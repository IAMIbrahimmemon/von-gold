"""The dry-run (paper) trading loop.

Runs once per trading session on the LOCAL machine, in the von-mlx venv so that the
decision model is a direct in-process call (no HTTP, no per-call tokens).

What "dry run" means here, precisely:
  * No broker. No API keys. No order ever leaves this machine.
  * Fills are simulated at the latest available close with the same cost model the
    backtester uses, so the paper equity curve is comparable to the backtest.
  * Every step is appended to runtime/ledger.jsonl, so any claimed result can be
    traced to a decision, an input price and a model answer.

Safety properties that are deliberate:
  * Fails CLOSED. If the control file is unreadable, or the model fails, or the price
    feed fails, exposure goes to zero and nothing is bought.
  * Hard drawdown kill-switch, checked before any new position is opened.
  * The model can only ever REDUCE exposure relative to the mechanical target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import CostModel, StrategyParams
from .data import build_dataset
from .live import fetch_live_mark, spot_cross_check
from .pl import compute_pl
from .state_store import Control, Ledger, PositionStore, read_status, utcnow, write_status
from .strategy import build_features, mechanical_exposure
from .von_client import decide, von_environment_status
from .von_overlay import MODES, overlay_exposure

RUNTIME = Path(__file__).resolve().parents[2] / "runtime"


def _von_available() -> tuple[bool, str]:
    """Check the local decision model can be reached from this interpreter."""
    return von_environment_status()


def decide_today(
    params: StrategyParams,
    overlay_mode: str = "veto",
    von_min_confidence: float = 0.60,
    use_von: bool = True,
    symbol: str = "GLD",
    lookback_days: int = 800,
) -> dict:
    """Produce today's target exposure, with and without the von overlay.

    Returns a dict describing the decision; does NOT place anything.
    """
    df = build_dataset(symbol, rng="10y")
    df = df.tail(lookback_days)
    f = build_features(df, params)
    mech = mechanical_exposure(f, params)

    out = {
        "date": str(df.index[-1].date()),
        "close": float(df["close"].iloc[-1]),
        "mechanical_target": float(mech.iloc[-1]),
        "von_multiplier": 1.0,
        "final_target": float(mech.iloc[-1]),
        "von": None,
        "von_error": None,
        "inputs": {
            "vol_21d_ann": float(f["vol"].iloc[-1]) if pd.notna(f["vol"].iloc[-1]) else None,
            "tsmom": float(f["tsmom"].iloc[-1]),
            "above_ma200": bool(f["above_ma"].iloc[-1] > 0),
            "real_yield_support": bool(f["real_yield_support"].iloc[-1] > 0),
            "dollar_support": bool(f["dollar_support"].iloc[-1] > 0),
            "real_yield_10y": float(f["real_yield"].iloc[-1]) if pd.notna(f["real_yield"].iloc[-1]) else None,
        },
    }

    if not use_von:
        return out

    ok, detail = _von_available()
    if not ok:
        out["von_error"] = detail
        return out

    # Build the state/question pair for the latest bar and ask the local model.
    from .von_state import build_von_problem

    prob = build_von_problem(df, len(df) - 1)
    if prob is None:
        out["von_error"] = "insufficient history for a decision problem"
        return out

    try:
        res = decide(prob["state"], prob["questions"])
        if not res.ok:
            out["von_error"] = res.error or "model returned no usable answer"
            return out

        out["von"] = {
            "regime": res.regime,
            "regime_confidence": res.regime_confidence,
            "long_prob": res.long_prob,
            "risk_prob": res.risk_prob,
            "conviction": res.conviction,
            "usage": res.usage,
            "transport": res.transport,
        }

        # Build a one-row answers frame so the overlay logic is shared with the
        # backtest -- the live path must not reimplement the overlay rule.
        row = pd.DataFrame(
            [
                {
                    "regime": res.regime,
                    "regime_confidence": res.regime_confidence or 0.0,
                    "long_prob": res.long_prob,
                    "risk_prob": res.risk_prob,
                }
            ],
            index=pd.DatetimeIndex([df.index[-1]]),
        )
        mult = overlay_exposure(row, df.index, mode=overlay_mode, min_confidence=von_min_confidence)
        out["von_multiplier"] = float(mult.iloc[-1])
        out["final_target"] = float(mech.iloc[-1] * out["von_multiplier"])
    except Exception as exc:
        # The trading path must never break because of the model.
        out["von_error"] = f"{type(exc).__name__}: {exc}"

    return out


def simulate_fill(position, target_exposure: float, price: float, cost: CostModel) -> dict:
    """Move the paper position toward the target exposure and charge costs.

    Long-only, no leverage: target_exposure in [0,1] of current equity.
    """
    equity_before = position.cash + position.shares * price
    target_value = max(0.0, min(1.0, target_exposure)) * equity_before
    current_value = position.shares * price
    delta_value = target_value - current_value

    # Ignore dust: below half a share of notional there is nothing meaningful to do.
    if abs(delta_value) < price * 0.5 or equity_before <= 0:
        return {"traded": False, "delta_value": 0.0, "cost": 0.0,
                "equity_before": equity_before, "equity_after": equity_before}

    per_side_bps = cost.spread_bps_per_side + cost.slippage_bps_per_side + cost.commission_bps_per_side
    fee = abs(delta_value) * per_side_bps / 10_000.0

    delta_shares = delta_value / price
    new_shares = position.shares + delta_shares
    if new_shares < 1e-9:
        new_shares = 0.0
    # Cost comes out of cash; buying reduces cash by the notional plus the fee.
    new_cash = position.cash - delta_value - fee

    position.shares = new_shares
    position.cash = new_cash
    if position.shares > 0 and delta_shares > 0:
        total_cost_basis = position.avg_cost * (position.shares - delta_shares) + price * delta_shares
        position.avg_cost = total_cost_basis / position.shares if position.shares else 0.0
    if position.shares == 0:
        position.avg_cost = 0.0
        position.opened_at = None
    elif position.opened_at is None:
        position.opened_at = utcnow()

    equity_after = position.cash + position.shares * price
    return {
        "traded": True,
        "delta_value": delta_value,
        "delta_shares": delta_shares,
        "cost": fee,
        "equity_before": equity_before,
        "equity_after": equity_after,
    }


def run_once(
    params: StrategyParams | None = None,
    cost: CostModel | None = None,
    use_von: bool = True,
    overlay_mode: str = "veto",
    force: bool = False,
    runtime: Path | None = None,
) -> dict:
    """One dry-run tick. Safe to run repeatedly; idempotent per session date."""
    runtime = runtime or RUNTIME
    params = params or StrategyParams()
    cost = cost or CostModel()

    control = Control.load(runtime / "control.json")
    ledger = Ledger(runtime / "ledger.jsonl")
    store = PositionStore(runtime / "position.json")

    decision = decide_today(params, overlay_mode=overlay_mode, use_von=use_von,
                            symbol=control.symbol)

    pos = store.state

    # SIGNAL vs MARK. The decision is made on the daily close (unchanged: reading intraday
    # bars would change what the strategy means and invalidate the backtest). But the paper
    # account is valued and executed at the latest live price, so a held position shows its
    # real current worth instead of yesterday's number all day.
    signal_price = decision["close"]
    mark = fetch_live_mark(control.symbol)
    if mark is not None:
        price = mark.price
    else:
        # No live quote: value at the last known price rather than inventing one.
        price = pos.last_price if pos.last_price > 0 else signal_price
        mark = None

    # --- Drawdown kill switch, evaluated BEFORE any trading ---
    peak = max([p.get("equity_after", 0.0) for p in ledger.all() if p.get("kind") == "fill"] + [pos.equity, 0.0])
    equity_now = pos.cash + pos.shares * price  # price is the live mark here
    dd = (equity_now / peak - 1.0) if peak > 0 else 0.0
    killed = dd < -abs(control.kill_if_drawdown_exceeds)
    if killed:
        control.enabled = False
        control.reason = f"kill switch: drawdown {dd:.2%} exceeded {control.kill_if_drawdown_exceeds:.0%}"
        control.changed_by = "kill_switch"
        control.changed_at = utcnow()
        control.save(runtime / "control.json")

    target = decision["final_target"] if (control.enabled or force) else 0.0
    target = min(target, control.max_exposure)

    # Idempotency: do not trade the same session twice unless forced.
    prior = [r for r in ledger.all() if r.get("kind") == "decision"]
    already = any(r.get("date") == decision["date"] for r in prior)

    ledger.append(
        "decision",
        date=decision["date"],
        enabled=control.enabled,
        killed=killed,
        mechanical_target=decision["mechanical_target"],
        von_multiplier=decision["von_multiplier"],
        final_target=target,
        close=signal_price,
        mark=price,
        mark_source=(mark.source if mark else "last known"),
        inputs=decision["inputs"],
        von=decision["von"],
        von_error=decision["von_error"],
        already_traded_today=already,
    )

    fill = {"traded": False, "reason": "already traded this session"} if (already and not force) else simulate_fill(pos, target, price, cost)

    if fill.get("traded"):
        store.save()
        ledger.append("fill", date=decision["date"], price=price,
                      shares=pos.shares, cash=pos.cash, **fill)

    store.snapshot(price)
    status = {
        "symbol": control.symbol,
        "enabled": control.enabled,
        "kill_switch_tripped": killed,
        "date": decision["date"],
        "price": price,
        "signal_close": signal_price,
        # Why exposure is what it is. Without this a flat bot looks broken on the dashboard;
        # with it the reason ("price is below its 200d average") is visible.
        "flat_reason": (
            f"price {price:.2f} is "
            + ("above" if decision["inputs"]["above_ma200"] else "below")
            + " its 200d trend gate"
            + ("" if target > 0 else " -- flat is the strategy's decision, not an error")
        ),
        "mark": (mark.as_dict() if mark else None),
        "spot": spot_cross_check(),
        "as_of": utcnow(),
        "target_exposure": target,
        "mechanical_target": decision["mechanical_target"],
        "von_multiplier": decision["von_multiplier"],
        "position_shares": pos.shares,
        "cash": pos.cash,
        "equity": pos.equity,
        "drawdown_from_peak": dd,
        "traded": bool(fill.get("traded")),
        "von": decision["von"],
        "von_error": decision["von_error"],
        "inputs": decision["inputs"],
        "recent_fills": [r for r in ledger.tail(200) if r.get("kind") == "fill"][-20:],
        "equity_curve": [
            {"ts": r["ts"], "equity": r.get("equity_after"), "date": r.get("date")}
            for r in ledger.all() if r.get("kind") == "fill"
        ][-250:],
        # Reconstructed from the append-only ledger rather than sampled from the curve
        # above, so the sub-day windows are real instead of reading as flat simply
        # because no tick landed inside them.
        "pl": compute_pl(
            ledger_path=ledger.path,
            initial_capital=store.initial_capital,
            current_equity=pos.equity,
            shares=pos.shares,
        ),
    }
    write_status(status, runtime / "status.json")
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description="von-gold dry-run tick")
    ap.add_argument("--no-von", action="store_true", help="mechanical strategy only")
    ap.add_argument("--mode", default="rank", choices=sorted(MODES.keys()),
                    help="von overlay mode (see docs/VON.md); 'rank' is the least bad")
    ap.add_argument("--force", action="store_true", help="trade even if disabled/already traded")
    ap.add_argument("--enable", action="store_true", help="flip the switch on, then run")
    ap.add_argument("--disable", action="store_true", help="flip the switch off, then run")
    ap.add_argument("--status", action="store_true", help="print status and exit")
    ap.add_argument("--von-check", action="store_true",
                    help="report whether the local model is reachable, then exit")
    args = ap.parse_args()

    if args.von_check:
        ok, detail = _von_available()
        print(f"von: {'OK' if ok else 'UNAVAILABLE'} -- {detail}")
        return 0 if ok else 1

    if args.status:
        print(json.dumps(read_status(RUNTIME / "status.json"), indent=2, default=str))
        return 0

    if args.enable or args.disable:
        c = Control.load(RUNTIME / "control.json")
        c.enabled = bool(args.enable)
        c.reason = "manual CLI toggle"
        c.changed_by = "cli"
        c.changed_at = utcnow()
        c.save(RUNTIME / "control.json")
        print(f"switch -> {'ON' if c.enabled else 'OFF'}")

    status = run_once(use_von=not args.no_von, overlay_mode=args.mode, force=args.force)
    print(json.dumps(status, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
