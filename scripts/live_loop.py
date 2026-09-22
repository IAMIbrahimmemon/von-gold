#!/usr/bin/env python
"""Live loop: decide on the real price, trade the paper account, publish in real time.

Three questions this file answers, each of which was a real complaint:

**"Why does it always say add_long?"** Because von's five-way argmax carries a baseline
prior. Measured over 270 days, one label won every single time regardless of the market.
The *probabilities* do move with state (open_long correlates r=+0.585 with momentum) but the
ranking does not. So the action label is displayed as a state readout and weighted at 0.0 --
it is not allowed to decide trades. See docs/ACTIONS.md.

**"Why is it still $10,000 with no trades?"** Because the strategy's trend gate is closed:
GLD sits below its 200-day average, so the mechanical target is 0 and the correct behaviour
is to sit flat. A trend follower is flat for long stretches; that is the strategy working.
`--live-signals` makes entries possible the moment the gate opens, and the reason for
standing flat is recorded every cycle and shown on the dashboard.

**"Make it simulate in real time and show it on the page."** Two mechanisms:
  * `--live-signals` re-derives the target from the LIVE price rather than yesterday's close,
    so a move across the 200-day average triggers an entry within one cycle, not overnight.
  * every decision is published to an ntfy topic, which the dashboard subscribes to over
    Server-Sent Events, so the page updates in about a second. (The route through
    raw.githubusercontent.com is stuck behind a 5-minute CDN cache: measured
    `Cache-Control: max-age=300`. GitHub cannot serve a 10-second feed, so this goes around it.)

Execution happens at the live mark through the same cost model the backtest uses, so the P/L
is a simulation of what those fills would have cost -- not a marked-to-market fiction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from vongold.action import decide_action  # noqa: E402
from vongold.config import CostModel, StrategyParams  # noqa: E402
from vongold.data import build_dataset  # noqa: E402
from vongold.dryrun import simulate_fill  # noqa: E402
from vongold.live import fetch_live_mark  # noqa: E402
from vongold.state_store import Ledger, PositionStore, utcnow  # noqa: E402
from vongold.strategy import apply_rebalance_band, build_features, mechanical_exposure  # noqa: E402
from vongold.von_state import build_von_problem  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime"

# The real-time transport. ntfy is used because it is free, needs no account, sends
# `Access-Control-Allow-Origin: *`, and serves ndjson over a plain GET -- so a static page can
# subscribe with EventSource and receive decisions in about a second.
DEFAULT_TOPIC = os.environ.get("VONGOLD_NTFY_TOPIC", "")
NTFY = "https://ntfy.sh"


def publish(topic: str, payload: dict, timeout: float = 6.0) -> bool:
    """Publish one decision to the realtime topic. Never raises into the trading path."""
    if not topic:
        return False
    try:
        req = urllib.request.Request(
            f"{NTFY}/{topic}",
            data=json.dumps(payload, default=str).encode(),
            headers={"Content-Type": "application/json", "X-Title": "von-gold"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


class LedgerState:
    """Everything the loop persists, in one place."""

    def __init__(self, runtime: Path, initial_capital: float = 10_000.0, feed_size: int = 80):
        self.runtime = runtime
        self.store = PositionStore(runtime / "position.json", initial_capital=initial_capital)
        self.ledger = Ledger(runtime / "ledger.jsonl")
        self.feed_path = runtime / "feed.json"
        self.feed: list[dict] = []
        self.feed_size = feed_size
        # Load the existing feed so a restart does not blank the dashboard.
        try:
            old = json.loads(self.feed_path.read_text())
            self.feed = list(reversed(old.get("decisions", [])))[-feed_size:]
        except Exception:
            self.feed = []

    def push(self, row: dict) -> None:
        self.feed.append(row)
        del self.feed[:-self.feed_size]

    def write_feed(self, symbol: str, mark, target: float, signal: str,
                   mechanical: float, mode: str, fills_count: int) -> None:
        pos = self.store.state
        equity = pos.cash + pos.shares * mark.price
        payload = {
            "symbol": symbol,
            "generated_at": utcnow(),
            "mode": mode,
            "price": round(mark.price, 4),
            "mark": mark.as_dict(),
            "target_exposure": round(target, 4),
            "mechanical_target": round(mechanical, 4),
            "signal": signal,
            "equity": round(equity, 2),
            "cash": round(pos.cash, 2),
            "shares": round(pos.shares, 6),
            "avg_cost": round(pos.avg_cost, 4),
            "initial_capital": self.store.initial_capital,
            "unrealized": round((mark.price - pos.avg_cost) * pos.shares, 2) if pos.shares else 0.0,
            "fills": fills_count,
            "flat_reason": signal,
            "decisions": self.feed[::-1],  # newest first, like a trade tape
        }
        tmp = self.feed_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=str))
        tmp.replace(self.feed_path)  # atomic: the dashboard never reads a half-written file


def live_features(df_base: pd.DataFrame, params: StrategyParams, price: float) -> pd.DataFrame:
    """Features recomputed with the live price as the latest close.

    Only the CLOSE of the final bar is replaced. The rest of the history is genuinely
    completed daily data; rewriting it would fabricate bars. This is the difference between
    "the strategy sees today's price" and "the strategy sees invented history".
    """
    d = df_base.copy()
    d.iloc[-1, d.columns.get_loc("close")] = float(price)
    return build_features(d, params)


def main() -> int:
    ap = argparse.ArgumentParser(description="live paper-trading loop")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--cycles", type=int, default=0, help="0 = run forever")
    ap.add_argument("--symbol", default="GLD")
    ap.add_argument("--runtime", type=Path, default=RUNTIME)
    ap.add_argument("--topic", default=DEFAULT_TOPIC,
                    help="ntfy topic for the realtime feed (empty = publishing disabled)")
    ap.add_argument("--publish-every", type=int, default=0,
                    help="also git-publish runtime/ every N decisions (0 = never)")
    ap.add_argument("--feed-size", type=int, default=80)
    ap.add_argument("--cache-unchanged", action="store_true",
                    help="skip the model call when the state is unchanged (von is deterministic)")
    ap.add_argument("--no-von", action="store_true", help="mechanical only; skip the model")
    # --- trading policy ---
    ap.add_argument("--live-signals", action="store_true",
                    help="derive the target from the LIVE price, so a move across the trend gate "
                         "can enter or exit within a cycle instead of overnight")
    ap.add_argument("--allow-entry", action="store_true",
                    help="permit opening/increasing a position at all")
    ap.add_argument("--no-brake", action="store_true",
                    help="disable the whipsaw brake (it damps signal churn; keep it on)")
    ap.add_argument("--initial-capital", type=float, default=10_000.0)
    args = ap.parse_args()

    st = LedgerState(args.runtime, args.initial_capital, args.feed_size)
    params = StrategyParams()
    cost = CostModel()

    df_base = build_dataset(args.symbol, rng="10y").tail(800)
    mech_daily = float(mechanical_exposure(build_features(df_base, params), params).iloc[-1])
    ma200 = float(df_base["close"].tail(200).mean())

    print("von-gold live loop")
    print(f"  symbol {args.symbol} | interval {args.interval}s | live_signals={args.live_signals} "
          f"| allow_entry={args.allow_entry}")
    print(f"  last close {df_base['close'].iloc[-1]:.2f} vs 200d MA {ma200:.2f} "
          f"-> daily-close target {mech_daily:.3f}")
    print(f"  realtime topic: {args.topic or '(disabled)'}")
    pos0 = st.store.state
    print(f"  account: cash ${pos0.cash:,.2f} shares {pos0.shares} equity ${pos0.equity:,.2f}")

    last_fp = None
    last_act = None
    cycles = 0
    cached_n = 0
    fills_n = sum(1 for r in st.ledger.all() if r.get("kind") == "fill")

    while True:
        cycles += 1
        t0 = time.time()

        mark = fetch_live_mark(args.symbol)
        if mark is None:
            print(f"[{utcnow()}] no live mark; skipping cycle")
            time.sleep(args.interval)
            continue

        # --- target ---------------------------------------------------------------
        if args.live_signals:
            f = live_features(df_base, params, mark.price)
            raw = mechanical_exposure(f, params)
            if not args.no_brake:
                # band=0 keeps hysteresis off (this is a fresh target each cycle) but still
                # applies the min-hold brake, which cuts churn at no cost in edge.
                raw = apply_rebalance_band(raw, band=0.0, min_hold_days=params.min_hold_days)
            mech = float(raw.iloc[-1])
            gate = "above" if mech > 0 else "below"
            signal = f"live price {mark.price:.2f} is {gate} its trend gate"
        else:
            mech = mech_daily
            signal = f"daily close {df_base['close'].iloc[-1]:.2f} vs 200d MA {ma200:.2f}"

        target = mech
        if not args.allow_entry:
            # Reduce-only: never open or add. The current exposure is the ceiling, so the loop
            # can still scale down or flatten. This is the safe default because intraday entry
            # is a strategy with no backtest behind it.
            pos = st.store.state
            held = (pos.shares * mark.price / pos.equity) if pos.equity > 0 else 0.0
            target = min(target, held)
            if mech > held + 1e-9:
                signal += " (entry disabled: holding flat)"

        # --- von (advisory; cannot move money at weight 0) ------------------------
        act = None
        cached = False
        if not args.no_von:
            df = live_features(df_base, params, mark.price)
            prob = build_von_problem(df, len(df) - 1)
            if prob is not None:
                fp = hashlib.sha256(prob["state"].encode()).hexdigest()[:16]
                if args.cache_unchanged and fp == last_fp and last_act is not None:
                    act, cached = last_act, True
                    cached_n += 1
                else:
                    act = decide_action(prob["state"])
                    last_fp, last_act = fp, act

        # --- execute --------------------------------------------------------------
        pos = st.store.state
        target = max(0.0, min(1.0, target))
        equity_before = pos.cash + pos.shares * mark.price
        fill = {"traded": False, "delta_shares": 0.0, "cost": 0.0}
        if equity_before > 0:
            fill = simulate_fill(pos, target, mark.price, cost)
            if fill.get("traded"):
                st.store.snapshot(mark.price)
                st.ledger.append("fill", symbol=args.symbol, price=mark.price,
                                 target_exposure=target, **fill)
                fills_n += 1

        equity = pos.cash + pos.shares * mark.price
        pnl = equity - st.store.initial_capital

        row = {
            "ts": utcnow(),
            "price": round(mark.price, 4),
            "action": (act.action if act else None),
            "confidence": (act.confidence if act else None),
            "probabilities": (act.probabilities if act else None),
            "conviction": (act.conviction if act else None),
            "latency_ms": int((time.time() - t0) * 1000),
            "cached": cached,
            "target_exposure": round(target, 4),
            "mechanical_target": round(mech, 4),
            "traded": bool(fill.get("traded")),
            "delta_shares": round(fill.get("delta_shares", 0.0) or 0.0, 6),
            "cost": round(fill.get("cost", 0.0) or 0.0, 4),
            "shares": round(pos.shares, 6),
            "cash": round(pos.cash, 2),
            "equity": round(equity, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / st.store.initial_capital * 100, 4),
            "market_open": mark.market_open,
            "age_minutes": round(mark.age_minutes, 1),
            "signal": signal,
        }
        st.push(row)
        st.write_feed(args.symbol, mark, target, signal, mech,
                      mode=("live-signals" if args.live_signals else "daily-signals"),
                      fills_count=fills_n)

        if args.topic:
            publish(args.topic, row)

        if args.publish_every and cycles % args.publish_every == 0:
            subprocess.run(["bash", str(REPO / "scripts" / "publish_status.sh"),
                            "runtime: live feed"], cwd=str(REPO),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        el = time.time() - t0
        tag = f"TRADE {row['delta_shares']:+.4f}sh" if row["traded"] else "no trade"
        print(f"[{row['ts'][11:19]}] ${mark.price:7.2f} tgt={target:.2f} {tag:20} "
              f"eq=${equity:9.2f} pnl=${pnl:+8.2f} | {(act.action if act else '-'):10} "
              f"| {el:.1f}s{' cached' if cached else ''}")

        if args.cycles and cycles >= args.cycles:
            break
        time.sleep(max(0.0, args.interval - el))

    print(f"stopped after {cycles} cycles ({cached_n} cached)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
