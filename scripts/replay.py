#!/usr/bin/env python
"""Replay the strategy over real history so the dashboard can be watched making trades.

Why this exists
---------------
The strategy trades on trend-gate crossings -- measured 12-14 times a year, with long flat
stretches in between. Gold is currently below its 200-day average, so the live loop will sit
flat until price rises ~4.5%. Watching a real-time bot wait is honest but useless for
verifying that the trading, the P&L bookkeeping and the dashboard all actually work.

This replays real historical bars through the SAME simulate_fill path the live loop uses, so
every fill, fee and realized-P&L figure is produced by the production code. It then writes a
replay feed the dashboard can display in a separate mode.

It is NOT a live simulation and does not pretend to be: the timestamps are compressed (one
trading day per --step-seconds) and the file is written to runtime/replay/ so it can never be
confused with the real account in runtime/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vongold.config import CostModel, StrategyParams  # noqa: E402
from vongold.data import build_dataset  # noqa: E402
from vongold.dryrun import simulate_fill  # noqa: E402
from vongold.live import fetch_live_mark  # noqa: E402
from vongold.state_store import PositionStore, utcnow  # noqa: E402
from vongold.strategy import build_features, mechanical_exposure  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="replay history through the real fill path")
    ap.add_argument("--days", type=int, default=250,
                    help="trading days to replay (250 ~= one year)")
    ap.add_argument("--step-seconds", type=float, default=1.0,
                    help="real seconds per replayed day")
    ap.add_argument("--runtime", type=Path, default=REPO / "runtime" / "replay")
    ap.add_argument("--feed-name", default="feed.json")
    ap.add_argument("--no-realtime", action="store_true", help="do not publish to ntfy")
    args = ap.parse_args()

    rt = args.runtime
    rt.mkdir(parents=True, exist_ok=True)

    # Start from a clean account so the replay's P&L is unambiguous.
    store = PositionStore(rt / "position.json", initial_capital=10_000.0)
    store.state.__dict__.update(cash=10_000.0, shares=0.0, avg_cost=0.0,
                                realized_pl=0.0, fees_paid=0.0,
                                trades=0, wins=0, losses=0, equity=10_000.0)
    store.save()
    pos = store.state
    cost = CostModel()
    params = StrategyParams()

    df = build_dataset("GLD", rng="10y")
    f = build_features(df, params)
    targets = mechanical_exposure(f, params)

    end = len(df) - 1
    start = max(260, end - args.days)
    print(f"replaying {start}..{end}  ({end - start} sessions, "
          f"{df.index[start].date()} -> {df.index[end].date()})")
    print(f"  {args.step_seconds}s per session | feed: {rt / args.feed_name}")

    topic = ""
    cfg = (REPO / "runtime" / "realtime.json")
    if not args.no_realtime and cfg.exists():
        topic = json.loads(cfg.read_text()).get("topic", "")

    feed: list[dict] = []
    fills = 0

    for i in range(start, end + 1):
        date = df.index[i]
        price = float(df["close"].iloc[i])
        target = float(targets.iloc[i])

        equity_before = pos.cash + pos.shares * price
        fill = simulate_fill(pos, target, price, cost)
        if fill.get("traded"):
            fills += 1
            store.snapshot(price)
        else:
            pos.last_price = price
            pos.equity = pos.cash + pos.shares * price

        equity = pos.cash + pos.shares * price
        pnl = equity - store.initial_capital
        row = {
            "ts": f"{date.date()}T00:00:00+00:00",
            "replay": True,
            "price": round(price, 4),
            "action": None,
            "confidence": None,
            "probabilities": None,
            "target_exposure": round(target, 4),
            "mechanical_target": round(target, 4),
            "traded": bool(fill.get("traded")),
            "delta_shares": round(fill.get("delta_shares", 0.0) or 0.0, 6),
            "cost": round(fill.get("cost", 0.0) or 0.0, 4),
            "realized_pl": round(fill.get("realized_pl", 0.0) or 0.0, 2),
            "shares": round(pos.shares, 6),
            "cash": round(pos.cash, 2),
            "equity": round(equity, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / store.initial_capital * 100, 4),
            "signal": ("above trend gate" if target > 0 else "below trend gate"),
        }
        feed.append(row)
        del feed[:-220]

        payload = {
            "symbol": "GLD",
            "generated_at": utcnow(),
            "mode": "replay",
            "replay": True,
            "replay_note": ("Historical replay through the production fill path. Timestamps are "
                            "compressed; this is not live trading."),
            "price": round(price, 4),
            "mark": {"price": price, "as_of": f"{date.date()}T00:00:00+00:00",
                     "source": "replay (daily close)", "is_intraday": False,
                     "market_open": False, "age_minutes": 0.0},
            "target_exposure": round(target, 4),
            "mechanical_target": round(target, 4),
            "signal": row["signal"],
            "equity": round(equity, 2),
            "cash": round(pos.cash, 2),
            "shares": round(pos.shares, 6),
            "avg_cost": round(pos.avg_cost, 4),
            "initial_capital": store.initial_capital,
            "unrealized": round(pos.unrealized_pl, 2),
            "realized_pl": round(pos.realized_pl, 2),
            "fees_paid": round(pos.fees_paid, 2),
            "trades": pos.trades,
            "wins": pos.wins,
            "losses": pos.losses,
            "total_pl": round(pos.realized_pl + pos.unrealized_pl, 2),
            "fills": fills,
            "flat_reason": row["signal"],
            "decisions": feed[::-1],
        }
        tmp = (rt / args.feed_name).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=str))
        tmp.replace(rt / args.feed_name)

        if topic:
            try:
                import urllib.request
                req = urllib.request.Request(f"https://ntfy.sh/{topic}",
                                             data=json.dumps(row, default=str).encode(),
                                             headers={"Content-Type": "application/json",
                                                      "X-Title": "von-gold replay"},
                                             method="POST")
                urllib.request.urlopen(req, timeout=5).read()
            except Exception:
                pass

        if fills and fill.get("traded"):
            print(f"  {date.date()} ${price:7.2f} target={target:.2f} "
                  f"{'BUY ' if row['delta_shares']>0 else 'SELL'} {row['delta_shares']:+.4f}sh "
                  f"realized=${row['realized_pl']:+.2f} equity=${equity:9.2f}")
        time.sleep(args.step_seconds)

    print(f"\ndone: {fills} fills, {len(feed)} decisions")
    print(f"  equity ${pos.cash:,.2f}  realized ${pos.realized_pl:+,.2f} "
          f"fees ${pos.fees_paid:,.2f}  {pos.wins}W/{pos.losses}L")
    print(f"  feed: {rt / args.feed_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
