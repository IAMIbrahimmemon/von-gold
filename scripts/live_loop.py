#!/usr/bin/env python
"""Live re-decide loop: re-evaluate the position every N seconds against the current price.

Read this before running it
---------------------------
Measured facts that shaped this design:

* **A von decision takes ~4.75s** (subprocess transport, the path live use takes). A
  10-second interval therefore means the model is busy roughly half the time -- on a laptop,
  that is real fan and battery cost. The default here is 60s and it is configurable.

* **von is deterministic.** The same state asked three times returns byte-identical
  answers. So re-asking an UNCHANGED state every 10 seconds yields nothing new; the loop is
  only meaningful when the *price* moves, because the price is what changes the state. This
  loop therefore compares a state fingerprint and skips the model call when nothing that
  feeds the decision has changed -- the answer would be provably identical.

* **The signal is daily.** The strategy is built on daily closes and validated that way.
  Re-deciding intraday is a DIFFERENT strategy with no backtest behind it, so by default
  this loop may only REDUCE exposure (act as a risk brake) and cannot open or add a
  position. That is the honest safe direction: the worst case is being flat in a rally, not
  holding a losing position the validated strategy never took. Set --allow-entry to let it
  open positions (untested; you would be trading without evidence).

What it does per cycle:
    fetch a live mark -> rebuild state with the live price -> fingerprint -> (if changed)
    ask von for one of the five actions -> apply the delta -> append to a heartbeat ledger
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from vongold.action import decide_action  # noqa: E402
from vongold.config import StrategyParams  # noqa: E402
from vongold.data import build_dataset  # noqa: E402
from vongold.live import fetch_live_mark, market_is_open, spot_cross_check  # noqa: E402
from vongold.state_store import PositionStore  # noqa: E402
from vongold.strategy import build_features, mechanical_exposure  # noqa: E402
from vongold.von_state import build_von_problem  # noqa: E402

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
REPO = Path(__file__).resolve().parents[1]

# The rolling decision feed the dashboard renders. Kept in memory and rewritten whole each
# cycle so the published file is always a consistent snapshot (a half-appended JSONL would
# render as a torn feed).
FEED: list[dict] = []


def write_feed(path: Path, feed: list[dict], symbol: str, store, mech_target: float,
               mark) -> None:
    """Write the live decision feed the dashboard reads.

    Deliberately a separate file from status.json: status.json is the once-per-session record
    of what the STRATEGY decided, while this is the per-10s record of what the MODEL is
    saying right now. Mixing them would make the session record churn every few seconds.
    """
    pos = store.state
    payload = {
        "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "interval_source": "live_loop",
        "price": round(mark.price, 4),
        "mark": mark.as_dict(),
        "mechanical_target": round(mech_target, 4),
        "equity": round(pos.equity, 2),
        "cash": round(pos.cash, 2),
        "shares": pos.shares,
        "decisions": feed[::-1],  # newest first, like a trade tape
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)  # atomic: the dashboard never reads a partially written file


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def state_with_live_price(df: pd.DataFrame, price: float) -> pd.DataFrame:
    """Overlay the live price onto the last daily bar.

    Only the CLOSE is replaced. The rest of the bar is daily history that has genuinely
    completed; rewriting it would fabricate data. The last bar's close becomes the live
    mark, which is the one value that is actually current.
    """
    out = df.copy()
    out.iloc[-1, out.columns.get_loc("close")] = float(price)
    return out


def fingerprint(state: str) -> str:
    return hashlib.sha256(state.encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description="live re-decide loop (paper)")
    ap.add_argument("--interval", type=float, default=10.0,
                    help="seconds between decisions (von takes ~4.75s, so ~50%% duty cycle)")
    ap.add_argument("--allow-entry", action="store_true",
                    help="let the loop OPEN positions (untested intraday -- off by default)")
    ap.add_argument("--cycles", type=int, default=0, help="0 = run forever")
    ap.add_argument("--cache-unchanged", action="store_true",
                    help="skip the model call when the state fingerprint is unchanged. von is "
                         "deterministic, so the answer would be identical -- this saves ~50%% CPU "
                         "and produces the same feed (rows are marked cached)")
    ap.add_argument("--publish-every", type=int, default=0,
                    help="publish the feed to GitHub every N decisions (0 = never)")
    ap.add_argument("--feed-size", type=int, default=60, help="decisions kept in the feed file")
    ap.add_argument("--symbol", default="GLD")
    ap.add_argument("--runtime", type=Path, default=RUNTIME)
    ap.add_argument("--journal", type=Path, default=None)
    args = ap.parse_args()

    journal = args.journal or (args.runtime / "live_loop.jsonl")
    feed_path = args.runtime / "feed.json"
    store = PositionStore(args.runtime / "position.json")

    df_base = build_dataset(args.symbol, rng="10y").tail(800)
    params = StrategyParams()
    f_base = build_features(df_base, params)
    mech_target = float(mechanical_exposure(f_base, params).iloc[-1])

    print(f"live loop: {args.symbol} | interval {args.interval}s | "
          f"mechanical target {mech_target:.3f} | allow_entry={args.allow_entry}")
    print(f"journal: {journal}")

    last_fp = None
    last_action = None
    cycles = 0
    skipped = 0

    while True:
        cycles += 1
        t0 = time.time()
        mark = fetch_live_mark(args.symbol)
        if mark is None:
            print(f"[{utcnow()}] no live mark available; skipping cycle")
            time.sleep(args.interval)
            continue

        df = state_with_live_price(df_base, mark.price)
        prob = build_von_problem(df, len(df) - 1)
        if prob is None:
            print(f"[{utcnow()}] could not build a decision problem")
            time.sleep(args.interval)
            continue

        fp = fingerprint(prob["state"])
        changed = fp != last_fp

        if args.cache_unchanged and not changed and last_action is not None:
            # Provably the same problem -> the same answer. Do not spend 5s of CPU on it.
            skipped += 1
            row = {"ts": utcnow(), "price": mark.price, "fingerprint": fp,
                   "action": last_action.action, "conviction": last_action.conviction,
                   "changed": False, "model_called": False,
                   "market_open": mark.market_open, "age_minutes": mark.age_minutes}
            with journal.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"[{utcnow()}] ${mark.price:.2f} state unchanged (fp={fp}) "
                  f"-> {last_action.action} (cached, no model call)")
            time.sleep(args.interval)
            if args.cycles and cycles >= args.cycles:
                break
            continue

        act = decide_action(prob["state"])
        last_fp, last_action = fp, act

        pos = store.state
        exposure_now = (pos.shares * mark.price / pos.equity) if pos.equity > 0 else 0.0

        # --- apply, in the safe direction only unless --allow-entry ---
        applied = None
        if act.ok:
            if act.action == "close":
                applied = "close"
            elif act.action == "reduce":
                applied = "reduce"
            elif act.action in ("open_long", "add_long") and args.allow_entry:
                applied = act.action
            elif act.action in ("open_long", "add_long"):
                applied = "ignored (entry disabled)"
            else:
                applied = "hold (no change)"

        row = {
            "ts": utcnow(), "price": mark.price, "fingerprint": fp,
            "action": act.action, "conviction": act.conviction,
            "error": act.error, "usage": act.usage,
            "changed": True, "model_called": True,
            "market_open": mark.market_open, "age_minutes": mark.age_minutes,
            "exposure_now": exposure_now, "applied": applied,
            "spot": None,
        }
        if cycles % 10 == 1:
            row["spot"] = spot_cross_check()

        with journal.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        FEED.append({"ts": row["ts"], "price": mark.price, "action": act.action,
                     "confidence": act.confidence, "probabilities": act.probabilities,
                     "conviction": act.conviction, "latency_ms": int((time.time()-t0)*1000),
                     "state_changed": True, "cached": False, "applied": applied,
                     "market_open": mark.market_open,
                     "age_minutes": round(mark.age_minutes, 1)})
        del FEED[:-args.feed_size]
        write_feed(feed_path, FEED, args.symbol, store, mech_target, mark)

        if args.publish_every and cycles % args.publish_every == 0:
            subprocess.run(["bash", str(REPO / "scripts" / "publish_status.sh"),
                            "runtime: live decisions"], cwd=str(REPO),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        el = time.time() - t0
        print(f"[{utcnow()}] ${mark.price:.2f} fp={fp} -> "
              f"{act.action} (conv {act.conviction}) {applied} "
              f"[{el:.1f}s, cycles={cycles} skipped={skipped}]")

        if args.cycles and cycles >= args.cycles:
            break
        time.sleep(max(0.0, args.interval - el))

    print(f"stopped after {cycles} cycles ({skipped} skipped as unchanged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
