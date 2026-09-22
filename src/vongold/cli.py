"""CLI entry point."""

from __future__ import annotations

import argparse
import json
import sys

from .config import BacktestConfig
from .data import build_dataset


def main() -> int:
    ap = argparse.ArgumentParser(prog="von-gold", description="Dry-run gold trading system")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_data = sub.add_parser("data", help="build/refresh the dataset")
    p_data.add_argument("--symbol", default="GLD")
    p_data.add_argument("--refresh", action="store_true")

    p_bt = sub.add_parser("backtest", help="run the mechanical backtest")
    p_bt.add_argument("--symbol", default="GLD")

    p_tick = sub.add_parser("tick", help="one dry-run trading session")
    p_tick.add_argument("--no-von", action="store_true")
    p_tick.add_argument("--force", action="store_true")
    p_tick.add_argument("--enable", action="store_true")
    p_tick.add_argument("--disable", action="store_true")
    p_tick.add_argument("--status", action="store_true")

    args = ap.parse_args()

    if args.cmd == "data":
        df = build_dataset(args.symbol, refresh=args.refresh)
        print(f"{len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
        return 0

    if args.cmd == "backtest":
        from .backtest import buy_and_hold, run_backtest, summarize

        cfg = BacktestConfig(symbol=args.symbol)
        df = build_dataset(args.symbol)
        print("buy & hold :", summarize(buy_and_hold(df)))
        print("mechanical :", summarize(run_backtest(df, cfg.params, cfg.cost)))
        return 0

    if args.cmd == "tick":
        from .dryrun import main as tick_main

        sys.argv = ["von-gold tick"] + [
            a for a in ("--no-von", "--force", "--enable", "--disable", "--status")
            if getattr(args, a.replace("--", "").replace("-", "_"))
        ]
        return tick_main()

    return 1


if __name__ == "__main__":
    sys.exit(main())
