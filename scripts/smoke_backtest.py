#!/usr/bin/env python
"""Smoke test: build the dataset and run the mechanical strategy vs buy & hold."""

from __future__ import annotations

import sys

import pandas as pd

from vongold.backtest import buy_and_hold, run_backtest, summarize
from vongold.config import BacktestConfig
from vongold.data import build_dataset


def main() -> int:
    cfg = BacktestConfig()
    print(f"building dataset for {cfg.symbol} ...", flush=True)
    df = build_dataset(cfg.symbol, rng="10y", refresh=True)
    print(f"  rows={len(df)}  {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  cols={list(df.columns)}")
    print("  macro coverage (non-null fraction):")
    for c in df.columns:
        if c.endswith("_available") or c in ("real_yield_10y", "dollar_index_broad"):
            print(f"    {c:28s} {df[c].notna().mean():.3f}")

    print("\n--- BUY & HOLD (GLD) ---")
    bh = buy_and_hold(df)
    print("  " + summarize(bh))

    print("\n--- MECHANICAL STRATEGY ---")
    res = run_backtest(df)
    print("  " + summarize(res))
    m = res.metrics
    print(f"  final equity  ${m['final_equity']:,.0f} from ${m['initial_capital']:,.0f}")
    print(f"  months positive {m['pct_months_positive']:.1%}  worst month {m['worst_month']:.2%}")
    print(f"  worst rolling 12m {m['worst_rolling_12m']:.2%}")

    print("\n  sample of the exposure path (last 8 rows):")
    tail = pd.DataFrame(
        {
            "close": df["close"].tail(8),
            "tsmom_sign": res.target.tail(8),
            "held": res.exposure.tail(8),
            "ret": res.returns.tail(8),
            "equity": res.equity.tail(8),
        }
    )
    print(tail.to_string())

    # --- Sanity checks that must hold or the engine is wrong ---
    failures = []
    if res.metrics["pct_time_in_market"] <= 0:
        failures.append("strategy was never in the market")
    if not res.exposure.index.equals(df.index):
        failures.append("exposure index does not match price index")
    # No-lookahead check: exposure on day t must equal target on day t-1.
    shifted = res.target.shift(1).fillna(0.0)
    if not shifted.equals(res.exposure):
        failures.append("exposure != lagged target (lookahead bug)")
    if failures:
        print("\nSANITY FAILURES:")
        for f in failures:
            print("  - " + f)
        return 1
    print("\nall sanity checks passed (no lookahead; strategy traded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
