#!/usr/bin/env python
"""Is the whipsaw brake real, or is the sweep's scatter just noise?

The brake sweep produced a non-monotonic Sharpe path across hold lengths (0.729 at
hold=0, 0.725 at 30, 0.629 at 15, 0.634 at 40). A real effect varies smoothly with its
own parameter; a jagged path usually means the parameter is not controlling anything
and the differences are resampling noise.

So: bootstrap the Sharpe DIFFERENCE between settings rather than comparing point
estimates. If the CI for the difference straddles zero, the brake is not an alpha tool
-- whatever else it is good for.

Also checks the original motivation: does the brake fix the range-bound decade
(1989-1998) that lost money gross through repeated flips?
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vongold.backtest import run_backtest
from vongold.config import CostModel, StrategyParams
from vongold.data import load_lbma_gold, load_local_parquet
from vongold.experiments import sharpe_of

TRADING_DAYS = 252
GLD_YF = Path("data/processed/gld_yfinance_max.parquet")
BLOCK = 21          # bootstrap block in days: preserves short-horizon autocorrelation
N_BOOT = 4000
RNG = np.random.default_rng(12345)


def block_bootstrap_diff(a: pd.Series, b: pd.Series, n_boot: int = N_BOOT,
                         block: int = BLOCK, rng: np.random.Generator = RNG):
    """Bootstrap CI for Sharpe(a) - Sharpe(b) using paired blocks.

    Blocks are drawn once and applied to BOTH series, so the comparison is paired and
    the (large) common market variation cancels instead of swamping the difference.
    """
    a = a.dropna().to_numpy()
    b = b.dropna().to_numpy()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    n_blocks = n // block
    if n_blocks < 5:
        return float("nan"), float("nan"), float("nan")
    diffs = np.empty(n_boot)
    starts_pool = np.arange(0, n - block + 1)
    for i in range(n_boot):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])
        aa, bb = a[idx], b[idx]
        sa = aa.mean() / aa.std(ddof=1) if aa.std(ddof=1) > 0 else 0.0
        sb = bb.mean() / bb.std(ddof=1) if bb.std(ddof=1) > 0 else 0.0
        diffs[i] = (sa - sb) * math.sqrt(TRADING_DAYS)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(diffs.mean())


def main() -> int:
    cost = CostModel()
    lbma = load_lbma_gold()
    gld = load_local_parquet(GLD_YF)

    def rets(df, hold):
        return run_backtest(df, params=StrategyParams(min_hold_days=hold), cost=cost).returns

    print("=" * 104)
    print("BOOTSTRAP TEST: is any brake setting distinguishable from hold=0?")
    print("=" * 104)
    print(f"  paired block bootstrap, {BLOCK}-day blocks, {N_BOOT} resamples\n")

    rows = []
    for name, df in (("LBMA 58y", lbma), ("GLD 22y", gld)):
        base = rets(df, 0)
        base_sr = sharpe_of(base) * math.sqrt(TRADING_DAYS)
        print(f"  {name} (hold=0 Sharpe {base_sr:.3f}):")
        for hold in (5, 10, 20, 30, 60):
            r = rets(df, hold)
            sr = sharpe_of(r) * math.sqrt(TRADING_DAYS)
            lo, hi, mean = block_bootstrap_diff(r, base)
            sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not distinguishable"
            rows.append({"window": name, "hold": hold, "sharpe": sr,
                         "d_sharpe": sr - base_sr, "ci_lo": lo, "ci_hi": hi,
                         "verdict": sig})
            print(f"    hold {hold:3d}d: Sharpe {sr:5.3f} (d {sr - base_sr:+.3f})  "
                  f"95% CI on the difference [{lo:+.3f}, {hi:+.3f}]  -> {sig}")
        print()

    tab = pd.DataFrame(rows)
    n_sig = int((tab["verdict"] == "SIGNIFICANT").sum())
    print(f"  RESULT: {n_sig}/{len(tab)} brake settings are statistically distinguishable from off.")
    if n_sig == 0:
        print("  => The brake is NOT an alpha improvement. Any apparent gain is noise.")
    else:
        sig_rows = tab[tab["verdict"] == "SIGNIFICANT"]
        print(sig_rows.to_string(index=False))

    # Original motivation: the range-bound decade.
    print("\n" + "=" * 104)
    print("DOES THE BRAKE FIX THE RANGE-BOUND DECADE? (1989-1998, the worst in the 58y run)")
    print("=" * 104)
    sl = lbma.loc["1989-01-01":"1998-12-31"]
    for hold in (0, 5, 10, 20, 30, 60):
        res = run_backtest(sl, params=StrategyParams(min_hold_days=hold), cost=cost)
        m = res.metrics
        print(f"  hold {hold:3d}d: ret {m['cagr']:+6.2%}  sharpe {m['sharpe']:5.2f}  "
              f"maxDD {m['max_drawdown']:7.1%}  switches {m['exposure_switches']:4d}  "
              f"cost {m['cost_drag_annual_bps']:5.1f}bps")

    # What the brake DOES reliably do: cut turnover. Quantify it, since that is a real
    # operational benefit even when Sharpe is unchanged.
    print("\n" + "=" * 104)
    print("WHAT THE BRAKE RELIABLY CHANGES: turnover and cost (the honest use for it)")
    print("=" * 104)
    print(f"  {'hold':>5s} | {'LBMA switches':>13s} {'cost bps/yr':>12s} | {'GLD switches':>12s} {'cost bps/yr':>12s}")
    for hold in (0, 5, 10, 20, 30, 60):
        a = run_backtest(lbma, params=StrategyParams(min_hold_days=hold), cost=cost).metrics
        b = run_backtest(gld, params=StrategyParams(min_hold_days=hold), cost=cost).metrics
        print(f"  {hold:5d} | {a['exposure_switches']:13d} {a['cost_drag_annual_bps']:12.1f} | "
              f"{b['exposure_switches']:12d} {b['cost_drag_annual_bps']:12.1f}")

    tab.to_csv("data/processed/brake_bootstrap.csv", index=False)
    print("\nwritten: data/processed/brake_bootstrap.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
