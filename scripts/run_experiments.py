#!/usr/bin/env python
"""Run the honest evaluation battery on the mechanical gold strategy."""

from __future__ import annotations

import sys

import pandas as pd

from vongold.backtest import buy_and_hold, run_backtest, summarize
from vongold.config import StrategyParams
from vongold.data import build_dataset
from vongold.experiments import (
    ablation,
    bootstrap_sharpe_ci,
    probabilistic_sharpe_ratio,
    sub_period_metrics,
    walk_forward,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:8.4f}")


def main() -> int:
    df = build_dataset("GLD", rng="10y")
    print(f"data: {len(df)} bars {df.index[0].date()} -> {df.index[-1].date()}\n")

    print("=" * 100)
    print("1. BENCHMARKS")
    print("=" * 100)
    bh = buy_and_hold(df)
    print(f"  buy & hold GLD : {summarize(bh)}")
    res_full = run_backtest(df)
    print(f"  mechanical     : {summarize(res_full)}")

    print("\n" + "=" * 100)
    print("2. ABLATION -- does each component earn its place?")
    print("=" * 100)
    ab = ablation(df)
    print(ab.to_string(index=False))

    print("\n" + "=" * 100)
    print("3. PER-YEAR STABILITY (mechanical strategy)")
    print("=" * 100)
    sp = sub_period_metrics(res_full, freq="YE")
    print(sp.to_string(index=False))

    print("\n" + "=" * 100)
    print("4. WALK-FORWARD FOLDS (5 equal slices)")
    print("=" * 100)
    wf = walk_forward(df, StrategyParams(), n_splits=5)
    print(wf.to_string(index=False))
    print(f"  folds beating buy&hold on Sharpe: {int(wf['beat_bh_sharpe'].sum())}/{len(wf)}")

    print("\n" + "=" * 100)
    print("5. STATISTICAL HONESTY")
    print("=" * 100)
    sr, lo, hi = bootstrap_sharpe_ci(res_full.returns)
    psr = probabilistic_sharpe_ratio(res_full.returns, benchmark_sr=0.0)
    print(f"  daily-Sharpe {sr:.4f}  95% block-bootstrap CI [{lo:.4f}, {hi:.4f}]")
    print(f"  PSR vs zero (per-period units): {psr:.4f}")
    print(f"  -> 'Sharpe > 0' holds at 95%? {'YES' if lo > 0 else 'NO'}")

    print("\n" + "=" * 100)
    print("6. SUB-PERIOD BREAKDOWN (10 slices, strategy vs buy&hold)")
    print("=" * 100)
    n = len(df)
    rows = []
    for i in range(10):
        sl = df.iloc[int(i * n / 10):int((i + 1) * n / 10)]
        if len(sl) < 60:
            continue
        a, b = run_backtest(sl), buy_and_hold(sl)
        rows.append(
            {
                "slice": f"{sl.index[0].date()}..{sl.index[-1].date()}",
                "strat_ret": a.equity.iloc[-1] / a.equity.iloc[0] - 1,
                "strat_dd": a.metrics["max_drawdown"],
                "bh_ret": b.equity.iloc[-1] / b.equity.iloc[0] - 1,
                "bh_dd": b.metrics["max_drawdown"],
                "strat_sharpe": a.metrics["sharpe"],
                "bh_sharpe": b.metrics["sharpe"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
