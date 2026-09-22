#!/usr/bin/env python
"""Diagnose WHY the strategy is flat so often, and test alternative designs."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from vongold.backtest import buy_and_hold, run_backtest, summarize
from vongold.config import CostModel, StrategyParams
from vongold.data import build_dataset
from vongold.experiments import deflated_sharpe_ratio
from vongold.strategy import build_features, mechanical_exposure

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:9.4f}")


def diag(df: pd.DataFrame) -> None:
    print("=" * 100)
    print("A. WHICH FILTER IS SUPPRESSING EXPOSURE?  (fraction of days each gate is ON)")
    print("=" * 100)
    p = StrategyParams()
    f = build_features(df, p)
    print(f"  tsmom > 0                : {(f['tsmom'] > 0).mean():.3f}")
    print(f"  tsmom == 0 (mixed)       : {(f['tsmom'] == 0).mean():.3f}")
    print(f"  above 200d MA            : {(f['above_ma'] > 0).mean():.3f}")
    print(f"  real-yield supportive    : {(f['real_yield_support'] > 0).mean():.3f}")
    print(f"  dollar supportive        : {(f['dollar_support'] > 0).mean():.3f}")
    allon = ((f["tsmom"] > 0) & (f["above_ma"] > 0) & (f["real_yield_support"] > 0)
             & (f["dollar_support"] > 0))
    print(f"  ALL four gates on        : {allon.mean():.3f}")
    print(f"  vol window available     : {f['vol'].notna().mean():.3f}")
    print(f"\n  mean tsmom               : {f['tsmom'].mean():.3f}")
    print(f"  mean vol (ann)           : {f['vol'].mean():.3f}")
    print(f"  implied size @12% target : {(p.target_vol_annual / f['vol']).clip(upper=1.0).mean():.3f}")

    print("\n  unconditional gold return stats:")
    r = f["ret1"].dropna()
    print(f"    ann return {r.mean()*252:.2%}  ann vol {r.std()*np.sqrt(252):.2%}")


def test_variants(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 100)
    print("B. ALTERNATIVE DESIGNS  (all long-only, max exposure 1.0)")
    print("=" * 100)
    c = CostModel()
    bh = buy_and_hold(df)

    rows = []

    def add(name: str, res, note: str = "") -> None:
        m = res.metrics
        rows.append({
            "design": name,
            "cagr": m["cagr"],
            "vol": m["vol_annual"],
            "sharpe": m["sharpe"],
            "max_dd": m["max_drawdown"],
            "calmar": m["calmar"],
            "in_mkt": m["pct_time_in_market"],
            "switches": m["exposure_switches"],
            "note": note,
        })

    add("buy & hold", bh, "benchmark")
    add("current default", run_backtest(df), "all 4 gates")

    # --- progressively simpler gates ---
    add("trend MA only", run_backtest(
        df, params=StrategyParams(use_real_yield_filter=False, use_dollar_filter=False)))
    add("trend MA + real-yield", run_backtest(
        df, params=StrategyParams(use_dollar_filter=False, use_real_yield_filter=True)))
    add("momentum only (no MA)", run_backtest(
        df, params=StrategyParams(trend_filter_days=0, use_real_yield_filter=False,
                                  use_dollar_filter=False)))

    # --- the key alternative: default LONG, use trend only as a crash brake ---
    # Mechanically: exposure = 1 when close > 200d MA else 0, vol-targeted.
    p = StrategyParams(momentum_lookbacks=(252,), trend_filter_days=200,
                       use_real_yield_filter=False, use_dollar_filter=False,
                       rebalance_band=0.5)
    add("long-biased 200MA brake", run_backtest(df, params=p), "no momentum sign")

    # --- vol target sweep with trend MA only ---
    for tv in (0.10, 0.12, 0.16, 0.20, 0.25, 1.00):
        p = StrategyParams(target_vol_annual=tv, use_real_yield_filter=False,
                           use_dollar_filter=False)
        add(f"trend MA, vol target {tv:.0%}", run_backtest(df, params=p))

    # --- MA length sweep ---
    for ma in (100, 150, 200, 250):
        p = StrategyParams(trend_filter_days=ma, use_real_yield_filter=False,
                           use_dollar_filter=False, target_vol_annual=0.16)
        add(f"MA{ma}, vol target 16%", run_backtest(df, params=p))

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


def sweep_robust(df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("C. MULTIPLE-TESTING: sweep MA x vol-target, then deflate the best Sharpe")
    print("=" * 100)
    c = CostModel()
    rows = []
    for ma in (50, 100, 150, 200, 250, 300):
        for tv in (0.08, 0.10, 0.12, 0.16, 0.20, 0.30):
            for rb in (0.05, 0.15, 0.30):
                p = StrategyParams(trend_filter_days=ma, target_vol_annual=tv,
                                   rebalance_band=rb, use_real_yield_filter=False,
                                   use_dollar_filter=False)
                r = run_backtest(df, params=p, cost=c)
                m = r.metrics
                rows.append({"ma": ma, "vol_target": tv, "band": rb,
                             "cagr": m["cagr"], "sharpe": m["sharpe"],
                             "max_dd": m["max_drawdown"], "in_mkt": m["pct_time_in_market"],
                             "_ret": r.returns})
    out = pd.DataFrame(rows)
    n_trials = len(out)
    sr_var = float(out["sharpe"].var(ddof=1) / 252)
    dsr, sr0 = [], []
    for r in out["_ret"]:
        d, s = deflated_sharpe_ratio(r, n_trials=n_trials, sr_variance=sr_var)
        dsr.append(d)
        sr0.append(s)
    out["dsr"] = dsr
    out["sr_expected_max_luck"] = sr0
    out = out.drop(columns=["_ret"])
    best = out.sort_values("sharpe", ascending=False).head(10)
    print(f"  n_trials = {n_trials}")
    print(f"  annualised-Sharpe you'd expect the BEST of {n_trials} random designs to show by luck:")
    print(f"    {sr0[0]*np.sqrt(252):.3f} annualised")
    print("\n  top 10 by raw Sharpe:")
    print(best.to_string(index=False))
    print("\n  median across all trials:")
    print(out[["cagr", "sharpe", "max_dd", "in_mkt"]].median().to_string())
    out.to_csv("data/processed/sweep_ma_vol.csv", index=False)
    print("\n  full sweep written to data/processed/sweep_ma_vol.csv")


def main() -> int:
    df = build_dataset("GLD", rng="10y")
    diag(df)
    test_variants(df)
    sweep_robust(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
