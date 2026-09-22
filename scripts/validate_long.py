#!/usr/bin/env python
"""Two tests the literature says you must run before believing anything.

**Test A -- the lagged macro gate.** The contemporaneous relationship between gold and
real yields / the dollar is powerful and well documented (Chicago Fed Letter 464; Erb &
Harvey "The Golden Dilemma"), BUT:
  * the Chicago Fed paper states the real-rate relationship "does not show up in these
    data before 2001";
  * Erb & Harvey show the headline -0.82 correlation collapses to -0.31 once a time trend
    is removed, and explicitly raise data mining as the explanation;
  * both are CONTEMPORANEOUS, not predictive.

A contemporaneous correlation is description, not edge. So this test uses only
information available at day t-1 to gate the day-t position, and reports whether the
gate adds anything over the ungated strategy. If it does not, the filter is decoration.

**Test B -- the overfitting battery.** Pre-registered parameter set, one shot on 58
years, with:
  * the Deflated Sharpe Ratio and the Sharpe the best of N trials would show by luck;
  * a block bootstrap CI (i.i.d. resampling understates uncertainty in a trending series);
  * minimum track record length -- how many years are needed before the observed Sharpe
    is statistically distinguishable from zero.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vongold.backtest import buy_and_hold, run_backtest
from vongold.config import CostModel, StrategyParams
from vongold.experiments import (
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_of,
)
from vongold.strategy import apply_rebalance_band, build_features, mechanical_exposure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from long_history import load_lbma  # reuse the loader, do not duplicate the parsing

pd.set_option("display.width", 220)
TRADING_DAYS = 252


def min_track_record_length(returns: pd.Series, benchmark_sr: float = 0.0,
                            confidence: float = 0.95) -> float:
    """Years needed for the observed Sharpe to beat `benchmark_sr` at `confidence`.

    Bailey & Lopez de Prado's minimum track record length, in per-period units then
    converted to years.
    """
    r = returns.dropna()
    sr = sharpe_of(r)
    if sr <= benchmark_sr or len(r) < 3:
        return float("inf")
    g3 = float(r.skew())
    g4 = float(r.kurtosis()) + 3.0
    # z for the one-sided confidence level
    from vongold.experiments import _norm_ppf

    z = _norm_ppf(confidence)
    denom = (sr - benchmark_sr) ** 2
    var_term = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2
    n = 1.0 + var_term * (z / math.sqrt(denom)) ** 2
    return float(n / TRADING_DAYS)


def main() -> int:
    cost = CostModel()
    lbma = load_lbma()
    # SHIPPED defaults (macro gates already off, band 0.10, brake 10) -- same reasoning
    # as long_history.py: the documented numbers must come from the shipped config.
    base_p = StrategyParams()

    print("=" * 100)
    print("TEST A: LAGGED MACRO GATE -- is the real-yield / dollar relationship tradable?")
    print("=" * 100)

    # Build the driver series from FRED CSVs already on disk.
    def fred(name: str) -> pd.Series | None:
        p = Path("data/raw") / f"fred_{name}.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
                      index=pd.to_datetime(df.iloc[:, 0]))
        s.index = s.index.astype("datetime64[ms]")
        return s[~s.index.duplicated(keep="last")].sort_index().dropna()

    # The tradable window is set by macro availability: TIPS from 2003, broad dollar 2006.
    gld = pd.read_parquet("data/processed/gld_yfinance_max.parquet")
    gld.columns = [str(c).lower().replace(" ", "_") for c in gld.columns]
    px = gld[["open", "high", "low", "close", "volume"]].copy()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None).normalize().astype("datetime64[ms]")

    ry = fred("DFII10")
    dx = fred("DTWEXBGS")
    if ry is None:
        print("  DFII10 not on disk -- skipping")
        return 1
    px["real_yield_10y"] = ry.reindex(px.index, method="ffill")
    if dx is not None:
        px["dollar_index_broad"] = dx.reindex(px.index, method="ffill")
    px = px.dropna(subset=["close"])
    print(f"  window with macro data: {px.index[0].date()} -> {px.index[-1].date()} ({len(px)} bars)\n")

    # Contemporaneous correlations, to reproduce the published direction.
    dr = px["close"].pct_change()
    if "real_yield_10y" in px:
        print(f"  contemporaneous corr(daily gold ret, d(real yield))  = "
              f"{dr.corr(px['real_yield_10y'].diff()):+.3f}")
    if "dollar_index_broad" in px:
        print(f"  contemporaneous corr(daily gold ret, d(dollar idx))  = "
              f"{dr.corr(px['dollar_index_broad'].diff()):+.3f}")

    # LAGGED predictive correlations: does yesterday's driver change predict today's return?
    print("\n  LAGGED predictive corr (driver change at t-1 vs gold return at t):")
    for col, label in (("real_yield_10y", "real yield"), ("dollar_index_broad", "dollar")):
        if col not in px:
            continue
        chg = px[col].diff()
        for lag in (1, 5, 21):
            c = dr.corr(chg.shift(lag))
            print(f"    {label:11s} lag {lag:2d}d: corr {c:+.4f}")

    # The real test: does the gate improve the strategy, or only reduce it?
    print("\n  strategy A/B (gate uses only data through day t-1, as the backtester enforces):")
    rows = []
    for label, p in (
        ("ungated (baseline)", base_p),
        ("+ real-yield gate", StrategyParams(use_real_yield_filter=True)),
        ("+ dollar gate", StrategyParams(use_dollar_filter=True)),
        ("+ both gates", StrategyParams(use_real_yield_filter=True, use_dollar_filter=True)),
    ):
        res = run_backtest(px, params=p, cost=cost)
        m = res.metrics
        rows.append({"config": label, "cagr": m["cagr"], "vol": m["vol_annual"],
                     "sharpe": m["sharpe"], "max_dd": m["max_drawdown"],
                     "calmar": m["calmar"], "in_mkt": m["pct_time_in_market"]})
    tab = pd.DataFrame(rows)
    base_sr = tab.iloc[0]["sharpe"]
    tab["d_sharpe"] = tab["sharpe"] - base_sr
    print(tab.to_string(index=False))
    print("\n  VERDICT: a gate earns its place only if d_sharpe > 0 on a LAGGED basis.")
    for _, r in tab.iloc[1:].iterrows():
        verdict = "ADDS" if r["d_sharpe"] > 0 else "HURTS"
        print(f"    {r['config']:20s} {verdict} (d_sharpe {r['d_sharpe']:+.3f})")

    print("\n" + "=" * 100)
    print("TEST B: OVERFITTING BATTERY on 58 years (pre-registered parameter set)")
    print("=" * 100)
    res = run_backtest(lbma, params=base_p, cost=cost)
    bh = buy_and_hold(lbma, cost=cost)
    r = res.returns.dropna()
    sr_daily, lo, hi = bootstrap_sharpe_ci(r, n_boot=3000, block=21)
    psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    mtrl = min_track_record_length(r, benchmark_sr=0.0, confidence=0.95)
    # We effectively examined a handful of configurations, not thousands. Declaring the
    # honest trial count is the whole point -- understating it inflates the DSR.
    n_trials = 12
    dsr, sr0 = deflated_sharpe_ratio(r, n_trials=n_trials,
                                     sr_variance=float(res.metrics["sharpe"] ** 2 / (12 * TRADING_DAYS)))

    print(f"  bars                     : {len(r)}")
    print(f"  observed annualised Sharpe: {res.metrics['sharpe']:.3f}  (buy & hold {bh.metrics['sharpe']:.3f})")
    print(f"  pct months positive       : {res.metrics['pct_months_positive']:.1%}")
    print(f"  block-bootstrap 95% CI    : [{lo * math.sqrt(TRADING_DAYS):.3f}, "
          f"{hi * math.sqrt(TRADING_DAYS):.3f}] annualised")
    print(f"    -> 'Sharpe > 0' at 95%? {'YES' if lo > 0 else 'NO'}")
    print(f"  Probabilistic Sharpe      : {psr:.4f}  ({'clears' if psr > 0.95 else 'below'} the 0.95 bar)")
    print(f"  Deflated Sharpe ({n_trials} trials): {dsr:.4f}")
    print(f"    Sharpe the best of {n_trials} random designs would show by luck: "
          f"{sr0 * math.sqrt(TRADING_DAYS):.3f} annualised")
    print(f"  Min track record length   : {mtrl:.1f} years at 95% confidence")

    # Rolling out-of-sample consistency: 10-year windows, does it ever go badly wrong?
    print("\n  rolling 10-year windows (strategy vs buy&hold Sharpe):")
    window = 10 * TRADING_DAYS
    wins = 0
    total = 0
    for start in range(0, len(r) - window, TRADING_DAYS):
        sl = r.iloc[start:start + window]
        sl_bh = bh.returns.iloc[start:start + window]
        s_sr = sharpe_of(sl) * math.sqrt(TRADING_DAYS)
        b_sr = sharpe_of(sl_bh) * math.sqrt(TRADING_DAYS)
        total += 1
        if s_sr > b_sr:
            wins += 1
    print(f"    strategy beat buy&hold in {wins}/{total} overlapping 10-year windows "
          f"({wins / max(total, 1):.1%})")

    pd.DataFrame(rows).to_csv("data/processed/macro_gate_test.csv", index=False)
    print("\nwritten: data/processed/macro_gate_test.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
