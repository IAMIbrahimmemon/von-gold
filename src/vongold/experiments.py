"""Experiment harness: component ablation, parameter sweeps, and overfitting tests.

The point of this module is to make it HARD to fool ourselves. Three disciplines:

1. **Ablation** -- every filter must earn its place. If removing a component does not
   hurt out-of-sample performance, it was decoration (or worse, a fitted artefact).
2. **Sub-period stability** -- a strategy whose edge lives in one 3-year window is a
   regime bet, not an edge. We report per-year and per-sub-period metrics.
3. **Multiple-testing correction** -- we test many configurations, so the best one is
   biased upward. The Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) adjusts
   the observed Sharpe for the number of trials and for non-normality of returns.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import pandas as pd

from .backtest import BacktestResult, buy_and_hold, run_backtest
from .config import BacktestConfig, CostModel, StrategyParams
from .strategy import build_features

EULER_MASCHERONI = 0.5772156649
TRADING_DAYS = 252


# --------------------------------------------------------------------------------------
# Statistical tools
# --------------------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def sharpe_of(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    # Non-annualised Sharpe, as the DSR formula expects per-period units.
    return float(r.mean() / r.std())


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """PSR: probability the true Sharpe exceeds `benchmark_sr`.

    Accounts for track-record length, skew and kurtosis (Bailey & Lopez de Prado).
    """
    r = returns.dropna()
    n = len(r)
    if n < 3:
        return 0.0
    sr = sharpe_of(r)
    g3 = float(r.skew())
    g4 = float(r.kurtosis()) + 3.0  # pandas reports excess kurtosis
    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2
    if denom <= 0:
        return 0.0
    z = (sr - benchmark_sr) * math.sqrt(n - 1) / math.sqrt(denom)
    return _norm_cdf(z)


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int,
                          sr_variance: float | None = None) -> tuple[float, float]:
    """DSR and the expected-maximum-Sharpe benchmark it is measured against.

    Returns (dsr, sr0) where sr0 is the Sharpe you would expect the BEST of
    `n_trials` independent random strategies to achieve by luck alone. Reporting
    the benchmark matters: a DSR below ~0.95 means the measured Sharpe is not
    distinguishable from selection noise.
    """
    r = returns.dropna()
    n = len(r)
    if n < 3 or n_trials < 1:
        return 0.0, 0.0
    sr = sharpe_of(r)
    var_sr = sr_variance if (sr_variance is not None and sr_variance > 0) else 1.0 / n
    g = EULER_MASCHERONI
    sr0 = math.sqrt(var_sr) * (
        (1.0 - g) * _norm_ppf(1.0 - 1.0 / n_trials)
        + g * _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    g3 = float(r.skew())
    g4 = float(r.kurtosis()) + 3.0
    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2
    if denom <= 0:
        return 0.0, float(sr0)
    z = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(denom)
    return _norm_cdf(z), float(sr0)


def sub_period_metrics(res: BacktestResult, freq: str = "YE") -> pd.DataFrame:
    """Metrics per calendar period -- the stability check."""
    r = res.returns
    rows = []
    for period, grp in r.groupby(pd.Grouper(freq=freq)):
        if len(grp) < 20:
            continue
        eq = (1.0 + grp).cumprod()
        dd = float((eq / eq.cummax() - 1.0).min())
        vol = grp.std() * np.sqrt(TRADING_DAYS)
        rows.append(
            {
                "period": period.year if freq.startswith("Y") else str(period.date()),
                "ret": float(eq.iloc[-1] - 1.0),
                "vol": float(vol),
                "sharpe": float(grp.mean() * TRADING_DAYS / vol) if vol > 0 else 0.0,
                "max_dd": dd,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Ablation
# --------------------------------------------------------------------------------------

def ablation(df: pd.DataFrame, base: StrategyParams | None = None,
             cost: CostModel | None = None) -> pd.DataFrame:
    """Disable one component at a time and measure the damage (or lack of it)."""
    base = base or StrategyParams()
    variants: list[tuple[str, StrategyParams]] = [("FULL (all components)", base)]

    off = replace(base, use_real_yield_filter=False)
    variants.append(("minus real-yield filter", off))
    off = replace(base, use_dollar_filter=False)
    variants.append(("minus dollar filter", off))
    off = replace(base, trend_filter_days=0)
    variants.append(("minus trend MA filter", off))
    off = replace(base, momentum_lookbacks=(252,))
    variants.append(("single 252d momentum", off))
    off = replace(base, momentum_lookbacks=(21,))
    variants.append(("single 21d momentum", off))
    off = replace(base, target_vol_annual=1.0)
    variants.append(("no vol targeting", off))
    off = replace(base, rebalance_band=0.0)
    variants.append(("no rebalance band", off))
    off = replace(base, atr_stop_mult=3.0)
    variants.append(("with 3xATR stop", off))

    rows = []
    for name, p in variants:
        try:
            res = run_backtest(df, params=p, cost=cost)
        except Exception as exc:
            rows.append({"variant": name, "error": str(exc)})
            continue
        m = res.metrics
        rows.append(
            {
                "variant": name,
                "cagr": m["cagr"],
                "vol": m["vol_annual"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_drawdown"],
                "calmar": m["calmar"],
                "in_mkt": m["pct_time_in_market"],
                "switches": m["exposure_switches"],
                "cost_bps": m["cost_drag_annual_bps"],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Sweep + multiple testing
# --------------------------------------------------------------------------------------

def sweep(df: pd.DataFrame, grid: dict[str, list], cost: CostModel | None = None,
          allow_short: bool = False) -> pd.DataFrame:
    """Grid search. `n_trials` in the result feeds the Deflated Sharpe Ratio."""
    keys = list(grid.keys())
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        kwargs = dict(zip(keys, combo))
        p = StrategyParams(**kwargs)
        try:
            res = run_backtest(df, params=p, cost=cost, allow_short=allow_short)
        except Exception:
            continue
        m = res.metrics
        rows.append({**kwargs, "cagr": m["cagr"], "sharpe": m["sharpe"],
                     "max_dd": m["max_drawdown"], "calmar": m["calmar"],
                     "in_mkt": m["pct_time_in_market"],
                     "_returns": res.returns})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    n_trials = len(out)
    sr_var = float(out["sharpe"].var(ddof=1) / TRADING_DAYS) if n_trials > 1 else None
    dsr_col, sr0_col = [], []
    for r in out["_returns"]:
        d, s0 = deflated_sharpe_ratio(r, n_trials=n_trials, sr_variance=sr_var)
        dsr_col.append(d)
        sr0_col.append(s0)
    out["deflated_sharpe"] = dsr_col
    out["benchmark_sr_expected_max"] = sr0_col
    out["n_trials"] = n_trials
    return out.drop(columns=["_returns"])


# --------------------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------------------

def walk_forward(df: pd.DataFrame, params: StrategyParams, cost: CostModel | None = None,
                 n_splits: int = 5, allow_short: bool = False) -> pd.DataFrame:
    """Report the strategy on each consecutive slice independently.

    Our parameters are fixed a priori (not fitted on this data), so this is a
    stability report rather than true walk-forward optimisation. If a strategy needs
    per-fold refitting to work, that is itself a finding.
    """
    n = len(df)
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    rows = []
    for i in range(n_splits):
        lo, hi = bounds[i], bounds[i + 1]
        if hi - lo < 60:
            continue
        sl = df.iloc[lo:hi]
        res = run_backtest(sl, params=params, cost=cost, allow_short=allow_short)
        bh = buy_and_hold(sl)
        m, bm = res.metrics, bh.metrics
        rows.append(
            {
                "fold": i + 1,
                "start": sl.index[0].date(),
                "end": sl.index[-1].date(),
                "strat_cagr": m["cagr"],
                "strat_sharpe": m["sharpe"],
                "strat_maxdd": m["max_drawdown"],
                "bh_cagr": bm["cagr"],
                "bh_sharpe": bm["sharpe"],
                "bh_maxdd": bm["max_drawdown"],
                "beat_bh_sharpe": m["sharpe"] > bm["sharpe"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_sharpe_ci(returns: pd.Series, n_boot: int = 2000, block: int = 21,
                        seed: int = 7) -> tuple[float, float, float]:
    """Stationary block bootstrap CI for the Sharpe ratio.

    Block resampling preserves some autocorrelation; i.i.d. resampling would
    understate the uncertainty of a trend-following equity curve.
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < block * 2:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        sd = sample.std()
        out[b] = sample.mean() / sd if sd > 0 else 0.0
    sr = float(r.mean() / r.std())
    return sr, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
