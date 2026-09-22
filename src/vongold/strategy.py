"""Signal construction for the gold strategy.

The mechanical core is a multi-horizon, volatility-targeted time-series-momentum
signal with two documented macro filters. Every input is lagged so that a decision
made on day t uses only information available at day t's close, and is executed on
day t+1.

Why this shape (see docs/STRATEGY.md for citations):
  * Time-series momentum is one of the few effects with out-of-sample evidence
    across asset classes including commodities (Moskowitz/Ooi/Pedersen 2012;
    Hurst/Ooi/Pedersen "A Century of Evidence on Trend-Following Investing").
  * Averaging several lookbacks reduces the variance of the signal and the
    sensitivity to any single window length (a classic overfitting defence).
  * Volatility targeting is documented to improve risk-adjusted returns and
    reduce drawdowns at moderate target levels (Harvey et al., "The Impact of
    Volatility Targeting").
  * Real yields and the dollar are gold's best-documented macro drivers; using
    them as slow filters rather than fast signals avoids competing with the
    faster trend signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyParams

TRADING_DAYS = 252


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Annualised close-to-close realised volatility."""
    ret = close.pct_change()
    return ret.rolling(window).std() * np.sqrt(TRADING_DAYS)


def time_series_momentum(close: pd.Series, lookbacks: tuple[int, ...]) -> pd.Series:
    """Average sign of momentum across several lookbacks, in [-1, 1].

    Averaging signs rather than raw returns keeps the signal bounded and stops the
    longest, largest-magnitude lookback from dominating purely by scale.
    """
    signs = []
    for k in lookbacks:
        mom = close / close.shift(k) - 1.0
        signs.append(np.sign(mom))
    if not signs:
        return pd.Series(0.0, index=close.index)
    return pd.concat(signs, axis=1).mean(axis=1).fillna(0.0)


def _optional_trend(series: pd.Series, days: int, falling_is_supportive: bool) -> pd.Series:
    """Return a 0/1 support flag from a driver series' own trend.

    `falling_is_supportive=True` for real yields and the dollar: gold benefits when
    the opportunity cost of holding it falls and when the dollar weakens.
    """
    delta = series.diff(days)
    flag = (delta <= 0) if falling_is_supportive else (delta >= 0)
    return flag.astype(float).fillna(0.5)


def build_features(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Compute all features needed for a decision on each row's date."""
    close = df["close"].astype(float)
    f = pd.DataFrame(index=df.index)

    f["close"] = close
    f["ret1"] = close.pct_change()
    f["vol"] = realized_vol(close, p.vol_window)
    f["tsmom"] = time_series_momentum(close, p.momentum_lookbacks)

    if p.trend_filter_days > 0:
        f["above_ma"] = (close > close.rolling(p.trend_filter_days).mean()).astype(float)
    else:
        f["above_ma"] = 1.0

    # --- Macro conditioning ---
    if p.use_real_yield_filter and "real_yield_10y" in df.columns:
        f["real_yield"] = df["real_yield_10y"]
        f["real_yield_support"] = _optional_trend(
            df["real_yield_10y"].astype(float), p.real_yield_trend_days, falling_is_supportive=True
        )
    else:
        f["real_yield"] = np.nan
        f["real_yield_support"] = 1.0

    if p.use_dollar_filter and "dollar_index_broad" in df.columns:
        f["dollar"] = df["dollar_index_broad"]
        f["dollar_support"] = _optional_trend(
            df["dollar_index_broad"].astype(float), p.dollar_trend_days, falling_is_supportive=True
        )
    else:
        f["dollar"] = np.nan
        f["dollar_support"] = 1.0

    # --- ATR for optional protective stops ---
    if p.atr_stop_mult > 0:
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                df["high"].astype(float) - df["low"].astype(float),
                (df["high"].astype(float) - prev_close).abs(),
                (df["low"].astype(float) - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        f["atr"] = tr.rolling(p.atr_window).mean()

    return f


def mechanical_exposure(f: pd.DataFrame, p: StrategyParams, allow_short: bool = False) -> pd.Series:
    """Target exposure BEFORE any decision-model overlay.

    Steps (each defensible on its own):
      1. direction  = average momentum sign (long-only => clipped to [0,1])
      2. size       = target_vol / realised_vol, capped (volatility targeting)
      3. filters    = trend MA and macro regime flags, applied multiplicatively
      4. cap        = max_exposure (no leverage)
    """
    direction = f["tsmom"].clip(-1, 1)
    if not allow_short:
        direction = direction.clip(lower=0.0)

    vol = f["vol"].replace(0.0, np.nan)
    size = (p.target_vol_annual / vol).clip(upper=p.max_exposure)
    # Before the vol window is full there is no estimate: stay flat rather than
    # guess a size.
    size = size.fillna(0.0)

    raw = direction * size * f["above_ma"] * f["real_yield_support"] * f["dollar_support"]
    if allow_short:
        return raw.clip(-p.max_exposure, p.max_exposure)
    return raw.clip(0.0, p.max_exposure)


def apply_rebalance_band(target: pd.Series, band: float) -> pd.Series:
    """Hysteresis: only move when the target has drifted more than `band`.

    Cuts turnover (and therefore cost) without changing the signal's direction.
    """
    out = np.empty(len(target), dtype=float)
    held = 0.0
    vals = target.to_numpy()
    for i, t in enumerate(vals):
        if not np.isfinite(t):
            t = 0.0
        if abs(t - held) >= band:
            held = t
        out[i] = held
    return pd.Series(out, index=target.index, name="target_exposure")
