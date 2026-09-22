"""Turn a trading day's numbers into a von decision problem.

This is the piece that lets a 395M non-autoregressive decision model act as a
trader without being an LLM. Two rules make it honest:

1. **Facts only, no conclusions.** The state text contains measured quantities
   (returns, spreads, percentiles) and never the words "bullish", "buy", "should".
   The model's job is to map evidence to a labelled category, not to be sold a story.
   Selling it a story would just make von echo our own bias back at us.
2. **Criteria are fixed per question and never tuned per-day.** Tuning the criteria
   text is a way to smuggle in a fitted parameter; we fix it once here and test it.

Everything is computed from data up to and including day t's close, matching the
backtester's no-lookahead convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .strategy import TRADING_DAYS, build_features, realized_vol

# --- Fixed question texts. Never varied per-day; changing these is a strategy change. ---

REGIME_CRITERIA = {
    "up_trend": "The price series shows a sustained directional advance across most lookbacks.",
    "down_trend": "The price series shows a sustained directional decline across most lookbacks.",
    "range": "The price series shows no sustained direction; movement is sideways or conflicting.",
}

LONG_CRITERIA = {
    "true": "Evidence supports holding a long position in gold today.",
    "false": "Evidence does not support holding a long position in gold today.",
}

RISK_CRITERIA = {
    "true": "Near-term downside risk to a long gold position is elevated above its normal level.",
    "false": "Near-term downside risk to a long gold position is at or below its normal level.",
}

CONVICTION_CRITERIA = [
    "No conviction: evidence is conflicting or absent",
    "Low conviction: weak or single-source evidence",
    "Moderate conviction: several indicators align",
    "High conviction: most indicators align consistently",
    "Very high conviction: all indicators align strongly",
]


def build_von_problem(
    df: pd.DataFrame,
    i: int,
    vol_history: pd.Series | None = None,
) -> dict[str, Any] | None:
    """Build the von decision problem for the trading day at positional index i.

    Returns None when there is not enough history to describe the day meaningfully.
    """
    if i < 260:
        return None

    row = df.iloc[i]
    close = float(row["close"])
    hist = df["close"].iloc[: i + 1].astype(float)

    def ret(days: int) -> float | None:
        if len(hist) <= days:
            return None
        prev = float(hist.iloc[-days - 1])
        return close / prev - 1.0 if prev else None

    def pctile_of_last(series: pd.Series, window: int = 1260) -> float | None:
        """Percentile of the latest observation within its own trailing history."""
        s = series.dropna().tail(window)
        if len(s) < 60:
            return None
        return float((s <= s.iloc[-1]).mean())

    def monthly_yoy(series: pd.Series) -> float | None:
        """True year-over-year change for a monthly series.

        The daily frame forward-fills monthly releases, so differencing by a fixed
        number of ROWS would compare against the wrong month and silently report a
        nonsense figure. Collapse to one observation per calendar month first.
        """
        s = series.dropna()
        if s.empty:
            return None
        monthly = s.groupby([s.index.year, s.index.month]).last()
        if len(monthly) < 13:
            return None
        prev = float(monthly.iloc[-13])
        return float(monthly.iloc[-1]) / prev - 1.0 if prev else None

    vol21 = realized_vol(hist, 21)
    vol63 = realized_vol(hist, 63)
    vol_pct = pctile_of_last(vol21) if len(vol21.dropna()) > 60 else None

    hi252 = float(hist.tail(252).max())
    lo252 = float(hist.tail(252).min())
    pos_in_range = (close - lo252) / (hi252 - lo252) if hi252 > lo252 else 0.5

    ma200 = float(hist.tail(200).mean()) if len(hist) >= 200 else float(hist.mean())
    dist_ma = close / ma200 - 1.0

    lines = [
        "GOLD PRICE EVIDENCE",
        f"last close: {close:.2f}",
    ]

    for label, d in (("5 trading days", 5), ("20 trading days", 20),
                     ("60 trading days", 60), ("120 trading days", 120),
                     ("250 trading days", 250)):
        r = ret(d)
        if r is not None:
            lines.append(f"return over {label}: {r * 100:+.2f}%")

    v21 = float(vol21.iloc[-1]) if pd.notna(vol21.iloc[-1]) else None
    v63 = float(vol63.iloc[-1]) if pd.notna(vol63.iloc[-1]) else None
    if v21 is not None:
        lines.append(f"annualised realised volatility (21d): {v21 * 100:.1f}%")
    if v63 is not None:
        lines.append(f"annualised realised volatility (63d): {v63 * 100:.1f}%")
    if vol_pct is not None:
        lines.append(
            f"21d volatility percentile versus its own 5-year history: {vol_pct * 100:.0f}th"
        )

    lines.append(f"position within 252-day low/high range: {pos_in_range * 100:.0f}% of range")
    lines.append(f"distance from 200-day moving average: {dist_ma * 100:+.2f}%")

    above = int((hist.tail(20).diff().dropna() > 0).sum())
    lines.append(f"count of positive daily closes in last 20 sessions: {above} of 20")

    # --- Macro drivers: gold's documented opportunity cost and currency headwind ---
    macro_lines: list[str] = []
    if "real_yield_10y" in df.columns:
        ry = df["real_yield_10y"].astype(float).iloc[: i + 1].dropna()
        if len(ry) > 70:
            chg = float(ry.iloc[-1] - ry.iloc[-64])
            macro_lines.append(
                f"10-year real yield: {ry.iloc[-1]:.2f}%, change over 3 months: {chg:+.2f} "
                f"percentage points"
            )
            macro_lines.append(
                f"10-year real yield percentile versus its own 5-year history: "
                f"{pctile_of_last(ry) * 100:.0f}th"
            )
    if "dollar_index_broad" in df.columns:
        dx = df["dollar_index_broad"].astype(float).iloc[: i + 1].dropna()
        if len(dx) > 70:
            dxchg = float(dx.iloc[-1] / dx.iloc[-64] - 1.0)
            macro_lines.append(
                f"broad dollar index: {dx.iloc[-1]:.1f}, change over 3 months: {dxchg * 100:+.2f}%"
            )
    if "cpi" in df.columns:
        yoy = monthly_yoy(df["cpi"].astype(float).iloc[: i + 1])
        if yoy is not None:
            macro_lines.append(f"US CPI inflation year over year: {yoy * 100:.2f}%")
    if "fed_funds_rate" in df.columns:
        ff = df["fed_funds_rate"].astype(float).iloc[: i + 1].dropna()
        if len(ff) > 70:
            macro_lines.append(
                f"effective federal funds rate: {ff.iloc[-1]:.2f}%, "
                f"change over 3 months: {float(ff.iloc[-1] - ff.iloc[-64]):+.2f} percentage points"
            )

    if macro_lines:
        lines.append("")
        lines.append("MACRO DRIVER EVIDENCE")
        lines.extend(macro_lines)

    state = "\n".join(lines)

    return {
        "state": state,
        "questions": {
            "regime": {"type": "choice", "instructions":
                       "Classify the prevailing directional character of the gold price series.",
                       "criteria": REGIME_CRITERIA},
            "long_justified": {"type": "noul", "instructions":
                               "Based only on the evidence provided, is holding a long "
                               "position in gold justified today?",
                               "criteria": LONG_CRITERIA},
            "risk_elevated": {"type": "noul", "instructions":
                              "Is the near-term downside risk of a long gold position "
                              "elevated relative to normal conditions?",
                              "criteria": RISK_CRITERIA},
            "conviction": {"type": "score", "instructions":
                           "Rate the strength of the evidence for holding gold today.",
                           "criteria": CONVICTION_CRITERIA},
        },
    }


def states_for_backtest(df: pd.DataFrame, start_index: int = 260) -> list[dict[str, Any]]:
    """Build the von problem for every day from `start_index` onward."""
    out = []
    for i in range(start_index, len(df)):
        prob = build_von_problem(df, i)
        if prob is None:
            continue
        prob["_date"] = df.index[i].date().isoformat()
        prob["_i"] = i
        out.append(prob)
    return out
