"""The von decision-model overlay.

Design contract (this is the honest part):

  * von NEVER sizes a position and never sets a number. It returns a label and a
    calibrated probability. Sizing stays with the volatility-targeted mechanical core.
  * von's contribution must be DEMONSTRATED. `evaluate_overlay` runs the overlay and
    its mechanical twin on identical data and reports whether von helped after costs
    and after multiple-testing correction. If it does not help, the honest output is
    "do not use the overlay" -- not a re-tuned overlay.
  * Because the overlay is evaluated on the same history used to design it, the
    result is in-sample. `overlay_out_of_sample` splits the history and reports the
    second half separately, which is the number that actually matters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import CostModel, StrategyParams

# How a von answer is turned into an exposure multiplier. Fixed, not fitted.
MODES = {
    # Only trade when von agrees the long is justified.
    "veto": {"disagree_scale": 0.0},
    # Halve exposure when von disagrees (softer).
    "halve": {"disagree_scale": 0.5},
    # Scale by von's stated probability directly.
    "prob": None,
    # Scale by the percentile RANK of today's read within its trailing distribution.
    # Measured best of the four on real gold data, though still within noise -- see
    # docs/VON.md. Prefer this over the thresholding modes: it has no free parameter and
    # consumes the ordering, which is the part of von's output that carried signal.
    "rank": None,
}


@dataclass
class VonAnswer:
    date: str
    regime: str
    regime_prob: float
    regime_confidence: float
    long_prob: float
    risk_prob: float
    conviction: float       # 0-4 scale from the score head
    conviction_conf: float
    error: str | None = None


def load_answers(path: str | Path) -> pd.DataFrame:
    """Load the JSONL produced by scripts/von_batch.py into a tidy frame."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("error"):
            rows.append({"date": rec.get("date"), "error": rec["error"]})
            continue
        a = rec.get("answers", {})
        reg = a.get("regime", {}) or {}
        rows.append(
            {
                "date": rec["date"],
                "regime": reg.get("choice"),
                "regime_prob": max((reg.get("probabilities") or {"": 0}).values(), default=0.0),
                "regime_confidence": reg.get("confidence", 0.0),
                "long_prob": (a.get("long_justified", {}) or {}).get("noul", np.nan),
                "risk_prob": (a.get("risk_elevated", {}) or {}).get("noul", np.nan),
                "conviction": (a.get("conviction", {}) or {}).get("score", np.nan),
                "conviction_conf": (a.get("conviction", {}) or {}).get("confidence", np.nan),
                "error": None,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def overlay_exposure(
    answers: pd.DataFrame,
    dates: pd.DatetimeIndex,
    mode: str = "veto",
    min_confidence: float = 0.60,
    risk_elevated_cut: float = 0.50,
) -> pd.Series:
    """Map von answers onto a per-day exposure MULTIPLIER in [0, 1].

    The multiplier is applied to the mechanical target. A multiplier of 1 leaves the
    mechanical decision untouched; 0 vetoes it. von can only reduce risk -- it can
    never create a position the mechanical core would not have taken. That asymmetry
    is deliberate: a 395M classifier has no business initiating risk.
    """
    ans = answers.reindex(dates).ffill()
    scale = pd.Series(1.0, index=dates)

    long_prob = ans["long_prob"].astype(float)
    risk_prob = ans["risk_prob"].astype(float)
    conf = ans["regime_confidence"].astype(float)

    if mode == "veto":
        disagree = (long_prob < 0.5) | (risk_prob >= risk_elevated_cut)
        low_conf = conf < min_confidence
        scale = np.where(disagree | low_conf, 0.0, 1.0)
        return pd.Series(scale, index=dates).fillna(1.0)

    if mode == "halve":
        disagree = (long_prob < 0.5) | (risk_prob >= risk_elevated_cut)
        scale = np.where(disagree, 0.5, 1.0)
        return pd.Series(scale, index=dates).fillna(1.0)

    if mode == "prob":
        # Blend von's probability with certainty by its own confidence.
        p = long_prob
        c = conf.fillna(0.0).clip(0, 1)
        adj = 0.5 + (p - 0.5) * c * 2.0
        # NO ANSWER MEANS NO VIEW: leave the mechanical decision untouched. Without
        # this, an absent/failed model answer would silently halve exposure -- a real
        # bug caught by tests/test_invariants.py, not a harmless default.
        adj = adj.where(p.notna() & (c > 0), 1.0)
        return pd.Series(adj, index=dates).clip(0.0, 1.0).fillna(1.0)

    if mode == "rank":
        # Use von's probability as a RELATIVE read, not an absolute level.
        #
        # Why: measured on 2,252 real decisions, von answered "range" 100% of the time
        # and never once returned P(long justified) above 0.27 -- its absolute
        # calibration is broken for this domain. But its answers were still positively
        # rank-correlated with forward gold returns (+0.16 at 21 days). So the level is
        # meaningless while the ordering carries a weak signal. This mode consumes only
        # the ordering: exposure scales with where today's read sits in the trailing
        # distribution of recent reads.
        #
        # This is deliberately NOT a fitted threshold. It is a percentile rank, which
        # has no free parameter to tune, and it cannot exceed 1.0 so it can only ever
        # reduce exposure relative to the mechanical target.
        p = long_prob
        if p.notna().sum() < 60:
            return pd.Series(1.0, index=dates)
        window = 252
        rank = p.rolling(window, min_periods=60).apply(
            lambda w: float((w <= w[-1]).mean()), raw=True
        )
        # Map the rank onto [floor, 1] so the worst-ranked reads trade smaller rather
        # than not at all: a rank-based tilt, not a second veto layer on top of the
        # mechanical gates.
        rank = rank.where(p.notna(), 1.0).fillna(1.0)
        return pd.Series(rank, index=dates).clip(0.0, 1.0)

    raise ValueError(f"unknown overlay mode: {mode}")


def evaluate_overlay(
    df: pd.DataFrame,
    answers: pd.DataFrame,
    base_params: StrategyParams | None = None,
    cost: CostModel | None = None,
    mode: str = "veto",
    min_confidence: float = 0.60,
    risk_elevated_cut: float = 0.50,
) -> dict:
    """Compare the mechanical strategy with and without the von overlay."""
    p = base_params or StrategyParams()
    c = cost or CostModel()

    base = run_backtest(df, params=p, cost=c)
    mult = overlay_exposure(answers, df.index, mode=mode, min_confidence=min_confidence,
                            risk_elevated_cut=risk_elevated_cut)
    with_von = run_backtest(df, params=p, cost=c, overlay_exposure=mult)

    b, v = base.metrics, with_von.metrics
    return {
        "mode": mode,
        "min_confidence": min_confidence,
        "risk_elevated_cut": risk_elevated_cut,
        "base": b,
        "von": v,
        "delta_sharpe": v["sharpe"] - b["sharpe"],
        "delta_cagr": v["cagr"] - b["cagr"],
        "delta_maxdd": v["max_drawdown"] - b["max_drawdown"],
        "delta_calmar": v["calmar"] - b["calmar"],
        "von_veto_rate": float((mult < 0.999).mean()),
        "base_result": base,
        "von_result": with_von,
        "multiplier": mult,
    }


def overlay_out_of_sample(
    df: pd.DataFrame,
    answers: pd.DataFrame,
    params: StrategyParams | None = None,
    cost: CostModel | None = None,
    split_frac: float = 0.5,
    **kw,
) -> list[dict]:
    """Run the overlay comparison on the first and second half independently.

    The second half is the number that matters: the question texts and MODE were
    designed while looking at the first half, so the first half flatters the overlay
    by construction.
    """
    dates = df.index
    k = int(len(dates) * split_frac)
    out = []
    for label, sl in (("first_half", df.iloc[:k]), ("second_half", df.iloc[k:])):
        if len(sl) < 120:
            continue
        ans = answers.reindex(sl.index).ffill()
        try:
            r = evaluate_overlay(sl, ans, params, cost, **kw)
        except Exception as exc:
            out.append({"period": label, "error": str(exc)})
            continue
        out.append(
            {
                "period": label,
                "start": sl.index[0].date(),
                "end": sl.index[-1].date(),
                "base_sharpe": r["base"]["sharpe"],
                "von_sharpe": r["von"]["sharpe"],
                "delta_sharpe": r["delta_sharpe"],
                "base_cagr": r["base"]["cagr"],
                "von_cagr": r["von"]["cagr"],
                "base_maxdd": r["base"]["max_drawdown"],
                "von_maxdd": r["von"]["max_drawdown"],
                "veto_rate": r["von_veto_rate"],
            }
        )
    return out


def answer_agreement(answers: pd.DataFrame) -> pd.DataFrame:
    """How well do von's own heads agree with each other?

    Regime says up_trend, long_justified says no, risk says elevated: those cannot all
    be right. High mutual disagreement means the model is not seeing a clear situation,
    which is itself a tradable signal (stand aside).
    """
    a = answers.dropna(subset=["long_prob", "risk_prob"]).copy()
    if a.empty:
        return a
    a["regime_says_up"] = (a["regime"] == "up_trend").astype(float)
    a["long_says_yes"] = (a["long_prob"] >= 0.5).astype(float)
    a["risk_says_calm"] = (a["risk_prob"] < 0.5).astype(float)
    a["n_agree"] = a[["regime_says_up", "long_says_yes", "risk_says_calm"]].sum(axis=1)
    return a
