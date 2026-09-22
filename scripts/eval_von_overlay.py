#!/usr/bin/env python
"""Does the local von decision model actually improve the gold strategy?

This is the experiment the whole design exists to answer honestly. It reports:
  1. what von actually said, in aggregate
  2. how often its own heads disagree with each other
  3. an A/B of the mechanical strategy with and without the overlay, per mode
  4. the same split into halves, so the second half reads out-of-sample
  5. whether the news features predict returns at all
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from vongold.backtest import buy_and_hold, run_backtest, summarize
from vongold.config import StrategyParams
from vongold.data import build_dataset
from vongold.news import news_daily_features, news_impact_report
from vongold.von_overlay import (
    answer_agreement,
    evaluate_overlay,
    load_answers,
    overlay_out_of_sample,
)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:9.4f}")


def main() -> int:
    df = build_dataset("GLD", rng="10y")
    answers = load_answers("data/processed/von_answers.jsonl")
    print(f"von answers: {len(answers)} rows {answers.index.min().date()} -> {answers.index.max().date()}")

    print("\n" + "=" * 100)
    print("1. WHAT DID VON ACTUALLY SAY?")
    print("=" * 100)
    print(f"  regime distribution:\n{answers['regime'].value_counts(dropna=False).to_string()}")
    print(f"\n  long_justified  : mean {answers['long_prob'].mean():.3f}  "
          f"min {answers['long_prob'].min():.3f}  max {answers['long_prob'].max():.3f}")
    print(f"  risk_elevated   : mean {answers['risk_prob'].mean():.3f}  "
          f"P(>=0.5) {(answers['risk_prob'] >= 0.5).mean():.3f}")
    print(f"  conviction 0-4  : mean {answers['conviction'].mean():.3f}")
    print(f"  regime confidence: mean {answers['regime_confidence'].mean():.3f}  "
          f"P(<0.60) {(answers['regime_confidence'] < 0.60).mean():.3f}")
    # von's answers must VARY with the evidence, otherwise the state text conveys nothing.
    print(f"\n  distinct regime values issued : {answers['regime'].nunique()} of 3")
    print(f"  long_prob distinct values     : {answers['long_prob'].nunique()} of {len(answers)}")
    print(f"  long_prob sd                  : {answers['long_prob'].std():.4f}")

    print("\n" + "=" * 100)
    print("2. INTERNAL AGREEMENT -- do von's own heads tell one story?")
    print("=" * 100)
    agr = answer_agreement(answers)
    print(agr["n_agree"].value_counts().sort_index().to_string())
    print(f"\n  all three heads agree       : {(agr['n_agree'] == 3).mean():.3f}")
    print(f"  all three heads disagree    : {(agr['n_agree'] <= 1).mean():.3f}")

    # --- The critical validation: does von's answer mean anything? ---
    print("\n" + "=" * 100)
    print("3. DOES VON'S ANSWER PREDICT ANYTHING?  (falsification test)")
    print("=" * 100)
    px = df["close"]
    for h in (1, 5, 21):
        fwd = (px.shift(-h) / px - 1.0).reindex(answers.index)
        j = pd.DataFrame({"long_prob": answers["long_prob"], "fwd": fwd}).dropna()
        if len(j) < 30:
            continue
        corr = j["long_prob"].corr(j["fwd"])
        # Split at von's own 0.5 threshold: does "justified" beat "not justified"?
        hi = j.loc[j["long_prob"] >= 0.5, "fwd"]
        lo = j.loc[j["long_prob"] < 0.5, "fwd"]
        print(f"  horizon {h:2d}d: corr(long_prob, fwd ret) = {corr:+.4f}   "
              f"mean fwd when P>=0.5: {hi.mean():+.4%} (n={len(hi)})   "
              f"when P<0.5: {lo.mean():+.4%} (n={len(lo)})"
              + ("   <-- INVERTED" if hi.mean() < lo.mean() else ""))
    for col, label in (("risk_prob", "risk_elevated"), ("conviction", "conviction")):
        fwd = (px.shift(-5) / px - 1.0).reindex(answers.index)
        j = pd.DataFrame({col: answers[col], "fwd": fwd}).dropna()
        if len(j) > 30:
            print(f"  horizon  5d: corr({label}, fwd ret) = {j[col].corr(j['fwd']):+.4f}")

    print("\n" + "=" * 100)
    print("4. A/B: MECHANICAL vs MECHANICAL+VON  (base = trend MA only, 10% vol target)")
    print("=" * 100)
    base_p = StrategyParams(use_real_yield_filter=False, use_dollar_filter=False,
                            target_vol_annual=0.10, rebalance_band=0.15)
    bh = buy_and_hold(df)
    mech = run_backtest(df, params=base_p)
    print(f"  buy & hold        : {summarize(bh)}")
    print(f"  mechanical        : {summarize(mech)}")

    rows = []
    for mode in ("veto", "halve", "prob", "rank"):
        for conf in (0.0, 0.60, 0.80):
            r = evaluate_overlay(df, answers, base_params=base_p, mode=mode, min_confidence=conf)
            v, b = r["von"], r["base"]
            rows.append({
                "mode": mode,
                "min_conf": conf,
                "cagr": v["cagr"], "sharpe": v["sharpe"], "max_dd": v["max_drawdown"],
                "calmar": v["calmar"], "in_mkt": v["pct_time_in_market"],
                "d_sharpe": r["delta_sharpe"], "d_cagr": r["delta_cagr"],
                "d_maxdd": r["delta_maxdd"],
                "veto_rate": r["von_veto_rate"],
            })
    out = pd.DataFrame(rows)
    print("\n  overlay variants (delta columns are vs the mechanical row above):")
    print(out.to_string(index=False))
    best = out.loc[out["sharpe"].idxmax()]
    print(f"\n  best overlay: mode={best['mode']} conf={best['min_conf']} "
          f"sharpe {best['sharpe']:.3f} vs mechanical {mech.metrics['sharpe']:.3f} "
          f"({best['d_sharpe']:+.3f})")

    print("\n" + "=" * 100)
    print("5. OUT OF SAMPLE: first half vs second half (second half is the real test)")
    print("=" * 100)
    for mode in ("veto", "prob", "rank"):
        print(f"\n  mode = {mode}")
        oos = overlay_out_of_sample(df, answers, base_p, mode=mode, min_confidence=0.60)
        print(pd.DataFrame(oos).to_string(index=False))

    print("\n" + "=" * 100)
    print("6. NEWS FEATURES -- do they predict gold returns?")
    print("=" * 100)
    try:
        news = news_daily_features()
        rep = news_impact_report(news, df)
        if rep.empty:
            print("  no measurable overlap (news feeds only carry recent headlines; see docs/DATA.md)")
        else:
            rep = rep.assign(abs_corr=lambda d: d["corr"].abs()).sort_values("abs_corr", ascending=False)
            print(rep.head(15).to_string(index=False))
    except Exception as exc:
        print(f"  news layer unavailable: {type(exc).__name__}: {exc}")

    out.to_csv("data/processed/overlay_results.csv", index=False)
    print("\nwritten: data/processed/overlay_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
