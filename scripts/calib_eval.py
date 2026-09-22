#!/usr/bin/env python
"""Score von's descriptive calibration against mechanical ground truth."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def main() -> int:
    truth = {r["date"]: r for r in json.loads(Path("data/processed/calib_truth.json").read_text())}
    rows = []
    for line in Path("data/processed/calib_answers.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        t = truth.get(rec["date"])
        if not t or rec.get("error"):
            continue
        a = rec.get("answers", {})
        got_dir = (a.get("trend_direction") or {}).get("choice")
        p_above = (a.get("above_average") or {}).get("noul")
        rows.append({
            "date": rec["date"],
            "von_dir": got_dir,
            "truth_dir": t["truth_dir"],
            "dir_ok": got_dir == t["truth_dir"],
            "von_above": (p_above >= 0.5) if p_above is not None else None,
            "truth_above": t["truth_above"],
            "above_ok": ((p_above >= 0.5) == t["truth_above"]) if p_above is not None else None,
            "von_above_p": p_above,
            "dir_conf": (a.get("trend_direction") or {}).get("confidence"),
            "dist_ma200": t["dist_ma200"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("no scored probes -- did the batch run?")
        return 1

    n = len(df)
    print(f"scored probes: {n}")
    print(f"\n  trend direction exact match : {df['dir_ok'].sum()}/{n} = {df['dir_ok'].mean():.3f}")
    print(f"  above-200MA match           : {df['above_ok'].sum()}/{n} = {df['above_ok'].mean():.3f}")

    print("\n  von's direction answer distribution:")
    print("   ", dict(Counter(df["von_dir"])))
    print("  ground truth distribution:")
    print("   ", dict(Counter(df["truth_dir"])))

    print("\n  von's above-average answers:")
    print("   ", dict(Counter(df["von_above"])))
    print("  ground truth:")
    print("   ", dict(Counter(df["truth_above"])))

    # A constant answer can still score well if the truth is imbalanced. Compare against
    # the majority-class baseline, which is the number that matters.
    maj_dir = Counter(df["truth_dir"]).most_common(1)[0]
    maj_above = Counter(df["truth_above"]).most_common(1)[0]
    print(f"\n  BASELINE: always answering '{maj_dir[0]}' scores {maj_dir[1]/n:.3f} on direction")
    print(f"  BASELINE: always answering {maj_above[0]} scores {maj_above[1]/n:.3f} on above-200MA")
    print(f"  von direction {'BEATS' if df['dir_ok'].mean() > maj_dir[1]/n else 'DOES NOT BEAT'} that baseline")
    print(f"  von above-200MA {'BEATS' if df['above_ok'].mean() > maj_above[1]/n else 'DOES NOT BEAT'} that baseline")

    # Where von is wrong on above-average, is the price near the average (a hard call)
    # or far from it (a clear miss)? That distinguishes calibration failure from noise.
    wrong = df[~df["above_ok"].astype(bool)]
    right = df[df["above_ok"].astype(bool)]
    if len(wrong) and len(right):
        print(f"\n  |distance from MA200| when WRONG: median {wrong['dist_ma200'].abs().median():.4f}")
        print(f"  |distance from MA200| when RIGHT: median {right['dist_ma200'].abs().median():.4f}")

    # Correlation of von's probability with the actual signed distance from the MA.
    j = df.dropna(subset=["von_above_p"])
    if len(j) > 10:
        print(f"\n  corr(von P(above), actual signed distance from MA200) = "
              f"{j['von_above_p'].corr(j['dist_ma200']):+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
