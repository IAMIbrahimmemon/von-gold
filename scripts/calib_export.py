#!/usr/bin/env python
"""Emit DESCRIPTIVE calibration probes for von, with ground truth attached.

The question this answers: can von even READ the market state?

Not "does it predict the future" -- that is the overlay experiment. Here we ask von
purely descriptive questions ("is price above its 200-session average?", "is the series
rising, falling or sideways?") and compare its answer to the mechanically-computed
truth from the same numbers. This isolates reading ability from forecasting ability.

If von cannot reproduce a moving-average comparison from explicitly supplied returns,
then a von overlay is not diagnosing anything -- it is a random number generator wearing
a lab coat, and the overlay should be disabled regardless of what its backtest says.

Writes JSONL consumable by scripts/von_batch.py. Truth labels are sidecar'd to
data/processed/calib_truth.json so the batch runner stays generic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from vongold.data import build_dataset

QUESTIONS = {
    "trend_direction": {
        "type": "choice",
        "instructions": "Look at the price returns provided below. Which best describes "
                        "the direction of this price series right now?",
        "criteria": {
            "rising": "Most recent prices are higher than earlier ones across the longer horizons.",
            "falling": "Most recent prices are lower than earlier ones across the longer horizons.",
            "sideways": "Prices are broadly unchanged across the longer horizons.",
        },
    },
    "above_average": {
        "type": "noul",
        "instructions": "Is the current price above its average over the last 200 sessions?",
        "criteria": {
            "true": "The current price is higher than the 200-session average.",
            "false": "The current price is lower than the 200-session average.",
        },
    },
}


def state_for(df, i: int) -> str:
    close = float(df["close"].iloc[i])
    hist = df["close"].iloc[: i + 1].astype(float)
    lines = [f"last close: {close:.2f}"]
    for lbl, d in (("5 sessions", 5), ("20 sessions", 20), ("60 sessions", 60),
                   ("120 sessions", 120), ("250 sessions", 250)):
        if len(hist) > d:
            lines.append(f"return over {lbl}: {(close / float(hist.iloc[-d - 1]) - 1) * 100:+.2f}%")
    return "\n".join(lines)


def main() -> int:
    n_probes = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    df = build_dataset("GLD", rng="10y")

    start = 300
    step = max(1, (len(df) - start) // n_probes)
    idx = list(range(start, len(df), step))[:n_probes]

    problems, truth = [], []
    for i in idx:
        hist = df["close"].iloc[: i + 1].astype(float)
        close = float(df["close"].iloc[i])
        ma200 = float(hist.tail(200).mean()) if len(hist) >= 200 else float(hist.mean())
        r20 = float(close / df["close"].iloc[i - 20] - 1) if i >= 20 else 0.0
        r250 = float(close / df["close"].iloc[i - 250] - 1) if i >= 250 else 0.0

        # Ground truth computed mechanically, from the same numbers von sees.
        if r20 > 0 and r250 > 0:
            truth_dir = "rising"
        elif r20 < 0 and r250 < 0:
            truth_dir = "falling"
        else:
            truth_dir = "sideways"

        date = df.index[i].date().isoformat()
        problems.append({"_date": date, "_i": i, "state": state_for(df, i),
                         "questions": QUESTIONS})
        truth.append({
            "date": date,
            "i": i,
            "truth_dir": truth_dir,
            "truth_above": close > ma200,
            "r20": r20,
            "r250": r250,
            "dist_ma200": close / ma200 - 1.0,
        })

    out = Path("data/processed/calib_problems.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for p in problems:
            fh.write(json.dumps(p) + "\n")
    Path("data/processed/calib_truth.json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {len(problems)} calibration probes -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
