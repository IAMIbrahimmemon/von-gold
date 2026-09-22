"""Probe: can von produce a 5-way {long, short, open, close, hold} action, and does it
discriminate between days?

Why this exists
---------------
The live bot currently asks von DESCRIPTIVE questions (regime, is a long justified, is risk
elevated, conviction). The proposal is to ask it for an explicit ACTION every 10 seconds:
long / short / open / close / hold.

Two things must be true for that to be worth building, and neither is obvious:

  1. **It must discriminate.** A model that answers "hold" on 2,252 of 2,252 days (which is
     what the current question set does -- it said "range" every single time) carries no
     information, and a 10-second loop over it just burns CPU.
  2. **The labels must be separable.** von is not an autoregressive LLM and does not reason
     step by step; it matches a state against criteria. Five mutually exclusive action
     labels are a harder ask than one yes/no, and the failure mode is collapsing onto the
     most generic label ("hold"), which looks plausible but decides nothing.

So: build the same state the live path builds, ask the 5-way question, and MEASURE the
distribution and agreement with what the mechanical strategy actually did on that day.

Writes a problems JSONL for `scripts/von_batch.py` (which loads the model once, in-process).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from vongold.data import build_dataset  # noqa: E402
from vongold.strategy import build_features, mechanical_exposure  # noqa: E402
from vongold.config import StrategyParams  # noqa: E402
from vongold.von_state import build_von_problem  # noqa: E402

# The question is imported from the package, NOT redefined here. A duplicated wording would
# mean the 270-day measurement describes a question the live bot never asks -- which is
# exactly the bug this replaced: the probe and the live loop had different criteria for the
# same five labels and produced different argmax answers for the same state.
from vongold.action import QUESTION  # noqa: E402

# Kept as an alias so older callers of this script keep working.
ACTION_CRITERIA = QUESTION["action"]["criteria"]


def build_questions(has_position: bool) -> dict:
    """The five-way action question, identical to what the live loop asks.

    `has_position` no longer changes the wording: the state text already says whether a
    position is held, so varying the criteria by position would mean two different questions
    and two different measurements.
    """
    return QUESTION


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/five_way_problems.jsonl")
    ap.add_argument("--n", type=int, default=240, help="number of historical days to sample")
    ap.add_argument("--lookback", type=int, default=800)
    args = ap.parse_args()

    df = build_dataset("GLD", rng="10y").tail(args.lookback)
    params = StrategyParams()
    f = build_features(df, params)
    # What the mechanical strategy ACTUALLY did, for agreement measurement.
    mech = mechanical_exposure(f, params)

    last = len(df) - 1
    start = 260
    if last <= start:
        print("not enough history", file=sys.stderr)
        return 1
    step = max(1, (last - start) // args.n)
    idx = list(range(start, last + 1, step))
    print(f"sampling {len(idx)} days from {df.index[start].date()} to {df.index[last].date()}")

    rows = []
    for i in idx:
        prob = build_von_problem(df, i)
        if prob is None:
            continue
        # Position state as the mechanical strategy saw it going into that day.
        prev = float(mech.iloc[i - 1]) if i > 0 else 0.0
        has_pos = prev > 1e-9
        q = build_questions(has_pos)
        rows.append({
            "_date": str(df.index[i].date()),
            "_mech_target": float(mech.iloc[i]),
            "_prev_exposure": prev,
            "_has_position": bool(has_pos),
            "_above_ma": bool(f["above_ma"].iloc[i] > 0),
            "_tsmom": float(f["tsmom"].iloc[i]),
            "state": prob["state"],
            "questions": q,
        })

    out = Path(args.out)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} problems -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
