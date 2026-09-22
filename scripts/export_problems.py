"""Export one von decision problem per trading day to JSONL."""
import json, sys
from pathlib import Path
from vongold.data import build_dataset
from vongold.von_state import states_for_backtest

df = build_dataset("GLD", rng="10y")
probs = states_for_backtest(df)
out = Path("data/processed/von_problems.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as fh:
    for p in probs:
        fh.write(json.dumps(p) + "\n")
print(f"wrote {len(probs)} problems -> {out}")
print("first date:", probs[0]["_date"], " last date:", probs[-1]["_date"])
print("\n--- sample state text ---")
print(probs[-1]["state"])
