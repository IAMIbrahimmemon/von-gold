#!/usr/bin/env python
"""Batch a whole backtest's worth of decision problems through local von.

Runs INSIDE the von-mlx venv, in-process, in the MAIN thread. Both details matter:

  * in-process (not HTTP) because loading the model once beats 2,500 subprocess
    launches, and because it keeps the run reproducible with no server to babysit.
  * main thread because MLX streams are thread-local. The von-mlx server works
    around this with a worker thread + job queue; a single loop over days avoids
    the problem entirely.

Usage (from the von-mlx venv):
    env -u PYTHONPATH .venv/bin/python <this script> --model-dir out/von-1.0-mlx/8bit \
        --problems problems.jsonl --out answers.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--problems", required=True, help="JSONL of {_date, state, questions}")
    ap.add_argument("--out", required=True, help="JSONL of {date, answers}")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from von_mlx import VonEngine  # imported late so --help works without MLX

    problems = [json.loads(line) for line in Path(args.problems).read_text().splitlines() if line.strip()]
    if args.limit:
        problems = problems[: args.limit]
    print(f"loaded {len(problems)} decision problems", flush=True)

    t0 = time.time()
    engine = VonEngine(args.model_dir)
    print(f"engine loaded in {time.time() - t0:.2f}s", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_err = 0
    t_start = time.time()
    with out_path.open("w") as fh:
        for k, prob in enumerate(problems):
            try:
                resp = engine.evaluate(state=prob["state"], questions=prob["questions"])
                payload = json.loads(resp.model_dump_json())
                rec = {
                    "date": prob["_date"],
                    "i": prob.get("_i"),
                    "answers": payload.get("answers", {}),
                    "usage": payload.get("usage", {}),
                }
            except Exception as exc:
                n_err += 1
                rec = {"date": prob.get("_date"), "i": prob.get("_i"), "error": str(exc)}
            fh.write(json.dumps(rec) + "\n")
            if (k + 1) % 250 == 0:
                el = time.time() - t_start
                rate = (k + 1) / el
                print(f"  {k + 1}/{len(problems)}  {rate:.1f}/s  elapsed {el:.0f}s", flush=True)

    el = time.time() - t_start
    print(f"done: {len(problems) - n_err} ok, {n_err} errors, {el:.0f}s "
          f"({len(problems) / el if el else 0:.1f} decisions/s)", flush=True)
    return 0 if n_err == 0 else 0  # errors are recorded per-row, not fatal


if __name__ == "__main__":
    sys.exit(main())
