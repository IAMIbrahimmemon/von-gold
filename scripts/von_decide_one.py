#!/usr/bin/env python
"""Answer ONE von decision problem, reading JSON from stdin and writing JSON to stdout.

Runs inside the von-mlx venv. This is the process boundary the dry-run runner uses:
the trading logic lives in the von-gold venv (pandas, backtester), while the model
lives in its own MLX venv. Keeping them separate avoids installing either environment's
dependencies into the other, and costs one process spawn per trading session -- which
for a daily job is irrelevant.

Why not HTTP: it would mean a long-running server to supervise, and a port to keep alive
for a job that runs once a day. A subprocess has no lifecycle and cannot be left down.

Stdin:  {"state": "...", "questions": {...}}
Stdout: {"answers": {...}, "usage": {...}}   (or {"error": "..."} with exit code 1)
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    raw = sys.stdin.read()
    try:
        problem = json.loads(raw)
    except json.JSONDecodeError as exc:
        json.dump({"error": f"invalid JSON on stdin: {exc}"}, sys.stdout)
        return 1

    model_dir = problem.pop("_model_dir", None) or sys.argv[1]
    try:
        from von_mlx import VonEngine
    except Exception as exc:
        json.dump({"error": f"von_mlx not importable: {exc}"}, sys.stdout)
        return 1

    try:
        engine = VonEngine(model_dir)
        resp = engine.evaluate(state=problem["state"], questions=problem["questions"])
        payload = json.loads(resp.model_dump_json())
        json.dump({"answers": payload.get("answers", {}),
                   "usage": payload.get("usage", {})}, sys.stdout)
        return 0
    except Exception as exc:
        json.dump({"error": f"{type(exc).__name__}: {exc}"}, sys.stdout)
        return 1


if __name__ == "__main__":
    sys.exit(main())
