"""Client for the local von decision model.

Two transports, chosen automatically:

  * **in-process** -- if `von_mlx` is importable in the current interpreter (i.e. we are
    already running inside the von-mlx venv, which is what the backtest batch does).
  * **subprocess** -- otherwise shell out to the von-mlx venv's python once per call.
    This is the path the daily dry-run runner uses, and it is deliberately the default
    for live use: the two environments stay separate (pandas in one, MLX in the other),
    nothing has to be installed twice, and there is no server to supervise.

Hard rule this client enforces: **it never raises into the trading path.** Any failure
returns `None`, and the caller treats a missing answer as "no view" rather than as a
trade signal. A model outage must not be able to move money.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VON_REPO = Path.home() / "vendor-mlx" / "von-mlx"
VON_PYTHON = VON_REPO / ".venv" / "bin" / "python"
VON_MODEL_DIR = VON_REPO / "out" / "von-1.0-mlx" / "8bit"
DECIDE_SCRIPT = REPO / "scripts" / "von_decide_one.py"


@dataclass
class VonResult:
    regime: str | None = None
    regime_confidence: float | None = None
    long_prob: float | None = None
    risk_prob: float | None = None
    conviction: float | None = None
    usage: dict | None = None
    error: str | None = None
    transport: str = "none"

    @property
    def ok(self) -> bool:
        return self.error is None and self.long_prob is not None


def von_importable_in_process() -> bool:
    try:
        import von_mlx  # noqa: F401
    except Exception:
        return False
    return True


def von_environment_status() -> tuple[bool, str]:
    """Report whether the local model is usable, with a specific reason if not."""
    if not VON_PYTHON.exists():
        return False, f"von venv python not found at {VON_PYTHON}"
    if not VON_MODEL_DIR.exists():
        return False, f"model weights not found at {VON_MODEL_DIR}"
    if not DECIDE_SCRIPT.exists():
        return False, f"decision script not found at {DECIDE_SCRIPT}"
    return True, (f"in-process" if von_importable_in_process() else f"subprocess via {VON_PYTHON}")


def _parse(payload: dict, transport: str) -> VonResult:
    if "error" in payload:
        return VonResult(error=payload["error"], transport=transport)
    a = payload.get("answers", {}) or {}
    reg = a.get("regime") or {}
    return VonResult(
        regime=reg.get("choice"),
        regime_confidence=reg.get("confidence"),
        long_prob=(a.get("long_justified") or {}).get("noul"),
        risk_prob=(a.get("risk_elevated") or {}).get("noul"),
        conviction=(a.get("conviction") or {}).get("score"),
        usage=payload.get("usage"),
        transport=transport,
    )


def decide(state: str, questions: dict, timeout: int = 180,
           model_dir: str | Path | None = None) -> VonResult:
    """Ask the local model one decision problem. Never raises."""
    problem = {"state": state, "questions": questions}

    # --- in-process ---
    if von_importable_in_process():
        try:
            from von_mlx import VonEngine

            engine = VonEngine(str(model_dir or VON_MODEL_DIR))
            resp = engine.evaluate(state=state, questions=questions)
            return _parse(json.loads(resp.model_dump_json()), "in-process")
        except Exception as exc:
            # Fall through to subprocess rather than giving up.
            inproc_err = f"{type(exc).__name__}: {exc}"
    else:
        inproc_err = None

    # --- subprocess ---
    if not VON_PYTHON.exists():
        return VonResult(error=inproc_err or f"von python missing at {VON_PYTHON}",
                         transport="unavailable")
    try:
        payload = dict(problem)
        payload["_model_dir"] = str(model_dir or VON_MODEL_DIR)
        proc = subprocess.run(
            [str(VON_PYTHON), str(DECIDE_SCRIPT)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(Path.home())},
        )
        if proc.returncode != 0 and not proc.stdout.strip():
            return VonResult(
                error=f"exit {proc.returncode}: {(proc.stderr or '').strip()[:300]}",
                transport="subprocess",
            )
        return _parse(json.loads(proc.stdout), "subprocess")
    except subprocess.TimeoutExpired:
        return VonResult(error=f"model call timed out after {timeout}s", transport="subprocess")
    except Exception as exc:
        return VonResult(error=f"{type(exc).__name__}: {exc}", transport="subprocess")


def main() -> int:
    """Diagnostic entry point: `python -m vongold.von_client`."""
    ok, detail = von_environment_status()
    print(f"von environment: {'OK' if ok else 'UNAVAILABLE'} -- {detail}")
    if not ok:
        return 1
    res = decide(
        state="GOLD PRICE EVIDENCE\nlast close: 398.38\nreturn over 20 trading days: -5.90%\n"
              "return over 250 trading days: +15.46%\n",
        questions={"long_justified": {
            "type": "noul",
            "instructions": "Based only on the evidence provided, is holding a long "
                            "position in gold justified today?",
            "criteria": {"true": "Evidence supports holding a long position.",
                         "false": "Evidence does not support holding a long position."}}},
    )
    print(f"transport: {res.transport}")
    print(f"answer: {res}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
