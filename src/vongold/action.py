"""The five-way action output: long / short / open / close / hold.

What this is
------------
The live bot asks von DESCRIPTIVE questions today (what regime, is a long justified, is risk
elevated, conviction) and turns the answers into an exposure with fixed rules. This module
asks von for an ACTION instead.

Design constraints, each of which shaped the code:

* **von is not an autoregressive reasoner.** It matches a state description against
  criteria. It cannot chain reasoning, so the criteria must state the observable condition
  for each label, and the state must contain the facts those conditions refer to. If the
  state does not mention whether a position is held, "close" and "hold" are unanswerable.

* **Five labels collapse.** A model that cannot separate the labels answers the most generic
  one every time -- which is exactly what happened with the previous question set ("range"
  on 2,252 of 2,252 days). So the answer is never trusted on its own: `probe_five_way.py`
  measures the distribution across real history, and `ACTION_PRIOR_WEIGHT` below gates how
  much the action may move exposure. A collapsed model produces a constant, and a constant
  overlay cannot move exposure in a way that matters.

* **Determinism is not a bug.** Measured: the same state asked three times returns byte-
  identical answers (von is not sampling). This is why a 10-second re-decide loop cannot
  produce new information from an unchanged state -- it is only meaningful when the *state*
  changes, i.e. when the price moves. See `decide_action`.

* **Fail closed.** Any error returns None, and None means "no view", never a trade.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[2]  # src/vongold/action.py -> repo root
VON_REPO = Path(os.environ.get("VON_MLX_DIR", Path.home() / "vendor-mlx" / "von-mlx"))
VON_PYTHON = VON_REPO / ".venv" / "bin" / "python"
VON_MODEL_DIR = VON_REPO / "out" / "von-1.0-mlx" / "8bit"
DECIDE_SCRIPT = REPO / "scripts" / "von_decide_one.py"

# The five actions, and the exposure delta each implies. Expressed as a *target adjustment*
# rather than an absolute, so the action composes with the mechanical strategy instead of
# replacing it: the action can scale or veto, never invent a position the strategy would
# not have taken.
ACTIONS = ("open_long", "add_long", "hold", "reduce", "close")

ACTION_TARGET = {
    "open_long": 1.0,    # take the mechanical target in full
    "add_long": 1.0,
    "hold": None,        # None == leave the mechanical target untouched
    "reduce": 0.5,       # halve it
    "close": 0.0,        # flatten
}

# How much the action is allowed to move exposure, 0..1. At 1.0 von fully overrides the
# mechanical target; at 0.0 it has no effect. Default is deliberately NOT 1.0: the action
# output is unvalidated until `scripts/eval_five_way.py` clears it, and a model that always
# answers "hold" or always answers "open_long" must not be able to steer the account.
ACTION_PRIOR_WEIGHT = float(os.environ.get("VONGOLD_ACTION_WEIGHT", "0.0"))

QUESTION = {
    "action": {
        "type": "choice",
        "instructions": (
            "You are deciding what to do with a gold position right now, based only on the "
            "evidence provided. Choose the single best action. Judge each option against its "
            "criterion and pick the one whose criterion the evidence matches most closely. "
            "Do not default to 'hold' unless the evidence is genuinely mixed."
        ),
        "criteria": {
            "open_long": "No position is held and the evidence points to rising prices: "
                         "momentum is positive over the longer horizons and price is above "
                         "its long-run average.",
            "add_long":  "A long position is already held and the evidence still points to "
                         "rising prices, so exposure should increase toward the target.",
            "hold":      "The evidence is mixed, conflicting, or weak, so the existing "
                         "position should be left unchanged.",
            "reduce":    "The evidence has weakened against the current position, so exposure "
                         "should be reduced but not eliminated.",
            "close":     "The evidence points clearly to falling prices, or risk is materially "
                         "elevated, so the position should be exited entirely.",
        },
    },
    "conviction": {
        "type": "score",
        "instructions": "How strong is the evidence for the action you chose?",
        # A `score` question takes a LIST of anchor descriptions (one per point on the
        # scale), not a dict. Sending a dict raises a pydantic ValidationError inside von and
        # the whole decision fails -- the `choice` question above does take a dict, so the two
        # types are not interchangeable. Learned by hitting it.
        "criteria": [
            "No conviction: the evidence is absent or contradictory",
            "Low conviction: a single weak indicator points one way",
            "Moderate conviction: several indicators align",
            "High conviction: most indicators align consistently",
            "Very high conviction: the evidence is unambiguous and fully supports the action",
        ],
    },
}


@dataclass
class Action:
    action: str | None = None
    conviction: float | None = None
    usage: dict | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.action in ACTIONS


def decide_action(state: str, timeout: float = 120.0) -> Action:
    """Ask von for one of the five actions. Never raises; returns an Action with .error."""
    if not VON_PYTHON.exists():
        return Action(error=f"von interpreter not found at {VON_PYTHON}")
    if not VON_MODEL_DIR.exists():
        return Action(error=f"von model not found at {VON_MODEL_DIR}")

    payload = json.dumps({"state": state, "questions": QUESTION})
    try:
        proc = subprocess.run(
            [str(VON_PYTHON), str(DECIDE_SCRIPT), str(VON_MODEL_DIR)],
            input=payload, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return Action(error=f"von timed out after {timeout}s")
    except Exception as exc:
        return Action(error=f"{type(exc).__name__}: {exc}")

    if proc.returncode != 0 and not proc.stdout.strip():
        return Action(error=f"von exited {proc.returncode}: {proc.stderr.strip()[:200]}")

    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return Action(error=f"unparseable von output: {exc}")

    if "error" in out:
        return Action(error=str(out["error"]))

    ans = out.get("answers", {})
    raw = ans.get("action", {}).get("choice")
    action = str(raw).strip().lower() if raw is not None else None
    # Normalize near-miss labels rather than discarding a usable answer.
    if action and action not in ACTIONS:
        for cand in ACTIONS:
            if cand in action:
                action = cand
                break
    conviction = ans.get("conviction", {}).get("score")
    return Action(action=action, conviction=conviction, usage=out.get("usage"))


def blend(mechanical_target: float, act: Action,
          weight: float | None = None) -> tuple[float, str]:
    """Combine the action with the mechanical target.

    Returns (target, why). The action can only move the target *toward* its own view by
    `weight`; at weight 0 the mechanical target passes through untouched, which is the
    default until the action output is validated.
    """
    w = ACTION_PRIOR_WEIGHT if weight is None else weight
    if not act.ok:
        return mechanical_target, f"mechanical (no action: {act.error})"
    implied = ACTION_TARGET.get(act.action) if act.action else None
    if implied is None:  # "hold"
        return mechanical_target, f"mechanical (von: hold, conviction {act.conviction})"
    blended = mechanical_target + w * (mechanical_target * implied - mechanical_target)
    return max(0.0, min(1.0, blended)), \
        f"von:{act.action} w={w} conviction={act.conviction}"
