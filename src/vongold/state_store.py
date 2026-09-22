"""Persistent state for the dry-run runner: the on/off switch, the ledger, status.

Everything lives in a single directory (`runtime/`) so the whole system's state can be
inspected, backed up, or deleted by a human without touching code.

Control channel: `runtime/control.json` holds `enabled`. The local runner polls it and
the deployed monitor writes it (via the GitHub API). That means the web button and the
local process agree on the same source of truth, and a human can always flip the switch
with a text editor if the network is down.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parents[2] / "runtime"


def _atomic_write(path: Path, text: str) -> None:
    """Write then rename, so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Control:
    """The on/off switch. Default OFF: a money-moving process must be started explicitly."""

    enabled: bool = False
    reason: str = "initial state"
    changed_at: str = field(default_factory=utcnow)
    changed_by: str = "system"
    # Guard rails a human can tighten from the monitor without editing code.
    max_exposure: float = 1.0
    kill_if_drawdown_exceeds: float = 0.25
    symbol: str = "GLD"

    @classmethod
    def load(cls, path: Path | None = None) -> "Control":
        p = (path or RUNTIME_DIR / "control.json")
        if not p.exists():
            c = cls()
            c.save(p)
            return c
        try:
            raw = json.loads(p.read_text())
            known = {f for f in cls.__dataclass_fields__}
            return cls(**{k: v for k, v in raw.items() if k in known})
        except Exception:
            # A corrupt control file must fail CLOSED.
            return cls(enabled=False, reason="control file unreadable; failing closed",
                       changed_by="system")

    def save(self, path: Path | None = None) -> Path:
        p = (path or RUNTIME_DIR / "control.json")
        _atomic_write(p, json.dumps(asdict(self), indent=2))
        return p


@dataclass
class PaperPosition:
    shares: float = 0.0
    avg_cost: float = 0.0
    cash: float = 0.0
    last_price: float = 0.0
    equity: float = 0.0
    opened_at: str | None = None

    @property
    def market_value(self) -> float:
        return self.shares * self.last_price


class Ledger:
    """Append-only JSONL of every decision, order and fill. The audit trail."""

    def __init__(self, path: Path | None = None):
        self.path = path or (RUNTIME_DIR / "ledger.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, **payload) -> dict:
        rec = {"ts": utcnow(), "kind": kind, **payload}
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        return rec

    def tail(self, n: int = 50) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text().splitlines()
        out = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    def all(self) -> list[dict]:
        return self.tail(n=10**9)


class PositionStore:
    """Current paper position + equity curve, persisted as JSON."""

    def __init__(self, path: Path | None = None, initial_capital: float = 10_000.0):
        self.path = path or (RUNTIME_DIR / "position.json")
        self.initial_capital = initial_capital
        self.state = self._load()

    def _load(self) -> PaperPosition:
        if not self.path.exists():
            # Seed the paper account with starting capital. Without this the account
            # begins at $0 and can never buy anything -- the first live tick exposed
            # exactly that (equity 0.0, no trade possible).
            return PaperPosition(cash=self.initial_capital)
        try:
            raw = json.loads(self.path.read_text())
            known = {f for f in PaperPosition.__dataclass_fields__}
            return PaperPosition(**{k: v for k, v in raw.items() if k in known})
        except Exception:
            return PaperPosition(cash=self.initial_capital)

    def save(self) -> None:
        _atomic_write(self.path, json.dumps(asdict(self.state), indent=2))

    def snapshot(self, price: float) -> PaperPosition:
        self.state.last_price = price
        self.state.equity = self.state.cash + self.state.shares * price
        self.save()
        return self.state


def write_status(payload: dict, path: Path | None = None) -> Path:
    """The blob the deployed monitor reads. Keep it small and self-describing."""
    p = path or (RUNTIME_DIR / "status.json")
    payload = {"generated_at": utcnow(), **payload}
    _atomic_write(p, json.dumps(payload, indent=2, default=str))
    return p


def read_status(path: Path | None = None) -> dict:
    p = path or (RUNTIME_DIR / "status.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}
