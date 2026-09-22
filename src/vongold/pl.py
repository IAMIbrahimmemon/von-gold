"""Reconstruct the paper account's P/L over arbitrary time windows.

Why this recomputes from the append-only ledger instead of sampling status.json:

The equity curve in status.json is only recorded when the bot ticks, so it is a *sample*
of history at whatever irregular moments the scheduler happened to run. Reconstructing
from the ledger -- which records every fill and every decision as an immutable event --
gives a curve that is correct at any granularity, can be re-derived after a schema change,
and cannot be silently corrupted by a missed tick. A sampled curve would report "past
hour" as a flat zero simply because no tick landed in that hour.

The one thing the ledger cannot invent is intraday prices. GLD only trades 09:30-16:00 ET
and the bot evaluates once per day after the close, so a "past hour" P/L is only
meaningful when the account is actually holding gold and the market is open. Rather than
fake interpolation, this module distinguishes:

  - a **held** position, where mark-to-market is real and the window P/L is genuine
    (using an intraday quote when one is available and fresh), from
  - a **flat** position, where the account is definitionally at $0 P/L for the window
    (no exposure -> no gains or losses), which is marked `flat_no_exposure` so the
    dashboard can say so plainly instead of implying the bot is inactive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

RUNTIME_DIR = Path(__file__).resolve().parents[2] / "runtime"
LEDGER_PATH = RUNTIME_DIR / "ledger.jsonl"
INITIAL_CAPITAL = 10_000.0

# Windows the dashboard reports. (label, timedelta or None for all-time)
WINDOWS: tuple[tuple[str, timedelta | None], ...] = (
    ("all_time", None),
    ("past_24h", timedelta(hours=24)),
    ("past_6h", timedelta(hours=6)),
    ("past_hour", timedelta(hours=1)),
)


@dataclass
class PlWindow:
    """P/L for one lookback window, in dollars and percent of capital."""

    label: str
    pl_abs: float
    pl_pct: float
    start_equity: float
    end_equity: float
    basis: str  # "reconstructed" | "flat_no_exposure"
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "pl_abs": round(self.pl_abs, 2),
            "pl_pct": round(self.pl_pct, 6),
            "start_equity": round(self.start_equity, 2),
            "end_equity": round(self.end_equity, 2),
            "basis": self.basis,
            "note": self.note,
        }


def _parse_ts(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def load_events(ledger_path: Path | None = None) -> list[dict]:
    """Read the append-only ledger, skipping any corrupt line rather than failing.

    A partially-written final line is expected when a tick is interrupted; losing that
    one event is correct, refusing to report at all is not.
    """
    p = ledger_path or LEDGER_PATH
    if not p.exists():
        return []
    events: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and "ts" in rec:
            events.append(rec)
    return events


def reconstruct_equity_curve(events: list[dict],
                             initial_capital: float = INITIAL_CAPITAL) -> list[tuple[datetime, float]]:
    """Build (timestamp, equity) pairs from ledger fills.

    The ledger's `cash` and `shares` fields are already post-trade snapshots, and the
    fill carries the price it executed at, so each fill yields an exact mark at that
    instant with no assumptions about what happened in between.
    """
    curve: list[tuple[datetime, float]] = []
    for ev in events:
        if ev.get("kind") != "fill":
            continue
        ts = _parse_ts(ev.get("ts", ""))
        if ts is None:
            continue
        try:
            cash = float(ev.get("cash", initial_capital))
            shares = float(ev.get("shares", 0.0))
            price = float(ev.get("price", 0.0))
        except (TypeError, ValueError):
            continue
        curve.append((ts, cash + shares * price))
    curve.sort(key=lambda x: x[0])
    return curve


def equity_at(curve: list[tuple[datetime, float]], when: datetime,
              fallback: float) -> float:
    """Equity as of `when`: the last known mark at or before it, else `fallback`.

    No interpolation -- inventing a value between two marks would be fabricating data.
    """
    value = fallback
    for ts, eq in curve:
        if ts <= when:
            value = eq
        else:
            break
    return value


def _fresh_intraday_mark(now: datetime) -> tuple[float | None, str]:
    """Try to get a live spot mark so a sub-day window on a held position is real.

    GLD's own last close is up to a day old, which makes an hourly P/L meaningless. These
    free no-key endpoints carry intraday freshness. Returned as (price, source) or
    (None, reason).
    """
    import requests

    try:
        r = requests.get("https://api.gold-api.com/price/XAU", timeout=8)
        if r.status_code == 200:
            payload = r.json()
            updated = _parse_ts(payload.get("updatedAt", ""))
            age_min = (now - updated).total_seconds() / 60.0 if updated else 10 ** 6
            if age_min <= 30 and payload.get("price"):
                return float(payload["price"]), "gold-api.com spot XAU/USD"
            return None, f"spot quote {age_min:.0f}min old"
    except Exception as exc:
        return None, f"spot quote unavailable ({type(exc).__name__})"
    return None, "spot quote unavailable"


def compute_pl(ledger_path: Path | None = None,
               initial_capital: float = INITIAL_CAPITAL,
               current_equity: float | None = None,
               shares: float | None = None,
               now: datetime | None = None,
               live_mark: tuple[float | None, str] | None = None,
               intraday_bars=None) -> dict[str, Any]:
    """Compute P/L for every lookback window, with the context needed to read it honestly.

    Per window the rule is:

      1. **Real** when the window is covered by actual price data -- for a held position
         that means marking the shares at the window's open and close.
      2. **"includes a trade"** when a fill landed inside the window but there is no
         intraday mark to price it at. Reported as reconstructed P/L with a flag, because
         the trade genuinely happened and the account genuinely moved.
      3. **"flat_no_exposure"** only when nothing was held at any point in the window --
         i.e. no fills occurred and no position was open. Then $0 is the exact answer.

    Note the deliberately avoided trap: "flat at both ends" does NOT imply "no P/L
    possible in between". A position opened and closed inside a 6-hour window leaves the
    account flat at both ends while realising a gain or loss in the middle. An earlier
    version of this function made exactly that error and reported -$64.50 alongside a
    note claiming no loss was possible.
    """
    now = now or datetime.now(timezone.utc)
    events = load_events(ledger_path)
    curve = reconstruct_equity_curve(events, initial_capital)

    if live_mark is None:
        live_mark = _fresh_intraday_mark(now) if shares else (None, "no position")
    _, mark_source = live_mark

    holdings = float(shares or 0.0)
    end_value = float(current_equity) if current_equity is not None else (
        curve[-1][1] if curve else initial_capital
    )

    if intraday_bars is None and holdings > 1e-9:
        intraday_bars = fetch_intraday_bars()

    windows: list[dict] = []
    for label, delta in WINDOWS:
        if delta is None:
            start_equity = initial_capital
            basis = "reconstructed"
            note = f"against the ${initial_capital:,.0f} starting stake"
        else:
            cutoff = now - delta
            start_equity = equity_at(curve, cutoff, initial_capital)
            basis = "reconstructed"
            note = ""

            fills = _fills_in_window(events, cutoff, now)
            held_then = _held_at(events, cutoff)

            if not fills and not held_then and holdings <= 1e-9:
                # Truly no exposure at any point -> $0 is exact, not an estimate.
                start_equity = end_value
                basis = "flat_no_exposure"
                note = ("flat for this whole window -- no position was open, so no gain "
                        "or loss was possible (this is not the same as the bot being off)")
            elif held_then and holdings > 1e-9 and intraday_bars is not None:
                # Mark the SAME share count at both ends: the window P/L is then purely
                # the price move, which is what "what did the market do to my stake"
                # means. Trades inside the window are already reflected in end_value.
                p_start = price_at(intraday_bars, cutoff)
                p_now = float(intraday_bars["Close"].iloc[-1])
                if p_start is not None and p_start > 0:
                    start_equity = end_value + holdings * (p_start - p_now)
                    basis = "reconstructed_intraday"
                    note = "marked with GLD 5-minute bars"
                else:
                    basis = "includes_a_trade"
                    note = "a trade occurred in this window; intraday marks unavailable"
            elif fills:
                basis = "includes_a_trade"
                note = "a trade occurred in this window"

        pl_abs = end_value - start_equity
        pl_pct = (pl_abs / start_equity) if start_equity else 0.0
        windows.append(PlWindow(label, pl_abs, pl_pct, start_equity, end_value, basis, note))

    return {
        "windows": [w.as_dict() for w in windows],
        "initial_capital": initial_capital,
        "current_equity": round(end_value, 2),
        "total_pl_abs": round(end_value - initial_capital, 2),
        "total_pl_pct": round((end_value - initial_capital) / initial_capital, 6),
        "shares": holdings,
        "ledger_events": len(events),
        "fills": sum(1 for e in events if e.get("kind") == "fill"),
        "mark_source": mark_source,
        "intraday_marks": intraday_bars is not None,
        "first_fill_ts": curve[0][0].isoformat() if curve else None,
        "last_fill_ts": curve[-1][0].isoformat() if curve else None,
        "generated_at": now.isoformat(),
    }


def _fills_in_window(events: list[dict], start: datetime, end: datetime) -> list[dict]:
    """Fills whose timestamp falls in (start, end]."""
    out = []
    for ev in events:
        if ev.get("kind") != "fill":
            continue
        ts = _parse_ts(ev.get("ts", ""))
        if ts is None:
            continue
        if start < ts <= end:
            out.append(ev)
    return out


def _held_at(events: list[dict], when: datetime) -> bool:
    """Whether a position was held at `when`, from the last fill at or before it."""
    held = False
    for ev in sorted((e for e in events if e.get("kind") == "fill"),
                     key=lambda e: str(e.get("ts", ""))):
        ts = _parse_ts(ev.get("ts", ""))
        if ts is None or ts > when:
            break
        try:
            held = float(ev.get("shares", 0.0)) > 1e-9
        except (TypeError, ValueError):
            continue
    return held


def fetch_intraday_bars(symbol: str = "GLD") -> "pd.DataFrame | None":
    """GLD intraday bars, for marking a sub-day window. None if unavailable.

    Uses GLD's own prices, NOT spot XAU: the share count is in GLD units (~$400) while
    spot gold is ~$4,340, so mixing the two would be off by an order of magnitude.
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        return None
    try:
        bars = yf.Ticker(symbol).history(period="2d", interval="5m", auto_adjust=False)
        if bars is None or len(bars) == 0:
            return None
        idx = pd.DatetimeIndex(bars.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        bars.index = idx
        return bars
    except Exception:
        return None


def price_at(bars, when: datetime) -> float | None:
    """Last intraday close at or before `when`. None if the window is not covered."""
    if bars is None or len(bars) == 0:
        return None
    try:
        naive = when.replace(tzinfo=None) if when.tzinfo else when
        eligible = bars.loc[:naive]
        if len(eligible) == 0:
            return None
        return float(eligible["Close"].iloc[-1])
    except Exception:
        return None
