"""Live price marks so the paper account reflects current value, not last night's close.

The strategy's SIGNAL is deliberately daily -- it is built from daily closes and reading
intraday bars would change its meaning. But the paper account's VALUE does not have to be
daily. Marking a position at yesterday's close makes a live dashboard show a stale number
all day, which reads as broken.

So the two are separated:

  * **signal**  <- daily bars, unchanged (no lookahead, same as the backtest)
  * **mark**    <- latest available price, intraday when the market is open

Trades execute at the live mark, which is what a real fill would do.

Freshness is stated, never assumed. Every mark carries the timestamp it came from and
whether it is intraday or a prior close, so the dashboard can say "as of 14:32 ET" instead
of implying a live quote it does not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any

# US equities regular session, in exchange-local terms.
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


@dataclass
class Mark:
    """A price observation with its provenance."""

    price: float
    as_of: datetime
    source: str
    is_intraday: bool
    market_open: bool
    age_minutes: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "price": round(self.price, 4),
            "as_of": self.as_of.isoformat(),
            "source": self.source,
            "is_intraday": self.is_intraday,
            "market_open": self.market_open,
            "age_minutes": round(self.age_minutes, 1),
        }


def market_is_open(now_utc: datetime | None = None) -> bool:
    """Whether the US cash session is currently open (holidays NOT handled).

    Deliberately simple: a holiday would be reported as open, and the worst consequence is
    that a mark is labelled intraday when the tape is actually stale. The `age_minutes`
    field is what callers should trust for freshness, not this flag.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    # ET is UTC-4 (EDT) or UTC-5 (EST). Use a fixed -4: being an hour off only shifts the
    # boundary, and age_minutes remains the authoritative freshness signal.
    et = now_utc - timedelta(hours=4)
    if et.weekday() >= 5:  # Sat/Sun
        return False
    return MARKET_OPEN <= et.time() <= MARKET_CLOSE


def fetch_live_mark(symbol: str = "GLD",
                    now_utc: datetime | None = None) -> Mark | None:
    """Latest price for `symbol`: intraday bar if available, else the last daily close.

    Returns None rather than a guessed value. A caller that cannot get a mark must fall
    back to a known price -- inventing one would silently corrupt the paper account.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    open_now = market_is_open(now_utc)

    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        return None

    # Prefer intraday bars; they are what make the dashboard feel live.
    for period, interval in (("1d", "1m"), ("5d", "5m"), ("1mo", "1h")):
        try:
            bars = yf.Ticker(symbol).history(period=period, interval=interval,
                                             auto_adjust=False)
            if bars is None or len(bars) == 0:
                continue
            idx = pd.DatetimeIndex(bars.index)
            last_ts = idx[-1]
            price = float(bars["Close"].iloc[-1])
            if not (price > 0):
                continue

            ts = last_ts.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = (now_utc - ts.astimezone(timezone.utc)).total_seconds() / 60.0
            # An intraday bar older than ~20 min means the session is not trading now.
            is_intraday = open_now and age <= 20
            return Mark(price=price, as_of=ts, source=f"yfinance {interval} bar",
                        is_intraday=is_intraday, market_open=open_now, age_minutes=age)
        except Exception:
            continue

    # Fall back to the daily close.
    try:
        daily = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=False)
        if daily is not None and len(daily) > 0:
            idx = pd.DatetimeIndex(daily.index)
            ts = idx[-1].to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return Mark(price=float(daily["Close"].iloc[-1]), as_of=ts,
                        source="yfinance daily close", is_intraday=False,
                        market_open=open_now,
                        age_minutes=(now_utc - ts.astimezone(timezone.utc)).total_seconds() / 60.0)
    except Exception:
        pass
    return None


def spot_cross_check(now_utc: datetime | None = None) -> dict[str, Any] | None:
    """Live spot XAU/USD, for sanity-checking that GLD is not stale.

    Not used to price the paper account: the position is held in GLD *shares*, and GLD
    trades near $400 while spot gold is near $4,340. Mixing the two would be off by an
    order of magnitude. This exists so a human can see both and notice if one is frozen.
    """
    import requests

    try:
        r = requests.get("https://api.gold-api.com/price/XAU", timeout=8)
        if r.status_code != 200:
            return None
        payload = r.json()
        return {
            "spot_xau_usd": float(payload.get("price")),
            "updated_at": payload.get("updatedAt"),
            "source": "gold-api.com",
        }
    except Exception:
        return None
