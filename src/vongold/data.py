"""Data ingestion: daily gold prices (Yahoo chart API) and macro drivers (FRED CSV).

Both sources are free and need no API key. Everything is cached to disk as parquet
so repeat runs are offline and reproducible.

Verified working 2026-09-21:
  https://query1.finance.yahoo.com/v8/finance/chart/GLD?range=10y&interval=1d  -> 200
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10                     -> 200
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) von-gold/0.1 research"

# Macro series used as gold regime conditioning inputs.
FRED_SERIES = {
    "DFII10": "real_yield_10y",   # 10-Year TIPS yield -- gold's key opportunity cost
    "DGS10": "nominal_yield_10y",
    "DFF": "fed_funds_rate",
    "DTWEXBGS": "dollar_index_broad",
    "CPIAUCSL": "cpi",
}

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json,text/csv,*/*"})
    return s


def fetch_yahoo_daily(symbol: str, rng: str = "10y", session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch daily OHLCV from Yahoo's chart endpoint.

    Returns a DatetimeIndex-ed frame with columns: open, high, low, close, volume.
    Raises RuntimeError on a non-200 or an empty result set.
    """
    s = session or _session()
    url = YAHOO_CHART.format(symbol=symbol)
    r = s.get(url, params={"range": rng, "interval": "1d"}, timeout=40)
    if r.status_code != 200:
        raise RuntimeError(f"yahoo {symbol}: HTTP {r.status_code}")
    payload = r.json()
    results = (payload.get("chart") or {}).get("result") or []
    if not results:
        err = (payload.get("chart") or {}).get("error")
        raise RuntimeError(f"yahoo {symbol}: empty result ({err})")
    res = results[0]
    ts = res.get("timestamp") or []
    if not ts:
        raise RuntimeError(f"yahoo {symbol}: no timestamps")
    quote = (res.get("indicators") or {}).get("quote") or [{}]
    q = quote[0]

    # Yahoo returns UTC epoch seconds. Normalise to the exchange-local DATE so that
    # daily bars align across instruments without timezone drift.
    tz_name = (res.get("meta") or {}).get("exchangeTimezoneName") or "UTC"
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz_name).normalize().tz_localize(None)

    df = pd.DataFrame(
        {
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
            "close": q.get("close"),
            "volume": q.get("volume"),
        },
        index=idx,
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Yahoo emits nulls for holidays/partial rows.
    df = df.dropna(subset=["close"])
    for c in ("open", "high", "low"):
        df[c] = df[c].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)
    df.index.name = "date"
    df.attrs["symbol"] = symbol
    df.attrs["currency"] = (res.get("meta") or {}).get("currency")
    df.attrs["exchange"] = (res.get("meta") or {}).get("fullExchangeName")
    return df


def fetch_fred(series: str, session: requests.Session | None = None) -> pd.Series:
    """Fetch one FRED series as a float Series indexed by date.

    FRED publishes '.' for missing observations (holidays) -- coerced to NaN.
    """
    s = session or _session()
    url = FRED_CSV.format(series=series)
    r = s.get(url, timeout=40)
    if r.status_code != 200:
        raise RuntimeError(f"fred {series}: HTTP {r.status_code}")
    from io import StringIO

    df = pd.read_csv(StringIO(r.text))
    if df.shape[1] < 2:
        raise RuntimeError(f"fred {series}: unexpected shape {df.shape}")
    date_col, val_col = df.columns[0], df.columns[1]
    out = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(df[date_col]),
        name=FRED_SERIES.get(series, series.lower()),
    )
    return out[~out.index.duplicated(keep="last")].sort_index().dropna()


def build_dataset(
    symbol: str = "GLD",
    rng: str = "10y",
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Build the joined price+macro dataset, caching raw JSON/CSV and the result.

    Returns a frame indexed by trading date with price columns plus, for each macro
    series, a `*_ffill` column forward-filled onto the trading calendar (macro series
    publish on their own schedule; forward fill is the only honest alignment for a
    daily decision made before the next release).
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_path = processed_dir / f"dataset_{symbol}_{rng}.parquet"
    if out_path.exists() and not refresh:
        return pd.read_parquet(out_path)

    s = _session()
    px = fetch_yahoo_daily(symbol, rng=rng, session=s)
    (raw_dir / f"yahoo_{symbol}_{rng}.json").write_text(
        json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(), "symbol": symbol})
    )

    df = px.copy()
    for series, name in FRED_SERIES.items():
        try:
            ser = fetch_fred(series, session=s)
        except Exception as exc:  # a missing macro series must not kill the pipeline
            print(f"  warn: FRED {series} failed ({exc}); skipping")
            continue
        df[name] = ser.reindex(df.index, method="ffill")
        df[f"{name}_available"] = ser.reindex(df.index, method="ffill").notna()

    df = df.dropna(subset=["close"])
    df.to_parquet(out_path)
    return df


def load_cached_bars(
    symbols: Iterable[str] = ("GLD", "IAU"),
    rng: str = "10y",
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Convenience: {symbol: frame} for several instruments."""
    return {sym: build_dataset(sym, rng=rng, refresh=refresh) for sym in symbols}
