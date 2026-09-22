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

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 von-gold/0.1 research"

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


def load_lbma_gold(path: str | Path | None = None) -> pd.DataFrame:
    """LBMA gold price fix (USD/troy oz), daily, 1968-present.

    58 years of spot gold is the only way to test a long-only strategy against the
    1980-2000 and 2013-2015 secular bear markets. A GLD-only backtest cannot see them.

    Source file is a JSON list of {"d": "YYYY-MM-DD", "v": [usd, gbp, eur]}. Only the
    USD leg is used. LBMA quotes once daily, so the OHLC envelope is flat -- anything
    that needs a true intrabar range (ATR stops) is meaningless on this series.

    Resolution order: explicit path, then the copy committed under data/raw/ (so the
    long-history test works on a fresh clone), then the research scratch directory.
    """
    if path:
        candidates = [Path(path)]
    else:
        candidates = [
            DEFAULT_RAW_DIR / "lbma_gold_pm.json",
            Path.home() / ".hermes" / "cache" / "scratch" / "goldresearch" / "lbma_gold_pm.json",
        ]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        raise FileNotFoundError(
            "LBMA series not found. Looked in: "
            + ", ".join(str(c) for c in candidates)
            + ". Fetch it or use build_dataset() for GLD only."
        )
    raw = json.loads(p.read_text())
    rows = []
    for rec in raw:
        v = rec.get("v") or []
        if not v or v[0] is None:
            continue
        try:
            rows.append({"date": pd.Timestamp(rec["d"]), "close": float(v[0])})
        except (ValueError, TypeError, KeyError):
            continue
    df = pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()
    df.index = df.index.astype("datetime64[ms]")
    for c in ("open", "high", "low"):
        df[c] = df["close"]
    df["volume"] = 0.0
    df.index.name = "date"
    df.attrs["symbol"] = "LBMA-GOLD-PM"
    df.attrs["source"] = "lbma"
    return df


def load_local_parquet(path: str | Path) -> pd.DataFrame:
    """Load a cached OHLCV parquet (e.g. the yfinance GLD full history).

    Normalises column names and the index so it can go straight into run_backtest.
    """
    h = pd.read_parquet(path)
    h.columns = [str(c).lower().replace(" ", "_") for c in h.columns]
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in h.columns]
    df = h[cols].copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize().astype("datetime64[ms]")
    df.index.name = "date"
    return df.dropna(subset=["close"])


def _session() -> requests.Session:
    """A session with a warmed cookie jar.

    Yahoo's chart API returns HTTP 429 for most requests from an IP with no cookie,
    and rate-limits bursts even with one. Measured: a bare curl to query1 returned 429
    on nearly every attempt, while warming the jar from finance.yahoo.com first made it
    succeed. Requests without a browser-like User-Agent also failed outright.
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json,text/csv,text/html,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        s.get("https://finance.yahoo.com/quote/GLD", timeout=15)
    except Exception:
        pass  # a failed warm-up is not fatal; the chart call may still work
    return s


def _throttle(session: requests.Session, seconds: float) -> None:
    """Sleep between Yahoo calls. The endpoint tolerates roughly 1 req / 5-10s per IP."""
    import time

    time.sleep(seconds)


# Independent cross-check source for GLD daily bars. Verified 200 without a key, but it
# requires a browser User-Agent and returns numbers as comma/dollar-formatted STRINGS.
NASDAQ_HISTORICAL = "https://api.nasdaq.com/api/quote/{symbol}/historical"


def fetch_nasdaq_daily(symbol: str = "GLD", fromdate: str = "2004-01-01",
                       session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch daily OHLCV from Nasdaq's public API.

    Used as a cross-check on Yahoo, not as the primary source: it is daily-only and its
    values ar  strings. Having two independent sources is how a stale or split-adjusted
    Yahoo bar gets caught.
    """
    s = session or _session()
    r = s.get(
        NASDAQ_HISTORICAL.format(symbol=symbol),
        params={"assetclass": "etf", "fromdate": fromdate, "limit": 9999},
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        timeout=40,
    )
    if r.status_code != 200:
        raise RuntimeError(f"nasdaq {symbol}: HTTP {r.status_code}")
    payload = r.json()
    rows = (((payload.get("data") or {}).get("tradesTable") or {}).get("rows")) or []
    if not rows:
        raise RuntimeError(f"nasdaq {symbol}: no rows")
    recs = []
    for row in rows:
        try:
            recs.append({
                "date": pd.to_datetime(row["date"], format="%m/%d/%Y"),
                "close": float(str(row["close"]).replace("$", "").replace(",", "")),
                "open": float(str(row["open"]).replace("$", "").replace(",", "")),
                "high": float(str(row["high"]).replace("$", "").replace(",", "")),
                "low": float(str(row["low"]).replace("$", "").replace(",", "")),
                "volume": float(str(row["volume"]).replace(",", "") or 0),
            })
        except (KeyError, ValueError, TypeError):
            continue
    if not recs:
        raise RuntimeError(f"nasdaq {symbol}: no parsable rows")
    df = pd.DataFrame(recs).set_index("date").sort_index()
    df.index = df.index.astype("datetime64[ms]")
    df.index.name = "date"
    df.attrs["symbol"] = symbol
    df.attrs["source"] = "nasdaq"
    return df


def fetch_yahoo_daily(symbol: str, rng: str = "10y", session: requests.Session | None = None,
                      retries: int = 3, retry_wait: float = 6.0,
                      min_rows: int = 0) -> pd.DataFrame:
    """Fetch daily OHLCV from Yahoo's chart endpoint.

    Returns a DatetimeIndex-ed frame with columns: open, high, low, close, volume.
    Raises RuntimeError on a non-200 or an empty result set.

    Retries on 429 (rate limit). `min_rows` guards against a subtle silent failure:
    Yahoo can answer 200 with a SHORT series when it decides to ignore the requested
    range, so a caller that asked for 20 years of history can receive a few hundred
    rows without any error. Passing min_rows turns that into a loud failure.
    """
    import time

    s = session or _session()
    url = YAHOO_CHART.format(symbol=symbol)
    last_err = None
    for attempt in range(retries + 1):
        r = s.get(url, params={"range": rng, "interval": "1d"}, timeout=40)
        if r.status_code == 429:
            last_err = f"HTTP 429 (rate limited) after {attempt + 1} attempts"
            if attempt < retries:
                time.sleep(retry_wait * (attempt + 1))
                continue
            raise RuntimeError(f"yahoo {symbol}: {last_err}")
        if r.status_code != 200:
            raise RuntimeError(f"yahoo {symbol}: HTTP {r.status_code}")
        payload = r.json()
        results = (payload.get("chart") or {}).get("result") or []
        if not results:
            err = (payload.get("chart") or {}).get("error")
            last_err = f"empty result ({err})"
            if attempt < retries:
                time.sleep(retry_wait)
                continue
            raise RuntimeError(f"yahoo {symbol}: {last_err}")
        res = results[0]
        ts = res.get("timestamp") or []
        if not ts:
            last_err = "no timestamps"
            if attempt < retries:
                time.sleep(retry_wait)
                continue
            raise RuntimeError(f"yahoo {symbol}: {last_err}")

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

        if min_rows and len(df) < min_rows:
            last_err = (f"only {len(df)} rows returned for range={rng} "
                        f"(expected at least {min_rows}) -- likely a silent range downgrade")
            if attempt < retries:
                time.sleep(retry_wait * (attempt + 1))
                continue
            raise RuntimeError(f"yahoo {symbol}: {last_err}")
        return df

    raise RuntimeError(f"yahoo {symbol}: {last_err or 'unknown failure'}")


def fetch_yfinance_daily(symbol: str = "GLD", period: str = "10y") -> pd.DataFrame:
    """Fetch daily OHLCV via the yfinance library.

    Preferred over raw Yahoo: the library performs the cookie/crumb handshake itself, so
    it keeps working when direct requests to query1/query2 return 429. Measured on this
    machine while raw curl was returning 429 on every attempt, yfinance returned fine.

    Raises RuntimeError if the library is missing or returns nothing -- callers fall back
    to the next source rather than silently proceeding with no data.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance not installed") from exc

    tk = yf.Ticker(symbol)
    h = tk.history(period=period, interval="1d", auto_adjust=False)
    if h is None or len(h) == 0:
        raise RuntimeError(f"yfinance {symbol}: empty result for period={period}")
    h.columns = [str(c).lower().replace(" ", "_") for c in h.columns]
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in h.columns]
    if "close" not in cols:
        raise RuntimeError(f"yfinance {symbol}: no close column (got {list(h.columns)})")
    df = h[cols].copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize().astype("datetime64[ms]")
    df.index.name = "date"
    for c in ("open", "high", "low"):
        if c in df.columns:
            df[c] = df[c].fillna(df["close"])
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0.0)
    df = df.dropna(subset=["close"])[~df.index.duplicated(keep="last")].sort_index()
    df.attrs["symbol"] = symbol
    df.attrs["source"] = "yfinance"
    return df


def fetch_prices(symbol: str = "GLD", rng: str = "10y",
                 session: requests.Session | None = None,
                 min_rows: int = 0) -> tuple[pd.DataFrame, str]:
    """Fetch daily prices, trying every source until one works.

    Returns (frame, source_name). A single-source fetch is a single point of failure: raw
    Yahoo was observed 429ing on *every* request from this IP while yfinance worked fine,
    so the order is yfinance -> raw Yahoo -> Nasdaq.

    Raises the collected errors only if all three fail, so a caller can never mistake
    "no data" for "flat market".
    """
    errors: list[str] = []
    for name, fn in (
        ("yfinance", lambda: fetch_yfinance_daily(symbol, period=rng)),
        ("yahoo", lambda: fetch_yahoo_daily(symbol, rng=rng, session=session, min_rows=min_rows)),
        ("nasdaq", lambda: fetch_nasdaq_daily(symbol)),
    ):
        try:
            df = fn()
            if min_rows and len(df) < min_rows and name != "nasdaq":
                raise RuntimeError(f"only {len(df)} rows (wanted {min_rows})")
            df.attrs["source"] = name
            return df, name
        except Exception as exc:  # try the next source
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"all price sources failed for {symbol} -- " + " | ".join(errors))


def latest_bar_age_days(df: pd.DataFrame, today: pd.Timestamp | None = None) -> int:
    """Calendar days between today and the last bar. Used to detect a stale cache."""
    if df is None or len(df) == 0:
        return 10 ** 6
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    return int((today - pd.Timestamp(df.index[-1]).normalize()).days)


def fetch_fred(series: str, session: requests.Session | None = None,
               timeout: float = 15.0) -> pd.Series:
    """Fetch one FRED series as a float Series indexed by date.

    FRED publishes '.' for missing observations (holidays) -- coerced to NaN.

    DO NOT reuse the browser-flavoured session used for Yahoo. Measured on this machine:
    fredgraph.csv with a Chrome User-Agent returns a connection-level failure (curl
    http=000), while the identical request with no User-Agent returns 200 and the full
    CSV. FRED evidently filters browser UA strings on this endpoint, so this uses a plain
    identifier instead. The earlier 40s timeout also meant five failing series cost ~200s
    per tick; 15s keeps a bad macro day cheap.
    """
    url = FRED_CSV.format(series=series)
    attempts = (
        {"User-Agent": "von-gold/0.1 (research; +https://fred.stlouisfed.org)"},
        {},  # second try with no UA at all
    )
    last_err: Exception | None = None
    r = None
    for headers in attempts:
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200 and len(r.text) > 100:
                break
            last_err = RuntimeError(f"HTTP {r.status_code}, {len(r.text)} bytes")
            r = None
        except Exception as exc:
            last_err = exc
            r = None
    if r is None:
        raise RuntimeError(f"fred {series}: {last_err}")

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
    max_stale_days: int = 4,
) -> pd.DataFrame:
    """Build the joined price+macro dataset, caching raw JSON/CSV and the result.

    Returns a frame indexed by trading date with price columns plus, for each macro
    series, a `*_ffill` column forward-filled onto the trading calendar (macro series
    publish on their own schedule; forward fill is the only honest alignment for a
    daily decision made before the next release).

    `max_stale_days` guards the cache. A pure "cache exists -> return it" check is a
    silent-failure trap for a daily bot: the cache is written once and then served
    forever, so the strategy would keep trading on the last day it happened to fetch,
    with no error anywhere. Observed in practice. The default of 4 covers a long weekend
    plus a holiday; anything older triggers a refetch.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_path = processed_dir / f"dataset_{symbol}_{rng}.parquet"
    if out_path.exists() and not refresh:
        cached = pd.read_parquet(out_path)
        age = latest_bar_age_days(cached)
        if age <= max_stale_days:
            return cached
        print(f"  cache is {age} days stale (last bar {cached.index[-1].date()}); refreshing")

    s = _session()
    # Ask for max history and demand a real answer: a bare "max" request was observed
    # returning 263 rows instead of ~5,500 (silent range downgrade / rate limiting).
    expected_min = 4000 if rng in ("max", "30y") else 0
    px, source = fetch_prices(symbol, rng=rng, session=s, min_rows=expected_min)
    (raw_dir / f"yahoo_{symbol}_{rng}.json").write_text(
        json.dumps({
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "source": source,
            "rows": int(len(px)),
            "last_bar": str(px.index[-1].date()),
        })
    )
    print(f"  prices via {source}: {len(px)} rows, last bar {px.index[-1].date()}")

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
    df.attrs["source"] = source
    df.to_parquet(out_path)
    return df


def load_cached_bars(
    symbols: Iterable[str] = ("GLD", "IAU"),
    rng: str = "10y",
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Convenience: {symbol: frame} for several instruments."""
    return {sym: build_dataset(sym, rng=rng, refresh=refresh) for sym in symbols}
