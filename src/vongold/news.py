"""News ingestion and local classification.

Two jobs:
  1. Pull gold-relevant headlines from free feeds (no API key).
  2. Turn them into something a decision can use WITHOUT an LLM.

Job 2 is where most "AI trading bot" designs quietly cheat: they call a frontier
model to read the news. That costs money, is non-reproducible, and for a backtest
would leak lookahead. Instead we do it deterministically:

  * **Lexical bucketing** -- a fixed, versioned keyword lexicon assigns each headline
    to a driver bucket (real yields, dollar, inflation, Fed policy, safe-haven demand,
    physical demand, positioning). The lexicon is checked into the repo and hashed, so
    a backtest can state exactly which lexicon produced it.
  * **Sentiment** via a small finance-tuned word list -- crude on purpose. We report
    its measured correlation with next-day gold returns rather than asserting it works.
  * **Novelty** -- headlines are deduplicated by token overlap so a story that runs
    40 times does not read as 40 signals.
  * **Volume spike** -- the count of gold-relevant headlines per day is itself a
    feature (attention/uncertainty proxy).

The resulting daily feature vector is what the strategy (and von) sees. If a headline
feed is unavailable, the feature degrades to neutral rather than blocking the run.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# --- Fixed, versioned lexicon. Changing this changes the strategy; bump LEXICON_VERSION. ---
LEXICON_VERSION = "1.0.0"

BUCKETS: dict[str, list[str]] = {
    "real_yields": [
        "real yield", "tips", "treasury yield", "yields rise", "yields fall", "bond yield",
        "interest rate", "rate cut", "rate hike", "fed rate", "yield curve",
    ],
    "dollar": [
        "dollar", "greenback", "dxy", "currency", "euro", "yen", "yuan", "devalu",
    ],
    "inflation": [
        "inflation", "cpi", "pce", "price index", "deflation", "stagflation",
        "cost of living", "consumer prices",
    ],
    "fed_policy": [
        "federal reserve", "fed ", "fomc", "powell", "central bank", "ecb", "boj",
        "monetary policy", "quantitative", "hawkish", "dovish",
    ],
    "safe_haven": [
        "safe haven", "safe-haven", "geopolitic", "war", "conflict", "tension",
        "crisis", "recession", "risk-off", "sanction", "tariff", "uncertainty",
    ],
    "physical_demand": [
        "central bank buying", "jewelry", "jewellery", "india", "china demand",
        "wedding season", "festival", "etf inflow", "etf outflow", "coin", "bar demand",
        "mine", "mining", "supply",
    ],
    "positioning": [
        "speculative", "net long", "net short", "comex", "futures position",
        "open interest", "short covering", "hedge fund",
    ],
}

POSITIVE = [
    "rise", "rises", "rally", "rallies", "gain", "gains", "surge", "soar", "climb",
    "jump", "high", "record", "boost", "support", "bullish", "upside", "strong",
    "demand", "inflow", "safe haven", "haven", "higher",
]
NEGATIVE = [
    "fall", "falls", "drop", "drops", "slide", "slump", "plunge", "tumble", "decline",
    "sink", "low", "weak", "bearish", "downside", "pressure", "outflow", "sell-off",
    "selloff", "lower", "loss", "losses", "retreat",
]

GOLD_TERMS = [
    "gold", "xauusd", "bullion", "gld", "precious metal", "gold price", "gold market",
    "gold futures", "gold etf",
]

TOKEN_RE = re.compile(r"[a-z0-9']+")


def lexicon_hash() -> str:
    payload = repr(sorted((k, sorted(v)) for k, v in BUCKETS.items())) + repr(sorted(POSITIVE)) + repr(sorted(NEGATIVE))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _is_gold_relevant(text: str) -> bool:
    t = text.lower()
    return any(term in t for term in GOLD_TERMS)


def classify_headline(title: str, summary: str = "") -> dict:
    """Deterministic classification of one headline. No model, no network."""
    text = f"{title} {summary}".lower()
    buckets = {name: int(any(term in text for term in terms)) for name, terms in BUCKETS.items()}
    pos = sum(1 for w in POSITIVE if w in text)
    neg = sum(1 for w in NEGATIVE if w in text)
    total = pos + neg
    sentiment = (pos - neg) / total if total else 0.0
    return {
        "buckets": buckets,
        "bucket_hits": sum(buckets.values()),
        "sentiment": sentiment,
        "tokens": frozenset(TOKEN_RE.findall(text)),
    }


def dedupe_headlines(items: list[dict], overlap_threshold: float = 0.7) -> list[dict]:
    """Drop near-duplicate stories by token-set Jaccard overlap.

    A single wire story gets republished by many outlets; without this, one event
    looks like a burst of independent confirmation.
    """
    kept: list[dict] = []
    seen: list[frozenset[str]] = []
    for it in items:
        toks = it["tokens"]
        if not toks:
            continue
        dup = False
        for prev in seen:
            if not prev:
                continue
            inter = len(toks & prev)
            union = len(toks | prev)
            if union and inter / union >= overlap_threshold:
                dup = True
                break
        if not dup:
            kept.append(it)
            seen.append(toks)
    return kept


DEFAULT_FEEDS = [
    # Verified reachable without a key (see docs/DATA.md).
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GLD&region=US&lang=en-US",
    "https://www.marketwatch.com/rss/topstories",
    "https://www.investing.com/rss/news_285.rss",
    "https://www.kitco.com/rss/",
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://news.google.com/rss/search?q=gold+price&hl=en-US&gl=US&ceid=US:en",
]


def fetch_feed(url: str, timeout: int = 20, session: requests.Session | None = None) -> list[dict]:
    """Fetch and parse one RSS/Atom feed. Returns [] on failure (never raises)."""
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", "Mozilla/5.0 von-gold/0.1")
    try:
        r = s.get(url, timeout=timeout)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
    except Exception:
        return []

    items = []
    # RSS 2.0
    for it in root.iter():
        tag = it.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        title, summary, pub = "", "", None
        for child in it:
            ctag = child.tag.split("}")[-1]
            if ctag == "title":
                title = (child.text or "").strip()
            elif ctag in ("description", "summary", "content"):
                summary = (child.text or "").strip()[:400]
            elif ctag in ("pubDate", "published", "updated"):
                pub = (child.text or "").strip()
        if not title:
            continue
        items.append({"title": title, "summary": summary, "published": pub, "feed": url})
    return items


def _parse_date(raw: str | None) -> pd.Timestamp | None:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return pd.Timestamp(datetime.strptime(raw, fmt)).tz_localize(None) if "%z" in fmt or fmt.endswith("Z") is False else pd.Timestamp(raw).tz_localize(None)
        except Exception:
            continue
    try:
        ts = pd.Timestamp(raw)
        return ts.tz_localize(None) if ts.tzinfo is None else ts.tz_convert(None)
    except Exception:
        return None


def news_daily_features(
    feeds: list[str] | None = None,
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Build a daily news-feature frame: bucket hit counts, sentiment, volume.

    Note on time alignment: a headline's date is when it was PUBLISHED. A strategy
    trading on day t may only use headlines timestamped at or before t's close, which
    is exactly what a date-indexed frame gives us when we then lag it by one day in
    the strategy.
    """
    feeds = feeds or DEFAULT_FEEDS
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists() and not refresh:
        return pd.read_parquet(cache)

    s = requests.Session()
    raw: list[dict] = []
    for url in feeds:
        got = fetch_feed(url, session=s)
        raw.extend(got)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(raw).to_parquet(cache.parent / "news_raw.parquet")

    relevant = [it for it in raw if _is_gold_relevant(f"{it['title']} {it['summary']}")]
    classified = []
    for it in relevant:
        c = classify_headline(it["title"], it["summary"])
        c["title"] = it["title"]
        c["date"] = _parse_date(it.get("published"))
        c["feed"] = it.get("feed")
        classified.append(c)

    classified = dedupe_headlines(classified)
    if not classified:
        return pd.DataFrame(
            columns=["n_headlines"] + [f"news_{b}" for b in BUCKETS] + ["news_sentiment", "news_bucket_hits"]
        )

    rows = []
    for c in classified:
        rec = {"date": c["date"], "sentiment": c["sentiment"], "bucket_hits": c["bucket_hits"]}
        for b in BUCKETS:
            rec[f"news_{b}"] = c["buckets"][b]
        rows.append(rec)
    df = pd.DataFrame(rows).dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame(columns=["n_headlines"] + [f"news_{b}" for b in BUCKETS] + ["news_sentiment", "news_bucket_hits"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # Normalise the dtype: pandas produces datetime64[us] here while the price frame
    # uses datetime64[ms]. Index intersection/join requires the SAME unit, or the
    # frames silently fail to align.
    df["date"] = df["date"].astype("datetime64[ms]")
    agg = df.drop(columns=["date"]).groupby(df["date"]).agg(
        {**{f"news_{b}": "sum" for b in BUCKETS}, "sentiment": "mean", "bucket_hits": "sum"}
    )
    agg["n_headlines"] = df.groupby(df["date"]).size()
    agg["news_sentiment"] = agg.pop("sentiment")
    if cache:
        agg.to_parquet(cache)
    return agg


def news_impact_report(news: pd.DataFrame, prices: pd.DataFrame, horizons=(1, 5, 10)) -> pd.DataFrame:
    """Measure whether the news features actually predict gold returns.

    Reported, never assumed. If the correlation is noise, the feature gets dropped
    from the strategy rather than kept because it sounds sophisticated.
    """
    if news.empty:
        return pd.DataFrame()
    # Reindex onto the PRICE calendar so weekend/holiday headlines land on the next
    # trading day. But do NOT count forward-filled days as observations: propagating
    # one headline across 40 silent sessions and then correlating would manufacture
    # hundreds of fake data points from a handful of real ones. So we track a coverage
    # flag and require the minimum sample on ACTUAL news days.
    union_idx = prices.index.union(news.index)
    aligned = news.reindex(union_idx).ffill().reindex(prices.index)
    has_news = news.reindex(union_idx).notna().any(axis=1).reindex(prices.index).fillna(False)
    aligned["has_news"] = has_news.astype(float)

    coverage_days = int(has_news.sum())
    px = prices["close"].pct_change()
    joined = aligned.join(px.rename("ret"), how="inner").dropna(subset=["ret"])

    MIN_OBS = 60
    shortfall_note = (
        f"news coverage is only {coverage_days} trading days; feeds carry recent "
        f"headlines only and there is no free historical archive (see docs/DATA.md). "
        f"Accumulate forward -- this is a missing result, not a negative one."
    )
    if coverage_days < MIN_OBS:
        return pd.DataFrame([{
            "feature": "(insufficient news coverage)",
            "horizon_days": 0,
            "n": coverage_days,
            "corr": float("nan"),
            "mean_fwd_when_active": float("nan"),
            "mean_fwd_when_quiet": float("nan"),
            "note": shortfall_note,
        }])

    rows = []
    feats = [c for c in joined.columns if c.startswith("news_") or c == "n_headlines"]
    # Restrict every measurement to genuine coverage days, plus a "days since a headline"
    # feature so that silence itself can be tested as an input.
    coverage = joined[joined["has_news"] > 0]
    for f in feats:
        for h in horizons:
            fwd = prices["close"].shift(-h) / prices["close"] - 1.0
            j = coverage[[f]].join(fwd.rename("fwd"), how="inner").dropna()
            if len(j) < MIN_OBS or j[f].std() == 0:
                continue
            rows.append(
                {
                    "feature": f,
                    "horizon_days": h,
                    "n": len(j),
                    "corr": float(j[f].corr(j["fwd"])),
                    "mean_fwd_when_active": float(j.loc[j[f] > 0, "fwd"].mean()) if (j[f] > 0).any() else np.nan,
                    "mean_fwd_when_quiet": float(j.loc[j[f] == 0, "fwd"].mean()) if (j[f] == 0).any() else np.nan,
                }
            )
    if not rows:
        return pd.DataFrame([{
            "feature": "(no feature met the minimum sample)",
            "horizon_days": 0,
            "n": coverage_days,
            "corr": float("nan"),
            "mean_fwd_when_active": float("nan"),
            "mean_fwd_when_quiet": float("nan"),
            "note": shortfall_note,
        }])
    return pd.DataFrame(rows)
