# Data sources — verified by direct call

Every endpoint below was called with `curl` on 2026-09-21 and the HTTP status recorded.
Anything not verified is marked as such. No endpoint here needs an API key.

## Prices

| Source | URL | Status | Depth | Notes |
|---|---|---|---|---|
| Yahoo chart, GLD | `https://query1.finance.yahoo.com/v8/finance/chart/GLD?range=10y&interval=1d` | **200** | 10y daily, 2,512 bars | Primary gold ETF series |
| Yahoo chart, GLD, `range=max` | same endpoint with `range=max` | **429 / silent-short** | unreliable | **Do not trust this.** See the warning below. |
| yfinance, GLD, `period=max` | `yfinance` package, daily | **200** | **5,493 bars, 2004-11-18 → 2026-09-21** | The working way to get full GLD history. Cached to `data/processed/gld_yfinance_max.parquet`. |
| Nasdaq historical API, GLD | `https://api.nasdaq.com/api/quote/GLD/historical?assetclass=etf&fromdate=2004-01-01&limit=9999` | **200** | 2,513 rows | **Independent cross-check.** Agrees with Yahoo to the cent on 2026-09-21 (398.38) and on the 2016 open (127.27). Use it to validate Yahoo. |
| LBMA gold PM fix | local file `data/raw/lbma_gold_pm.json` (committed, 915KB) | **on disk** | **14,687 daily, 1968-04-01 → 2026-09-21** | **58 years** of spot gold. The only way to test against the 1980-2000 and 2013-2015 secular bears. |
| Yahoo chart, IAU | `https://query1.finance.yahoo.com/v8/finance/chart/IAU?range=10y&interval=1d` | **200** | 10y daily | Cheaper expense-ratio twin of GLD; cross-check |
| Yahoo chart, GC=F | `https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range=10y&interval=1d` | **200** | 10y daily (continuous futures) | Closest to spot gold; has roll artefacts |
| Yahoo chart, XAUUSD=X | `https://query1.finance.yahoo.com/v8/finance/chart/XAUUSD=X` | **404** | — | Does **not** work; don't build on it |
| Stooq CSV | `https://stooq.com/q/d/l/?s=xauusd&i=d` | **blocked** | — | Returns a JavaScript proof-of-work bot wall, not CSV |

### ⚠️ Yahoo silently returns a SHORT series, not an error

Observed in practice: `range=max` returned **200 OK with 263 rows** instead of ~5,500, and
later returned hard **429**s. A caller that trusts the status code gets a truncated price
history and a corrupted backtest with no warning from anywhere.

`fetch_yahoo_daily(..., min_rows=N)` now turns that into a loud `RuntimeError` instead, and
`build_dataset` demands ≥4,000 rows for `max`/`30y` requests. Prefer `yfinance` for full
history, and cross-check the last close against Nasdaq's independent API.

Requires a `User-Agent` header. Without one Yahoo may refuse.

### Price sources are tried in order, and FRED must NOT see a browser UA

Two failures found in production, both silent, both now fixed and pinned by tests:

**1. Single-source fetch is a single point of failure.** Raw requests to
`query1.finance.yahoo.com` were observed returning **429 on every attempt** from this IP
while `yfinance` — which performs its own cookie/crumb handshake — returned the full
series fine. `fetch_prices()` now tries **yfinance → raw Yahoo → Nasdaq** and reports which
source it used (recorded in `data/raw/yahoo_GLD_10y.json`). All three must fail before it
raises, so "no data" can never be mistaken for "flat market".

**2. FRED rejects the browser User-Agent that Yahoo requires.** Measured directly:

```
fredgraph.csv?id=DFII10  with Chrome UA     -> http=000 (connection failure)
fredgraph.csv?id=DFII10  with no UA         -> http=200, 98,955 bytes
```

Reusing one session for both hosts made **every** macro series fail, each burning a 40s
timeout — ~200s per tick — and silently dropped all five macro columns. `fetch_fred()` now
sends a plain identifier and uses a 15s timeout. Full rebuild went **210s → 16.8s** with
all macro series loading.

### ⚠️ A cache with no freshness check makes a daily bot trade stale prices forever

`build_dataset()` originally did: *if the parquet exists, return it*. The cache is written
once and then served forever, so the dry run would keep trading on the last day it happened
to fetch — with **no error anywhere**, because every downstream number (vol, MA200,
momentum) is computed from the same stale frame and looks perfectly self-consistent.

`max_stale_days=4` (long weekend + holiday) now forces a refetch. Verified by truncating
the cache by 30 days: the next call logged `cache is 31 days stale; refreshing` and
refetched automatically.

### Choosing the instrument

`GLD` is the default because it is what a retail account can actually buy and sell, so
the simulated costs are real. `GC=F` tracks spot more closely but is a rolling futures
contract — its back-adjusted price series contains roll gaps a naive backtest will
mistake for returns. Spot XAU/USD would be ideal but has no free, reliable daily source.
`IAU` is worth running as a robustness check: same exposure, ~0.25% vs ~0.40% fee, so
the strategy should look *better* there, and if it does not, something is wrong.

## Macro drivers (FRED, public CSV, no key)

`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>`

| Series | Meaning | Status | Role in the strategy |
|---|---|---|---|
| `DFII10` | 10-year TIPS (real) yield | **200** | Gold's opportunity cost. Best-documented macro driver. |
| `DGS10` | 10-year nominal yield | **200** | Context / inflation-expectation proxy |
| `DFF` | Effective federal funds rate | **200** | Policy regime |
| `CPIAUCSL` | CPI, monthly | **200** | Inflation backdrop |
| `DTWEXBGS` | Broad trade-weighted dollar index | **200** | Currency headwind |

Publication lag is real anyway (a FRED observation dated Friday may not have been visible
until the following week) — a known, small optimism in the backtest. A stricter build would
shift each series by its actual release lag; that is listed as future work rather than
silently assumed away.

**Note:** as of the long-history work, both macro filters are OFF by default because a
lagged A/B showed them *removing* Sharpe (see `docs/STRATEGY.md`). These series are still
fetched and reported on the dashboard; they no longer gate the position. That removes the
publication-lag optimism from the *result* (it only affects reporting now).

## News

`news.py` ships these feeds. Reachability measured on the same date:

| Feed | URL | Items seen |
|---|---|---|
| Google News, gold query | `https://news.google.com/rss/search?q=gold+price&hl=en-US&gl=US&ceid=US:en` | **100** |
| Federal Reserve press releases | `https://www.federalreserve.gov/feeds/press_all.xml` | **20** |
| MarketWatch top stories | `https://www.marketwatch.com/rss/topstories` | **10** |
| Investing.com commodities | `https://www.investing.com/rss/news_285.rss` | **10** |
| Yahoo Finance headline, GLD | `https://feeds.finance.yahoo.com/rss/2.0/headline?s=GLD&region=US&lang=en-US` | **0** (dead) |
| Kitco | `https://www.kitco.com/rss/` | **0** (dead) |

The two dead feeds are left in the list on purpose: `fetch_feed` returns `[]` on failure
instead of raising, so a feed dying never breaks a run. Feeds are one input among
several, and the system is designed to work with none of them.

### The lookahead problem with news

A feed serves *current* headlines. You cannot retrieve "what headlines existed on
2021-03-04" from any of these. So there is **no honest long history of news features**
available here, and a backtest that splices today's feed into 10 years of prices would be
committing a serious lookahead error.

Consequences, stated plainly:

* The news layer is wired and measurable **going forward** (it accumulates), not
  retroactively.
* `news_impact_report()` measures whether the features predict gold returns on whatever
  history has accumulated, rather than asserting they do.
* The strategy does **not** currently size on news. `news_daily_features()` produces the
  features and the report tells you whether they earned a place. Until a feature shows a
  stable relationship out of sample, adding it would be decoration.

Historical news would need a paid archive (RavenPack, GDELT's full history, a
NewsAPI/Bloomberg licence). GDELT's free tier is the only realistic no-cost option for
history and is noted as a future path, not wired in.
