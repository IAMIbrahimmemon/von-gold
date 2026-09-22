# Data sources — verified by direct call

Every endpoint below was called with `curl` on 2026-09-21 and the HTTP status recorded.
Anything not verified is marked as such. No endpoint here needs an API key.

## Prices

| Source | URL | Status | Depth | Notes |
|---|---|---|---|---|
| Yahoo chart, GLD | `https://query1.finance.yahoo.com/v8/finance/chart/GLD?range=10y&interval=1d` | **200** | 10y daily, 2,512 bars | Primary gold ETF series |
| Yahoo chart, IAU | `https://query1.finance.yahoo.com/v8/finance/chart/IAU?range=10y&interval=1d` | **200** | 10y daily | Cheaper expense ratio twin of GLD; cross-check |
| Yahoo chart, GC=F | `https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range=10y&interval=1d` | **200** | 10y daily (continuous futures) | Closest to spot gold; has roll artefacts |
| Yahoo chart, XAUUSD=X | `https://query1.finance.yahoo.com/v8/finance/chart/XAUUSD=X` | **404** | — | Does **not** work; don't build on it |
| Stooq CSV | `https://stooq.com/q/d/l/?s=xauusd&i=d` | **blocked** | — | Returns a JavaScript proof-of-work bot wall, not CSV |

Requires a `User-Agent` header. Without one Yahoo may refuse.

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

`DTWEXBGS` is published weekly and lagged; `DFII10` and `DGS10` daily. All are
forward-filled onto the trading calendar in `data.py`. **Forward-filling is a
no-lookahead-safe alignment for a decision made before the next release** — but the
publication lag is real (a FRED observation dated Friday may not have been visible
until the following week). This is a known, small optimism in the backtest. A stricter
build would shift each series by its actual release lag; that is listed as future work
rather than silently assumed away.

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
