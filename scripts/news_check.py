"""Fetch the news feeds and measure whether the features predict gold returns."""
import sys, pandas as pd
from vongold.data import build_dataset
from vongold.news import DEFAULT_FEEDS, fetch_feed, news_daily_features, news_impact_report, lexicon_hash

print("lexicon hash:", lexicon_hash())
print("\n--- feed reachability ---")
for url in DEFAULT_FEEDS:
    items = fetch_feed(url)
    print(f"  {len(items):4d} items  {url}")

print("\n--- building daily news features ---")
news = news_daily_features(refresh=True)
print(f"  news days: {len(news)}   date range: {news.index.min() if len(news) else 'n/a'} -> {news.index.max() if len(news) else 'n/a'}")
if len(news):
    print(news.tail(10).to_string())

px = build_dataset("GLD", rng="10y")
rep = news_impact_report(news, px)
print("\n--- does news predict gold returns? (in-sample correlation) ---")
if rep.empty:
    print("  no overlap between news dates and price dates")
else:
    rep = rep.assign(abs_corr=lambda d: d["corr"].abs()).sort_values("abs_corr", ascending=False)
    print(rep.head(25).to_string(index=False))
