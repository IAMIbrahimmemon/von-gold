#!/usr/bin/env python
"""Long-history robustness: does the strategy's edge survive 58 years?

The 2016-2026 GLD window is a mostly-bullish decade, which flatters a long-only
strategy. This test removes that objection by running on:

  1. **LBMA gold PM fix, 1968-2026 (58 years)** -- includes the 1970s bull, the
     1980-2000 secular bear (gold fell ~60% peak to trough and stayed down for two
     decades), the 2001-2011 bull, and the 2013-2015 bear. If the strategy only works
     in bull decades, this will show it.
  2. **GLD, 2004-2026** -- the actual tradable instrument, full history including 2008.

Macro filters (real yield, dollar) are DISABLED here on purpose: TIPS data only starts
2003 and the broad dollar index 2006, so they cannot exist for 58 years. That is also
the configuration the ablation already showed is fine, so this is a clean test of the
core trend + volatility-targeting mechanism.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vongold.backtest import BacktestResult, buy_and_hold, run_backtest
from vongold.config import CostModel, StrategyParams
from vongold.data import load_lbma_gold, load_local_parquet

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

LBMA = Path.home() / ".hermes" / "cache" / "scratch" / "goldresearch" / "lbma_gold_pm.json"
GLD_YF = Path("data/processed/gld_yfinance_max.parquet")


def load_lbma(path: Path = LBMA) -> pd.DataFrame:
    """LBMA gold price, USD per troy ounce, daily. v[0] is USD (v[1]/v[2] are GBP/EUR)."""
    raw = json.loads(path.read_text())
    rows = []
    for rec in raw:
        v = rec.get("v") or []
        if not v or v[0] is None:
            continue
        rows.append({"date": pd.Timestamp(rec["d"]), "close": float(v[0])})
    df = pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()
    df.index = df.index.astype("datetime64[ms]")
    # LBMA quotes once daily: synthesise a flat intrabar envelope so the ATR-based code
    # paths (unused here) do not see NaNs.
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    df["volume"] = 0.0
    df.index.name = "date"
    return df


def load_gld_full(path: Path = GLD_YF) -> pd.DataFrame:
    h = pd.read_parquet(path)
    h.columns = [str(c).lower().replace(" ", "_") for c in h.columns]
    df = h.rename(columns={"adj_close": "close_adj"})[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index = df.index.astype("datetime64[ms]")
    df.index.name = "date"
    return df


def per_period(res: BacktestResult, freq: str) -> pd.DataFrame:
    r = res.returns
    rows = []
    periods = list(r.groupby(pd.Grouper(freq=freq)))
    for idx, (period, grp) in enumerate(periods):
        if len(grp) < 40:
            continue
        eq = (1.0 + grp).cumprod()
        dd = float((eq / eq.cummax() - 1.0).min())
        vol = grp.std() * np.sqrt(252)
        # Label a decade by the window it actually covers (period end is the boundary
        # date, so a "10YE" bucket ending 1978-12-31 spans 1968-1977).
        if freq.endswith("10YE"):
            label = f"{period.year - 9}-{period.year}"
        elif freq.startswith("Y"):
            label = str(period.year)
        else:
            label = str(period.date())
        rows.append({
            "period": str(period.date()),
            "label": label,
            "ret": float(eq.iloc[-1] - 1.0),
            "vol": float(vol),
            "sharpe": float(grp.mean() * 252 / vol) if vol > 0 else 0.0,
            "max_dd": dd,
        })
    return pd.DataFrame(rows)


def report(name: str, df: pd.DataFrame, params: StrategyParams, cost: CostModel) -> dict:
    res = run_backtest(df, params=params, cost=cost)
    bh = buy_and_hold(df, cost=cost)
    m, b = res.metrics, bh.metrics
    print(f"\n{'=' * 100}\n{name}   ({len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()})\n{'=' * 100}")
    print(f"  buy & hold  : CAGR {b['cagr']:7.2%}  vol {b['vol_annual']:6.2%}  "
          f"Sharpe {b['sharpe']:5.2f}  maxDD {b['max_drawdown']:8.2%}  Calmar {b['calmar']:5.2f}")
    print(f"  strategy    : CAGR {m['cagr']:7.2%}  vol {m['vol_annual']:6.2%}  "
          f"Sharpe {m['sharpe']:5.2f}  maxDD {m['max_drawdown']:8.2%}  Calmar {m['calmar']:5.2f}")
    print(f"  in-market {m['pct_time_in_market']:.1%}  switches {m['exposure_switches']}  "
          f"cost {m['cost_drag_annual_bps']:.1f}bps/yr  "
          f"final ${m['final_equity']:,.0f} from ${m['initial_capital']:,.0f}")
    print(f"  vs buy&hold: Sharpe {m['sharpe'] - b['sharpe']:+.2f}  "
          f"CAGR {m['cagr'] - b['cagr']:+.2%}  maxDD {m['max_drawdown'] - b['max_drawdown']:+.2%}")

    decade = per_period(res, "10YE")
    dec_bh = per_period(bh, "10YE")
    if not decade.empty:
        print("\n  by decade (strategy vs buy&hold):")
        for _, row in decade.iterrows():
            b_row = dec_bh[dec_bh["period"] == row["period"]]
            bh_s = (f"   bh: ret {b_row['ret'].iloc[0]:+7.1%} sharpe {b_row['sharpe'].iloc[0]:5.2f} "
                    f"maxDD {b_row['max_dd'].iloc[0]:7.1%}") if len(b_row) else ""
            print(f"    {row['label']:>6s}  strat: ret {row['ret']:+7.1%}  vol {row['vol']:5.1%}  "
                  f"sharpe {row['sharpe']:5.2f}  maxDD {row['max_dd']:7.1%}{bh_s}")

    yearly = per_period(res, "YE")
    pos = int((yearly["ret"] > 0).sum())
    print(f"\n  positive years: {pos}/{len(yearly)}   "
          f"worst year {yearly['ret'].min():+.1%}   best {yearly['ret'].max():+.1%}")

    return {
        "name": name, "bars": len(df),
        "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
        "strat_sharpe": m["sharpe"], "bh_sharpe": b["sharpe"],
        "strat_cagr": m["cagr"], "bh_cagr": b["cagr"],
        "strat_maxdd": m["max_drawdown"], "bh_maxdd": b["max_drawdown"],
        "strat_calmar": m["calmar"], "bh_calmar": b["calmar"],
        "in_market": m["pct_time_in_market"],
        "positive_years": pos, "total_years": len(yearly),
    }


def main() -> int:
    cost = CostModel()
    # Use the SHIPPED DEFAULTS exactly. These numbers are what the documentation quotes,
    # so the script must not carry its own parameter overrides -- otherwise the docs and
    # the deliverable drift apart. (Earlier versions of this script passed band=0.15 while
    # the shipped default was 0.10, which made the quoted figures unreproducible.)
    p = StrategyParams()

    out = []
    gld = load_local_parquet(GLD_YF)
    out.append(report("GLD full history (tradable ETF)", gld, p, cost))

    lbma = load_lbma()
    # sanity-check the series against a known print before trusting it
    peak = lbma["close"].loc["2011-08-01":"2011-10-01"].max()
    print(f"\nLBMA sanity check -- 2011 autumn peak: ${peak:,.2f} (historically ~$1,900)")
    out.append(report("LBMA gold PM fix (58 years, spot)", lbma, p, cost))

    # The bear-market decades isolated, which is the whole point of the exercise.
    print(f"\n{'=' * 100}\nSECULAR BEAR WINDOWS ISOLATED\n{'=' * 100}")
    for label, lo, hi in (
        ("1980-2000 bear (gold fell ~60%)", "1980-01-01", "2000-12-31"),
        ("2013-2015 bear", "2013-01-01", "2015-12-31"),
        ("1970s bull", "1971-01-01", "1979-12-31"),
        ("2001-2011 bull", "2001-01-01", "2011-12-31"),
    ):
        sl = lbma.loc[lo:hi]
        if len(sl) < 200:
            continue
        res = run_backtest(sl, params=p, cost=cost)
        bh = buy_and_hold(sl, cost=cost)
        m, b = res.metrics, bh.metrics
        verdict = "BEATS" if m["sharpe"] > b["sharpe"] else "trails"
        print(f"  {label:34s} strat Sharpe {m['sharpe']:5.2f} (ret {m['cagr']:+7.1%}, "
              f"DD {m['max_drawdown']:7.1%})  |  bh Sharpe {b['sharpe']:5.2f} "
              f"(ret {b['cagr']:+7.1%}, DD {b['max_drawdown']:7.1%})  -> {verdict}")

    pd.DataFrame(out).to_csv("data/processed/long_history.csv", index=False)
    print("\nwritten: data/processed/long_history.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
