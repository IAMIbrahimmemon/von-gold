# Strategy — what the bot does, and why

Written to be falsifiable. Every claim is either backed by a citation or by a number
this repo produced. Where a component did not earn its place, that is recorded as a
negative result rather than quietly removed.

## The core idea

Gold is a **trending, macro-driven, non-cash-flowing asset**. It has no earnings, so
there is no valuation anchor. What moves it is the direction of two things:

1. **Its own price momentum.** Time-series momentum is one of the few effects with
   out-of-sample evidence across asset classes, commodities included
   (Moskowitz, Ooi & Pedersen, *Time Series Momentum*, JFE 2012; Hurst, Ooi & Pedersen,
   *A Century of Evidence on Trend-Following Investing*, 2017).
2. **The real interest rate.** Gold pays no coupon, so the 10-year TIPS yield is its
   literal opportunity cost, and it is the most cited empirical driver of gold.

The bot therefore takes long positions in gold when trend and macro regime agree, and
sizes them by volatility. It is long-only and unlevered by default: a $10k account
holding at most $10k of GLD.

## The strategy in full

Decision made on day *t*'s close, executed at day *t+1*'s close.

```
direction  = mean( sign(close_t / close_{t-k} - 1) for k in {21, 63, 126, 252} )
size       = min( target_vol / realised_vol_21d , 1.0 )
gates      = [close > MA200]  ×  [real yield falling]  ×  [dollar falling]
target     = direction × size × gates        (clipped to [0, 1])
```

Then hysteresis: the target only moves when it has drifted more than `rebalance_band`
from the current holding, which cuts turnover without changing direction.

### Why each piece

**Multiple momentum lookbacks, averaged as signs.** Averaging *signs* rather than raw
returns keeps the signal bounded in [-1,1] and stops the 252-day lookback from
dominating purely because its returns are numerically larger. Averaging several windows
is the standard defence against a single window length being a lucky choice. The
ablation below shows the average is not worse than any single window — which is the
honest way to say it earned its place (it is more robust, not higher-returning).

**Volatility targeting.** Scaling exposure inversely with realised volatility is
documented to improve risk-adjusted returns and reduce drawdowns at moderate target
levels (Harvey et al., *The Impact of Volatility Targeting*, 2018). It is the single
most valuable component here: at a 10% target it cut max drawdown from -26.5% to -18.0%
while keeping Sharpe at 0.79.

**The 200-day moving average gate.** The pure trend brake. Cheapest crash protection
available and it is what makes the risk-adjusted numbers competitive with buy-and-hold.

**Real-yield filter.** Condition longs on the 10-year real yield *falling* over 3 months.
Economically motivated, and the ablation confirms it is roughly value-neutral on its own
(Sharpe 0.72 → 0.70 without it), so treat it as a modest tilt that reduces exposure in
hostile regimes rather than a source of return.

**Dollar filter — REMOVED as a default.** The broad dollar index strengthening is a
textbook gold headwind, so this filter was included in the first build. It **destroyed**
performance: Sharpe 0.73 → 0.44, because the dollar index and gold are correlated enough
that the filter mostly duplicated the price trend while lagging it. This is recorded as a
negative result. The code still supports it (`use_dollar_filter`), off by default.

### Costs charged

| Item | Assumption |
|---|---|
| ETF expense ratio | 40 bps/yr (GLD), accrued daily |
| Half-spread | 1.5 bps per side (GLD's real quoted spread is ~0.3 bps, so this is 5× conservative) |
| Slippage | 1.5 bps per side |
| Commission | 0 |
| **Round trip** | **6 bps** |

Cash earns nothing. A long-only strategy that is often out of the market would gain from
T-bill interest on the idle sleeve, so ignoring it is the conservative direction.

## Measured results (GLD, 2016-09-22 → 2026-09-21, 2,512 bars)

Read this table with the caveat that the same 10 years produced all of it.

| | CAGR | Vol | Sharpe | Max DD | Calmar | Time in market |
|---|---|---|---|---|---|---|
| Buy & hold GLD | 11.65% | 16.4% | 0.76 | **-26.5%** | 0.44 | 100% |
| Early version (all 4 gates) | 2.13% | 5.1% | 0.44 | -17.7% | 0.12 | 21% |
| Trend MA only, 10% vol target | 5.83% | 7.5% | **0.79** | -18.0% | 0.32 | 56% |
| Best of 108 swept configs | 6.02% | — | 0.93 | -13.9% | — | 56% |

**The honest summary: this strategy does not beat buy-and-hold on return. It beats it on
risk-adjusted return, and it roughly halves the drawdown — at the cost of giving up about
half the return.** Anyone who tells you a mechanical gold bot "always wins" is describing
a backtest they have not stress-tested.

### Does the edge survive scrutiny?

| Test | Result |
|---|---|
| Block-bootstrap 95% CI on daily Sharpe | [-0.015, 0.072] → **includes zero** |
| Probabilistic Sharpe (PSR vs 0) | 0.914 → below the 0.95 bar |
| Deflated Sharpe, 108 trials | 0.983 → the best swept config **does** clear the bar |
| Sharpe the best of 108 random designs would show by luck | 0.236 annualised (vs 0.93 achieved) |
| Folds beating buy-and-hold on Sharpe | 3 of 5 |
| Per-year: positive | 6 of 11 years |

The raw strategy's Sharpe is **not** statistically distinguishable from zero on this
sample. The deflated-Sharpe result is the interesting one: the trend+vol-targeting family
clears the multiple-testing bar, which is evidence for *trend following and volatility
targeting on gold generally*, not for my specific parameter choice. That distinction is
the whole point of computing it.

## What did NOT work

Recorded because a strategy document that only lists successes is a sales document.

1. **Dollar-index filter** — actively harmful (Sharpe 0.73 → 0.44).
2. **ATR trailing stop** — did nothing at all in the ablation. Because the 200-day MA
   gate and the momentum sign already exit on the same conditions, an additional stop
   only fired when the position was already flat. Kept in the code, off by default.
3. **The `momentum sign` as the primary driver** — the single largest drag. It was flat
   79% of the time, mostly because averaging four lookbacks yields exactly 0 whenever
   the windows disagree (which is often). It is more useful as a *risk reducer* than as
   a *return driver*.
4. **Real-yield filter as a return source** — value-neutral; a modest tilt only.

## Open questions / next steps

* **Release lags.** FRED series are forward-filled without accounting for publication
  lag, a small optimism. Shifting each by its real release delay would tighten it.
* **Sub-ETF choice.** IAU has a lower expense ratio for identical exposure. If the
  strategy cannot beat buy-and-hold on IAU too, something is wrong with the engine.
* **News features are forward-only.** No honest history exists for free (see DATA.md),
  so news is measured, not yet used.
* **Short side.** Gold had a brutal 2013-2015 bear market; this window (from 2016) is
  mostly up. The long-only restriction flatters the result. `allow_short=True` exists in
  the backtester and remains untested.

## The decision-model overlay

`von` (a 395M ModernBERT-Language non-autoregressive decision model, running locally via
MLX) is wired as an **overlay that can only reduce exposure**. It never sizes, never
initiates, and never emits prose.

The contract, so the test is fair:

* It sees measured facts only (returns, volatility percentiles, real yield levels).
  The state text never contains "bullish" or "buy" — selling it a narrative would just
  make it echo our own bias back.
* Question texts and option criteria are fixed once, not tuned per day.
* It is measured against its own mechanical twin in the same run. If it does not help
  after costs, the honest report is "do not enable it", not a re-tuned prompt.

Modes: `veto` (trade only when von agrees), `halve`, `prob` (scale by von's probability
weighted by its own confidence). See `docs/VON.md` for the results.

## Is it a good idea to use von for this?

Yes for a specific, narrow job, and no for a broader one.

**Good fit.** von is a calibrated classifier that answers in one forward pass at ~18ms
with zero LLM tokens, and it is deterministic and reproducible. For deciding *"given
these measured facts, is this one of the situations where my trend system historically
got chopped up?"*, that is exactly the right shape of tool. It is also free and local, so
2,250 decisions cost nothing.

**Bad fit.** von cannot read news, reason about a Fed statement, or produce a novel
thesis — it maps a text state to a labelled category. Anything generative needs a real
LLM. And a 395M model is not a forecaster: it has no more information than the numbers
you hand it. If the numbers do not contain the edge, von cannot supply it.

**So the design question is not "can von trade?"** — it is "does a second, independent
read of the same evidence add anything over a moving average?" That is an empirical
question with a measurable answer, which is what `docs/VON.md` reports.
