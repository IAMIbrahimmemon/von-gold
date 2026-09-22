# Strategy — what the bot does, and why

Written to be falsifiable. Every claim is either backed by a citation or by a number this
repo produced and can re-produce. Where a component did not earn its place, it is recorded
as a **negative result** rather than quietly deleted — the negative results are the most
valuable thing in this file, because they are what stop the strategy from becoming a curve
fit.

Reproduce everything here:

```bash
env -u PYTHONPATH .venv/bin/python scripts/long_history.py    # 58y + 22y headline
env -u PYTHONPATH .venv/bin/python scripts/validate_long.py   # macro-gate A/B + overfitting battery
env -u PYTHONPATH .venv/bin/python scripts/brake_test.py      # whipsaw brake, bootstrap-tested
env -u PYTHONPATH .venv/bin/python scripts/sweep_whipsaw.py   # parameter-family spread
env -u PYTHONPATH .venv/bin/python -m pytest tests -q         # 34 invariants
```

## The core idea

Gold is a **trending, macro-driven, non-cash-flowing asset**. It has no earnings, so there
is no valuation anchor. Two things have out-of-sample published support for moving it:

1. **Its own price momentum.** Time-series momentum is one of the few effects with
   out-of-sample evidence across asset classes, commodities included (Moskowitz, Ooi &
   Pedersen, *Time Series Momentum*, JFE 2012; Hurst, Ooi & Pedersen, *A Century of
   Evidence on Trend-Following Investing*, 2017).
2. **The real interest rate.** Gold pays no coupon, so the 10-year TIPS yield is its
   literal opportunity cost (Chicago Fed Letter 464; Erb & Harvey, *The Golden Dilemma*).

The bot goes long gold when trend is up and volatility is controllable, and **goes flat
otherwise**. Long-only, unlevered: a $10k account holds at most $10k of GLD. It cycles in
and out repeatedly — that is the mechanism, not a defect. The edge is in avoiding the
losing stretches, and over 58 years it spends ~50% of the time in the market.

## The strategy in full

A decision made on day *t*'s close is executed at day *t+1*'s close.

```
direction = mean( sign(close_t / close_{t-k} - 1) for k in {21, 63, 126, 252} )
size      = min( 0.10 / realised_vol_21d , 1.0 )
gate      = [close > MA200]
raw       = direction × size × gate            (clipped to [0, 1]: long-only, unlevered)
target    = hysteresis(raw, band=0.10, min_hold=10)
held      = target shifted forward one day
```

Three components, each defensible alone.

**Multi-horizon momentum (21/63/126/252d).** Averaging the *signs* rather than raw returns
keeps the signal bounded in [−1, 1] and stops the longest lookback dominating purely by
scale. Several horizons is a deliberate overfitting defence — any single window length is
an arbitrary choice. Honest note: averaging four signs yields exactly **0** whenever the
windows disagree, which is often (the strategy is flat most days because of this). It is
more useful as a *risk reducer* than a *return driver*.

**Trend gate (price > 200-day MA).** The single most important component. It is what
converts "hold gold" into "hold gold when it is working", and it is the cheapest crash
protection available.

**Volatility targeting (10% annualised).** Size = target vol ÷ realised vol, capped at
1.0. Harvey et al. (*The Impact of Volatility Targeting*) document improved risk-adjusted
returns at moderate targets. Gold's own vol is ~19%, so a 10% target is a deliberate
de-risking to roughly half of gold's natural risk.

**Hysteresis + whipsaw brake.** The target only moves when it has drifted more than
`rebalance_band` (0.10), and at most once per `min_hold_days` (10). This is a **cost**
mechanism, not an alpha claim — see the negative result below.

### Costs charged

| Item | Assumption |
|---|---|
| ETF expense ratio | 40 bps/yr (GLD), accrued daily |
| Half-spread | 1.5 bps per side (GLD's real quoted spread is ~0.3 bps, so this is ~5× conservative) |
| Slippage | 1.5 bps per side |
| Commission | 0 |
| **Round trip** | **6 bps** |

Cash earns nothing. A long-only strategy that is often flat would gain from T-bill interest
on the idle sleeve, so ignoring it is the conservative direction. Actual drag: **23 bps/yr**
at `min_hold_days=10`.

## The evidence

### Headline: 58 years of gold, not 10

A GLD-only backtest starts in 2004 and therefore only ever sees a mostly-bullish gold
market. That was a real limitation of the first version of this document, so the strategy
is now also tested on the **LBMA gold PM fix, 1968–2026 (14,687 daily observations)**,
which contains:

- the 1970s bull (gold ~$35 → ~$850),
- the **1980–2000 secular bear** (−72.5% peak-to-trough, two decades flat-to-down),
- the 2001–2011 bull,
- the **2013–2015 bear**.

| | CAGR | Vol | Sharpe | Max DD | Calmar | Time in market |
|---|---|---|---|---|---|---|
| **LBMA 58y — buy & hold** | 8.04% | 19.3% | 0.497 | **−72.5%** | 0.11 | 100% |
| **LBMA 58y — strategy** | 4.88% | 7.1% | **0.710** | **−21.8%** | 0.22 | 50.1% |
| GLD 22y — buy & hold | 10.15% | 18.3% | 0.621 | −46.5% | 0.22 | 100% |
| GLD 22y — strategy | 5.00% | 7.4% | 0.693 | −22.5% | 0.22 | 56.6% |

$10,000 → **$160,486** over 58 years with the strategy, vs **~$900,000** buy & hold. That
gap is the price of the drawdown protection, and it is worth stating in dollars, not
just percentages.

### The 58-year result is the load-bearing one

The strategy's drawdown across *two* secular bear markets is about **a third** of buy &
hold's. In the isolated bear windows:

| Window | Strategy | Buy & hold | Verdict |
|---|---|---|---|
| 1980–2000 bear | Sharpe −0.10, DD **−21.8%** | Sharpe −0.09, DD **−72.5%** | Sharpe level; **drawdown 3.3× better** |
| 2013–2015 bear | Sharpe −0.78, DD −5.4% | Sharpe −0.88, DD −38.8% | strategy wins |
| 1970s bull | Sharpe 1.68, DD −12.5% | Sharpe 1.32, DD −47.3% | strategy wins |
| 2001–2011 bull | Sharpe 0.79, DD −19.6% | Sharpe 0.90, DD −29.7% | buy & hold wins |

Across decades it beats buy & hold on Sharpe in **5 of 7** (4 of 6 excluding the partial
first window), and dominates on drawdown in **all 7**. The 1980–2000 bear is the honest
weak spot for Sharpe: the strategy roughly matches buy & hold there *while drawing down a
third as much*. Anyone who claims this wins on Sharpe everywhere has not run this table.

The decade detail also shows where the return goes: 1999–2008 returned +32% against buy &
hold's +190%, because the trend gate sat out much of the 2000s rally. That is the trade
being made, stated plainly.

### Honest limits — read this part

| Check | Result | Reading |
|---|---|---|
| Block-bootstrap 95% CI, strategy Sharpe | [0.458, 0.986] | excludes zero |
| Block-bootstrap 95% CI, buy & hold Sharpe | [0.245, 0.757] | also excludes zero |
| **Do those two CIs overlap?** | **Yes** | **the Sharpe *edge* is NOT statistically significant** |
| Probabilistic Sharpe (PSR vs 0) | 1.0000 | the strategy is distinguishable from nothing |
| Deflated Sharpe (12 declared trials) | 0.9978 | survives multiple-testing correction |
| Sharpe the best of 12 random designs shows by luck | 0.341 | vs 0.710 achieved |
| Minimum track record length | 5.2 years | at 95% confidence |
| Rolling 10-year windows beating buy & hold | 32/49 (65%) | consistent, with real losing stretches |
| Positive months | 33% | flat most of the time, by design |
| Positive years (58y) | 33/59 | worst year −10.6%, best +48.6% |

**What is reliably true:** the drawdown reduction is large, consistent across seven
decades, and survives every robustness check. Vol is roughly halved.

**What is not established:** that the strategy *beats buy-and-hold on Sharpe*. The point
estimate favours it (+0.213 over 58 years) but the confidence intervals overlap. Anyone
claiming a statistically significant Sharpe edge here is overreading the data.

**What is honestly given up:** roughly 3–5% of annual return. This is a **risk-reduction
strategy**, not a return-maximisation strategy.

**Nobody can build a strategy that "always wins".** Gold fell 72% once and stayed down for
twenty years. Any bot that claims to profit in every period is either fitted to the past or
lying. The realistic goal is to participate in gold's upside while cutting the pain enough
that a human can actually hold through it — and to be honest about the cost of that.

## Negative results — components tested and rejected

These matter more than the components that survived. Each was implemented, measured, and
switched off on the strength of the measurement.

### 1. The macro filters, tested correctly, HURT

Real yields and the dollar are gold's best-documented drivers, so the first build gated
longs on both. Then the relationship was tested with the correct *lag*.

The published relationships are **contemporaneous** — gold and real yields move together in
the same period. That is description, not tradeable edge; to trade it you need yesterday's
driver to predict today's return. It does not:

| Driver change at *t−1* vs gold return at *t* | Correlation |
|---|---|
| Real yield, lag 1d | −0.012 |
| Real yield, lag 5d | −0.003 |
| Real yield, lag 21d | −0.030 |
| Dollar index, lag 1d | +0.016 |
| Dollar index, lag 21d | +0.010 |

For comparison, the *contemporaneous* correlations are −0.248 (real yield) and −0.338
(dollar). The signal is real and it is entirely in the same period, where it cannot be
traded.

A lagged A/B on 22 years, gate using only information available through *t−1*:

| Configuration | Sharpe | Δ vs ungated |
|---|---|---|
| ungated (baseline, shipped config) | 0.693 | — |
| + real-yield gate | 0.639 | **−0.054** |
| + dollar gate | 0.412 | **−0.281** |
| + both gates | 0.409 | **−0.285** |

Both gates **remove** Sharpe. Off by default. `use_real_yield_filter` / `use_dollar_filter`
remain switchable so the claim can be re-tested. Erb & Harvey raised data mining as the
likely explanation for the headline gold/real-rate correlation, and the Chicago Fed paper
notes the relationship "does not show up in these data before 2001" — this result is
consistent with both.

### 2. The whipsaw brake is not an alpha tool

The 58-year run's worst decade (1989–1998, range-bound) lost money through repeated
in/out flips — 114 switches. A `min_hold_days` brake is the standard remedy and the
point-estimate sweep looked encouraging. It is not real:

- A paired block bootstrap on the Sharpe *difference* found **0 of 10** settings
  distinguishable from off, on either window. The sweep's scatter (0.729 → 0.725 → 0.629 →
  0.662 → 0.634 across hold lengths) is the signature of noise: a real effect varies
  smoothly with its own parameter.
- It does not fix the range-bound decade either.

Retained at `min_hold_days=10` for a **cost** reason: it cuts turnover ~55% and cost from
~39 to ~23 bps/yr. When two settings are statistically indistinguishable, take the cheaper
one — cost is deterministic, alpha estimates are not.

**And it is a genuine trade-off, not a free lunch.** The brake delays exits as well as
entries. With it on, the 1980–2000 bear's Sharpe slips from +0.04 to −0.10 because the
strategy is slower to step aside. That is exactly what the bootstrap said would happen: the
Sharpe difference is not distinguishable from zero in either direction, while turnover and
cost fall reliably. This is what "statistically indistinguishable" means in practice — it
is not a licence to also claim the better point estimate.

### 3. ATR stop-loss — no effect

Tested at several multiples; changed results within noise. Removed (`atr_stop_mult=0`). The
trend gate already does the exit work; an extra stop only fired when the position was
already flat.

### 4. A dollar-value filter — actively harmful

An early version required a minimum dollar move before trading. Sharpe 0.73 → 0.44.
Removed.

### 5. The von decision model — wired in, measurable, OFF

See `docs/VON.md`. Short version: von reads state competently (correlation +0.75 with
actual distance from the 200-day MA) but its probabilities are not usable as a signal, it
never once rated a long justified over 2,252 days, and out-of-sample its veto cost −1.03
Sharpe. It adds nothing the mechanical layer does not already contain, because
distance-from-MA is already an input. **Off by default, as a measured conclusion.**

### 6. Parameter sensitivity — the family is stable

A 40-configuration sweep across trend-MA (100/150/200/250) × momentum horizons × brake
settings spanned Sharpe 0.605–0.738, median 0.705, against buy & hold's 0.497. **100% of
the grid beat buy & hold.** A family where nearly every cell wins is evidence of a real
effect; a family where one cell wins and the rest lose is a curve fit.

## Why the drawdown control is the real product

For someone holding this with real money, the drawdown number decides whether the strategy
survives contact with their own nerve. A **−72.5%** drawdown takes two decades to recover
from and almost nobody holds through it. **−21.8%** is holdable. The strategy is therefore
best understood as *"own gold's upside, with a third of the pain, and accept a lower
average return for it."*

## What would falsify this

Stated in advance so it cannot be rationalised later:

1. **Vol targeting fails if** realised vol no longer predicts near-term vol.
2. **The trend component fails if** gold stops trending — a long pure mean-reversion
   regime. The 1989–1998 decade is the closest historical example: the strategy lost 9.1%
   over ten years there (vs buy & hold's −32.6%), the only clearly negative decade.
3. **The whole thing fails if** costs rise far above assumption. It pays ~23 bps/yr. A
   broker charging 25 bps per side would consume the edge.
4. **Live divergence from backtest is the real test.** The dry-run ledger exists to catch
   that; a persistent gap between the paper ledger and the backtest would mean the
   execution model is wrong.

## Open questions

* **Release lags.** FRED series are forward-filled without accounting for publication lag,
  a small optimism. Shifting each by its real release delay would tighten it. (Note: the
  macro filters are off anyway, so this currently affects only reporting.)
* **Sub-ETF choice.** IAU has a lower expense ratio for identical exposure.
* **News features are forward-only.** No honest free history exists (see `docs/DATA.md`),
  so news accumulates forward and is measured, not yet used as a signal.
* **Short side.** `allow_short=True` exists in the backtester and is untested. It would
  matter in a decade like 1980–2000.

## The decision-model overlay

`von` (a 395M ModernBERT-Language non-autoregressive decision model, running locally via
MLX) is wired as an **overlay that can only reduce exposure**. It never sizes, never
initiates, and never emits prose.

The contract, so the test is fair:

* It sees measured facts only (returns, volatility percentiles, real yield levels). The
  state text never contains "bullish" or "buy" — selling it a narrative would just make it
  echo our own bias back.
* Question texts and option criteria are fixed once, not tuned per day.
* It is measured against its own mechanical twin in the same run. If it does not help after
  costs, the honest report is "do not enable it", not a re-tuned prompt.

Modes: `veto` (trade only when von agrees), `halve`, `rank`, `prob` (scale by von's
probability weighted by its own confidence). Results in `docs/VON.md`.

One structural safeguard worth stating: the whipsaw brake is applied to the **mechanical**
signal only, *before* any overlay multiplier. Applying it afterwards would let the brake
delay a safety veto for up to `min_hold_days` — the bot would keep holding through a
signal that said get out. Caught by a test; a brake may smooth a signal, it must never
delay a kill switch.

## Is it a good idea to use von for this?

Yes for a specific, narrow job, and no for a broader one.

**Good fit.** von is a calibrated classifier that answers in one forward pass at ~18ms with
zero LLM tokens, deterministic and reproducible. For deciding *"given these measured facts,
is this one of the situations where my trend system historically got chopped up?"*, that is
the right shape of tool. It is free and local, so 2,250 decisions cost nothing.

**Bad fit.** von cannot read news, reason about a Fed statement, or produce a novel thesis
— it maps a text state to a labelled category. Anything generative needs a real LLM. A
395M model is not a forecaster: it has no more information than the numbers you hand it. If
the numbers do not contain the edge, von cannot supply it.

**Measured answer, not a guess:** it did not help, so it is off. See `docs/VON.md` for the
numbers. The design question was never "can von trade?" but "does a second, independent
read of the same evidence add anything over a moving average?" — and the answer here is no,
because distance-from-MA is already an input.

## Files

| File | Role |
|---|---|
| `src/vongold/strategy.py` | signals: momentum, vol, trend gate, hysteresis, brake |
| `src/vongold/backtest.py` | event loop, cost application, metrics; enforces the t→t+1 lag |
| `src/vongold/config.py` | all parameters, with the measurement behind each default |
| `src/vongold/data.py` | Yahoo/FRED/LBMA loaders, with silent-failure guards |
| `src/vongold/dryrun.py` | live paper-trading tick + state persistence |
| `src/vongold/news.py` | headline ingestion + local (no-LLM) classification |
| `docs/VON.md` | the von overlay experiment and its negative result |
| `docs/DATA.md` | verified data endpoints |
