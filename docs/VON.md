# The von decision-model overlay — results

**Bottom line: von is wired in, fully testable, and OFF by default. On 9 years of real
gold data it did not improve the strategy. The reason is specific and interesting, and
it points at where von *would* help.**

Everything below comes from `scripts/eval_von_overlay.py` and `scripts/calib_eval.py`
running against 2,252 real decisions and 300 calibration probes. No numbers here are
estimated.

## What was run

Every trading day from 2017-10-04 to 2026-09-21 (2,252 days), von was given a text state
containing only measured facts — returns over 5/20/60/120/250 days, annualised realised
volatility and its percentile, position in the 252-day range, distance from the 200-day
moving average, and the 10-year real yield, dollar index, CPI and fed funds rate with
their 3-month changes — and asked four questions:

| Question | Type | Output |
|---|---|---|
| Classify the prevailing direction of the price series | `choice` | up_trend / down_trend / range |
| Is holding a long position justified today? | `noul` | probability |
| Is near-term downside risk elevated? | `noul` | probability |
| Rate the strength of the evidence | `score` | 0-4 |

Cost: **zero LLM tokens**, ~450ms per day, 17 minutes for the whole history, entirely
local.

## Finding 1: von's absolute calibration is broken for this domain

| | Measured |
|---|---|
| Regime answers | **"range" 2,252 times out of 2,252 (100%)** |
| Distinct regime values issued | 1 of 3 |
| P(long justified) | mean 0.070, **max 0.266 — never once above 0.5** |
| P(risk elevated) | mean 0.614 — above 0.5 on 96.9% of days |
| Regime confidence | mean 0.549, below 0.60 on 98.8% of days |
| Internal agreement | 3 heads disagreed on **100%** of days |

von never rated a long gold position justified, and never even reached a coin-flip. It
also consistently said "range" while gold rose 12.8% annualised. So the *absolute
probability scale* is compressed and offset — its 0.5 threshold carries no meaning here.
That alone disqualifies any mode that thresholds von's probability.

## Finding 2: but von's *ranking* does carry signal

| Horizon | corr(P(long justified), forward gold return) |
|---|---|
| 1 day | +0.028 |
| 5 days | +0.087 |
| 21 days | **+0.162** |

Monotone, and growing with horizon. von's answers are not noise — they are positively
rank-correlated with what gold does next. The level is meaningless; the ordering is not.

## Finding 3: von can actually read the state (this is the important one)

300 descriptive probes, ground truth computed mechanically from the same numbers:

| Probe | von | Baseline | Verdict |
|---|---|---|---|
| "Is price above its 200-session average?" | **0.783** | 0.753 (always "yes") | Beats baseline |
| "Is the series rising, falling or sideways?" | 0.390 | 0.470 (always "rising") | **Does not beat baseline** |

The direction question is a clear failure — von said "sideways" 298 of 300 times. But the
moving-average question is far more revealing:

```
corr(von P(above MA200),  actual signed distance from MA200) = +0.7516
median |distance from MA| when von is RIGHT : 7.90%
median |distance from MA| when von is WRONG : 2.46%
```

**+0.75 correlation with a continuous market quantity is not a coin flip.** von is
accurately reading how far price sits from its own moving average, and it only errs when
price is within ~2.5% of that average — genuinely ambiguous territory. The model is
competent at this task; its *thresholded label* is what fails, not its perception.

## Finding 4: the competence is redundant, and that is why the overlay fails

This is the whole explanation. von's single strongest measured skill is estimating the
position relative to the 200-day moving average (+0.75). That quantity **is already an
input to the mechanical strategy** — `above_ma` is one of its gates, and the momentum
sign is a multi-horizon version of the same idea.

So von is doing a good job of reading a number we already compute. A second, noisier
view of information already in the signal cannot add return; it can only add variance.
That is exactly what the A/B shows.

## The A/B, measured

Base: trend-MA + 10% vol target, no macro filters. Costs charged at 6bps round trip.

| Variant | CAGR | Sharpe | Max DD | Δ Sharpe | Time in mkt |
|---|---|---|---|---|---|
| Buy & hold | 11.65% | 0.755 | -26.5% | — | 100% |
| **Mechanical (no von)** | 5.67% | **0.760** | -18.7% | — | 55.1% |
| von `veto` | 0.21% | 0.161 | -4.1% | **-0.599** | 1.9% |
| von `halve` | 3.64% | 0.877 | -8.6% | +0.117 | 59.6% |
| von `prob` | 0.97% | 0.608 | -4.1% | -0.152 | 74.6% |
| von `rank` | 3.95% | 0.732 | -9.1% | -0.028 | 67.6% |

### The `halve` result is an artifact, not a win

`halve` beats the mechanical strategy by +0.117 Sharpe and halves drawdown. It looks like
a success. It is not, for two reasons that a less careful evaluation would have missed:

1. **The veto rate was identical — 0.8965 — at every confidence setting** (0.0, 0.60,
   0.80). If von were genuinely discriminating, tightening the confidence threshold would
   change how often it acts. It does not, because von's confidence sat below 0.60 on 98.8%
   of days, so the threshold was never binding. The "result" is produced by a single
   constant decision applied 90% of the time, not by von reading each day.
2. **Out of sample it collapses.** Split at the midpoint:

| Mode | First half Δ Sharpe | Second half Δ Sharpe |
|---|---|---|
| `halve` | +0.10 | **-1.03** (veto rate 100%) |
| `veto` | -0.18 | **-1.13** (veto rate 100%) |
| `rank` | -0.13 | +0.02 |

In the first half von's constant de-risking happened to help. In the second half it
vetoed 100% of days and the strategy went completely flat — Sharpe 0.00, CAGR 0.00%.

**A constant decision cannot be a strategy.** `halve`'s positive full-period number is
one half of luck masking the other half of failure. `rank` — the only mode that consumes
von's *relative* read without thresholding — lands within noise of the mechanical
baseline (+0.02 out of sample, -0.03 full period).

## Conclusion

**Do not enable the von overlay on the mechanical gold signal.** The measured reasons:

1. Its absolute probabilities are unusable (never crossed 0.5 in 2,252 days).
2. Modes that threshold those probabilities are therefore constant decisions, and a
   constant decision is not a strategy — the apparent win inverts out of sample.
3. Its one genuine competence (+0.75 correlation with distance from the 200-day MA) is
   already an input to the mechanical strategy, so it is redundant rather than additive.

This is a negative result, and it is reported as one. It was not "fixed" by re-tuning
thresholds or question wording until something looked good — that would have been
overfitting dressed up as engineering.

## Where von would genuinely help

The finding above says something useful: **von adds value where it reads information the
mechanical strategy cannot compute at all.** Distance from a moving average is arithmetic;
the mechanical layer already has it. What the mechanical layer has *no access to* is text:

* **News and headlines.** Wire von in as an NLI/entailment classifier over headlines —
  does this story entail a real-yield move, dollar strength, or safe-haven demand? This is
  a genuine information channel, not a re-derivation of arithmetic.
* **Official statements.** Fed minutes and press releases: von's `choice`/`noul` heads
  over sentence-level text is exactly the shape for extracting a stance.
* **Event gating.** "Is there an unscheduled macro event in the last 12 hours?" is a
  classification question the price series cannot answer by construction.

**The blocker for all three is the same, and it is not von:** there is no free, honest
historical news archive (see `docs/DATA.md`). A backtest that splices today's headlines
into 2021 would be committing lookahead. So this channel has to be *accumulated forward*
before it can be tested — which is precisely what the running dry run is set up to do.

## Reproduce

```bash
# 1. build states for every day (fast, no model)
env -u PYTHONPATH .venv/bin/python scripts/export_problems.py

# 2. run von over them -- MUST use the von-mlx venv (MLX streams are thread-local;
#    this runs in the main thread in-process, ~17 min for 2,252 days)
cd "$VON_MLX_DIR" && env -u PYTHONPATH .venv/bin/python \
  ~/Documents/GitHub/von-gold/scripts/von_batch.py \
  --model-dir out/von-1.0-mlx/8bit \
  --problems ~/Documents/GitHub/von-gold/data/processed/von_problems.jsonl \
  --out ~/Documents/GitHub/von-gold/data/processed/von_answers.jsonl

# 3. the A/B and falsification tests
env -u PYTHONPATH .venv/bin/python scripts/eval_von_overlay.py

# 4. the calibration probes
env -u PYTHONPATH .venv/bin/python scripts/calib_export.py 300
#    ... run von_batch.py on calib_problems.jsonl -> calib_answers.jsonl ...
env -u PYTHONPATH .venv/bin/python scripts/calib_eval.py
```
