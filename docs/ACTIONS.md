# The five-way action output — measured, and why it is off

The requested output was five actions: **long / short / open / close / hold**, re-decided
every 10 seconds. All three parts were built and measured. This records what the
measurements said, because two of the three did not survive contact with the data.

## 1. The five-way question: built, and von collapses to one label

`src/vongold/action.py` asks von for a single `choice` among five labels. The probe
(`scripts/probe_five_way.py` -> `scripts/von_batch.py`) ran it over **270 real trading days**
of GLD history, each day given the same state description the live path builds.

```
action distribution over 270 days
   reduce        270    100.0%
```

One distinct label out of five. Not "mostly reduce" — *always* reduce.

The comparison that makes this damning:

| | |
|---|---|
| Days price was **above** its 200-day MA | 236 |
| ...on which von said **reduce** | **236** |
| Days the mechanical strategy wanted exposure >5% | 218 |
| ...on which von said **reduce** | **218** |

So on 236 days when gold was in an uptrend — the exact situation where "reduce" is the wrong
answer — von said reduce. Its confidence was ~7% with a standard deviation of 0.011, i.e.
noise around a constant.

This is the same failure as the previous descriptive question set, which answered "range" on
2,252 of 2,252 days. The pattern is now established across two independent question sets:
**von does not discriminate when asked to classify a market state.** It is a 395M
non-autoregressive decision model; it matches a state against criteria, and when the criteria
are not separable it returns whichever label is most generic.

Consequence: the action output is wired in but **`ACTION_PRIOR_WEIGHT` defaults to 0.0**, so
a constant label cannot move exposure. A model that always says "reduce" would otherwise
quietly halve the account's exposure forever while appearing to work.

## 2. Re-deciding every 10 seconds: built, but the interval buys nothing

`scripts/live_loop.py` does re-decide on a timer. Two measurements set the default interval
at 60s rather than 10s:

* **A von decision takes ~4.75s** (subprocess transport — the path live use takes). At 10s
  the model is busy ~half of every cycle on a laptop.
* **von is deterministic.** The same state asked three times returns byte-identical answers
  (`range / 0.523 / 0.0527 / 0.5351` every time) — it does not sample. Therefore re-asking an
  *unchanged* state every 10 seconds provably cannot produce a different answer.

That second point is the important one: the thing that changes between seconds is the
**price**, and only the price. So the loop fingerprints the state and skips the model call
when the fingerprint is unchanged, which is a pure saving with zero behavioural difference.

```
[06:18:07] $398.35 fp=b7e1ffb2897ae6cb -> add_long (conv 2.15) ignored (entry disabled)
[06:18:08] $398.35 state unchanged (fp=b7e1ffb2897ae6cb) -> add_long (cached, no model call)
```

The signal is also **daily by construction**. The strategy was validated on daily closes; an
intraday re-decide is a different strategy with no backtest behind it. So the loop may only
**reduce** exposure by default. Opening intraday is behind `--allow-entry` and is untested —
using it means trading without evidence.

## 3. "short": measured, and it is a no-op

Shorting is supported in the engine (`allow_short=True`) and was tested over 58 years of LBMA
data:

| | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| long-only | 4.88% | 0.710 | −21.76% |
| long + short | 4.89% | **0.711** | −19.60% |

**Sharpe changes by +0.002.** The reason is structural: short exposure fires on only 1.58% of
days, because the trend gate `above_ma` is 0/1 and multiplies a *negative* direction to zero.
The gate that keeps the strategy out of downtrends also disables the short.

Offering a user "short" when it contributes 0.002 Sharpe would imply an edge that the
measurement does not support. It is left in the engine, documented, and not exposed.

## What this means for the bot

The mechanical strategy carries the account. von is off by default for decisions, and the
five-way output is inert until something shows it discriminates — which it currently does not.
The honest summary is that this model is good at *reading state* and useless at *choosing
actions*, and both of those were measured rather than assumed.

## Reproducing

```bash
# 271 problems over real history -> answers
env -u PYTHONPATH .venv/bin/python scripts/probe_five_way.py --out /tmp/five_way_problems.jsonl --n 240
env -u PYTHONPATH ~/vendor-mlx/von-mlx/.venv/bin/python scripts/von_batch.py \
  --model-dir ~/vendor-mlx/von-mlx/out/von-1.0-mlx/8bit \
  --problems /tmp/five_way_problems.jsonl --out /tmp/five_way_answers.jsonl

# the live loop (safe mode: may only reduce)
env -u PYTHONPATH .venv/bin/python scripts/live_loop.py --interval 60
```

## Gotcha: question criteria shapes are not interchangeable

`choice` criteria take a **dict** of label -> condition. `score` criteria take a **list** of
anchors. Sending a dict for a `score` question raises a pydantic `ValidationError` inside von
and the *entire* decision fails. This silently voided a full 270-day probe run before it was
caught. Both shapes are pinned by tests in `tests/test_invariants.py`.
