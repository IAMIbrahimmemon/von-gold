# The five-way action output — measured, and why it is off

The requested output was five actions: **long / short / open / close / hold**, re-decided
every 10 seconds. All three parts were built and measured. This records what the
measurements said, because two of the three did not survive contact with the data.

## 1. The five-way question: built, and the argmax collapses

`src/vongold/action.py` asks von for a single `choice` among five labels. The probe
(`scripts/probe_five_way.py` -> `scripts/von_batch.py`) ran it over **270 real trading days**
of GLD history, each day given the same state description the live path builds.

> The probe **imports** the question from `vongold.action` rather than restating it. An earlier
> version defined its own copy of the criteria with different wording, so it measured a
> *different question* from the one the bot asks — the documented result did not describe the
> deployed behaviour, and the winner differed between the two ("reduce" in the probe,
> "add_long" live) on the very same state. A test now pins them together.

Using the canonical wording:

```
argmax distribution over 270 days
   add_long      270    100.0%
```

One distinct winner out of five. Not "mostly add_long" — *always* add_long.

| option | mean prob | sd | min | max |
|---|---|---|---|---|
| `open_long` | 0.0302 | 0.0006 | 0.0288 | 0.0321 |
| **`add_long`** | **0.3746** | 0.0065 | 0.3540 | 0.3937 |
| `hold` | 0.3104 | 0.0089 | 0.2796 | 0.3396 |
| `reduce` | 0.1949 | 0.0059 | 0.1827 | 0.2174 |
| `close` | 0.0899 | 0.0024 | 0.0848 | 0.0992 |

Top probability averages 0.375 with a standard deviation of 0.0065 — the model never gets
confident, and never reorders. This is the same failure as the descriptive question set
("range" on 2,252 of 2,252 days), now reproduced across three independent question sets.

### The important nuance: the values DO carry information

The winner is static, but the distribution is **not**. The per-option probabilities track
market state and can be used as a readout:

| option | correlation with momentum | with above-200MA |
|---|---|---|
| `open_long` | **+0.585** | **+0.467** |
| `add_long` | −0.005 | −0.046 |
| `hold` | −0.373 | −0.232 |
| `reduce` | +0.399 | +0.312 |
| `close` | +0.270 | +0.101 |

`open_long` at r=+0.585 against momentum is the strongest state relationship measured anywhere
in this project. von is genuinely reading the market through these probabilities — it simply
never lets any option cross the argmax threshold, because one label carries a baseline prior
the others cannot overcome.

**What does not work:** none of the probabilities predict the next day's return (|r| ≤ 0.117,
n=270). So this is a description of the present, not a forecast of the future — the same
conclusion as the macro gates.

Consequence: the action output is wired in, and **`ACTION_PRIOR_WEIGHT` defaults to 0.0**, so
a static winner cannot move exposure. The dashboard shows the full distribution and says
plainly that the ranking is static.

## 2. Re-deciding every 10 seconds: built, and the interval buys nothing extra

`scripts/live_loop.py` re-decides on a timer, default **10 seconds** as requested. Two
measurements matter here:

* **A von decision takes ~3-8s** (subprocess transport, the path live use takes). At a 10s
  interval the model is busy most of each cycle on a laptop.
* **von is deterministic.** The same state asked three times returns byte-identical answers
  (`range / 0.523 / 0.0527 / 0.5351` every time) — it does not sample. Therefore re-asking an
  *unchanged* state cannot produce a different answer, and the feed will repeat the same values
  until the **price** moves. Only the price changes the state.

So by default the loop always calls the model (so the feed shows a real latency and cadence),
and `--cache-unchanged` makes it skip provably-identical calls, which halves CPU with no change
in output. Rows are marked `cached` when skipped.

The signal is also **daily by construction**. The strategy was validated on daily closes; an
intraday re-decide is a different strategy with no backtest behind it. So the loop may only
**reduce** exposure by default. Opening intraday is behind `--allow-entry` and is untested.

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
