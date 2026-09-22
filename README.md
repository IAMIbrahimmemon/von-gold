# von-gold

A **dry-run (paper) trading system for gold** that runs on your own machine, driven by
your local **von** decision model (395M MLX, ~18ms, **zero LLM tokens**) — with the
honest finding that von does not improve it, and a mechanical strategy that does what a
professional trend-following program would do.

Over **58 years of gold** (LBMA, 1968-2026, 14,687 daily bars — includes the 1980-2000
secular bear and the 2013-2015 bear):

```
                          CAGR    Vol    Sharpe   max DD    time in market
Buy & hold gold           8.04%   19.3%   0.497    -72.5%       100%
von-gold mechanical       4.88%    7.1%   0.710    -21.8%        50%
```

**Read that honestly:** it does not beat buy-and-hold on return — it gives up ~3-5%/yr.
It beats it on **drawdown**, cutting -72.5% to -21.8%, and its Sharpe point estimate is
higher (0.710 vs 0.497) though the bootstrap confidence intervals overlap, so that
advantage is **not statistically significant**. The drawdown reduction is the large,
reliable effect.

**Nobody can build a strategy that always wins.** Gold fell 72% once and stayed down for
twenty years. Any pitch promising a system that "always comes out as a win" is describing
a backtest nobody stress-tested — see `docs/STRATEGY.md` for the tests this one passes and
fails, including the negative results.

## The one-paragraph version

Gold trends and it has no cash flow, so what moves it is (a) its own price momentum and
(b) the 10-year real yield, which is its literal opportunity cost. The bot holds gold when
its own trend is up (price above the 200-day average), sizes the position by volatility
targeting 10% annualised, and sits flat otherwise — going in and out repeatedly is the
mechanism, not a defect. Decision made at day *t*'s close, filled at day *t+1*'s close,
with 6bps round-trip costs charged. Long-only, unlevered. Runs once per session on the
Mac, writes its state to a file, and a Vercel page shows you how it's doing with an
on/off switch.

Note what is *not* in there: the macro filters. Real yields and the dollar are gold's
best-documented drivers, so they were built in first — then tested properly and found to
**hurt** on a lagged basis (-0.054 and -0.281 Sharpe). The documented relationship is
contemporaneous, which is description, not tradeable edge. They are off by default.

## Quick start

```bash
cd ~/Documents/GitHub/von-gold

# 1. backtest it
env -u PYTHONPATH .venv/bin/python scripts/run_experiments.py

# 2. is the local model reachable?
env -u PYTHONPATH .venv/bin/python -c "import sys;sys.path.insert(0,'src');from vongold.von_client import main;main()"

# 3. one dry-run session, switch ON, asking von
env -u PYTHONPATH .venv/bin/python -c "import sys;sys.path.insert(0,'src');from vongold.dryrun import main;sys.argv=['x','--enable','--force'];main()"

# 4. stop it
env -u PYTHONPATH .venv/bin/python -c "import sys;sys.path.insert(0,'src');from vongold.dryrun import main;sys.argv=['x','--disable','--force'];main()"

# tests
env -u PYTHONPATH .venv/bin/python -m pytest tests -q      # 34 passing

# the long-history harness (58 years) -- this is the one that matters
env -u PYTHONPATH .venv/bin/python scripts/long_history.py
```

**Always `env -u PYTHONPATH`.** A Hermes/agent session exports a `PYTHONPATH` that points
at its own venv and silently shadows this one's packages — you get `ModuleNotFoundError`
for something you can see is installed.

## Watching it: profit / loss windows

The dashboard shows P/L against your original $10,000 for **all time, past 24 hours, past
6 hours, and past hour**.

These are **reconstructed from the append-only ledger**, not sampled from a saved equity
curve. That distinction matters: the curve is only recorded when the bot ticks, so a
sampled version would report "past hour" as flat purely because no tick landed in that
hour. Rebuilding from the ledger's fill events keeps every window exact.

Each window is labelled with what it's actually telling you:

| Label | Meaning |
|---|---|
| `against the $10,000 starting stake` | all-time P/L, the number you care about |
| `includes a trade` | a fill landed in this window, so the move is real but not purely mark-to-market |
| `intraday marks` | window priced with GLD 5-minute bars (position held at both ends) |
| `flat — no position` | no position was open at any point in the window, so $0 is exact |

**Read the sub-day windows with this in mind:** the bot is deliberately flat roughly half
the time, and GLD only trades 09:30–16:00 ET. A "past hour" of $0 usually means *no
exposure*, not *no data* — the label says which. Sub-day P/L is only informative while a
position is open.

One trap that had to be avoided explicitly: a window that is **flat at both ends is not
the same as a window with no P/L**. A position opened and closed inside six hours leaves
the account flat at both ends while realising a gain in the middle. The first version of
this code reported the right dollar figure while labelling it "no loss possible" — a
contradiction the tests now pin against.

## Architecture

Three environments, deliberately kept apart:

```
von-gold/.venv        pandas, numpy, the backtester, the strategy, the CLI
von-mlx/.venv         MLX + the von weights   (already on your machine)
Vercel                the dashboard, reads status.json from the repo, writes control.json
```

The trading logic and the model live in **separate interpreters**, talking over one
subprocess per trading session (~450ms, once a day). This is on purpose: installing MLX
into the strategy venv, or pandas into the model venv, would couple two environments that
have no reason to know about each other. There is also no server to supervise — a
subprocess has no lifecycle and cannot be left accidentally down.

| File | Role |
|---|---|
| `src/vongold/data.py` | Yahoo/FRED/LBMA loaders, with silent-failure guards |
| `src/vongold/strategy.py` | Trend + vol-targeting signal, macro gates, rebalance band |
| `src/vongold/backtest.py` | Event-driven backtester with real costs, metrics |
| `src/vongold/experiments.py` | Ablation, sweeps, walk-forward, Deflated Sharpe, bootstrap |
| `src/vongold/von_state.py` | Turns a trading day's numbers into a von decision problem |
| `src/vongold/von_client.py` | Talks to local von (in-process or subprocess), never raises |
| `src/vongold/von_overlay.py` | The overlay contract + A/B harness |
| `src/vongold/news.py` | Free RSS feeds, deterministic lexical classification, impact test |
| `src/vongold/dryrun.py` | The paper-trading loop, kill switch, fill simulation |
| `src/vongold/state_store.py` | The on/off switch, the ledger, the status blob |
| `src/vongold/pl.py` | Reconstructs P/L for every window from the ledger |
| `web/` | Vercel dashboard + control endpoint |

### Safety properties (all tested)

* **No broker, no keys, no orders.** Paper only. Fills simulated at the close with the
  same cost model the backtester uses.
* **Fails closed.** Unreadable control file → disabled. Model failure → the answer is
  treated as "no view", never as a signal. Price feed failure → flat.
* **Drawdown kill-switch.** Checked before any trade; trips the switch off permanently
  until you re-enable it deliberately.
* **von can only ever REDUCE exposure**, never create it. A 395M classifier has no
  business initiating risk.
* **No lookahead**, verified two ways: exposure must equal the lagged target, and
  truncating the future must not change past decisions.
* **Idempotent per session** — running the tick twice does not double-trade.

## The von verdict — the part you actually asked about

You asked whether using von for this is a good idea. Measured answer, from 2,252 real
decisions over 9 years and 300 calibration probes:

**von is competent at reading market state, and that competence is redundant here.**

| Finding | Number |
|---|---|
| Regime labels issued | **"range" 2,252/2,252 times (100%)** |
| P(long justified) | mean 0.070, **max 0.266 — never crossed 0.5** |
| Its 3 heads agreed with each other | **0% of days** |
| corr(P(long justified), forward 21-day return) | **+0.162** (monotone in horizon) |
| corr(P(above 200-day MA), actual distance from MA) | **+0.752** |
| "Is price above its 200-session average?" accuracy | 78.3% vs 75.3% baseline |
| "Rising/falling/sideways?" accuracy | 39.0% — **worse than the 47% baseline** |

So: its absolute probabilities are unusable, but its *ranking* carries a weak real signal,
and its perception of *where price sits relative to its moving average* is genuinely good
(+0.75 correlation, erring only when price is within ~2.5% of the average).

**That last one is why the overlay fails.** Distance from the 200-day MA is already an
input to the mechanical strategy. von is doing a good job reading a number we already
compute, so it can only add variance, not information.

The A/B, with the trap called out:

| Mode | Sharpe | Δ vs mechanical | Out-of-sample Δ |
|---|---|---|---|
| mechanical (no von) | 0.760 | — | — |
| `veto` | 0.161 | **-0.60** | -1.13 |
| `halve` | 0.877 | +0.12 | **-1.03** |
| `prob` | 0.608 | -0.15 | -1.03 |
| `rank` | 0.732 | -0.03 | +0.02 |

`halve` looks like a win at +0.12 Sharpe — **it is not.** The veto rate was identical
(0.8965) at every confidence threshold, meaning the result comes from one constant
decision applied 90% of the time, not from von reading each day. Out of sample it vetoed
100% of days and the strategy went flat. **A constant decision is not a strategy.**

`rank` — the only mode that uses von's ordering without thresholding — lands within noise.

**Conclusion: the overlay is OFF by default, and that is a measured result, not a
placeholder.** Full write-up in `docs/VON.md`.

### Where von would genuinely help

von adds value where it reads information the mechanical layer *cannot compute at all* —
text. News headlines, Fed minutes, "is there an unscheduled macro event in the last 12
hours?" are classification problems the price series cannot answer by construction, and
von's `choice`/`noul`/`score` heads are exactly the right shape for them.

**The blocker is not von, it's data:** there is no free, honest *historical* news archive
(DATA.md). A backtest that splices today's headlines into 2021 is committing lookahead. So
this channel must be accumulated forward — which is what the running dry run does.

## What did not work (kept honest)

1. **Both macro filters, tested with the correct lag** — real-yield gate **-0.054**
   Sharpe, dollar gate **-0.281**. The published gold/real-rate relationship is
   contemporaneous (corr -0.248 same-day) and carries nothing at a 1-day lag (-0.012).
   Description, not edge.
2. **The whipsaw brake as an alpha tool** — a paired block bootstrap found **0 of 10**
   hold settings distinguishable from off. The sweep's scatter across hold lengths (0.729
   / 0.725 / 0.629 / 0.662 / 0.634) is the signature of noise. Retained at 10 days purely
   because it cuts cost ~55% (39 → 23 bps/yr) between two statistically identical
   settings.
3. **ATR trailing stop** — did nothing; the 200-day gate already exits on those days.
4. **A dollar-value filter** — Sharpe 0.73 → **0.44**.
5. **Momentum sign as the main driver** — flat most of the time; averaging four lookbacks
   gives exactly 0 whenever the windows disagree, which is often. More useful as a risk
   reducer than a return driver.

## Deployment (not yet done — needs your go-ahead)

Two pieces, both of which leave this machine:

1. **Dashboard on Vercel.** Push the repo, then `cd web && vercel --prod`. The on/off
   switch needs a `GITHUB_TOKEN` env var (fine-grained PAT, *Contents: read and write* on
   this repo only) so the page can write `runtime/control.json`. Without it the dashboard
   is read-only and you flip the switch locally.
2. **Scheduled tick on the Mac.** `scripts/local.vongold.dryrun.plist.template` →
   substitute the repo path → `~/Library/LaunchAgents/`. Weekdays 13:15 local, after the
   US close. A Hermes session cannot arm a LaunchAgent (`launchctl bootstrap` is blocked
   for the agent) — you run one command:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.vongold.dryrun.plist
```

## Verification status

**Verified (actually executed):**

* **43/43** invariant tests pass, including the two no-lookahead checks, the kill-switch
  ordering regression, the stale-cache guard, the price-source fallback chain, the six
  P/L window tests, and the "defaults encode measured findings" test
* Every documented number regenerates from `scripts/*.py` on the shipped defaults — no
  parameter overrides hiding in the scripts that produced the tables
* Full data rebuild timed end-to-end: **16.8s** (was 210s before the FRED UA fix)
* Backtests over **14,687 real LBMA bars (58y)** and 5,493 real GLD bars (22y): headline,
  macro-gate A/B, overfitting battery, brake bootstrap, 40-config family sweep — all on
  real data
* 2,252 von decisions + 300 calibration probes, **0 errors**, in-process on MLX
* Live tick end-to-end: switch ON → von answered (294 tokens) → correct flat decision;
  switch OFF → exposure collapsed to zero
* Fill simulator round-trip conserves cash minus exactly 2× the cost
* Dashboard rendered in headless Chrome with real status data
* Independent cross-check of the live decision: GLD last 398.38 vs 200d MA 416.28 →
  below the gate; 21d vol 23.9% → correct flat position, matching the backtest engine
* Price-source fallback exercised for real: with raw Yahoo 429ing, `fetch_prices` fell
  through to yfinance and returned 2,512 bars
* Stale-cache guard exercised for real: cache truncated by 30 days → auto-refetched

**Validated (checked, not executed):**

* The Vercel control endpoint's write path — `GITHUB_TOKEN` is not configured, so
  `POST /api/control` was never exercised against GitHub. Its read path is what the
  preview used.
* The LaunchAgent plist — written and path-checked, never bootstrapped (needs your shell).

**Known gaps, stated plainly:**

* FRED series are forward-filled without modelling publication lag — a small optimism.
* **The Sharpe edge over buy-and-hold is NOT statistically significant** — strategy CI
  [0.458, 0.986] overlaps buy & hold's [0.245, 0.757], even over 58 years. The strategy's
  own Sharpe excludes zero (PSR 1.0000) and clears Deflated Sharpe at 12 declared trials,
  but "beats buy & hold on Sharpe" is not established. The drawdown claim is.
* ~3-5%/yr of return is deliberately given up. This is a risk-reduction strategy.
* 26 losing years out of 59; worst year -10.6%. Positive months only 33%.
* In the 1980-2000 bear the strategy does **not** beat buy & hold on Sharpe (+/-0.01 at
  -0.10 vs -0.09) — it wins there only on drawdown (-21.8% vs -72.5%).
* News features cannot be backtested for lack of honest history; measured, not used.
* `allow_short` exists but is untested (would matter in a decade like 1980-2000).

## Docs

* `docs/STRATEGY.md` — the strategy, every component's justification, the negative results, citations, and the honest limits
* `docs/VON.md` — the full overlay write-up and how to reproduce it
* `docs/DATA.md` — every data endpoint, verified with status codes
