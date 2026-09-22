# von-gold

A **dry-run (paper) trading system for gold** that runs on your own machine, driven by
your local **von** decision model (395M MLX, ~18ms, **zero LLM tokens**) — with the
honest finding that von does not improve it, and a mechanical strategy that does what a
professional trend-following program would do.

```
Buy & hold GLD        11.65% CAGR   Sharpe 0.76   max DD -26.5%
von-gold mechanical    5.83% CAGR   Sharpe 0.79   max DD -18.0%
```

**Read that honestly: it does not beat buy-and-hold on return. It beats it on
risk-adjusted return and roughly halves the drawdown, at the cost of about half the
return.** That is what a real trend-following overlay looks like on a single commodity.
Any pitch promising a system that "always comes out as a win" is describing a backtest
nobody stress-tested — see `docs/STRATEGY.md` for the statistical tests this one fails
and passes.

## The one-paragraph version

Gold trends and it has no cash flow, so what moves it is (a) its own price momentum and
(b) the 10-year real yield, which is its literal opportunity cost. The bot holds gold
when trend and macro regime agree, sizes the position by volatility, and sits flat
otherwise. Decision made at day *t*'s close, filled at day *t+1*'s close, with 6bps of
round-trip costs charged. Long-only, unlevered. Runs once per session on the Mac, writes
its state to a file, and a Vercel page shows you how it's doing with an on/off switch.

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
env -u PYTHONPATH .venv/bin/python -m pytest tests -q      # 24 passing
```

**Always `env -u PYTHONPATH`.** A Hermes/agent session exports a `PYTHONPATH` that points
at its own venv and silently shadows this one's packages — you get `ModuleNotFoundError`
for something you can see is installed.

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
| `src/vongold/data.py` | Yahoo daily prices + FRED macro series, cached to parquet |
| `src/vongold/strategy.py` | Trend + vol-targeting signal, macro gates, rebalance band |
| `src/vongold/backtest.py` | Event-driven backtester with real costs, metrics |
| `src/vongold/experiments.py` | Ablation, sweeps, walk-forward, Deflated Sharpe, bootstrap |
| `src/vongold/von_state.py` | Turns a trading day's numbers into a von decision problem |
| `src/vongold/von_client.py` | Talks to local von (in-process or subprocess), never raises |
| `src/vongold/von_overlay.py` | The overlay contract + A/B harness |
| `src/vongold/news.py` | Free RSS feeds, deterministic lexical classification, impact test |
| `src/vongold/dryrun.py` | The paper-trading loop, kill switch, fill simulation |
| `src/vongold/state_store.py` | The on/off switch, the ledger, the status blob |
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

1. **Dollar-index filter** — Sharpe 0.73 → **0.44**. The dollar and gold are correlated
   enough that the filter duplicated the price trend while lagging it.
2. **ATR trailing stop** — did nothing; the 200-day gate already exits on those days.
3. **Momentum sign as the main driver** — flat 79% of the time; averaging four lookbacks
   gives exactly 0 whenever the windows disagree, which is often.
4. **Real-yield filter as a return source** — value-neutral. A modest tilt only.

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

* 24/24 invariant tests pass, including the two no-lookahead checks
* Backtests over 2,512 real GLD bars; ablation, 108-config sweep, walk-forward, bootstrap
  and Deflated Sharpe all run on real data
* 2,252 von decisions + 300 calibration probes, **0 errors**, in-process on MLX
* Live tick end-to-end: switch ON → von answered (294 tokens) → correct flat decision;
  switch OFF → exposure collapsed to zero
* Fill simulator round-trip conserves cash minus exactly 2× the cost
* Dashboard rendered in headless Chrome with real status data

**Validated (checked, not executed):**

* The Vercel control endpoint's write path — `GITHUB_TOKEN` is not configured, so
  `POST /api/control` was never exercised against GitHub. Its read path is what the
  preview used.
* The LaunchAgent plist — written and path-checked, never bootstrapped (needs your shell).

**Known gaps, stated plainly:**

* FRED series are forward-filled without modelling publication lag — a small optimism.
* The 2016→2026 window is a mostly-bullish decade for gold; the long-only restriction
  flatters the result and no 2013-2015 bear market is in sample.
* News features cannot be backtested for lack of honest history; they are measured, not used.
* The strategy's own Sharpe is **not** statistically distinguishable from zero
  (bootstrap CI [-0.015, 0.072] includes zero). The *family* — trend + vol targeting on
  gold — does clear the Deflated Sharpe bar at 108 trials. That distinction is the point.

## Docs

* `docs/STRATEGY.md` — the strategy, every component's justification, the negative results, citations
* `docs/VON.md` — the full overlay write-up and how to reproduce it
* `docs/DATA.md` — every data endpoint, verified with status codes
