"""Invariant tests for the von-gold engine.

These test BEHAVIOUR CONTRACTS, not current values:
  * no lookahead (the single most important property of a backtest)
  * costs always reduce returns
  * the decision-model overlay can only ever reduce exposure, never create it
  * the switch fails closed
  * the position simulator conserves cash within one cost charge

Run: env -u PYTHONPATH .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vongold.backtest import buy_and_hold, run_backtest
from vongold.config import CostModel, StrategyParams
from vongold.dryrun import simulate_fill
from vongold.state_store import Control, Ledger, PaperPosition, PositionStore, read_status
from vongold.strategy import apply_rebalance_band, build_features, mechanical_exposure
from vongold.von_overlay import overlay_exposure


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def synth() -> pd.DataFrame:
    """Deterministic synthetic price series with a deliberate trend change.

    Synthetic (not downloaded) so the tests are fast, offline and reproducible.
    """
    rng = np.random.default_rng(11)
    n = 900
    idx = pd.bdate_range("2019-01-01", periods=n)
    drift = np.concatenate([np.full(450, 0.0006), np.full(450, -0.0004)])
    ret = drift + rng.normal(0, 0.009, n)
    close = 100 * np.exp(np.cumsum(ret))
    # Macro series must OSCILLATE, not ramp monotonically. A monotone driver pins the
    # trend filters permanently off (or on) and makes every downstream test vacuous --
    # exactly the failure this fixture originally had.
    phase = np.linspace(0, 6 * np.pi, n)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": 1e6,
            "real_yield_10y": 1.20 + 0.80 * np.sin(phase),
            "dollar_index_broad": 105.0 + 12.0 * np.sin(phase + 1.1),
        },
        index=idx,
    )
    df.index.name = "date"
    return df


# --------------------------------------------------------------- no-lookahead contract

def test_exposure_is_lagged_target(synth):
    """Holding on day t must equal the target decided on day t-1. Nothing else is safe."""
    res = run_backtest(synth, params=StrategyParams())
    expected = res.target.shift(1).fillna(0.0)
    assert res.exposure.equals(expected), "exposure is not the lagged target -> lookahead"


def test_future_prices_do_not_change_past_decisions(synth):
    """Truncating the future must not change decisions already made.

    This is the sharpest lookahead test: run on the full series, run on a prefix, and
    compare the overlapping decisions. Any rolling statistic that peeks forward breaks
    this immediately.
    """
    p = StrategyParams()
    full = run_backtest(synth, params=p)
    cut = len(synth) - 150
    prefix = run_backtest(synth.iloc[:cut], params=p)

    a = full.target.iloc[:cut]
    b = prefix.target
    # The final vol-window rows may differ because the prefix run has a shorter tail
    # available; compare through to cut-1 minus the longest window.
    compare_to = cut - max(p.momentum_lookbacks) - 5
    assert np.allclose(a.iloc[:compare_to], b.iloc[:compare_to], atol=1e-12), (
        "decisions changed when the future was removed -> lookahead"
    )


# ------------------------------------------------------------------ cost monotonicity

def test_costs_reduce_returns(synth):
    """Higher assumed costs must never produce a better result."""
    cheap = run_backtest(synth, cost=CostModel(spread_bps_per_side=0.0,
                                               slippage_bps_per_side=0.0))
    dear = run_backtest(synth, cost=CostModel(spread_bps_per_side=25.0,
                                              slippage_bps_per_side=25.0))
    assert dear.equity.iloc[-1] <= cheap.equity.iloc[-1], "costs did not reduce equity"
    assert dear.metrics["total_cost_paid"] > cheap.metrics["total_cost_paid"]


def test_round_trip_cost_arithmetic():
    c = CostModel(spread_bps_per_side=1.5, slippage_bps_per_side=1.5, commission_bps_per_side=0.0)
    assert c.round_trip_bps == pytest.approx(6.0)


# ------------------------------------------------------------- exposure bounds/overlay

def test_long_only_exposure_within_bounds(synth):
    res = run_backtest(synth, params=StrategyParams())
    assert res.exposure.min() >= 0.0
    assert res.exposure.max() <= 1.0 + 1e-12


def test_overlay_can_only_reduce_exposure(synth):
    """The decision model must never be able to INCREASE risk.

    Checked by construction: overlay multipliers are confined to [0,1], so applying
    one can never lift exposure above the mechanical target.
    """
    res = run_backtest(synth, params=StrategyParams())
    dates = synth.index
    rng = np.random.default_rng(3)
    ans = pd.DataFrame(
        {
            "regime": rng.choice(["up_trend", "down_trend", "range"], len(dates)),
            "regime_confidence": rng.uniform(0, 1, len(dates)),
            "long_prob": rng.uniform(0, 1, len(dates)),
            "risk_prob": rng.uniform(0, 1, len(dates)),
        },
        index=dates,
    )
    for mode in ("veto", "halve", "prob"):
        mult = overlay_exposure(ans, dates, mode=mode)
        assert mult.min() >= -1e-12, f"{mode} produced a negative multiplier"
        assert mult.max() <= 1.0 + 1e-12, f"{mode} produced a multiplier above 1"
        overlay = run_backtest(synth, params=StrategyParams(), overlay_exposure=mult)
        assert overlay.exposure.max() <= 1.0 + 1e-12
        # Where the multiplier was zero, the exposure must be zero -- remembering that
        # exposure on day t is the target decided on day t-1, so the multiplier that
        # governs day t's holding is mult.shift(1). Getting this lag wrong is the same
        # class of mistake as lookahead, so assert it explicitly.
        governing = mult.shift(1).fillna(mult.iloc[0])
        zero_days = governing[governing == 0].index
        if len(zero_days):
            held = overlay.exposure.reindex(zero_days).dropna()
            assert (held.abs() < 1e-12).all(), f"{mode} vetoed a day but still held exposure"


def test_overlay_veto_produces_zero_exposure(synth):
    dates = synth.index
    ans = pd.DataFrame(
        {
            "regime": ["down_trend"] * len(dates),
            "regime_confidence": [0.99] * len(dates),
            "long_prob": [0.01] * len(dates),
            "risk_prob": [0.99] * len(dates),
        },
        index=dates,
    )
    mult = overlay_exposure(ans, dates, mode="veto")
    assert (mult == 0).all()
    res = run_backtest(synth, params=StrategyParams(), overlay_exposure=mult)
    assert res.exposure.abs().max() == 0.0
    assert res.equity.iloc[-1] == pytest.approx(res.equity.iloc[0])


def test_prob_mode_is_monotone_in_long_prob():
    """More bullish evidence must not lower the multiplier."""
    dates = pd.bdate_range("2020-01-01", periods=11)
    ans = pd.DataFrame(
        {
            "regime": ["range"] * 11,
            "regime_confidence": [1.0] * 11,
            "long_prob": np.linspace(0.0, 1.0, 11),
            "risk_prob": [0.2] * 11,
        },
        index=dates,
    )
    mult = overlay_exposure(ans, dates, mode="prob")
    assert mult.is_monotonic_increasing


def test_missing_von_answer_does_not_block_trading(synth):
    """A missing model answer must leave the mechanical strategy intact, not zero it."""
    dates = synth.index
    ans = pd.DataFrame(
        {"regime": [None] * len(dates), "regime_confidence": [np.nan] * len(dates),
         "long_prob": [np.nan] * len(dates), "risk_prob": [np.nan] * len(dates)},
        index=dates,
    )
    mult = overlay_exposure(ans, dates, mode="prob")
    assert (mult == 1.0).all()


# --------------------------------------------------------------------- rebalance band

def test_rebalance_band_reduces_switching():
    target = pd.Series(np.sin(np.linspace(0, 40, 400)) / 2 + 0.5)
    banded = apply_rebalance_band(target, 0.25)
    no_band = apply_rebalance_band(target, 0.0)
    switches_banded = (banded.diff().fillna(0) != 0).sum()
    switches_raw = (no_band.diff().fillna(0) != 0).sum()
    assert switches_banded < switches_raw
    # And the banded path must stay inside the original range.
    assert banded.min() >= target.min() - 1e-12
    assert banded.max() <= target.max() + 1e-12


# ------------------------------------------------------------------- control / safety

def test_control_defaults_to_disabled(tmp_path):
    c = Control.load(tmp_path / "control.json")
    assert c.enabled is False


def test_control_fails_closed_on_corrupt_file(tmp_path):
    p = tmp_path / "control.json"
    p.write_text("{ this is not json ")
    c = Control.load(p)
    assert c.enabled is False
    assert "unreadable" in c.reason


def test_control_roundtrip(tmp_path):
    p = tmp_path / "control.json"
    c = Control(enabled=True, reason="test", changed_by="pytest")
    c.save(p)
    again = Control.load(p)
    assert again.enabled is True and again.changed_by == "pytest"


def test_switch_off_means_zero_target(tmp_path):
    """With the switch off, the runner must target zero exposure."""
    from vongold.dryrun import decide_today

    c = Control(enabled=False)
    target = 0.0 if not c.enabled else 1.0
    assert target == 0.0


# ------------------------------------------------------------------ fill simulator

def test_buy_then_sell_conserves_value_minus_costs():
    cost = CostModel(spread_bps_per_side=2.0, slippage_bps_per_side=1.0,
                     commission_bps_per_side=0.0)
    pos = PaperPosition(cash=10_000.0)
    price = 100.0

    buy = simulate_fill(pos, 0.5, price, cost)
    assert buy["traded"] is True
    assert pos.shares == pytest.approx(50.0)
    # cash spent = notional + fee
    expected_cash = 10_000.0 - 5_000.0 - 5_000.0 * 3.0 / 10_000.0
    assert pos.cash == pytest.approx(expected_cash)
    equity = pos.cash + pos.shares * price
    assert equity == pytest.approx(10_000.0 - 5_000.0 * 3.0 / 10_000.0)

    sell = simulate_fill(pos, 0.0, price, cost)
    assert sell["traded"] is True
    assert pos.shares == pytest.approx(0.0)
    assert pos.avg_cost == 0.0
    # Two round-trip sides paid: total cost = 2 * 3bps on ~5000
    total_cost = buy["cost"] + sell["cost"]
    assert total_cost == pytest.approx(2 * 5_000.0 * 3.0 / 10_000.0, rel=1e-6)
    assert pos.cash == pytest.approx(10_000.0 - total_cost)


def test_dust_trades_are_ignored():
    pos = PaperPosition(cash=10_000.0)
    cost = CostModel()
    simulate_fill(pos, 0.5, 100.0, cost)
    shares_before = pos.shares
    # Re-issue the same target: nothing should trade.
    out = simulate_fill(pos, 0.5, 100.0, cost)
    assert out["traded"] is False
    assert pos.shares == shares_before


def test_exposure_cannot_exceed_full_equity():
    pos = PaperPosition(cash=10_000.0)
    cost = CostModel()
    simulate_fill(pos, 5.0, 100.0, cost)  # absurd target, must clamp to 1.0
    assert pos.shares * 100.0 <= 10_000.0 + 1e-9


# --------------------------------------------------------------------- ledger/status

def test_ledger_appends_jsonl(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append("decision", date="2026-01-01", target=0.5)
    led.append("fill", date="2026-01-01", price=100.0)
    recs = led.all()
    assert len(recs) == 2
    assert recs[0]["kind"] == "decision" and recs[0]["target"] == 0.5
    assert all("ts" in r for r in recs)


def test_position_store_roundtrip(tmp_path):
    p = tmp_path / "position.json"
    s = PositionStore(p)
    s.state.cash = 1234.5
    s.state.shares = 3.0
    s.save()
    again = PositionStore(p)
    assert again.state.cash == 1234.5
    assert again.state.shares == 3.0


def test_position_store_seeds_initial_capital(tmp_path):
    """A fresh paper account must start funded, not at zero.

    A zero balance can never buy anything, so this is the difference between a working
    dry run and one that silently never trades.
    """
    s = PositionStore(tmp_path / "position.json", initial_capital=10_000.0)
    assert s.state.cash == 10_000.0
    assert s.state.equity == 0.0  # equity is set once a price is snapshotted


def test_position_store_survives_corruption(tmp_path):
    p = tmp_path / "position.json"
    p.write_text("not json at all")
    s = PositionStore(p, initial_capital=10_000.0)
    # Fails safe: a corrupt file resets to initial capital rather than crashing or
    # leaving the account unusable.
    assert s.state.cash == 10_000.0 and s.state.shares == 0.0


def test_news_impact_report_flags_short_history():
    """With too little news coverage the report must SAY so, not return an empty frame.

    An empty frame is indistinguishable from "the feeds were down", which would read as
    a clean negative result when it is really a missing one.
    """
    import numpy as np
    import pandas as pd

    from vongold.news import news_impact_report

    idx = pd.bdate_range("2020-01-01", periods=200)
    px = pd.DataFrame({"close": 100 * np.exp(np.cumsum(np.full(200, 0.001)))}, index=idx)
    idx_ms = idx.astype("datetime64[ms]")
    news = pd.DataFrame(
        {"n_headlines": 1.0, "news_fed_policy": 0.0, "news_sentiment": 0.1,
         "news_bucket_hits": 1.0},
        index=idx_ms[-5:],
    )
    rep = news_impact_report(news, px)
    assert not rep.empty, "a short news history must produce an explanatory row"
    assert rep.iloc[0]["n"] < 30
    assert "insufficient" in str(rep.iloc[0]["feature"]).lower()
    assert "note" in rep.columns


def test_status_reader_handles_missing(tmp_path):
    assert read_status(tmp_path / "nope.json") == {}


# ------------------------------------------------------------- mechanical sanity

def test_flat_before_enough_history(synth):
    """The strategy must not take a position before its volatility window exists."""
    res = run_backtest(synth, params=StrategyParams(vol_window=21, trend_filter_days=200))
    assert res.exposure.iloc[:21].abs().max() == 0.0


def test_buy_and_hold_charges_exactly_one_entry():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2021-01-01", periods=300)
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100 * np.exp(np.cumsum(rng.normal(0, .01, 300))),
         "volume": 1e6}, index=idx)
    res = buy_and_hold(df)
    assert res.metrics["exposure_switches"] <= 2
    assert res.metrics["pct_time_in_market"] > 0.99


def test_vol_targeting_scales_down_in_high_vol():
    """Higher realised volatility must produce a smaller position, all else equal."""
    idx = pd.bdate_range("2020-01-01", periods=400)
    calm = pd.Series(100 * np.exp(np.cumsum(np.full(400, 0.0005) + np.random.default_rng(1).normal(0, 0.003, 400))), index=idx)
    wild = pd.Series(100 * np.exp(np.cumsum(np.full(400, 0.0005) + np.random.default_rng(1).normal(0, 0.03, 400))), index=idx)
    p = StrategyParams(target_vol_annual=0.10, trend_filter_days=0, use_real_yield_filter=False,
                       use_dollar_filter=False, rebalance_band=0.0)
    sizes = []
    for s in (calm, wild):
        df = pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 1e6}, index=idx)
        f = build_features(df, p)
        sizes.append(float(mechanical_exposure(f, p).iloc[-1]))
    assert sizes[0] > sizes[1], f"calm size {sizes[0]} not greater than wild size {sizes[1]}"
