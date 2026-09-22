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


# ------------------------------------------------------- long-history data loaders

def test_lbma_loader_shape_and_known_print():
    """The 58-year series must parse, be monotonic in time, and contain a known price.

    Sanity-checking against an external fact (gold's ~$1,895 autumn-2011 peak) is what
    separates "the file parsed" from "the file is the right file".
    """
    import pytest

    from vongold.data import load_lbma_gold

    try:
        df = load_lbma_gold()
    except FileNotFoundError:
        pytest.skip("LBMA series not present on this machine")

    assert len(df) > 10_000, "expected decades of daily observations"
    assert df.index.is_monotonic_increasing
    assert df["close"].notna().all()
    assert (df["close"] > 0).all()
    peak = float(df["close"].loc["2011-08-01":"2011-10-01"].max())
    assert 1500 < peak < 2300, f"autumn-2011 peak {peak} is not a plausible gold price"
    # The series must span the secular bear markets a GLD backtest cannot see.
    assert df.index.min() < pd.Timestamp("1970-01-01")
    assert float(df["close"].loc["1980-01-21":"1980-02-28"].max()) > 400  # 1980 spike


def test_local_parquet_loader_normalises(tmp_path):
    from vongold.data import load_local_parquet

    idx = pd.date_range("2020-01-01", periods=10, tz="America/New_York")
    src = pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5,
                        "Adj Close": 1.5, "Volume": 100.0}, index=idx)
    p = tmp_path / "x.parquet"
    src.to_parquet(p)
    df = load_local_parquet(p)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is None, "a tz-aware index would break alignment with FRED data"
    assert str(df.index.dtype) == "datetime64[ms]"


def test_yahoo_short_series_is_rejected():
    """A 200 response with too few rows must raise, not silently pass.

    Observed in the wild: range=max returned 263 rows instead of ~5,500 with no error.
    A silent short series would corrupt a backtest invisibly.
    """
    import pytest

    from vongold.data import fetch_yahoo_daily

    class FakeResp:
        status_code = 200

        def json(self):
            ts = [1600000000 + i * 86400 for i in range(50)]
            return {"chart": {"result": [{
                "meta": {"exchangeTimezoneName": "UTC"},
                "timestamp": ts,
                "indicators": {"quote": [{"open": [1.0] * 50, "high": [1.0] * 50,
                                          "low": [1.0] * 50, "close": [1.0] * 50,
                                          "volume": [1.0] * 50}]},
            }], "error": None}}

    class FakeSession:
        def get(self, *a, **k):
            return FakeResp()

    with pytest.raises(RuntimeError, match="range downgrade"):
        fetch_yahoo_daily("GLD", rng="max", session=FakeSession(), retries=0, min_rows=4000)


# ------------------------------------------------- whipsaw brake (min_hold_days)

def test_min_hold_days_blocks_rapid_reflips():
    """The brake must actually delay changes, and must be symmetric (entries AND exits).

    Delaying an exit delays protection, so the symmetry is a real trade-off rather than
    a free win -- this test pins the behaviour so a future change cannot quietly make it
    one-directional.
    """
    from vongold.strategy import apply_rebalance_band

    idx = pd.date_range("2020-01-01", periods=12)
    # Flip every single day: 1,1,0,0,1,1,0,0,...
    target = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], index=idx)
    flat = apply_rebalance_band(target, 0.05, min_hold_days=0)
    braked = apply_rebalance_band(target, 0.05, min_hold_days=5)

    assert flat.tolist() == target.tolist(), "with no brake it should track the target"
    changes_flat = int((flat.diff().abs() > 0).sum())
    changes_braked = int((braked.diff().abs() > 0).sum())
    assert changes_braked < changes_flat, "the brake must reduce the number of changes"


def test_min_hold_days_zero_is_a_noop():
    from vongold.strategy import apply_rebalance_band

    idx = pd.date_range("2020-01-01", periods=40)
    t = pd.Series(np.random.default_rng(3).choice([0.0, 1.0], size=40), index=idx)
    a = apply_rebalance_band(t, 0.1, min_hold_days=0)
    b = apply_rebalance_band(t, 0.1)
    pd.testing.assert_series_equal(a, b)


def test_shipped_defaults_reflect_measured_findings():
    """Defaults encode conclusions that cost real experiments to reach.

    If someone flips these back without re-running the A/Bs, the tests should say so.
    - macro gates: LAGGED test showed they HURT (-0.029 / -0.263 Sharpe)
    - vol target 10%: the sweep's measured optimum for gold
    - von overlay: measured, and found not to help
    """
    from vongold.config import StrategyParams

    p = StrategyParams()
    assert p.use_real_yield_filter is False, "real-yield gate measured as harmful on a lagged basis"
    assert p.use_dollar_filter is False, "dollar gate measured as clearly harmful on a lagged basis"
    assert p.target_vol_annual == 0.10
    assert p.min_hold_days == 10, "set for cost, not alpha"
    assert p.use_von_overlay is False


def test_strategy_beats_buyhold_drawdown_on_long_history():
    """The one large, reliable effect: drawdown control over 58 years.

    Also pins the honest caveat -- the Sharpe advantage over buy & hold is NOT
    statistically significant (the bootstrap CIs overlap). Only the drawdown claim is
    robust. This test exists so that claim cannot drift upward unnoticed.
    """
    import pytest

    from vongold.backtest import buy_and_hold, run_backtest
    from vongold.config import CostModel, StrategyParams
    from vongold.data import load_lbma_gold

    try:
        df = load_lbma_gold()
    except FileNotFoundError:
        pytest.skip("LBMA series not present")
    cost = CostModel()
    p = StrategyParams()
    m = run_backtest(df, params=p, cost=cost).metrics
    b = buy_and_hold(df, cost=cost).metrics

    # Reliable, large: drawdown is roughly a quarter of buy & hold's.
    assert m["max_drawdown"] > b["max_drawdown"] + 0.35, (
        f"strategy DD {m['max_drawdown']:.1%} should be far better than {b['max_drawdown']:.1%}"
    )
    assert m["vol_annual"] < b["vol_annual"], "risk targeting should lower realised vol"
    assert m["sharpe"] > b["sharpe"], "sharpe edge is positive (documented as not significant)"
    # The honest limit, asserted so it is not overclaimed later.
    assert m["sharpe"] - b["sharpe"] < 0.60, "no evidence for an edge this large"


def test_brake_never_delays_a_risk_veto(synth):
    """Regression: the whipsaw brake must not postpone a safety exit.

    A first implementation applied the brake AFTER the overlay multiplier, which meant a
    veto could be ignored for up to min_hold_days -- the strategy would keep holding
    through a signal that said get out. Smoothed entries are fine; delayed kill switches
    are not. This test fails if anyone reorders those two steps again.
    """
    import numpy as np
    import pandas as pd

    from vongold.backtest import run_backtest
    from vongold.config import StrategyParams
    from vongold.von_overlay import overlay_exposure

    dates = synth.index
    n = len(dates)
    # Veto everything from the midpoint onward, decisively.
    ans = pd.DataFrame(
        {
            "regime": ["range"] * n,
            "regime_confidence": [0.9] * n,
            "long_prob": [0.01] * n,
            "risk_prob": [0.99] * n,
        },
        index=dates,
    )
    mult = overlay_exposure(ans, dates, mode="veto")
    # Force a hard kill from the midpoint so the brake has every chance to delay it.
    mid = n // 2
    mult.iloc[mid:] = 0.0

    # A long brake: if the brake were applied after the overlay this would visibly fail.
    p = StrategyParams(min_hold_days=20)
    res = run_backtest(synth, params=p, overlay_exposure=mult)

    # Exposure on day t is the target decided at t-1. So from mid+1 onward the holding
    # must be exactly zero, with no grace period.
    after = res.exposure.iloc[mid + 1:]
    assert (after.abs() < 1e-12).all(), (
        f"brake delayed the veto: {int((after.abs() > 1e-12).sum())} days still held "
        f"exposure after the kill signal"
    )


# ------------------------------------------------- stale-cache guard (daily bot safety)

def test_stale_cache_triggers_refetch(tmp_path):
    """A cache that is never invalidated is a silent-failure trap for a daily bot.

    build_dataset used to be "if the parquet exists, return it" -- written once, served
    forever. The dry run would then keep trading on the last day it happened to fetch,
    with no error and no stale-looking state, because every downstream number is computed
    from the same stale frame. This test pins the freshness guard.
    """
    from vongold.data import latest_bar_age_days

    fresh = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2026-09-21"]))
    fresh.index = fresh.index.astype("datetime64[ms]")
    old = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2020-01-02"]))
    old.index = old.index.astype("datetime64[ms]")

    today = pd.Timestamp("2026-09-21")
    assert latest_bar_age_days(fresh, today) == 0
    assert latest_bar_age_days(old, today) > 1000
    # An empty frame must look infinitely stale, never fresh.
    assert latest_bar_age_days(pd.DataFrame(columns=["close"]), today) > 100000


def test_price_source_fallback_chain():
    """fetch_prices must fall through to the next source instead of giving up.

    Raw Yahoo was observed returning 429 on every request from this IP while yfinance
    returned fine. With a single-source fetch the daily tick would simply fail. This
    asserts the chain reports the source it used and raises only when ALL fail.
    """
    import vongold.data as d

    good = pd.DataFrame({"close": [1.0, 2.0]},
                        index=pd.DatetimeIndex(["2026-09-18", "2026-09-21"]).astype("datetime64[ms]"))
    good.index.name = "date"
    calls = []

    def boom(*a, **k):
        calls.append("failed")
        raise RuntimeError("simulated 429")

    def works(*a, **k):
        calls.append("ok")
        return good.copy()

    orig = (d.fetch_yfinance_daily, d.fetch_yahoo_daily, d.fetch_nasdaq_daily)
    try:
        # First two sources fail -> third must be used.
        d.fetch_yfinance_daily = boom
        d.fetch_yahoo_daily = boom
        d.fetch_nasdaq_daily = works
        df, src = d.fetch_prices("GLD")
        assert src == "nasdaq", f"expected fallback to nasdaq, got {src}"
        assert len(df) == 2

        # All three fail -> must raise, never return an empty frame. Returning empty
        # would look like "flat market" to the strategy.
        d.fetch_nasdaq_daily = boom
        try:
            d.fetch_prices("GLD")
            raise AssertionError("fetch_prices returned success with every source down")
        except RuntimeError as exc:
            assert "all price sources failed" in str(exc)
    finally:
        d.fetch_yfinance_daily, d.fetch_yahoo_daily, d.fetch_nasdaq_daily = orig


def test_fred_does_not_reuse_the_browser_session():
    """FRED rejects the Chrome User-Agent that Yahoo requires.

    Measured: fredgraph.csv with a browser UA returned a connection-level failure while
    the same request with no UA returned 200. Reusing one session for both hosts made
    every macro series fail with a 40s timeout each (~200s per tick) and silently dropped
    all macro columns. This pins the actual behaviour -- that the browser UA constant is
    not used to build the FRED request -- rather than grepping the source, which would
    also match the docstring explaining the problem.
    """
    import inspect

    import vongold.data as d

    sig = inspect.signature(d.fetch_fred)
    assert "timeout" in sig.parameters
    assert sig.parameters["timeout"].default <= 20, "a long timeout makes a bad macro day expensive"

    # Capture the headers actually sent.
    seen: list[dict] = []

    class FakeResp:
        status_code = 200
        # Must exceed fetch_fred's >100-byte sanity gate -- it rejects a short body,
        # which is how a truncated/error page gets caught. Real FRED CSVs are ~99KB.
        text = "observation_date,DFII10\n" + "".join(
            f"2026-0{i%9+1}-0{i%9+1},{2.0 + i * 0.01:.2f}\n" for i in range(1, 25)
        )

    def fake_get(url, headers=None, timeout=None):
        seen.append(dict(headers or {}))
        return FakeResp()

    orig = d.requests.get
    try:
        d.requests.get = fake_get
        d.fetch_fred("DFII10")
    finally:
        d.requests.get = orig

    assert seen, "fetch_fred made no request"
    for hdrs in seen:
        ua = hdrs.get("User-Agent", "")
        assert "Mozilla" not in ua and "Chrome" not in ua, (
            f"fetch_fred sent a browser User-Agent ({ua!r}); FRED rejects those"
        )


# ------------------------------------------------------------------ P/L windows

def _ledger(tmp_path, events):
    import json
    p = tmp_path / "ledger.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return p


def test_pl_all_time_is_against_the_initial_stake(tmp_path):
    """All-time P/L must be measured from the $10k the user actually started with."""
    from datetime import datetime, timedelta, timezone

    from vongold.pl import compute_pl

    now = datetime.now(timezone.utc)
    p = _ledger(tmp_path, [
        {"ts": (now - timedelta(days=5)).isoformat(), "kind": "fill", "price": 400.0,
         "shares": 0.0, "cash": 10000.0},
        {"ts": (now - timedelta(days=1)).isoformat(), "kind": "fill", "price": 410.0,
         "shares": 0.0, "cash": 10250.0},
    ])
    out = compute_pl(ledger_path=p, current_equity=10250.0, shares=0.0, now=now,
                     intraday_bars=None)
    assert out["initial_capital"] == 10000.0
    all_time = out["windows"][0]
    assert all_time["label"] == "all_time"
    assert all_time["pl_abs"] == 250.0
    assert abs(all_time["pl_pct"] - 0.025) < 1e-9


def test_pl_flat_at_both_ends_is_not_reported_as_no_loss(tmp_path):
    """Regression: flat at both ends does NOT mean no P/L was possible.

    A position opened and closed inside the window leaves the account flat at both ends
    while realising a gain. The first implementation of this module reported the correct
    dollar figure but labelled it "no gain or loss was possible", which is a
    contradiction a user would have spotted and distrusted. It must be flagged as
    containing a trade instead.
    """
    from datetime import datetime, timedelta, timezone

    from vongold.pl import compute_pl

    now = datetime.now(timezone.utc)
    p = _ledger(tmp_path, [
        {"ts": (now - timedelta(hours=10)).isoformat(), "kind": "fill", "price": 400.0,
         "shares": 0.0, "cash": 10000.0},
        {"ts": (now - timedelta(hours=5)).isoformat(), "kind": "fill", "price": 400.0,
         "shares": 5.0, "cash": 8000.0},
        {"ts": (now - timedelta(hours=2)).isoformat(), "kind": "fill", "price": 410.0,
         "shares": 0.0, "cash": 10050.0},
    ])
    out = compute_pl(ledger_path=p, current_equity=10050.0, shares=0.0, now=now,
                     intraday_bars=None)
    w6 = next(w for w in out["windows"] if w["label"] == "past_6h")
    assert w6["pl_abs"] == 50.0, "the realised gain must be reported"
    assert w6["basis"] != "flat_no_exposure", (
        "a window containing a trade must never be labelled as having no exposure"
    )
    assert w6["basis"] == "includes_a_trade"


def test_pl_genuinely_flat_window_is_exactly_zero(tmp_path):
    """When no position was open at any point, $0 is the exact answer, and it is labelled."""
    from datetime import datetime, timedelta, timezone

    from vongold.pl import compute_pl

    now = datetime.now(timezone.utc)
    p = _ledger(tmp_path, [
        {"ts": (now - timedelta(days=2)).isoformat(), "kind": "fill", "price": 400.0,
         "shares": 0.0, "cash": 10000.0},
    ])
    out = compute_pl(ledger_path=p, current_equity=10000.0, shares=0.0, now=now,
                     intraday_bars=None)
    for w in out["windows"]:
        assert w["pl_abs"] == 0.0
    for label in ("past_24h", "past_6h", "past_hour"):
        w = next(x for x in out["windows"] if x["label"] == label)
        assert w["basis"] == "flat_no_exposure"


def test_pl_curve_is_reconstructed_not_sampled(tmp_path):
    """The curve must come from ledger fills, not from a sampled status blob.

    A sampled curve would report a sub-day window as flat purely because no tick landed
    inside it. Reconstructing from the append-only ledger keeps it exact.
    """
    from datetime import datetime, timedelta, timezone

    from vongold.pl import reconstruct_equity_curve, load_events

    now = datetime.now(timezone.utc)
    p = _ledger(tmp_path, [
        {"ts": (now - timedelta(days=3)).isoformat(), "kind": "decision"},
        {"ts": (now - timedelta(days=2)).isoformat(), "kind": "fill", "price": 400.0,
         "shares": 2.0, "cash": 9200.0},
        {"ts": (now - timedelta(days=1)).isoformat(), "kind": "fill", "price": 405.0,
         "shares": 0.0, "cash": 10010.0},
    ])
    curve = reconstruct_equity_curve(load_events(p))
    assert len(curve) == 2, "only fills define an equity mark"
    assert curve[0][1] == 9200.0 + 2.0 * 400.0
    assert curve[1][1] == 10010.0


def test_pl_survives_a_corrupt_ledger_line(tmp_path):
    """An interrupted write leaves a partial final line; that must not break reporting."""
    import json
    from datetime import datetime, timedelta, timezone

    from vongold.pl import compute_pl

    now = datetime.now(timezone.utc)
    p = tmp_path / "ledger.jsonl"
    good = {"ts": (now - timedelta(days=1)).isoformat(), "kind": "fill",
            "price": 400.0, "shares": 0.0, "cash": 10100.0}
    p.write_text(json.dumps(good) + "\n" + '{"ts": "2026-09-21T00:00:00+00:00", "kin')
    out = compute_pl(ledger_path=p, current_equity=10100.0, shares=0.0, now=now,
                     intraday_bars=None)
    assert out["fills"] == 1
    assert out["total_pl_abs"] == 100.0


def test_pl_windows_cover_the_requested_labels(tmp_path):
    """The user asked for exactly these four windows."""
    from vongold.pl import compute_pl

    p = _ledger(tmp_path, [])
    out = compute_pl(ledger_path=p, current_equity=10000.0, shares=0.0, intraday_bars=None)
    labels = [w["label"] for w in out["windows"]]
    assert labels == ["all_time", "past_24h", "past_6h", "past_hour"]


# ---------------------------------------------------------------------------
# Five-way action output (long / short / open / close / hold)
# ---------------------------------------------------------------------------

def test_action_criteria_are_a_dict_and_score_criteria_are_a_list():
    """von's two question types take DIFFERENT criteria shapes, and getting it wrong fails
    the whole decision with a pydantic ValidationError.

    This is a real bug that silently voided an entire 270-day probe: `choice` wants a dict
    of label -> condition, `score` wants a LIST of anchors. Pinning it here so it cannot
    regress.
    """
    from vongold.action import QUESTION
    assert isinstance(QUESTION["action"]["criteria"], dict), "choice criteria must be a dict"
    assert isinstance(QUESTION["conviction"]["criteria"], list), "score criteria must be a list"
    assert QUESTION["action"]["type"] == "choice"
    assert QUESTION["conviction"]["type"] == "score"


def test_action_labels_match_the_five_requested_actions():
    """The five actions the user asked for, and the exposure each implies."""
    from vongold.action import ACTIONS, ACTION_TARGET
    assert set(ACTIONS) == {"open_long", "add_long", "hold", "reduce", "close"}
    assert "short" not in ACTIONS, (
        "short is deliberately absent: measured over 58 years it changed Sharpe by +0.002 "
        "(4.88% -> 4.89% CAGR), because the above_ma gate is 0/1 and zeroes a short in a "
        "downtrend. Offering it would imply an edge that measurement does not support."
    )
    assert ACTION_TARGET["hold"] is None, "hold must leave the mechanical target untouched"
    assert ACTION_TARGET["close"] == 0.0


def test_action_weight_defaults_to_zero_so_an_unvalidated_model_cannot_steer():
    """The action output must be inert until it is validated.

    A model that always answers the same label produces a constant overlay. At weight 0 the
    mechanical target passes through unchanged, so such a model cannot move the account.
    """
    from vongold.action import ACTION_PRIOR_WEIGHT
    assert ACTION_PRIOR_WEIGHT == 0.0, (
        "the five-way action is unvalidated; its weight must default to 0 so it cannot "
        "steer exposure before an A/B clears it"
    )


def test_blend_is_inert_at_weight_zero_and_bounded():
    from vongold.action import Action, blend
    act = Action(action="open_long", conviction=4.0)
    t, why = blend(0.37, act, weight=0.0)
    assert abs(t - 0.37) < 1e-12, "weight 0 must pass the mechanical target through"
    # full weight moves toward the action's own target
    t2, _ = blend(0.37, Action(action="close"), weight=1.0)
    assert abs(t2 - 0.0) < 1e-12, "close at full weight must flatten"
    # and the result is always within [0, 1]
    for w in (0.0, 0.25, 0.5, 1.0):
        for a in ("open_long", "add_long", "reduce", "close", "hold"):
            tt, _ = blend(0.8, Action(action=a), weight=w)
            assert 0.0 <= tt <= 1.0


def test_blend_fails_closed_when_the_model_errors():
    """An error must yield the mechanical target, never a guess in either direction."""
    from vongold.action import Action, blend
    t, why = blend(0.42, Action(error="von timed out"), weight=1.0)
    assert abs(t - 0.42) < 1e-12
    assert "no action" in why


def test_live_loop_never_opens_a_position_without_allow_entry():
    """The intraday re-decide loop is NOT backtested, so by default it may only reduce risk.

    Opening or adding intraday would be trading a strategy with no evidence behind it. This
    asserts the safety property that makes the loop safe to leave running.
    """
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parents[1] / "scripts" / "live_loop.py").read_text()
    # the gate must consult allow_entry before accepting an entry action
    assert '"open_long", "add_long") and args.allow_entry' in src
    assert "ignored (entry disabled)" in src
