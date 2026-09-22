"""Configuration for the von-gold dry-run system.

Cost model defaults are deliberately pessimistic. A backtest that cannot survive
honest costs is not a strategy, it is a curve fit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CostModel:
    """Realistic frictions for trading a gold ETF (GLD/IAU) on daily bars.

    - expense_ratio: annual, accrued per calendar day held (GLD ~0.40%).
    - spread_bps: half-spread paid on each side. GLD quoted spread is ~1 cent on a
      ~$300-400 price = ~0.25-0.35 bps; we charge 1.5 bps per side to also absorb
      market-impact and the fact that retail fills are rarely at the touch.
    - slippage_bps: extra per side to cover timing/market impact.
    - commission_bps: per-side commission. Most brokers are $0 for ETFs.
    """

    expense_ratio_annual: float = 0.0040
    spread_bps_per_side: float = 1.5
    slippage_bps_per_side: float = 1.5
    commission_bps_per_side: float = 0.0

    @property
    def round_trip_bps(self) -> float:
        """Total cost in bps for entering and exiting once (excl. expense ratio)."""
        per_side = (
            self.spread_bps_per_side
            + self.slippage_bps_per_side
            + self.commission_bps_per_side
        )
        return 2.0 * per_side


@dataclass(frozen=True)
class StrategyParams:
    """Parameters for the trend/vol-targeted gold strategy.

    Nothing here is fitted to gold's recent history. Defaults are the conventional
    values from the published literature (see docs/STRATEGY.md) so that an
    out-of-sample result means something.
    """

    # Multi-horizon time-series momentum lookbacks, in trading days.
    momentum_lookbacks: tuple[int, ...] = (21, 63, 126, 252)
    # Trend filter: only take longs above this moving average (0 disables).
    trend_filter_days: int = 200
    # Realized-vol window (days) for scaling.
    vol_window: int = 21
    # Volatility target, annualized. Harvey et al. show vol targeting improves
    # risk-adjusted returns at moderate levels.
    target_vol_annual: float = 0.12
    # Cap on gross exposure (1.0 = fully invested, no leverage).
    max_exposure: float = 1.0
    # Do not bother trading tiny adjustments: rebalance only when target exposure
    # moves more than this from current.
    rebalance_band: float = 0.10
    # ATR-based protective stop (in ATR units). 0 disables.
    atr_stop_mult: float = 0.0
    atr_window: int = 14

    # --- Macro conditioning (slow regime tilt, documented drivers) ---
    # Condition longs on the 10-year real yield trend. Real yields are gold's
    # single best-documented macro driver.
    use_real_yield_filter: bool = True
    real_yield_trend_days: int = 63
    # Condition on the broad dollar index trend (headwind when strengthening).
    use_dollar_filter: bool = True
    dollar_trend_days: int = 63

    # --- Decision-model overlay ---
    # OFF BY DEFAULT, and that is a measured conclusion, not a placeholder.
    # See docs/VON.md: over 2,252 real trading days von never rated a long justified
    # (max P=0.27), and its one genuine competence -- reading distance from the 200-day
    # moving average (corr +0.75) -- is already an input to this strategy. Modes that
    # threshold its probability therefore reduce to a constant decision, which looks
    # good in one half of the sample and collapses in the other. Enable it to test a
    # hypothesis, not because it is expected to help.
    use_von_overlay: bool = False
    # Minimum von confidence to act on a von answer.
    von_min_confidence: float = 0.60
    # Exposure multiplier applied when von disagrees with the mechanical signal.
    von_disagree_scale: float = 0.0
    von_url: str = "http://127.0.0.1:8100"
    # Overlay mode: veto | halve | prob | rank  (see von_overlay.MODES)
    von_mode: str = "rank"


@dataclass
class BacktestConfig:
    symbol: str = "GLD"
    start: str = "2010-01-01"
    end: str | None = None
    initial_capital: float = 10_000.0
    # Signal computed on day t is executed on day t+1 (no lookahead).
    execution_lag_days: int = 1
    cost: CostModel = field(default_factory=CostModel)
    params: StrategyParams = field(default_factory=StrategyParams)


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def save_config(cfg: BacktestConfig, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(cfg), indent=2))
    return p


def load_config(path: str | Path) -> BacktestConfig:
    raw = json.loads(Path(path).read_text())
    cost = CostModel(**raw.pop("cost", {}))
    params_raw = raw.pop("params", {})
    if "momentum_lookbacks" in params_raw:
        params_raw["momentum_lookbacks"] = tuple(params_raw["momentum_lookbacks"])
    params = StrategyParams(**params_raw)
    return BacktestConfig(cost=cost, params=params, **raw)
