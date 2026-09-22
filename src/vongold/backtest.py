"""Event-driven daily backtester with honest frictions, plus performance metrics.

Execution model (deliberately conservative):
  * A decision made from day t's close is executed at day t+1's close
    (`execution_lag_days=1`). No same-bar fills, ever.
  * Returns are close-to-close, so the strategy earns the return of the day
    AFTER the decision.
  * Transaction costs are charged per side on the traded notional, plus the ETF
    expense ratio accrued daily on the held notional.
  * Cash earns nothing (conservative: T-bill interest on the un-invested sleeve
    would flatter a long-only strategy and is not the point of this test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CostModel, StrategyParams
from .strategy import build_features, mechanical_exposure, apply_rebalance_band

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    exposure: pd.Series
    target: pd.Series
    trades: pd.Series          # traded notional (|delta exposure|) per day
    costs: pd.Series           # cost in return units per day
    params: StrategyParams
    cost_model: CostModel
    metrics: dict

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "close": self.target.index.to_series().map(lambda _: np.nan),
            }
        ).drop(columns=["close"]).join(
            [
                self.equity.rename("equity"),
                self.returns.rename("strategy_ret"),
                self.exposure.rename("exposure"),
                self.target.rename("target"),
                self.trades.rename("traded"),
                self.costs.rename("cost"),
            ]
        )


def run_backtest(
    df: pd.DataFrame,
    params: StrategyParams | None = None,
    cost: CostModel | None = None,
    initial_capital: float = 10_000.0,
    execution_lag_days: int = 1,
    allow_short: bool = False,
    overlay_exposure: pd.Series | None = None,
    target_override: pd.Series | None = None,
) -> BacktestResult:
    """Run the mechanical strategy; optionally apply a pre-computed overlay.

    `target_override` lets a caller supply a fully-formed target exposure series
    (used by the von overlay A/B and by the parameter sweep).
    """
    p = params or StrategyParams()
    c = cost or CostModel()

    f = build_features(df, p)
    if target_override is not None:
        target = target_override.reindex(f.index).fillna(0.0)
    else:
        target = mechanical_exposure(f, p, allow_short=allow_short)

    if overlay_exposure is not None:
        target = target * overlay_exposure.reindex(f.index).fillna(1.0)

    target = apply_rebalance_band(target.clip(-p.max_exposure, p.max_exposure), p.rebalance_band)

    # Decision on t -> held over t+1. shift() moves the target forward so that
    # holding[t+1] == decided[t].
    held = target.shift(execution_lag_days).fillna(0.0)

    asset_ret = f["ret1"].fillna(0.0)
    gross = held * asset_ret

    traded = held.diff().abs().fillna(held.abs())
    per_side = c.spread_bps_per_side + c.slippage_bps_per_side + c.commission_bps_per_side
    trade_cost = traded * (per_side / 10_000.0)
    expense = held.abs() * (c.expense_ratio_annual / TRADING_DAYS)

    net = gross - trade_cost - expense

    equity = initial_capital * (1.0 + net).cumprod()

    result = BacktestResult(
        equity=equity,
        returns=net,
        exposure=held,
        target=target,
        trades=traded,
        costs=trade_cost + expense,
        params=p,
        cost_model=c,
        metrics={},
    )
    result.metrics = compute_metrics(result, asset_ret, c)
    result.metrics["final_equity"] = float(equity.iloc[-1])
    result.metrics["initial_capital"] = float(initial_capital)
    return result


def buy_and_hold(df: pd.DataFrame, cost: CostModel | None = None, initial_capital: float = 10_000.0) -> BacktestResult:
    """Benchmark: buy the ETF on the first bar and hold to the end."""
    c = cost or CostModel()
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0.0)

    held = pd.Series(1.0, index=close.index)
    held.iloc[0] = 0.0  # buy at the first close
    traded = held.diff().abs().fillna(1.0)
    per_side = c.spread_bps_per_side + c.slippage_bps_per_side + c.commission_bps_per_side
    trade_cost = traded * (per_side / 10_000.0)
    expense = held.abs() * (c.expense_ratio_annual / TRADING_DAYS)
    net = held * ret - trade_cost - expense
    equity = initial_capital * (1.0 + net).cumprod()

    res = BacktestResult(
        equity=equity,
        returns=net,
        exposure=held,
        target=held,
        trades=traded,
        costs=trade_cost + expense,
        params=StrategyParams(),
        cost_model=c,
        metrics={},
    )
    res.metrics = compute_metrics(res, ret, c)
    res.metrics["final_equity"] = float(equity.iloc[-1])
    return res


def compute_metrics(res: BacktestResult, asset_ret: pd.Series, c: CostModel) -> dict:
    r = res.returns.dropna()
    if len(r) < 2:
        return {}
    eq = res.equity
    years = len(r) / TRADING_DAYS
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() * TRADING_DAYS) / vol if vol > 0 else 0.0

    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (r.mean() * TRADING_DAYS) / downside if downside > 0 else 0.0

    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    active = res.exposure.abs() > 1e-9
    exposure_turnover = float(res.trades.sum())
    n_switches = int((np.sign(res.exposure).diff().fillna(0) != 0).sum())

    # Monthly consistency: how often is a calendar month positive?
    monthly = (1.0 + r).resample("ME").prod() - 1.0
    monthly = monthly.dropna()

    # Rolling 12-month worst case, a practical risk-of-ruin proxy.
    roll12 = (1.0 + r).rolling(TRADING_DAYS).apply(np.prod, raw=True) - 1.0

    return {
        "bars": int(len(r)),
        "years": round(float(years), 2),
        "cagr": float(cagr),
        "vol_annual": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
        "hit_rate_daily": float((r > 0).mean()),
        "pct_time_in_market": float(active.mean()),
        "avg_exposure_when_active": float(res.exposure[active].abs().mean()) if active.any() else 0.0,
        "total_traded_notional": exposure_turnover,
        "exposure_switches": n_switches,
        "months": int(len(monthly)),
        "pct_months_positive": float((monthly > 0).mean()) if len(monthly) else 0.0,
        "worst_month": float(monthly.min()) if len(monthly) else 0.0,
        "best_month": float(monthly.max()) if len(monthly) else 0.0,
        "worst_rolling_12m": float(roll12.min()) if roll12.notna().any() else 0.0,
        "total_cost_paid": float(res.costs.sum()),
        "cost_drag_annual_bps": float(res.costs.sum() / years * 10_000.0) if years > 0 else 0.0,
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
    }


def summarize(res: BacktestResult) -> str:
    m = res.metrics
    return (
        f"CAGR {m['cagr']:6.2%} | vol {m['vol_annual']:6.2%} | Sharpe {m['sharpe']:5.2f} | "
        f"maxDD {m['max_drawdown']:7.2%} | Calmar {m['calmar']:5.2f} | "
        f"in-market {m['pct_time_in_market']:5.1%} | switches {m['exposure_switches']:4d} | "
        f"cost {m['cost_drag_annual_bps']:5.1f}bps/yr"
    )
