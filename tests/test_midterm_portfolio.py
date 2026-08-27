from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from ashare_lab.analytics.adaptive_portfolio import AdaptiveRiskBudget
from ashare_lab.services.build_midterm_portfolio import (
    MidtermPortfolioStatus,
    _select_stock_count,
    build_midterm_portfolio,
)


def _breakout_history(seed: int, *, periods: int = 620, downtrend: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=periods)
    if downtrend:
        close = np.linspace(100.0, 70.0, periods)
    else:
        # Keep the synthetic history in a clear, non-parabolic ordered trend;
        # randomness remains large enough to exercise downside-risk metrics.
        returns = 0.00075 + rng.normal(0.0, 0.002, periods)
        close = 80.0 * np.cumprod(1.0 + returns)
        # A calm platform followed by a fresh, turnover-confirmed close breakout.
        base = float(close[-22])
        close[-21:-1] = base * (1.0 + np.linspace(-0.008, 0.008, 20))
        close[-1] = float(close[-21:-1].max() * 1.018)
    open_price = close * (1.0 + rng.normal(0.0, 0.001, periods))
    high = np.maximum(open_price, close) * 1.006
    low = np.minimum(open_price, close) * 0.994
    amount = np.full(periods, 100_000_000.0)
    amount[-1] = 140_000_000.0
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "amount_cny": amount,
        }
    )


def _universe() -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    industries = ("医药", "电力", "软件", "消费", "机械", "材料", "银行", "交通")
    histories: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, object]] = {}
    for index, industry in enumerate(industries):
        symbol = f"{600000 + index}.SH"
        histories[symbol] = _breakout_history(index + 1)
        metadata[symbol] = {
            "name": f"合成{index}",
            "industry": industry,
            "is_st": False,
            "is_delisting": False,
            "is_suspended": False,
            "is_limit_up_at_cutoff": False,
            "is_buyable_at_cutoff": True,
            "fundamental_gate": "pass",
            "announcement_gate": "pass",
            "balance_sheet_strength_score": 0.60,
        }
    return histories, metadata


def _loose_budget() -> AdaptiveRiskBudget:
    return AdaptiveRiskBudget(
        max_annual_downside_volatility=0.60,
        max_rolling_drawdown_60_p90=0.50,
        max_es95_5d=0.30,
        max_down_period_correlation=1.0,
        max_position_downside_risk_contribution=0.80,
        holding_period_sessions=65,
        holding_period_cost_rate=0.0,
        minimum_observations=500,
        minimum_holding_period_samples=8,
    )


def test_builds_one_adaptive_research_portfolio_and_orders_weights() -> None:
    histories, metadata = _universe()
    cutoff = histories[next(iter(histories))]["trade_date"].iloc[-1]
    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=13,
        risk_budget=_loose_budget(),
        require_index_regime=False,
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=24,
        minimum_universe_size=3,
    )

    assert result.status == MidtermPortfolioStatus.RESEARCH_ONLY
    assert 3 <= len(result.positions) <= 5
    assert result.borrowed_weight == 0.0
    assert abs(sum(item.weight for item in result.positions) + result.cash_weight - 1.0) < 1e-10
    assert [item.weight for item in result.positions] == sorted(
        (item.weight for item in result.positions), reverse=True
    )
    assert all(item.entry_pattern.value != "no_signal" for item in result.positions)
    assert result.evaluation is not None
    assert result.evaluation.risk_budget.passed


def test_downtrend_and_fundamental_veto_cannot_reenter_through_score() -> None:
    histories, metadata = _universe()
    cutoff = histories[next(iter(histories))]["trade_date"].iloc[-1]
    histories["600000.SH"] = _breakout_history(100, downtrend=True)
    metadata["600001.SH"]["fundamental_gate"] = "veto"

    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=13,
        risk_budget=_loose_budget(),
        require_index_regime=False,
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=24,
        minimum_universe_size=3,
    )

    selected = {item.symbol for item in result.positions}
    assert "600000.SH" not in selected
    assert "600001.SH" not in selected
    reasons = {item.symbol: item.reasons for item in result.exclusions}
    assert any(reason.startswith("stage_not_entry_ready") for reason in reasons["600000.SH"])
    assert "fundamental_veto" in reasons["600001.SH"]


def test_risk_budget_failure_returns_cash_instead_of_weak_names() -> None:
    histories, metadata = _universe()
    cutoff = histories[next(iter(histories))]["trade_date"].iloc[-1]
    impossible = AdaptiveRiskBudget(
        max_annual_downside_volatility=0.000001,
        max_rolling_drawdown_60_p90=0.000001,
        max_es95_5d=0.000001,
        max_down_period_correlation=-0.99,
        max_position_downside_risk_contribution=0.21,
        holding_period_sessions=65,
        minimum_observations=500,
        minimum_holding_period_samples=8,
    )

    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=13,
        risk_budget=impossible,
        require_index_regime=False,
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=16,
        minimum_universe_size=3,
    )

    assert result.status == MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO
    assert result.positions == ()
    assert result.cash_weight == 1.0


def test_unknown_fundamental_and_announcement_evidence_is_not_filled_neutral() -> None:
    histories, metadata = _universe()
    cutoff = histories[next(iter(histories))]["trade_date"].iloc[-1]
    for item in metadata.values():
        item.pop("fundamental_gate")
        item.pop("announcement_gate")

    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=13,
        risk_budget=_loose_budget(),
        require_index_regime=False,
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=24,
        minimum_universe_size=3,
    )

    assert result.status == MidtermPortfolioStatus.VALIDATION_NOT_READY
    assert all(item.evidence_unknown for item in result.positions)
    assert result.evidence_review_required is True
    assert any("待补充" in warning for warning in result.warnings)


def test_small_partial_universe_is_data_not_ready() -> None:
    histories, metadata = _universe()
    cutoff = histories[next(iter(histories))]["trade_date"].iloc[-1]

    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=13,
        require_index_regime=False,
    )

    assert result.status == MidtermPortfolioStatus.DATA_NOT_READY
    assert result.positions == ()
    assert "minimum_1000_required" in result.reasons[0]


def test_unknown_execution_eligibility_produces_provisional_positions() -> None:
    histories, metadata = _universe()
    cutoff = histories[next(iter(histories))]["trade_date"].iloc[-1]
    for item in metadata.values():
        item["is_buyable_at_cutoff"] = None

    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=13,
        risk_budget=_loose_budget(),
        require_index_regime=False,
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=24,
        minimum_universe_size=3,
    )

    assert result.status == MidtermPortfolioStatus.VALIDATION_NOT_READY
    assert result.positions
    assert all(
        "formation_execution_evidence_unknown" in item.evidence_unknown
        for item in result.positions
    )


def _selection_row(
    stock_count: int,
    *,
    lcb: float,
    correlation: float,
    contribution: float,
):
    metrics = SimpleNamespace(
        holding_period_return_lcb=lcb,
        annual_downside_volatility=0.10,
        rolling_max_drawdown_60_p90=0.08,
        es95_5d=0.03,
        max_down_period_correlation=correlation,
        max_position_downside_risk_contribution=contribution,
    )
    evaluation = SimpleNamespace(metrics=metrics)
    selected = tuple(SimpleNamespace(symbol=str(index)) for index in range(stock_count))
    return (lcb, evaluation, selected)


def test_five_stocks_replace_four_only_for_material_diversification() -> None:
    four = _selection_row(4, lcb=0.05, correlation=0.60, contribution=0.32)
    weak_five = _selection_row(5, lcb=0.06, correlation=0.58, contribution=0.30)
    strong_five = _selection_row(5, lcb=0.06, correlation=0.54, contribution=0.28)

    chosen = _select_stock_count({3: [], 4: [four], 5: [weak_five]})
    assert chosen is four

    chosen = _select_stock_count({3: [], 4: [four], 5: [strong_five]})
    assert chosen is strong_five
