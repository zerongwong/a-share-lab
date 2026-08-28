from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ashare_lab.analytics.adaptive_portfolio import AdaptiveRiskBudget
from ashare_lab.analytics.cycle_policy import PriceCycleState
from ashare_lab.analytics.entry_readiness import EntryPattern
from ashare_lab.analytics.indicators import enrich_indicators
from ashare_lab.analytics.levels import build_horizon_levels
from ashare_lab.services.build_midterm_portfolio import (
    _MOMENTUM_WEIGHTS,
    HOLDING_PERIOD_SESSIONS,
    CandidateAction,
    ConditionalEntryPlanKind,
    MidtermPortfolioStatus,
    _build_one_week_conditional_entry_plan,
    _build_one_week_price_observation_plan,
    _RejectedPortfolioEvaluation,
    _select_observation_portfolio,
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


def _loose_budget(*, holding_sessions: int = 60) -> AdaptiveRiskBudget:
    return AdaptiveRiskBudget(
        max_annual_downside_volatility=0.60,
        max_rolling_drawdown_60_p90=0.50,
        max_es95_5d=0.30,
        max_down_period_correlation=1.0,
        max_position_downside_risk_contribution=0.80,
        holding_period_sessions=holding_sessions,
        holding_period_cost_rate=0.0,
        minimum_observations=500,
        minimum_holding_period_samples=8,
    )


def _downtrend_repair_indices(dates: pd.Series) -> dict[str, pd.DataFrame]:
    count = len(dates)
    decline_count = count - 20
    close = np.concatenate(
        (
            np.linspace(145.0, 80.0, decline_count),
            np.linspace(80.0, 81.5, 20),
        )
    )
    return {
        code: pd.DataFrame(
            {
                "trade_date": pd.DatetimeIndex(dates),
                "close": close * scale,
                "historical_backtest_eligible": True,
                "common_cutoff_date": pd.Timestamp(dates.iloc[-1]),
            }
        )
        for code, scale in {"000001": 1.0, "000300": 1.1, "000905": 0.9}.items()
    }


def _uptrend_indices(dates: pd.Series) -> dict[str, pd.DataFrame]:
    close = np.linspace(80.0, 145.0, len(dates))
    return {
        code: pd.DataFrame(
            {
                "trade_date": pd.DatetimeIndex(dates),
                "close": close * scale,
                "historical_backtest_eligible": True,
                "common_cutoff_date": pd.Timestamp(dates.iloc[-1]),
            }
        )
        for code, scale in {"000001": 1.0, "000300": 1.1, "000905": 0.9}.items()
    }


def test_builds_one_adaptive_research_portfolio_and_orders_weights() -> None:
    histories, metadata = _universe()
    cutoff = histories[next(iter(histories))]["trade_date"].iloc[-1]
    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=13,
        market_index_histories=_uptrend_indices(histories[next(iter(histories))]["trade_date"]),
        risk_budget=_loose_budget(),
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=24,
        minimum_universe_size=3,
    )

    assert result.status == MidtermPortfolioStatus.VALIDATION_NOT_READY
    assert "nonproduction_small_universe_or_negative_lcb_override" in result.reasons
    assert 3 <= len(result.positions) <= 5
    assert result.borrowed_weight == 0.0
    assert abs(sum(item.weight for item in result.positions) + result.cash_weight - 1.0) < 1e-10
    assert (
        abs(
            sum(item.operational_account_weight or 0.0 for item in result.positions)
            + result.cash_weight
            - 1.0
        )
        < 1e-10
    )
    assert [item.operational_account_weight for item in result.positions] == sorted(
        (item.operational_account_weight for item in result.positions), reverse=True
    )
    assert all(item.entry_pattern.value != "no_signal" for item in result.positions)
    assert all(item.conditional_entry_plan is not None for item in result.positions)
    assert all(
        item.conditional_entry_plan is not None
        and item.conditional_entry_plan.data_cutoff == cutoff
        and item.conditional_entry_plan.sessions == 5
        for item in result.research_candidates
    )
    assert all(
        item.conditional_entry_plan is not None
        and item.conditional_entry_plan.kind is ConditionalEntryPlanKind.VOLUME_BREAKOUT
        and item.conditional_entry_plan.trigger_price is not None
        and item.conditional_entry_plan.trigger_price > item.breakout_line
        for item in result.research_candidates
    )
    assert result.evaluation is not None
    assert result.evaluation.risk_budget.passed
    assert result.evaluation.stock_sleeve_weight_step == 0.10
    action_sleeve = [item.operational_stock_sleeve_weight or 0.0 for item in result.positions]
    assert sum(action_sleeve) == pytest.approx(1.0)
    assert all(np.isclose(weight * 10, round(weight * 10)) for weight in action_sleeve)


def test_two_week_horizon_runs_with_ten_session_risk_contract() -> None:
    histories, metadata = _universe()
    first = histories[next(iter(histories))]
    cutoff = first["trade_date"].iloc[-1]

    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=2,
        market_index_histories=_uptrend_indices(first["trade_date"]),
        risk_budget=_loose_budget(holding_sessions=10),
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=24,
        minimum_universe_size=3,
    )

    assert result.holding_weeks == 2
    assert result.status is not MidtermPortfolioStatus.DATA_NOT_READY
    assert result.evaluation is not None
    assert result.evaluation.metrics.holding_period_sessions == 10


def test_risk_off_indices_continue_screen_and_return_research_candidates() -> None:
    histories, metadata = _universe()
    first = histories[next(iter(histories))]
    cutoff = first["trade_date"].iloc[-1]

    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=13,
        market_index_histories=_downtrend_repair_indices(first["trade_date"]),
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=24,
        minimum_universe_size=3,
    )

    assert result.status is not MidtermPortfolioStatus.DATA_NOT_READY
    assert result.price_cycle is not None
    assert result.price_cycle.state is PriceCycleState.DOWNTREND_REPAIR
    assert result.price_cycle.research_continues is True
    assert len(result.research_candidates) == 4
    assert all(
        item.action
        in {
            CandidateAction.CONDITIONAL_ENTRY,
            CandidateAction.WAIT_CONFIRMATION,
            CandidateAction.OBSERVE_ONLY,
        }
        for item in result.research_candidates
    )
    assert result.entry_ready_count >= 3
    assert result.research_evaluation is not None
    assert result.evaluated_portfolio_count > 0
    assert result.research_stock_exposure <= 0.30
    assert result.research_cash_weight >= 0.70
    assert all(item.research_weight is not None for item in result.research_candidates)
    research_sleeve = [
        item.operational_stock_sleeve_weight or 0.0 for item in result.research_candidates
    ]
    assert sum(research_sleeve) == pytest.approx(1.0)
    assert all(np.isclose(weight * 10, round(weight * 10)) for weight in research_sleeve)


def test_one_week_entry_plan_uses_horizon_levels_not_the_breakout_line() -> None:
    frame = _breakout_history(42)
    cutoff = pd.Timestamp(frame.iloc[-1]["trade_date"]).normalize()
    levels = build_horizon_levels(enrich_indicators(frame))[0]

    pullback = _build_one_week_conditional_entry_plan(
        frame,
        cutoff=cutoff,
        action=CandidateAction.CONDITIONAL_ENTRY,
        entry_pattern=EntryPattern.HEALTHY_PULLBACK,
        evidence_passed=True,
    )
    reclaim = _build_one_week_conditional_entry_plan(
        frame,
        cutoff=cutoff,
        action=CandidateAction.CONDITIONAL_ENTRY,
        entry_pattern=EntryPattern.BREAKOUT_RECLAIM,
        evidence_passed=True,
    )
    breakout = _build_one_week_conditional_entry_plan(
        frame,
        cutoff=cutoff,
        action=CandidateAction.CONDITIONAL_ENTRY,
        entry_pattern=EntryPattern.VOLUME_BREAKOUT,
        evidence_passed=True,
    )

    assert pullback is not None
    assert pullback.price_low == levels.pullback_entry_low
    assert pullback.price_high == levels.pullback_entry_high
    assert reclaim is not None
    assert reclaim.trigger_price == levels.entry_trigger
    assert breakout is not None
    assert breakout.trigger_price == levels.breakout_trigger


@pytest.mark.parametrize(
    "action",
    (CandidateAction.WAIT_CONFIRMATION, CandidateAction.OBSERVE_ONLY),
)
def test_non_actionable_candidate_has_no_entry_price_plan(action: CandidateAction) -> None:
    frame = _breakout_history(43)
    cutoff = pd.Timestamp(frame.iloc[-1]["trade_date"]).normalize()

    assert (
        _build_one_week_conditional_entry_plan(
            frame,
            cutoff=cutoff,
            action=action,
            entry_pattern=EntryPattern.VOLUME_BREAKOUT,
            evidence_passed=True,
        )
        is None
    )
    observation = _build_one_week_price_observation_plan(
        frame,
        cutoff=cutoff,
        entry_pattern=EntryPattern.VOLUME_BREAKOUT,
    )
    assert observation is not None
    assert observation.trigger_price is not None
    assert observation.confirmation_rule is not None


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
        market_index_histories=_uptrend_indices(histories[next(iter(histories))]["trade_date"]),
        risk_budget=_loose_budget(),
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
        holding_period_sessions=60,
        minimum_observations=500,
        minimum_holding_period_samples=8,
    )

    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=13,
        market_index_histories=_uptrend_indices(histories[next(iter(histories))]["trade_date"]),
        risk_budget=impossible,
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=16,
        minimum_universe_size=3,
    )

    assert result.status == MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO
    assert result.positions == ()
    assert result.stock_exposure == 0.0
    assert result.cash_weight == 1.0
    assert result.research_evaluation is None
    assert result.observation_evaluation is not None
    assert result.observation_evaluation.risk_budget.passed is False
    assert result.observation_rejection_reasons
    assert len(result.research_candidates) == 4
    observation_weights = [
        item.observation_stock_sleeve_weight or 0.0 for item in result.research_candidates
    ]
    assert sum(observation_weights) == pytest.approx(1.0)
    assert all(np.isclose(weight * 10, round(weight * 10)) for weight in observation_weights)
    assert all(item.research_weight is None for item in result.research_candidates)
    assert all(item.operational_account_weight is None for item in result.research_candidates)
    assert all(item.price_observation_plan is not None for item in result.research_candidates)


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
        market_index_histories=_uptrend_indices(histories[next(iter(histories))]["trade_date"]),
        risk_budget=_loose_budget(),
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=24,
        minimum_universe_size=3,
    )

    assert result.status == MidtermPortfolioStatus.VALIDATION_NOT_READY
    assert result.positions == ()
    assert result.stock_exposure == 0.0
    assert result.cash_weight == 1.0
    assert result.research_evaluation is not None
    assert result.evaluated_portfolio_count > 0
    assert (
        abs(
            sum(item.research_weight or 0.0 for item in result.research_candidates)
            + result.research_cash_weight
            - 1.0
        )
        < 1e-10
    )
    assert all(item.evidence_unknown for item in result.research_candidates)
    assert all(item.conditional_entry_plan is None for item in result.research_candidates)
    assert all(item.price_observation_plan is not None for item in result.research_candidates)
    assert result.evidence_review_required is True
    assert any("尚未接齐" in warning for warning in result.warnings)


def test_small_partial_universe_is_data_not_ready() -> None:
    histories, metadata = _universe()
    cutoff = histories[next(iter(histories))]["trade_date"].iloc[-1]

    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=13,
    )

    assert result.status == MidtermPortfolioStatus.DATA_NOT_READY
    assert result.positions == ()
    assert "minimum_1000_required" in result.reasons[0]


def test_core_index_evidence_cannot_be_disabled() -> None:
    histories, metadata = _universe()
    cutoff = histories[next(iter(histories))]["trade_date"].iloc[-1]

    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=13,
        minimum_universe_size=3,
    )

    assert result.status is MidtermPortfolioStatus.DATA_NOT_READY
    assert result.reasons == ("core_index_regime_required_but_missing",)


def test_main_holding_period_session_contract_is_unique() -> None:
    assert HOLDING_PERIOD_SESSIONS == {1: 5, 2: 10, 4: 20, 13: 60, 26: 120, 52: 252}
    assert _MOMENTUM_WEIGHTS[2] == ((10, 0.50), (20, 0.30), (60, 0.20))
    assert sum(weight for _sessions, weight in _MOMENTUM_WEIGHTS[2]) == pytest.approx(1.0)


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
        market_index_histories=_uptrend_indices(histories[next(iter(histories))]["trade_date"]),
        risk_budget=_loose_budget(),
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=24,
        minimum_universe_size=3,
    )

    assert result.status == MidtermPortfolioStatus.VALIDATION_NOT_READY
    assert result.positions == ()
    assert result.research_candidates
    assert all(
        "formation_execution_evidence_unknown" in item.evidence_unknown
        for item in result.research_candidates
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


def _observation_row(
    symbols: tuple[str, ...],
    *,
    violations: tuple[str, ...],
    overrun: float,
    lcb: float,
) -> _RejectedPortfolioEvaluation:
    return _RejectedPortfolioEvaluation(
        evaluation=SimpleNamespace(metrics=SimpleNamespace(holding_period_return_lcb=lcb)),
        selected=tuple(SimpleNamespace(symbol=symbol) for symbol in symbols),
        rejection_reasons=violations,
        normalized_overrun=overrun,
    )


def test_observation_selection_prefers_four_then_violations_overrun_lcb_and_code() -> None:
    three = _observation_row(
        ("1", "2", "3"),
        violations=("risk",),
        overrun=0.01,
        lcb=0.20,
    )
    four_many = _observation_row(
        ("1", "2", "3", "9"),
        violations=("risk", "lcb"),
        overrun=0.01,
        lcb=0.20,
    )
    four_large = _observation_row(
        ("1", "2", "3", "8"),
        violations=("risk",),
        overrun=0.20,
        lcb=0.30,
    )
    four_lower_lcb = _observation_row(
        ("1", "2", "3", "7"),
        violations=("risk",),
        overrun=0.10,
        lcb=0.05,
    )
    four_code_later = _observation_row(
        ("1", "2", "4", "6"),
        violations=("risk",),
        overrun=0.10,
        lcb=0.10,
    )
    four_chosen = _observation_row(
        ("1", "2", "3", "6"),
        violations=("risk",),
        overrun=0.10,
        lcb=0.10,
    )

    chosen = _select_observation_portfolio(
        {
            3: [three],
            4: [four_many, four_large, four_lower_lcb, four_code_later, four_chosen],
            5: [],
        }
    )

    assert chosen is four_chosen
