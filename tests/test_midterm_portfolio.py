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
from ashare_lab.analytics.multi_timeframe import (
    MULTI_TIMEFRAME_IMPLEMENTATION_STATUS,
    ExecutionState,
    StructureState,
    assess_multi_timeframe,
)
from ashare_lab.services.build_midterm_portfolio import (
    CENTRAL_MULTI_TIMEFRAME_IMPLEMENTATION_STATUS,
    HOLDING_PERIOD_SESSIONS,
    CandidateAction,
    ConditionalEntryPlanKind,
    MidtermPortfolioStatus,
    _build_horizon_price_observation_plan,
    _build_one_week_conditional_entry_plan,
    _build_one_week_price_observation_plan,
    _holding_risk_history_exclusion_reasons,
    _observation_entry_pattern,
    _RejectedPortfolioEvaluation,
    _select_observation_portfolio,
    _select_stock_count,
    _viable_sort_key,
    build_midterm_portfolio,
    horizon_history_requirements,
)


def test_observation_pattern_never_falls_back_to_legacy_sixty_day_entry() -> None:
    candidate = SimpleNamespace(
        timeframe=SimpleNamespace(
            execution=SimpleNamespace(state=ExecutionState.WAIT_CONFIRMATION),
            structure=SimpleNamespace(state=StructureState.NEAR_BREAKOUT),
        ),
        entry=SimpleNamespace(pattern=EntryPattern.HEALTHY_PULLBACK),
    )

    assert _observation_entry_pattern(candidate) is EntryPattern.VOLUME_BREAKOUT


def test_horizon_plan_separates_daily_entry_from_primary_structure_risk_line() -> None:
    frame = _breakout_history(77)
    cutoff = pd.Timestamp(frame.iloc[-1]["trade_date"]).normalize()
    timeframe = assess_multi_timeframe(
        frame,
        as_of=cutoff,
        holding_weeks=13,
    )

    plan = _build_horizon_price_observation_plan(
        frame,
        cutoff=cutoff,
        holding_weeks=13,
        timeframe=timeframe,
        entry_pattern=EntryPattern.HEALTHY_PULLBACK,
    )

    assert plan is not None
    assert plan.price_source_timeframe == "daily"
    assert plan.primary_structure_timeframe == "weekly_completed"
    assert plan.invalidation_source_timeframe == "weekly_completed"
    assert plan.reduction_review_source_timeframe == "weekly_completed"
    assert plan.entry_reference_price == pytest.approx(timeframe.execution.breakout_line)
    assert plan.primary_structure_reference_price == pytest.approx(
        timeframe.structure.breakout_line
    )
    assert "日线执行线" in (plan.confirmation_rule or "")


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
    assert (
        result.central_implementation_status
        == CENTRAL_MULTI_TIMEFRAME_IMPLEMENTATION_STATUS
        == "partial_multiframe"
    )
    assert (
        result.multi_timeframe_component_status
        == MULTI_TIMEFRAME_IMPLEMENTATION_STATUS
        == "analytics_core_only"
    )
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
        and item.conditional_entry_plan.sessions == 60
        and item.conditional_entry_plan.horizon == "三个月"
        and item.conditional_entry_plan.structure_timeframe == "weekly_completed"
        for item in result.research_candidates
    )
    for item in result.research_candidates:
        expected_activity = float(
            pd.to_numeric(histories[item.symbol]["amount_cny"], errors="coerce").tail(20).median()
        )
        assert item.conditional_entry_plan is not None
        assert item.conditional_entry_plan.confirmation_activity_metric == "amount_cny"
        assert item.conditional_entry_plan.confirmation_activity_min == pytest.approx(
            1.20 * expected_activity
        )
    assert all(
        item.conditional_entry_plan is not None
        and item.conditional_entry_plan.kind is ConditionalEntryPlanKind.VOLUME_BREAKOUT
        and item.conditional_entry_plan.trigger_price is not None
        and item.conditional_entry_plan.confirmation_activity_min is not None
        and item.conditional_entry_plan.confirmation_activity_min > 0.0
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


def test_one_year_short_risk_history_is_audited_per_stock_not_global_data_failure() -> None:
    histories, metadata = _universe()
    for index, symbol in enumerate(histories):
        histories[symbol] = _breakout_history(index + 101, periods=2087)
    short_symbol = next(iter(histories))
    histories[short_symbol] = histories[short_symbol].tail(620).reset_index(drop=True)
    first_long = next(frame for symbol, frame in histories.items() if symbol != short_symbol)
    cutoff = first_long["trade_date"].iloc[-1]

    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=52,
        market_index_histories=_uptrend_indices(first_long["trade_date"]),
        risk_budget=_loose_budget(holding_sessions=252),
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=8,
        minimum_universe_size=3,
    )

    short_candidate = next(
        item for item in result.research_candidates if item.symbol == short_symbol
    )
    assert result.status is not MidtermPortfolioStatus.DATA_NOT_READY
    assert result.horizon_candidate_count >= 4
    assert result.risk_history_eligible_candidate_count == (result.horizon_candidate_count - 1)
    assert result.risk_history_ineligible_candidate_count == 1
    assert 3 <= len(result.research_candidates) <= 5
    assert short_candidate.risk_history_available is False
    assert short_candidate.risk_history_available_returns == 619
    assert short_candidate.risk_history_required_returns == 2016
    assert short_candidate.research_weight is None
    assert short_candidate.operational_account_weight is None
    assert short_candidate.operational_stock_sleeve_weight is None
    assert short_candidate.conditional_entry_plan is None
    assert short_candidate.action is CandidateAction.WAIT_CONFIRMATION
    assert "risk_history_unavailable_for_portfolio_weighting" in (short_candidate.action_reasons)
    assert any(
        reason.startswith(
            "insufficient_holding_risk_history:available_619_returns;"
            "required_2016_returns;holding_252_sessions;"
        )
        for reason in short_candidate.risk_history_reasons
    )
    assert short_symbol not in {position.symbol for position in result.positions}
    assert result.research_evaluation is not None
    assert short_symbol not in {
        position.symbol for position in result.research_evaluation.positions
    }


def test_one_year_all_short_risk_histories_return_no_portfolio_not_data_not_ready() -> None:
    histories, metadata = _universe()
    first = histories[next(iter(histories))]
    cutoff = first["trade_date"].iloc[-1]

    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=52,
        market_index_histories=_uptrend_indices(first["trade_date"]),
        risk_budget=_loose_budget(holding_sessions=252),
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=8,
        minimum_universe_size=3,
    )

    assert result.status is MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO
    assert result.status is not MidtermPortfolioStatus.DATA_NOT_READY
    assert result.horizon_candidate_count >= 3
    assert result.risk_history_eligible_candidate_count == 0
    assert result.risk_history_ineligible_candidate_count == result.horizon_candidate_count
    assert result.research_candidates
    assert all(not item.risk_history_available for item in result.research_candidates)
    assert all(item.research_weight is None for item in result.research_candidates)
    assert all(item.operational_account_weight is None for item in result.research_candidates)
    assert all(item.conditional_entry_plan is None for item in result.research_candidates)


def test_holding_risk_history_requirement_keeps_all_adaptive_sample_gates() -> None:
    dates = pd.bdate_range("2020-01-02", periods=2017)
    full = pd.Series(np.full(2016, 0.001), index=dates[1:])
    short = full.iloc[:-1]
    budget = _loose_budget(holding_sessions=252)

    assert _holding_risk_history_exclusion_reasons(full, budget) == ()
    assert _holding_risk_history_exclusion_reasons(short, budget) == (
        "insufficient_holding_risk_history:available_2015_returns;"
        "required_2016_returns;holding_252_sessions;"
        "minimum_8_nonoverlapping_samples",
    )


@pytest.mark.parametrize(
    ("holding_weeks", "slow_timeframe", "primary_timeframe", "plan_sessions"),
    (
        (4, "weekly_completed", "daily", 20),
        (13, "monthly_completed", "weekly_completed", 60),
        (26, "monthly_completed", "weekly_completed", 120),
    ),
)
def test_service_exposes_the_selected_horizons_independent_timeframe_contract(
    holding_weeks: int,
    slow_timeframe: str,
    primary_timeframe: str,
    plan_sessions: int,
) -> None:
    histories, metadata = _universe()
    required_prices = max(620, plan_sessions * 8 + 1)
    if required_prices > 620:
        for index, symbol in enumerate(histories):
            histories[symbol] = _breakout_history(index + 201, periods=required_prices)
    first = histories[next(iter(histories))]
    cutoff = first["trade_date"].iloc[-1]

    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=holding_weeks,
        market_index_histories=_uptrend_indices(first["trade_date"]),
        risk_budget=_loose_budget(holding_sessions=HOLDING_PERIOD_SESSIONS[holding_weeks]),
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=24,
        minimum_universe_size=3,
    )

    assert result.research_candidates
    for candidate in result.research_candidates:
        assert candidate.timeframe is not None
        assert candidate.timeframe.slow_direction.timeframe.value == slow_timeframe
        assert candidate.timeframe.structure.timeframe.value == primary_timeframe
        assert candidate.price_observation_plan is not None
        assert candidate.price_observation_plan.sessions == plan_sessions
        assert candidate.price_observation_plan.structure_timeframe == primary_timeframe


def test_horizon_execution_is_not_blocked_by_the_legacy_fixed_60_day_gate() -> None:
    histories, metadata = _universe()
    first = histories[next(iter(histories))]
    cutoff = first["trade_date"].iloc[-1]
    for frame in histories.values():
        # Multi-timeframe execution accepts >=1.05 activity confirmation while
        # the legacy daily execution check retains its stricter >=1.10 gate.
        # The selected horizon is execution-ready.  The legacy 60-session
        # checker remains below its old 1.10 gate, but it is audit-only and
        # must not block the new horizon action layer.
        frame.loc[frame.index[-1], "amount_cny"] = 107_000_000.0

    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=13,
        market_index_histories=_uptrend_indices(first["trade_date"]),
        risk_budget=_loose_budget(),
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=24,
        minimum_universe_size=3,
    )

    assert result.horizon_candidate_count >= 3
    assert result.entry_ready_count >= 3
    assert result.research_candidates
    assert all(
        candidate.action is CandidateAction.CONDITIONAL_ENTRY
        for candidate in result.research_candidates
    )
    assert result.positions
    assert result.stock_exposure > 0.0


def test_late_stage_vertical_acceleration_is_a_shared_candidate_hard_veto() -> None:
    histories, metadata = _universe()
    symbol = next(iter(histories))
    frame = histories[symbol]
    base = float(frame.loc[frame.index[-21], "close"])
    accelerated = np.linspace(base * 1.02, base * 1.65, 20)
    frame.loc[frame.index[-20:], "close"] = accelerated
    frame.loc[frame.index[-20:], "open"] = accelerated * 0.995
    frame.loc[frame.index[-20:], "high"] = accelerated * 1.006
    frame.loc[frame.index[-20:], "low"] = accelerated * 0.989
    frame.loc[frame.index[-1], "amount_cny"] = 180_000_000.0
    cutoff = frame["trade_date"].iloc[-1]

    result = build_midterm_portfolio(
        histories,
        metadata,
        as_of=cutoff,
        holding_weeks=13,
        market_index_histories=_uptrend_indices(frame["trade_date"]),
        risk_budget=_loose_budget(),
        minimum_historical_return_lcb=-1.0,
        candidate_pool_size=8,
        beam_width=24,
        minimum_universe_size=3,
    )

    exclusion = next(item for item in result.exclusions if item.symbol == symbol)
    assert "late_stage_acceleration_hard_freeze" in exclusion.reasons
    assert symbol not in {item.symbol for item in result.research_candidates}


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
    assert "horizon_13_week_structure_not_qualified" in reasons["600000.SH"]
    assert any(reason.startswith("slow_monthly_completed:down") for reason in reasons["600000.SH"])
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


@pytest.mark.parametrize(
    ("holding_weeks", "qualification", "risk_prices", "read_sessions"),
    (
        (1, 252, 252, 322),
        (2, 252, 252, 322),
        (4, 252, 252, 322),
        (13, 252, 481, 551),
        (26, 280, 961, 1031),
        (52, 540, 2017, 2087),
    ),
)
def test_central_horizon_history_contract_separates_qualification_and_risk_depth(
    holding_weeks: int,
    qualification: int,
    risk_prices: int,
    read_sessions: int,
) -> None:
    requirement = horizon_history_requirements(holding_weeks)

    assert requirement.qualification_minimum_sessions == qualification
    assert requirement.risk_minimum_price_sessions == risk_prices
    assert requirement.history_read_sessions == read_sessions


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
    horizon_drawdown: float = 0.08,
    legacy_drawdown_60: float = 0.08,
):
    metrics = SimpleNamespace(
        holding_period_return_lcb=lcb,
        annual_downside_volatility=0.10,
        rolling_max_drawdown_60_p90=legacy_drawdown_60,
        horizon_rolling_max_drawdown_p90=horizon_drawdown,
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


def test_final_portfolio_order_uses_horizon_drawdown_not_legacy_60_day_value() -> None:
    first = _selection_row(
        4,
        lcb=0.05,
        correlation=0.60,
        contribution=0.32,
        horizon_drawdown=0.09,
        legacy_drawdown_60=0.01,
    )
    second = _selection_row(
        4,
        lcb=0.05,
        correlation=0.60,
        contribution=0.32,
        horizon_drawdown=0.05,
        legacy_drawdown_60=0.50,
    )

    ordered = sorted((first, second), key=_viable_sort_key)

    assert ordered[0] is second


def test_five_stock_diversification_uses_horizon_drawdown_tolerance() -> None:
    four = _selection_row(
        4,
        lcb=0.05,
        correlation=0.60,
        contribution=0.32,
        horizon_drawdown=0.08,
        legacy_drawdown_60=0.01,
    )
    five = _selection_row(
        5,
        lcb=0.06,
        correlation=0.54,
        contribution=0.28,
        horizon_drawdown=0.084,
        legacy_drawdown_60=0.90,
    )

    chosen = _select_stock_count({3: [], 4: [four], 5: [five]})

    assert chosen is five


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
