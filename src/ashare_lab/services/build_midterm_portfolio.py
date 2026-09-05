"""Build one adaptive 3--5 stock medium-term A-share research portfolio.

This is the main strategy service for the post-prototype model.  It keeps the
legacy fixed-four builder intact for archive compatibility, but it does not use
its composite factor score or fixed 3:2:2:1 allocation.

The service is deliberately deterministic and separates opportunity research
from current entry permission:

* market evidence changes exposure, risk budgets, and entry strictness but
  never stops an otherwise data-complete full-market screen;
* price/turnover structure is a hard candidate gate;
* fundamentals and official announcements can veto, never boost ranking;
* every 3-, 4-, and 5-stock set is evaluated with the same downside-risk
  engine; a bounded beam is a documented computational approximation;
* only risk-budget-passing sets with a positive historical holding-period lower
  bound may be returned, and the result remains ``RESEARCH_ONLY`` until genuine
  point-in-time walk-forward validation exists.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from ashare_lab.analytics.adaptive_portfolio import (
    FIVE_DAY_SESSIONS,
    MINIMUM_FIVE_DAY_SAMPLES,
    MINIMUM_ROLLING_DRAWDOWN_WINDOWS,
    ROLLING_DRAWDOWN_SESSIONS,
    AdaptiveCandidate,
    AdaptivePortfolioDataError,
    AdaptivePortfolioEvaluation,
    AdaptiveRiskBudget,
    optimize_adaptive_portfolio,
)
from ashare_lab.analytics.cycle_policy import (
    EntryStrictness,
    PriceCycleAssessment,
    PriceCycleState,
    assess_price_cycle,
)
from ashare_lab.analytics.entry_readiness import (
    EntryPattern,
    EntryReadinessAssessment,
    assess_entry_readiness,
)
from ashare_lab.analytics.index_regime import (
    IndexRegimeAssessment,
    IndexRegimeState,
    assess_index_regime,
)
from ashare_lab.analytics.indicators import enrich_indicators
from ashare_lab.analytics.levels import build_horizon_levels
from ashare_lab.analytics.market_regime import (
    MarketRegimeAssessment,
    MarketRegimeState,
    assess_market_regime,
)
from ashare_lab.analytics.medium_term_stage import assess_medium_term_stage
from ashare_lab.analytics.multi_timeframe import (
    MULTI_TIMEFRAME_IMPLEMENTATION_STATUS,
    MULTI_TIMEFRAME_METHOD_VERSION,
    ExecutionState,
    MultiTimeframeAssessment,
    MultiTimeframeDataError,
    StructureState,
    assess_multi_timeframe,
    build_completed_timeframes,
    horizon_contract,
)
from ashare_lab.services.review_active_holdings import HOLDING_TREE_METHOD_VERSION, _candidate_stop

RESEARCH_DISCLAIMER = (
    "本结果是共同截止日历史数据上的确定性研究筛选。持有期收益下界尚未经过严格的"
    "point-in-time walk-forward、费用、滑点和不可成交验证，不是未来收益预测、上涨概率、"
    "投资建议或最大回撤保证。"
)
MIDTERM_METHOD_VERSION = "midterm-maintrend-multitimeframe-v0.8.0"
CENTRAL_MULTI_TIMEFRAME_IMPLEMENTATION_STATUS = "partial_multiframe"
MAX_INITIAL_ENTRY_RISK = 0.08


class MidtermPortfolioStatus(StrEnum):
    DATA_NOT_READY = "data_not_ready"
    VALIDATION_NOT_READY = "validation_not_ready"
    NO_ELIGIBLE_PORTFOLIO = "no_eligible_portfolio"
    RESEARCH_ONLY = "research_only"


class CandidateAction(StrEnum):
    """Current action state; a research candidate is not automatically buyable."""

    CONDITIONAL_ENTRY = "conditional_entry"
    WAIT_CONFIRMATION = "wait_confirmation"
    OBSERVE_ONLY = "observe_only"


class ConditionalEntryPlanKind(StrEnum):
    """Horizon-aware daily execution level; never an unconditional buy price."""

    HEALTHY_PULLBACK = "healthy_pullback_range"
    RECLAIM = "reclaim_close_confirmation"
    VOLUME_BREAKOUT = "volume_breakout_close_confirmation"


@dataclass(frozen=True, slots=True)
class ConditionalEntryPlan:
    """Structured horizon-aware daily execution plan at the common cutoff.

    The containing field determines its role.  ``price_observation_plan`` is
    diagnostic only, while ``conditional_entry_plan`` additionally requires
    the candidate-level evidence and entry gates to pass.
    """

    kind: ConditionalEntryPlanKind
    data_cutoff: pd.Timestamp
    horizon: str
    sessions: int
    price_low: float | None = None
    price_high: float | None = None
    trigger_price: float | None = None
    confirmation_rule: str | None = None
    structure_timeframe: str | None = None
    structure_cutoff: pd.Timestamp | None = None
    execution_lookback_sessions: int | None = None
    method_version: str = MULTI_TIMEFRAME_METHOD_VERSION
    price_source_timeframe: str | None = None
    primary_structure_timeframe: str | None = None
    primary_structure_cutoff: pd.Timestamp | None = None
    invalidation_price: float | None = None
    reduction_review_price: float | None = None
    entry_reference_price: float | None = None
    primary_structure_reference_price: float | None = None
    invalidation_source_timeframe: str | None = None
    reduction_review_source_timeframe: str | None = None
    confirmation_activity_metric: str | None = None
    confirmation_activity_min: float | None = None
    # New plans assess risk from their prospective buy price, never today's
    # close.  Missing values on legacy archived plans remain unavailable.
    initial_risk_reference_price: float | None = None
    initial_risk_fraction: float | None = None
    maximum_entry_price: float | None = None
    initial_risk_qualified: bool | None = None
    initial_risk_reason: str | None = None
    initial_protection_support: float | None = None
    initial_protection_atr: float | None = None
    initial_protection_evidence_date: str | None = None
    initial_protection_atr_cutoff: str | None = None
    initial_protection_method_version: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateExclusion:
    symbol: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MidtermCandidate:
    symbol: str
    name: str
    industry: str
    signal_score: float
    entry: EntryReadinessAssessment
    returns: pd.Series
    evidence_unknown: tuple[str, ...]
    absolute_return_60: float
    relative_strength_percentile: float
    downside_capture_ratio: float | None
    ma20_above_ma60: bool
    timeframe: MultiTimeframeAssessment
    horizon_absolute_return: float
    relative_strength_sessions: int
    # Appended for positional compatibility.  Structural qualification and
    # deep-history risk eligibility are deliberately separate states.
    risk_history_available: bool = True
    risk_history_reasons: tuple[str, ...] = ()
    risk_history_available_returns: int | None = None
    risk_history_required_returns: int | None = None
    price_observation_plan: ConditionalEntryPlan | None = None


@dataclass(frozen=True, slots=True)
class MidtermResearchCandidate:
    rank: int
    symbol: str
    name: str
    industry: str
    signal_score: float
    entry_pattern: EntryPattern
    breakout_line: float | None
    days_since_breakout: int | None
    action: CandidateAction
    action_reasons: tuple[str, ...]
    absolute_return_60: float
    relative_strength_percentile: float
    downside_capture_ratio: float | None
    evidence_unknown: tuple[str, ...]
    research_weight: float | None = None
    downside_risk_contribution: float | None = None
    operational_account_weight: float | None = None
    operational_stock_sleeve_weight: float | None = None
    conditional_entry_plan: ConditionalEntryPlan | None = None
    observation_stock_sleeve_weight: float | None = None
    price_observation_plan: ConditionalEntryPlan | None = None
    timeframe: MultiTimeframeAssessment | None = None
    horizon_absolute_return: float | None = None
    relative_strength_sessions: int | None = None
    risk_history_available: bool = True
    risk_history_reasons: tuple[str, ...] = ()
    risk_history_available_returns: int | None = None
    risk_history_required_returns: int | None = None


@dataclass(frozen=True, slots=True)
class MidtermSelectedPosition:
    rank: int
    symbol: str
    name: str
    industry: str
    weight: float
    signal_score: float
    entry_pattern: EntryPattern
    breakout_line: float
    days_since_breakout: int
    annual_downside_volatility: float
    downside_risk_contribution: float
    evidence_unknown: tuple[str, ...]
    operational_account_weight: float | None = None
    operational_stock_sleeve_weight: float | None = None
    conditional_entry_plan: ConditionalEntryPlan | None = None
    price_observation_plan: ConditionalEntryPlan | None = None
    timeframe: MultiTimeframeAssessment | None = None
    horizon_absolute_return: float | None = None
    relative_strength_sessions: int | None = None


@dataclass(frozen=True, slots=True)
class MidtermPortfolioResult:
    status: MidtermPortfolioStatus
    data_cutoff: pd.Timestamp | None
    holding_weeks: int
    positions: tuple[MidtermSelectedPosition, ...] = ()
    stock_exposure: float = 0.0
    cash_weight: float = 1.0
    borrowed_weight: float = 0.0
    evaluation: AdaptivePortfolioEvaluation | None = None
    entry_ready_count: int = 0
    search_pool_count: int = 0
    evaluated_portfolio_count: int = 0
    exclusions: tuple[CandidateExclusion, ...] = ()
    market_regime: MarketRegimeAssessment | None = None
    index_regime: IndexRegimeAssessment | None = None
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence_review_required: bool = False
    method_version: str = MIDTERM_METHOD_VERSION
    disclaimer: str = RESEARCH_DISCLAIMER
    # Appended fields preserve the positional layout of the original result
    # object while exposing the candidate and deployment layers separately.
    research_candidates: tuple[MidtermResearchCandidate, ...] = ()
    actionable_candidate_count: int = 0
    price_cycle: PriceCycleAssessment | None = None
    research_evaluation: AdaptivePortfolioEvaluation | None = None
    research_stock_exposure: float = 0.0
    research_cash_weight: float = 1.0
    action_evaluated_portfolio_count: int = 0
    observation_evaluation: AdaptivePortfolioEvaluation | None = None
    observation_rejection_reasons: tuple[str, ...] = ()
    multi_timeframe_method_version: str = MULTI_TIMEFRAME_METHOD_VERSION
    horizon_candidate_count: int = 0
    risk_history_eligible_candidate_count: int = 0
    risk_history_ineligible_candidate_count: int = 0
    # Repository-level truth is separate from the analytics component marker.
    # The central contract remains partial until the immutable six-horizon
    # archive, official-calendar boundaries and validation path are complete.
    central_implementation_status: str = CENTRAL_MULTI_TIMEFRAME_IMPLEMENTATION_STATUS
    multi_timeframe_component_status: str = MULTI_TIMEFRAME_IMPLEMENTATION_STATUS


@dataclass(frozen=True, slots=True)
class _CandidateInput:
    symbol: str
    name: str
    industry: str
    entry: EntryReadinessAssessment
    timeframe: MultiTimeframeAssessment
    returns: pd.Series
    evidence_unknown: tuple[str, ...]
    risk_history_available: bool
    risk_history_reasons: tuple[str, ...]
    risk_history_available_returns: int
    risk_history_required_returns: int


@dataclass(frozen=True, slots=True)
class _RejectedPortfolioEvaluation:
    """One structurally evaluable allocation that failed a research gate."""

    evaluation: AdaptivePortfolioEvaluation
    selected: tuple[MidtermCandidate, ...]
    rejection_reasons: tuple[str, ...]
    normalized_overrun: float


# One shared trading-session convention for the main strategy.  Month/quarter/
# half-year/year horizons follow the research contract rather than multiplying
# calendar weeks by five and silently turning 13 weeks into 65 sessions.
HOLDING_PERIOD_SESSIONS: dict[int, int] = {
    1: 5,
    2: 10,
    4: 20,
    13: 60,
    26: 120,
    52: 252,
}

_HORIZON_PLAN_LABELS: dict[int, str] = {
    1: "一周",
    2: "两周",
    4: "一个月",
    13: "三个月",
    26: "六个月",
    52: "一年",
}


@dataclass(frozen=True, slots=True)
class HorizonHistoryRequirements:
    """Separate full-market qualification from finalist risk history depth."""

    holding_weeks: int
    holding_period_sessions: int
    qualification_minimum_sessions: int
    risk_minimum_price_sessions: int
    history_read_sessions: int


def horizon_history_requirements(holding_weeks: int) -> HorizonHistoryRequirements:
    """Return one central, no-lookahead history contract for a holding horizon.

    Qualification uses only the selected multi-timeframe core plus a 252-price
    baseline.  Eight non-overlapping holding samples remain mandatory for the
    downstream LCB/risk engine, while a 70-session read buffer absorbs calendar
    and alignment differences without treating that deeper history as a
    full-market coverage gate.
    """

    if holding_weeks not in HOLDING_PERIOD_SESSIONS:
        raise ValueError("holding_weeks must be one of 1, 2, 4, 13, 26 or 52")
    holding_sessions = HOLDING_PERIOD_SESSIONS[holding_weeks]
    contract = horizon_contract(holding_weeks)
    qualification = max(252, contract.minimum_daily_sessions)
    risk_minimum = max(qualification, holding_sessions * 8 + 1)
    history_read = max(qualification + 70, holding_sessions * 8 + 71)
    return HorizonHistoryRequirements(
        holding_weeks=holding_weeks,
        holding_period_sessions=holding_sessions,
        qualification_minimum_sessions=qualification,
        risk_minimum_price_sessions=risk_minimum,
        history_read_sessions=history_read,
    )


def build_midterm_portfolio(
    histories: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    as_of: object,
    holding_weeks: int = 13,
    market_index_histories: Mapping[str, pd.DataFrame] | None = None,
    risk_budget: AdaptiveRiskBudget | None = None,
    candidate_pool_size: int = 36,
    beam_width: int = 128,
    minimum_universe_size: int = 1_000,
    minimum_balance_sheet_strength: float = 0.10,
    minimum_historical_return_lcb: float = 0.0,
) -> MidtermPortfolioResult:
    """Return one risk-budget-passing 3--5 stock research portfolio.

    Optional metadata keys ``fundamental_gate`` and ``announcement_gate`` use
    ``pass``, ``veto`` or ``unknown``.  The boolean aliases
    ``fundamental_veto`` and ``announcement_veto`` are also accepted.  Vetoes
    exclude before ranking.  Missing evidence is recorded on the finalist and
    never translated into a neutral score.
    """

    if holding_weeks not in HOLDING_PERIOD_SESSIONS:
        raise ValueError("holding_weeks must be one of 1, 2, 4, 13, 26 or 52")
    if candidate_pool_size < 5:
        raise ValueError("candidate_pool_size must be at least five")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if minimum_universe_size < 3:
        raise ValueError("minimum_universe_size must be at least three")
    if not 0.0 <= minimum_balance_sheet_strength <= 1.0:
        raise ValueError("minimum_balance_sheet_strength must be in [0, 1]")
    if not math.isfinite(minimum_historical_return_lcb):
        raise ValueError("minimum_historical_return_lcb must be finite")

    cutoff = _normalize_cutoff(as_of)
    if cutoff is None:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.DATA_NOT_READY,
            data_cutoff=None,
            holding_weeks=holding_weeks,
            reasons=("invalid_data_cutoff",),
        )

    normalized_histories = _normalize_mapping(histories, "history")
    normalized_metadata = _normalize_mapping(metadata, "metadata")
    nonproduction_override = (
        len(normalized_histories) < 1_000 or minimum_historical_return_lcb < 0.0
    )
    if not normalized_histories:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.DATA_NOT_READY,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            reasons=("empty_history_universe",),
        )
    if len(normalized_histories) < minimum_universe_size:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.DATA_NOT_READY,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            reasons=(
                f"only_{len(normalized_histories)}_symbols_in_universe;"
                f"minimum_{minimum_universe_size}_required",
            ),
        )

    market_regime = assess_market_regime(
        normalized_histories,
        cutoff,
        minimum_symbols=minimum_universe_size,
    )
    if market_regime.state == MarketRegimeState.UNAVAILABLE:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.DATA_NOT_READY,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            market_regime=market_regime,
            reasons=("market_regime_unavailable",),
        )
    if not market_index_histories:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.DATA_NOT_READY,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            market_regime=market_regime,
            reasons=("core_index_regime_required_but_missing",),
        )
    index_regime = (
        None
        if market_index_histories is None
        else assess_index_regime(market_index_histories, cutoff)
    )
    if index_regime is not None and index_regime.state == IndexRegimeState.UNAVAILABLE:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.DATA_NOT_READY,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            market_regime=market_regime,
            index_regime=index_regime,
            reasons=("core_index_regime_unavailable",),
        )
    price_cycle = (
        assess_price_cycle(market_regime, index_regime) if index_regime is not None else None
    )
    if price_cycle is not None and price_cycle.state == PriceCycleState.UNAVAILABLE:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.DATA_NOT_READY,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            market_regime=market_regime,
            index_regime=index_regime,
            price_cycle=price_cycle,
            reasons=("price_cycle_evidence_unavailable",),
        )

    contract = horizon_contract(holding_weeks)
    holding_sessions = HOLDING_PERIOD_SESSIONS[holding_weeks]
    budget = _resolve_risk_budget(
        price_cycle,
        risk_budget,
        holding_sessions=holding_sessions,
    )
    raw_candidates: list[_CandidateInput] = []
    exclusions: list[CandidateExclusion] = []
    for symbol, frame in sorted(normalized_histories.items()):
        item = normalized_metadata.get(symbol)
        hard_reasons = _metadata_hard_gate(
            symbol,
            item,
            minimum_balance_sheet_strength=minimum_balance_sheet_strength,
        )
        if hard_reasons:
            exclusions.append(CandidateExclusion(symbol, hard_reasons))
            continue
        assert item is not None
        try:
            timeframe = assess_multi_timeframe(
                frame,
                as_of=cutoff,
                holding_weeks=holding_weeks,
            )
        except MultiTimeframeDataError as exc:
            exclusions.append(CandidateExclusion(symbol, (f"multi_timeframe_data:{exc}",)))
            continue
        try:
            returns = _returns_at_cutoff(frame, cutoff)
        except ValueError as exc:
            exclusions.append(CandidateExclusion(symbol, (f"invalid_returns:{exc}",)))
            continue
        try:
            stage = assess_medium_term_stage((1.0 + returns).cumprod())
        except (TypeError, ValueError):
            exclusions.append(CandidateExclusion(symbol, ("late_stage_guard_unavailable",)))
            continue
        if stage.hard_freeze_new_entry:
            exclusions.append(
                CandidateExclusion(
                    symbol,
                    ("late_stage_acceleration_hard_freeze", *stage.reasons),
                )
            )
            continue
        if not timeframe.candidate_qualified:
            exclusions.append(
                CandidateExclusion(
                    symbol,
                    _multi_timeframe_exclusion_reasons(timeframe),
                )
            )
            continue
        risk_history_required_returns = _required_holding_risk_return_observations(budget)
        risk_history_reasons = _holding_risk_history_exclusion_reasons(returns, budget)
        # Retained only as a legacy audit record.  It no longer controls
        # candidate membership or action permission; those are owned by the
        # selected horizon's multi-timeframe execution contract below.
        entry = assess_entry_readiness(frame, as_of=cutoff)
        unknown = _unknown_evidence(item)
        raw_candidates.append(
            _CandidateInput(
                symbol=symbol,
                name=str(item.get("name", symbol)).strip() or symbol,
                industry=str(item.get("industry", "")).strip(),
                entry=entry,
                timeframe=timeframe,
                returns=returns,
                evidence_unknown=unknown,
                risk_history_available=not risk_history_reasons,
                risk_history_reasons=risk_history_reasons,
                risk_history_available_returns=len(returns),
                risk_history_required_returns=risk_history_required_returns,
            )
        )

    benchmark_returns = _core_index_benchmark_returns(market_index_histories, cutoff)
    relative_strength = _full_universe_relative_strength(
        normalized_histories,
        cutoff,
        sessions=contract.relative_strength_sessions,
    )
    candidates = (
        _rank_candidates(
            raw_candidates,
            holding_weeks,
            relative_strength,
            benchmark_returns,
        )
        if raw_candidates
        else []
    )
    # Freeze the same plan used by the action gate and by the user-facing
    # shortlist before any actionable portfolio is searched.
    candidates = [
        replace(
            candidate,
            price_observation_plan=_build_horizon_price_observation_plan(
                normalized_histories.get(candidate.symbol),
                cutoff=cutoff,
                holding_weeks=holding_weeks,
                timeframe=candidate.timeframe,
                entry_pattern=_observation_entry_pattern(candidate),
            ),
        )
        for candidate in candidates
    ]
    candidate_actions = {
        candidate.symbol: _candidate_action(candidate, price_cycle) for candidate in candidates
    }
    horizon_candidate_count = len(candidates)
    risk_eligible_candidates = [
        candidate for candidate in candidates if candidate.risk_history_available
    ]
    risk_history_eligible_candidate_count = len(risk_eligible_candidates)
    risk_history_ineligible_candidate_count = (
        horizon_candidate_count - risk_history_eligible_candidate_count
    )
    daily_entry_ready_count = sum(candidate.timeframe.execution_ready for candidate in candidates)
    research_candidates = _build_research_shortlist(
        candidates,
        candidate_actions,
        holding_weeks=holding_weeks,
        histories=normalized_histories,
        cutoff=cutoff,
    )

    if len(candidates) < 3:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            research_candidates=research_candidates,
            entry_ready_count=daily_entry_ready_count,
            horizon_candidate_count=horizon_candidate_count,
            risk_history_eligible_candidate_count=risk_history_eligible_candidate_count,
            risk_history_ineligible_candidate_count=risk_history_ineligible_candidate_count,
            actionable_candidate_count=sum(
                action is CandidateAction.CONDITIONAL_ENTRY
                for action, _reasons in candidate_actions.values()
            ),
            exclusions=tuple(exclusions),
            market_regime=market_regime,
            index_regime=index_regime,
            price_cycle=price_cycle,
            reasons=(f"only_{len(candidates)}_horizon_candidates;minimum_three",),
        )

    if len(risk_eligible_candidates) < 3:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            research_candidates=research_candidates,
            entry_ready_count=daily_entry_ready_count,
            horizon_candidate_count=horizon_candidate_count,
            risk_history_eligible_candidate_count=risk_history_eligible_candidate_count,
            risk_history_ineligible_candidate_count=risk_history_ineligible_candidate_count,
            actionable_candidate_count=0,
            exclusions=tuple(exclusions),
            market_regime=market_regime,
            index_regime=index_regime,
            price_cycle=price_cycle,
            reasons=(
                f"only_{risk_history_eligible_candidate_count}_risk_history_eligible_candidates;"
                "minimum_three",
            ),
            warnings=(
                "结构候选已经生成，但具备完整持有期风险与LCB历史的股票不足3只；"
                "历史不足者仅保留研究观察，不进入组合搜索或权重计算。",
            ),
        )

    # Research-set optimisation is independent from current deployment
    # permission.  This lets a defensive market or incomplete review evidence
    # retain a diversified, risk-evaluated 3--5-name research set while the
    # action layer correctly stays at 100% cash.
    research_pool = risk_eligible_candidates[:candidate_pool_size]
    research_viable, research_rejected, research_evaluated = _search_candidate_portfolios(
        research_pool,
        budget=budget,
        beam_width=beam_width,
        minimum_historical_return_lcb=minimum_historical_return_lcb,
    )
    research_best: AdaptivePortfolioEvaluation | None = None
    observation_best: AdaptivePortfolioEvaluation | None = None
    observation_rejection_reasons: tuple[str, ...] = ()
    if any(research_viable.values()):
        for rows in research_viable.values():
            rows.sort(key=_viable_sort_key)
        _, research_best, research_selected = _select_stock_count(research_viable)
        research_candidates = _build_research_shortlist(
            candidates,
            candidate_actions,
            holding_weeks=holding_weeks,
            histories=normalized_histories,
            cutoff=cutoff,
            preferred_symbols=tuple(
                position.symbol
                for position in sorted(
                    research_best.positions,
                    key=lambda item: (-item.weight, -item.signal_score, item.symbol),
                )
            ),
            evaluation=research_best,
        )
    elif any(research_rejected.values()):
        rejected = _select_observation_portfolio(research_rejected)
        observation_best = rejected.evaluation
        observation_rejection_reasons = rejected.rejection_reasons
        research_candidates = _build_research_shortlist(
            candidates,
            candidate_actions,
            holding_weeks=holding_weeks,
            histories=normalized_histories,
            cutoff=cutoff,
            preferred_symbols=tuple(
                position.symbol
                for position in sorted(
                    observation_best.positions,
                    key=lambda item: (-item.weight, -item.signal_score, item.symbol),
                )
            ),
            observation_evaluation=observation_best,
        )

    actionable_candidates = [
        candidate
        for candidate in risk_eligible_candidates
        if candidate_actions[candidate.symbol][0] is CandidateAction.CONDITIONAL_ENTRY
    ]
    search_pool = actionable_candidates[:candidate_pool_size]
    if len(search_pool) < 3:
        evidence_blocks_action = any(
            candidate.evidence_unknown for candidate in research_candidates
        )
        return MidtermPortfolioResult(
            status=(
                MidtermPortfolioStatus.VALIDATION_NOT_READY
                if evidence_blocks_action
                else MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO
            ),
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            research_candidates=research_candidates,
            entry_ready_count=daily_entry_ready_count,
            horizon_candidate_count=horizon_candidate_count,
            risk_history_eligible_candidate_count=risk_history_eligible_candidate_count,
            risk_history_ineligible_candidate_count=risk_history_ineligible_candidate_count,
            actionable_candidate_count=len(actionable_candidates),
            search_pool_count=len(search_pool),
            evaluated_portfolio_count=research_evaluated,
            exclusions=tuple(exclusions),
            market_regime=market_regime,
            index_regime=index_regime,
            price_cycle=price_cycle,
            research_evaluation=research_best,
            research_stock_exposure=(
                0.0 if research_best is None else research_best.stock_exposure
            ),
            research_cash_weight=(1.0 if research_best is None else research_best.cash_weight),
            observation_evaluation=observation_best,
            observation_rejection_reasons=observation_rejection_reasons,
            reasons=(
                "research_candidates_generated_but_fewer_than_three_pass_current_entry_policy",
            ),
            warnings=(
                (
                    "研究候选的财务、公告或可买性证据尚未接齐，行动层保持现金。"
                    if evidence_blocks_action
                    else "研究候选不等于当前可以买；本轮周期介入门后不足3只，行动层保持现金。"
                ),
            ),
            evidence_review_required=evidence_blocks_action,
        )
    viable_by_count, _action_rejected, action_evaluated = _search_candidate_portfolios(
        search_pool,
        budget=budget,
        beam_width=beam_width,
        minimum_historical_return_lcb=minimum_historical_return_lcb,
    )

    if not any(viable_by_count.values()):
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            research_candidates=research_candidates,
            entry_ready_count=daily_entry_ready_count,
            horizon_candidate_count=horizon_candidate_count,
            risk_history_eligible_candidate_count=risk_history_eligible_candidate_count,
            risk_history_ineligible_candidate_count=risk_history_ineligible_candidate_count,
            actionable_candidate_count=len(actionable_candidates),
            search_pool_count=len(search_pool),
            evaluated_portfolio_count=research_evaluated,
            action_evaluated_portfolio_count=action_evaluated,
            exclusions=tuple(exclusions),
            market_regime=market_regime,
            index_regime=index_regime,
            price_cycle=price_cycle,
            research_evaluation=research_best,
            research_stock_exposure=(
                0.0 if research_best is None else research_best.stock_exposure
            ),
            research_cash_weight=(1.0 if research_best is None else research_best.cash_weight),
            observation_evaluation=observation_best,
            observation_rejection_reasons=observation_rejection_reasons,
            reasons=("no_3_to_5_stock_set_passed_grid_industry_and_risk_budgets",),
            warnings=(
                "研究候选已经生成，但没有3至5股组合同时通过10%操作档、"
                "行业集中度与当前周期的下行风险预算。"
            ),
        )

    for rows in viable_by_count.values():
        rows.sort(key=_viable_sort_key)
    _, best, selected = _select_stock_count(viable_by_count)
    selected_by_symbol = {item.symbol: item for item in selected}
    ordered_positions = sorted(
        best.positions,
        key=lambda item: (-item.weight, -item.signal_score, item.symbol),
    )
    exact_target_by_symbol = dict(best.exact_target_weights)
    research_candidates = _build_research_shortlist(
        candidates,
        candidate_actions,
        holding_weeks=holding_weeks,
        histories=normalized_histories,
        cutoff=cutoff,
        preferred_symbols=tuple(position.symbol for position in ordered_positions),
        evaluation=best,
    )
    entry_plan_by_symbol = {
        candidate.symbol: candidate.conditional_entry_plan for candidate in research_candidates
    }
    price_observation_plan_by_symbol = {
        candidate.symbol: candidate.price_observation_plan for candidate in research_candidates
    }
    positions = tuple(
        MidtermSelectedPosition(
            rank=rank,
            symbol=position.symbol,
            name=selected_by_symbol[position.symbol].name,
            industry=position.industry,
            weight=exact_target_by_symbol[position.symbol],
            signal_score=position.signal_score,
            entry_pattern=_observation_entry_pattern(selected_by_symbol[position.symbol]),
            breakout_line=float(
                selected_by_symbol[position.symbol].timeframe.structure.breakout_line
            ),
            days_since_breakout=int(
                selected_by_symbol[position.symbol].timeframe.structure.days_or_bars_since_breakout
                or 0
            ),
            annual_downside_volatility=position.annual_downside_volatility,
            downside_risk_contribution=position.downside_risk_contribution,
            evidence_unknown=selected_by_symbol[position.symbol].evidence_unknown,
            operational_account_weight=position.weight,
            operational_stock_sleeve_weight=position.weight / best.stock_exposure,
            conditional_entry_plan=entry_plan_by_symbol.get(position.symbol),
            price_observation_plan=price_observation_plan_by_symbol.get(position.symbol),
            timeframe=selected_by_symbol[position.symbol].timeframe,
            horizon_absolute_return=selected_by_symbol[position.symbol].horizon_absolute_return,
            relative_strength_sessions=selected_by_symbol[
                position.symbol
            ].relative_strength_sessions,
        )
        for rank, position in enumerate(ordered_positions, start=1)
    )
    warnings: list[str] = [
        "组合收益下界是历史非重叠窗口代理，is_out_of_sample=false；正式验证前只作研究。"
    ]
    if price_cycle is not None:
        warnings.append(
            f"价格周期为“{price_cycle.label}”；股票敞口上限"
            f"{price_cycle.policy.max_stock_exposure:.0%}，参数尚待严格walk-forward验证。"
        )
    unknown_symbols = [position.symbol for position in positions if position.evidence_unknown]
    if unknown_symbols:
        warnings.append(
            "以下入选标的仍有财务或官方公告证据待补充，不应据此直接交易："
            + "、".join(unknown_symbols)
        )
    if nonproduction_override:
        warnings.append(
            "本轮使用了小样本或负收益下界研究覆盖，仅供测试；状态不得升级为RESEARCH_ONLY。"
        )
    review_required = bool(unknown_symbols) or nonproduction_override
    final_reasons: list[str] = []
    if unknown_symbols:
        final_reasons.append("provisional_positions_require_evidence_and_execution_review")
    if nonproduction_override:
        final_reasons.append("nonproduction_small_universe_or_negative_lcb_override")
    return MidtermPortfolioResult(
        status=(
            MidtermPortfolioStatus.VALIDATION_NOT_READY
            if review_required
            else MidtermPortfolioStatus.RESEARCH_ONLY
        ),
        data_cutoff=cutoff,
        holding_weeks=holding_weeks,
        positions=positions,
        research_candidates=research_candidates,
        stock_exposure=best.stock_exposure,
        cash_weight=best.cash_weight,
        borrowed_weight=0.0,
        evaluation=best,
        entry_ready_count=daily_entry_ready_count,
        horizon_candidate_count=horizon_candidate_count,
        risk_history_eligible_candidate_count=risk_history_eligible_candidate_count,
        risk_history_ineligible_candidate_count=risk_history_ineligible_candidate_count,
        actionable_candidate_count=len(actionable_candidates),
        search_pool_count=len(search_pool),
        evaluated_portfolio_count=research_evaluated,
        action_evaluated_portfolio_count=action_evaluated,
        exclusions=tuple(exclusions),
        market_regime=market_regime,
        index_regime=index_regime,
        price_cycle=price_cycle,
        research_evaluation=best,
        research_stock_exposure=best.stock_exposure,
        research_cash_weight=best.cash_weight,
        warnings=tuple(warnings),
        reasons=tuple(final_reasons),
        evidence_review_required=review_required,
    )


def _normalize_cutoff(value: object) -> pd.Timestamp | None:
    try:
        cutoff = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(cutoff):
        return None
    if cutoff.tz is not None:
        cutoff = cutoff.tz_localize(None)
    return cutoff.normalize()


def _finite_optional(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _normalize_mapping(values: Mapping[str, Any], label: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_symbol, value in values.items():
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in normalized:
            raise ValueError(f"{label} symbols must be non-blank and unique")
        normalized[symbol] = value
    return normalized


def _metadata_hard_gate(
    symbol: str,
    item: Mapping[str, Any] | None,
    *,
    minimum_balance_sheet_strength: float,
) -> tuple[str, ...]:
    if item is None:
        return ("metadata_missing",)
    reasons: list[str] = []
    for flag in ("is_st", "is_delisting"):
        if flag not in item or not isinstance(item[flag], (bool, np.bool_)):
            reasons.append(f"{flag}_must_be_explicit_boolean")
    if reasons:
        return tuple(reasons)
    name = str(item.get("name", symbol)).replace(" ", "").upper()
    if bool(item["is_st"]) or name.startswith(("ST", "*ST")):
        reasons.append("st_stock_excluded")
    if bool(item["is_delisting"]) or "退" in name:
        reasons.append("delisting_stock_excluded")
    if item.get("is_suspended") is True:
        reasons.append("suspended_stock_excluded")
    if item.get("is_buyable_at_cutoff") is False:
        reasons.append("formation_unbuyable_at_cutoff")
    if item.get("is_limit_up_at_cutoff") is True:
        reasons.append("formation_limit_up_excluded_pending_execution_check")
    industry = str(item.get("industry", "")).strip()
    if not industry:
        reasons.append("industry_missing")
    if item.get("fundamental_veto") is True or _gate_value(item.get("fundamental_gate")) == "veto":
        reasons.append("fundamental_veto")
    if (
        item.get("announcement_veto") is True
        or _gate_value(item.get("announcement_gate")) == "veto"
    ):
        reasons.append("official_announcement_veto")
    balance = item.get("balance_sheet_strength_score")
    if balance is not None:
        try:
            score = float(balance)
        except (TypeError, ValueError):
            reasons.append("invalid_balance_sheet_strength")
        else:
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                reasons.append("invalid_balance_sheet_strength")
            elif score < minimum_balance_sheet_strength:
                reasons.append("balance_sheet_strength_veto")
    return tuple(reasons)


def _gate_value(value: object) -> str:
    return str(value).strip().lower() if value is not None else "unknown"


def _unknown_evidence(item: Mapping[str, Any]) -> tuple[str, ...]:
    unknown: list[str] = []
    if item.get("is_buyable_at_cutoff") is not True:
        unknown.append("formation_execution_evidence_unknown")
    if (
        item.get("fundamental_veto") is not True
        and _gate_value(item.get("fundamental_gate")) != "pass"
    ):
        unknown.append("fundamental_evidence_unknown")
    if (
        item.get("announcement_veto") is not True
        and _gate_value(item.get("announcement_gate")) != "pass"
    ):
        unknown.append("official_announcement_evidence_unknown")
    return tuple(unknown)


def _multi_timeframe_exclusion_reasons(
    assessment: MultiTimeframeAssessment,
) -> tuple[str, ...]:
    """Return compact, horizon-specific hard-gate reasons for audit output."""

    reasons: list[str] = [
        f"horizon_{assessment.holding_weeks}_week_structure_not_qualified",
        f"slow_{assessment.slow_direction.timeframe.value}:"
        f"{assessment.slow_direction.direction.value}",
        f"primary_{assessment.structure.timeframe.value}:{assessment.structure.state.value}",
    ]
    if assessment.above_daily_anchor is not True:
        reasons.append(f"below_daily_ma{assessment.contract.daily_anchor_sessions}_anchor")
    if assessment.incomplete_week_excluded:
        reasons.append("incomplete_current_week_excluded")
    if assessment.incomplete_month_excluded:
        reasons.append("incomplete_current_month_excluded")
    return tuple(reasons)


def _returns_at_cutoff(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    if "trade_date" not in frame or "close" not in frame:
        raise ValueError("trade_date_and_close_required")
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    if bool(dates.isna().any()) or bool(close.isna().any()):
        raise ValueError("invalid_date_or_close")
    clean = pd.Series(close.to_numpy(dtype=float), index=pd.DatetimeIndex(dates).normalize())
    clean = clean.loc[clean.index <= cutoff].sort_index()
    if clean.index.has_duplicates or clean.empty or clean.index[-1] != cutoff:
        raise ValueError("duplicate_or_stale_cutoff")
    returns = clean.pct_change(fill_method=None).dropna()
    if not bool(np.isfinite(returns.to_numpy()).all()) or bool((returns <= -1.0).any()):
        raise ValueError("invalid_simple_returns")
    return returns


def _holding_risk_history_exclusion_reasons(
    returns: pd.Series,
    budget: AdaptiveRiskBudget,
) -> tuple[str, ...]:
    """Audit a single candidate before any 3--5-name risk combination.

    This mirrors, without weakening, every history-length requirement in the
    adaptive risk engine.  Alignment and portfolio-level degeneracy remain
    fail-closed inside that engine.
    """

    required_returns = _required_holding_risk_return_observations(budget)
    available_returns = len(returns)
    if available_returns >= required_returns:
        return ()
    return (
        "insufficient_holding_risk_history:"
        f"available_{available_returns}_returns;required_{required_returns}_returns;"
        f"holding_{budget.holding_period_sessions}_sessions;"
        f"minimum_{budget.minimum_holding_period_samples}_nonoverlapping_samples",
    )


def _required_holding_risk_return_observations(budget: AdaptiveRiskBudget) -> int:
    """Return the unchanged adaptive-engine history floor for one stock."""

    return max(
        budget.minimum_observations,
        FIVE_DAY_SESSIONS * MINIMUM_FIVE_DAY_SAMPLES,
        ROLLING_DRAWDOWN_SESSIONS + MINIMUM_ROLLING_DRAWDOWN_WINDOWS - 1,
        budget.holding_period_sessions + MINIMUM_ROLLING_DRAWDOWN_WINDOWS - 1,
        budget.holding_period_sessions * budget.minimum_holding_period_samples,
    )


def _core_index_benchmark_returns(
    histories: Mapping[str, pd.DataFrame] | None,
    cutoff: pd.Timestamp,
) -> pd.Series | None:
    """Build an equal-weight core-index return proxy for downside capture."""

    if not histories:
        return None
    series: list[pd.Series] = []
    for frame in histories.values():
        if not isinstance(frame, pd.DataFrame) or "close" not in frame:
            return None
        if "trade_date" in frame:
            raw_dates = frame["trade_date"]
        elif "date" in frame:
            raw_dates = frame["date"]
        else:
            raw_dates = frame.index
        dates = pd.to_datetime(raw_dates, errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        if bool(pd.isna(dates).any()) or bool(close.isna().any()):
            return None
        values = pd.Series(
            close.to_numpy(dtype=float),
            index=pd.DatetimeIndex(dates).tz_localize(None).normalize(),
        )
        values = values.loc[values.index <= cutoff].sort_index()
        if values.empty or values.index.has_duplicates or values.index[-1] != cutoff:
            return None
        returns = values.pct_change(fill_method=None).dropna()
        if not bool(np.isfinite(returns.to_numpy()).all()):
            return None
        series.append(returns)
    aligned = pd.concat(series, axis=1, join="inner").dropna()
    if len(aligned) < 61:
        return None
    benchmark = aligned.mean(axis=1)
    benchmark.name = "core_index_equal_weight_return"
    return benchmark


def _full_universe_relative_strength(
    histories: Mapping[str, pd.DataFrame],
    cutoff: pd.Timestamp,
    *,
    sessions: int,
) -> dict[str, float]:
    """Return exact horizon-session return percentiles across the full universe."""

    if sessions < 1:
        raise ValueError("relative-strength sessions must be positive")

    values: dict[str, float] = {}
    for symbol, frame in histories.items():
        if "trade_date" not in frame or "close" not in frame:
            continue
        dates = pd.to_datetime(frame["trade_date"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        if bool(dates.isna().any()) or bool(close.isna().any()):
            continue
        series = pd.Series(
            close.to_numpy(dtype=float),
            index=pd.DatetimeIndex(dates).tz_localize(None).normalize(),
        )
        series = series.loc[series.index <= cutoff].sort_index()
        if (
            len(series) < sessions + 1
            or series.index.has_duplicates
            or series.index[-1] != cutoff
            or bool((series <= 0.0).any())
        ):
            continue
        value = float(series.iloc[-1] / series.iloc[-sessions - 1] - 1.0)
        if math.isfinite(value):
            values[symbol] = value
    if not values:
        return {}
    ranked = pd.Series(values, dtype=float).rank(method="average", pct=True)
    return {str(symbol): float(value) for symbol, value in ranked.items()}


def _full_universe_relative_strength_60(
    histories: Mapping[str, pd.DataFrame],
    cutoff: pd.Timestamp,
) -> dict[str, float]:
    """Compatibility wrapper for callers that explicitly need sixty sessions."""

    return _full_universe_relative_strength(histories, cutoff, sessions=60)


def _downside_capture_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series | None,
) -> float | None:
    if benchmark_returns is None:
        return None
    aligned = pd.concat((returns, benchmark_returns), axis=1, join="inner").dropna().tail(60)
    if aligned.shape[1] != 2:
        return None
    down = aligned.loc[aligned.iloc[:, 1] < 0.0]
    if len(down) < 10:
        return None
    denominator = float(down.iloc[:, 1].mean())
    numerator = float(down.iloc[:, 0].mean())
    if not math.isfinite(denominator) or not math.isfinite(numerator) or denominator >= 0.0:
        return None
    ratio = numerator / denominator
    return float(ratio) if math.isfinite(ratio) else None


def _risk_budget_for_cycle(
    cycle: PriceCycleAssessment | None,
    *,
    holding_sessions: int,
) -> AdaptiveRiskBudget:
    common = {
        "holding_period_sessions": holding_sessions,
        "minimum_observations": max(160, holding_sessions * 8),
        "minimum_holding_period_samples": 8,
        "holding_period_cost_rate": 0.002,
    }
    if cycle is None:
        return AdaptiveRiskBudget(**common)
    policy = cycle.policy
    return AdaptiveRiskBudget(
        max_annual_downside_volatility=policy.max_annual_downside_volatility,
        max_rolling_drawdown_60_p90=policy.max_rolling_drawdown_60_p90,
        max_es95_5d=policy.max_es95_5d,
        max_down_period_correlation=policy.max_down_period_correlation,
        max_position_downside_risk_contribution=(policy.max_position_downside_risk_contribution),
        industry_weight_limit=policy.industry_weight_limit,
        maximum_stock_exposure=policy.max_stock_exposure,
        max_horizon_rolling_drawdown_p90=policy.max_rolling_drawdown_60_p90,
        **common,
    )


def _resolve_risk_budget(
    cycle: PriceCycleAssessment | None,
    requested: AdaptiveRiskBudget | None,
    *,
    holding_sessions: int,
) -> AdaptiveRiskBudget:
    """Apply caller settings only when they are at least as strict as the cycle.

    A caller may raise data requirements, cost assumptions or confidence, and
    may lower every risk/exposure ceiling.  It cannot use a custom budget to
    escape a defensive cycle policy.
    """

    cycle_budget = _risk_budget_for_cycle(cycle, holding_sessions=holding_sessions)
    if requested is None:
        return cycle_budget
    if requested.holding_period_sessions != holding_sessions:
        raise ValueError("risk_budget.holding_period_sessions must match HOLDING_PERIOD_SESSIONS")
    if cycle is None:
        return requested
    requested_exposure = (
        cycle_budget.maximum_stock_exposure
        if requested.maximum_stock_exposure is None
        else requested.maximum_stock_exposure
    )
    assert cycle_budget.maximum_stock_exposure is not None
    cycle_horizon_drawdown_limit = (
        cycle_budget.max_horizon_rolling_drawdown_p90
        if cycle_budget.max_horizon_rolling_drawdown_p90 is not None
        else cycle_budget.max_rolling_drawdown_60_p90
    )
    requested_horizon_drawdown_limit = (
        requested.max_horizon_rolling_drawdown_p90
        if requested.max_horizon_rolling_drawdown_p90 is not None
        else requested.max_rolling_drawdown_60_p90
    )
    return AdaptiveRiskBudget(
        max_annual_downside_volatility=min(
            cycle_budget.max_annual_downside_volatility,
            requested.max_annual_downside_volatility,
        ),
        max_rolling_drawdown_60_p90=min(
            cycle_budget.max_rolling_drawdown_60_p90,
            requested.max_rolling_drawdown_60_p90,
        ),
        max_es95_5d=min(cycle_budget.max_es95_5d, requested.max_es95_5d),
        max_down_period_correlation=min(
            cycle_budget.max_down_period_correlation,
            requested.max_down_period_correlation,
        ),
        max_position_downside_risk_contribution=min(
            cycle_budget.max_position_downside_risk_contribution,
            requested.max_position_downside_risk_contribution,
        ),
        industry_weight_limit=min(
            cycle_budget.industry_weight_limit,
            requested.industry_weight_limit,
        ),
        maximum_stock_exposure=min(
            cycle_budget.maximum_stock_exposure,
            requested_exposure,
        ),
        holding_period_sessions=holding_sessions,
        holding_period_cost_rate=max(
            cycle_budget.holding_period_cost_rate,
            requested.holding_period_cost_rate,
        ),
        lcb_confidence=max(cycle_budget.lcb_confidence, requested.lcb_confidence),
        minimum_observations=max(
            cycle_budget.minimum_observations,
            requested.minimum_observations,
        ),
        minimum_down_periods=max(
            cycle_budget.minimum_down_periods,
            requested.minimum_down_periods,
        ),
        minimum_holding_period_samples=max(
            cycle_budget.minimum_holding_period_samples,
            requested.minimum_holding_period_samples,
        ),
        max_horizon_rolling_drawdown_p90=min(
            cycle_horizon_drawdown_limit,
            requested_horizon_drawdown_limit,
        ),
    )


def _candidate_action(
    candidate: MidtermCandidate,
    cycle: PriceCycleAssessment | None,
) -> tuple[CandidateAction, tuple[str, ...]]:
    """Apply cycle-specific entry confirmation without changing research rank."""

    if not candidate.risk_history_available:
        return CandidateAction.WAIT_CONFIRMATION, (
            "risk_history_unavailable_for_portfolio_weighting",
            *candidate.risk_history_reasons,
        )
    plan = candidate.price_observation_plan
    if plan is None:
        return CandidateAction.OBSERVE_ONLY, ("initial_entry_risk_structure_plan_unavailable",)
    if plan.initial_risk_qualified is not True:
        return CandidateAction.OBSERVE_ONLY, (
            plan.initial_risk_reason or "initial_entry_risk_not_assessed",
        )
    strictness = EntryStrictness.STANDARD if cycle is None else cycle.policy.entry_strictness
    if not candidate.timeframe.execution_ready:
        action = (
            CandidateAction.OBSERVE_ONLY
            if strictness is EntryStrictness.EXCEPTION_ONLY
            else CandidateAction.WAIT_CONFIRMATION
        )
        return action, (
            f"horizon_daily_execution_not_ready:{candidate.timeframe.execution.state.value}",
        )
    if candidate.evidence_unknown:
        return (
            CandidateAction.WAIT_CONFIRMATION,
            ("fundamental_announcement_or_execution_evidence_requires_review",),
        )
    if strictness is EntryStrictness.STANDARD:
        return CandidateAction.CONDITIONAL_ENTRY, (
            "standard_multi_timeframe_and_daily_entry_confirmed",
        )

    execution = candidate.timeframe.execution
    checks: tuple[tuple[bool, str], ...]
    if strictness is EntryStrictness.TIGHT:
        checks = (
            (
                candidate.horizon_absolute_return > 0.0,
                f"{candidate.relative_strength_sessions}_session_absolute_return_not_positive",
            ),
            (
                execution.activity_ratio is not None and execution.activity_ratio >= 1.20,
                "horizon_execution_activity_ratio_below_1_20",
            ),
            (
                execution.distance_to_average is not None and execution.distance_to_average <= 0.08,
                "horizon_execution_average_distance_above_8pct",
            ),
        )
    else:
        checks = (
            (
                candidate.horizon_absolute_return > 0.0,
                f"{candidate.relative_strength_sessions}_session_absolute_return_not_positive",
            ),
            (
                candidate.relative_strength_percentile >= 0.90,
                "relative_strength_not_top_decile",
            ),
            (
                candidate.timeframe.slow_direction.qualified,
                "slow_timeframe_direction_not_qualified",
            ),
            (
                candidate.downside_capture_ratio is not None
                and candidate.downside_capture_ratio <= 0.80,
                "downside_capture_above_0_80_or_unavailable",
            ),
            (
                execution.activity_ratio is not None and execution.activity_ratio >= 1.30,
                "horizon_execution_activity_ratio_below_1_30",
            ),
            (
                execution.distance_to_average is not None and execution.distance_to_average <= 0.06,
                "horizon_execution_average_distance_above_6pct",
            ),
        )
        if strictness is EntryStrictness.EXCEPTION_ONLY:
            checks = (
                *checks,
                (
                    execution.state is ExecutionState.READY_PULLBACK
                    or candidate.timeframe.structure.state is StructureState.HEALTHY_PULLBACK,
                    "downtrend_pressure_requires_pullback_or_reclaim",
                ),
            )
    failures = tuple(reason for passed, reason in checks if not passed)
    if not failures:
        return CandidateAction.CONDITIONAL_ENTRY, (f"{strictness.value}_entry_confirmed",)
    action = (
        CandidateAction.OBSERVE_ONLY
        if strictness is EntryStrictness.EXCEPTION_ONLY
        else CandidateAction.WAIT_CONFIRMATION
    )
    return action, failures


def _build_research_shortlist(
    candidates: list[MidtermCandidate],
    actions: Mapping[str, tuple[CandidateAction, tuple[str, ...]]],
    *,
    holding_weeks: int,
    histories: Mapping[str, pd.DataFrame],
    cutoff: pd.Timestamp,
    preferred_symbols: tuple[str, ...] | None = None,
    evaluation: AdaptivePortfolioEvaluation | None = None,
    observation_evaluation: AdaptivePortfolioEvaluation | None = None,
) -> tuple[MidtermResearchCandidate, ...]:
    if preferred_symbols:
        by_symbol = {candidate.symbol: candidate for candidate in candidates}
        selected = [by_symbol[symbol] for symbol in preferred_symbols if symbol in by_symbol]
    else:
        selected = candidates[: min(4, len(candidates))]
    # Keep a bounded sample of structural candidates whose deep risk history
    # is unavailable visible even though they can never receive a portfolio
    # weight.  The user-facing shortlist must remain a genuine 3--5 name list;
    # aggregate result counts expose the complete structural set.
    selected_symbols = {candidate.symbol for candidate in selected}
    risk_history_audit = [
        candidate for candidate in candidates if not candidate.risk_history_available
    ][:4]
    for candidate in risk_history_audit:
        if candidate.symbol not in selected_symbols and len(selected) < 5:
            selected.append(candidate)
            selected_symbols.add(candidate.symbol)
    evaluation_by_symbol = (
        {}
        if evaluation is None
        else {position.symbol: position for position in evaluation.positions}
    )
    exact_target_by_symbol = {} if evaluation is None else dict(evaluation.exact_target_weights)
    observation_by_symbol = (
        {}
        if observation_evaluation is None
        else {position.symbol: position for position in observation_evaluation.positions}
    )
    rows: list[MidtermResearchCandidate] = []
    for rank, candidate in enumerate(selected, start=1):
        action = actions[candidate.symbol][0]
        observation_pattern = _observation_entry_pattern(candidate)
        price_observation_plan = candidate.price_observation_plan
        rows.append(
            MidtermResearchCandidate(
                rank=rank,
                symbol=candidate.symbol,
                name=candidate.name,
                industry=candidate.industry,
                signal_score=candidate.signal_score,
                entry_pattern=observation_pattern,
                breakout_line=_finite_optional(candidate.timeframe.structure.breakout_line),
                days_since_breakout=(
                    None
                    if candidate.timeframe.structure.days_or_bars_since_breakout is None
                    else int(candidate.timeframe.structure.days_or_bars_since_breakout)
                ),
                action=action,
                action_reasons=actions[candidate.symbol][1],
                absolute_return_60=candidate.absolute_return_60,
                relative_strength_percentile=candidate.relative_strength_percentile,
                downside_capture_ratio=candidate.downside_capture_ratio,
                evidence_unknown=candidate.evidence_unknown,
                research_weight=exact_target_by_symbol.get(candidate.symbol),
                downside_risk_contribution=(
                    None
                    if candidate.symbol not in evaluation_by_symbol
                    else evaluation_by_symbol[candidate.symbol].downside_risk_contribution
                ),
                operational_account_weight=(
                    None
                    if candidate.symbol not in evaluation_by_symbol
                    else evaluation_by_symbol[candidate.symbol].weight
                ),
                operational_stock_sleeve_weight=(
                    None
                    if candidate.symbol not in evaluation_by_symbol
                    else evaluation_by_symbol[candidate.symbol].weight / evaluation.stock_exposure
                ),
                conditional_entry_plan=(
                    price_observation_plan
                    if action is CandidateAction.CONDITIONAL_ENTRY
                    and not candidate.evidence_unknown
                    and price_observation_plan is not None
                    and price_observation_plan.initial_risk_qualified is True
                    else None
                ),
                observation_stock_sleeve_weight=(
                    None
                    if candidate.symbol not in observation_by_symbol
                    else observation_by_symbol[candidate.symbol].weight
                    / observation_evaluation.stock_exposure
                ),
                price_observation_plan=price_observation_plan,
                timeframe=candidate.timeframe,
                horizon_absolute_return=candidate.horizon_absolute_return,
                relative_strength_sessions=candidate.relative_strength_sessions,
                risk_history_available=candidate.risk_history_available,
                risk_history_reasons=candidate.risk_history_reasons,
                risk_history_available_returns=candidate.risk_history_available_returns,
                risk_history_required_returns=candidate.risk_history_required_returns,
            )
        )
    return tuple(rows)


def _observation_entry_pattern(candidate: MidtermCandidate) -> EntryPattern:
    """Map horizon evidence to a daily observation-plan shape without granting entry."""

    if candidate.timeframe.execution.state is ExecutionState.READY_PULLBACK:
        return EntryPattern.HEALTHY_PULLBACK
    if candidate.timeframe.execution.state is ExecutionState.READY_BREAKOUT:
        return EntryPattern.VOLUME_BREAKOUT
    if candidate.timeframe.structure.state is StructureState.HEALTHY_PULLBACK:
        return EntryPattern.BREAKOUT_RECLAIM
    # A structurally qualified candidate can only be BREAKOUT, NEAR_BREAKOUT,
    # or HEALTHY_PULLBACK.  Never let the compatibility-only fixed 60-session
    # assessment change the selected horizon's displayed price-plan shape.
    return EntryPattern.VOLUME_BREAKOUT


def _build_horizon_price_observation_plan(
    frame: pd.DataFrame | None,
    *,
    cutoff: pd.Timestamp,
    holding_weeks: int,
    timeframe: MultiTimeframeAssessment,
    entry_pattern: EntryPattern,
) -> ConditionalEntryPlan | None:
    """Build the daily execution condition for one independently assessed horizon.

    Slow and primary bars decide whether the security belongs in the horizon's
    candidate pool.  The displayed price remains a complete-daily-bar
    execution condition, but its breakout lookback and reference line come
    from that horizon's explicit contract rather than the legacy fixed five-day
    plan.  This keeps a monthly/weekly trend decision separate from an order
    price while still producing one practical conditional level.
    """

    if timeframe.holding_weeks != holding_weeks or timeframe.data_cutoff != cutoff:
        return None
    if not isinstance(frame, pd.DataFrame) or "trade_date" not in frame.columns:
        return None
    prepared = frame.copy()
    parsed_dates = pd.to_datetime(prepared["trade_date"], errors="coerce")
    if bool(parsed_dates.isna().any()):
        return None
    try:
        parsed_dates = parsed_dates.dt.tz_localize(None)
    except (AttributeError, TypeError):
        return None
    prepared["trade_date"] = parsed_dates.dt.normalize()
    prepared = prepared.loc[prepared["trade_date"] <= cutoff].sort_values("trade_date")
    if prepared.empty or pd.Timestamp(prepared.iloc[-1]["trade_date"]).normalize() != cutoff:
        return None
    try:
        enriched = enrich_indicators(prepared)
        atr = float(enriched.iloc[-1]["atr14"])
    except (KeyError, TypeError, ValueError):
        return None
    execution_line = _finite_optional(timeframe.execution.breakout_line)
    primary_structure_line = _finite_optional(timeframe.structure.breakout_line)
    if not timeframe.structure.qualified or primary_structure_line is None:
        # A daily trigger cannot manufacture a missing holding-horizon
        # structure or a protection line simply to satisfy the risk cap.
        return None
    try:
        # The initial line must match the existing holding-review method:
        # confirmed holding-horizon base support minus its 0.5 ATR buffer,
        # rounded down to a price tick.  The former breakout-line-minus-ATR
        # observation line is not a substitute for the actual protection.
        protection = _candidate_stop(
            build_completed_timeframes(prepared, as_of=cutoff),
            timeframe,
            entry_date=cutoff.date(),
        )
    except (MultiTimeframeDataError, ValueError, TypeError, KeyError):
        return None
    entry_line = execution_line if execution_line is not None else primary_structure_line
    price_source_timeframe = (
        "daily" if execution_line is not None else timeframe.structure.timeframe.value
    )
    risk_reference_line = (
        primary_structure_line if primary_structure_line is not None else entry_line
    )
    risk_source_timeframe = (
        timeframe.structure.timeframe.value
        if primary_structure_line is not None
        else price_source_timeframe
    )
    if entry_line is None or risk_reference_line is None or not math.isfinite(atr) or atr <= 0.0:
        return None

    common = {
        "data_cutoff": cutoff,
        "horizon": _HORIZON_PLAN_LABELS[holding_weeks],
        "sessions": HOLDING_PERIOD_SESSIONS[holding_weeks],
        "structure_timeframe": timeframe.structure.timeframe.value,
        "structure_cutoff": timeframe.structure_bar_cutoff,
        "execution_lookback_sessions": timeframe.contract.execution_breakout_sessions,
        "method_version": MIDTERM_METHOD_VERSION,
        "price_source_timeframe": price_source_timeframe,
        "primary_structure_timeframe": timeframe.structure.timeframe.value,
        "primary_structure_cutoff": timeframe.structure_bar_cutoff,
        "invalidation_price": protection.stop,
        "reduction_review_price": round(max(0.01, risk_reference_line - 0.50 * atr), 4),
        "entry_reference_price": round(entry_line, 4),
        "primary_structure_reference_price": (
            None if primary_structure_line is None else round(primary_structure_line, 4)
        ),
        "invalidation_source_timeframe": risk_source_timeframe,
        "reduction_review_source_timeframe": risk_source_timeframe,
        "initial_protection_support": protection.support,
        "initial_protection_atr": protection.atr14,
        "initial_protection_evidence_date": protection.evidence_date.isoformat(),
        "initial_protection_atr_cutoff": protection.atr_cutoff.isoformat(),
        "initial_protection_method_version": HOLDING_TREE_METHOD_VERSION,
    }
    if entry_pattern is EntryPattern.HEALTHY_PULLBACK:
        reference_label = "日线执行线" if price_source_timeframe == "daily" else "期限主结构线"
        return _with_initial_entry_risk(
            ConditionalEntryPlan(
                kind=ConditionalEntryPlanKind.HEALTHY_PULLBACK,
                price_low=round(max(0.01, entry_line - 0.25 * atr), 4),
                price_high=round(entry_line + 0.15 * atr, 4),
                confirmation_rule=f"回踩{reference_label}附近且完整日线未失效",
                **common,
            )
        )
    if entry_pattern is EntryPattern.BREAKOUT_RECLAIM:
        return _with_initial_entry_risk(
            ConditionalEntryPlan(
                kind=ConditionalEntryPlanKind.RECLAIM,
                trigger_price=round(entry_line + 0.05 * atr, 4),
                confirmation_rule="完整日线收盘重新站回期限执行线",
                **common,
            )
        )
    activity_confirmation = _activity_confirmation_threshold(prepared)
    if activity_confirmation is None:
        return None
    activity_metric, activity_min = activity_confirmation
    activity_label = "成交额" if activity_metric == "amount_cny" else "成交量"
    return _with_initial_entry_risk(
        ConditionalEntryPlan(
            kind=ConditionalEntryPlanKind.VOLUME_BREAKOUT,
            trigger_price=round(entry_line + 0.10 * atr, 4),
            confirmation_activity_metric=activity_metric,
            confirmation_activity_min=activity_min,
            confirmation_rule=(
                f"完整日线收盘越过期限执行线，且{activity_label}达到20日中位数1.2倍"
            ),
            **common,
        )
    )


def _with_initial_entry_risk(plan: ConditionalEntryPlan) -> ConditionalEntryPlan:
    """Assess a structural stop without moving it to force an eight-percent fit.

    For a range the upper bound is the worst permitted buy price.  For a
    breakout/reclaim use the frozen trigger.  A higher actual fill requires
    a fresh check against ``maximum_entry_price``; the cap is a planned price
    distance, not a guarantee on realizable losses or corporate-action basis.
    """

    stop = _finite_optional(plan.invalidation_price)
    is_range = plan.kind is ConditionalEntryPlanKind.HEALTHY_PULLBACK
    reference = _finite_optional(plan.price_high if is_range else plan.trigger_price)
    low = _finite_optional(plan.price_low) if is_range else reference
    maximum = (
        float(
            (Decimal(str(stop)) / Decimal("0.92")).quantize(Decimal("0.0001"), rounding=ROUND_FLOOR)
        )
        if stop is not None
        else None
    )
    risk = None if reference is None or stop is None else (reference - stop) / reference
    if (
        stop is None
        or _finite_optional(plan.primary_structure_reference_price) is None
        or plan.primary_structure_cutoff is None
        or plan.invalidation_source_timeframe is None
    ):
        qualified, reason = False, "initial_entry_risk_structure_unavailable"
    elif reference is None or low is None or low > reference:
        qualified, reason = False, "initial_entry_risk_buy_price_unavailable_or_invalid"
    elif stop >= low:
        qualified, reason = False, "initial_entry_risk_structure_stop_not_below_buy_price"
    elif risk is not None and risk > MAX_INITIAL_ENTRY_RISK + 1e-12:
        qualified, reason = False, "initial_entry_risk_exceeds_8pct_wait_for_better_setup"
    elif Decimal(str(low)).quantize(Decimal("0.01"), rounding=ROUND_CEILING) > Decimal(
        str(min(reference, maximum) if is_range else maximum)
    ).quantize(Decimal("0.01"), rounding=ROUND_FLOOR):
        qualified, reason = False, "initial_entry_risk_no_executable_price_tick"
    else:
        qualified, reason = True, "initial_entry_risk_within_8pct_of_planned_buy_price"
    return replace(
        plan,
        initial_risk_reference_price=reference,
        initial_risk_fraction=risk,
        maximum_entry_price=maximum,
        initial_risk_qualified=qualified,
        initial_risk_reason=reason,
    )


def _build_one_week_conditional_entry_plan(
    frame: pd.DataFrame | None,
    *,
    cutoff: pd.Timestamp,
    action: CandidateAction,
    entry_pattern: EntryPattern,
    evidence_passed: bool,
) -> ConditionalEntryPlan | None:
    """Build a fail-closed one-week plan from data at the shared cutoff.

    The Edwards--Magee breakout line is evidence for the trend gate.  It is not
    reused as an entry price.  All displayed prices come from the one-week
    horizon support/resistance and ATR plan.
    """

    if action is not CandidateAction.CONDITIONAL_ENTRY or not evidence_passed:
        return None
    return _build_one_week_price_observation_plan(
        frame,
        cutoff=cutoff,
        entry_pattern=entry_pattern,
    )


def _build_one_week_price_observation_plan(
    frame: pd.DataFrame | None,
    *,
    cutoff: pd.Timestamp,
    entry_pattern: EntryPattern,
) -> ConditionalEntryPlan | None:
    """Build a neutral one-week price observation plan for a ranked candidate.

    This calculation depends only on common-cutoff price structure.  It grants
    no buy permission and remains separate from ``conditional_entry_plan``.
    """

    if not isinstance(frame, pd.DataFrame) or "trade_date" not in frame.columns:
        return None
    prepared = frame.copy()
    parsed_dates = pd.to_datetime(prepared["trade_date"], errors="coerce")
    if bool(parsed_dates.isna().any()):
        return None
    try:
        parsed_dates = parsed_dates.dt.tz_localize(None)
    except (AttributeError, TypeError):
        return None
    prepared["trade_date"] = parsed_dates.dt.normalize()
    prepared = prepared.loc[prepared["trade_date"] <= cutoff].sort_values("trade_date")
    if prepared.empty or pd.Timestamp(prepared.iloc[-1]["trade_date"]).normalize() != cutoff:
        return None
    try:
        enriched = enrich_indicators(prepared)
        one_week = next(level for level in build_horizon_levels(enriched) if level.sessions == 5)
    except (KeyError, StopIteration, TypeError, ValueError):
        return None
    common = {
        "data_cutoff": cutoff,
        "horizon": one_week.horizon,
        "sessions": one_week.sessions,
    }
    if entry_pattern is EntryPattern.HEALTHY_PULLBACK:
        return ConditionalEntryPlan(
            kind=ConditionalEntryPlanKind.HEALTHY_PULLBACK,
            price_low=one_week.pullback_entry_low,
            price_high=one_week.pullback_entry_high,
            **common,
        )
    if entry_pattern is EntryPattern.BREAKOUT_RECLAIM:
        return ConditionalEntryPlan(
            kind=ConditionalEntryPlanKind.RECLAIM,
            trigger_price=one_week.entry_trigger,
            **common,
        )
    if entry_pattern is EntryPattern.VOLUME_BREAKOUT:
        activity_confirmation = _activity_confirmation_threshold(prepared)
        if activity_confirmation is None:
            return None
        activity_metric, activity_min = activity_confirmation
        activity_label = "成交额" if activity_metric == "amount_cny" else "成交量"
        return ConditionalEntryPlan(
            kind=ConditionalEntryPlanKind.VOLUME_BREAKOUT,
            trigger_price=one_week.breakout_trigger,
            confirmation_activity_metric=activity_metric,
            confirmation_activity_min=activity_min,
            confirmation_rule=(
                f"以收盘价确认突破，且当日{activity_label}不低于截止日已冻结的"
                "20期中位数1.2倍；盘中瞬间越线不算有效突破"
            ),
            **common,
        )
    return None


def _activity_confirmation_threshold(frame: pd.DataFrame) -> tuple[str, float] | None:
    """Freeze one cutoff-known 1.2x 20-session activity threshold.

    CSMAR-derived histories may not carry share volume, while they do carry
    turnover amount.  Prefer amount so an archived breakout remains replayable
    across the immutable history and the verified unadjusted overlay; fall back to share
    volume only when a complete amount window is unavailable.  A partial or
    non-positive window cannot be filled or guessed.
    """

    if frame.empty:
        return None
    for column in ("amount_cny", "volume_shares"):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").tail(20)
        if len(values) != 20 or bool(values.isna().any()):
            continue
        median = float(values.median())
        if math.isfinite(median) and median > 0.0:
            return column, round(1.20 * median, 4)
    return None


def _rank_candidates(
    rows: list[_CandidateInput],
    holding_weeks: int,
    relative_strength: Mapping[str, float],
    benchmark_returns: pd.Series | None = None,
) -> list[MidtermCandidate]:
    raw: list[dict[str, float]] = []
    contract = horizon_contract(holding_weeks)
    for row in rows:
        returns = row.returns
        close_proxy = (1.0 + returns).cumprod()
        features: dict[str, float] = {}
        downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
        features["downside_volatility"] = float(
            np.sqrt(np.mean(np.square(downside))) * math.sqrt(252)
        )
        equity = (1.0 + returns).cumprod()
        features["max_drawdown"] = abs(float((equity / equity.cummax() - 1.0).min()))
        features["multi_timeframe"] = row.timeframe.score
        features["absolute_return_60"] = float(close_proxy.iloc[-1] / close_proxy.iloc[-61] - 1.0)
        features["horizon_absolute_return"] = float(
            close_proxy.iloc[-1] / close_proxy.iloc[-contract.relative_strength_sessions - 1] - 1.0
        )
        features["ma20_above_ma60"] = float(
            close_proxy.tail(20).mean() > close_proxy.tail(60).mean()
        )
        raw.append(features)
    if any(not all(math.isfinite(value) for value in item.values()) for item in raw):
        raise ValueError("candidate ranking contains non-finite evidence")

    percentiles: dict[str, list[float]] = {}
    for key in raw[0]:
        values = [item[key] for item in raw]
        ranks = pd.Series(values).rank(method="average", pct=True).to_list()
        percentiles[key] = [float(value) for value in ranks]

    ranked: list[MidtermCandidate] = []
    for index, row in enumerate(rows):
        if row.symbol not in relative_strength:
            raise ValueError(
                f"{row.symbol}: full-universe {contract.relative_strength_sessions}-session "
                "relative strength missing"
            )
        horizon_momentum = percentiles["horizon_absolute_return"][index]
        downside_quality = 1.0 - percentiles["downside_volatility"][index]
        drawdown_quality = 1.0 - percentiles["max_drawdown"][index]
        risk_quality = 0.60 * downside_quality + 0.40 * drawdown_quality
        # The holding-period assessment now owns the largest share of the
        # ranking.  Its completed slow/primary bars and daily execution state
        # replace the legacy common daily-entry score.  The only price-return
        # rank is the selected horizon's own 5/10/20/60/120/252-session return;
        # longer horizons no longer reuse one shared 20/60/120 daily blend.
        # Full-universe relative strength is separate cross-sectional evidence;
        # neither input is a future return probability.
        signal = (
            0.55 * row.timeframe.score
            + 0.15 * horizon_momentum
            + 0.10 * float(relative_strength[row.symbol])
            + 0.20 * risk_quality
        )
        downside_capture = _downside_capture_ratio(row.returns, benchmark_returns)
        ranked.append(
            MidtermCandidate(
                symbol=row.symbol,
                name=row.name,
                industry=row.industry,
                signal_score=round(float(signal), 6),
                entry=row.entry,
                returns=row.returns,
                evidence_unknown=row.evidence_unknown,
                absolute_return_60=raw[index]["absolute_return_60"],
                relative_strength_percentile=float(relative_strength[row.symbol]),
                downside_capture_ratio=downside_capture,
                ma20_above_ma60=bool(raw[index]["ma20_above_ma60"]),
                timeframe=row.timeframe,
                horizon_absolute_return=raw[index]["horizon_absolute_return"],
                relative_strength_sessions=contract.relative_strength_sessions,
                risk_history_available=row.risk_history_available,
                risk_history_reasons=row.risk_history_reasons,
                risk_history_available_returns=row.risk_history_available_returns,
                risk_history_required_returns=row.risk_history_required_returns,
            )
        )
    return sorted(ranked, key=lambda item: (-item.signal_score, item.symbol))


def _beam_candidate_sets(
    candidates: list[MidtermCandidate],
    stock_count: int,
    beam_width: int,
) -> tuple[tuple[int, ...], ...]:
    correlations: dict[tuple[int, int], float] = {}
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            aligned = pd.concat(
                (candidates[left].returns, candidates[right].returns),
                axis=1,
                join="inner",
            ).dropna()
            correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            correlations[(left, right)] = correlation if math.isfinite(correlation) else 1.0

    states: list[tuple[int, ...]] = [()]
    for _depth in range(stock_count):
        expanded: list[tuple[int, ...]] = []
        for state in states:
            start = state[-1] + 1 if state else 0
            remaining = stock_count - len(state) - 1
            stop = len(candidates) - remaining
            for index in range(start, stop):
                expanded.append((*state, index))
        expanded.sort(
            key=lambda state: (
                -_partial_set_score(state, candidates, correlations),
                tuple(candidates[index].symbol for index in state),
            )
        )
        states = expanded[:beam_width]
        if not states:
            break
    return tuple(states)


def _partial_set_score(
    indices: tuple[int, ...],
    candidates: list[MidtermCandidate],
    correlations: Mapping[tuple[int, int], float],
) -> float:
    base = float(np.mean([candidates[index].signal_score for index in indices]))
    industries = {candidates[index].industry for index in indices}
    diversity = len(industries) / len(indices)
    pairs = [
        correlations[(left, right)]
        for offset, left in enumerate(indices)
        for right in indices[offset + 1 :]
    ]
    correlation_penalty = max(0.0, float(np.mean(pairs))) if pairs else 0.0
    return base + 0.04 * diversity - 0.10 * correlation_penalty


def _evaluate_candidate_set(
    selected: tuple[MidtermCandidate, ...],
    budget: AdaptiveRiskBudget,
) -> AdaptivePortfolioEvaluation:
    aligned = pd.concat(
        {candidate.symbol: candidate.returns for candidate in selected},
        axis=1,
        join="inner",
    ).dropna()
    adaptive = tuple(
        AdaptiveCandidate(
            symbol=candidate.symbol,
            industry=candidate.industry,
            signal_score=candidate.signal_score,
            returns=aligned[candidate.symbol],
        )
        for candidate in selected
    )
    return optimize_adaptive_portfolio(adaptive, budget=budget)


def _search_candidate_portfolios(
    search_pool: list[MidtermCandidate],
    *,
    budget: AdaptiveRiskBudget,
    beam_width: int,
    minimum_historical_return_lcb: float,
) -> tuple[
    dict[
        int,
        list[tuple[float, AdaptivePortfolioEvaluation, tuple[MidtermCandidate, ...]]],
    ],
    dict[int, list[_RejectedPortfolioEvaluation]],
    int,
]:
    """Evaluate bounded 3/4/5-name searches under one immutable budget.

    Structurally evaluable failures are retained only for an explicitly
    non-actionable observation layer.  They never enter ``viable_by_count``.
    """

    viable_by_count: dict[
        int,
        list[tuple[float, AdaptivePortfolioEvaluation, tuple[MidtermCandidate, ...]]],
    ] = {3: [], 4: [], 5: []}
    rejected_by_count: dict[int, list[_RejectedPortfolioEvaluation]] = {
        3: [],
        4: [],
        5: [],
    }
    evaluated = 0
    for stock_count in (3, 4, 5):
        if len(search_pool) < stock_count:
            continue
        for indices in _beam_candidate_sets(search_pool, stock_count, beam_width):
            selected = tuple(search_pool[index] for index in indices)
            try:
                adaptive = _evaluate_candidate_set(selected, budget)
            except AdaptivePortfolioDataError:
                continue
            evaluated += 1
            rejection_reasons = list(adaptive.risk_budget.violations)
            if adaptive.metrics.holding_period_return_lcb < minimum_historical_return_lcb:
                rejection_reasons.append("holding_period_return_lcb_below_minimum")
            if rejection_reasons:
                rejected_by_count[stock_count].append(
                    _RejectedPortfolioEvaluation(
                        evaluation=adaptive,
                        selected=selected,
                        rejection_reasons=tuple(rejection_reasons),
                        normalized_overrun=_normalized_rejection_overrun(
                            adaptive,
                            budget=budget,
                            minimum_historical_return_lcb=minimum_historical_return_lcb,
                        ),
                    )
                )
                continue
            viable_by_count[stock_count].append(
                (adaptive.metrics.holding_period_return_lcb, adaptive, selected)
            )
    return viable_by_count, rejected_by_count, evaluated


def _select_observation_portfolio(
    rejected_by_count: Mapping[int, list[_RejectedPortfolioEvaluation]],
) -> _RejectedPortfolioEvaluation:
    """Select one deterministic rejected allocation for observation only.

    Four names remain the attention default.  If no four-name set can even be
    evaluated structurally, use five and then three.  Within that count choose
    fewer violated gates, smaller normalized excess, the higher historical LCB,
    and finally the lexicographically smaller symbol tuple.
    """

    for stock_count in (4, 5, 3):
        rows = rejected_by_count.get(stock_count, [])
        if rows:
            return min(rows, key=_observation_sort_key)
    raise RuntimeError("at least one rejected observation portfolio is required")


def _observation_sort_key(
    row: _RejectedPortfolioEvaluation,
) -> tuple[int, float, float, tuple[str, ...]]:
    return (
        len(row.rejection_reasons),
        row.normalized_overrun,
        -row.evaluation.metrics.holding_period_return_lcb,
        tuple(sorted(candidate.symbol for candidate in row.selected)),
    )


def _normalized_rejection_overrun(
    evaluation: AdaptivePortfolioEvaluation,
    *,
    budget: AdaptiveRiskBudget,
    minimum_historical_return_lcb: float,
) -> float:
    """Return a dimensionless, deterministic diagnostic constraint excess."""

    metrics = evaluation.metrics

    def positive_ratio(value: float, limit: float) -> float:
        return max(0.0, value / limit - 1.0)

    industry_max = max(weight for _industry, weight in evaluation.industry_weights)
    correlation_scale = max(1e-6, 1.0 - budget.max_down_period_correlation)
    horizon_drawdown_limit = (
        budget.max_horizon_rolling_drawdown_p90
        if budget.max_horizon_rolling_drawdown_p90 is not None
        else budget.max_rolling_drawdown_60_p90
    )
    return float(
        positive_ratio(
            metrics.annual_downside_volatility,
            budget.max_annual_downside_volatility,
        )
        + positive_ratio(
            metrics.horizon_rolling_max_drawdown_p90,
            horizon_drawdown_limit,
        )
        + positive_ratio(metrics.es95_5d, budget.max_es95_5d)
        + max(
            0.0,
            (metrics.max_down_period_correlation - budget.max_down_period_correlation)
            / correlation_scale,
        )
        + positive_ratio(
            metrics.max_position_downside_risk_contribution,
            budget.max_position_downside_risk_contribution,
        )
        + positive_ratio(industry_max, budget.industry_weight_limit)
        + max(
            0.0,
            minimum_historical_return_lcb - metrics.holding_period_return_lcb,
        )
        / max(abs(minimum_historical_return_lcb), 0.01)
    )


def _viable_sort_key(
    row: tuple[float, AdaptivePortfolioEvaluation, tuple[MidtermCandidate, ...]],
) -> tuple[float, float, float, float, float, float, tuple[str, ...]]:
    objective, evaluation, selected = row
    metrics = evaluation.metrics
    return (
        -objective,
        metrics.annual_downside_volatility,
        metrics.horizon_rolling_max_drawdown_p90,
        metrics.es95_5d,
        metrics.max_down_period_correlation,
        metrics.max_position_downside_risk_contribution,
        tuple(sorted(item.symbol for item in selected)),
    )


def _select_stock_count(
    viable_by_count: Mapping[
        int,
        list[tuple[float, AdaptivePortfolioEvaluation, tuple[MidtermCandidate, ...]]],
    ],
) -> tuple[float, AdaptivePortfolioEvaluation, tuple[MidtermCandidate, ...]]:
    """Choose four normally; use three for scarcity and five for real diversification.

    A viable four-stock set is the baseline.  A five-stock set may replace it
    only when it has a higher conservative return lower bound, does not worsen
    the three path/tail risk measures by more than 50 basis points, and improves
    either down-period correlation by five points or maximum single-name risk
    contribution by three points.  Three stocks are selected only when no
    four-stock set passes every hard risk budget.  If a five-stock set is the
    only diversified feasible construction, it remains preferable to a
    concentrated three-stock fallback.
    """

    best_three = viable_by_count.get(3, [])[:1]
    best_four = viable_by_count.get(4, [])[:1]
    best_five = viable_by_count.get(5, [])[:1]
    if best_four:
        baseline = best_four[0]
        for five_stock_row in viable_by_count.get(5, []):
            if _five_stock_diversification_is_material(baseline[1], five_stock_row[1]):
                return five_stock_row
        return baseline
    if best_five:
        return best_five[0]
    if best_three:
        return best_three[0]
    raise RuntimeError("at least one viable 3-to-5 stock set is required")


def _five_stock_diversification_is_material(
    four: AdaptivePortfolioEvaluation,
    five: AdaptivePortfolioEvaluation,
) -> bool:
    four_metrics = four.metrics
    five_metrics = five.metrics
    if five_metrics.holding_period_return_lcb <= four_metrics.holding_period_return_lcb:
        return False
    path_and_tail_not_worse = (
        five_metrics.annual_downside_volatility <= four_metrics.annual_downside_volatility + 0.005
        and five_metrics.horizon_rolling_max_drawdown_p90
        <= four_metrics.horizon_rolling_max_drawdown_p90 + 0.005
        and five_metrics.es95_5d <= four_metrics.es95_5d + 0.005
    )
    diversification_improved = (
        five_metrics.max_down_period_correlation <= four_metrics.max_down_period_correlation - 0.05
        or five_metrics.max_position_downside_risk_contribution
        <= four_metrics.max_position_downside_risk_contribution - 0.03
    )
    return path_and_tail_not_worse and diversification_improved
