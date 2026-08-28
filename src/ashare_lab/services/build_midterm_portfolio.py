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
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from ashare_lab.analytics.adaptive_portfolio import (
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

RESEARCH_DISCLAIMER = (
    "本结果是共同截止日历史数据上的确定性研究筛选。持有期收益下界尚未经过严格的"
    "point-in-time walk-forward、费用、滑点和不可成交验证，不是未来收益预测、上涨概率、"
    "投资建议或最大回撤保证。"
)
MIDTERM_METHOD_VERSION = "midterm-maintrend-v0.6.0"


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
    """One-week conditional entry level; never an unconditional buy price."""

    HEALTHY_PULLBACK = "healthy_pullback_range"
    RECLAIM = "reclaim_close_confirmation"
    VOLUME_BREAKOUT = "volume_breakout_close_confirmation"


@dataclass(frozen=True, slots=True)
class ConditionalEntryPlan:
    """Structured one-week price plan generated at the common cutoff.

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


@dataclass(frozen=True, slots=True)
class MidtermResearchCandidate:
    rank: int
    symbol: str
    name: str
    industry: str
    signal_score: float
    entry_pattern: EntryPattern
    breakout_line: float
    days_since_breakout: int
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


@dataclass(frozen=True, slots=True)
class _RejectedPortfolioEvaluation:
    """One structurally evaluable allocation that failed a research gate."""

    evaluation: AdaptivePortfolioEvaluation
    selected: tuple[MidtermCandidate, ...]
    rejection_reasons: tuple[str, ...]
    normalized_overrun: float


_MOMENTUM_WEIGHTS: dict[int, tuple[tuple[int, float], ...]] = {
    1: ((5, 0.50), (20, 0.30), (60, 0.20)),
    # Two weeks deliberately avoids five-session chase.  Ten-day momentum leads
    # at 50%, while 20-day confirmation and a modest 60-day trend anchor retain
    # 30% and 20% so one fresh move cannot erase the short/medium-term context.
    2: ((10, 0.50), (20, 0.30), (60, 0.20)),
    4: ((20, 0.50), (60, 0.35), (120, 0.15)),
    13: ((20, 0.20), (60, 0.50), (120, 0.30)),
    26: ((20, 0.10), (60, 0.35), (120, 0.55)),
    52: ((20, 0.10), (60, 0.25), (120, 0.65)),
}

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

    if holding_weeks not in _MOMENTUM_WEIGHTS:
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

    raw_candidates: list[tuple[str, str, str, EntryReadinessAssessment, pd.Series, tuple[str, ...]]]
    raw_candidates = []
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
        entry = assess_entry_readiness(frame, as_of=cutoff)
        if not entry.ready:
            exclusions.append(CandidateExclusion(symbol, entry.reasons))
            continue
        try:
            returns = _returns_at_cutoff(frame, cutoff)
        except ValueError as exc:
            exclusions.append(CandidateExclusion(symbol, (f"invalid_returns:{exc}",)))
            continue
        unknown = _unknown_evidence(item)
        raw_candidates.append(
            (
                symbol,
                str(item.get("name", symbol)).strip() or symbol,
                str(item.get("industry", "")).strip(),
                entry,
                returns,
                unknown,
            )
        )

    benchmark_returns = _core_index_benchmark_returns(market_index_histories, cutoff)
    relative_strength_60 = _full_universe_relative_strength_60(
        normalized_histories,
        cutoff,
    )
    candidates = (
        _rank_candidates(
            raw_candidates,
            holding_weeks,
            relative_strength_60,
            benchmark_returns,
        )
        if raw_candidates
        else []
    )
    candidate_actions = {
        candidate.symbol: _candidate_action(candidate, price_cycle) for candidate in candidates
    }
    research_candidates = _build_research_shortlist(
        candidates,
        candidate_actions,
        histories=normalized_histories,
        cutoff=cutoff,
    )

    if len(candidates) < 3:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            research_candidates=research_candidates,
            entry_ready_count=len(candidates),
            actionable_candidate_count=sum(
                action is CandidateAction.CONDITIONAL_ENTRY
                for action, _reasons in candidate_actions.values()
            ),
            exclusions=tuple(exclusions),
            market_regime=market_regime,
            index_regime=index_regime,
            price_cycle=price_cycle,
            reasons=(f"only_{len(candidates)}_entry_ready_candidates;minimum_three",),
        )

    holding_sessions = HOLDING_PERIOD_SESSIONS[holding_weeks]
    budget = _resolve_risk_budget(
        price_cycle,
        risk_budget,
        holding_sessions=holding_sessions,
    )

    # Research-set optimisation is independent from current deployment
    # permission.  This lets a defensive market or incomplete review evidence
    # retain a diversified, risk-evaluated 3--5-name research set while the
    # action layer correctly stays at 100% cash.
    research_pool = candidates[:candidate_pool_size]
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
        for candidate in candidates
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
            entry_ready_count=len(candidates),
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
            entry_ready_count=len(candidates),
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
            entry_pattern=selected_by_symbol[position.symbol].entry.pattern,
            breakout_line=float(selected_by_symbol[position.symbol].entry.breakout_line),
            days_since_breakout=int(selected_by_symbol[position.symbol].entry.days_since_breakout),
            annual_downside_volatility=position.annual_downside_volatility,
            downside_risk_contribution=position.downside_risk_contribution,
            evidence_unknown=selected_by_symbol[position.symbol].evidence_unknown,
            operational_account_weight=position.weight,
            operational_stock_sleeve_weight=position.weight / best.stock_exposure,
            conditional_entry_plan=entry_plan_by_symbol.get(position.symbol),
            price_observation_plan=price_observation_plan_by_symbol.get(position.symbol),
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
        entry_ready_count=len(candidates),
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


def _full_universe_relative_strength_60(
    histories: Mapping[str, pd.DataFrame],
    cutoff: pd.Timestamp,
) -> dict[str, float]:
    """Return exact 60-session return percentiles across the supplied universe."""

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
            len(series) < 61
            or series.index.has_duplicates
            or series.index[-1] != cutoff
            or bool((series <= 0.0).any())
        ):
            continue
        value = float(series.iloc[-1] / series.iloc[-61] - 1.0)
        if math.isfinite(value):
            values[symbol] = value
    if not values:
        return {}
    ranked = pd.Series(values, dtype=float).rank(method="average", pct=True)
    return {str(symbol): float(value) for symbol, value in ranked.items()}


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
    )


def _candidate_action(
    candidate: MidtermCandidate,
    cycle: PriceCycleAssessment | None,
) -> tuple[CandidateAction, tuple[str, ...]]:
    """Apply cycle-specific entry confirmation without changing research rank."""

    if candidate.evidence_unknown:
        return (
            CandidateAction.WAIT_CONFIRMATION,
            ("fundamental_announcement_or_execution_evidence_requires_review",),
        )
    strictness = EntryStrictness.STANDARD if cycle is None else cycle.policy.entry_strictness
    if strictness is EntryStrictness.STANDARD:
        return CandidateAction.CONDITIONAL_ENTRY, ("standard_entry_structure_confirmed",)

    entry = candidate.entry
    checks: tuple[tuple[bool, str], ...]
    if strictness is EntryStrictness.TIGHT:
        checks = (
            (candidate.absolute_return_60 > 0.0, "sixty_session_absolute_return_not_positive"),
            (
                entry.breakout_amount_ratio is not None and entry.breakout_amount_ratio >= 1.20,
                "breakout_amount_ratio_below_1_20",
            ),
            (
                entry.distance_ma20_ratio is not None and entry.distance_ma20_ratio <= 0.08,
                "distance_ma20_ratio_above_8pct",
            ),
            (
                entry.distance_ma20_atr is not None and entry.distance_ma20_atr <= 2.0,
                "distance_ma20_atr_above_2",
            ),
        )
    else:
        checks = (
            (candidate.absolute_return_60 > 0.0, "sixty_session_absolute_return_not_positive"),
            (
                candidate.relative_strength_percentile >= 0.90,
                "relative_strength_not_top_decile",
            ),
            (candidate.ma20_above_ma60, "ma20_not_above_ma60"),
            (
                candidate.downside_capture_ratio is not None
                and candidate.downside_capture_ratio <= 0.80,
                "downside_capture_above_0_80_or_unavailable",
            ),
            (
                entry.breakout_amount_ratio is not None and entry.breakout_amount_ratio >= 1.30,
                "breakout_amount_ratio_below_1_30",
            ),
            (
                entry.distance_ma20_ratio is not None and entry.distance_ma20_ratio <= 0.06,
                "distance_ma20_ratio_above_6pct",
            ),
            (
                entry.distance_ma20_atr is not None and entry.distance_ma20_atr <= 1.50,
                "distance_ma20_atr_above_1_5",
            ),
        )
        if strictness is EntryStrictness.EXCEPTION_ONLY:
            checks = (
                *checks,
                (
                    entry.pattern in {EntryPattern.HEALTHY_PULLBACK, EntryPattern.BREAKOUT_RECLAIM},
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
        price_observation_plan = _build_one_week_price_observation_plan(
            histories.get(candidate.symbol),
            cutoff=cutoff,
            entry_pattern=candidate.entry.pattern,
        )
        rows.append(
            MidtermResearchCandidate(
                rank=rank,
                symbol=candidate.symbol,
                name=candidate.name,
                industry=candidate.industry,
                signal_score=candidate.signal_score,
                entry_pattern=candidate.entry.pattern,
                breakout_line=float(candidate.entry.breakout_line),
                days_since_breakout=int(candidate.entry.days_since_breakout),
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
                    else None
                ),
                observation_stock_sleeve_weight=(
                    None
                    if candidate.symbol not in observation_by_symbol
                    else observation_by_symbol[candidate.symbol].weight
                    / observation_evaluation.stock_exposure
                ),
                price_observation_plan=price_observation_plan,
            )
        )
    return tuple(rows)


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
        return ConditionalEntryPlan(
            kind=ConditionalEntryPlanKind.VOLUME_BREAKOUT,
            trigger_price=one_week.breakout_trigger,
            confirmation_rule=one_week.breakout_confirmation_rule,
            **common,
        )
    return None


def _rank_candidates(
    rows: list[tuple[str, str, str, EntryReadinessAssessment, pd.Series, tuple[str, ...]]],
    holding_weeks: int,
    relative_strength_60: Mapping[str, float],
    benchmark_returns: pd.Series | None = None,
) -> list[MidtermCandidate]:
    raw: list[dict[str, float]] = []
    for _, _, _, entry, returns, _ in rows:
        close_proxy = (1.0 + returns).cumprod()
        features: dict[str, float] = {}
        for sessions, _weight in _MOMENTUM_WEIGHTS[holding_weeks]:
            features[f"momentum_{sessions}"] = (
                float(close_proxy.iloc[-1] / close_proxy.iloc[-sessions - 1] - 1.0)
                if len(close_proxy) > sessions
                else float("nan")
            )
        downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
        features["downside_volatility"] = float(
            np.sqrt(np.mean(np.square(downside))) * math.sqrt(252)
        )
        equity = (1.0 + returns).cumprod()
        features["max_drawdown"] = abs(float((equity / equity.cummax() - 1.0).min()))
        features["entry"] = entry.score
        features["absolute_return_60"] = float(close_proxy.iloc[-1] / close_proxy.iloc[-61] - 1.0)
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
    for index, (symbol, name, industry, entry, returns, unknown) in enumerate(rows):
        if symbol not in relative_strength_60:
            raise ValueError(f"{symbol}: full-universe 60-session relative strength missing")
        momentum = sum(
            weight * percentiles[f"momentum_{sessions}"][index]
            for sessions, weight in _MOMENTUM_WEIGHTS[holding_weeks]
        )
        downside_quality = 1.0 - percentiles["downside_volatility"][index]
        drawdown_quality = 1.0 - percentiles["max_drawdown"][index]
        risk_quality = 0.60 * downside_quality + 0.40 * drawdown_quality
        signal = 0.55 * entry.score + 0.25 * momentum + 0.20 * risk_quality
        downside_capture = _downside_capture_ratio(returns, benchmark_returns)
        ranked.append(
            MidtermCandidate(
                symbol=symbol,
                name=name,
                industry=industry,
                signal_score=round(float(signal), 6),
                entry=entry,
                returns=returns,
                evidence_unknown=unknown,
                absolute_return_60=raw[index]["absolute_return_60"],
                # The defensive entry contract is explicitly a 60-session
                # relative-strength test.  Keep it separate from the
                # horizon-dependent blended momentum used in ranking.
                relative_strength_percentile=float(relative_strength_60[symbol]),
                downside_capture_ratio=downside_capture,
                ma20_above_ma60=bool(raw[index]["ma20_above_ma60"]),
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
    return float(
        positive_ratio(
            metrics.annual_downside_volatility,
            budget.max_annual_downside_volatility,
        )
        + positive_ratio(
            metrics.rolling_max_drawdown_60_p90,
            budget.max_rolling_drawdown_60_p90,
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
        metrics.rolling_max_drawdown_60_p90,
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
        and five_metrics.rolling_max_drawdown_60_p90
        <= four_metrics.rolling_max_drawdown_60_p90 + 0.005
        and five_metrics.es95_5d <= four_metrics.es95_5d + 0.005
    )
    diversification_improved = (
        five_metrics.max_down_period_correlation <= four_metrics.max_down_period_correlation - 0.05
        or five_metrics.max_position_downside_risk_contribution
        <= four_metrics.max_position_downside_risk_contribution - 0.03
    )
    return path_and_tail_not_worse and diversification_improved
