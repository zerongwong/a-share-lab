"""Daily, close-confirmed research review of the user's persistent holdings.

The service manages existing positions; it does not rerank candidates, replace
membership, send notifications, connect to a broker, or place orders.  Its
protective line is a documented research default inspired by pivot/trailing
ideas associated with Edwards--Magee.  It is not claimed to reproduce an
unavailable book chapter or a uniquely correct stop algorithm.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import ROUND_FLOOR, Decimal
from enum import StrEnum
from math import isfinite
from typing import Final

import pandas as pd

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.analytics.indicators import atr
from ashare_lab.analytics.multi_timeframe import (
    ExecutionState,
    MultiTimeframeAssessment,
    MultiTimeframeDataError,
    StructureState,
    TrendDirection,
    assess_multi_timeframe,
    build_completed_timeframes,
)
from ashare_lab.analytics.trend import confirmed_swings
from ashare_lab.ports.market_data import normalize_symbol
from ashare_lab.services.holding_ledger import (
    ActiveHolding,
    ActiveHoldingPortfolio,
    HoldingKnowledgeContext,
    get_active_holding_portfolio,
    resolve_current_holding_context,
)

HOLDING_TREE_METHOD_VERSION: Final = "magee-inspired-pivot-trailing-v0.1.0"
ATR_BUFFER_MULTIPLE: Final = 0.50


class HoldingAction(StrEnum):
    HOLD = "hold"
    TIGHTEN = "tighten"
    REDUCE = "reduce"
    EXIT = "exit"
    REVIEW = "review"


class HoldingReviewRowStatus(StrEnum):
    READY = "ready"
    DATA_NOT_READY = "data_not_ready"


class HoldingReviewSummaryStatus(StrEnum):
    READY = "ready"
    PARTIAL = "partial"
    DATA_NOT_READY = "data_not_ready"
    NO_HOLDINGS = "no_holdings"


@dataclass(frozen=True, slots=True)
class CompanyActionClearance:
    """Independent or explicit-local evidence covering the review cutoff."""

    symbol: str
    through_date: date
    clear: bool
    source: str
    evidence_id: str


@dataclass(frozen=True, slots=True)
class HoldingTreeReviewRow:
    symbol: str
    name: str
    holding_weeks: int
    holding_version: int
    position_key: str
    status: HoldingReviewRowStatus
    action: HoldingAction
    latest_close: float | None
    cost_price: float | None
    stock_sleeve_weight: float
    account_weight: float | None
    candidate_stop: float | None
    previous_stop: float | None
    effective_stop: float | None
    stop_raised: bool
    close_below_stop: bool | None
    source_timeframe: str | None
    evidence_date: date | None
    slow_direction: str | None
    primary_structure: str | None
    daily_execution: str | None
    reasons: tuple[str, ...]
    decision_layer: str = "holding_management"
    candidate_rank_used: bool = False
    next_session_only: bool = True
    auto_order_allowed: bool = False
    replacement_requested: bool = False
    company_action_clear: bool | None = None
    company_action_evidence_id: str | None = None
    company_action_evidence_source: str | None = None
    company_action_clear_through: date | None = None
    method_version: str = HOLDING_TREE_METHOD_VERSION

    @property
    def urgent(self) -> bool:
        return self.action in {HoldingAction.REDUCE, HoldingAction.EXIT} or (
            self.action is HoldingAction.REVIEW
            and any("company_action_evidence_blocks" in reason for reason in self.reasons)
        )


@dataclass(frozen=True, slots=True)
class HoldingTreeReviewSummary:
    status: HoldingReviewSummaryStatus
    portfolio_id: str | None
    holding_version: int | None
    holding_weeks: int | None
    reviewed_at: datetime
    data_cutoff: date | None
    rows: tuple[HoldingTreeReviewRow, ...]
    reasons: tuple[str, ...] = ()
    method_version: str = HOLDING_TREE_METHOD_VERSION
    membership_changed: bool = False
    holding_weeks_changed: bool = False
    auto_order_allowed: bool = False

    @property
    def urgent_rows(self) -> tuple[HoldingTreeReviewRow, ...]:
        return tuple(row for row in self.rows if row.urgent)


def build_holding_tree_review(
    repository: SQLiteRepository,
    histories: Mapping[str, pd.DataFrame],
    *,
    as_of: object,
    verified_data_cutoff: object | None = None,
    verified_close: bool = True,
    reviewed_at: datetime | None = None,
    persist: bool = True,
    company_action_clear_by_symbol: Mapping[str, CompanyActionClearance] | None = None,
    holding_context: HoldingKnowledgeContext | None = None,
) -> HoldingTreeReviewSummary:
    """Stable integration alias for the evening digest and scheduled sync."""

    return review_active_holdings(
        repository,
        histories,
        as_of=as_of,
        verified_data_cutoff=verified_data_cutoff,
        verified_close=verified_close,
        reviewed_at=reviewed_at,
        persist=persist,
        company_action_clear_by_symbol=company_action_clear_by_symbol,
        holding_context=holding_context,
    )


def review_active_holdings(
    repository: SQLiteRepository,
    histories: Mapping[str, pd.DataFrame],
    *,
    as_of: object,
    verified_data_cutoff: object | None = None,
    verified_close: bool = True,
    reviewed_at: datetime | None = None,
    persist: bool = True,
    company_action_clear_by_symbol: Mapping[str, CompanyActionClearance] | None = None,
    holding_context: HoldingKnowledgeContext | None = None,
) -> HoldingTreeReviewSummary:
    """Review the newest explicit holding snapshot on verified completed bars.

    A rank falling from yesterday's candidate list is intentionally ignored.
    ``EXIT`` or ``REDUCE`` is a next-session research response to completed-bar
    weakness, not an automatic order.  Removing a position still requires a
    separate explicit holding-ledger update from the user.
    """

    review_time = _review_time(reviewed_at)
    as_of_date = _as_date(as_of)
    cutoff = _as_date(as_of if verified_data_cutoff is None else verified_data_cutoff)
    if holding_context is None:
        portfolio = get_active_holding_portfolio(
            repository,
            as_of=min(as_of_date, cutoff),
        )
    else:
        portfolio = resolve_current_holding_context(repository, holding_context)
    if portfolio is None or portfolio.status != "active" or not portfolio.positions:
        return HoldingTreeReviewSummary(
            status=HoldingReviewSummaryStatus.NO_HOLDINGS,
            portfolio_id=(None if portfolio is None else portfolio.id),
            holding_version=(None if portfolio is None else portfolio.version),
            holding_weeks=(None if portfolio is None else portfolio.holding_weeks),
            reviewed_at=review_time,
            data_cutoff=None,
            rows=(),
            reasons=("no_user_confirmed_active_holdings",),
        )

    if cutoff > as_of_date:
        return _global_failure(
            portfolio,
            review_time=review_time,
            reason="verified_cutoff_after_as_of_rejected",
        )
    if holding_context is not None and holding_context.known_at > review_time:
        return _global_failure(
            portfolio,
            review_time=review_time,
            reason="holding_knowledge_time_after_review_time_rejected",
        )
    normalized_histories = _normalize_histories(histories)
    clearances = _normalize_company_action_clearances(company_action_clear_by_symbol or {})
    rows = tuple(
        _review_one(
            repository,
            portfolio,
            holding,
            normalized_histories.get(holding.symbol),
            cutoff=cutoff,
            verified_close=verified_close,
            review_time=review_time,
            persist=persist,
            company_action_clearance=clearances.get(holding.symbol),
        )
        for holding in portfolio.positions
        if holding.status == "active"
    )
    ready_count = sum(row.status is HoldingReviewRowStatus.READY for row in rows)
    if ready_count == len(rows):
        status = HoldingReviewSummaryStatus.READY
        reasons: tuple[str, ...] = ()
    elif ready_count:
        status = HoldingReviewSummaryStatus.PARTIAL
        reasons = ("one_or_more_holdings_failed_closed",)
    else:
        status = HoldingReviewSummaryStatus.DATA_NOT_READY
        reasons = ("all_holdings_failed_closed",)
    return HoldingTreeReviewSummary(
        status=status,
        portfolio_id=portfolio.id,
        holding_version=portfolio.version,
        holding_weeks=portfolio.holding_weeks,
        reviewed_at=review_time,
        data_cutoff=cutoff,
        rows=rows,
        reasons=reasons,
    )


def _review_one(
    repository: SQLiteRepository,
    portfolio: ActiveHoldingPortfolio,
    holding: ActiveHolding,
    frame: pd.DataFrame | None,
    *,
    cutoff: date,
    verified_close: bool,
    review_time: datetime,
    persist: bool,
    company_action_clearance: CompanyActionClearance | None,
) -> HoldingTreeReviewRow:
    if not verified_close:
        return _failed_row(
            repository,
            portfolio,
            holding,
            cutoff=cutoff,
            review_time=review_time,
            reason="close_not_verified",
            persist=persist,
        )
    if holding.entry_date > cutoff:
        return _failed_row(
            repository,
            portfolio,
            holding,
            cutoff=cutoff,
            review_time=review_time,
            reason="holding_entry_date_after_verified_cutoff",
            persist=persist,
        )
    if frame is None:
        return _failed_row(
            repository,
            portfolio,
            holding,
            cutoff=cutoff,
            review_time=review_time,
            reason="holding_history_missing",
            persist=persist,
        )

    stored = repository.get_holding_protective_stop(holding.position_key)
    if stored is not None and _stored_stop_cutoff(stored) > cutoff:
        return _failed_row(
            repository,
            portfolio,
            holding,
            cutoff=cutoff,
            review_time=review_time,
            reason="protective_stop_cutoff_after_review_rejected",
            persist=persist,
        )

    try:
        bars = build_completed_timeframes(frame, as_of=cutoff)
        if bars.data_cutoff.date() != cutoff:
            raise MultiTimeframeDataError("holding_close_cutoff_mismatch")
        assessment = assess_multi_timeframe(
            frame,
            as_of=cutoff,
            holding_weeks=portfolio.holding_weeks,
        )
        if (
            assessment.slow_direction.direction is TrendDirection.INSUFFICIENT
            or assessment.structure.state is StructureState.INSUFFICIENT
            or assessment.execution.state is ExecutionState.INSUFFICIENT
        ):
            raise MultiTimeframeDataError("holding_multitimeframe_history_insufficient")
        candidate = _candidate_stop(
            bars,
            assessment,
            entry_date=holding.entry_date,
        )
    except (MultiTimeframeDataError, ValueError, TypeError) as exc:
        return _failed_row(
            repository,
            portfolio,
            holding,
            cutoff=cutoff,
            review_time=review_time,
            reason=f"holding_data_not_ready:{exc}",
            persist=persist,
        )

    latest_close = float(bars.daily.iloc[-1]["close"])
    if stored is None:
        previous_stop = None
    elif _stored_stop_cutoff(stored) == cutoff:
        previous_stop = None if stored["previous_stop"] is None else float(stored["previous_stop"])
    else:
        previous_stop = float(stored["effective_stop"])
    effective_stop = max(candidate.stop, previous_stop or candidate.stop)
    effective_stop = _price_floor(effective_stop)
    stop_raised = previous_stop is not None and effective_stop > previous_stop + 0.005
    close_below_stop = latest_close < effective_stop
    raw_action, action_reasons = _holding_action(
        assessment,
        close_below_stop=close_below_stop,
        stop_raised=stop_raised,
    )
    company_action_clear = bool(
        company_action_clearance is not None
        and company_action_clearance.clear
        and company_action_clearance.through_date >= cutoff
    )
    company_action_detected = bool(
        company_action_clearance is not None and not company_action_clearance.clear
    )
    if company_action_clear:
        action = raw_action
        row_status = HoldingReviewRowStatus.READY
        company_action_reasons = ("independent_company_action_clearance_covers_cutoff",)
    elif raw_action is HoldingAction.HOLD and not company_action_detected:
        action = HoldingAction.HOLD
        row_status = HoldingReviewRowStatus.READY
        company_action_reasons = (
            "company_action_clearance_missing_non_destructive_hold_only",
            "candidate_stop_not_persisted_without_company_action_clearance",
        )
    else:
        action = HoldingAction.REVIEW
        row_status = HoldingReviewRowStatus.DATA_NOT_READY
        evidence_state = (
            "company_action_detected"
            if company_action_detected
            else "company_action_clearance_missing_or_stale"
        )
        company_action_reasons = (
            f"company_action_evidence_blocks_{raw_action.value}:{evidence_state}",
            "verify_ex_rights_dividend_bonus_and_allotment_announcements_before_action",
            "candidate_stop_not_persisted_without_company_action_clearance",
        )
    reasons = (
        "holding_management_is_separate_from_candidate_ranking",
        "daily_rank_decline_does_not_trigger_replacement",
        f"primary_stop_source:{candidate.source_timeframe}",
        f"atr_buffer_multiple:{ATR_BUFFER_MULTIPLE:.2f}",
        *action_reasons,
        *company_action_reasons,
    )
    row = HoldingTreeReviewRow(
        symbol=holding.symbol,
        name=holding.name,
        holding_weeks=portfolio.holding_weeks,
        holding_version=portfolio.version,
        position_key=holding.position_key,
        status=row_status,
        action=action,
        latest_close=latest_close,
        cost_price=holding.cost_price,
        stock_sleeve_weight=holding.stock_sleeve_weight,
        account_weight=holding.account_weight,
        candidate_stop=candidate.stop,
        previous_stop=previous_stop,
        effective_stop=effective_stop,
        stop_raised=stop_raised,
        close_below_stop=close_below_stop,
        source_timeframe=candidate.source_timeframe,
        evidence_date=candidate.evidence_date,
        slow_direction=assessment.slow_direction.direction.value,
        primary_structure=assessment.structure.state.value,
        daily_execution=assessment.execution.state.value,
        reasons=reasons,
        company_action_clear=(
            None if company_action_clearance is None else company_action_clearance.clear
        ),
        company_action_evidence_id=(
            None if company_action_clearance is None else company_action_clearance.evidence_id
        ),
        company_action_evidence_source=(
            None if company_action_clearance is None else company_action_clearance.source
        ),
        company_action_clear_through=(
            None if company_action_clearance is None else company_action_clearance.through_date
        ),
    )
    if persist:
        evidence_hash = _evidence_hash(
            holding,
            cutoff,
            bars.daily.iloc[-1].to_dict(),
            candidate.stop,
            company_action_clearance,
        )
        stop_state = None
        if company_action_clear:
            stop_state = {
                "position_key": holding.position_key,
                "symbol": holding.symbol,
                "entry_date": holding.entry_date,
                "effective_stop": effective_stop,
                "candidate_stop": candidate.stop,
                "previous_stop": previous_stop,
                "data_cutoff": cutoff,
                "source_timeframe": candidate.source_timeframe,
                "evidence_date": candidate.evidence_date,
                "holding_version": portfolio.version,
                "method_version": HOLDING_TREE_METHOD_VERSION,
                "details_json": {
                    "atr14": candidate.atr14,
                    "atr_cutoff": candidate.atr_cutoff.isoformat(),
                    "atr_buffer_multiple": ATR_BUFFER_MULTIPLE,
                    "support": candidate.support,
                    "support_kind": candidate.support_kind,
                    "calendar_boundary": "existing_multi_timeframe_conservative_fallback",
                    "company_action_evidence_id": company_action_clearance.evidence_id,
                },
                "updated_at": review_time,
            }
        try:
            repository.record_holding_review(
                _review_record(
                    portfolio,
                    holding,
                    row,
                    cutoff=cutoff,
                    review_time=review_time,
                    evidence_hash=evidence_hash,
                ),
                stop_state=stop_state,
            )
        except ValueError as exc:
            if str(exc) != "Holding protection data cutoff cannot move backwards":
                raise
            return _failed_row(
                repository,
                portfolio,
                holding,
                cutoff=cutoff,
                review_time=review_time,
                reason="protective_stop_cutoff_after_review_rejected",
                persist=True,
            )
    return row


@dataclass(frozen=True, slots=True)
class _CandidateStop:
    stop: float
    support: float
    atr14: float
    support_kind: str
    source_timeframe: str
    evidence_date: date
    atr_cutoff: date


def _candidate_stop(
    bars: object,
    assessment: MultiTimeframeAssessment,
    *,
    entry_date: date,
) -> _CandidateStop:
    contract = assessment.contract
    primary = bars.for_timeframe(contract.structure_timeframe)
    if primary.empty:
        raise MultiTimeframeDataError("primary_structure_bars_missing")
    window = primary.tail(contract.structure_lookback_bars + 5).reset_index(drop=True)
    lows = [
        point
        for point in confirmed_swings(window, left=2, right=2)
        if point.kind == "low" and point.trade_date.date() > entry_date
    ]
    if lows:
        point = lows[-1]
        support = float(point.price)
        evidence_date = point.trade_date.date()
        atr_cutoff = point.confirmed_at.date()
        support_kind = "confirmed_reaction_low"
    else:
        entry_bars = build_completed_timeframes(bars.daily, as_of=entry_date)
        entry_primary = entry_bars.for_timeframe(contract.structure_timeframe)
        base_count = min(contract.structure_base_bars, max(1, len(entry_primary) - 1))
        base = entry_primary.iloc[-base_count - 1 : -1]
        if base.empty:
            raise MultiTimeframeDataError("entry_cutoff_primary_structure_floor_unavailable")
        support_index = base["low"].astype(float).idxmin()
        support = float(base.loc[support_index, "low"])
        evidence_date = pd.Timestamp(base.loc[support_index, "trade_date"]).date()
        atr_cutoff = entry_bars.data_cutoff.date()
        support_kind = "entry_cutoff_primary_structure_floor"
    atr_rows = bars.daily.loc[pd.to_datetime(bars.daily["trade_date"]).dt.date <= atr_cutoff]
    daily_atr = float(atr(atr_rows, 14).iloc[-1])
    if not isfinite(daily_atr) or daily_atr <= 0.0:
        raise MultiTimeframeDataError("atr14_unavailable_at_stop_evidence_cutoff")
    candidate = _price_floor(support - ATR_BUFFER_MULTIPLE * daily_atr)
    if not isfinite(candidate) or candidate <= 0.0:
        raise MultiTimeframeDataError("candidate_protection_line_invalid")
    return _CandidateStop(
        stop=candidate,
        support=support,
        atr14=daily_atr,
        support_kind=support_kind,
        source_timeframe=contract.structure_timeframe.value,
        evidence_date=evidence_date,
        atr_cutoff=atr_cutoff,
    )


def _holding_action(
    assessment: MultiTimeframeAssessment,
    *,
    close_below_stop: bool,
    stop_raised: bool,
) -> tuple[HoldingAction, tuple[str, ...]]:
    if close_below_stop:
        return HoldingAction.EXIT, (
            "complete_close_confirmed_below_effective_stop",
            "handle_only_at_next_tradable_session_subject_to_t_plus_one_suspension_and_limit_down",
            "exited_weight_remains_cash_until_explicit_new_plan",
        )
    weakness_signals: list[str] = []
    if assessment.slow_direction.direction is TrendDirection.DOWN:
        weakness_signals.append("slow_direction_down")
    if assessment.structure.state is StructureState.FAILED:
        weakness_signals.append("primary_structure_failed")
    if (
        assessment.above_daily_anchor is False
        and assessment.execution.state is ExecutionState.FAILED
    ):
        weakness_signals.append("daily_execution_failed_below_anchor")
    if weakness_signals:
        confirmation = (
            "multiple_timeframe_weakness_confirmed"
            if len(weakness_signals) >= 2
            else "single_dimension_weakness_warning_not_multi_timeframe_confirmation"
        )
        return HoldingAction.REDUCE, (
            confirmation,
            f"weakness_signals:{','.join(weakness_signals)}",
            "completed_bar_weakness_requires_staged_reduction_review",
            "handle_only_at_next_tradable_session_no_auto_order",
            "reduced_weight_remains_cash_no_automatic_replacement",
        )
    if stop_raised:
        return HoldingAction.TIGHTEN, (
            "strong_or_intact_holding_retained",
            "confirmed_pivot_raised_protection_line",
            "effective_stop_never_moves_down",
        )
    return HoldingAction.HOLD, (
        "no_completed_close_exit_or_reduce_signal",
        "strong_or_intact_holding_retained_without_rank_churn",
    )


def _failed_row(
    repository: SQLiteRepository,
    portfolio: ActiveHoldingPortfolio,
    holding: ActiveHolding,
    *,
    cutoff: date,
    review_time: datetime,
    reason: str,
    persist: bool,
    company_action_clearance: CompanyActionClearance | None = None,
) -> HoldingTreeReviewRow:
    stored = repository.get_holding_protective_stop(holding.position_key)
    stored_is_future = stored is not None and _stored_stop_cutoff(stored) > cutoff
    effective = None if stored is None or stored_is_future else float(stored["effective_stop"])
    reasons = (
        reason,
        *(("future_protective_stop_not_used",) if stored_is_future else ()),
        "fail_closed_no_holding_action",
        "candidate_ranking_not_used",
    )
    row = HoldingTreeReviewRow(
        symbol=holding.symbol,
        name=holding.name,
        holding_weeks=portfolio.holding_weeks,
        holding_version=portfolio.version,
        position_key=holding.position_key,
        status=HoldingReviewRowStatus.DATA_NOT_READY,
        action=HoldingAction.REVIEW,
        latest_close=None,
        cost_price=holding.cost_price,
        stock_sleeve_weight=holding.stock_sleeve_weight,
        account_weight=holding.account_weight,
        candidate_stop=None,
        previous_stop=effective,
        effective_stop=effective,
        stop_raised=False,
        close_below_stop=None,
        source_timeframe=None,
        evidence_date=None,
        slow_direction=None,
        primary_structure=None,
        daily_execution=None,
        reasons=reasons,
        company_action_clear=(
            None if company_action_clearance is None else company_action_clearance.clear
        ),
        company_action_evidence_id=(
            None if company_action_clearance is None else company_action_clearance.evidence_id
        ),
        company_action_evidence_source=(
            None if company_action_clearance is None else company_action_clearance.source
        ),
        company_action_clear_through=(
            None if company_action_clearance is None else company_action_clearance.through_date
        ),
    )
    if persist:
        evidence_hash = hashlib.sha256(
            f"{holding.position_key}|{portfolio.version}|{cutoff}|{reason}".encode()
        ).hexdigest()
        repository.record_holding_review(
            _review_record(
                portfolio,
                holding,
                row,
                cutoff=cutoff,
                review_time=review_time,
                evidence_hash=evidence_hash,
            )
        )
    return row


def _review_record(
    portfolio: ActiveHoldingPortfolio,
    holding: ActiveHolding,
    row: HoldingTreeReviewRow,
    *,
    cutoff: date,
    review_time: datetime,
    evidence_hash: str,
) -> dict[str, object]:
    identity = hashlib.sha256(
        (
            f"{holding.position_key}|{portfolio.version}|{cutoff}|"
            f"{HOLDING_TREE_METHOD_VERSION}|{evidence_hash}"
        ).encode()
    ).hexdigest()[:32]
    return {
        "id": f"holding-review:{identity}",
        "revision_id": portfolio.id,
        "position_id": holding.id,
        "position_key": holding.position_key,
        "symbol": holding.symbol,
        "name": holding.name,
        "holding_weeks": portfolio.holding_weeks,
        "holding_version": portfolio.version,
        "reviewed_at": review_time,
        "data_cutoff": cutoff,
        "status": row.status.value,
        "holding_action": row.action.value,
        "latest_close": row.latest_close,
        "candidate_stop": row.candidate_stop,
        "previous_stop": row.previous_stop,
        "effective_stop": row.effective_stop,
        "close_below_stop": row.close_below_stop,
        "source_timeframe": row.source_timeframe,
        "evidence_date": row.evidence_date,
        "reason_json": list(row.reasons),
        "company_action_clear": row.company_action_clear,
        "company_action_evidence_id": row.company_action_evidence_id,
        "company_action_evidence_source": row.company_action_evidence_source,
        "company_action_clear_through": row.company_action_clear_through,
        "evidence_hash": evidence_hash,
        "method_version": HOLDING_TREE_METHOD_VERSION,
        "created_at": review_time,
    }


def _global_failure(
    portfolio: ActiveHoldingPortfolio,
    *,
    review_time: datetime,
    reason: str,
) -> HoldingTreeReviewSummary:
    rows = tuple(
        HoldingTreeReviewRow(
            symbol=holding.symbol,
            name=holding.name,
            holding_weeks=portfolio.holding_weeks,
            holding_version=portfolio.version,
            position_key=holding.position_key,
            status=HoldingReviewRowStatus.DATA_NOT_READY,
            action=HoldingAction.REVIEW,
            latest_close=None,
            cost_price=holding.cost_price,
            stock_sleeve_weight=holding.stock_sleeve_weight,
            account_weight=holding.account_weight,
            candidate_stop=None,
            previous_stop=None,
            effective_stop=None,
            stop_raised=False,
            close_below_stop=None,
            source_timeframe=None,
            evidence_date=None,
            slow_direction=None,
            primary_structure=None,
            daily_execution=None,
            reasons=(reason, "fail_closed_no_holding_action"),
        )
        for holding in portfolio.positions
    )
    return HoldingTreeReviewSummary(
        status=HoldingReviewSummaryStatus.DATA_NOT_READY,
        portfolio_id=portfolio.id,
        holding_version=portfolio.version,
        holding_weeks=portfolio.holding_weeks,
        reviewed_at=review_time,
        data_cutoff=None,
        rows=rows,
        reasons=(reason,),
    )


def _evidence_hash(
    holding: ActiveHolding,
    cutoff: date,
    latest: Mapping[str, object],
    candidate_stop: float,
    company_action_clearance: CompanyActionClearance | None,
) -> str:
    payload = {
        "symbol": holding.symbol,
        "position_key": holding.position_key,
        "holding_version": holding.version,
        "cutoff": cutoff.isoformat(),
        "latest": {
            key: str(latest.get(key)) for key in ("trade_date", "open", "high", "low", "close")
        },
        "candidate_stop": candidate_stop,
        "company_action_clearance": (
            None
            if company_action_clearance is None
            else {
                "clear": company_action_clearance.clear,
                "through_date": company_action_clearance.through_date.isoformat(),
                "source": company_action_clearance.source,
                "evidence_id": company_action_clearance.evidence_id,
            }
        ),
        "method_version": HOLDING_TREE_METHOD_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _normalize_histories(
    histories: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    normalized: dict[str, pd.DataFrame] = {}
    for raw_symbol, frame in histories.items():
        try:
            symbol = normalize_symbol(str(raw_symbol))
        except ValueError:
            continue
        if symbol in normalized:
            raise ValueError(f"Duplicate normalized holding history: {symbol}")
        normalized[symbol] = frame
    return normalized


def _normalize_company_action_clearances(
    clearances: Mapping[str, CompanyActionClearance],
) -> dict[str, CompanyActionClearance]:
    normalized: dict[str, CompanyActionClearance] = {}
    for raw_symbol, evidence in clearances.items():
        if not isinstance(evidence, CompanyActionClearance):
            raise TypeError("company action evidence must use CompanyActionClearance")
        symbol = normalize_symbol(str(raw_symbol))
        if normalize_symbol(evidence.symbol) != symbol:
            raise ValueError("company action evidence symbol mismatch")
        if not evidence.source.strip() or not evidence.evidence_id.strip():
            raise ValueError("company action evidence source and id cannot be blank")
        normalized[symbol] = evidence
    return normalized


def _as_date(value: object) -> date:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("as_of and verified_data_cutoff must be valid dates") from exc
    if pd.isna(timestamp):
        raise ValueError("as_of and verified_data_cutoff must be valid dates")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.date()


def _stored_stop_cutoff(stored: Mapping[str, object]) -> date:
    try:
        return date.fromisoformat(str(stored["data_cutoff"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("stored holding protection cutoff is invalid") from exc


def _review_time(value: datetime | None) -> datetime:
    result = datetime.now(UTC) if value is None else value
    if not isinstance(result, datetime) or result.tzinfo is None:
        raise ValueError("reviewed_at must be timezone-aware")
    return result


def _price_floor(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_FLOOR))
