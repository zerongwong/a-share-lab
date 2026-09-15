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
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import ROUND_FLOOR, Decimal
from enum import StrEnum
from math import isfinite
from typing import Final
from zoneinfo import ZoneInfo

import pandas as pd

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.analytics.cost_stop import (
    COST_STOP_METHOD_VERSION,
    CostStopObservation,
    observe_cost_stop,
)
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

HOLDING_TREE_METHOD_VERSION: Final = "magee-inspired-pivot-trailing-v0.2.0"
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
    from_date: date | None = None


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
    company_action_clear_from: date | None = None
    cost_stop: float | None = None
    cost_stop_touched_on: date | None = None

    @property
    def urgent(self) -> bool:
        return self.action in {HoldingAction.REDUCE, HoldingAction.EXIT} or (
            self.action is HoldingAction.REVIEW
            and any(
                "company_action_evidence_blocks" in reason
                or reason.startswith("cost_stop_pending_breach")
                for reason in self.reasons
            )
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
    continuous_profile: bool = False,
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
            continuous_profile=continuous_profile,
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
    continuous_profile: bool = False,
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
    if (
        holding.cost_price is None
        and stored is not None
        and stored.get("details_json", {}).get("cost_stop_touched_on") is not None
    ):
        return _failed_row(
            repository,
            portfolio,
            holding,
            cutoff=cutoff,
            review_time=review_time,
            reason="confirmed_cost_missing_after_cost_stop_breach",
            persist=persist,
        )

    cost_observation = None
    company_action_clear = _company_action_is_clear(
        company_action_clearance, holding, cutoff, require_start=continuous_profile
    )
    try:
        # A missing/invalid volume series must not hide a valid cost-loss
        # observation; full technical assessment still validates its own data.
        price_columns = ["trade_date", "open", "high", "low", "close"]
        price_frame = (
            frame.loc[:, price_columns] if all(c in frame for c in price_columns) else frame
        )
        bars = build_completed_timeframes(price_frame, as_of=cutoff)
        if bars.data_cutoff.date() != cutoff:
            raise MultiTimeframeDataError("holding_close_cutoff_mismatch")
        if frame.attrs.get("adjustment", "none") != "none":
            raise MultiTimeframeDataError("holding_prices_must_be_unadjusted")
        if holding.cost_price is not None:
            remembered = {} if stored is None else stored.get("details_json", {})
            touched = remembered.get("cost_stop_touched_on")
            from ashare_lab.services.intraday_alert_store import confirmed_cost_touch

            intraday_touch = confirmed_cost_touch(
                repository, holding.position_key, cutoff=cutoff, known_at=review_time
            )
            remembered_touch = None if touched is None else date.fromisoformat(touched)
            if intraday_touch is not None:
                remembered_touch = (
                    min(remembered_touch, intraday_touch) if remembered_touch else intraday_touch
                )
            cost_observation = observe_cost_stop(
                bars.daily,
                cost_price=holding.cost_price,
                entry_date=holding.entry_date,
                remembered_line=remembered.get("cost_stop"),
                remembered_touch=remembered_touch,
                cost_update_date=portfolio.effective_at.astimezone(
                    ZoneInfo("Asia/Shanghai")
                ).date(),
                remembered_history=remembered.get("cost_stop_history"),
            )
        from ashare_lab.analytics.continuous_signals import CONTINUOUS_SIGNAL_CONTRACT

        profile_kwargs = (
            {"signal_contract": CONTINUOUS_SIGNAL_CONTRACT} if continuous_profile else {}
        )
        assessment = assess_multi_timeframe(
            frame,
            as_of=cutoff,
            holding_weeks=portfolio.holding_weeks,
            **profile_kwargs,
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
        failed = _failed_row(
            repository,
            portfolio,
            holding,
            cutoff=cutoff,
            review_time=review_time,
            reason=f"holding_data_not_ready:{exc}",
            persist=persist and cost_observation is None,
            company_action_clearance=company_action_clearance,
        )
        if cost_observation is None:
            return failed
        # Technical history/ATR failure cannot mask an independently verified
        # cost-loss breach. The corporate-action interval still has to pass.
        return _cost_only_review(
            repository,
            portfolio,
            holding,
            failed,
            cost_observation,
            cutoff=cutoff,
            review_time=review_time,
            persist=persist,
            company_action_clearance=company_action_clearance,
        )

    latest_close = float(bars.daily.iloc[-1]["close"])
    if stored is None:
        previous_stop = None
    elif _stored_stop_cutoff(stored) == cutoff:
        previous_stop = None if stored["previous_stop"] is None else float(stored["previous_stop"])
    else:
        previous_stop = float(stored["effective_stop"])
    # Even a same-cutoff replay under a new signal profile must respect the
    # latest stored line, not just yesterday's previous_stop.
    stored_floor = candidate.stop if stored is None else float(stored["effective_stop"])
    effective_stop = max(
        _price_floor(candidate.stop), previous_stop or candidate.stop, stored_floor
    )
    if cost_observation is not None:
        # Do not round the hard boundary down to a greater-than-8% loss.
        effective_stop = max(effective_stop, cost_observation.line)
    stop_raised = previous_stop is not None and effective_stop > previous_stop + 0.005
    close_below_stop = latest_close < effective_stop
    raw_action, action_reasons = _holding_action(
        assessment,
        close_below_stop=close_below_stop,
        stop_raised=stop_raised,
    )
    cost_reasons = _cost_reasons(cost_observation)
    if cost_observation is not None and cost_observation.touched_on is not None:
        raw_action = HoldingAction.EXIT
        action_reasons = (
            "handle_only_at_next_tradable_session_subject_to_t_plus_one_suspension_and_limit_down",
            "exited_weight_remains_cash_until_explicit_new_plan",
        )
    # Actual-cost comparisons always require the entire holding interval,
    # even for callers retaining the legacy horizon transport field.
    if cost_observation is not None:
        company_action_clear = _company_action_is_clear(
            company_action_clearance, holding, cutoff, require_start=True
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
        *cost_reasons,
        *company_action_reasons,
        *(
            (f"company_action_coverage_from:{company_action_clearance.from_date.isoformat()}",)
            if company_action_clearance is not None
            and company_action_clearance.from_date is not None
            else ()
        ),
        *(("signal_profile:continuous_daily_weekly_v1;no_expiry",) if continuous_profile else ()),
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
        cost_stop=None if cost_observation is None else cost_observation.line,
        cost_stop_touched_on=(None if cost_observation is None else cost_observation.touched_on),
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
        company_action_clear_from=(
            None if company_action_clearance is None else company_action_clearance.from_date
        ),
        method_version=(
            f"{HOLDING_TREE_METHOD_VERSION}+continuous-v1"
            if continuous_profile
            else HOLDING_TREE_METHOD_VERSION
        ),
    )
    if persist:
        evidence_hash = _evidence_hash(
            holding,
            cutoff,
            bars.daily.iloc[-1].to_dict(),
            candidate.stop,
            company_action_clearance,
            cost_observation,
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
                "method_version": row.method_version,
                "details_json": {
                    "atr14": candidate.atr14,
                    "atr_cutoff": candidate.atr_cutoff.isoformat(),
                    "atr_buffer_multiple": ATR_BUFFER_MULTIPLE,
                    "support": candidate.support,
                    "support_kind": candidate.support_kind,
                    "cost_stop": row.cost_stop
                    if cost_observation is not None
                    else ({} if stored is None else stored.get("details_json", {})).get(
                        "cost_stop"
                    ),
                    "cost_stop_touched_on": (
                        row.cost_stop_touched_on.isoformat()
                        if row.cost_stop_touched_on is not None
                        else ({} if stored is None else stored.get("details_json", {})).get(
                            "cost_stop_touched_on"
                        )
                    ),
                    "calendar_boundary": "existing_multi_timeframe_conservative_fallback",
                    "cost_stop_history": cost_observation.line_history
                    if cost_observation is not None
                    else ({} if stored is None else stored.get("details_json", {})).get(
                        "cost_stop_history"
                    ),
                    "company_action_evidence_id": company_action_clearance.evidence_id,
                    "company_action_clear_from": None
                    if company_action_clearance.from_date is None
                    else company_action_clearance.from_date.isoformat(),
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


def _company_action_is_clear(evidence, holding, cutoff, *, require_start):
    return bool(
        evidence is not None
        and evidence.clear
        and evidence.through_date >= cutoff
        and (
            not require_start
            or (evidence.from_date is not None and evidence.from_date <= holding.entry_date)
        )
    )


def _cost_reasons(observation: CostStopObservation | None) -> tuple[str, ...]:
    if observation is None:
        return ("cost_stop_unavailable_without_confirmed_cost",)
    reasons = (f"cost_stop_method:{COST_STOP_METHOD_VERSION}",)
    if observation.touched_on is not None:
        reasons += (
            f"cost_stop_8pct_touched:{observation.touched_on.isoformat()}",
            "cost_stop_alert_latched_until_confirmed_position_closed",
        )
    return reasons


def _cost_only_review(
    repository,
    portfolio,
    holding,
    failed,
    observation,
    *,
    cutoff,
    review_time,
    persist,
    company_action_clearance,
):
    """Independent safety guard when a technical pattern cannot be assessed."""
    clear = _company_action_is_clear(company_action_clearance, holding, cutoff, require_start=True)
    breached = observation.touched_on is not None
    reasons = tuple(
        r
        for r in failed.reasons
        if r != "fail_closed_no_holding_action" and not r.startswith("cost_stop_pending_breach:")
    )
    reasons += _cost_reasons(observation)
    if breached and not clear:
        reasons += (
            "company_action_evidence_blocks_exit:cost_comparability_unverified",
            "candidate_stop_not_persisted_without_company_action_clearance",
        )
    elif breached:
        reasons += (
            "cost_stop_exit_independent_of_technical_history",
            "handle_only_at_next_tradable_session_subject_to_t_plus_one_suspension_and_limit_down",
        )
    row = replace(
        failed,
        status=HoldingReviewRowStatus.READY if breached and clear else failed.status,
        action=HoldingAction.EXIT if breached and clear else HoldingAction.REVIEW,
        latest_close=observation.latest_close,
        cost_stop=observation.line,
        cost_stop_touched_on=observation.touched_on,
        effective_stop=max(failed.effective_stop or observation.line, observation.line),
        close_below_stop=observation.latest_close
        < max(failed.effective_stop or observation.line, observation.line),
        reasons=reasons,
    )
    if persist:
        # Persist the hard line and breach even without a usable technical
        # pivot; future missing data must not quietly erase this warning.
        stop_state = None
        if clear:
            stop_state = {
                "position_key": holding.position_key,
                "symbol": holding.symbol,
                "entry_date": holding.entry_date,
                "effective_stop": row.effective_stop,
                "candidate_stop": observation.line,
                "previous_stop": failed.effective_stop,
                "data_cutoff": cutoff,
                "source_timeframe": "confirmed_cost",
                "evidence_date": observation.touched_on or cutoff,
                "holding_version": portfolio.version,
                "method_version": row.method_version,
                "details_json": {
                    "cost_stop": observation.line,
                    "cost_stop_touched_on": None
                    if observation.touched_on is None
                    else observation.touched_on.isoformat(),
                    "cost_stop_history": observation.line_history,
                },
                "updated_at": review_time,
            }
        evidence_hash = _evidence_hash(
            holding,
            cutoff,
            {"close": observation.latest_close},
            row.effective_stop,
            company_action_clearance,
            observation,
        )
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
    remembered_touch = (
        None
        if stored is None or stored_is_future
        else stored.get("details_json", {}).get("cost_stop_touched_on")
    )
    from ashare_lab.services.intraday_alert_store import confirmed_cost_touch

    intraday_touch = confirmed_cost_touch(
        repository, holding.position_key, cutoff=cutoff, known_at=review_time
    )
    if intraday_touch is not None:
        remembered_touch = (
            min(remembered_touch, intraday_touch.isoformat())
            if remembered_touch
            else intraday_touch.isoformat()
        )
    if remembered_touch is not None:
        reasons += (f"cost_stop_pending_breach:{remembered_touch}",)
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
        company_action_clear_from=(
            None if company_action_clearance is None else company_action_clearance.from_date
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
            f"{row.method_version}|{evidence_hash}"
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
        "method_version": row.method_version,
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
    cost_observation: CostStopObservation | None = None,
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
        "confirmed_cost": holding.cost_price,
        "cost_observation": None
        if cost_observation is None
        else {
            "line": cost_observation.line,
            "touched_on": None
            if cost_observation.touched_on is None
            else cost_observation.touched_on.isoformat(),
        },
        "cost_line_history": None if cost_observation is None else cost_observation.line_history,
        "company_action_clearance": (
            None
            if company_action_clearance is None
            else {
                "clear": company_action_clearance.clear,
                "through_date": company_action_clearance.through_date.isoformat(),
                "source": company_action_clearance.source,
                "evidence_id": company_action_clearance.evidence_id,
                **(
                    {"from_date": company_action_clearance.from_date.isoformat()}
                    if company_action_clearance.from_date is not None
                    else {}
                ),
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
