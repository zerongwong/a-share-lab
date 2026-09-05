"""Pure, fail-closed settlement of archived recommendation cohorts.

The module deliberately performs no I/O.  A caller supplies immutable archive
records plus a cross-section of verified, unadjusted daily bars and the
verified exchange-session sequence.  The result can then be persisted and/or
rendered by a separate adapter.

Two outcome populations are kept separate:

* ``ACTION_SIMULATION`` confirms an archived condition on ``plan_for_date``,
  simulates entry at the next verified session's open, and exits at the close
  of the Nth holding session.  An untriggered allocation remains cash and is
  never redistributed to triggered names.
* ``OBSERVATION_SIMULATION`` measures an accepted original risk-qualified or
  observation-only basket; ``RECONSTRUCTED_OBSERVATION`` does the same for an
  explicitly reconstructed historical archive.  Both use either the
  plan-session close or an archived reference price and remain outside the
  action-performance population.

All returns are decimal unadjusted *price* returns (``0.03 == 3%``), not total
returns.  Corporate-action evidence must explicitly cover every evaluated
symbol.  Missing, invalid, suspended, or corporate-action-affected evidence
does not silently become a zero return.
"""

from __future__ import annotations

import calendar
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from typing import Literal

import pandas as pd

from ashare_lab.ports.market_data import normalize_symbol

HORIZON_SESSIONS_BY_WEEKS: Mapping[int, int] = {
    1: 5,
    2: 10,
    4: 20,
    13: 60,
    26: 120,
    52: 252,
}
PERFORMANCE_METHOD_VERSION = "recommendation-maturity-settlement-v0.1.0"


class ArchiveNature(StrEnum):
    ORIGINAL = "original"
    RECONSTRUCTED = "reconstructed"


class CohortNature(StrEnum):
    ACTION_QUALIFIED = "action_qualified"
    RISK_QUALIFIED = "risk_qualified"
    OBSERVATION_ONLY = "observation_only"


class EvaluationMode(StrEnum):
    ACTION_SIMULATION = "action_simulation"
    OBSERVATION_SIMULATION = "observation_simulation"
    RECONSTRUCTED_OBSERVATION = "reconstructed_observation"


class HoldingClock(StrEnum):
    TRADING_SESSIONS = "trading_sessions"
    CALENDAR = "calendar"


class EntryPlanKind(StrEnum):
    HEALTHY_PULLBACK = "healthy_pullback_range"
    RECLAIM = "reclaim_close_confirmation"
    VOLUME_BREAKOUT = "volume_breakout_close_confirmation"


class ObservationAnchor(StrEnum):
    PLAN_SESSION_CLOSE = "plan_session_close"
    ARCHIVED_REFERENCE_PRICE = "archived_reference_price"


class MemberSettlementStatus(StrEnum):
    PENDING = "pending_maturity_session"
    SETTLED = "settled"
    EXPIRED_UNTRIGGERED = "expired_untriggered"
    ENTRY_PRICE_LIMIT_EXCEEDED = "entry_price_limit_exceeded"
    ENTRY_INVALIDATED_BEFORE_FILL = "entry_invalidated_before_fill"
    ARCHIVE_INELIGIBLE = "archive_ineligible"
    ENTRY_RULE_INCOMPLETE = "entry_rule_incomplete"
    SOURCE_ADJUSTMENT_INVALID = "source_adjustment_invalid"
    PLAN_SESSION_MISSING = "plan_session_missing"
    PLAN_PRICE_MISSING = "plan_price_missing"
    PLAN_PRICE_INVALID = "plan_price_invalid"
    PLAN_SESSION_SUSPENDED = "plan_session_suspended"
    ENTRY_PRICE_MISSING = "entry_price_missing"
    ENTRY_PRICE_INVALID = "entry_price_invalid"
    ENTRY_SESSION_SUSPENDED = "entry_session_suspended"
    MATURITY_PRICE_MISSING = "maturity_price_missing"
    MATURITY_PRICE_INVALID = "maturity_price_invalid"
    MATURITY_SESSION_SUSPENDED = "maturity_session_suspended"
    CORPORATE_ACTION_EVIDENCE_UNKNOWN = "corporate_action_evidence_unknown"
    CORPORATE_ACTION_DETECTED = "corporate_action_detected"


class BatchSettlementStatus(StrEnum):
    PENDING = "pending"
    SETTLED = "settled"
    SETTLED_PARTIAL_ENTRY = "settled_partial_entry"
    NO_ENTRY = "no_entry"
    DATA_QUALITY_FAILURE = "data_quality_failure"


@dataclass(frozen=True, slots=True)
class ArchivedRecommendationReport:
    report_id: str
    decision_date: date
    common_cutoff: date
    plan_for_date: date
    archive_nature: ArchiveNature
    delivery_accepted: bool


@dataclass(frozen=True, slots=True)
class ArchivedRecommendationBatch:
    batch_id: str
    report_id: str
    holding_weeks: int
    holding_sessions: int
    evaluation_mode: EvaluationMode
    cohort_nature: CohortNature
    stock_exposure: float | None
    holding_clock: HoldingClock = HoldingClock.TRADING_SESSIONS


@dataclass(frozen=True, slots=True)
class ArchivedEntryPlan:
    # Legacy archives may contain only validity metadata.  Keep them readable,
    # but settlement will fail closed until a structured kind is present.
    kind: EntryPlanKind | None
    price_low: float | None = None
    price_high: float | None = None
    trigger_price: float | None = None
    invalidation_price: float | None = None
    confirmation_activity_metric: str | None = None
    confirmation_activity_min: float | None = None
    # Read compatibility for archives created before activity metric names
    # were frozen.  New archives never write this volume-only field.
    confirmation_volume_min: float | None = None
    maximum_entry_price: float | None = None


@dataclass(frozen=True, slots=True)
class ArchivedRecommendationMember:
    member_id: str
    batch_id: str
    symbol: str
    name: str
    operational_stock_sleeve_weight: float
    operational_account_weight: float | None
    entry_plan: ArchivedEntryPlan | None = None
    observation_anchor: ObservationAnchor | None = None
    reference_price: float | None = None


@dataclass(frozen=True, slots=True)
class VerifiedDailyEvidence:
    """Verified input evidence for one settlement pass.

    ``session_dates`` contains market-wide sessions, including dates on which
    an individual security was suspended.  ``corporate_action_coverage_symbols``
    is explicit: an absent symbol means that corporate-action evidence is
    unknown and therefore cannot produce an unadjusted price return.
    """

    prices: pd.DataFrame
    session_dates: tuple[date, ...]
    source_adjustment: Literal["none"] | str = "none"
    corporate_action_coverage_symbols: frozenset[str] = frozenset()
    corporate_action_dates_by_symbol: Mapping[str, frozenset[date]] = field(default_factory=dict)
    suspended_dates_by_symbol: Mapping[str, frozenset[date]] = field(default_factory=dict)
    invalid_dates_by_symbol: Mapping[str, frozenset[date]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecommendationMemberPerformance:
    member_id: str
    symbol: str
    name: str
    status: MemberSettlementStatus
    evaluation_mode: EvaluationMode
    plan_date: date
    entry_date: date | None
    entry_price: float | None
    published_reference_price: float | None
    maturity_date: date | None
    maturity_close: float | None
    unadjusted_price_return: float | None
    stock_sleeve_contribution: float | None
    account_contribution: float | None
    condition_triggered: bool | None
    simulated_action_return: float | None
    simulated_action_stock_sleeve_contribution: float | None
    simulated_action_account_contribution: float | None
    operational_stock_sleeve_weight: float
    operational_account_weight: float | None
    holding_sessions_observed: int | None
    company_action_clear: bool | None
    reason_code: str
    method_version: str = PERFORMANCE_METHOD_VERSION
    raw_unadjusted_price_change: float | None = None
    maximum_entry_price: float | None = None
    entry_execution_rule_version: str = "legacy-next-open-v1"


@dataclass(frozen=True, slots=True)
class RecommendationBatchPerformance:
    batch_id: str
    report_id: str
    holding_weeks: int
    holding_sessions: int
    evaluation_mode: EvaluationMode
    cohort_nature: CohortNature
    status: BatchSettlementStatus
    plan_date: date
    maturity_date: date | None
    stock_sleeve_return: float | None
    account_return: float | None
    simulated_action_stock_sleeve_return: float | None
    simulated_action_account_return: float | None
    entered_stock_sleeve_weight: float
    entered_account_weight: float | None
    stock_sleeve_cash_weight: float | None
    account_cash_weight: float | None
    members: tuple[RecommendationMemberPerformance, ...]
    data_cutoff: date | None
    reason_code: str
    method_version: str = PERFORMANCE_METHOD_VERSION
    raw_unadjusted_stock_sleeve_change: float | None = None
    raw_unadjusted_account_change: float | None = None
    holding_clock: HoldingClock = HoldingClock.TRADING_SESSIONS


def archived_report_from_mapping(row: Mapping[str, object]) -> ArchivedRecommendationReport:
    """Adapt one repository row without coupling the pure engine to SQLite."""

    return ArchivedRecommendationReport(
        report_id=str(_required(row, "report_id", "id")),
        decision_date=_as_date(_required(row, "decision_date")),
        common_cutoff=_as_date(_required(row, "common_cutoff", "common_cutoff_date")),
        plan_for_date=_as_date(_required(row, "plan_for_date")),
        archive_nature=ArchiveNature(str(_required(row, "archive_nature"))),
        delivery_accepted=_as_bool(row.get("delivery_accepted", False)),
    )


def archived_batch_from_mapping(row: Mapping[str, object]) -> ArchivedRecommendationBatch:
    """Adapt a stored horizon batch while preserving the immutable session count."""

    stock_exposure = _optional_float(_first_present(row, "stock_exposure", "action_stock_exposure"))
    metadata = row.get("metadata_json") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not isinstance(metadata, Mapping):
        raise ValueError("batch metadata must be an object")
    return ArchivedRecommendationBatch(
        batch_id=str(_required(row, "batch_id", "id")),
        report_id=str(_required(row, "report_id")),
        holding_weeks=int(_required(row, "holding_weeks")),
        holding_sessions=int(_required(row, "holding_sessions")),
        evaluation_mode=EvaluationMode(str(_required(row, "evaluation_mode"))),
        cohort_nature=CohortNature(str(_required(row, "cohort_nature"))),
        stock_exposure=stock_exposure,
        holding_clock=HoldingClock(str(metadata.get("holding_clock", "trading_sessions"))),
    )


def archived_member_from_mapping(row: Mapping[str, object]) -> ArchivedRecommendationMember:
    """Adapt one stored member, including a JSON or mapping entry-plan payload."""

    plan_payload = _first_present(row, "entry_plan", "entry_plan_json")
    plan = _entry_plan_from_payload(plan_payload)
    anchor_value = _first_present(row, "observation_anchor", "observation_anchor_kind")
    return ArchivedRecommendationMember(
        member_id=str(_required(row, "member_id", "id")),
        batch_id=str(_required(row, "batch_id")),
        symbol=str(_required(row, "symbol")),
        name=str(row.get("name") or row.get("stock_name") or row.get("symbol") or ""),
        operational_stock_sleeve_weight=float(
            _required(
                row,
                "operational_stock_sleeve_weight",
                "stock_sleeve_weight",
                "sleeve_weight",
            )
        ),
        operational_account_weight=_optional_float(
            _first_present(row, "operational_account_weight", "account_weight")
        ),
        entry_plan=plan,
        observation_anchor=(None if anchor_value is None else ObservationAnchor(str(anchor_value))),
        reference_price=_optional_float(
            _first_present(
                row,
                "reference_price",
                "entry_reference_price",
                "published_reference_price",
            )
        ),
    )


def member_performance_record(
    result: RecommendationMemberPerformance,
) -> dict[str, object]:
    """Return a repository-friendly mapping; dates and enums are serialized."""

    return {
        "member_id": result.member_id,
        "status": result.status.value,
        # The frozen published reference is the primary paper-evaluation price.
        "entry_date": _iso_date(result.plan_date),
        "entry_price": result.published_reference_price,
        "maturity_date": _iso_date(result.maturity_date),
        "maturity_close": result.maturity_close,
        "realized_return": result.unadjusted_price_return,
        "holding_sessions_observed": result.holding_sessions_observed,
        "reason_code": result.reason_code,
        "company_action_clear": result.company_action_clear,
        "data_cutoff": _iso_date(result.maturity_date),
        "method_version": result.method_version,
        "details_json": {
            "evaluation_mode": result.evaluation_mode.value,
            "condition_triggered": result.condition_triggered,
            "simulated_entry_date": _iso_date(result.entry_date),
            "simulated_entry_price": result.entry_price,
            "simulated_action_return": result.simulated_action_return,
            "stock_sleeve_contribution": result.stock_sleeve_contribution,
            "account_contribution": result.account_contribution,
            "simulated_action_stock_sleeve_contribution": (
                result.simulated_action_stock_sleeve_contribution
            ),
            "simulated_action_account_contribution": (result.simulated_action_account_contribution),
            "raw_unadjusted_price_change": result.raw_unadjusted_price_change,
            "maximum_entry_price": result.maximum_entry_price,
            "entry_execution_rule_version": result.entry_execution_rule_version,
            "legacy_entry_execution_compatibility": result.maximum_entry_price is None,
        },
    }


def batch_performance_record(
    result: RecommendationBatchPerformance,
) -> dict[str, object]:
    """Return the primary paper metric plus explicit execution-simulation details."""

    return {
        "batch_id": result.batch_id,
        "status": result.status.value,
        "maturity_date": _iso_date(result.maturity_date),
        "stock_sleeve_return": result.stock_sleeve_return,
        "account_return": result.account_return,
        "entered_stock_sleeve_weight": result.entered_stock_sleeve_weight,
        "entered_account_weight": result.entered_account_weight,
        "cash_weight": result.account_cash_weight,
        "resolved_member_count": sum(
            member.status is not MemberSettlementStatus.PENDING for member in result.members
        ),
        "total_member_count": len(result.members),
        "reason_code": result.reason_code,
        "data_cutoff": _iso_date(result.data_cutoff),
        "method_version": result.method_version,
        "details_json": {
            "evaluation_mode": result.evaluation_mode.value,
            "cohort_nature": result.cohort_nature.value,
            "holding_weeks": result.holding_weeks,
            "holding_sessions": result.holding_sessions,
            "holding_clock": result.holding_clock.value,
            "performance_basis": "fixed_horizon_price_simulation_not_dynamic_holding_returns",
            "simulated_action_stock_sleeve_return": (result.simulated_action_stock_sleeve_return),
            "simulated_action_account_return": result.simulated_action_account_return,
            "stock_sleeve_cash_weight": result.stock_sleeve_cash_weight,
            "raw_unadjusted_stock_sleeve_change": (result.raw_unadjusted_stock_sleeve_change),
            "raw_unadjusted_account_change": result.raw_unadjusted_account_change,
        },
    }


class _PriceState(StrEnum):
    AVAILABLE = "available"
    MISSING = "missing"
    INVALID = "invalid"
    SUSPENDED = "suspended"


@dataclass(frozen=True, slots=True)
class _PriceRead:
    state: _PriceState
    value: float | None = None


@dataclass(frozen=True, slots=True)
class _PreparedEvidence:
    prices: pd.DataFrame
    sessions: tuple[date, ...]
    corporate_action_coverage_symbols: frozenset[str]
    corporate_action_dates_by_symbol: Mapping[str, frozenset[date]]
    suspended_dates_by_symbol: Mapping[str, frozenset[date]]
    invalid_dates_by_symbol: Mapping[str, frozenset[date]]


_FATAL_MEMBER_STATUSES = frozenset(
    status
    for status in MemberSettlementStatus
    if status
    not in {
        MemberSettlementStatus.PENDING,
        MemberSettlementStatus.SETTLED,
        MemberSettlementStatus.EXPIRED_UNTRIGGERED,
        MemberSettlementStatus.ENTRY_PRICE_LIMIT_EXCEEDED,
        MemberSettlementStatus.ENTRY_INVALIDATED_BEFORE_FILL,
    }
)


def settle_recommendation_performance(
    *,
    report: ArchivedRecommendationReport,
    batch: ArchivedRecommendationBatch,
    members: Sequence[ArchivedRecommendationMember],
    evidence: VerifiedDailyEvidence,
) -> RecommendationBatchPerformance:
    """Settle one immutable horizon batch without reading or writing state.

    Legacy maturity is session based. If ``plan_for_date`` is session index ``i``,
    both supported modes mature at ``i + holding_sessions``.  For action
    simulation, a confirmed plan enters at session ``i + 1`` open, making the
    maturity close the Nth holding-session close. Explicit calendar archives
    use the calendar anniversary, rolling non-sessions forward without changing
    the archived clock or legacy session counts.
    """

    normalized_members = tuple(members)
    _validate_archive(report, batch, normalized_members)
    prepared = _prepare_evidence(evidence, normalized_members)
    data_cutoff = prepared.sessions[-1] if prepared.sessions else None

    if evidence.source_adjustment != "none":
        return _uniform_failure(
            report=report,
            batch=batch,
            members=normalized_members,
            data_cutoff=data_cutoff,
            status=MemberSettlementStatus.SOURCE_ADJUSTMENT_INVALID,
        )

    try:
        plan_index = prepared.sessions.index(report.plan_for_date)
    except ValueError:
        if data_cutoff is None or data_cutoff < report.plan_for_date:
            return _pending_batch(
                report=report,
                batch=batch,
                members=normalized_members,
                maturity_date=None,
                data_cutoff=data_cutoff,
            )
        return _uniform_failure(
            report=report,
            batch=batch,
            members=normalized_members,
            data_cutoff=data_cutoff,
            status=MemberSettlementStatus.PLAN_SESSION_MISSING,
        )

    maturity_date = _maturity_date(report, batch, prepared.sessions, plan_index)
    member_results = tuple(
        _settle_member(
            report=report,
            batch=batch,
            member=member,
            prepared=prepared,
            plan_index=plan_index,
            maturity_date=maturity_date,
        )
        for member in normalized_members
    )
    return _aggregate_batch(
        report=report,
        batch=batch,
        member_results=member_results,
        maturity_date=maturity_date,
        data_cutoff=data_cutoff,
    )


def calendar_maturity_target(plan_date: date, holding_weeks: int) -> date:
    """Calendar anniversaries clamp month-end, then execution rolls forward."""

    if holding_weeks in (1, 2):
        return plan_date + timedelta(days=holding_weeks * 7)
    months = {4: 1, 13: 3, 26: 6, 52: 12}.get(holding_weeks)
    if months is None:
        raise ValueError("unsupported calendar horizon")
    month_index = plan_date.year * 12 + plan_date.month - 1 + months
    year, month_zero = divmod(month_index, 12)
    month = month_zero + 1
    day = min(plan_date.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _maturity_date(report, batch, sessions, plan_index) -> date | None:
    if batch.holding_clock is HoldingClock.CALENDAR:
        target = calendar_maturity_target(report.plan_for_date, batch.holding_weeks)
        return next((session for session in sessions if session >= target), None)
    index = plan_index + batch.holding_sessions
    return sessions[index] if index < len(sessions) else None


def _validate_archive(
    report: ArchivedRecommendationReport,
    batch: ArchivedRecommendationBatch,
    members: tuple[ArchivedRecommendationMember, ...],
) -> None:
    if not report.report_id or batch.report_id != report.report_id or not batch.batch_id:
        raise ValueError("report and batch archive identifiers are inconsistent")
    expected_sessions = HORIZON_SESSIONS_BY_WEEKS.get(batch.holding_weeks)
    if expected_sessions is None or batch.holding_sessions != expected_sessions:
        raise ValueError("holding horizon must use the immutable session mapping")
    if report.common_cutoff > report.plan_for_date:
        raise ValueError("common_cutoff cannot be later than plan_for_date")
    if not members:
        raise ValueError("a recommendation batch must contain at least one member")
    if len({member.member_id for member in members}) != len(members):
        raise ValueError("member_id values must be unique")
    normalized_symbols = [normalize_symbol(member.symbol) for member in members]
    if len(set(normalized_symbols)) != len(normalized_symbols):
        raise ValueError("member symbols must be unique within a batch")
    for member in members:
        if member.batch_id != batch.batch_id:
            raise ValueError("member belongs to a different batch")
        _require_weight(member.operational_stock_sleeve_weight, "stock-sleeve weight")
        if member.operational_account_weight is not None:
            _require_weight(member.operational_account_weight, "account weight")

    sleeve_sum = sum(member.operational_stock_sleeve_weight for member in members)
    if sleeve_sum > 1.0 + 1e-8:
        raise ValueError("stock-sleeve weights cannot exceed 100%")

    if batch.evaluation_mode is EvaluationMode.ACTION_SIMULATION:
        if report.archive_nature is not ArchiveNature.ORIGINAL or not report.delivery_accepted:
            raise ValueError("action simulation requires an accepted original archive")
        if batch.cohort_nature is not CohortNature.ACTION_QUALIFIED:
            raise ValueError("action simulation requires an action-qualified cohort")
        if batch.stock_exposure is None:
            raise ValueError("action simulation requires archived stock exposure")
        _require_weight(batch.stock_exposure, "stock exposure")
        for member in members:
            if (
                member.entry_plan is None
                or member.operational_account_weight is None
                or not _valid_positive(member.reference_price)
            ):
                raise ValueError(
                    "action members require entry plans, published reference prices, "
                    "and account weights"
                )
            expected_account = member.operational_stock_sleeve_weight * batch.stock_exposure
            if not math.isclose(
                member.operational_account_weight,
                expected_account,
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise ValueError("account weight does not match stock-sleeve weight and exposure")
    elif batch.evaluation_mode is EvaluationMode.OBSERVATION_SIMULATION:
        if report.archive_nature is not ArchiveNature.ORIGINAL or not report.delivery_accepted:
            raise ValueError("observation simulation requires an accepted original archive")
        if batch.cohort_nature not in {
            CohortNature.RISK_QUALIFIED,
            CohortNature.OBSERVATION_ONLY,
        }:
            raise ValueError("observation simulation cannot use an action cohort")
        for member in members:
            if member.observation_anchor is None:
                raise ValueError("observation members require an explicit anchor")
            if (
                member.observation_anchor is ObservationAnchor.ARCHIVED_REFERENCE_PRICE
                and not _valid_positive(member.reference_price)
            ):
                raise ValueError("reference-price observation requires a positive reference price")
    else:
        if batch.cohort_nature is not CohortNature.OBSERVATION_ONLY:
            raise ValueError("reconstructed evaluation must remain observation-only")
        if report.archive_nature is not ArchiveNature.RECONSTRUCTED:
            raise ValueError("reconstructed observation requires reconstructed archive nature")
        for member in members:
            if member.observation_anchor is None:
                raise ValueError("observation members require an explicit anchor")
            if (
                member.observation_anchor is ObservationAnchor.ARCHIVED_REFERENCE_PRICE
                and not _valid_positive(member.reference_price)
            ):
                raise ValueError("reference-price observation requires a positive reference price")


def _require_weight(value: float, label: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be a finite decimal between zero and one")


def _prepare_evidence(
    evidence: VerifiedDailyEvidence,
    members: tuple[ArchivedRecommendationMember, ...],
) -> _PreparedEvidence:
    sessions = tuple(sorted(set(evidence.session_dates)))
    if sessions != evidence.session_dates:
        raise ValueError("session_dates must be unique and strictly increasing")
    required = {"symbol", "trade_date", "open", "close"}
    missing = required - set(evidence.prices.columns)
    if missing:
        raise ValueError("verified prices are missing columns: " + ", ".join(sorted(missing)))

    wanted = {normalize_symbol(member.symbol) for member in members}
    frame = evidence.prices.copy()
    frame["_normalized_symbol"] = frame["symbol"].map(_try_normalize_symbol)
    frame = frame.loc[frame["_normalized_symbol"].isin(wanted)].copy()
    parsed_dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    if bool(parsed_dates.isna().any()):
        raise ValueError("verified prices contain invalid trade_date values")
    try:
        parsed_dates = parsed_dates.dt.tz_localize(None)
    except TypeError as exc:
        raise ValueError("verified price dates must use one consistent timezone") from exc
    frame["_trade_date"] = parsed_dates.dt.date
    duplicate_mask = frame.duplicated(["_normalized_symbol", "_trade_date"], keep=False)
    if bool(duplicate_mask.any()):
        duplicates = frame.loc[duplicate_mask, ["_normalized_symbol", "_trade_date"]]
        duplicate_pairs = set(duplicates.itertuples(index=False, name=None))
    else:
        duplicate_pairs = set()
    frame["_duplicate"] = [
        (symbol, trade_date) in duplicate_pairs
        for symbol, trade_date in zip(
            frame["_normalized_symbol"], frame["_trade_date"], strict=True
        )
    ]

    return _PreparedEvidence(
        prices=frame,
        sessions=sessions,
        corporate_action_coverage_symbols=frozenset(
            _normalized_mapping_keys(evidence.corporate_action_coverage_symbols)
        ),
        corporate_action_dates_by_symbol=_normalize_date_mapping(
            evidence.corporate_action_dates_by_symbol
        ),
        suspended_dates_by_symbol=_normalize_date_mapping(evidence.suspended_dates_by_symbol),
        invalid_dates_by_symbol=_normalize_date_mapping(evidence.invalid_dates_by_symbol),
    )


def _settle_member(
    *,
    report: ArchivedRecommendationReport,
    batch: ArchivedRecommendationBatch,
    member: ArchivedRecommendationMember,
    prepared: _PreparedEvidence,
    plan_index: int,
    maturity_date: date | None,
) -> RecommendationMemberPerformance:
    if batch.evaluation_mode is EvaluationMode.ACTION_SIMULATION:
        return _settle_action_member(
            report=report,
            batch=batch,
            member=member,
            prepared=prepared,
            plan_index=plan_index,
            maturity_date=maturity_date,
        )
    return _settle_observation_member(
        report=report,
        batch=batch,
        member=member,
        prepared=prepared,
        maturity_date=maturity_date,
    )


def _settle_action_member(
    *,
    report: ArchivedRecommendationReport,
    batch: ArchivedRecommendationBatch,
    member: ArchivedRecommendationMember,
    prepared: _PreparedEvidence,
    plan_index: int,
    maturity_date: date | None,
) -> RecommendationMemberPerformance:
    symbol = normalize_symbol(member.symbol)
    plan = member.entry_plan
    if plan is None or not _entry_plan_complete(plan):
        return _member_failure(
            report, batch, member, maturity_date, MemberSettlementStatus.ENTRY_RULE_INCOMPLETE
        )

    plan_price = _read_price(prepared, symbol, report.plan_for_date, "close")
    price_failure = _price_failure_status(plan_price, stage="plan")
    if price_failure is not None:
        return _member_failure(report, batch, member, maturity_date, price_failure)
    trigger_state = _evaluate_trigger(prepared, symbol, report.plan_for_date, plan)
    if isinstance(trigger_state, MemberSettlementStatus):
        return _member_failure(report, batch, member, maturity_date, trigger_state)
    if not trigger_state:
        if maturity_date is None:
            return _member_result(
                report=report,
                batch=batch,
                member=member,
                status=MemberSettlementStatus.PENDING,
                published_reference_price=member.reference_price,
                condition_triggered=False,
                reason_code=MemberSettlementStatus.PENDING.value,
            )
        return _complete_member(
            report=report,
            batch=batch,
            member=member,
            prepared=prepared,
            published_reference_price=float(member.reference_price),
            maturity_date=maturity_date,
            condition_triggered=False,
        )

    entry_index = plan_index + 1
    if entry_index >= len(prepared.sessions):
        return _member_result(
            report=report,
            batch=batch,
            member=member,
            status=MemberSettlementStatus.PENDING,
            maturity_date=maturity_date,
            published_reference_price=member.reference_price,
            condition_triggered=True,
            reason_code=MemberSettlementStatus.PENDING.value,
        )
    entry_date = prepared.sessions[entry_index]
    entry_read = _read_price(prepared, symbol, entry_date, "open")
    entry_failure = _price_failure_status(entry_read, stage="entry")
    if entry_failure is not None:
        return _member_failure(
            report,
            batch,
            member,
            maturity_date,
            entry_failure,
            entry_date=entry_date,
        )
    execution_block = None
    if plan.maximum_entry_price is not None:
        if float(entry_read.value) > plan.maximum_entry_price:
            execution_block = MemberSettlementStatus.ENTRY_PRICE_LIMIT_EXCEEDED
        elif float(entry_read.value) <= float(plan.invalidation_price):
            execution_block = MemberSettlementStatus.ENTRY_INVALIDATED_BEFORE_FILL
    if execution_block is not None:
        # Confirmation and execution are separate: a gap beyond either frozen
        # entry bound leaves this allocation in cash, without a later intraday
        # fill being inferred from OHLC data. Legacy plans retain their clock
        # and execution rules when no maximum-entry contract was archived.
        if maturity_date is None:
            return _member_result(
                report=report,
                batch=batch,
                member=member,
                status=MemberSettlementStatus.PENDING,
                published_reference_price=member.reference_price,
                condition_triggered=True,
                reason_code=execution_block.value,
            )
        return _complete_member(
            report=report,
            batch=batch,
            member=member,
            prepared=prepared,
            published_reference_price=float(member.reference_price),
            maturity_date=maturity_date,
            condition_triggered=True,
            execution_block=execution_block,
        )
    if maturity_date is None:
        return _member_result(
            report=report,
            batch=batch,
            member=member,
            status=MemberSettlementStatus.PENDING,
            entry_date=entry_date,
            entry_price=entry_read.value,
            published_reference_price=member.reference_price,
            condition_triggered=True,
            reason_code=MemberSettlementStatus.PENDING.value,
        )
    return _complete_member(
        report=report,
        batch=batch,
        member=member,
        prepared=prepared,
        published_reference_price=float(member.reference_price),
        maturity_date=maturity_date,
        condition_triggered=True,
        simulated_entry_date=entry_date,
        simulated_entry_price=float(entry_read.value),
    )


def _settle_observation_member(
    *,
    report: ArchivedRecommendationReport,
    batch: ArchivedRecommendationBatch,
    member: ArchivedRecommendationMember,
    prepared: _PreparedEvidence,
    maturity_date: date | None,
) -> RecommendationMemberPerformance:
    if member.observation_anchor is ObservationAnchor.ARCHIVED_REFERENCE_PRICE:
        entry_read = _PriceRead(_PriceState.AVAILABLE, member.reference_price)
    else:
        entry_read = _read_price(
            prepared,
            normalize_symbol(member.symbol),
            report.plan_for_date,
            "close",
        )
    entry_failure = _price_failure_status(entry_read, stage="plan")
    if entry_failure is not None:
        return _member_failure(report, batch, member, maturity_date, entry_failure)
    if maturity_date is None:
        return _member_result(
            report=report,
            batch=batch,
            member=member,
            status=MemberSettlementStatus.PENDING,
            entry_date=report.plan_for_date,
            entry_price=entry_read.value,
            published_reference_price=entry_read.value,
            reason_code=MemberSettlementStatus.PENDING.value,
        )
    return _complete_member(
        report=report,
        batch=batch,
        member=member,
        prepared=prepared,
        published_reference_price=float(entry_read.value),
        maturity_date=maturity_date,
        condition_triggered=None,
    )


def _complete_member(
    *,
    report: ArchivedRecommendationReport,
    batch: ArchivedRecommendationBatch,
    member: ArchivedRecommendationMember,
    prepared: _PreparedEvidence,
    published_reference_price: float,
    maturity_date: date,
    condition_triggered: bool | None,
    simulated_entry_date: date | None = None,
    simulated_entry_price: float | None = None,
    execution_block: MemberSettlementStatus | None = None,
) -> RecommendationMemberPerformance:
    symbol = normalize_symbol(member.symbol)
    maturity_read = _read_price(prepared, symbol, maturity_date, "close")
    maturity_failure = _price_failure_status(maturity_read, stage="maturity")
    if maturity_failure is not None:
        return _member_failure(
            report,
            batch,
            member,
            maturity_date,
            maturity_failure,
            entry_date=simulated_entry_date,
            entry_price=simulated_entry_price,
            published_reference_price=published_reference_price,
            condition_triggered=condition_triggered,
        )

    company_action_state = _company_action_state(
        prepared,
        symbol=symbol,
        start=report.plan_for_date,
        end=maturity_date,
    )
    raw_price_change = float(maturity_read.value) / published_reference_price - 1.0
    if company_action_state is None:
        return _member_failure(
            report,
            batch,
            member,
            maturity_date,
            MemberSettlementStatus.CORPORATE_ACTION_EVIDENCE_UNKNOWN,
            entry_date=simulated_entry_date,
            entry_price=simulated_entry_price,
            published_reference_price=published_reference_price,
            maturity_close=maturity_read.value,
            company_action_clear=None,
            condition_triggered=condition_triggered,
            raw_unadjusted_price_change=raw_price_change,
        )
    if company_action_state is False:
        return _member_failure(
            report,
            batch,
            member,
            maturity_date,
            MemberSettlementStatus.CORPORATE_ACTION_DETECTED,
            entry_date=simulated_entry_date,
            entry_price=simulated_entry_price,
            published_reference_price=published_reference_price,
            maturity_close=maturity_read.value,
            company_action_clear=False,
            condition_triggered=condition_triggered,
            raw_unadjusted_price_change=raw_price_change,
        )

    unadjusted_return = raw_price_change
    simulated_return = (
        None
        if simulated_entry_price is None
        else float(maturity_read.value) / simulated_entry_price - 1.0
    )
    status = (
        execution_block
        if execution_block is not None
        else MemberSettlementStatus.EXPIRED_UNTRIGGERED
        if condition_triggered is False
        else MemberSettlementStatus.SETTLED
    )
    return _member_result(
        report=report,
        batch=batch,
        member=member,
        status=status,
        entry_date=simulated_entry_date,
        entry_price=simulated_entry_price,
        published_reference_price=published_reference_price,
        maturity_date=maturity_date,
        maturity_close=float(maturity_read.value),
        unadjusted_price_return=unadjusted_return,
        stock_sleeve_contribution=(member.operational_stock_sleeve_weight * unadjusted_return),
        account_contribution=(
            None
            if member.operational_account_weight is None
            else member.operational_account_weight * unadjusted_return
        ),
        condition_triggered=condition_triggered,
        simulated_action_return=simulated_return,
        simulated_action_stock_sleeve_contribution=(
            None
            if simulated_return is None
            else member.operational_stock_sleeve_weight * simulated_return
        ),
        simulated_action_account_contribution=(
            None
            if simulated_return is None or member.operational_account_weight is None
            else member.operational_account_weight * simulated_return
        ),
        holding_sessions_observed=sum(
            report.plan_for_date < session <= maturity_date for session in prepared.sessions
        ),
        company_action_clear=True,
        reason_code=status.value,
        raw_unadjusted_price_change=raw_price_change,
    )


def _entry_plan_complete(plan: ArchivedEntryPlan) -> bool:
    if plan.kind is None:
        return False
    if plan.invalidation_price is not None and not _valid_positive(plan.invalidation_price):
        return False
    if plan.maximum_entry_price is not None:
        if not _valid_positive(plan.maximum_entry_price) or not _valid_positive(
            plan.invalidation_price
        ):
            return False
        if (
            plan.invalidation_price is not None
            and plan.maximum_entry_price <= plan.invalidation_price
        ):
            return False
    if plan.kind is EntryPlanKind.HEALTHY_PULLBACK:
        return (
            _valid_positive(plan.price_low)
            and _valid_positive(plan.price_high)
            and float(plan.price_low) <= float(plan.price_high)
        )
    if plan.kind is EntryPlanKind.RECLAIM:
        return _valid_positive(plan.trigger_price)
    if not _valid_positive(plan.trigger_price):
        return False
    metric, minimum = _activity_confirmation(plan)
    return metric in {"amount_cny", "volume_shares"} and _valid_positive(minimum)


def _evaluate_trigger(
    prepared: _PreparedEvidence,
    symbol: str,
    plan_date: date,
    plan: ArchivedEntryPlan,
) -> bool | MemberSettlementStatus:
    close_read = _read_price(prepared, symbol, plan_date, "close")
    failure = _price_failure_status(close_read, stage="plan")
    if failure is not None:
        return failure
    close = float(close_read.value)
    if plan.invalidation_price is not None and close < plan.invalidation_price:
        return False
    if plan.kind is EntryPlanKind.RECLAIM:
        return close >= float(plan.trigger_price)
    if plan.kind is EntryPlanKind.VOLUME_BREAKOUT:
        metric, minimum = _activity_confirmation(plan)
        activity_read = _read_price(prepared, symbol, plan_date, metric)
        failure = _price_failure_status(activity_read, stage="plan", allow_zero=True)
        if failure is not None:
            return failure
        return close >= float(plan.trigger_price) and float(activity_read.value) >= float(minimum)

    low_read = _read_price(prepared, symbol, plan_date, "low")
    high_read = _read_price(prepared, symbol, plan_date, "high")
    for read in (low_read, high_read):
        failure = _price_failure_status(read, stage="plan")
        if failure is not None:
            return failure
    return float(low_read.value) <= float(plan.price_high) and float(high_read.value) >= float(
        plan.price_low
    )


def _read_price(
    prepared: _PreparedEvidence,
    symbol: str,
    trade_date: date,
    column: str,
) -> _PriceRead:
    if trade_date in prepared.suspended_dates_by_symbol.get(symbol, frozenset()):
        return _PriceRead(_PriceState.SUSPENDED)
    if trade_date in prepared.invalid_dates_by_symbol.get(symbol, frozenset()):
        return _PriceRead(_PriceState.INVALID)
    rows = prepared.prices.loc[
        (prepared.prices["_normalized_symbol"] == symbol)
        & (prepared.prices["_trade_date"] == trade_date)
    ]
    if rows.empty:
        return _PriceRead(_PriceState.MISSING)
    if len(rows) != 1 or bool(rows.iloc[0]["_duplicate"]):
        return _PriceRead(_PriceState.INVALID)
    row = rows.iloc[0]
    if "is_suspended" in rows.columns and bool(row["is_suspended"]):
        return _PriceRead(_PriceState.SUSPENDED)
    if "volume_shares" in rows.columns:
        volume = _finite_optional(row["volume_shares"])
        if volume is not None and volume <= 0.0:
            return _PriceRead(_PriceState.SUSPENDED)
    if column not in rows.columns:
        return _PriceRead(_PriceState.MISSING)
    value = _finite_optional(row[column])
    if value is None:
        return _PriceRead(_PriceState.INVALID)
    if column != "volume_shares" and value <= 0.0:
        return _PriceRead(_PriceState.INVALID)
    if column == "volume_shares" and value < 0.0:
        return _PriceRead(_PriceState.INVALID)
    return _PriceRead(_PriceState.AVAILABLE, value)


def _price_failure_status(
    read: _PriceRead,
    *,
    stage: Literal["plan", "entry", "maturity"],
    allow_zero: bool = False,
) -> MemberSettlementStatus | None:
    del allow_zero  # Zero validity is handled by the requested column in _read_price.
    if read.state is _PriceState.AVAILABLE:
        return None
    mapping = {
        "plan": {
            _PriceState.MISSING: MemberSettlementStatus.PLAN_PRICE_MISSING,
            _PriceState.INVALID: MemberSettlementStatus.PLAN_PRICE_INVALID,
            _PriceState.SUSPENDED: MemberSettlementStatus.PLAN_SESSION_SUSPENDED,
        },
        "entry": {
            _PriceState.MISSING: MemberSettlementStatus.ENTRY_PRICE_MISSING,
            _PriceState.INVALID: MemberSettlementStatus.ENTRY_PRICE_INVALID,
            _PriceState.SUSPENDED: MemberSettlementStatus.ENTRY_SESSION_SUSPENDED,
        },
        "maturity": {
            _PriceState.MISSING: MemberSettlementStatus.MATURITY_PRICE_MISSING,
            _PriceState.INVALID: MemberSettlementStatus.MATURITY_PRICE_INVALID,
            _PriceState.SUSPENDED: MemberSettlementStatus.MATURITY_SESSION_SUSPENDED,
        },
    }
    return mapping[stage][read.state]


def _company_action_state(
    prepared: _PreparedEvidence,
    *,
    symbol: str,
    start: date,
    end: date,
) -> bool | None:
    if symbol not in prepared.corporate_action_coverage_symbols:
        return None
    actions = prepared.corporate_action_dates_by_symbol.get(symbol, frozenset())
    return not any(start <= action_date <= end for action_date in actions)


def _activity_confirmation(plan: ArchivedEntryPlan) -> tuple[str, float | None]:
    """Return the immutable activity field and threshold for one plan.

    The legacy fallback exists only so already archived local records remain
    readable.  It does not invent a missing threshold.
    """

    if plan.confirmation_activity_metric is not None:
        return plan.confirmation_activity_metric, plan.confirmation_activity_min
    return "volume_shares", plan.confirmation_volume_min


def _aggregate_batch(
    *,
    report: ArchivedRecommendationReport,
    batch: ArchivedRecommendationBatch,
    member_results: tuple[RecommendationMemberPerformance, ...],
    maturity_date: date | None,
    data_cutoff: date | None,
) -> RecommendationBatchPerformance:
    fatal = [result for result in member_results if result.status in _FATAL_MEMBER_STATUSES]
    raw_complete = bool(member_results) and all(
        result.raw_unadjusted_price_change is not None for result in member_results
    )
    raw_stock_sleeve_change = (
        sum(
            result.operational_stock_sleeve_weight * float(result.raw_unadjusted_price_change)
            for result in member_results
        )
        if raw_complete
        else None
    )
    raw_account_change = (
        None
        if not raw_complete
        or any(result.operational_account_weight is None for result in member_results)
        else sum(
            float(result.operational_account_weight) * float(result.raw_unadjusted_price_change)
            for result in member_results
        )
    )
    # No cohort is mature before its session-based due date.  We may already
    # know that an archived rule is incomplete, but surfacing that as a final
    # review before maturity would send an "到期日未知" notification and then
    # suppress the real due-date result.  Keep the batch retryable until the
    # immutable horizon actually elapses.
    if maturity_date is None:
        status = BatchSettlementStatus.PENDING
        stock_sleeve_return = None
        account_return = None
        simulated_action_stock_sleeve_return = None
        simulated_action_account_return = None
        reason_code = BatchSettlementStatus.PENDING.value
    elif fatal:
        status = BatchSettlementStatus.DATA_QUALITY_FAILURE
        stock_sleeve_return = None
        account_return = None
        simulated_action_stock_sleeve_return = None
        simulated_action_account_return = None
        reason_code = fatal[0].reason_code
    elif any(result.status is MemberSettlementStatus.PENDING for result in member_results):
        status = BatchSettlementStatus.PENDING
        stock_sleeve_return = None
        account_return = None
        simulated_action_stock_sleeve_return = None
        simulated_action_account_return = None
        reason_code = BatchSettlementStatus.PENDING.value
    else:
        evaluated = [
            result
            for result in member_results
            if result.status
            in {
                MemberSettlementStatus.SETTLED,
                MemberSettlementStatus.EXPIRED_UNTRIGGERED,
                MemberSettlementStatus.ENTRY_PRICE_LIMIT_EXCEEDED,
                MemberSettlementStatus.ENTRY_INVALIDATED_BEFORE_FILL,
            }
        ]
        settled = [
            result for result in evaluated if result.status is MemberSettlementStatus.SETTLED
        ]
        untriggered = [
            result
            for result in member_results
            if result.status
            in {
                MemberSettlementStatus.EXPIRED_UNTRIGGERED,
                MemberSettlementStatus.ENTRY_PRICE_LIMIT_EXCEEDED,
                MemberSettlementStatus.ENTRY_INVALIDATED_BEFORE_FILL,
            }
        ]
        stock_sleeve_return = sum(float(result.stock_sleeve_contribution) for result in evaluated)
        account_return = (
            None
            if any(result.account_contribution is None for result in evaluated)
            else sum(float(result.account_contribution) for result in evaluated)
        )
        simulated_action_stock_sleeve_return = (
            None
            if batch.evaluation_mode is not EvaluationMode.ACTION_SIMULATION
            else sum(float(result.simulated_action_stock_sleeve_contribution) for result in settled)
        )
        simulated_action_account_return = (
            None
            if batch.evaluation_mode is not EvaluationMode.ACTION_SIMULATION
            or any(result.simulated_action_account_contribution is None for result in settled)
            else sum(float(result.simulated_action_account_contribution) for result in settled)
        )
        if untriggered and settled:
            status = BatchSettlementStatus.SETTLED_PARTIAL_ENTRY
        elif untriggered:
            status = BatchSettlementStatus.NO_ENTRY
        else:
            status = BatchSettlementStatus.SETTLED
        reason_code = status.value

    entered = [
        result
        for result in member_results
        if result.status in {MemberSettlementStatus.SETTLED, MemberSettlementStatus.PENDING}
        and result.entry_price is not None
    ]
    entered_sleeve = _unit_weight_sum(result.operational_stock_sleeve_weight for result in entered)
    entered_account = (
        None
        if any(result.operational_account_weight is None for result in entered)
        else _unit_weight_sum(float(result.operational_account_weight) for result in entered)
    )
    cash_is_meaningful = batch.evaluation_mode is EvaluationMode.ACTION_SIMULATION
    return RecommendationBatchPerformance(
        batch_id=batch.batch_id,
        report_id=report.report_id,
        holding_weeks=batch.holding_weeks,
        holding_sessions=batch.holding_sessions,
        evaluation_mode=batch.evaluation_mode,
        cohort_nature=batch.cohort_nature,
        status=status,
        plan_date=report.plan_for_date,
        maturity_date=maturity_date,
        stock_sleeve_return=stock_sleeve_return,
        account_return=account_return,
        simulated_action_stock_sleeve_return=simulated_action_stock_sleeve_return,
        simulated_action_account_return=simulated_action_account_return,
        entered_stock_sleeve_weight=entered_sleeve,
        entered_account_weight=entered_account,
        stock_sleeve_cash_weight=(max(0.0, 1.0 - entered_sleeve) if cash_is_meaningful else None),
        account_cash_weight=(
            max(0.0, 1.0 - entered_account)
            if cash_is_meaningful and entered_account is not None
            else None
        ),
        members=member_results,
        data_cutoff=data_cutoff,
        reason_code=reason_code,
        raw_unadjusted_stock_sleeve_change=raw_stock_sleeve_change,
        raw_unadjusted_account_change=raw_account_change,
        holding_clock=batch.holding_clock,
    )


def _unit_weight_sum(values) -> float:
    """Remove only boundary-sized roundoff; reject materially invalid exposure."""

    total = math.fsum(values)
    if not math.isfinite(total) or total < -1e-8 or total > 1.0 + 1e-8:
        raise ValueError("aggregate allocation weight is outside [0, 1]")
    return min(1.0, max(0.0, total))


def _pending_batch(
    *,
    report: ArchivedRecommendationReport,
    batch: ArchivedRecommendationBatch,
    members: tuple[ArchivedRecommendationMember, ...],
    maturity_date: date | None,
    data_cutoff: date | None,
) -> RecommendationBatchPerformance:
    results = tuple(
        _member_result(
            report=report,
            batch=batch,
            member=member,
            status=MemberSettlementStatus.PENDING,
            maturity_date=maturity_date,
            reason_code=MemberSettlementStatus.PENDING.value,
        )
        for member in members
    )
    return _aggregate_batch(
        report=report,
        batch=batch,
        member_results=results,
        maturity_date=maturity_date,
        data_cutoff=data_cutoff,
    )


def _uniform_failure(
    *,
    report: ArchivedRecommendationReport,
    batch: ArchivedRecommendationBatch,
    members: tuple[ArchivedRecommendationMember, ...],
    data_cutoff: date | None,
    status: MemberSettlementStatus,
) -> RecommendationBatchPerformance:
    results = tuple(_member_failure(report, batch, member, None, status) for member in members)
    return _aggregate_batch(
        report=report,
        batch=batch,
        member_results=results,
        maturity_date=None,
        data_cutoff=data_cutoff,
    )


def _member_failure(
    report: ArchivedRecommendationReport,
    batch: ArchivedRecommendationBatch,
    member: ArchivedRecommendationMember,
    maturity_date: date | None,
    status: MemberSettlementStatus,
    *,
    entry_date: date | None = None,
    entry_price: float | None = None,
    published_reference_price: float | None = None,
    maturity_close: float | None = None,
    raw_unadjusted_price_change: float | None = None,
    company_action_clear: bool | None = None,
    condition_triggered: bool | None = None,
) -> RecommendationMemberPerformance:
    return _member_result(
        report=report,
        batch=batch,
        member=member,
        status=status,
        entry_date=entry_date,
        entry_price=entry_price,
        published_reference_price=published_reference_price,
        maturity_date=maturity_date,
        maturity_close=maturity_close,
        raw_unadjusted_price_change=raw_unadjusted_price_change,
        company_action_clear=company_action_clear,
        condition_triggered=condition_triggered,
        reason_code=status.value,
    )


def _member_result(
    *,
    report: ArchivedRecommendationReport,
    batch: ArchivedRecommendationBatch,
    member: ArchivedRecommendationMember,
    status: MemberSettlementStatus,
    reason_code: str,
    entry_date: date | None = None,
    entry_price: float | None = None,
    published_reference_price: float | None = None,
    maturity_date: date | None = None,
    maturity_close: float | None = None,
    raw_unadjusted_price_change: float | None = None,
    unadjusted_price_return: float | None = None,
    stock_sleeve_contribution: float | None = None,
    account_contribution: float | None = None,
    condition_triggered: bool | None = None,
    simulated_action_return: float | None = None,
    simulated_action_stock_sleeve_contribution: float | None = None,
    simulated_action_account_contribution: float | None = None,
    holding_sessions_observed: int | None = None,
    company_action_clear: bool | None = None,
) -> RecommendationMemberPerformance:
    return RecommendationMemberPerformance(
        member_id=member.member_id,
        symbol=normalize_symbol(member.symbol),
        name=member.name,
        status=status,
        evaluation_mode=batch.evaluation_mode,
        plan_date=report.plan_for_date,
        entry_date=entry_date,
        entry_price=entry_price,
        published_reference_price=published_reference_price,
        maturity_date=maturity_date,
        maturity_close=maturity_close,
        raw_unadjusted_price_change=raw_unadjusted_price_change,
        unadjusted_price_return=unadjusted_price_return,
        stock_sleeve_contribution=stock_sleeve_contribution,
        account_contribution=account_contribution,
        condition_triggered=condition_triggered,
        simulated_action_return=simulated_action_return,
        simulated_action_stock_sleeve_contribution=(simulated_action_stock_sleeve_contribution),
        simulated_action_account_contribution=simulated_action_account_contribution,
        operational_stock_sleeve_weight=member.operational_stock_sleeve_weight,
        operational_account_weight=member.operational_account_weight,
        holding_sessions_observed=holding_sessions_observed,
        company_action_clear=company_action_clear,
        reason_code=reason_code,
        maximum_entry_price=(
            None if member.entry_plan is None else member.entry_plan.maximum_entry_price
        ),
        entry_execution_rule_version=(
            "archived-entry-price-bounds-v1"
            if member.entry_plan is not None and member.entry_plan.maximum_entry_price is not None
            else "legacy-next-open-v1"
        ),
    )


def _try_normalize_symbol(value: object) -> str | None:
    try:
        return normalize_symbol(str(value))
    except ValueError:
        return None


def _normalized_mapping_keys(values: Sequence[str] | frozenset[str]) -> tuple[str, ...]:
    return tuple(normalize_symbol(value) for value in values)


def _normalize_date_mapping(
    values: Mapping[str, frozenset[date]],
) -> dict[str, frozenset[date]]:
    return {normalize_symbol(symbol): frozenset(dates) for symbol, dates in values.items()}


def _finite_optional(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _valid_positive(value: object) -> bool:
    parsed = _finite_optional(value)
    return parsed is not None and parsed > 0.0


def _entry_plan_from_payload(value: object) -> ArchivedEntryPlan | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("entry_plan_json is not valid JSON") from exc
    elif isinstance(value, Mapping):
        payload = value
    else:
        raise ValueError("entry plan must be a mapping or JSON object")
    if not isinstance(payload, Mapping):
        raise ValueError("entry plan JSON must contain an object")
    kind_value = payload.get("kind")
    return ArchivedEntryPlan(
        kind=None if kind_value is None else EntryPlanKind(str(kind_value)),
        price_low=_optional_float(payload.get("price_low")),
        price_high=_optional_float(payload.get("price_high")),
        trigger_price=_optional_float(payload.get("trigger_price")),
        invalidation_price=_optional_float(payload.get("invalidation_price")),
        confirmation_activity_metric=_optional_text(payload.get("confirmation_activity_metric")),
        confirmation_activity_min=_optional_float(payload.get("confirmation_activity_min")),
        confirmation_volume_min=_optional_float(payload.get("confirmation_volume_min")),
        maximum_entry_price=_optional_float(payload.get("maximum_entry_price")),
    )


def _required(row: Mapping[str, object], *keys: str) -> object:
    value = _first_present(row, *keys)
    if value is None or value == "":
        raise ValueError("missing required archive field: " + "/".join(keys))
    return value


def _first_present(row: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _as_date(value: object) -> date:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("archive date is invalid")
    return parsed.date()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no", ""}:
        return False
    raise ValueError("boolean archive field is invalid")


def _optional_float(value: object) -> float | None:
    return None if value is None or value == "" else float(value)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _iso_date(value: date | None) -> str | None:
    return None if value is None else value.isoformat()
