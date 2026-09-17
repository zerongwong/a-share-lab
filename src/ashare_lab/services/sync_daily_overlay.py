"""Synchronize completed daily increments into a verified market overlay."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Any

import pandas as pd

from ashare_lab.adapters.market_overlay_store import (
    MarketOverlayStore,
    normalize_overlay_daily,
)
from ashare_lab.domain.data_sources import DataAction, RightsPolicy
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.daily_increment import AssetKind, DailyIncrementBatch, DailyIncrementPort
from ashare_lab.ports.market_data import normalize_symbol


class DailyOverlaySyncStatus(StrEnum):
    VERIFIED = "verified"
    UNCHANGED = "unchanged"
    FAILED = "failed"


class DailyOverlayFailureStage(StrEnum):
    """Stable, non-secret stage identifiers for failed daily synchronizations."""

    CALENDAR = "calendar"
    STOCK_MASTER = "stock_master"
    STOCK_DAILY = "stock_daily"
    STOCK_VALIDATION = "stock_validation"
    INDEPENDENT_VERIFICATION = "independent_verification"
    INDEX_DAILY = "index_daily"
    INDEX_VALIDATION = "index_validation"
    UNIT_VALIDATION = "unit_validation"
    PERSISTENCE = "persistence"
    UNKNOWN = "unknown"


_STOCK_UNEXPECTED_COUNT = re.compile(
    r"^stock increment returned unrequested symbol count: (?P<count>\d+)$"
)
_INDEX_MISSING_COUNT = re.compile(r"^core index increment missing symbol count: (?P<count>\d+)$")
_INDEX_UNEXPECTED_COUNT = re.compile(
    r"^core index increment returned unrequested symbol count: (?P<count>\d+)$"
)


@dataclass(frozen=True, slots=True)
class DailyOverlaySyncResult:
    source_id: str
    trade_date: date
    previous_cutoff: date
    verified_cutoff: date
    status: DailyOverlaySyncStatus
    expected_stock_count: int
    stock_count: int = 0
    stock_coverage_ratio: float = 0.0
    index_count: int = 0
    stock_checksum: str = ""
    index_checksum: str = ""
    run_id: str = ""
    quarantine_path: str | None = None
    reason: str = ""
    failure_stage: str = ""
    failure_provider: str = ""
    failure_status: str = ""
    reason_code: str = ""


@dataclass(frozen=True, slots=True)
class DailyOverlayRangeReport:
    source_id: str
    baseline_cutoff: date
    requested_through: date
    started_cutoff: date
    verified_cutoff: date
    expected_sessions: tuple[date, ...]
    completed_sessions: tuple[date, ...]
    results: tuple[DailyOverlaySyncResult, ...]
    ready_through_requested_date: bool


def sync_daily_overlay(
    provider: DailyIncrementPort,
    store: MarketOverlayStore,
    *,
    target_date: date,
    previous_trade_date: date,
    core_index_symbols: Sequence[str],
    stock_symbols: Sequence[str] | None = None,
    required_stock_coverage_ratio: float = 0.98,
    cutoff_timestamp: int | None = None,
    rights_policy: RightsPolicy | None = None,
    clock: Callable[[], datetime] | None = None,
) -> DailyOverlaySyncResult:
    """Fetch, stage and atomically verify one unadjusted market session.

    Stocks and indices are fetched independently and staged as separate
    artifacts.  Any provider, date, schema, identity, coverage or core-index
    failure quarantines the run and leaves the prior verified cutoff intact.
    """

    source_id = _provider_id(provider)
    _validate_options(
        target_date=target_date,
        previous_trade_date=previous_trade_date,
        required_stock_coverage_ratio=required_stock_coverage_ratio,
        core_index_symbols=core_index_symbols,
    )
    if rights_policy is not None:
        rights_policy.require(source_id, DataAction.MARKET_DATA_READ)
        rights_policy.require(source_id, DataAction.MARKET_DATA_CACHE)
    now = clock or (lambda: datetime.now(UTC))
    try:
        stocks_requested = tuple(stock_symbols or provider.fetch_cn_stock_symbols())
        if not stocks_requested:
            raise DataUnavailableError("provider stock universe is empty")
    except Exception as exc:  # noqa: BLE001 - fail closed at metadata boundary
        return _failed_result(
            source_id=source_id,
            trade_date=target_date,
            previous_cutoff=previous_trade_date,
            expected_stock_count=0,
            stage=DailyOverlayFailureStage.STOCK_MASTER,
            exc=exc,
        )
    core_requested = tuple(core_index_symbols)
    try:
        run = store.begin_staging(
            source_id=source_id,
            trade_date=target_date,
            receipt={
                "provider": source_id,
                "target_date": target_date,
                "previous_trade_date": previous_trade_date,
                "state": "started",
            },
        )
    except Exception as exc:  # noqa: BLE001 - fail closed at local persistence boundary
        return _failed_result(
            source_id=source_id,
            trade_date=target_date,
            previous_cutoff=previous_trade_date,
            expected_stock_count=len(stocks_requested),
            stage=DailyOverlayFailureStage.PERSISTENCE,
            exc=exc,
        )
    receipt: dict[str, Any] = {
        "provider": source_id,
        "target_date": target_date,
        "previous_trade_date": previous_trade_date,
        "adjustment": "none",
    }
    stage = DailyOverlayFailureStage.STOCK_DAILY
    try:
        stock_batch = provider.fetch_daily_increment(
            stocks_requested,
            target_date,
            cutoff_timestamp=cutoff_timestamp,
            asset_kind="stocks",
        )
        stage = DailyOverlayFailureStage.PERSISTENCE
        store.stage_asset(run, "stocks", stock_batch.frame)
        stage = DailyOverlayFailureStage.STOCK_VALIDATION
        stocks = _validate_batch(
            stock_batch,
            requested_symbols=stocks_requested,
            target_date=target_date,
            source_id=source_id,
            asset_kind="stocks",
        )
        stock_expected = _normalize_configured_symbols(stocks_requested)
        unexpected_stocks = set(stocks["symbol"]) - set(stock_expected)
        if unexpected_stocks:
            raise DataQualityError(
                f"stock increment returned unrequested symbol count: {len(unexpected_stocks)}"
            )
        stock_coverage = len(set(stocks["symbol"])) / len(stock_expected)
        if stock_coverage + 1e-12 < required_stock_coverage_ratio:
            raise DataQualityError(
                f"stock coverage {stock_coverage:.4f} is below {required_stock_coverage_ratio:.4f}"
            )
        receipt["stocks"] = _batch_receipt(stock_batch, asset_kind="stocks")
        stage = DailyOverlayFailureStage.PERSISTENCE
        store.update_staging_receipt(run, receipt)

        stage = DailyOverlayFailureStage.INDEX_DAILY
        index_batch = provider.fetch_daily_increment(
            core_requested,
            target_date,
            cutoff_timestamp=cutoff_timestamp,
            asset_kind="indices",
        )
        stage = DailyOverlayFailureStage.PERSISTENCE
        store.stage_asset(run, "indices", index_batch.frame)
        stage = DailyOverlayFailureStage.INDEX_VALIDATION
        indices = _validate_batch(
            index_batch,
            requested_symbols=core_requested,
            target_date=target_date,
            source_id=source_id,
            asset_kind="indices",
        )
        stage = DailyOverlayFailureStage.UNIT_VALIDATION
        _validate_unit_audit_pair(stock_batch, index_batch)
        stage = DailyOverlayFailureStage.INDEX_VALIDATION
        core_normalized = _normalize_configured_symbols(core_requested)
        actual_indices = set(indices["symbol"])
        missing_indices = set(core_normalized) - actual_indices
        unexpected_indices = actual_indices - set(core_normalized)
        if missing_indices:
            raise DataQualityError(
                f"core index increment missing symbol count: {len(missing_indices)}"
            )
        if unexpected_indices:
            raise DataQualityError(
                f"core index increment returned unrequested symbol count: {len(unexpected_indices)}"
            )
        receipt["indices"] = _batch_receipt(index_batch, asset_kind="indices")
        stage = DailyOverlayFailureStage.PERSISTENCE
        store.update_staging_receipt(run, receipt)

        stage = DailyOverlayFailureStage.PERSISTENCE
        summary = store.commit_verified(
            run,
            stocks=stocks,
            indices=indices,
            previous_trade_date=previous_trade_date,
            expected_stock_count=len(stock_expected),
            stock_coverage_ratio=stock_coverage,
            core_index_symbols=core_normalized,
            receipt=receipt,
            verified_at=_aware_utc(now(), "clock"),
        )
        status = (
            DailyOverlaySyncStatus.UNCHANGED
            if summary.unchanged
            else DailyOverlaySyncStatus.VERIFIED
        )
        return DailyOverlaySyncResult(
            source_id=source_id,
            trade_date=target_date,
            previous_cutoff=previous_trade_date,
            verified_cutoff=target_date,
            status=status,
            expected_stock_count=summary.expected_stock_count,
            stock_count=summary.stock_count,
            stock_coverage_ratio=summary.stock_coverage_ratio,
            index_count=summary.index_count,
            stock_checksum=summary.stock_checksum,
            index_checksum=summary.index_checksum,
            run_id=summary.run_id,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed at provider boundary
        stage = _refine_failure_stage(stage, exc)
        diagnostic = _failure_diagnostic(source_id=source_id, stage=stage, exc=exc)
        quarantine_path: str | None = None
        if run.path.is_dir():
            quarantine_path = str(
                store.quarantine(
                    run,
                    reason=diagnostic["reason"],
                    failed_at=_aware_utc(now(), "clock"),
                )
            )
        return _failed_result(
            source_id=source_id,
            trade_date=target_date,
            previous_cutoff=previous_trade_date,
            expected_stock_count=len(stocks_requested),
            quarantine_path=quarantine_path,
            stage=stage,
            exc=exc,
            diagnostic=diagnostic,
        )


def sync_daily_overlay_range(
    provider: DailyIncrementPort,
    store: MarketOverlayStore,
    *,
    baseline_cutoff: date,
    through_date: date,
    core_index_symbols: Sequence[str],
    stock_symbols: Sequence[str] | None = None,
    required_stock_coverage_ratio: float = 0.98,
    cutoff_timestamp_by_date: Mapping[date, int] | None = None,
    rights_policy: RightsPolicy | None = None,
    clock: Callable[[], datetime] | None = None,
) -> DailyOverlayRangeReport:
    """Catch up every provider-confirmed session after an immutable baseline.

    The manifest's ``previous_trade_date`` chain is authoritative.  The
    provider calendar supplies the missing open sessions, so weekends and
    statutory holidays are never guessed.  The loop stops on the first failed
    date; a later date therefore cannot conceal a gap or advance the cutoff.
    """

    source_id = _provider_id(provider)
    if through_date < baseline_cutoff:
        raise ValueError("through_date cannot precede baseline_cutoff")
    if rights_policy is not None:
        rights_policy.require(source_id, DataAction.MARKET_DATA_READ)
        rights_policy.require(source_id, DataAction.MARKET_DATA_CACHE)
    chain = store.verified_dates_from(
        source_id=source_id,
        baseline_cutoff=baseline_cutoff,
        through_date=through_date,
    )
    started_cutoff = chain[-1] if chain else baseline_cutoff
    if started_cutoff >= through_date:
        return DailyOverlayRangeReport(
            source_id=source_id,
            baseline_cutoff=baseline_cutoff,
            requested_through=through_date,
            started_cutoff=started_cutoff,
            verified_cutoff=started_cutoff,
            expected_sessions=(),
            completed_sessions=(),
            results=(),
            ready_through_requested_date=True,
        )
    try:
        calendar = _normalize_calendar(
            provider.fetch_cn_trading_days(started_cutoff + timedelta(days=1), through_date),
            start=started_cutoff + timedelta(days=1),
            end=through_date,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed at metadata boundary
        failure = _failed_result(
            source_id=source_id,
            trade_date=through_date,
            previous_cutoff=started_cutoff,
            expected_stock_count=0,
            stage=DailyOverlayFailureStage.CALENDAR,
            exc=exc,
        )
        return DailyOverlayRangeReport(
            source_id=source_id,
            baseline_cutoff=baseline_cutoff,
            requested_through=through_date,
            started_cutoff=started_cutoff,
            verified_cutoff=started_cutoff,
            expected_sessions=(),
            completed_sessions=(),
            results=(failure,),
            ready_through_requested_date=False,
        )
    if not calendar:
        return DailyOverlayRangeReport(
            source_id=source_id,
            baseline_cutoff=baseline_cutoff,
            requested_through=through_date,
            started_cutoff=started_cutoff,
            verified_cutoff=started_cutoff,
            expected_sessions=(),
            completed_sessions=(),
            results=(),
            ready_through_requested_date=True,
        )
    try:
        stocks_requested = tuple(stock_symbols or provider.fetch_cn_stock_symbols())
        if not stocks_requested:
            raise DataUnavailableError("provider stock universe is empty")
    except Exception as exc:  # noqa: BLE001 - fail closed at metadata boundary
        failure = _failed_result(
            source_id=source_id,
            trade_date=calendar[0],
            previous_cutoff=started_cutoff,
            expected_stock_count=0,
            stage=DailyOverlayFailureStage.STOCK_MASTER,
            exc=exc,
        )
        return DailyOverlayRangeReport(
            source_id=source_id,
            baseline_cutoff=baseline_cutoff,
            requested_through=through_date,
            started_cutoff=started_cutoff,
            verified_cutoff=started_cutoff,
            expected_sessions=calendar,
            completed_sessions=(),
            results=(failure,),
            ready_through_requested_date=False,
        )
    results: list[DailyOverlaySyncResult] = []
    completed: list[date] = []
    current_cutoff = started_cutoff
    cutoffs = cutoff_timestamp_by_date or {}
    for target_date in calendar:
        result = sync_daily_overlay(
            provider,
            store,
            target_date=target_date,
            previous_trade_date=current_cutoff,
            core_index_symbols=core_index_symbols,
            stock_symbols=stocks_requested,
            required_stock_coverage_ratio=required_stock_coverage_ratio,
            cutoff_timestamp=cutoffs.get(target_date),
            rights_policy=rights_policy,
            clock=clock,
        )
        results.append(result)
        if result.status is DailyOverlaySyncStatus.FAILED:
            break
        completed.append(target_date)
        current_cutoff = target_date
    return DailyOverlayRangeReport(
        source_id=source_id,
        baseline_cutoff=baseline_cutoff,
        requested_through=through_date,
        started_cutoff=started_cutoff,
        verified_cutoff=current_cutoff,
        expected_sessions=calendar,
        completed_sessions=tuple(completed),
        results=tuple(results),
        ready_through_requested_date=(not calendar or current_cutoff == calendar[-1]),
    )


def _validate_batch(
    batch: DailyIncrementBatch,
    *,
    requested_symbols: Sequence[str],
    target_date: date,
    source_id: str,
    asset_kind: AssetKind,
) -> pd.DataFrame:
    if not isinstance(batch, DailyIncrementBatch):
        # Permit structurally compatible implementations while preserving the
        # same audit fields required by DailyIncrementPort.
        required = {
            "frame",
            "target_date",
            "requested_symbols",
            "received_symbols",
            "fetched_at",
            "trace_ids",
            "provider",
            "cutoff_timestamp",
        }
        missing = [name for name in sorted(required) if not hasattr(batch, name)]
        if missing:
            raise DataQualityError("daily increment receipt is missing: " + ", ".join(missing))
    if batch.target_date != target_date:
        raise DataQualityError("daily increment target_date does not match request")
    if str(batch.provider).strip().lower() != source_id:
        raise DataQualityError("daily increment provider does not match adapter")
    _aware_utc(batch.fetched_at, "batch.fetched_at")
    if isinstance(batch.cutoff_timestamp, bool) or not isinstance(batch.cutoff_timestamp, int):
        raise DataQualityError("daily increment cutoff_timestamp must be an integer")
    if batch.cutoff_timestamp <= 0:
        raise DataQualityError("daily increment cutoff_timestamp must be positive")
    requested = tuple(dict.fromkeys(str(value).strip().upper() for value in requested_symbols))
    batch_requested = tuple(
        dict.fromkeys(str(value).strip().upper() for value in batch.requested_symbols)
    )
    received = tuple(str(value).strip().upper() for value in batch.received_symbols)
    if set(batch_requested) != set(requested):
        raise DataQualityError("daily increment receipt requested_symbols do not match")
    if len(set(received)) != len(received):
        raise DataQualityError("daily increment receipt contains duplicate received_symbols")
    if not set(received).issubset(set(batch_requested)):
        raise DataQualityError("daily increment receipt contains unrequested received_symbols")
    return normalize_overlay_daily(
        batch.frame,
        expected_date=target_date,
        source_id=source_id,
        asset_kind=asset_kind,
    )


def _batch_receipt(batch: DailyIncrementBatch, *, asset_kind: AssetKind) -> dict[str, Any]:
    receipt = {
        "asset_kind": asset_kind,
        "provider": str(batch.provider),
        "target_date": batch.target_date,
        "requested_symbols": tuple(batch.requested_symbols),
        "received_symbols": tuple(batch.received_symbols),
        "coverage_ratio": float(batch.coverage_ratio),
        "fetched_at": batch.fetched_at,
        "trace_ids": tuple(batch.trace_ids),
        "cutoff_timestamp": int(batch.cutoff_timestamp),
    }
    if batch.metadata_sources:
        receipt["metadata_sources"] = batch.metadata_sources
    for field in (
        "unit_contract_version",
        "unit_resolution_method_version",
        "amount_multiplier_to_cny",
    ):
        value = str(getattr(batch, field, "") or "").strip()
        if value:
            receipt[field] = value
    return receipt


def _validate_unit_audit_pair(
    stock_batch: DailyIncrementBatch,
    index_batch: DailyIncrementBatch,
) -> None:
    fields = (
        "unit_contract_version",
        "unit_resolution_method_version",
        "amount_multiplier_to_cny",
    )
    stocks = tuple(str(getattr(stock_batch, field, "") or "").strip() for field in fields)
    indices = tuple(str(getattr(index_batch, field, "") or "").strip() for field in fields)
    if not any(stocks) and not any(indices):
        return
    if not all(stocks) or not all(indices):
        raise DataQualityError("daily increment unit audit metadata is incomplete")
    if stocks != indices:
        raise DataQualityError("stock and index unit audit metadata do not match")


def _normalize_configured_symbols(values: Sequence[str]) -> tuple[str, ...]:
    try:
        normalized = tuple(
            dict.fromkeys(normalize_symbol(str(value).strip().upper()) for value in values)
        )
    except ValueError as exc:
        raise DataQualityError(f"configured symbol is invalid: {exc}") from exc
    if not normalized:
        raise DataQualityError("configured symbols cannot be empty")
    if len(normalized) != len(tuple(values)):
        raise DataQualityError("configured symbols contain duplicates after normalization")
    return tuple(sorted(normalized))


def _normalize_calendar(
    values: Sequence[date],
    *,
    start: date,
    end: date,
) -> tuple[date, ...]:
    if any(not isinstance(value, date) or isinstance(value, datetime) for value in values):
        raise DataQualityError("provider calendar must contain date values")
    sessions = tuple(values)
    if len(set(sessions)) != len(sessions):
        raise DataQualityError("provider calendar contains duplicate sessions")
    if tuple(sorted(sessions)) != sessions:
        raise DataQualityError("provider calendar is not sorted")
    if any(value < start or value > end for value in sessions):
        raise DataQualityError("provider calendar returned a session outside the request")
    return sessions


def _provider_id(provider: DailyIncrementPort) -> str:
    raw = getattr(provider, "provider", None)
    if raw is None:
        raw = getattr(provider, "source_id", None)
    if hasattr(raw, "value"):
        raw = raw.value
    normalized = str(raw or "").strip().lower()
    if not normalized:
        raise ValueError("daily increment provider must expose provider or source_id")
    return normalized


def _validate_options(
    *,
    target_date: date,
    previous_trade_date: date,
    required_stock_coverage_ratio: float,
    core_index_symbols: Sequence[str],
) -> None:
    if previous_trade_date >= target_date:
        raise ValueError("previous_trade_date must precede target_date")
    if not 0 < required_stock_coverage_ratio <= 1:
        raise ValueError("required_stock_coverage_ratio must be in (0, 1]")
    if not core_index_symbols:
        raise ValueError("at least one core index is required")


def _aware_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DataQualityError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


_SAFE_COVERAGE = re.compile(
    r"^stock coverage (?P<actual>\d+(?:\.\d+)?) is below (?P<required>\d+(?:\.\d+)?)$"
)
_PRE_SANITIZED_DEPENDENCY = re.compile(
    r"^(?:免费交易日历|免费证券主表|Tushare股票日线|独立日线核验|"
    r"免费核心指数)(?:质量校验失败|不可用|调用失败)，原始错误已脱敏。$"
)
_STOCK_MASTER_CONSENSUS_FAILURE = re.compile(
    r"^证券主表共识未通过（reason=(?P<reason>[a-z0-9_]+);"
    r"available=(?:none|(?:baostock|official_exchange|tushare)"
    r"(?:,(?:baostock|official_exchange|tushare))*);"
    r"unavailable=(?:none|(?:baostock|official_exchange|tushare)"
    r"(?:,(?:baostock|official_exchange|tushare))*)）。$"
)


def _failed_result(
    *,
    source_id: str,
    trade_date: date,
    previous_cutoff: date,
    expected_stock_count: int,
    stage: DailyOverlayFailureStage,
    exc: Exception,
    quarantine_path: str | None = None,
    diagnostic: dict[str, str] | None = None,
) -> DailyOverlaySyncResult:
    values = diagnostic or _failure_diagnostic(source_id=source_id, stage=stage, exc=exc)
    return DailyOverlaySyncResult(
        source_id=source_id,
        trade_date=trade_date,
        previous_cutoff=previous_cutoff,
        verified_cutoff=previous_cutoff,
        status=DailyOverlaySyncStatus.FAILED,
        expected_stock_count=expected_stock_count,
        quarantine_path=quarantine_path,
        reason=values["reason"],
        failure_stage=values["stage"],
        failure_provider=values["provider"],
        failure_status=values["status"],
        reason_code=values["reason_code"],
    )


def _failure_diagnostic(
    *,
    source_id: str,
    stage: DailyOverlayFailureStage,
    exc: Exception,
) -> dict[str, str]:
    status = _failure_status(exc)
    reason_code = _reason_code(stage, status, exc)
    provider = _failure_provider(source_id, stage)
    detail = _safe_failure_detail(stage, exc)
    reason = (
        f"reason_code={reason_code};stage={stage.value};provider={provider};"
        f"status={status};detail={detail}"
    )
    return {
        "reason": reason,
        "stage": stage.value,
        "provider": provider,
        "status": status,
        "reason_code": reason_code,
    }


def _failure_status(exc: Exception) -> str:
    if isinstance(exc, DataQualityError):
        return "quality_rejected"
    if isinstance(exc, DataUnavailableError):
        return "unavailable"
    return "call_failed"


def _refine_failure_stage(
    stage: DailyOverlayFailureStage,
    exc: Exception,
) -> DailyOverlayFailureStage:
    message = str(exc)
    if "独立日线核验" in message:
        return DailyOverlayFailureStage.INDEPENDENT_VERIFICATION
    return stage


def _failure_provider(source_id: str, stage: DailyOverlayFailureStage) -> str:
    if source_id != "zero_budget_eod":
        return source_id if source_id in {"infoway"} else "configured_provider"
    return {
        DailyOverlayFailureStage.CALENDAR: "free_calendar_chain",
        DailyOverlayFailureStage.STOCK_MASTER: "free_stock_master_chain",
        DailyOverlayFailureStage.STOCK_DAILY: "tushare_daily",
        DailyOverlayFailureStage.STOCK_VALIDATION: "tushare_daily",
        DailyOverlayFailureStage.INDEPENDENT_VERIFICATION: "independent_verifier_chain",
        DailyOverlayFailureStage.INDEX_DAILY: "free_index_chain",
        DailyOverlayFailureStage.INDEX_VALIDATION: "free_index_chain",
        DailyOverlayFailureStage.UNIT_VALIDATION: "composite_unit_contract",
        DailyOverlayFailureStage.PERSISTENCE: "local_overlay_store",
        DailyOverlayFailureStage.UNKNOWN: "zero_budget_eod",
    }[stage]


def _reason_code(
    stage: DailyOverlayFailureStage,
    status: str,
    exc: Exception,
) -> str:
    message = " ".join(str(exc).split())
    consensus = _STOCK_MASTER_CONSENSUS_FAILURE.fullmatch(message)
    if stage is DailyOverlayFailureStage.STOCK_MASTER and consensus:
        return f"stock_master_{consensus.group('reason')}"
    lowered = message.lower()
    if status == "quality_rejected" and any(
        marker in lowered
        for marker in (
            "单位合同",
            "成交额字段",
            "amount_volume",
            "amount/volume",
            "成交额与成交量",
        )
    ):
        return f"{stage.value}_provider_unit_contract_changed"
    if stage is DailyOverlayFailureStage.STOCK_VALIDATION:
        if _SAFE_COVERAGE.fullmatch(message):
            return "stock_coverage_below_threshold"
        if _STOCK_UNEXPECTED_COUNT.fullmatch(message):
            return "stock_identity_unrequested"
        if "duplicate" in lowered or "重复" in message:
            return "stock_identity_duplicate"
    if stage is DailyOverlayFailureStage.INDEX_VALIDATION:
        if _INDEX_MISSING_COUNT.fullmatch(message):
            return "core_index_incomplete"
        if _INDEX_UNEXPECTED_COUNT.fullmatch(message):
            return "core_index_unrequested"
    if stage is DailyOverlayFailureStage.UNIT_VALIDATION:
        return "unit_contract_mismatch"
    if stage is DailyOverlayFailureStage.INDEPENDENT_VERIFICATION:
        return (
            "independent_verification_mismatch"
            if status == "quality_rejected"
            else "independent_verification_unavailable"
        )
    suffix = {
        "quality_rejected": "quality_rejected",
        "unavailable": "unavailable",
        "call_failed": "call_failed",
    }[status]
    return f"{stage.value}_{suffix}"


def _safe_failure_detail(stage: DailyOverlayFailureStage, exc: Exception) -> str:
    """Return only repository-owned or explicitly pre-sanitized diagnostic text."""

    message = " ".join(str(exc).split())
    if stage is DailyOverlayFailureStage.STOCK_MASTER and _STOCK_MASTER_CONSENSUS_FAILURE.fullmatch(
        message
    ):
        return message
    coverage = _SAFE_COVERAGE.fullmatch(message)
    if coverage:
        return f"stock coverage {coverage.group('actual')} is below {coverage.group('required')}"
    if _PRE_SANITIZED_DEPENDENCY.fullmatch(message):
        return message[:240]
    lowered = message.lower()
    stock_unexpected = _STOCK_UNEXPECTED_COUNT.fullmatch(message)
    if stage is DailyOverlayFailureStage.STOCK_VALIDATION:
        if stock_unexpected:
            return f"stock increment rejected unexpected symbol count={stock_unexpected.group('count')}"
        if "duplicate" in lowered or "重复" in message:
            return "stock increment identity contains duplicates"
    index_missing = _INDEX_MISSING_COUNT.fullmatch(message)
    index_unexpected = _INDEX_UNEXPECTED_COUNT.fullmatch(message)
    if stage is DailyOverlayFailureStage.INDEX_VALIDATION:
        if index_missing:
            return f"core index increment missing symbol count={index_missing.group('count')}"
        if index_unexpected:
            return f"core index increment rejected unexpected symbol count={index_unexpected.group('count')}"
        if "duplicate" in lowered or "重复" in message:
            return "core index increment identity contains duplicates"
    if stage is DailyOverlayFailureStage.UNIT_VALIDATION and "unit audit metadata" in lowered:
        return message[:240]
    labels = {
        DailyOverlayFailureStage.CALENDAR: "交易日历暂未取得；保留原有已验证数据并停止更新。",
        DailyOverlayFailureStage.STOCK_MASTER: "证券主表暂未取得；保留原有已验证数据并停止更新。",
        DailyOverlayFailureStage.STOCK_DAILY: "股票日线暂未取得；该交易日未登记。",
        DailyOverlayFailureStage.STOCK_VALIDATION: "股票日线未通过完整性校验；该交易日未登记。",
        DailyOverlayFailureStage.INDEPENDENT_VERIFICATION: "独立日线核验未通过；该交易日未登记。",
        DailyOverlayFailureStage.INDEX_DAILY: "核心指数日线暂未取得；该交易日未登记。",
        DailyOverlayFailureStage.INDEX_VALIDATION: "核心指数日线未通过完整性校验；该交易日未登记。",
        DailyOverlayFailureStage.UNIT_VALIDATION: "行情单位合同未通过一致性校验；该交易日未登记。",
        DailyOverlayFailureStage.PERSISTENCE: "本地增量写入或核验提交失败；原有已验证数据保持不变。",
        DailyOverlayFailureStage.UNKNOWN: "每日数据同步未完成；原有已验证数据保持不变。",
    }
    return labels[stage]


def _safe_reason(exc: Exception) -> str:
    """Compatibility helper for callers outside this module's staged path."""

    return _failure_diagnostic(
        source_id="unknown",
        stage=DailyOverlayFailureStage.UNKNOWN,
        exc=exc,
    )["reason"]
