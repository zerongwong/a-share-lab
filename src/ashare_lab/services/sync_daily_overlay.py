"""Synchronize completed daily increments into a verified market overlay."""

from __future__ import annotations

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
    stocks_requested = tuple(stock_symbols or provider.fetch_cn_stock_symbols())
    if not stocks_requested:
        raise DataUnavailableError("provider stock universe is empty")
    core_requested = tuple(core_index_symbols)
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
    receipt: dict[str, Any] = {
        "provider": source_id,
        "target_date": target_date,
        "previous_trade_date": previous_trade_date,
        "adjustment": "none",
    }
    try:
        stock_batch = provider.fetch_daily_increment(
            stocks_requested,
            target_date,
            cutoff_timestamp=cutoff_timestamp,
            asset_kind="stocks",
        )
        store.stage_asset(run, "stocks", stock_batch.frame)
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
                "stock increment returned unrequested symbols: "
                + ", ".join(sorted(unexpected_stocks)[:10])
            )
        stock_coverage = len(set(stocks["symbol"])) / len(stock_expected)
        if stock_coverage + 1e-12 < required_stock_coverage_ratio:
            raise DataQualityError(
                f"stock coverage {stock_coverage:.4f} is below {required_stock_coverage_ratio:.4f}"
            )
        receipt["stocks"] = _batch_receipt(stock_batch, asset_kind="stocks")
        store.update_staging_receipt(run, receipt)

        index_batch = provider.fetch_daily_increment(
            core_requested,
            target_date,
            cutoff_timestamp=cutoff_timestamp,
            asset_kind="indices",
        )
        store.stage_asset(run, "indices", index_batch.frame)
        indices = _validate_batch(
            index_batch,
            requested_symbols=core_requested,
            target_date=target_date,
            source_id=source_id,
            asset_kind="indices",
        )
        core_normalized = _normalize_configured_symbols(core_requested)
        actual_indices = set(indices["symbol"])
        missing_indices = set(core_normalized) - actual_indices
        unexpected_indices = actual_indices - set(core_normalized)
        if missing_indices:
            raise DataQualityError(
                "core index increment is incomplete: " + ", ".join(sorted(missing_indices))
            )
        if unexpected_indices:
            raise DataQualityError(
                "core index increment returned unrequested symbols: "
                + ", ".join(sorted(unexpected_indices))
            )
        receipt["indices"] = _batch_receipt(index_batch, asset_kind="indices")
        store.update_staging_receipt(run, receipt)

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
        quarantine_path: str | None = None
        if run.path.is_dir():
            quarantine_path = str(
                store.quarantine(
                    run,
                    reason=_safe_reason(exc),
                    failed_at=_aware_utc(now(), "clock"),
                )
            )
        return DailyOverlaySyncResult(
            source_id=source_id,
            trade_date=target_date,
            previous_cutoff=previous_trade_date,
            verified_cutoff=previous_trade_date,
            status=DailyOverlaySyncStatus.FAILED,
            expected_stock_count=len(stocks_requested),
            quarantine_path=quarantine_path,
            reason=_safe_reason(exc),
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
    calendar = _normalize_calendar(
        provider.fetch_cn_trading_days(started_cutoff + timedelta(days=1), through_date),
        start=started_cutoff + timedelta(days=1),
        end=through_date,
    )
    stocks_requested = tuple(stock_symbols or provider.fetch_cn_stock_symbols())
    if not stocks_requested:
        raise DataUnavailableError("provider stock universe is empty")
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
    return {
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


def _safe_reason(exc: Exception) -> str:
    message = " ".join(str(exc).split())[:1200]
    return f"{type(exc).__name__}: {message or 'overlay verification failed'}"
