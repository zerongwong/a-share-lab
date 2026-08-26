"""Date-major ingestion of licensed full-market A-share daily data."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from math import ceil
from typing import Any

import pandas as pd

from ashare_lab.adapters.parquet_market_store import (
    ParquetMarketStore,
    StoredCrossSectionSummary,
    normalize_daily_cross_section,
    normalize_security_master,
)
from ashare_lab.domain.data_sources import DataAction, RightsPolicy, SourceId
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.bulk_market_data import BulkMarketDataPort
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS, normalize_symbol


class TradeDateSyncStatus(StrEnum):
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TradeDateSyncResult:
    trade_date: date
    status: TradeDateSyncStatus
    fetched_symbols: int = 0
    stored_symbols: int = 0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class FullMarketSyncReport:
    source_id: str
    as_of: date
    adjustment: str
    bootstrap_start: date
    started_at: datetime
    completed_at: datetime
    master_rows: int
    active_symbols: int
    current_session_expected_symbols: int
    portfolio_eligible_symbols: int
    mature_eligible_symbols: int
    calendar_sessions: int
    attempted_sessions: int
    updated_sessions: int
    unchanged_sessions: int
    failed_sessions: int
    stored_sessions: int
    session_coverage_ratio: float
    locally_cached_symbols: int
    current_symbols: int
    strategy_ready_symbols: int
    current_symbol_coverage_ratio: float
    strategy_coverage_ratio: float
    required_coverage_ratio: float
    market_cutoff: date | None
    insufficient_history_symbols: int
    minimum_sessions: int
    ready_for_full_market_screen: bool
    warnings: tuple[str, ...]
    results: tuple[TradeDateSyncResult, ...]
    report_json_path: str | None = None
    report_symbol_detail_path: str | None = None
    report_date_detail_path: str | None = None

    def summary_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("results", None)
        return payload


def sync_full_market_daily(
    provider: BulkMarketDataPort,
    store: ParquetMarketStore,
    rights_policy: RightsPolicy,
    *,
    as_of: date,
    bootstrap_start: date,
    adjustment: str = "none",
    overlap_sessions: int = 5,
    minimum_sessions: int = 252,
    required_coverage_ratio: float = 0.98,
    clock: Callable[[], datetime] | None = None,
) -> FullMarketSyncReport:
    """Synchronize all A-shares with one provider call per open session.

    The first run loops the provider's trading calendar.  Later runs fetch only
    missing sessions plus a small overlap window for vendor corrections.  The
    UI never invokes this function; it reads the completed Parquet dataset and
    the report only after the quality gate passes.
    """

    _validate_options(
        as_of=as_of,
        bootstrap_start=bootstrap_start,
        overlap_sessions=overlap_sessions,
        minimum_sessions=minimum_sessions,
        required_coverage_ratio=required_coverage_ratio,
    )
    now = clock or (lambda: datetime.now(UTC))
    started_at = _aware_utc(now(), "started_at")
    source_id = SourceId(provider.source_id)

    # Access and local caching are separate contractual rights.  Both are
    # checked before a provider call or filesystem mutation.
    rights_policy.require(source_id, DataAction.MARKET_DATA_READ)
    rights_policy.require(source_id, DataAction.MARKET_DATA_CACHE)

    raw_master = provider.fetch_security_master(as_of)
    master = normalize_security_master(raw_master, source_id=source_id, as_of=as_of)
    store.write_security_master(master, source_id=source_id, as_of=as_of)
    active = _active_master(master, as_of)
    if active.empty:
        raise DataUnavailableError(f"{source_id.value} 证券主表没有当前上市A股")

    calendar = _normalize_calendar(
        provider.fetch_trade_calendar(bootstrap_start, as_of),
        start=bootstrap_start,
        end=as_of,
    )
    if not calendar:
        raise DataUnavailableError("供应商交易日历在请求区间内为空")

    manifest_before = _cross_section_manifest_lookup(
        store, source_id=source_id, adjustment=adjustment
    )
    stored_before = set(manifest_before)
    future_dates = sorted(value for value in stored_before if value > as_of)
    if future_dates:
        raise DataQualityError(
            "本地缓存晚于分析截止日，拒绝用于历史时点以免前视偏差："
            + ", ".join(value.isoformat() for value in future_dates[:5])
        )

    missing = set(calendar) - stored_before
    overlap = set(calendar[-overlap_sessions:]) if overlap_sessions else set()
    requested_dates = sorted(missing | overlap)
    known_symbols = set(master["symbol"].astype(str))
    results: dict[date, TradeDateSyncResult] = {}
    summaries: list[StoredCrossSectionSummary] = []

    for trade_date in requested_dates:
        try:
            raw_daily = provider.fetch_daily_for_trade_date(
                trade_date,
                adjust=adjustment,
            )
            canonical = _validate_market_cross_section(
                raw_daily,
                source_id=source_id,
                trade_date=trade_date,
                known_symbols=known_symbols,
            )
            summary = store.write_daily_cross_section(
                trade_date,
                canonical,
                source_id=source_id,
                adjustment=adjustment,
            )
            summaries.append(summary)
            previous = manifest_before.get(trade_date)
            status = (
                TradeDateSyncStatus.UNCHANGED
                if previous is not None and str(previous["checksum"]) == summary.checksum
                else TradeDateSyncStatus.UPDATED
            )
            results[trade_date] = TradeDateSyncResult(
                trade_date=trade_date,
                status=status,
                fetched_symbols=int(canonical["symbol"].nunique()),
                stored_symbols=summary.symbol_count,
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider boundary per date
            previous = manifest_before.get(trade_date)
            results[trade_date] = TradeDateSyncResult(
                trade_date=trade_date,
                status=TradeDateSyncStatus.FAILED,
                stored_symbols=int(previous["symbol_count"]) if previous else 0,
                reason=_safe_reason("trade_date_fetch_failed", exc),
            )

    store.upsert_cross_section_manifest(summaries)
    completed_at = _aware_utc(now(), "completed_at")
    if completed_at < started_at:
        raise ValueError("clock returned completed_at before started_at")

    report, symbol_details, date_details = _build_quality_report(
        store=store,
        master=master,
        active=active,
        calendar=calendar,
        results=results,
        source_id=source_id,
        adjustment=adjustment,
        as_of=as_of,
        bootstrap_start=bootstrap_start,
        started_at=started_at,
        completed_at=completed_at,
        minimum_sessions=minimum_sessions,
        required_coverage_ratio=required_coverage_ratio,
    )
    json_path, symbol_path, date_path = store.write_sync_report(
        report.summary_dict(),
        symbol_details,
        date_details,
        source_id=source_id,
        completed_at=completed_at,
    )
    return replace(
        report,
        report_json_path=str(json_path),
        report_symbol_detail_path=str(symbol_path),
        report_date_detail_path=str(date_path),
    )


def _validate_market_cross_section(
    frame: pd.DataFrame,
    *,
    source_id: SourceId,
    trade_date: date,
    known_symbols: set[str],
) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise DataUnavailableError(f"{trade_date.isoformat()} 全市场日线为空")
    required = {"symbol", *CANONICAL_DAILY_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise DataQualityError("全市场日线缺少字段：" + ", ".join(sorted(missing)))
    output = frame.loc[:, ["symbol", *CANONICAL_DAILY_COLUMNS]].copy()
    try:
        output["symbol"] = output["symbol"].map(_normalize_symbol_cell)
    except ValueError as exc:
        raise DataQualityError(f"全市场日线包含无效代码：{exc}") from exc
    unexpected = set(output["symbol"]) - known_symbols
    if unexpected:
        raise DataQualityError(
            "供应商返回证券主表之外的代码，已拒绝整日截面：" + ", ".join(sorted(unexpected)[:10])
        )
    labels = output["source"].fillna("").astype(str).str.strip()
    correct_source = labels.map(
        lambda value: value == source_id.value or value.startswith(f"{source_id.value}:")
    )
    if not bool(correct_source.all()):
        raise DataQualityError(f"全市场日线source必须属于{source_id.value}，禁止混入其他供应商")
    return normalize_daily_cross_section(output, expected_date=trade_date)


def _build_quality_report(
    *,
    store: ParquetMarketStore,
    master: pd.DataFrame,
    active: pd.DataFrame,
    calendar: tuple[date, ...],
    results: dict[date, TradeDateSyncResult],
    source_id: SourceId,
    adjustment: str,
    as_of: date,
    bootstrap_start: date,
    started_at: datetime,
    completed_at: datetime,
    minimum_sessions: int,
    required_coverage_ratio: float,
) -> tuple[FullMarketSyncReport, pd.DataFrame, pd.DataFrame]:
    manifest = _cross_section_manifest_lookup(store, source_id=source_id, adjustment=adjustment)
    calendar_set = set(calendar)
    stored_sessions = calendar_set & set(manifest)
    target_cutoff = calendar[-1]
    market_cutoff = target_cutoff if target_cutoff in stored_sessions else None
    session_coverage = len(stored_sessions) / len(calendar)

    coverage = store.build_symbol_coverage(
        source_id=source_id,
        adjustment=adjustment,
        start=bootstrap_start,
        end=as_of,
    )
    coverage_by_symbol = {
        str(row.symbol): row._asdict() for row in coverage.itertuples(index=False)
    }
    current_symbols: set[str] = set()
    if market_cutoff is not None:
        latest = store.read_daily_cross_section(
            market_cutoff,
            source_id=source_id,
            adjustment=adjustment,
        )
        current_symbols = set(latest["symbol"].astype(str))

    active_symbol_set = set(active["symbol"].astype(str))
    cached_active = active_symbol_set & set(coverage_by_symbol)
    current_expected = set(active.loc[~active["is_suspended"], "symbol"].astype(str))
    current_present = current_expected & current_symbols
    eligible_mask = ~(active["is_st"] | active["is_delisting"] | active["is_suspended"])
    eligible = set(active.loc[eligible_mask, "symbol"].astype(str))
    maturity_days = ceil(minimum_sessions * 365 / 252) + 30
    mature_cutoff = pd.Timestamp(as_of - timedelta(days=maturity_days))
    mature = set(
        active.loc[eligible_mask & (active["list_date"] <= mature_cutoff), "symbol"].astype(str)
    )

    symbol_rows: list[dict[str, Any]] = []
    strategy_ready = 0
    insufficient = 0
    for master_row in active.to_dict(orient="records"):
        symbol = str(master_row["symbol"])
        history = coverage_by_symbol.get(symbol)
        row_count = int(history["row_count"]) if history else 0
        first_date = pd.Timestamp(history["first_trade_date"]).date() if history else None
        last_date = pd.Timestamp(history["last_trade_date"]).date() if history else None
        is_mature = symbol in mature
        sufficient = row_count >= minimum_sessions
        is_current = symbol in current_symbols or bool(master_row["is_suspended"])
        is_ready = bool(symbol in eligible and is_mature and sufficient and is_current)
        if is_ready:
            strategy_ready += 1
        if is_mature and not sufficient:
            insufficient += 1
        reasons: list[str] = []
        if history is None:
            reasons.append("missing_history")
        if symbol in current_expected and symbol not in current_symbols:
            reasons.append("missing_latest_cross_section")
        if is_mature and not sufficient:
            reasons.append(f"insufficient_history:{row_count}<{minimum_sessions}")
        if symbol not in eligible:
            reasons.append("portfolio_rule_excluded")
        if symbol in eligible and not is_mature:
            reasons.append("recent_ipo_not_required_for_strategy_coverage")
        symbol_rows.append(
            {
                "symbol": symbol,
                "name": master_row["name"],
                "exchange": master_row["exchange"],
                "board": master_row["board"],
                "industry": master_row["industry"],
                "is_st": bool(master_row["is_st"]),
                "is_delisting": bool(master_row["is_delisting"]),
                "is_suspended": bool(master_row["is_suspended"]),
                "stored_rows": row_count,
                "first_trade_date": first_date,
                "last_trade_date": last_date,
                "portfolio_eligible": symbol in eligible,
                "mature_eligible": is_mature,
                "is_current": is_current,
                "strategy_ready": is_ready,
                "quality_reasons": "|".join(reasons),
            }
        )

    date_rows: list[dict[str, Any]] = []
    for trade_date in calendar:
        sync = results.get(trade_date)
        stored = manifest.get(trade_date)
        date_rows.append(
            {
                "trade_date": trade_date,
                "was_requested": sync is not None,
                "sync_status": sync.status.value if sync else "cached_not_refetched",
                "sync_reason": sync.reason if sync else "",
                "fetched_symbols": sync.fetched_symbols if sync else 0,
                "stored_symbols": int(stored["symbol_count"]) if stored else 0,
                "is_cached": stored is not None,
                "checksum": str(stored["checksum"]) if stored else "",
            }
        )

    current_symbol_coverage = (
        len(current_present) / len(current_expected) if current_expected else 0.0
    )
    strategy_coverage = strategy_ready / len(mature) if mature else 0.0
    updated = sum(item.status is TradeDateSyncStatus.UPDATED for item in results.values())
    unchanged = sum(item.status is TradeDateSyncStatus.UNCHANGED for item in results.values())
    failed = sum(item.status is TradeDateSyncStatus.FAILED for item in results.values())
    warnings: list[str] = []
    if failed:
        warnings.append(f"本次有{failed}个交易日同步失败；详见日期质量报告")
    if session_coverage < required_coverage_ratio:
        warnings.append(
            f"历史交易日覆盖率{session_coverage:.2%}低于门槛{required_coverage_ratio:.2%}"
        )
    if current_symbol_coverage < required_coverage_ratio:
        warnings.append(
            f"最新截面股票覆盖率{current_symbol_coverage:.2%}低于门槛{required_coverage_ratio:.2%}"
        )
    if strategy_coverage < required_coverage_ratio:
        warnings.append(
            f"策略历史覆盖率{strategy_coverage:.2%}低于门槛{required_coverage_ratio:.2%}"
        )
    ready = bool(
        market_cutoff is not None
        and session_coverage >= required_coverage_ratio
        and current_symbol_coverage >= required_coverage_ratio
        and strategy_coverage >= required_coverage_ratio
    )
    if not ready:
        warnings.append("未达到全市场筛选门槛，UI不得标称已扫描全部A股")

    report = FullMarketSyncReport(
        source_id=source_id.value,
        as_of=as_of,
        adjustment=adjustment,
        bootstrap_start=bootstrap_start,
        started_at=started_at,
        completed_at=completed_at,
        master_rows=len(master),
        active_symbols=len(active),
        current_session_expected_symbols=len(current_expected),
        portfolio_eligible_symbols=len(eligible),
        mature_eligible_symbols=len(mature),
        calendar_sessions=len(calendar),
        attempted_sessions=len(results),
        updated_sessions=updated,
        unchanged_sessions=unchanged,
        failed_sessions=failed,
        stored_sessions=len(stored_sessions),
        session_coverage_ratio=session_coverage,
        locally_cached_symbols=len(cached_active),
        current_symbols=len(current_present),
        strategy_ready_symbols=strategy_ready,
        current_symbol_coverage_ratio=current_symbol_coverage,
        strategy_coverage_ratio=strategy_coverage,
        required_coverage_ratio=required_coverage_ratio,
        market_cutoff=market_cutoff,
        insufficient_history_symbols=insufficient,
        minimum_sessions=minimum_sessions,
        ready_for_full_market_screen=ready,
        warnings=tuple(warnings),
        results=tuple(results[value] for value in sorted(results)),
    )
    return (
        report,
        pd.DataFrame(symbol_rows).sort_values("symbol").reset_index(drop=True),
        pd.DataFrame(date_rows).sort_values("trade_date").reset_index(drop=True),
    )


def _active_master(master: pd.DataFrame, as_of: date) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of)
    return (
        master.loc[
            (master["list_date"] <= cutoff)
            & (master["delist_date"].isna() | (master["delist_date"] > cutoff))
        ]
        .sort_values("symbol")
        .reset_index(drop=True)
    )


def _normalize_calendar(
    values: Sequence[date],
    *,
    start: date,
    end: date,
) -> tuple[date, ...]:
    normalized: list[date] = []
    for value in values:
        parsed = pd.Timestamp(value)
        if pd.isna(parsed):
            raise DataQualityError("交易日历包含无效日期")
        if parsed.tzinfo is not None:
            parsed = parsed.tz_localize(None)
        trade_date = parsed.date()
        if not start <= trade_date <= end:
            raise DataQualityError("交易日历返回了请求范围外日期")
        normalized.append(trade_date)
    if len(set(normalized)) != len(normalized):
        raise DataQualityError("交易日历包含重复日期")
    return tuple(sorted(normalized))


def _cross_section_manifest_lookup(
    store: ParquetMarketStore,
    *,
    source_id: SourceId,
    adjustment: str,
) -> dict[date, dict[str, Any]]:
    manifest = store.read_cross_section_manifest()
    selected = manifest.loc[
        (manifest["source_id"] == source_id.value) & (manifest["adjustment"] == adjustment)
    ]
    return {
        pd.Timestamp(row.trade_date).date(): row._asdict()
        for row in selected.itertuples(index=False)
    }


def _normalize_symbol_cell(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return normalize_symbol(text.zfill(6))


def _safe_reason(prefix: str, error: Exception) -> str:
    # Provider exceptions can accidentally echo credentials.  Keep only a
    # short scrubbed message and never persist the original exception object.
    message = str(error).replace("\n", " ").replace("\r", " ")[:160]
    for marker in ("token", "password", "secret", "api_key", "access_token"):
        if marker in message.lower():
            message = "provider_error_redacted"
            break
    return f"{prefix}:{type(error).__name__}:{message}"


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _validate_options(
    *,
    as_of: date,
    bootstrap_start: date,
    overlap_sessions: int,
    minimum_sessions: int,
    required_coverage_ratio: float,
) -> None:
    if bootstrap_start > as_of:
        raise ValueError("bootstrap_start不能晚于as_of")
    if not 0 <= overlap_sessions <= 30:
        raise ValueError("overlap_sessions必须在0到30之间")
    if minimum_sessions < 1:
        raise ValueError("minimum_sessions必须为正整数")
    if not 0.0 < required_coverage_ratio <= 1.0:
        raise ValueError("required_coverage_ratio必须在0到1之间")
