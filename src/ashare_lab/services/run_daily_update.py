"""User-facing orchestration for the read-only CSMAR daily EOD update.

Only a macOS Keychain loader can provide the Infoway credential.  The local
CSMAR DuckDB is opened read-only and is never passed to an ingestion service;
all newer rows are written to the provider-isolated market overlay.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from ashare_lab.adapters.infoway_eod import (
    InfowayEodMarketData,
    InfowayEodUnitContract,
)
from ashare_lab.adapters.macos_keychain import load_infoway_api_key
from ashare_lab.adapters.market_overlay_store import MarketOverlayStore
from ashare_lab.bootstrap import application_data_dir
from ashare_lab.domain.data_sources import RightsPolicy, SourceRegistry
from ashare_lab.domain.errors import DataUnavailableError
from ashare_lab.services.sync_daily_overlay import (
    DailyOverlayRangeReport,
    DailyOverlaySyncStatus,
    sync_daily_overlay_range,
)

CN_TIMEZONE = ZoneInfo("Asia/Shanghai")
SAFE_EOD_READY_TIME = time(15, 30)

# Versioned, explicit contract requested by the application owner.  Infoway's
# public China A-share overview and batch-kline example identify ``v`` as trade
# volume and ``vw`` as turnover value.  A private, read-only account calibration
# confirmed that historical responses use 100-share lots for ``v`` and CNY for
# ``vw``.  No provider row or reversible raw value is retained in this source
# repository.  The
# adapter independently enforces
# low * volume <= amount <= high * volume for every row.  A response that uses
# only ``vm``, contains both fields, or violates the invariant is rejected and
# quarantined; the code never tries another field or guesses a multiplier.
INFOWAY_EOD_UNIT_CONTRACT_VERSION = "infoway-cn-eod-v3-2026-08-27"
INFOWAY_EOD_UNIT_RESOLUTION_METHOD_VERSION = "batch-price-band-v1"
INFOWAY_EOD_UNIT_REFERENCE = (
    "Infoway official China A-Shares Data API overview and POST batch candlestick example "
    "(https://docs.infoway.io/en-docs/readme/china-a-shares-data-api; "
    "https://docs.infoway.io/en-docs/rest-api/market-data/post-candlestick), "
    "plus a private read-only account calibration whose raw rows are deliberately not "
    "stored in the repository: historical v is 100-share lots and historical vw is CNY; "
    "same-session calibration found that provisional "
    "vw requires x100 while v remains 100-share lots; contract version "
    "infoway-cn-eod-v3-2026-08-27; method batch-price-band-v1; runtime invariant "
    "low*volume_shares<=amount_cny<=high*volume_shares"
)
INFOWAY_EOD_UNIT_CONTRACT = InfowayEodUnitContract(
    volume_multiplier_to_shares=Decimal("100"),
    amount_field="vw",
    amount_multiplier_to_cny=Decimal("1"),
    provisional_amount_multiplier_to_cny=Decimal("100"),
    contract_version=INFOWAY_EOD_UNIT_CONTRACT_VERSION,
    resolution_method_version=INFOWAY_EOD_UNIT_RESOLUTION_METHOD_VERSION,
    verified_reference=INFOWAY_EOD_UNIT_REFERENCE,
)

DEFAULT_INFOWAY_CORE_INDICES = (
    "000001.SH",
    "000300.SH",
    "000852.SH",
    "399905.SZ",
    "399001.SZ",
    "399006.SZ",
)


@dataclass(frozen=True, slots=True)
class QuarantinedDailyUpdate:
    trade_date: date
    reason: str
    path: str | None


@dataclass(frozen=True, slots=True)
class DailyUpdateReport:
    source_id: str
    historical_baseline_cutoff: date
    requested_complete_date: date
    latest_complete_session: date
    automatic_increment_cutoff: date | None
    common_cutoff: date
    updated_sessions: tuple[date, ...]
    unchanged_sessions: tuple[date, ...]
    quarantined_failures: tuple[QuarantinedDailyUpdate, ...]
    provider_contract_changed: bool
    current_through_latest_complete_session: bool
    unit_contract_version: str
    unit_resolution_method_version: str
    market_scope: str
    csmar_mutated: bool
    range_report: DailyOverlayRangeReport


def latest_complete_cn_candidate(now: datetime) -> date:
    """Return the latest calendar date that may contain a finalized CN close.

    Before 15:30 Asia/Shanghai, today's accumulating quote is never treated as
    a completed-bar candidate.  At and after 15:30 the existing trading-calendar,
    completeness and row-quality gates still decide whether the date can advance;
    an incomplete provider snapshot is quarantined rather than accepted.  Weekends
    and statutory holidays are resolved later by the provider's official CN trading
    calendar, rather than guessed here.
    """

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(CN_TIMEZONE)
    if local.time().replace(tzinfo=None) < SAFE_EOD_READY_TIME:
        return local.date() - timedelta(days=1)
    return local.date()


def read_csmar_baseline_cutoff(
    dataset_root: str | Path,
    *,
    through_date: date,
) -> date:
    """Read the latest eligible CSMAR session through a read-only connection."""

    database_path = Path(dataset_root).expanduser().resolve() / "csmar.duckdb"
    if not database_path.is_file():
        raise DataUnavailableError(f"CSMAR本地数据库不存在：{database_path}")
    try:
        import duckdb
    except ImportError as exc:
        raise DataUnavailableError("缺少duckdb依赖，无法读取CSMAR基线截止日") from exc
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        cutoff = connection.execute(
            "SELECT max(trade_date) FROM daily_bars WHERE trade_date <= ?",
            [through_date],
        ).fetchone()[0]
    finally:
        connection.close()
    if cutoff is None:
        raise DataUnavailableError(f"CSMAR在{through_date.isoformat()}及以前没有可用日线基线")
    return pd.Timestamp(cutoff).date()


def run_daily_update(
    *,
    csmar_root: str | Path | None = None,
    overlay_root: str | Path | None = None,
    now: datetime | None = None,
    core_index_symbols: tuple[str, ...] = DEFAULT_INFOWAY_CORE_INDICES,
    required_stock_coverage_ratio: float = 0.98,
    _api_key_loader: Callable[[], str | None] = load_infoway_api_key,
    _provider_factory: Callable[..., Any] = InfowayEodMarketData,
    _rights_policy: RightsPolicy | None = None,
) -> DailyUpdateReport:
    """Update every missing completed session without modifying CSMAR.

    ``_api_key_loader`` and ``_provider_factory`` are dependency seams for
    offline tests.  Production callers do not accept a secret argument: the
    credential can enter only through ``load_infoway_api_key``.
    """

    resolved_now = now or datetime.now(UTC)
    requested_date = latest_complete_cn_candidate(resolved_now)
    data_root = Path(csmar_root or application_data_dir() / "cache" / "csmar")
    increment_root = Path(overlay_root or application_data_dir() / "cache" / "market_overlay")
    baseline_cutoff = read_csmar_baseline_cutoff(
        data_root,
        through_date=requested_date,
    )
    api_key = _api_key_loader()
    if not api_key or not str(api_key).strip():
        raise DataUnavailableError(
            "尚未在macOS钥匙串保存Infoway API密钥。请先轮换曾在聊天或截图中出现的旧密钥，"
            "再在自动数据更新页面保存新密钥。"
        )
    if not core_index_symbols:
        raise ValueError("core_index_symbols cannot be empty")
    rights_policy = _rights_policy or RightsPolicy(SourceRegistry.load_default())
    store = MarketOverlayStore(increment_root)
    provider = _provider_factory(
        str(api_key).strip(),
        unit_contract=INFOWAY_EOD_UNIT_CONTRACT,
    )
    try:
        range_report = sync_daily_overlay_range(
            provider,
            store,
            baseline_cutoff=baseline_cutoff,
            through_date=requested_date,
            core_index_symbols=core_index_symbols,
            required_stock_coverage_ratio=required_stock_coverage_ratio,
            rights_policy=rights_policy,
            clock=lambda: resolved_now.astimezone(UTC),
        )
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    chain = store.verified_dates_from(
        source_id="infoway",
        baseline_cutoff=baseline_cutoff,
        through_date=requested_date,
    )
    automatic_cutoff = chain[-1] if chain else None
    common_cutoff = automatic_cutoff or baseline_cutoff
    latest_session = (
        range_report.expected_sessions[-1] if range_report.expected_sessions else common_cutoff
    )
    updated = tuple(
        item.trade_date
        for item in range_report.results
        if item.status is DailyOverlaySyncStatus.VERIFIED
    )
    unchanged = tuple(
        item.trade_date
        for item in range_report.results
        if item.status is DailyOverlaySyncStatus.UNCHANGED
    )
    failures = tuple(
        QuarantinedDailyUpdate(
            trade_date=item.trade_date,
            reason=item.reason,
            path=item.quarantine_path,
        )
        for item in range_report.results
        if item.status is DailyOverlaySyncStatus.FAILED
    )
    contract_changed = any(_looks_like_unit_contract_change(item.reason) for item in failures)
    return DailyUpdateReport(
        source_id="infoway",
        historical_baseline_cutoff=baseline_cutoff,
        requested_complete_date=requested_date,
        latest_complete_session=latest_session,
        automatic_increment_cutoff=automatic_cutoff,
        common_cutoff=common_cutoff,
        updated_sessions=updated,
        unchanged_sessions=unchanged,
        quarantined_failures=failures,
        provider_contract_changed=contract_changed,
        current_through_latest_complete_session=(common_cutoff == latest_session),
        unit_contract_version=INFOWAY_EOD_UNIT_CONTRACT_VERSION,
        unit_resolution_method_version=INFOWAY_EOD_UNIT_RESOLUTION_METHOD_VERSION,
        market_scope="沪深A股及配置的沪深核心指数；Infoway当前清单不含北交所",
        csmar_mutated=False,
        range_report=range_report,
    )


def _looks_like_unit_contract_change(reason: str) -> bool:
    normalized = str(reason).lower()
    markers = (
        "单位合同",
        "成交额字段",
        "vw",
        "vm",
        "amount_volume",
        "amount/volume",
        "成交额与成交量",
    )
    return any(marker in normalized for marker in markers)
