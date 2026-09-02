"""Orchestrate the free, independently verified A-share EOD overlay.

The immutable CSMAR database remains read-only.  New sessions are written to
their own ``zero_budget_eod`` chain, so legacy Infoway rows are retained for
audit but can never be mixed into a new research cutoff.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ashare_lab.adapters.akshare_eod_verifier import AKShareEodVerifier
from ashare_lab.adapters.baostock_eod import (
    BAOSTOCK_CORE_INDEX_SYMBOLS,
    BaoStockEodMarketData,
)
from ashare_lab.adapters.macos_keychain import load_tushare_token
from ashare_lab.adapters.market_overlay_store import MarketOverlayStore
from ashare_lab.adapters.tushare_daily import TushareDailyClient
from ashare_lab.adapters.zero_budget_eod import (
    ZERO_BUDGET_EOD_PROVIDER,
    ZERO_BUDGET_UNIT_CONTRACT_VERSION,
    ZERO_BUDGET_UNIT_RESOLUTION_METHOD_VERSION,
    ZeroBudgetEodMarketData,
)
from ashare_lab.bootstrap import application_data_dir
from ashare_lab.domain.data_sources import DataAction, RightsPolicy, SourceId, SourceRegistry
from ashare_lab.domain.errors import DataUnavailableError
from ashare_lab.services.run_daily_update import (
    DailyUpdateReport,
    QuarantinedDailyUpdate,
    latest_complete_cn_candidate,
    read_csmar_baseline_cutoff,
)
from ashare_lab.services.sync_daily_overlay import (
    DailyOverlaySyncStatus,
    sync_daily_overlay_range,
)


def run_zero_budget_daily_update(
    *,
    csmar_root: str | Path | None = None,
    overlay_root: str | Path | None = None,
    now: datetime | None = None,
    core_index_symbols: tuple[str, ...] = BAOSTOCK_CORE_INDEX_SYMBOLS,
    required_stock_coverage_ratio: float = 0.98,
    _token_loader: Callable[[], str | None] = load_tushare_token,
    _baostock_factory: Callable[..., Any] = BaoStockEodMarketData,
    _tushare_factory: Callable[..., Any] = TushareDailyClient,
    _verifier_factory: Callable[..., Any] = AKShareEodVerifier,
    _provider_factory: Callable[..., Any] = ZeroBudgetEodMarketData,
    _rights_policy: RightsPolicy | None = None,
) -> DailyUpdateReport:
    """Fill every missing completed session through the three-source gate."""

    resolved_now = now or datetime.now(UTC)
    requested_date = latest_complete_cn_candidate(resolved_now)
    data_root = Path(csmar_root or application_data_dir() / "cache" / "csmar")
    increment_root = Path(
        overlay_root or application_data_dir() / "cache" / "market_overlay"
    )
    baseline_cutoff = read_csmar_baseline_cutoff(data_root, through_date=requested_date)
    token = _token_loader()
    if not token or not str(token).strip():
        raise DataUnavailableError(
            "尚未在macOS钥匙串保存Tushare Token。请在自动数据更新页面保存；"
            "不要把Token发送到聊天、日志或GitHub。"
        )
    if not core_index_symbols:
        raise ValueError("core_index_symbols cannot be empty")

    rights_policy = _rights_policy or RightsPolicy(SourceRegistry.load_default())
    for source_id, actions in (
        (SourceId.TUSHARE, (DataAction.MARKET_DATA_READ, DataAction.MARKET_DATA_CACHE)),
        (
            SourceId.BAOSTOCK,
            (
                DataAction.MARKET_DATA_READ,
                DataAction.MARKET_DATA_CACHE,
                DataAction.METADATA_READ,
            ),
        ),
        (SourceId.AKSHARE, (DataAction.MARKET_DATA_READ,)),
        (
            SourceId.ZERO_BUDGET_EOD,
            (DataAction.MARKET_DATA_READ, DataAction.MARKET_DATA_CACHE),
        ),
    ):
        for action in actions:
            rights_policy.require(source_id, action)

    store = MarketOverlayStore(increment_root)
    components: list[object] = []
    try:
        baostock = _safe_construct(
            "BaoStock组件",
            lambda: _baostock_factory(clock=lambda: resolved_now.astimezone(UTC)),
        )
        components.append(baostock)
        tushare = _safe_construct(
            "Tushare组件",
            lambda: _tushare_factory(
                str(token).strip(),
                clock=lambda: resolved_now.astimezone(UTC),
            ),
        )
        components.append(tushare)
        verifier = _safe_construct(
            "AKShare核验组件",
            lambda: _verifier_factory(clock=lambda: resolved_now.astimezone(UTC)),
        )
        components.append(verifier)
        provider = _safe_construct(
            "零预算三源组合",
            lambda: _provider_factory(
                baostock=baostock,
                tushare=tushare,
                verifier=verifier,
            ),
        )
        components.append(provider)
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
        closed: set[int] = set()
        for component in reversed(components):
            if id(component) in closed:
                continue
            closed.add(id(component))
            close = getattr(component, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()

    chain = store.verified_dates_from(
        source_id=ZERO_BUDGET_EOD_PROVIDER,
        baseline_cutoff=baseline_cutoff,
        through_date=requested_date,
    )
    automatic_cutoff = chain[-1] if chain else None
    common_cutoff = automatic_cutoff or baseline_cutoff
    latest_session = (
        range_report.expected_sessions[-1]
        if range_report.expected_sessions
        else common_cutoff
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
    return DailyUpdateReport(
        source_id=ZERO_BUDGET_EOD_PROVIDER,
        historical_baseline_cutoff=baseline_cutoff,
        requested_complete_date=requested_date,
        latest_complete_session=latest_session,
        automatic_increment_cutoff=automatic_cutoff,
        common_cutoff=common_cutoff,
        updated_sessions=updated,
        unchanged_sessions=unchanged,
        quarantined_failures=failures,
        provider_contract_changed=any(
            _looks_like_contract_change(item.reason) for item in failures
        ),
        current_through_latest_complete_session=(common_cutoff == latest_session),
        unit_contract_version=ZERO_BUDGET_UNIT_CONTRACT_VERSION,
        unit_resolution_method_version=ZERO_BUDGET_UNIT_RESOLUTION_METHOD_VERSION,
        market_scope=(
            "沪深A股：Tushare免费未复权日线；BaoStock交易日历、证券清单及六核心指数；"
            "AKShare确定性抽样交叉核验；不含北交所"
        ),
        csmar_mutated=False,
        range_report=range_report,
    )


def _looks_like_contract_change(reason: str) -> bool:
    normalized = str(reason).lower()
    return any(
        marker in normalized
        for marker in (
            "dataqualityerror",
            "质量校验失败",
            "单位",
            "unit",
            "volume",
            "amount",
            "成交量",
            "成交额",
            "字段",
            "schema",
        )
    )


def _safe_construct(label: str, factory: Callable[[], Any]) -> Any:
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001 - credential-bearing construction boundary
        raise DataUnavailableError(
            f"{label}初始化失败（{type(exc).__name__}），原始错误已脱敏。"
        ) from None


__all__ = ["run_zero_budget_daily_update"]
