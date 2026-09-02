"""Fail-closed composition of free A-share end-of-day components.

BaoStock is authoritative only for the China calendar, listed SSE/SZSE stock
universe and the six core-index bars.  Tushare supplies the stock cross
section, while AKShare independently verifies that cross section.  AKShare is
never a repair or fallback source: any result other than ``VERIFIED`` fails the
entire stock batch.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from ashare_lab.adapters.akshare_eod_verifier import AKShareVerificationStatus
from ashare_lab.adapters.baostock_eod import BAOSTOCK_CORE_INDEX_SYMBOLS
from ashare_lab.adapters.market_overlay_store import normalize_overlay_daily
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.daily_increment import AssetKind, DailyIncrementBatch

ZERO_BUDGET_EOD_PROVIDER = "zero_budget_eod"
ZERO_BUDGET_STOCK_SOURCE = "zero_budget_eod:tushare:daily_unadjusted:stocks"
ZERO_BUDGET_INDEX_SOURCE = "zero_budget_eod:baostock:eod_unadjusted:indices"
ZERO_BUDGET_UNIT_CONTRACT_VERSION = (
    "zero-budget-eod-tushare-stocks-baostock-indices-v1"
)
ZERO_BUDGET_UNIT_RESOLUTION_METHOD_VERSION = (
    "static-units-plus-akshare-independent-verification-v1"
)
ZERO_BUDGET_AMOUNT_MULTIPLIER_TO_CNY = "tushare=1000;baostock=1"

_CN_TIMEZONE = ZoneInfo("Asia/Shanghai")
_CN_CLOSE = time(15, 0)
_STOCK_SYMBOL = re.compile(r"^(?P<code>\d{6})\.(?P<exchange>SH|SZ)$")
_TRACE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class ZeroBudgetEodMarketData:
    """Compose BaoStock, Tushare and AKShare into ``DailyIncrementPort``.

    All three components are injected explicitly.  This class never receives,
    stores or formats a Tushare token, which keeps secret acquisition outside
    the daily-increment boundary.
    """

    provider = ZERO_BUDGET_EOD_PROVIDER
    source_id = ZERO_BUDGET_EOD_PROVIDER

    def __init__(
        self,
        *,
        baostock: object,
        tushare: object,
        verifier: object,
    ) -> None:
        for name, component in (
            ("baostock", baostock),
            ("tushare", tushare),
            ("verifier", verifier),
        ):
            if component is None:
                raise ValueError(f"{name} component is required")
        self._baostock = baostock
        self._tushare = tushare
        self._verifier = verifier

    def fetch_cn_trading_days(self, start: date, end: date) -> tuple[date, ...]:
        if not _exact_date(start) or not _exact_date(end):
            raise TypeError("start and end must be date values")
        if start > end:
            raise ValueError("start cannot be after end")
        method = _required_method(self._baostock, "fetch_cn_trading_days", "BaoStock交易日历")
        raw = _dependency_call("BaoStock交易日历", lambda: method(start, end))
        if not isinstance(raw, tuple) or any(not _exact_date(value) for value in raw):
            raise DataQualityError("BaoStock交易日历没有返回严格的date元组。")
        if len(raw) != len(set(raw)) or tuple(sorted(raw)) != raw:
            raise DataQualityError("BaoStock交易日历包含重复或乱序交易日。")
        if any(value < start or value > end for value in raw):
            raise DataQualityError("BaoStock交易日历包含请求区间外日期。")
        return raw

    def fetch_cn_stock_symbols(self) -> tuple[str, ...]:
        method = _required_method(self._baostock, "fetch_cn_stock_symbols", "BaoStock股票清单")
        raw = _dependency_call("BaoStock股票清单", method)
        return _normalize_stock_symbols(raw, label="BaoStock股票清单", sort=True)

    def fetch_daily_increment(
        self,
        symbols: Sequence[str],
        target_date: date,
        *,
        cutoff_timestamp: int | None = None,
        asset_kind: AssetKind = "stocks",
    ) -> DailyIncrementBatch:
        if not _exact_date(target_date):
            raise TypeError("target_date must be a date")
        if asset_kind == "stocks":
            return self._fetch_stock_increment(symbols, target_date, cutoff_timestamp)
        if asset_kind == "indices":
            return self._fetch_index_increment(symbols, target_date, cutoff_timestamp)
        raise ValueError("asset_kind must be stocks or indices")

    def _fetch_stock_increment(
        self,
        symbols: Sequence[str],
        target_date: date,
        cutoff_timestamp: int | None,
    ) -> DailyIncrementBatch:
        requested = _normalize_stock_symbols(symbols, label="请求股票", sort=False)
        requested_codes = tuple(value[:6] for value in requested)
        cutoff = _resolve_cutoff_timestamp(target_date, cutoff_timestamp)

        fetch = _dependency_call(
            "Tushare股票日线",
            lambda: _required_method(
                self._tushare,
                "fetch_daily",
                "Tushare股票日线",
            )(target_date),
        )
        frame, fetched_at, trace_ids = _validate_tushare_fetch(fetch, target_date)
        _require_completed_session(target_date, fetched_at)

        requested_by_code = dict(zip(requested_codes, requested, strict=True))
        available = set(frame["symbol"])
        received_codes = tuple(code for code in requested_codes if code in available)
        if not received_codes:
            raise DataUnavailableError("Tushare股票日线未收到任何请求股票。")
        filtered = frame.loc[frame["symbol"].isin(received_codes)].copy()
        if len(filtered) != len(received_codes) or bool(filtered["symbol"].duplicated().any()):
            raise DataQualityError("Tushare股票日线过滤后身份数量或唯一性不一致。")
        if not set(filtered["symbol"]).issubset(requested_by_code):
            raise DataQualityError("Tushare股票日线过滤后仍包含未请求股票。")

        verification = _dependency_call(
            "AKShare独立核验",
            lambda: _required_method(
                self._verifier,
                "verify_stock_frame",
                "AKShare独立核验",
            )(filtered.copy(), target_date, received_codes),
        )
        status = getattr(verification, "status", None)
        if status is not AKShareVerificationStatus.VERIFIED:
            if status is AKShareVerificationStatus.MISMATCH:
                raise DataQualityError("AKShare独立核验不是VERIFIED，股票批次已拒绝。")
            raise DataUnavailableError("AKShare独立核验不可用，股票批次已拒绝。")

        output = filtered.sort_values("symbol").reset_index(drop=True)
        output["source"] = ZERO_BUDGET_STOCK_SOURCE
        received = tuple(requested_by_code[code] for code in received_codes)
        return _composite_batch(
            frame=output,
            target_date=target_date,
            requested_symbols=requested,
            received_symbols=received,
            fetched_at=fetched_at,
            trace_ids=_prefix_trace_ids(trace_ids, "tushare"),
            cutoff_timestamp=cutoff,
        )

    def _fetch_index_increment(
        self,
        symbols: Sequence[str],
        target_date: date,
        cutoff_timestamp: int | None,
    ) -> DailyIncrementBatch:
        requested = _normalize_index_symbols(symbols)
        if set(requested) != set(BAOSTOCK_CORE_INDEX_SYMBOLS):
            raise DataQualityError("零预算组合只接受已验证的六核心指数集合。")
        cutoff = _resolve_cutoff_timestamp(target_date, cutoff_timestamp)
        batch = _dependency_call(
            "BaoStock核心指数",
            lambda: _required_method(
                self._baostock,
                "fetch_core_index_daily",
                "BaoStock核心指数",
            )(target_date, cutoff_timestamp=cutoff),
        )
        if not isinstance(batch, DailyIncrementBatch):
            raise DataQualityError("BaoStock核心指数没有返回DailyIncrementBatch。")
        if batch.provider != "baostock" or batch.target_date != target_date:
            raise DataQualityError("BaoStock核心指数批次的来源或日期不一致。")
        batch_requested = _normalize_index_symbols(batch.requested_symbols)
        batch_received = _normalize_index_symbols(batch.received_symbols)
        if set(batch_requested) != set(requested):
            raise DataQualityError("BaoStock核心指数批次的请求身份不一致。")
        if set(batch_received) != set(requested):
            raise DataQualityError("BaoStock核心指数批次不完整。")
        if batch.cutoff_timestamp != cutoff:
            raise DataQualityError("BaoStock核心指数批次的cutoff不一致。")
        fetched_at = _aware_utc(batch.fetched_at, "BaoStock核心指数抓取时间")
        _require_completed_session(target_date, fetched_at)
        frame = normalize_overlay_daily(
            batch.frame,
            expected_date=target_date,
            source_id="baostock",
            asset_kind="indices",
        )
        expected_codes = {value[:6] for value in requested}
        if set(frame["symbol"]) != expected_codes or len(frame) != len(expected_codes):
            raise DataQualityError("BaoStock核心指数文件与请求身份不一致。")
        frame["source"] = ZERO_BUDGET_INDEX_SOURCE
        return _composite_batch(
            frame=frame,
            target_date=target_date,
            requested_symbols=requested,
            received_symbols=requested,
            fetched_at=fetched_at,
            trace_ids=_prefix_trace_ids(batch.trace_ids, "baostock"),
            cutoff_timestamp=cutoff,
        )


def _validate_tushare_fetch(
    fetch: object,
    target_date: date,
) -> tuple[pd.DataFrame, datetime, tuple[str, ...]]:
    required = (
        "frame",
        "requested_trade_date",
        "received_trade_dates",
        "received_symbols",
        "fetched_at",
        "trace_ids",
        "provider",
    )
    if any(not hasattr(fetch, field) for field in required):
        raise DataQualityError("Tushare股票日线回执缺少审计字段。")
    if fetch.provider != "tushare" or fetch.requested_trade_date != target_date:
        raise DataQualityError("Tushare股票日线回执的来源或请求日期不一致。")
    if fetch.received_trade_dates != (target_date,):
        raise DataQualityError("Tushare股票日线回执包含错误交易日期。")
    fetched_at = _aware_utc(fetch.fetched_at, "Tushare股票日线抓取时间")
    receipt_symbols = _normalize_received_codes(fetch.received_symbols)
    trace_ids = _normalize_trace_ids(fetch.trace_ids)
    frame = normalize_overlay_daily(
        fetch.frame,
        expected_date=target_date,
        source_id="tushare",
        asset_kind="stocks",
    )
    frame_symbols = tuple(frame["symbol"].astype(str))
    if set(frame_symbols) != set(receipt_symbols) or len(frame_symbols) != len(receipt_symbols):
        raise DataQualityError("Tushare股票日线文件与回执身份集合不一致。")
    return frame, fetched_at, trace_ids


def _composite_batch(
    *,
    frame: pd.DataFrame,
    target_date: date,
    requested_symbols: tuple[str, ...],
    received_symbols: tuple[str, ...],
    fetched_at: datetime,
    trace_ids: tuple[str, ...],
    cutoff_timestamp: int,
) -> DailyIncrementBatch:
    return DailyIncrementBatch(
        frame=frame,
        target_date=target_date,
        requested_symbols=requested_symbols,
        received_symbols=received_symbols,
        fetched_at=fetched_at,
        trace_ids=trace_ids,
        provider=ZERO_BUDGET_EOD_PROVIDER,
        cutoff_timestamp=cutoff_timestamp,
        unit_contract_version=ZERO_BUDGET_UNIT_CONTRACT_VERSION,
        unit_resolution_method_version=ZERO_BUDGET_UNIT_RESOLUTION_METHOD_VERSION,
        amount_multiplier_to_cny=ZERO_BUDGET_AMOUNT_MULTIPLIER_TO_CNY,
    )


def _normalize_stock_symbols(
    values: object,
    *,
    label: str,
    sort: bool,
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise DataQualityError(f"{label}不是证券代码序列。")
    symbols: list[str] = []
    codes: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            raise DataQualityError(f"{label}含非字符串证券代码。")
        symbol = raw.strip().upper()
        match = _STOCK_SYMBOL.fullmatch(symbol)
        if match is None:
            raise DataQualityError(f"{label}含无效沪深A股代码。")
        code = match.group("code")
        exchange = match.group("exchange")
        if (exchange == "SH" and not code.startswith("6")) or (
            exchange == "SZ" and not code.startswith(("0", "3"))
        ):
            raise DataQualityError(f"{label}的代码与交易所后缀冲突。")
        if symbol in symbols or code in codes:
            raise DataQualityError(f"{label}含重复证券代码。")
        symbols.append(symbol)
        codes.add(code)
    if not symbols:
        raise DataUnavailableError(f"{label}为空。")
    return tuple(sorted(symbols) if sort else symbols)


def _normalize_index_symbols(values: object) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise DataQualityError("核心指数请求不是代码序列。")
    normalized: list[str] = []
    for raw in values:
        if not isinstance(raw, str):
            raise DataQualityError("核心指数请求含非字符串代码。")
        symbol = raw.strip().upper()
        if symbol not in BAOSTOCK_CORE_INDEX_SYMBOLS:
            raise DataQualityError("核心指数请求含未验证代码。")
        if symbol in normalized:
            raise DataQualityError("核心指数请求含重复代码。")
        normalized.append(symbol)
    if not normalized:
        raise DataUnavailableError("核心指数请求为空。")
    return tuple(normalized)


def _normalize_received_codes(values: object) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise DataQualityError("Tushare回执证券身份不是代码序列。")
    normalized: list[str] = []
    for raw in values:
        if not isinstance(raw, str) or re.fullmatch(r"\d{6}", raw) is None:
            raise DataQualityError("Tushare回执含无效证券身份。")
        if raw in normalized:
            raise DataQualityError("Tushare回执含重复证券身份。")
        normalized.append(raw)
    if not normalized:
        raise DataUnavailableError("Tushare回执没有任何证券身份。")
    return tuple(normalized)


def _normalize_trace_ids(values: object) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise DataQualityError("供应商trace_ids不是序列。")
    traces: list[str] = []
    for raw in values:
        if not isinstance(raw, str) or _TRACE_ID.fullmatch(raw) is None:
            raise DataQualityError("供应商trace_id格式无效。")
        if raw in traces:
            raise DataQualityError("供应商trace_id重复。")
        traces.append(raw)
    return tuple(traces)


def _prefix_trace_ids(values: object, provider: str) -> tuple[str, ...]:
    return tuple(f"{provider}:{value}" for value in _normalize_trace_ids(values))


def _required_method(component: object, name: str, label: str) -> Callable[..., Any]:
    method = getattr(component, name, None)
    if not callable(method):
        raise DataUnavailableError(f"{label}组件缺少所需接口。")
    return method


def _dependency_call[T](label: str, operation: Callable[[], T]) -> T:
    try:
        return operation()
    except DataQualityError:
        raise DataQualityError(f"{label}质量校验失败，原始错误已脱敏。") from None
    except DataUnavailableError:
        raise DataUnavailableError(f"{label}不可用，原始错误已脱敏。") from None
    except Exception:
        raise DataUnavailableError(f"{label}调用失败，原始错误已脱敏。") from None


def _resolve_cutoff_timestamp(target_date: date, value: int | None) -> int:
    if value is None:
        return int(
            datetime.combine(target_date, time(23, 59, 59), tzinfo=_CN_TIMEZONE).timestamp()
        )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("cutoff_timestamp must be a positive Unix-seconds integer")
    try:
        local = datetime.fromtimestamp(value, tz=_CN_TIMEZONE)
    except (OverflowError, OSError, ValueError):
        raise ValueError("cutoff_timestamp is outside the supported Unix range") from None
    if local.date() != target_date or local.time().replace(tzinfo=None) < _CN_CLOSE:
        raise ValueError("cutoff_timestamp must be on target_date at or after CN close")
    return value


def _require_completed_session(target_date: date, fetched_at: datetime) -> None:
    local = fetched_at.astimezone(_CN_TIMEZONE)
    if target_date > local.date():
        raise DataQualityError("目标交易日尚未发生。")
    if target_date == local.date() and local.time().replace(tzinfo=None) < _CN_CLOSE:
        raise DataQualityError("目标交易日尚未收盘，拒绝盘中日线。")


def _aware_utc(value: object, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DataQualityError(f"{label}必须包含时区。")
    return value.astimezone(UTC)


def _exact_date(value: object) -> bool:
    return isinstance(value, date) and not isinstance(value, datetime)


__all__ = [
    "ZERO_BUDGET_AMOUNT_MULTIPLIER_TO_CNY",
    "ZERO_BUDGET_EOD_PROVIDER",
    "ZERO_BUDGET_INDEX_SOURCE",
    "ZERO_BUDGET_STOCK_SOURCE",
    "ZERO_BUDGET_UNIT_CONTRACT_VERSION",
    "ZERO_BUDGET_UNIT_RESOLUTION_METHOD_VERSION",
    "ZeroBudgetEodMarketData",
]
