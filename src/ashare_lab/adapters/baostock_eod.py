"""Free BaoStock metadata and core-index end-of-day component.

This deliberately small adapter supplies only the pieces for which BaoStock
has a clear, free data contract: the China trading calendar, the current
listed SSE/SZSE A-share identifiers, and one completed unadjusted daily bar for
each of the six research core indices.  It is not a full-market stock-bar
provider and must not be presented as one.

BaoStock is optional.  Importing this module never imports ``baostock``; the
package is loaded only when a live method is called, and tests can inject a
fully offline fake module.
"""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.daily_increment import DailyIncrementBatch
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS

CN_TIMEZONE = ZoneInfo("Asia/Shanghai")
CN_CLOSE = time(15, 0)

# Canonical application identities.  BaoStock uses the corresponding
# lower-case prefix form internally (for example ``sh.000300``).
BAOSTOCK_CORE_INDEX_SYMBOLS = (
    "000001.SH",
    "000300.SH",
    "000852.SH",
    "000905.SH",
    "399001.SZ",
    "399006.SZ",
)
BAOSTOCK_CORE_INDEX_PROVIDER_CODES = {
    "000001.SH": "sh.000001",
    "000300.SH": "sh.000300",
    "000852.SH": "sh.000852",
    "000905.SH": "sh.000905",
    "399001.SZ": "sz.399001",
    "399006.SZ": "sz.399006",
}

BAOSTOCK_INDEX_UNIT_CONTRACT_VERSION = "baostock-index-daily-shares-cny-v1"
BAOSTOCK_INDEX_UNIT_RESOLUTION_METHOD_VERSION = "provider-static-field-contract-v1"

_CALENDAR_FIELDS = ("calendar_date", "is_trading_day")
_STOCK_BASIC_FIELDS = ("code", "code_name", "ipoDate", "outDate", "type", "status")
_INDEX_FIELDS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
)
_INDEX_FIELDS_TEXT = ",".join(_INDEX_FIELDS)
_PROVIDER_SECURITY = re.compile(r"^(?P<exchange>sh|sz|bj)\.(?P<code>\d{6})$")
_EXTERNAL_INDEX = re.compile(r"^\d{6}\.(?:SH|SZ)$")
_MAX_CALENDAR_ROWS = 20_000
_MAX_SECURITY_ROWS = 20_000
_SOURCE = "baostock:eod_unadjusted:indices"


def _load_baostock() -> object:
    """Load the optional dependency only for an actual provider call."""

    try:
        return importlib.import_module("baostock")
    except ImportError as exc:
        raise DataUnavailableError(
            "未安装BaoStock；该免费组件只有在显式配置可选依赖后才能联网读取。"
        ) from exc


class BaoStockEodMarketData:
    """Strict read-only BaoStock calendar, universe and core-index client."""

    provider = "baostock"

    def __init__(
        self,
        *,
        module_loader: Callable[[], object] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._module_loader = module_loader or _load_baostock
        self._clock = clock or (lambda: datetime.now(UTC))

    def fetch_cn_trading_days(self, start: date, end: date) -> tuple[date, ...]:
        """Return verified open sessions, requiring a complete calendar range."""

        if start > end:
            raise ValueError("start cannot be after end")
        with self._session() as module:
            return self._fetch_cn_trading_days(module, start, end)

    def fetch_cn_stock_symbols(self) -> tuple[str, ...]:
        """Return the current listed SSE/SZSE A-share identifiers.

        ``query_stock_basic`` is used instead of a same-day traded list so a
        currently suspended listed stock is not silently removed from the
        universe.  B-shares, Beijing securities, funds and indices are outside
        this component's declared scope.
        """

        with self._session() as module:
            query = _required_callable(module, "query_stock_basic")
            result = _provider_call("证券清单", query)
            rows = _read_result(
                result,
                expected_fields=_STOCK_BASIC_FIELDS,
                operation="证券清单",
                maximum_rows=_MAX_SECURITY_ROWS,
            )

        symbols: list[str] = []
        for row in rows:
            provider_code, _, _, _, security_type, status = row
            type_text = _strict_text(security_type, "type", "证券清单")
            status_text = _strict_text(status, "status", "证券清单")
            if not type_text.isdigit() or status_text not in {"0", "1"}:
                raise DataQualityError("BaoStock证券清单的type或status值无效。")
            if type_text != "1" or status_text != "1":
                continue

            raw_code = _strict_text(provider_code, "code", "证券清单").lower()
            match = _PROVIDER_SECURITY.fullmatch(raw_code)
            if match is None:
                raise DataQualityError("BaoStock当前上市股票包含无法识别的证券代码。")
            exchange = match.group("exchange")
            code = match.group("code")
            if exchange == "bj":
                continue
            if exchange == "sh" and code.startswith("900"):
                continue
            if exchange == "sz" and code.startswith("200"):
                continue
            if exchange == "sh" and code.startswith("6"):
                symbols.append(f"{code}.SH")
                continue
            if exchange == "sz" and code.startswith(("00", "30")):
                symbols.append(f"{code}.SZ")
                continue
            raise DataQualityError(
                "BaoStock当前上市沪深股票出现未纳入已验证A股代码规则的新前缀。"
            )

        if not symbols:
            raise DataUnavailableError("BaoStock没有返回当前上市的沪深A股清单。")
        if len(symbols) != len(set(symbols)):
            raise DataQualityError("BaoStock当前沪深A股清单包含重复证券代码。")
        return tuple(sorted(symbols))

    def fetch_core_index_daily(
        self,
        target_date: date,
        *,
        cutoff_timestamp: int | None = None,
    ) -> DailyIncrementBatch:
        """Return exactly six completed, unadjusted core-index daily bars."""

        fetched_at = _aware_utc(self._clock())
        _require_completed_session(target_date, fetched_at)
        cutoff = _resolve_cutoff_timestamp(target_date, cutoff_timestamp)

        with self._session() as module:
            sessions = self._fetch_cn_trading_days(module, target_date, target_date)
            if sessions != (target_date,):
                raise DataQualityError("BaoStock交易日历未确认目标日期为A股交易日。")

            query = _required_callable(module, "query_history_k_data_plus")
            normalized_rows: list[dict[str, object]] = []
            for external_symbol in BAOSTOCK_CORE_INDEX_SYMBOLS:
                provider_code = BAOSTOCK_CORE_INDEX_PROVIDER_CODES[external_symbol]
                result = _provider_call(
                    "核心指数日线",
                    query,
                    provider_code,
                    _INDEX_FIELDS_TEXT,
                    start_date=target_date.isoformat(),
                    end_date=target_date.isoformat(),
                    frequency="d",
                    adjustflag="3",
                )
                rows = _read_result(
                    result,
                    expected_fields=_INDEX_FIELDS,
                    operation=f"核心指数日线 {external_symbol}",
                    maximum_rows=2,
                )
                if not rows:
                    raise DataUnavailableError(
                        f"BaoStock没有返回{target_date.isoformat()}的{external_symbol}日线。"
                    )
                if len(rows) != 1:
                    raise DataQualityError(
                        f"BaoStock {external_symbol}在指定交易日返回了重复日线。"
                    )
                normalized_rows.append(
                    _normalize_index_row(
                        rows[0],
                        external_symbol=external_symbol,
                        provider_code=provider_code,
                        target_date=target_date,
                        fetched_at=fetched_at,
                    )
                )

        frame = pd.DataFrame(
            normalized_rows,
            columns=["symbol", *CANONICAL_DAILY_COLUMNS],
        ).sort_values("symbol", kind="stable", ignore_index=True)
        return DailyIncrementBatch(
            frame=frame,
            target_date=target_date,
            requested_symbols=BAOSTOCK_CORE_INDEX_SYMBOLS,
            received_symbols=BAOSTOCK_CORE_INDEX_SYMBOLS,
            fetched_at=fetched_at,
            trace_ids=(),
            provider=self.provider,
            cutoff_timestamp=cutoff,
            unit_contract_version=BAOSTOCK_INDEX_UNIT_CONTRACT_VERSION,
            unit_resolution_method_version=BAOSTOCK_INDEX_UNIT_RESOLUTION_METHOD_VERSION,
            amount_multiplier_to_cny="1",
        )

    def _fetch_cn_trading_days(
        self,
        module: object,
        start: date,
        end: date,
    ) -> tuple[date, ...]:
        query = _required_callable(module, "query_trade_dates")
        result = _provider_call(
            "交易日历",
            query,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )
        rows = _read_result(
            result,
            expected_fields=_CALENDAR_FIELDS,
            operation="交易日历",
            maximum_rows=_MAX_CALENDAR_ROWS,
        )
        if not rows:
            raise DataUnavailableError("BaoStock交易日历返回空结果。")

        expected_dates = _calendar_dates(start, end)
        flags: dict[date, str] = {}
        for raw_date, raw_flag in rows:
            parsed = _parse_iso_date(raw_date, "交易日历")
            if parsed < start or parsed > end:
                raise DataQualityError("BaoStock交易日历返回请求区间外日期。")
            if parsed in flags:
                raise DataQualityError("BaoStock交易日历包含重复日期。")
            flag = _strict_text(raw_flag, "is_trading_day", "交易日历")
            if flag not in {"0", "1"}:
                raise DataQualityError("BaoStock交易日历包含无效交易日标记。")
            flags[parsed] = flag
        if set(flags) != set(expected_dates):
            raise DataQualityError("BaoStock交易日历没有完整覆盖请求的自然日区间。")
        return tuple(value for value in expected_dates if flags[value] == "1")

    @contextmanager
    def _session(self) -> Iterator[object]:
        try:
            module = self._module_loader()
        except DataUnavailableError:
            raise
        except ImportError as exc:
            raise DataUnavailableError("BaoStock可选依赖不可用。") from exc
        except Exception as exc:  # noqa: BLE001 - provider import boundary
            raise DataUnavailableError(
                f"BaoStock组件加载失败：{type(exc).__name__}。"
            ) from exc

        login = _required_callable(module, "login")
        login_result = _provider_call("登录", login)
        _require_success(login_result, "登录")
        try:
            yield module
        finally:
            logout = getattr(module, "logout", None)
            if callable(logout):
                with suppress(Exception):
                    logout()


def _required_callable(module: object, name: str) -> Callable[..., Any]:
    value = getattr(module, name, None)
    if not callable(value):
        raise DataUnavailableError(f"当前BaoStock版本缺少{name}接口。")
    return value


def _provider_call(operation: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - normalize provider boundary
        raise DataUnavailableError(
            f"BaoStock{operation}请求不可用：{type(exc).__name__}。"
        ) from exc


def _require_success(result: object, operation: str) -> None:
    if getattr(result, "error_code", None) != "0":
        raise DataUnavailableError(f"BaoStock{operation}返回失败状态。")


def _read_result(
    result: object,
    *,
    expected_fields: Sequence[str],
    operation: str,
    maximum_rows: int,
) -> tuple[tuple[object, ...], ...]:
    _require_success(result, operation)
    raw_fields = getattr(result, "fields", None)
    if (
        not isinstance(raw_fields, Sequence)
        or isinstance(raw_fields, (str, bytes))
        or any(not isinstance(field, str) for field in raw_fields)
    ):
        raise DataQualityError(f"BaoStock{operation}响应缺少结构化字段清单。")
    if tuple(raw_fields) != tuple(expected_fields):
        raise DataQualityError(f"BaoStock{operation}响应字段与请求合同不一致。")

    next_row = getattr(result, "next", None)
    get_row = getattr(result, "get_row_data", None)
    if not callable(next_row) or not callable(get_row):
        raise DataQualityError(f"BaoStock{operation}响应不是可迭代结果集。")

    rows: list[tuple[object, ...]] = []
    while True:
        try:
            has_row = next_row()
        except Exception as exc:  # noqa: BLE001 - provider iterator boundary
            raise DataUnavailableError(
                f"BaoStock{operation}结果读取失败：{type(exc).__name__}。"
            ) from exc
        if not isinstance(has_row, bool):
            raise DataQualityError(f"BaoStock{operation}迭代状态不是布尔值。")
        if not has_row:
            break
        try:
            raw_row = get_row()
        except Exception as exc:  # noqa: BLE001 - provider iterator boundary
            raise DataUnavailableError(
                f"BaoStock{operation}结果读取失败：{type(exc).__name__}。"
            ) from exc
        if (
            not isinstance(raw_row, Sequence)
            or isinstance(raw_row, (str, bytes))
            or len(raw_row) != len(expected_fields)
        ):
            raise DataQualityError(f"BaoStock{operation}返回行宽与字段合同不一致。")
        rows.append(tuple(raw_row))
        if len(rows) > maximum_rows:
            raise DataQualityError(f"BaoStock{operation}返回行数超过安全上限。")
    _require_success(result, operation)
    return tuple(rows)


def _normalize_index_row(
    row: Sequence[object],
    *,
    external_symbol: str,
    provider_code: str,
    target_date: date,
    fetched_at: datetime,
) -> dict[str, object]:
    (
        raw_date,
        raw_code,
        raw_open,
        raw_high,
        raw_low,
        raw_close,
        raw_preclose,
        raw_volume,
        raw_amount,
    ) = row
    if _parse_iso_date(raw_date, external_symbol) != target_date:
        raise DataQualityError(f"BaoStock {external_symbol}日线日期与目标交易日不一致。")
    if _strict_text(raw_code, "code", external_symbol).lower() != provider_code:
        raise DataQualityError(f"BaoStock {external_symbol}日线证券代码与请求不一致。")

    open_price = _positive_decimal(raw_open, "open", external_symbol)
    high = _positive_decimal(raw_high, "high", external_symbol)
    low = _positive_decimal(raw_low, "low", external_symbol)
    close = _positive_decimal(raw_close, "close", external_symbol)
    previous_close = _positive_decimal(raw_preclose, "preclose", external_symbol)
    volume_shares = _positive_decimal(raw_volume, "volume", external_symbol)
    amount_cny = _positive_decimal(raw_amount, "amount", external_symbol)
    if high < max(open_price, close, low) or low > min(open_price, close, high):
        raise DataQualityError(f"BaoStock {external_symbol}日线开高低收关系无效。")

    if _EXTERNAL_INDEX.fullmatch(external_symbol) is None:
        raise DataQualityError("BaoStock核心指数的外部证券标识无效。")
    return {
        "symbol": external_symbol[:6],
        "trade_date": target_date,
        "open": float(open_price),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "prev_close": float(previous_close),
        # BaoStock's daily field contract defines volume as shares and amount
        # as CNY.  There is intentionally no inferred or empirical multiplier.
        "volume_shares": float(volume_shares),
        "amount_cny": float(amount_cny),
        "turnover_pct": math.nan,
        "source": _SOURCE,
        "retrieved_at": fetched_at.isoformat().replace("+00:00", "Z"),
    }


def _strict_text(value: object, field: str, operation: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataQualityError(f"BaoStock{operation}的{field}字段缺失或类型无效。")
    return value.strip()


def _parse_iso_date(value: object, operation: str) -> date:
    text = _strict_text(value, "date", operation)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) is None:
        raise DataQualityError(f"BaoStock{operation}包含无效日期。")
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise DataQualityError(f"BaoStock{operation}包含无效日期。") from exc


def _positive_decimal(value: object, field: str, symbol: str) -> Decimal:
    if not isinstance(value, str) or not value.strip():
        raise DataQualityError(f"BaoStock {symbol}日线{field}缺失或类型无效。")
    try:
        parsed = Decimal(value.strip())
    except (InvalidOperation, ValueError) as exc:
        raise DataQualityError(f"BaoStock {symbol}日线{field}不是有效数值。") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise DataQualityError(f"BaoStock {symbol}日线{field}必须为正的有限数值。")
    return parsed


def _calendar_dates(start: date, end: date) -> tuple[date, ...]:
    return tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DataQualityError("BaoStock抓取时间必须包含时区。")
    return value.astimezone(UTC)


def _require_completed_session(target_date: date, fetched_at: datetime) -> None:
    local = fetched_at.astimezone(CN_TIMEZONE)
    if target_date > local.date():
        raise DataQualityError("BaoStock目标交易日尚未发生。")
    if target_date == local.date() and local.time().replace(tzinfo=None) < CN_CLOSE:
        raise DataQualityError("BaoStock目标交易日尚未收盘，拒绝盘中日线。")


def _resolve_cutoff_timestamp(target_date: date, value: int | None) -> int:
    if value is None:
        return int(datetime.combine(target_date, time(23, 59, 59), tzinfo=CN_TIMEZONE).timestamp())
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("cutoff_timestamp must be a positive Unix-seconds integer")
    try:
        local = datetime.fromtimestamp(value, tz=CN_TIMEZONE)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("cutoff_timestamp is outside the supported Unix range") from exc
    if local.date() != target_date or local.time().replace(tzinfo=None) < CN_CLOSE:
        raise ValueError("cutoff_timestamp must be on target_date at or after CN close")
    return value
