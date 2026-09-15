"""Private child protocol for bounded BaoStock stock-bar verification."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout, suppress
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from ashare_lab.adapters.baostock_eod import (
    _load_baostock,
    _provider_call,
    _read_result,
    _require_completed_session,
    _require_success,
    _required_callable,
)
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError

_SYMBOL = re.compile(r"^[0-9]{6}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FIELDS = (
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
_FIELDS_TEXT = ",".join(_FIELDS)
_MAX_REQUEST_BYTES = 32_768
_MAX_SYMBOLS = 64
_METHOD_VERSION = "baostock-unadjusted-daily-sample-v1"


def _read_sample(
    module: object,
    *,
    symbols: tuple[str, ...],
    target_date: date,
    retrieved_at: datetime,
) -> dict[str, object]:
    _require_completed_session(target_date, retrieved_at)
    login = _required_callable(module, "login")
    login_result = _provider_call("登录", login)
    _require_success(login_result, "登录")
    try:
        query = _required_callable(module, "query_history_k_data_plus")
        rows: list[dict[str, object]] = []
        for symbol in symbols:
            provider_code = _provider_code(symbol)
            result = _provider_call(
                "个股日线核验",
                query,
                provider_code,
                _FIELDS_TEXT,
                start_date=target_date.isoformat(),
                end_date=target_date.isoformat(),
                frequency="d",
                adjustflag="3",
            )
            raw_rows = _read_result(
                result,
                expected_fields=_FIELDS,
                operation=f"个股日线核验 {symbol}",
                maximum_rows=2,
            )
            if not raw_rows:
                raise DataUnavailableError("BaoStock个股日线核验缺少目标交易日。")
            if len(raw_rows) != 1:
                raise DataQualityError("BaoStock个股日线核验在目标交易日返回重复行。")
            rows.append(
                _normalize_row(
                    raw_rows[0],
                    symbol=symbol,
                    provider_code=provider_code,
                    target_date=target_date,
                )
            )
    finally:
        logout = getattr(module, "logout", None)
        if callable(logout):
            with suppress(Exception):
                logout()
    return {
        "provider": "baostock",
        "method_version": _METHOD_VERSION,
        "target_date": target_date.isoformat(),
        "retrieved_at": retrieved_at.isoformat(),
        "rows": rows,
    }


def _normalize_row(
    raw: Sequence[object],
    *,
    symbol: str,
    provider_code: str,
    target_date: date,
) -> dict[str, object]:
    (
        raw_date,
        raw_code,
        raw_open,
        raw_high,
        raw_low,
        raw_close,
        raw_prev_close,
        raw_volume,
        raw_amount,
    ) = raw
    if raw_date != target_date.isoformat():
        raise DataQualityError("BaoStock个股日线核验日期与请求不一致。")
    if raw_code != provider_code:
        raise DataQualityError("BaoStock个股日线核验证券身份与请求不一致。")
    open_price = _decimal(raw_open, positive=True)
    high = _decimal(raw_high, positive=True)
    low = _decimal(raw_low, positive=True)
    close = _decimal(raw_close, positive=True)
    previous_close = _decimal(raw_prev_close, positive=True)
    volume = _decimal(raw_volume, positive=False)
    amount = _decimal(raw_amount, positive=False)
    if high < max(open_price, low, close) or low > min(open_price, high, close):
        raise DataQualityError("BaoStock个股日线核验开高低收关系无效。")
    if (volume == 0) != (amount == 0):
        raise DataQualityError("BaoStock个股日线核验成交量额零值状态冲突。")
    return {
        "symbol": symbol,
        "trade_date": target_date.isoformat(),
        "open": float(open_price),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "prev_close": float(previous_close),
        # BaoStock's daily contract is shares and CNY; no inferred multiplier.
        "volume_shares": float(volume),
        "amount_cny": float(amount),
    }


def _provider_code(symbol: str) -> str:
    if _SYMBOL.fullmatch(symbol) is None:
        raise DataQualityError("BaoStock个股日线核验证券代码无效。")
    if symbol.startswith("6"):
        return f"sh.{symbol}"
    if symbol.startswith(("0", "3")):
        return f"sz.{symbol}"
    raise DataQualityError("BaoStock个股日线核验证券代码前缀无效。")


def _decimal(value: object, *, positive: bool) -> Decimal:
    if not isinstance(value, str) or not value.strip():
        raise DataQualityError("BaoStock个股日线核验行情字段缺失或类型无效。")
    try:
        parsed = Decimal(value.strip())
    except (InvalidOperation, ValueError):
        raise DataQualityError("BaoStock个股日线核验行情字段不是有效数值。") from None
    if not parsed.is_finite() or parsed < 0 or (positive and parsed == 0):
        raise DataQualityError("BaoStock个股日线核验行情字段超出有效范围。")
    return parsed


def _request() -> tuple[tuple[str, ...], date, datetime]:
    raw = sys.stdin.read(_MAX_REQUEST_BYTES + 1)
    if len(raw.encode("utf-8")) > _MAX_REQUEST_BYTES:
        raise DataQualityError("BaoStock个股日线核验请求异常过大。")
    request = json.loads(raw)
    if not isinstance(request, dict) or set(request) != {
        "operation",
        "target_date",
        "now",
        "symbols",
    }:
        raise DataQualityError("BaoStock个股日线核验请求结构无效。")
    if request.get("operation") != "verify_stock_sample":
        raise DataQualityError("BaoStock个股日线核验操作无效。")
    raw_target = request.get("target_date")
    if not isinstance(raw_target, str) or _DATE.fullmatch(raw_target) is None:
        raise DataQualityError("BaoStock个股日线核验请求日期无效。")
    try:
        target_date = date.fromisoformat(raw_target)
        retrieved_at = datetime.fromisoformat(request["now"])
    except (KeyError, TypeError, ValueError):
        raise DataQualityError("BaoStock个股日线核验请求时间无效。") from None
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise DataQualityError("BaoStock个股日线核验请求时间必须包含时区。")
    raw_symbols = request.get("symbols")
    if (
        not isinstance(raw_symbols, list)
        or not 1 <= len(raw_symbols) <= _MAX_SYMBOLS
        or any(
            not isinstance(value, str) or _SYMBOL.fullmatch(value) is None for value in raw_symbols
        )
        or tuple(sorted(raw_symbols)) != tuple(raw_symbols)
        or len(set(raw_symbols)) != len(raw_symbols)
    ):
        raise DataQualityError("BaoStock个股日线核验请求证券身份无效。")
    symbols = tuple(raw_symbols)
    return symbols, target_date, retrieved_at


def main() -> int:
    try:
        symbols, target_date, retrieved_at = _request()
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            module = _load_baostock()
            result = _read_sample(
                module,
                symbols=symbols,
                target_date=target_date,
                retrieved_at=retrieved_at,
            )
        print(json.dumps({"result": result}, separators=(",", ":")))
        return 0
    except Exception as exc:  # provider details must not cross the child boundary
        error = "quality" if isinstance(exc, DataQualityError) else "unavailable"
        print(json.dumps({"error": error}, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
