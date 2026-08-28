"""Strict Infoway adapter for completed, unadjusted A-share daily bars.

Infoway uses the same authenticated HTTP surface for current and historical
candles.  This adapter subclasses the existing realtime client so request
sanitization and the shared free-plan rate limiter remain the single source of
truth.  It adds the stronger identity, date, unit and OHLC validations required
before a response can enter a research database.
"""

from __future__ import annotations

import math
import re
import time as time_module
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from ashare_lab.adapters.infoway_realtime import InfowayRealtimeMarketData
from ashare_lab.domain.data_sources import SourceId
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.daily_increment import AssetKind, DailyIncrementBatch
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS

_CN_TIMEZONE = ZoneInfo("Asia/Shanghai")
_CN_CLOSE = time(15, 0)
_MAX_BATCH = 100
_PROVIDER_SYMBOL = re.compile(r"^(?P<code>\d{6})\.(?P<exchange>SH|SZ)$")
_SH_A_SHARE_PREFIXES = ("600", "601", "603", "605", "688", "689")
_SZ_A_SHARE_PREFIXES = ("000", "001", "002", "003", "300", "301", "302")
_DATE_FORMAT = "%Y%m%d"
_SOURCE_LABELS: dict[AssetKind, str] = {
    "stocks": "infoway:eod_unadjusted:stocks",
    "indices": "infoway:eod_unadjusted:indices",
}
_EOD_MINIMUM_INTERVAL_SECONDS = 1.2
_MAX_429_RETRIES = 2
_MIN_429_DELAY_SECONDS = 2.0
_MAX_429_DELAY_SECONDS = 30.0
_HISTORICAL_AMOUNT_MULTIPLIER_TO_CNY = Decimal("1")
_SAME_SESSION_PROVISIONAL_AMOUNT_MULTIPLIER_TO_CNY = Decimal("100")
_VERSION_TOKEN = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True, slots=True)
class InfowayEodUnitContract:
    """Explicit, externally verified mapping from provider units to canonical units.

    Infoway's public pages have used inconsistent prose for ``vw``.  Therefore
    the adapter has no implicit unit default.  A caller must supply the mapping
    from its current account data dictionary or written provider confirmation.
    Runtime amount/volume consistency is checked as an additional guard; it is
    not treated as proof of the provider's contractual unit definition.
    """

    volume_multiplier_to_shares: Decimal
    amount_field: str
    amount_multiplier_to_cny: Decimal
    provisional_amount_multiplier_to_cny: Decimal
    contract_version: str
    resolution_method_version: str
    verified_reference: str
    volume_semantics: str = "shares"
    amount_semantics: str = "turnover_value_cny"

    def __post_init__(self) -> None:
        try:
            volume_multiplier = Decimal(str(self.volume_multiplier_to_shares))
            amount_multiplier = Decimal(str(self.amount_multiplier_to_cny))
            provisional_multiplier = Decimal(str(self.provisional_amount_multiplier_to_cny))
        except InvalidOperation as exc:
            raise ValueError("unit multipliers must be finite decimal values") from exc
        if not volume_multiplier.is_finite() or volume_multiplier <= 0:
            raise ValueError("volume_multiplier_to_shares must be positive and finite")
        if not amount_multiplier.is_finite() or amount_multiplier <= 0:
            raise ValueError("amount_multiplier_to_cny must be positive and finite")
        if amount_multiplier != _HISTORICAL_AMOUNT_MULTIPLIER_TO_CNY:
            raise ValueError("historical amount_multiplier_to_cny must be explicitly 1")
        if (
            not provisional_multiplier.is_finite()
            or provisional_multiplier != _SAME_SESSION_PROVISIONAL_AMOUNT_MULTIPLIER_TO_CNY
        ):
            raise ValueError("provisional_amount_multiplier_to_cny must be explicitly 100")
        if self.amount_field not in {"vw", "vm"}:
            raise ValueError("amount_field must be explicitly verified as 'vw' or 'vm'")
        if self.volume_semantics != "shares":
            raise ValueError("volume_semantics must explicitly be 'shares'")
        if self.amount_semantics != "turnover_value_cny":
            raise ValueError("amount_semantics must explicitly be 'turnover_value_cny'")
        if not self.verified_reference.strip():
            raise ValueError("verified_reference is required; units cannot be guessed")
        for name, value in (
            ("contract_version", self.contract_version),
            ("resolution_method_version", self.resolution_method_version),
        ):
            if not value.strip() or _VERSION_TOKEN.fullmatch(value.strip()) is None:
                raise ValueError(f"{name} must be a non-empty audit-safe version token")
        object.__setattr__(self, "volume_multiplier_to_shares", volume_multiplier)
        object.__setattr__(self, "amount_multiplier_to_cny", amount_multiplier)
        object.__setattr__(
            self,
            "provisional_amount_multiplier_to_cny",
            provisional_multiplier,
        )
        object.__setattr__(self, "contract_version", self.contract_version.strip())
        object.__setattr__(
            self,
            "resolution_method_version",
            self.resolution_method_version.strip(),
        )


@dataclass(frozen=True, slots=True)
class _PreparedTargetBar:
    provider_symbol: str
    target_date: date
    open_price: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    previous_close: Decimal
    volume_shares: Decimal
    amount_raw: Decimal


class InfowayEodMarketData(InfowayRealtimeMarketData):
    """Rate-limited CN calendar, symbol-list and daily-increment adapter."""

    source_id = SourceId.INFOWAY

    def __init__(
        self,
        api_key: str,
        *,
        unit_contract: InfowayEodUnitContract | None = None,
        minimum_interval_seconds: float = _EOD_MINIMUM_INTERVAL_SECONDS,
        sleeper: Callable[[float], None] = time_module.sleep,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key,
            minimum_interval_seconds=minimum_interval_seconds,
            sleeper=sleeper,
            **kwargs,
        )
        self._unit_contract = unit_contract
        self._retry_sleeper = sleeper
        self._resolved_amount_multiplier_by_date: dict[date, Decimal] = {}

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> tuple[Any, str]:
        """Issue one sanitized request with an EOD-only bounded 429 retry."""

        for attempt in range(_MAX_429_RETRIES + 1):
            self._limiter.wait()
            try:
                response = self._client.request(
                    method,
                    f"{self._base_url}{path}",
                    headers={"apiKey": self._api_key, "Accept": "application/json"},
                    params=params,
                    json=json,
                )
                response.raise_for_status()
                document = response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 429 and attempt < _MAX_429_RETRIES:
                    delay = _bounded_retry_after_seconds(
                        exc.response.headers.get("Retry-After"),
                        now=_aware_utc(self._utcnow()),
                    )
                    self._retry_sleeper(delay)
                    continue
                if status == 429:
                    raise DataUnavailableError(
                        "Infoway 请求达到套餐限额，有限重试后仍不可用。"
                    ) from exc
                if status in {401, 403}:
                    raise DataUnavailableError("Infoway 鉴权或套餐权限不足。") from exc
                raise DataUnavailableError(f"Infoway HTTP 请求失败（状态码 {status}）。") from exc
            except (httpx.HTTPError, ValueError) as exc:
                raise DataUnavailableError(f"Infoway 请求不可用：{type(exc).__name__}。") from exc

            if not isinstance(document, dict):
                raise DataQualityError("Infoway 响应不是JSON对象。")
            ret = document.get("ret")
            if ret != 200:
                # Provider messages may echo request details.  Do not include them.
                raise DataUnavailableError(f"Infoway 返回失败状态：{ret!r}。")
            trace_id = document.get("traceId")
            return document.get("data"), trace_id if isinstance(trace_id, str) else ""

        raise AssertionError("bounded retry loop exhausted without returning or raising")

    def fetch_cn_trading_days(self, start: date, end: date) -> tuple[date, ...]:
        if start > end:
            raise ValueError("start cannot be after end")
        payload, _ = self._request_json(
            "GET",
            "/common/basic/markets/trading_days",
            params={
                "market": "CN",
                "beginDay": start.strftime(_DATE_FORMAT),
                "endDay": end.strftime(_DATE_FORMAT),
            },
        )
        if not isinstance(payload, Mapping):
            raise DataQualityError("Infoway CN交易日历数据结构异常。")

        full_days = _parse_calendar_values(payload.get("trade_days"), "trade_days")
        half_days = _parse_calendar_values(payload.get("half_trade_days"), "half_trade_days")
        if set(full_days).intersection(half_days):
            raise DataQualityError("Infoway CN交易日历的全天与半天日期重复。")
        sessions = full_days + half_days
        if len(set(sessions)) != len(sessions):
            raise DataQualityError("Infoway CN交易日历包含重复日期。")
        if any(value < start or value > end for value in sessions):
            raise DataQualityError("Infoway CN交易日历返回请求区间外日期。")
        return tuple(sorted(sessions))

    def fetch_cn_stock_symbols(self) -> tuple[str, ...]:
        records = self.fetch_symbol_list()
        symbols: list[str] = []
        for record in records:
            raw_symbol = record.get("symbol")
            if not isinstance(raw_symbol, str):
                raise DataQualityError("Infoway 沪深股票清单包含无效证券代码。")
            index_flag = record.get("index")
            if not isinstance(index_flag, bool):
                raise DataQualityError("Infoway STOCK_CN清单的index字段缺失或不是布尔值。")
            if index_flag:
                continue
            symbol = raw_symbol.strip().upper()
            match = _PROVIDER_SYMBOL.fullmatch(symbol)
            if match is None:
                # STOCK_CN currently documents SSE/SZSE equities.  Ignore any
                # explicitly different market products instead of inventing a
                # mapping for indices, sectors or future exchanges.
                continue
            if not _is_a_share_provider_symbol(match):
                continue
            exchange = record.get("exchange")
            if exchange not in {None, ""}:
                expected = {"SH", "SSE"} if match.group("exchange") == "SH" else {"SZ", "SZSE"}
                if str(exchange).strip().upper() not in expected:
                    raise DataQualityError("Infoway 股票代码后缀与交易所字段冲突。")
            symbols.append(symbol)

        if not symbols:
            raise DataUnavailableError("Infoway STOCK_CN清单没有可识别的沪深股票。")
        if len(set(symbols)) != len(symbols):
            raise DataQualityError("Infoway 沪深股票清单包含重复证券代码。")
        return tuple(sorted(symbols))

    def fetch_daily_increment(
        self,
        symbols: Sequence[str],
        target_date: date,
        *,
        cutoff_timestamp: int | None = None,
        asset_kind: AssetKind = "stocks",
    ) -> DailyIncrementBatch:
        contract = self._require_unit_contract()
        normalized_asset_kind = _validate_asset_kind(asset_kind)
        requested = _normalize_provider_symbols(symbols)
        fetched_at = _aware_utc(self._utcnow())
        _require_completed_session(target_date, fetched_at)
        cutoff = _resolve_cutoff_timestamp(target_date, cutoff_timestamp)

        sessions = self.fetch_cn_trading_days(target_date, target_date)
        if sessions != (target_date,):
            raise DataQualityError("Infoway交易日历未确认目标日期为CN交易日。")

        prepared_rows: list[_PreparedTargetBar] = []
        trace_ids: list[str] = []
        for batch in _chunks(requested, _MAX_BATCH):
            payload, trace_id = self._request_json(
                "POST",
                "/stock/v2/batch_kline",
                json={
                    "klineType": 8,
                    "klineNum": 2,
                    "codes": ",".join(batch),
                    "timestamp": cutoff,
                },
            )
            grouped = _group_kline_payload(payload, requested_symbols=batch)
            for symbol in batch:
                bars = grouped.get(symbol, ())
                if not bars:
                    continue
                prepared = _prepare_target_bar(
                    symbol,
                    bars,
                    target_date=target_date,
                    contract=contract,
                )
                if prepared is None:
                    # An empty, stale-only or explicit zero-trade response is
                    # normal for a suspended security.  Keep it out of the
                    # frame and let the caller's market-coverage gate decide.
                    continue
                prepared_rows.append(prepared)
            if trace_id:
                trace_ids.append(trace_id)

        if not prepared_rows:
            raise DataUnavailableError(f"Infoway没有返回{target_date.isoformat()}的已完成CN日线。")
        amount_multiplier = self._resolve_batch_amount_multiplier(
            prepared_rows,
            target_date=target_date,
            fetched_at=fetched_at,
            asset_kind=normalized_asset_kind,
            contract=contract,
        )
        rows = [
            _normalize_prepared_bar(
                prepared,
                amount_multiplier_to_cny=amount_multiplier,
                retrieved_at=fetched_at,
                asset_kind=normalized_asset_kind,
            )
            for prepared in prepared_rows
        ]
        # The per-date decision becomes reusable only after the complete batch
        # has passed parsing, uniqueness and cross-symbol consistency checks.
        self._resolved_amount_multiplier_by_date[target_date] = amount_multiplier
        frame = pd.DataFrame(rows, columns=["symbol", *CANONICAL_DAILY_COLUMNS])
        return DailyIncrementBatch(
            frame=frame.sort_values("symbol").reset_index(drop=True),
            target_date=target_date,
            requested_symbols=requested,
            received_symbols=tuple(row.provider_symbol for row in prepared_rows),
            fetched_at=fetched_at,
            trace_ids=tuple(trace_ids),
            provider=self.provider,
            cutoff_timestamp=cutoff,
            unit_contract_version=contract.contract_version,
            unit_resolution_method_version=contract.resolution_method_version,
            amount_multiplier_to_cny=str(amount_multiplier),
        )

    def _resolve_batch_amount_multiplier(
        self,
        rows: Sequence[_PreparedTargetBar],
        *,
        target_date: date,
        fetched_at: datetime,
        asset_kind: AssetKind,
        contract: InfowayEodUnitContract,
    ) -> Decimal:
        local_fetch = fetched_at.astimezone(_CN_TIMEZONE)
        same_session_after_close = (
            target_date == local_fetch.date()
            and local_fetch.time().replace(tzinfo=None) >= _CN_CLOSE
        )
        if asset_kind == "indices":
            if not same_session_after_close:
                return contract.amount_multiplier_to_cny
            inherited = self._resolved_amount_multiplier_by_date.get(target_date)
            if inherited is None:
                raise DataQualityError(
                    "Infoway同日指数成交额倍率尚未由完整股票批次唯一锁定，拒绝单独猜测。"
                )
            return inherited

        allowed = [contract.amount_multiplier_to_cny]
        if same_session_after_close:
            allowed.append(contract.provisional_amount_multiplier_to_cny)
        decisions: dict[str, Decimal] = {}
        for row in rows:
            matches = tuple(
                multiplier
                for multiplier in allowed
                if _amount_volume_is_consistent(
                    volume_shares=row.volume_shares,
                    amount_cny=row.amount_raw * multiplier,
                    low=row.low,
                    high=row.high,
                )
            )
            if len(matches) != 1:
                raise DataQualityError(
                    f"Infoway {row.provider_symbol} 成交额倍率无法由量价区间唯一判定，"
                    "整批单位合同验证失败。"
                )
            decisions[row.provider_symbol] = matches[0]
        selected = set(decisions.values())
        if len(selected) != 1:
            raise DataQualityError(
                "Infoway股票批次出现不同成交额倍率，拒绝混用历史与同日临时口径。"
            )
        return next(iter(selected))

    def _require_unit_contract(self) -> InfowayEodUnitContract:
        if self._unit_contract is None:
            raise DataQualityError(
                "Infoway成交量与成交额单位尚未由当前数据字典验证，拒绝猜测vw/vm含义。"
            )
        return self._unit_contract


def _parse_calendar_values(value: object, field: str) -> tuple[date, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DataQualityError(f"Infoway CN交易日历字段{field}结构异常。")
    parsed: list[date] = []
    for item in value:
        if not re.fullmatch(r"\d{8}", item):
            raise DataQualityError(f"Infoway CN交易日历字段{field}包含无效日期。")
        try:
            parsed.append(datetime.strptime(item, _DATE_FORMAT).date())
        except ValueError as exc:
            raise DataQualityError(f"Infoway CN交易日历字段{field}包含无效日期。") from exc
    if len(set(parsed)) != len(parsed):
        raise DataQualityError(f"Infoway CN交易日历字段{field}包含重复日期。")
    return tuple(parsed)


def _normalize_provider_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols))
    if not normalized:
        raise ValueError("at least one SSE/SZSE symbol is required")
    if any(_PROVIDER_SYMBOL.fullmatch(symbol) is None for symbol in normalized):
        raise ValueError("symbols must use Infoway six-digit .SH or .SZ format")
    return normalized


def _is_a_share_provider_symbol(match: re.Match[str]) -> bool:
    code = match.group("code")
    exchange = match.group("exchange")
    if exchange == "SH":
        return code.startswith(_SH_A_SHARE_PREFIXES)
    return code.startswith(_SZ_A_SHARE_PREFIXES)


def _validate_asset_kind(value: object) -> AssetKind:
    if value not in {"stocks", "indices"}:
        raise ValueError("asset_kind must be 'stocks' or 'indices'")
    return cast(AssetKind, value)


def _chunks(values: Sequence[str], size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(values[index : index + size]) for index in range(0, len(values), size))


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DataQualityError("Infoway抓取时间必须包含时区。")
    return value.astimezone(UTC)


def _bounded_retry_after_seconds(value: str | None, *, now: datetime) -> float:
    delay: float | None = None
    if value is not None:
        try:
            delay = float(value.strip())
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None or retry_at.utcoffset() is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                delay = (retry_at.astimezone(UTC) - now).total_seconds()
            except (TypeError, ValueError, OverflowError):
                delay = None
    if delay is None or not math.isfinite(delay):
        delay = _MIN_429_DELAY_SECONDS
    return min(
        max(delay, _MIN_429_DELAY_SECONDS),
        _MAX_429_DELAY_SECONDS,
    )


def _require_completed_session(target_date: date, fetched_at: datetime) -> None:
    local = fetched_at.astimezone(_CN_TIMEZONE)
    if target_date > local.date():
        raise DataQualityError("目标交易日尚未发生。")
    if target_date == local.date() and local.time().replace(tzinfo=None) < _CN_CLOSE:
        raise DataQualityError("目标交易日尚未收盘，拒绝把盘中K线写成完整日线。")


def _resolve_cutoff_timestamp(target_date: date, value: int | None) -> int:
    if value is None:
        return int(datetime.combine(target_date, time(23, 59, 59), tzinfo=_CN_TIMEZONE).timestamp())
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("cutoff_timestamp must be a positive Unix-seconds integer")
    try:
        local = datetime.fromtimestamp(value, tz=_CN_TIMEZONE)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("cutoff_timestamp is outside the supported Unix range") from exc
    if local.date() != target_date or local.time().replace(tzinfo=None) < _CN_CLOSE:
        raise ValueError("cutoff_timestamp must be on target_date at or after CN close")
    return value


def _group_kline_payload(
    payload: object,
    *,
    requested_symbols: Sequence[str],
) -> dict[str, tuple[Mapping[str, object], ...]]:
    payload = _unwrap_kline_envelope(payload)
    if isinstance(payload, Mapping):
        records: list[object] = [payload]
    elif isinstance(payload, list):
        records = list(payload)
    else:
        raise DataQualityError("Infoway 日K数据结构异常。")
    if any(not isinstance(item, Mapping) for item in records):
        raise DataQualityError("Infoway 日K数据结构异常。")

    requested = set(requested_symbols)
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for item in records:
        raw_symbol = item.get("s", item.get("symbol"))
        if not isinstance(raw_symbol, str):
            raise DataQualityError("Infoway 日K响应缺少证券代码。")
        symbol = raw_symbol.strip().upper()
        if symbol not in requested:
            raise DataQualityError("Infoway 日K返回未请求的证券代码。")

        if "respList" in item:
            nested = item["respList"]
            if not isinstance(nested, list) or any(not isinstance(bar, Mapping) for bar in nested):
                raise DataQualityError("Infoway 日K respList结构异常。")
            grouped.setdefault(symbol, []).extend(nested)
        else:
            grouped.setdefault(symbol, []).append(item)
    return {symbol: tuple(bars) for symbol, bars in grouped.items()}


def _unwrap_kline_envelope(payload: object) -> object:
    """Remove provider envelopes without inventing a missing symbol identity."""

    current = payload
    for _ in range(3):
        if not isinstance(current, Mapping):
            return current
        if "s" in current or "symbol" in current:
            return current
        if "data" in current:
            current = current["data"]
            continue
        if "respList" in current:
            nested = current["respList"]
            if not isinstance(nested, list) or any(
                not isinstance(item, Mapping) for item in nested
            ):
                raise DataQualityError("Infoway 日K respList包装结构异常。")
            # A wrapper may contain symbol-level records.  Bare bars have no
            # independently verifiable identity and remain a quality error.
            if nested and all("s" not in item and "symbol" not in item for item in nested):
                raise DataQualityError("Infoway 日K包装缺少可验证的证券代码。")
            current = nested
            continue
        return current
    raise DataQualityError("Infoway 日K包装层级异常。")


def _prepare_target_bar(
    provider_symbol: str,
    bars: Sequence[Mapping[str, object]],
    *,
    target_date: date,
    contract: InfowayEodUnitContract,
) -> _PreparedTargetBar | None:
    dated: list[tuple[int, date, Mapping[str, object]]] = []
    seen_timestamps: set[int] = set()
    for bar in bars:
        timestamp = _parse_unix_seconds(bar.get("t"))
        if timestamp in seen_timestamps:
            raise DataQualityError(f"Infoway {provider_symbol} 日K包含重复时间戳。")
        seen_timestamps.add(timestamp)
        bar_date = datetime.fromtimestamp(timestamp, tz=_CN_TIMEZONE).date()
        if bar_date > target_date:
            raise DataQualityError(f"Infoway {provider_symbol} 日K晚于目标日期，截止时间未生效。")
        dated.append((timestamp, bar_date, bar))

    targets = [item for item in dated if item[1] == target_date]
    if not targets:
        # A suspended stock commonly returns its last traded daily bars.  No
        # stale bar is relabelled; it simply remains missing for target_date.
        return None
    if len(targets) != 1:
        raise DataQualityError(f"Infoway {provider_symbol} 日K目标日期不符，拒绝用其他日期替代。")
    target_timestamp, _, target = targets[0]

    volume_raw = _nonnegative_decimal(target.get("v"), "volume", provider_symbol)
    amount_keys = [
        field for field in ("vw", "vm") if field in target and target.get(field) is not None
    ]
    if len(amount_keys) > 1:
        raise DataQualityError(f"Infoway {provider_symbol} 日K同时包含vw与vm，成交额语义冲突。")
    if amount_keys != [contract.amount_field]:
        raise DataQualityError(f"Infoway {provider_symbol} 日K成交额字段与已验证单位合同不一致。")
    amount_raw = _nonnegative_decimal(target.get(contract.amount_field), "amount", provider_symbol)
    volume_shares = volume_raw * contract.volume_multiplier_to_shares
    if volume_shares == 0 and amount_raw == 0:
        # Some vendors emit a same-date placeholder for suspended securities.
        # Its price fields are not a tradable daily bar and must not enter the
        # research history, even if they repeat the previous close.
        return None
    if volume_shares == 0 or amount_raw == 0:
        raise DataQualityError(f"Infoway {provider_symbol} 成交量与成交额零值不一致，数据损坏。")

    previous = [item for item in dated if item[0] < target_timestamp]
    if not previous:
        if len(dated) == 1:
            # A newly listed security may legitimately have only its first
            # completed daily bar.  It cannot supply a verified prev_close, so
            # omit it and let the full-market coverage gate handle the gap.
            return None
        raise DataQualityError(f"Infoway {provider_symbol} 日K缺少可验证的前收盘价。")
    previous_bar = max(previous, key=lambda item: item[0])[2]

    open_price = _positive_decimal(target.get("o"), "open", provider_symbol)
    high = _positive_decimal(target.get("h"), "high", provider_symbol)
    low = _positive_decimal(target.get("l"), "low", provider_symbol)
    close = _positive_decimal(target.get("c"), "close", provider_symbol)
    previous_close = _positive_decimal(previous_bar.get("c"), "prev_close", provider_symbol)
    if high < max(open_price, low, close) or low > min(open_price, high, close):
        raise DataQualityError(f"Infoway {provider_symbol} 日K开高低收关系无效。")

    return _PreparedTargetBar(
        provider_symbol=provider_symbol,
        target_date=target_date,
        open_price=open_price,
        high=high,
        low=low,
        close=close,
        previous_close=previous_close,
        volume_shares=volume_shares,
        amount_raw=amount_raw,
    )


def _normalize_prepared_bar(
    prepared: _PreparedTargetBar,
    *,
    amount_multiplier_to_cny: Decimal,
    retrieved_at: datetime,
    asset_kind: AssetKind,
) -> dict[str, object]:
    code = _PROVIDER_SYMBOL.fullmatch(prepared.provider_symbol)
    if code is None:  # Already validated; keep this branch fail-closed.
        raise DataQualityError("Infoway 日K证券代码无法标准化。")
    amount_cny = prepared.amount_raw * amount_multiplier_to_cny
    return {
        "symbol": code.group("code"),
        "trade_date": prepared.target_date,
        "open": float(prepared.open_price),
        "high": float(prepared.high),
        "low": float(prepared.low),
        "close": float(prepared.close),
        "prev_close": float(prepared.previous_close),
        "volume_shares": float(prepared.volume_shares),
        "amount_cny": float(amount_cny),
        "turnover_pct": math.nan,
        "source": _SOURCE_LABELS[asset_kind],
        "retrieved_at": retrieved_at.isoformat().replace("+00:00", "Z"),
    }


def _parse_unix_seconds(value: object) -> int:
    if isinstance(value, bool):
        raise DataQualityError("Infoway 日K时间戳无效。")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"\d{1,10}", value):
        parsed = int(value)
    else:
        raise DataQualityError("Infoway 日K时间戳必须是Unix秒。")
    try:
        datetime.fromtimestamp(parsed, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise DataQualityError("Infoway 日K时间戳超出有效范围。") from exc
    return parsed


def _positive_decimal(value: object, field: str, symbol: str) -> Decimal:
    parsed = _decimal(value, field, symbol)
    if parsed <= 0:
        raise DataQualityError(f"Infoway {symbol} 日K {field}必须大于0。")
    return parsed


def _nonnegative_decimal(value: object, field: str, symbol: str) -> Decimal:
    parsed = _decimal(value, field, symbol)
    if parsed < 0:
        raise DataQualityError(f"Infoway {symbol} 日K {field}不能为负。")
    return parsed


def _decimal(value: object, field: str, symbol: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise DataQualityError(f"Infoway {symbol} 日K {field}缺失或非数值。")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise DataQualityError(f"Infoway {symbol} 日K {field}缺失或非数值。") from exc
    if not parsed.is_finite():
        raise DataQualityError(f"Infoway {symbol} 日K {field}不是有限数值。")
    return parsed


def _amount_volume_is_consistent(
    *,
    volume_shares: Decimal,
    amount_cny: Decimal,
    low: Decimal,
    high: Decimal,
) -> bool:
    if volume_shares <= 0 or amount_cny <= 0:
        return False
    implied_price = amount_cny / volume_shares
    tolerance = Decimal("0.005")
    return not (
        implied_price < low * (Decimal(1) - tolerance)
        or implied_price > high * (Decimal(1) + tolerance)
    )
