"""Infoway HTTP adapter for personal, non-commercial A-share research.

The adapter intentionally has no browser/session integration and never stores
the API key.  A caller supplies the key from a local secret store or process
environment.  Errors are sanitized so credentials cannot appear in logs.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.realtime_market_data import CurrentFundamentalsSnapshot, RealtimeBatch

_MAX_BATCH = 100
_FUNDAMENTALS_MAX_BATCH = 500
_FREE_MIN_INTERVAL_SECONDS = 1.0
_SAFE_SYMBOL = re.compile(r"^[A-Z0-9._-]{1,32}$")


class _RateLimiter:
    def __init__(
        self,
        minimum_interval: float,
        *,
        clock: Callable[[], float],
        sleeper: Callable[[float], None],
    ) -> None:
        if minimum_interval < 0:
            raise ValueError("minimum_interval cannot be negative")
        self.minimum_interval = minimum_interval
        self.clock = clock
        self.sleeper = sleeper
        self._last_request_at: float | None = None

    def wait(self) -> None:
        now = self.clock()
        if self._last_request_at is not None:
            remaining = self.minimum_interval - (now - self._last_request_at)
            if remaining > 0:
                self.sleeper(remaining)
                now = self.clock()
        self._last_request_at = now


class InfowayRealtimeMarketData:
    """Small, rate-limited client for Infoway's documented A-share endpoints."""

    provider = "infoway"

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        base_url: str = "https://data.infoway.io",
        minimum_interval_seconds: float = _FREE_MIN_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        utcnow: Callable[[], datetime] | None = None,
    ) -> None:
        normalized_key = api_key.strip()
        if not normalized_key:
            raise ValueError("Infoway API key is required")
        if not base_url.startswith("https://"):
            raise ValueError("Infoway base_url must use HTTPS")
        self._api_key = normalized_key
        self._client = client or httpx.Client(timeout=httpx.Timeout(20.0))
        self._owns_client = client is None
        self._base_url = base_url.rstrip("/")
        self._limiter = _RateLimiter(
            minimum_interval_seconds,
            clock=clock,
            sleeper=sleeper,
        )
        self._utcnow = utcnow or (lambda: datetime.now(UTC))

    @classmethod
    def from_environment(cls, variable: str = "INFOWAY_API_KEY", **kwargs: Any):
        """Build from a process-local secret without ever printing its value."""

        value = os.environ.get(variable, "")
        if not value.strip():
            raise DataUnavailableError(
                f"本机尚未配置 {variable}。请在本机私密配置中保存，勿发送到聊天。"
            )
        return cls(value, **kwargs)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_symbol_list(self) -> tuple[dict[str, Any], ...]:
        payload, _ = self._request_json(
            "GET",
            "/common/basic/symbols",
            params={"type": "STOCK_CN"},
        )
        records = _as_record_list(payload, "symbol_list")
        _require_unique_symbols(records, "symbol_list")
        return tuple(records)

    def fetch_symbol_fundamentals(
        self,
        symbols: Sequence[str],
    ) -> CurrentFundamentalsSnapshot:
        """Fetch Infoway's current Symbol Fundamentals snapshot.

        The endpoint exposes current symbol metadata plus values such as EPS,
        EPS TTM, BPS and dividend yield.  It does not expose the disclosure
        timeline required for point-in-time historical research, and this
        method deliberately leaves those provider values untransformed.
        """

        normalized = _normalize_symbols(symbols)
        records: list[dict[str, Any]] = []
        traces: list[str] = []
        for batch in _chunks(normalized, _FUNDAMENTALS_MAX_BATCH):
            payload, trace_id = self._request_json(
                "GET",
                "/common/basic/symbols/info",
                params={"type": "STOCK_CN", "symbols": ",".join(batch)},
            )
            batch_records = _as_record_list(payload, "symbol_fundamentals")
            _validate_symbol_records(
                batch_records,
                "symbol_fundamentals",
                requested_symbols=batch,
            )
            records.extend(batch_records)
            if trace_id:
                traces.append(trace_id)

        _validate_symbol_records(
            records,
            "symbol_fundamentals",
            requested_symbols=normalized,
        )
        received = tuple(str(record["symbol"]).upper() for record in records)
        return CurrentFundamentalsSnapshot(
            records=tuple(records),
            requested_symbols=normalized,
            received_symbols=received,
            fetched_at=self._utcnow(),
            trace_ids=tuple(traces),
            provider=self.provider,
        )

    def fetch_recent_klines(
        self,
        symbols: Sequence[str],
        *,
        interval: int = 1,
        count: int = 2,
    ) -> RealtimeBatch:
        if interval not in range(1, 13):
            raise ValueError("interval must be one of Infoway kline types 1..12")
        if not 1 <= count <= 500:
            raise ValueError("count must be between 1 and 500")
        normalized = _normalize_symbols(symbols)
        if len(normalized) > 1 and count > 2:
            raise ValueError("Infoway multi-symbol requests support at most 2 bars per symbol")

        records: list[dict[str, Any]] = []
        traces: list[str] = []
        for batch in _chunks(normalized, _MAX_BATCH):
            payload, trace_id = self._request_json(
                "POST",
                "/stock/v2/batch_kline",
                json={"klineType": interval, "klineNum": count, "codes": ",".join(batch)},
            )
            records.extend(_as_record_list(payload, "kline"))
            if trace_id:
                traces.append(trace_id)
        return self._build_batch("kline", normalized, records, traces)

    def fetch_latest_trades(self, symbols: Sequence[str]) -> RealtimeBatch:
        return self._fetch_path_batches("latest_trade", "/stock/batch_trade", symbols)

    def fetch_depth(self, symbols: Sequence[str]) -> RealtimeBatch:
        return self._fetch_path_batches("market_depth", "/stock/batch_depth", symbols)

    def _fetch_path_batches(
        self,
        dataset: str,
        endpoint: str,
        symbols: Sequence[str],
    ) -> RealtimeBatch:
        normalized = _normalize_symbols(symbols)
        records: list[dict[str, Any]] = []
        traces: list[str] = []
        for batch in _chunks(normalized, _MAX_BATCH):
            encoded = quote(",".join(batch), safe=",._-")
            payload, trace_id = self._request_json("GET", f"{endpoint}/{encoded}")
            records.extend(_as_record_list(payload, dataset))
            if trace_id:
                traces.append(trace_id)
        return self._build_batch(dataset, normalized, records, traces)

    def _build_batch(
        self,
        dataset: str,
        requested: tuple[str, ...],
        records: list[dict[str, Any]],
        traces: list[str],
    ) -> RealtimeBatch:
        received: list[str] = []
        for record in records:
            symbol = record.get("s", record.get("symbol"))
            if isinstance(symbol, str) and symbol:
                received.append(symbol.upper())
        return RealtimeBatch(
            dataset=dataset,
            records=tuple(records),
            requested_symbols=requested,
            received_symbols=tuple(dict.fromkeys(received)),
            fetched_at=self._utcnow(),
            trace_ids=tuple(traces),
            provider=self.provider,
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> tuple[Any, str]:
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
            if status == 429:
                raise DataUnavailableError("Infoway 请求达到套餐限额，请稍后重试。") from exc
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


def _normalize_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols))
    if not normalized:
        raise ValueError("at least one symbol is required")
    invalid = [symbol for symbol in normalized if not _SAFE_SYMBOL.fullmatch(symbol)]
    if invalid:
        raise ValueError("symbol contains unsupported characters")
    return normalized


def _chunks(values: Sequence[str], size: int) -> Iterable[tuple[str, ...]]:
    for index in range(0, len(values), size):
        yield tuple(values[index : index + size])


def _as_record_list(payload: Any, dataset: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise DataQualityError(f"Infoway {dataset} 数据结构异常。")
    return list(payload)


def _require_unique_symbols(records: Sequence[dict[str, Any]], dataset: str) -> None:
    symbols = [item.get("symbol") for item in records]
    valid = [item for item in symbols if isinstance(item, str) and item]
    if len(valid) != len(records) or len(set(valid)) != len(valid):
        raise DataQualityError(f"Infoway {dataset} 缺少或重复证券代码。")


def _validate_symbol_records(
    records: Sequence[dict[str, Any]],
    dataset: str,
    *,
    requested_symbols: Sequence[str],
) -> None:
    """Validate identity without guessing optional provider field types."""

    symbols = [record.get("symbol") for record in records]
    if any(
        not isinstance(symbol, str) or not symbol or not _SAFE_SYMBOL.fullmatch(symbol.upper())
        for symbol in symbols
    ):
        raise DataQualityError(f"Infoway {dataset} 缺少或包含无效证券代码。")
    normalized = [str(symbol).upper() for symbol in symbols]
    if len(set(normalized)) != len(normalized):
        raise DataQualityError(f"Infoway {dataset} 包含重复证券代码。")
    requested = set(requested_symbols)
    if not set(normalized).issubset(requested):
        raise DataQualityError(f"Infoway {dataset} 返回了未请求的证券代码。")
