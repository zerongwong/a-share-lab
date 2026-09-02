"""Strict single-session Tushare Pro daily cross-section adapter.

This module is intentionally lower level than :class:`DailyIncrementPort`.
It performs one bounded retry sequence for ``daily(trade_date=...)`` and
converts the provider's unadjusted A-share rows into the canonical daily
schema.  Trading calendars, security masters, credentials and scheduling
remain the caller's responsibility.
"""

from __future__ import annotations

import importlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from types import ModuleType

import numpy as np
import pandas as pd

from ashare_lab.domain.data_sources import SourceId
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS

TUSHARE_DAILY_ROW_LIMIT = 6_000
TUSHARE_DAILY_SOURCE = "tushare:daily_unadjusted:stocks"
TUSHARE_DAILY_RETRY_DELAYS_SECONDS = (2.0, 5.0)

_PROVIDER_CODE = re.compile(r"^(?P<symbol>\d{6})\.(?:SH|SZ|BJ)$")
_REQUIRED_COLUMNS = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
)
_PRICE_COLUMNS = ("open", "high", "low", "close", "prev_close")


def _default_client_factory(token: str) -> object:
    try:
        module: ModuleType = importlib.import_module("tushare")
    except ImportError:
        raise DataUnavailableError("未安装 Tushare SDK，无法读取免费日线。") from None
    factory = getattr(module, "pro_api", None)
    if not callable(factory):
        raise DataUnavailableError("当前 Tushare SDK 不提供 Pro API 客户端。")
    return factory(token)


@dataclass(frozen=True, slots=True)
class TushareDailyFetch:
    """Auditable receipt for one date-only Tushare request.

    ``received_symbols`` describes only what the free ``daily`` endpoint
    returned.  There is deliberately no ``requested_symbols`` field because
    this endpoint does not provide an independent security master from which
    full-market coverage could be proven.
    """

    frame: pd.DataFrame
    requested_trade_date: date
    received_trade_dates: tuple[date, ...]
    received_symbols: tuple[str, ...]
    fetched_at: datetime
    trace_ids: tuple[str, ...]
    provider: str = SourceId.TUSHARE.value


class TushareDailyClient:
    """Fetch and normalize one full-market, unadjusted A-share daily table.

    A ready Pro client can be injected by tests or a higher-level composition
    root.  Otherwise a token is passed once to Tushare's client factory and is
    deliberately not retained by this adapter.
    """

    provider = SourceId.TUSHARE.value

    def __init__(
        self,
        token: str | None = None,
        *,
        client: object | None = None,
        client_factory: Callable[[str], object] | None = None,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if client is not None:
            if token is not None or client_factory is not None:
                raise ValueError("client 与 token/client_factory 不能同时提供")
            self._client = client
        else:
            resolved_token = str(token or "").strip()
            if not resolved_token:
                raise ValueError("Tushare token 不能为空")
            factory = client_factory or _default_client_factory
            try:
                self._client = factory(resolved_token)
            except DataUnavailableError:
                raise
            except Exception:
                # Provider exceptions may echo credentials or request details.
                raise DataUnavailableError("Tushare 客户端初始化失败，原始错误已脱敏。") from None
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleeper = sleeper or time.sleep

    def fetch_daily(self, trade_date: date) -> TushareDailyFetch:
        """Return one canonical cross-section after a bounded same-source retry.

        Only provider unavailability and an empty response are retryable.
        Schema, unit, identity and price-quality failures still fail closed on
        the first observation and can never be hidden by a later response.
        """

        target_date = _validate_target_date(trade_date)
        endpoint = getattr(self._client, "daily", None)
        if not callable(endpoint):
            raise DataUnavailableError("Tushare Pro 客户端没有 daily 日线接口。")
        attempt_count = len(TUSHARE_DAILY_RETRY_DELAYS_SECONDS) + 1
        for attempt in range(attempt_count):
            try:
                raw = endpoint(trade_date=target_date.strftime("%Y%m%d"))
            except Exception:
                # Never propagate provider text: it may contain a token or
                # request.  A bounded retry remains on the same provider and
                # does not weaken the cross-source verification contract.
                if attempt + 1 == attempt_count:
                    raise DataUnavailableError(
                        "Tushare 日线请求失败，有限重试后仍不可用，原始错误已脱敏。"
                    ) from None
                self._sleeper(TUSHARE_DAILY_RETRY_DELAYS_SECONDS[attempt])
                continue

            fetched_at = _aware_utc(self._clock())
            try:
                frame = normalize_tushare_daily(
                    raw,
                    target_date=target_date,
                    retrieved_at=fetched_at,
                )
            except DataUnavailableError:
                if attempt + 1 == attempt_count:
                    raise
                self._sleeper(TUSHARE_DAILY_RETRY_DELAYS_SECONDS[attempt])
                continue
            return TushareDailyFetch(
                frame=frame,
                requested_trade_date=target_date,
                received_trade_dates=(target_date,),
                received_symbols=tuple(frame["symbol"].astype(str)),
                fetched_at=fetched_at,
                # The Tushare Python SDK's DataFrame response exposes no
                # provider trace identifier.  An empty tuple is honest and
                # auditable; do not invent a local value that could be
                # mistaken for a provider trace.
                trace_ids=(),
            )
        raise AssertionError("bounded Tushare retry loop exhausted unexpectedly")


# Compatibility name for code that describes adapters as market-data objects.
# Neither name implements DailyIncrementPort: the free endpoint has no calendar,
# independent requested universe or core-index surface.
TushareDailyMarketData = TushareDailyClient


def normalize_tushare_daily(
    raw: pd.DataFrame,
    *,
    target_date: date,
    retrieved_at: datetime,
) -> pd.DataFrame:
    """Convert one Tushare ``daily`` result without inferring missing fields."""

    target = _validate_target_date(target_date)
    retrieved = _utc_iso(retrieved_at)
    if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
        raise DataUnavailableError(f"Tushare 在 {target.isoformat()} 没有返回日线数据。")
    if not isinstance(raw, pd.DataFrame):
        raise DataQualityError("Tushare 日线响应不是 DataFrame。")
    if isinstance(raw.columns, pd.MultiIndex):
        raise DataQualityError("Tushare 日线响应含多层列，字段合同不明确。")
    if len(raw) >= TUSHARE_DAILY_ROW_LIMIT:
        raise DataQualityError(
            "Tushare 日线响应达到 6000 行接口上限，可能已截断，拒绝进入研究。"
        )

    missing = set(_REQUIRED_COLUMNS).difference(raw.columns)
    if missing:
        raise DataQualityError("Tushare 日线缺少字段：" + "、".join(sorted(missing)))

    provider_codes = raw["ts_code"].astype("string").str.strip().str.upper()
    valid_codes = provider_codes.str.fullmatch(_PROVIDER_CODE).fillna(False)
    if not bool(valid_codes.all()):
        raise DataQualityError(
            f"Tushare 日线含 {int((~valid_codes).sum())} 行无效 A 股代码。"
        )
    if bool(provider_codes.duplicated().any()):
        raise DataQualityError("Tushare 日线含重复证券代码。")
    symbols = provider_codes.str.extract(_PROVIDER_CODE, expand=True)["symbol"]
    if bool(symbols.duplicated().any()):
        raise DataQualityError("Tushare 日线去除市场后缀后证券代码重复。")

    date_text = raw["trade_date"].astype("string").str.strip()
    dates = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
    if bool(dates.isna().any()):
        raise DataQualityError("Tushare 日线含无效 trade_date。")
    if not bool((dates.dt.date == target).all()):
        raise DataQualityError("Tushare 日线混入目标交易日以外的数据。")

    numeric: dict[str, pd.Series] = {}
    for column in (
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ):
        values = pd.to_numeric(raw[column], errors="coerce")
        finite = pd.Series(np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)), index=raw.index)
        if bool(values.isna().any()) or not bool(finite.all()):
            raise DataQualityError(f"Tushare 日线字段 {column} 含空值或非有限数。")
        numeric[column] = values.astype(float)

    output = pd.DataFrame(index=raw.index)
    output["symbol"] = symbols.astype(str)
    output["trade_date"] = pd.Timestamp(target)
    for source, destination in (
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("pre_close", "prev_close"),
    ):
        output[destination] = numeric[source]

    if bool((output[list(_PRICE_COLUMNS)] <= 0).any(axis=None)):
        raise DataQualityError("Tushare 日线 OHLC 与 pre_close 必须为正数。")
    if bool((output["high"] < output[["open", "low", "close"]].max(axis=1)).any()):
        raise DataQualityError("Tushare 日线最高价低于开盘、最低或收盘价。")
    if bool((output["low"] > output[["open", "high", "close"]].min(axis=1)).any()):
        raise DataQualityError("Tushare 日线最低价高于开盘、最高或收盘价。")

    # ``pre_close`` is the exchange reference close and can be adjusted on an
    # ex-right/ex-dividend session.  Do not compare it with the preceding raw
    # close from another history endpoint.  Instead, require Tushare's same-row
    # change and percentage fields to agree with the published reference close
    # at their documented display precision.
    derived_change = output["close"] - output["prev_close"]
    if not bool(
        np.isclose(
            derived_change.to_numpy(dtype=float),
            numeric["change"].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.011,
        ).all()
    ):
        raise DataQualityError("Tushare 日线昨收与涨跌额不一致。")
    derived_pct = derived_change / output["prev_close"] * 100.0
    if not bool(
        np.isclose(
            derived_pct.to_numpy(dtype=float),
            numeric["pct_chg"].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.011,
        ).all()
    ):
        raise DataQualityError("Tushare 日线昨收与涨跌幅不一致。")

    volume_lots = numeric["vol"]
    amount_thousand_cny = numeric["amount"]
    if bool((volume_lots < 0).any()) or bool((amount_thousand_cny < 0).any()):
        raise DataQualityError("Tushare 日线成交量与成交额不能为负数。")
    volume_shares = volume_lots * 100.0
    rounded_shares = volume_shares.round()
    if not bool(np.isclose(volume_shares, rounded_shares, rtol=0.0, atol=1e-6).all()):
        raise DataQualityError("Tushare vol 按手换算后不是整数股，单位合同可能变化。")
    output["volume_shares"] = rounded_shares.astype("Int64")
    output["amount_cny"] = amount_thousand_cny * 1_000.0
    output["turnover_pct"] = float("nan")

    _validate_amount_volume_consistency(output)
    output["source"] = TUSHARE_DAILY_SOURCE
    output["retrieved_at"] = retrieved
    output = output.loc[:, ["symbol", *CANONICAL_DAILY_COLUMNS]]
    output = output.sort_values("symbol").reset_index(drop=True)
    output.attrs.update(
        {
            "provider": SourceId.TUSHARE.value,
            "target_date": target.isoformat(),
            "adjustment": "none",
            "raw_row_count": len(raw),
            "provider_row_limit": TUSHARE_DAILY_ROW_LIMIT,
            "unit_contract": "vol=100_shares;amount=1000_cny",
        }
    )
    return output


def _validate_amount_volume_consistency(frame: pd.DataFrame) -> None:
    volume = frame["volume_shares"].astype(float)
    amount = frame["amount_cny"].astype(float)
    if bool(((volume == 0) != (amount == 0)).any()):
        raise DataQualityError("Tushare 日线零成交量与零成交额状态不一致。")
    traded = volume > 0
    if not bool(traded.any()):
        return
    lower = frame.loc[traded, "low"] * volume.loc[traded]
    upper = frame.loc[traded, "high"] * volume.loc[traded]
    observed = amount.loc[traded]
    # Tushare publishes amount in thousands of CNY, so one CNY of tolerance
    # covers provider rounding after conversion without masking a unit change.
    if bool(((observed + 1.0 < lower) | (observed - 1.0 > upper)).any()):
        raise DataQualityError("Tushare 成交额/成交量换算后的均价超出当日价格区间。")


def _validate_target_date(value: date) -> date:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError("trade_date 必须是 date，不能是 datetime 或字符串")
    return value


def _utc_iso(value: datetime) -> str:
    return _aware_utc(value).isoformat().replace("+00:00", "Z")


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DataQualityError("retrieved_at 必须是带时区的 datetime。")
    return value.astimezone(UTC)


__all__ = [
    "TUSHARE_DAILY_ROW_LIMIT",
    "TUSHARE_DAILY_RETRY_DELAYS_SECONDS",
    "TUSHARE_DAILY_SOURCE",
    "TushareDailyClient",
    "TushareDailyFetch",
    "TushareDailyMarketData",
    "normalize_tushare_daily",
]
