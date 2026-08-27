from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import pytest

from ashare_lab.adapters.infoway_eod import (
    InfowayEodMarketData,
    InfowayEodUnitContract,
)
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.daily_increment import DailyIncrementPort
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS

CN = ZoneInfo("Asia/Shanghai")
TARGET = date(2026, 8, 26)
NOW = datetime(2026, 8, 27, 8, 0, tzinfo=UTC)


def _timestamp(value: date, at: time = time(15, 0)) -> int:
    return int(datetime.combine(value, at, tzinfo=CN).timestamp())


def _units(
    *,
    amount_field: str = "vw",
    volume_multiplier: str = "1",
    amount_multiplier: str = "1",
) -> InfowayEodUnitContract:
    return InfowayEodUnitContract(
        volume_multiplier_to_shares=Decimal(volume_multiplier),
        amount_field=amount_field,
        amount_multiplier_to_cny=Decimal(amount_multiplier),
        verified_reference="account-data-dictionary-2026-08-27",
    )


def _calendar_response(*days: date) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "ret": 200,
            "traceId": "calendar-trace",
            "data": {
                "trade_days": [value.strftime("%Y%m%d") for value in days],
                "half_trade_days": [],
            },
        },
    )


def _bar(
    value: date,
    *,
    close: str,
    amount_field: str = "vw",
    amount: str = "10500",
) -> dict[str, str]:
    return {
        "t": str(_timestamp(value)),
        "o": "10.20",
        "h": "11.00",
        "l": "10.00",
        "c": close,
        "v": "1000",
        amount_field: amount,
    }


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _api(handler, **kwargs) -> InfowayEodMarketData:
    return InfowayEodMarketData(
        "eod-private-secret",
        unit_contract=_units(),
        client=_client(handler),
        minimum_interval_seconds=0,
        utcnow=lambda: NOW,
        **kwargs,
    )


def test_calendar_uses_cn_range_and_merges_half_days() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(
            200,
            json={
                "ret": 200,
                "data": {
                    "trade_days": ["20260824", "20260825"],
                    "half_trade_days": ["20260826"],
                },
            },
        )

    api = _api(handler)
    result = api.fetch_cn_trading_days(date(2026, 8, 24), TARGET)

    assert result == (date(2026, 8, 24), date(2026, 8, 25), TARGET)
    assert seen == {"market": "CN", "beginDay": "20260824", "endDay": "20260826"}
    assert isinstance(api, DailyIncrementPort)


@pytest.mark.parametrize(
    "data, match",
    [
        ({"trade_days": ["2026-08-26"], "half_trade_days": []}, "无效日期"),
        ({"trade_days": ["20260826", "20260826"], "half_trade_days": []}, "重复"),
        ({"trade_days": ["20260827"], "half_trade_days": []}, "区间外"),
        ({"trade_days": ["20260826"], "half_trade_days": ["20260826"]}, "重复"),
    ],
)
def test_calendar_rejects_ambiguous_or_out_of_range_values(
    data: dict[str, list[str]],
    match: str,
) -> None:
    api = _api(lambda _: httpx.Response(200, json={"ret": 200, "data": data}))
    with pytest.raises(DataQualityError, match=match):
        api.fetch_cn_trading_days(TARGET, TARGET)


def test_symbol_list_returns_only_consistent_sse_szse_stock_codes() -> None:
    api = _api(
        lambda _: httpx.Response(
            200,
            json={
                "ret": 200,
                "data": [
                    {"symbol": "600000.SH", "exchange": "SH", "index": False},
                    {"symbol": "000001.SZ", "exchange": "SZ", "index": False},
                    {"symbol": "600001.SH", "exchange": "SSE", "index": False},
                    {"symbol": "000002.SZ", "exchange": "SZSE", "index": False},
                    {"symbol": "600002.SH", "index": False},
                    {"symbol": "689009.SH", "exchange": "SH", "index": False},
                    {"symbol": "302132.SZ", "exchange": "SZSE", "index": False},
                    {"symbol": "200011.SZ", "exchange": "SZSE", "index": False},
                    {"symbol": "201872.SZ", "exchange": "SZ", "index": False},
                    {"symbol": "899999.BJ", "exchange": "BSE", "index": False},
                    {"symbol": "000001.SH", "index": True},
                    {"symbol": "000300.SH", "exchange": "SSE", "index": True},
                    {"symbol": "399001.SZ", "exchange": "SZ", "index": True},
                ],
            },
        )
    )

    assert api.fetch_cn_stock_symbols() == (
        "000001.SZ",
        "000002.SZ",
        "302132.SZ",
        "600000.SH",
        "600001.SH",
        "600002.SH",
        "689009.SH",
    )


def test_symbol_list_rejects_exchange_suffix_conflict() -> None:
    api = _api(
        lambda _: httpx.Response(
            200,
            json={
                "ret": 200,
                "data": [{"symbol": "600000.SH", "exchange": "SZSE", "index": False}],
            },
        )
    )
    with pytest.raises(DataQualityError, match="冲突"):
        api.fetch_cn_stock_symbols()


@pytest.mark.parametrize(
    "record", [{"symbol": "600000.SH"}, {"symbol": "600000.SH", "index": "false"}]
)
def test_symbol_list_requires_explicit_boolean_index_flag(record: dict[str, object]) -> None:
    api = _api(
        lambda _: httpx.Response(
            200,
            json={"ret": 200, "data": [record]},
        )
    )

    with pytest.raises(DataQualityError, match="index字段"):
        api.fetch_cn_stock_symbols()


def test_daily_increment_batches_100_and_normalizes_nested_response() -> None:
    kline_requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trading_days"):
            return _calendar_response(TARGET)
        assert request.url.path == "/stock/v2/batch_kline"
        body = json.loads(request.content)
        kline_requests.append(body)
        records = []
        for symbol in body["codes"].split(","):
            records.append(
                {
                    "s": symbol,
                    # The endpoint commonly returns newest-first.  The adapter
                    # must order by timestamp rather than response position.
                    "respList": [
                        _bar(TARGET, close="10.50"),
                        _bar(date(2026, 8, 25), close="10.10", amount="10100"),
                    ],
                }
            )
        return httpx.Response(
            200,
            json={"ret": 200, "traceId": f"trace-{len(kline_requests)}", "data": records},
        )

    symbols = tuple(f"{number:06d}.SZ" for number in range(205))
    result = _api(handler).fetch_daily_increment(symbols, TARGET)

    assert len(kline_requests) == 3
    assert [len(request["codes"].split(",")) for request in kline_requests] == [100, 100, 5]
    assert all(request["klineType"] == 8 for request in kline_requests)
    assert all(request["klineNum"] == 2 for request in kline_requests)
    assert all(
        request["timestamp"] == _timestamp(TARGET, time(23, 59, 59)) for request in kline_requests
    )
    assert result.trace_ids == ("trace-1", "trace-2", "trace-3")
    assert result.coverage_ratio == 1.0
    assert list(result.frame.columns) == ["symbol", *CANONICAL_DAILY_COLUMNS]
    assert len(result.frame) == 205
    first = result.frame.iloc[0]
    assert first["symbol"] == "000000"
    assert first["trade_date"] == TARGET
    assert first["prev_close"] == pytest.approx(10.1)
    assert first["volume_shares"] == pytest.approx(1000)
    assert first["amount_cny"] == pytest.approx(10_500)
    assert first["source"] == "infoway:eod_unadjusted:stocks"
    assert first["retrieved_at"] == "2026-08-27T08:00:00Z"
    assert pd.isna(first["turnover_pct"])


def test_daily_increment_accepts_flat_records_and_explicit_cutoff() -> None:
    cutoff = _timestamp(TARGET, time(16, 0))
    seen_cutoff: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trading_days"):
            return _calendar_response(TARGET)
        body = json.loads(request.content)
        seen_cutoff.append(body["timestamp"])
        return httpx.Response(
            200,
            json={
                "ret": 200,
                "data": [
                    {"s": "600000.SH", **_bar(date(2026, 8, 25), close="10.10")},
                    {"s": "600000.SH", **_bar(TARGET, close="10.50")},
                ],
            },
        )

    result = _api(handler).fetch_daily_increment(["600000.SH"], TARGET, cutoff_timestamp=cutoff)

    assert seen_cutoff == [cutoff]
    assert result.received_symbols == ("600000.SH",)
    assert result.frame.loc[0, "symbol"] == "600000"


def test_realistic_index_scale_uses_explicit_kind_and_skips_stock_vwap_check() -> None:
    previous = {
        "t": str(_timestamp(date(2026, 8, 25))),
        "o": "4500.00",
        "h": "4550.00",
        "l": "4480.00",
        "c": "4520.00",
        "v": "170000000",
        "vw": "450000000000.00",
    }
    target = {
        "t": str(_timestamp(TARGET)),
        "o": "4542.00",
        "h": "4612.00",
        "l": "4530.00",
        "c": "4600.00",
        "v": "184578169",
        "vw": "489057278414.77",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trading_days"):
            return _calendar_response(TARGET)
        return httpx.Response(
            200,
            json={
                "ret": 200,
                "data": [{"s": "000300.SH", "respList": [target, previous]}],
            },
        )

    api = InfowayEodMarketData(
        "index-secret",
        unit_contract=_units(volume_multiplier="100"),
        client=_client(handler),
        minimum_interval_seconds=0,
        utcnow=lambda: NOW,
    )
    index_result = api.fetch_daily_increment(["000300.SH"], TARGET, asset_kind="indices")

    row = index_result.frame.iloc[0]
    assert row["volume_shares"] == pytest.approx(18_457_816_900)
    assert row["amount_cny"] == pytest.approx(489_057_278_414.77)
    assert row["high"] == pytest.approx(4612)
    assert row["source"] == "infoway:eod_unadjusted:indices"

    # The same suffix does not silently identify an index.  The default stock
    # path retains the strict amount/volume-implied-price validation.
    with pytest.raises(DataQualityError, match="单位不可验证"):
        api.fetch_daily_increment(["000300.SH"], TARGET)


def test_invalid_asset_kind_fails_before_network() -> None:
    called = False

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    api = _api(handler)
    with pytest.raises(ValueError, match="asset_kind"):
        api.fetch_daily_increment(["000300.SH"], TARGET, asset_kind="fund")  # type: ignore[arg-type]
    assert called is False


def test_missing_symbol_is_visible_in_coverage_instead_of_fabricated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trading_days"):
            return _calendar_response(TARGET)
        return httpx.Response(
            200,
            json={
                "ret": 200,
                "data": [
                    {
                        "s": "000001.SZ",
                        "respList": [
                            _bar(date(2026, 8, 25), close="10.10"),
                            _bar(TARGET, close="10.50"),
                        ],
                    },
                    {"s": "000002.SZ", "respList": []},
                ],
            },
        )

    result = _api(handler).fetch_daily_increment(["000001.SZ", "000002.SZ"], TARGET)
    assert result.coverage_ratio == pytest.approx(0.5)
    assert result.received_symbols == ("000001.SZ",)


def test_stale_or_zero_trade_suspension_is_missing_not_batch_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trading_days"):
            return _calendar_response(TARGET)
        return httpx.Response(
            200,
            json={
                "ret": 200,
                "data": [
                    {
                        "s": "000001.SZ",
                        "respList": [
                            _bar(date(2026, 8, 25), close="10.10"),
                            _bar(TARGET, close="10.50"),
                        ],
                    },
                    {
                        "s": "000002.SZ",
                        "respList": [
                            _bar(date(2026, 8, 22), close="10.00"),
                            _bar(date(2026, 8, 25), close="10.10"),
                        ],
                    },
                    {
                        "s": "000003.SZ",
                        "respList": [
                            {
                                "t": str(_timestamp(TARGET)),
                                "o": "0",
                                "h": "0",
                                "l": "0",
                                "c": "0",
                                "v": "0",
                                "vw": "0",
                            }
                        ],
                    },
                ],
            },
        )

    result = _api(handler).fetch_daily_increment(["000001.SZ", "000002.SZ", "000003.SZ"], TARGET)

    assert result.received_symbols == ("000001.SZ",)
    assert result.coverage_ratio == pytest.approx(1 / 3)
    assert result.frame["symbol"].tolist() == ["000001"]


def test_zero_trade_placeholder_with_nonzero_amount_is_data_damage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trading_days"):
            return _calendar_response(TARGET)
        return httpx.Response(
            200,
            json={
                "ret": 200,
                "data": [
                    {
                        "s": "000001.SZ",
                        "respList": [
                            {
                                **_bar(TARGET, close="10.50"),
                                "v": "0",
                                "vw": "100",
                            }
                        ],
                    }
                ],
            },
        )

    with pytest.raises(DataQualityError, match="零值不一致"):
        _api(handler).fetch_daily_increment(["000001.SZ"], TARGET)


@pytest.mark.parametrize("wrapper", ["respList", "data_respList"])
def test_daily_increment_unwraps_symbol_level_provider_envelopes(wrapper: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trading_days"):
            return _calendar_response(TARGET)
        symbol_records = [
            {
                "s": "600000.SH",
                "respList": [
                    _bar(date(2026, 8, 25), close="10.10"),
                    _bar(TARGET, close="10.50"),
                ],
            }
        ]
        data = (
            {"respList": symbol_records}
            if wrapper == "respList"
            else {"data": {"respList": symbol_records}}
        )
        return httpx.Response(200, json={"ret": 200, "data": data})

    result = _api(handler).fetch_daily_increment(["600000.SH"], TARGET)
    assert result.received_symbols == ("600000.SH",)
    assert result.frame.loc[0, "symbol"] == "600000"


def test_bare_bar_envelope_without_symbol_identity_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trading_days"):
            return _calendar_response(TARGET)
        return httpx.Response(
            200,
            json={
                "ret": 200,
                "data": {
                    "respList": [
                        _bar(date(2026, 8, 25), close="10.10"),
                        _bar(TARGET, close="10.50"),
                    ]
                },
            },
        )

    with pytest.raises(DataQualityError, match="缺少可验证的证券代码"):
        _api(handler).fetch_daily_increment(["600000.SH"], TARGET)


def test_no_unit_contract_fails_before_network_and_never_reveals_key() -> None:
    called = False

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    api = InfowayEodMarketData(
        "never-print-this-secret",
        client=_client(handler),
        minimum_interval_seconds=0,
        utcnow=lambda: NOW,
    )
    with pytest.raises(DataQualityError, match="拒绝猜测") as exc_info:
        api.fetch_daily_increment(["000001.SZ"], TARGET)
    assert called is False
    assert "never-print-this-secret" not in str(exc_info.value)


@pytest.mark.parametrize(
    "target_bar, contract, match",
    [
        (
            {**_bar(TARGET, close="10.50"), "vm": "10500"},
            _units(),
            "同时包含vw与vm",
        ),
        (_bar(TARGET, close="10.50", amount_field="vm"), _units(), "字段.*不一致"),
        (_bar(TARGET, close="10.50", amount="10.50"), _units(), "单位不可验证"),
        (
            _bar(TARGET, close="10.50", amount="10500"),
            _units(volume_multiplier="100"),
            "单位不可验证",
        ),
    ],
)
def test_amount_field_or_unit_ambiguity_fails_closed(
    target_bar: dict[str, str],
    contract: InfowayEodUnitContract,
    match: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trading_days"):
            return _calendar_response(TARGET)
        return httpx.Response(
            200,
            json={
                "ret": 200,
                "data": [
                    {
                        "s": "000001.SZ",
                        "respList": [
                            _bar(date(2026, 8, 25), close="10.10"),
                            target_bar,
                        ],
                    }
                ],
            },
        )

    api = InfowayEodMarketData(
        "do-not-leak",
        unit_contract=contract,
        client=_client(handler),
        minimum_interval_seconds=0,
        utcnow=lambda: NOW,
    )
    with pytest.raises(DataQualityError, match=match) as exc_info:
        api.fetch_daily_increment(["000001.SZ"], TARGET)
    assert "do-not-leak" not in str(exc_info.value)


def test_stale_only_or_first_listing_bar_is_missing_for_coverage_gate() -> None:
    responses = iter(
        [
            [
                _bar(date(2026, 8, 24), close="10.00"),
                _bar(date(2026, 8, 25), close="10.10"),
            ],
            [_bar(TARGET, close="10.50")],
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trading_days"):
            return _calendar_response(TARGET)
        return httpx.Response(
            200,
            json={"ret": 200, "data": [{"s": "000001.SZ", "respList": next(responses)}]},
        )

    api = _api(handler)
    with pytest.raises(DataUnavailableError, match="没有返回"):
        api.fetch_daily_increment(["000001.SZ"], TARGET)
    with pytest.raises(DataUnavailableError, match="没有返回"):
        api.fetch_daily_increment(["000001.SZ"], TARGET)


@pytest.mark.parametrize(
    ("bars", "match"),
    [
        (
            [
                {**_bar(TARGET, close="10.40"), "t": str(_timestamp(TARGET, time(14, 59)))},
                _bar(TARGET, close="10.50"),
            ],
            "目标日期不符",
        ),
        (
            [
                _bar(TARGET, close="10.50"),
                _bar(date(2026, 8, 27), close="10.60"),
            ],
            "晚于目标日期",
        ),
    ],
)
def test_multiple_bars_with_invalid_previous_bar_timing_fail_closed(
    bars: list[dict[str, str]],
    match: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trading_days"):
            return _calendar_response(TARGET)
        return httpx.Response(
            200,
            json={"ret": 200, "data": [{"s": "000001.SZ", "respList": bars}]},
        )

    with pytest.raises(DataQualityError, match=match):
        _api(handler).fetch_daily_increment(["000001.SZ"], TARGET)


def test_intraday_target_and_non_trading_target_are_rejected() -> None:
    called = False

    def intraday_handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    intraday = InfowayEodMarketData(
        "intraday-secret",
        unit_contract=_units(),
        client=_client(intraday_handler),
        minimum_interval_seconds=0,
        utcnow=lambda: datetime(2026, 8, 26, 6, 59, tzinfo=UTC),
    )
    with pytest.raises(DataQualityError, match="尚未收盘"):
        intraday.fetch_daily_increment(["000001.SZ"], TARGET)
    assert called is False

    closed_day = _api(lambda _: _calendar_response())
    with pytest.raises(DataQualityError, match="未确认"):
        closed_day.fetch_daily_increment(["000001.SZ"], TARGET)


@pytest.mark.parametrize(
    "cutoff",
    [
        _timestamp(date(2026, 8, 25), time(23, 59)),
        _timestamp(TARGET, time(14, 59)),
        1_796_000_000_000,
    ],
)
def test_invalid_cutoff_timestamp_is_rejected(cutoff: int) -> None:
    api = _api(lambda _: httpx.Response(500))
    with pytest.raises(ValueError, match="cutoff_timestamp"):
        api.fetch_daily_increment(["000001.SZ"], TARGET, cutoff_timestamp=cutoff)


def test_empty_completed_payload_is_unavailable_and_http_errors_redact_key() -> None:
    def empty_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trading_days"):
            return _calendar_response(TARGET)
        return httpx.Response(200, json={"ret": 200, "data": []})

    with pytest.raises(DataUnavailableError, match="没有返回"):
        _api(empty_handler).fetch_daily_increment(["000001.SZ"], TARGET)

    def auth_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trading_days"):
            return _calendar_response(TARGET)
        return httpx.Response(401, json={"apiKey": "eod-private-secret"})

    with pytest.raises(DataUnavailableError) as exc_info:
        _api(auth_handler).fetch_daily_increment(["000001.SZ"], TARGET)
    assert "eod-private-secret" not in str(exc_info.value)
    assert "鉴权" in str(exc_info.value)


def test_eod_default_rate_limit_is_one_point_two_seconds() -> None:
    times = iter([0.0, 0.2, 1.2])
    sleeps: list[float] = []
    api = InfowayEodMarketData(
        "secret",
        unit_contract=_units(),
        client=_client(lambda _: _calendar_response(TARGET)),
        clock=lambda: next(times),
        sleeper=sleeps.append,
        utcnow=lambda: NOW,
    )

    api.fetch_cn_trading_days(TARGET, TARGET)
    api.fetch_cn_trading_days(TARGET, TARGET)

    assert sleeps == pytest.approx([1.0])


def test_internal_calendar_429_retries_then_daily_increment_succeeds() -> None:
    request_number = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_number
        request_number += 1
        if request_number == 1:
            return _calendar_response(TARGET)
        if request_number == 2:
            return httpx.Response(429, headers={"Retry-After": "3"})
        if request_number == 3:
            return _calendar_response(TARGET)
        assert request.url.path == "/stock/v2/batch_kline"
        return httpx.Response(
            200,
            json={
                "ret": 200,
                "data": [
                    {
                        "s": "000001.SZ",
                        "respList": [
                            _bar(date(2026, 8, 25), close="10.10"),
                            _bar(TARGET, close="10.50"),
                        ],
                    }
                ],
            },
        )

    api = _api(handler, sleeper=sleeps.append)
    assert api.fetch_cn_trading_days(TARGET, TARGET) == (TARGET,)
    result = api.fetch_daily_increment(["000001.SZ"], TARGET)

    assert result.received_symbols == ("000001.SZ",)
    assert request_number == 4
    assert sleeps == pytest.approx([3.0])


def test_continuous_429_stops_after_two_retries_and_redacts_secret() -> None:
    requests = 0
    sleeps: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(429, json={"apiKey": "eod-private-secret"})

    api = _api(handler, sleeper=sleeps.append)
    with pytest.raises(DataUnavailableError, match="有限重试") as exc_info:
        api.fetch_cn_trading_days(TARGET, TARGET)

    assert requests == 3
    assert sleeps == pytest.approx([2.0, 2.0])
    assert "eod-private-secret" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [("0.5", 2.0), ("999", 30.0), ("not-a-delay", 2.0)],
)
def test_retry_after_wait_is_bounded(
    retry_after: str,
    expected: float,
) -> None:
    requests = 0
    sleeps: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(429, headers={"Retry-After": retry_after})
        return _calendar_response(TARGET)

    api = _api(handler, sleeper=sleeps.append)
    assert api.fetch_cn_trading_days(TARGET, TARGET) == (TARGET,)
    assert sleeps == pytest.approx([expected])
