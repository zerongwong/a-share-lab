from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from ashare_lab.adapters.infoway_realtime import InfowayRealtimeMarketData
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_symbol_list_uses_stock_cn_and_never_puts_key_in_url() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("apiKey")
        return httpx.Response(
            200,
            json={
                "ret": 200,
                "msg": "success",
                "traceId": "trace-1",
                "data": [{"symbol": "000001.SZ", "name_cn": "平安银行"}],
            },
        )

    api = InfowayRealtimeMarketData(
        "local-secret",
        client=_client(handler),
        minimum_interval_seconds=0,
    )
    records = api.fetch_symbol_list()

    assert records[0]["symbol"] == "000001.SZ"
    assert "type=STOCK_CN" in seen["url"]
    assert "local-secret" not in seen["url"]
    assert seen["key"] == "local-secret"


def test_symbol_fundamentals_batches_at_five_hundred_and_marks_current_snapshot() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        symbols = request.url.params["symbols"].split(",")
        return httpx.Response(
            200,
            json={
                "ret": 200,
                "traceId": f"trace-{len(requests)}",
                "data": [
                    {
                        "symbol": symbol,
                        "eps": "1.20",
                        "eps_ttm": "1.35",
                        "bps": "5.60",
                        "dividend_yield": "2.1",
                    }
                    for symbol in symbols
                ],
            },
        )

    symbols = [f"{number:06d}.SZ" for number in range(501)]
    api = InfowayRealtimeMarketData(
        "fundamentals-secret",
        client=_client(handler),
        minimum_interval_seconds=0,
        utcnow=lambda: datetime(2026, 8, 25, tzinfo=UTC),
    )
    result = api.fetch_symbol_fundamentals(symbols)

    assert len(requests) == 2
    assert all(request.method == "GET" for request in requests)
    assert all(request.url.path == "/common/basic/symbols/info" for request in requests)
    assert all(request.url.params["type"] == "STOCK_CN" for request in requests)
    assert len(requests[0].url.params["symbols"].split(",")) == 500
    assert "fundamentals-secret" not in "".join(str(request.url) for request in requests)
    assert result.coverage_ratio == 1.0
    assert result.snapshot_scope == "current"
    assert result.is_point_in_time_history is False
    assert result.historical_backtest_eligible is False
    assert result.fetched_at == datetime(2026, 8, 25, tzinfo=UTC)
    assert result.records[0]["eps_ttm"] == "1.35"


@pytest.mark.parametrize(
    "records, match",
    [
        ([{"eps_ttm": "1.0"}], "无效证券代码"),
        (
            [{"symbol": "000001.SZ"}, {"symbol": "000001.sz"}],
            "重复证券代码",
        ),
        ([{"symbol": "600000.SH"}], "未请求的证券代码"),
    ],
)
def test_symbol_fundamentals_rejects_invalid_or_ambiguous_identity(
    records: list[dict[str, str]],
    match: str,
) -> None:
    api = InfowayRealtimeMarketData(
        "secret-not-in-errors",
        client=_client(
            lambda _: httpx.Response(
                200,
                json={"ret": 200, "data": records},
            )
        ),
        minimum_interval_seconds=0,
    )

    with pytest.raises(DataQualityError, match=match) as exc_info:
        api.fetch_symbol_fundamentals(["000001.SZ"])
    assert "secret-not-in-errors" not in str(exc_info.value)


def test_symbol_fundamentals_rejects_non_list_payload() -> None:
    api = InfowayRealtimeMarketData(
        "secret",
        client=_client(
            lambda _: httpx.Response(
                200,
                json={"ret": 200, "data": {"symbol": "000001.SZ"}},
            )
        ),
        minimum_interval_seconds=0,
    )

    with pytest.raises(DataQualityError, match="数据结构异常"):
        api.fetch_symbol_fundamentals(["000001.SZ"])


def test_symbol_fundamentals_uses_free_plan_rate_limit_between_batches() -> None:
    times = iter([0.0, 0.25, 1.0])
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        symbols = request.url.params["symbols"].split(",")
        return httpx.Response(
            200,
            json={"ret": 200, "data": [{"symbol": symbol} for symbol in symbols]},
        )

    api = InfowayRealtimeMarketData(
        "secret",
        client=_client(handler),
        minimum_interval_seconds=1.0,
        clock=lambda: next(times),
        sleeper=sleeps.append,
    )
    api.fetch_symbol_fundamentals([f"{number:06d}.SZ" for number in range(501)])

    assert sleeps == pytest.approx([0.75])


def test_kline_chunks_more_than_one_hundred_symbols() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = request.read().decode()
        codes = __import__("json").loads(body)["codes"].split(",")
        return httpx.Response(
            200,
            json={
                "ret": 200,
                "traceId": f"trace-{len(requests)}",
                "data": [{"s": code, "respList": []} for code in codes],
            },
        )

    symbols = [f"{number:06d}.SZ" for number in range(205)]
    api = InfowayRealtimeMarketData(
        "secret",
        client=_client(handler),
        minimum_interval_seconds=0,
        utcnow=lambda: datetime(2026, 8, 25, tzinfo=UTC),
    )
    result = api.fetch_recent_klines(symbols)

    assert len(requests) == 3
    assert all(request.method == "POST" for request in requests)
    assert result.coverage_ratio == 1.0
    assert result.trace_ids == ("trace-1", "trace-2", "trace-3")


def test_multi_symbol_kline_rejects_more_than_two_bars() -> None:
    api = InfowayRealtimeMarketData(
        "secret",
        client=_client(lambda _: httpx.Response(500)),
        minimum_interval_seconds=0,
    )
    with pytest.raises(ValueError, match="at most 2"):
        api.fetch_recent_klines(["000001.SZ", "600000.SH"], count=3)


def test_key_is_redacted_from_authentication_error() -> None:
    api = InfowayRealtimeMarketData(
        "super-secret-value",
        client=_client(lambda _: httpx.Response(401, json={"apiKey": "super-secret-value"})),
        minimum_interval_seconds=0,
    )
    with pytest.raises(DataUnavailableError) as exc_info:
        api.fetch_latest_trades(["000001.SZ"])
    assert "super-secret-value" not in str(exc_info.value)
    assert "鉴权" in str(exc_info.value)


def test_free_plan_rate_limiter_waits_between_batches() -> None:
    times = iter([0.0, 0.2, 1.0])
    sleeps: list[float] = []

    def clock() -> float:
        return next(times)

    def handler(request: httpx.Request) -> httpx.Response:
        codes = str(request.url).rsplit("/", 1)[-1].split(",")
        return httpx.Response(
            200,
            json={"ret": 200, "data": [{"s": code} for code in codes]},
        )

    symbols = [f"{number:06d}.SZ" for number in range(101)]
    api = InfowayRealtimeMarketData(
        "secret",
        client=_client(handler),
        minimum_interval_seconds=1.0,
        clock=clock,
        sleeper=sleeps.append,
    )
    api.fetch_latest_trades(symbols)
    assert sleeps == pytest.approx([0.8])


def test_missing_environment_key_fails_without_revealing_secrets(monkeypatch) -> None:
    monkeypatch.delenv("INFOWAY_API_KEY", raising=False)
    with pytest.raises(DataUnavailableError, match="勿发送到聊天"):
        InfowayRealtimeMarketData.from_environment()
