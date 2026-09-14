from __future__ import annotations

import json
import subprocess
from datetime import UTC, date, datetime
from types import SimpleNamespace

import httpx
import pytest

from ashare_lab.adapters.bounded_baostock import BoundedBaoStockEod
from ashare_lab.adapters.free_eod_fallback import FreeEodMetadataFallback
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError

TARGET = date(2026, 9, 11)
NOW = datetime(2026, 9, 14, 1, tzinfo=UTC)
DAYS = (date(2026, 9, 10), TARGET)


def fail(*_args, **_kwargs):
    raise DataUnavailableError("synthetic unavailable")


def primary(error=DataUnavailableError):
    def broken(*_args, **_kwargs):
        raise error("synthetic")

    return SimpleNamespace(
        fetch_cn_trading_days=broken, fetch_cn_stock_symbols=broken, fetch_core_index_daily=broken
    )


def backup():
    return SimpleNamespace(
        fetch_cn_trading_days=lambda *_: DAYS, fetch_cn_stock_symbols=lambda: ("600000.SH",)
    )


def test_metadata_failover_on_unavailability_only():
    with httpx.Client() as client:
        adapter = FreeEodMetadataFallback(primary(), backup(), client=client)
        assert adapter.fetch_cn_trading_days(*DAYS) == DAYS
        assert adapter.fetch_cn_stock_symbols() == ("600000.SH",)
        assert set(adapter.metadata_sources.values()) == {"tushare"}
        adapter.primary = primary(DataQualityError)
        with pytest.raises(DataQualityError):
            adapter.fetch_cn_stock_symbols()


def response(request, *, mismatch=False, wrong_date=False):
    if "eastmoney.com" in request.url.host:
        code = request.url.params["secid"].split(".")[1]
        rows = [f"{d},10,10.5,11,9,100,100000,1" for d in DAYS]
        if wrong_date:
            rows = rows[:1]
        return httpx.Response(200, json={"data": {"code": code, "klines": rows}})
    code = request.url.params["param"].split(",")[0]
    rows = [[str(d), "10", "10.5", "11", "9", "100", {}, "1", "10"] for d in DAYS]
    if mismatch:
        rows[-1][2] = "12"
    return httpx.Response(200, json={"data": {code: {"day": rows}}})


def test_all_six_indices_have_crosschecked_units_and_explicit_provenance():
    calls = []

    def handler(request):
        calls.append(request)
        return response(request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        adapter = FreeEodMetadataFallback(primary(), backup(), client=client, clock=lambda: NOW)
        batch = adapter.fetch_core_index_daily(TARGET, cutoff_timestamp=123)
    assert len(calls) == 12
    assert len(batch.frame) == 6
    assert batch.provider == "akshare"
    assert batch.frame.volume_shares.eq(10000).all()
    assert batch.frame.amount_cny.eq(100000).all()
    assert batch.frame.prev_close.eq(10.5).all()
    assert batch.frame.source.eq("akshare:eastmoney:indices:tx_verified").all()
    assert batch.unit_resolution_method_version == "em_tx_all_six_crosscheck-v1"


@pytest.mark.parametrize(
    "kwargs,error",
    [({"mismatch": True}, DataQualityError), ({"wrong_date": True}, DataUnavailableError)],
)
def test_index_missing_or_mismatched_data_cannot_be_published(kwargs, error):
    with httpx.Client(transport=httpx.MockTransport(lambda r: response(r, **kwargs))) as client:
        adapter = FreeEodMetadataFallback(primary(), backup(), client=client, clock=lambda: NOW)
        with pytest.raises(error):
            adapter.fetch_core_index_daily(TARGET, cutoff_timestamp=123)


def test_isolated_baostock_timeout_does_not_leak_child_payload():
    def runner(*args, **kwargs):
        assert kwargs["timeout"] == 1
        assert json.loads(kwargs["input"])["operation"] == "calendar"
        raise subprocess.TimeoutExpired("safe", 1, output="sensitive diagnostics")

    with pytest.raises(DataUnavailableError) as caught:
        BoundedBaoStockEod(timeout=1, runner=runner).fetch_cn_trading_days(*DAYS)
    assert "sensitive" not in str(caught.value)


def test_isolated_baostock_quality_failure_is_not_downgraded():
    def runner(*_, **__):
        return SimpleNamespace(returncode=2, stdout='{"error":"quality"}')

    with pytest.raises(DataQualityError):
        BoundedBaoStockEod(runner=runner).fetch_cn_stock_symbols()


def test_isolated_baostock_reads_dates_without_console_noise():
    def runner(*_, **__):
        return SimpleNamespace(returncode=0, stdout='{"result":["2026-09-10","2026-09-11"]}')

    assert BoundedBaoStockEod(runner=runner).fetch_cn_trading_days(*DAYS) == DAYS


@pytest.mark.parametrize("payload", ["null", "[]", "{}"])
def test_invalid_child_protocol_is_unavailable(payload):
    def runner(*_, **__):
        return SimpleNamespace(returncode=0, stdout=payload)

    with pytest.raises(DataUnavailableError):
        BoundedBaoStockEod(runner=runner).fetch_cn_stock_symbols()


def test_evening_calendar_uses_backup_only_after_unavailability():
    from ashare_lab.cli.evening_digest import resolve_next_zero_budget_trading_day

    calls = []

    def calendar(start, end):
        calls.append((start, end))
        return (date(2026, 9, 14), date(2026, 9, 15))

    assert resolve_next_zero_budget_trading_day(
        TARGET, _provider_factory=primary, _calendar_backup=calendar
    ) == date(2026, 9, 14)
    assert len(calls) == 1
    with pytest.raises(DataQualityError):
        resolve_next_zero_budget_trading_day(
            TARGET, _provider_factory=lambda: primary(DataQualityError), _calendar_backup=calendar
        )
    assert len(calls) == 1


def test_evening_calendar_cannot_accept_a_date_outside_its_request():
    from ashare_lab.cli.evening_digest import resolve_next_zero_budget_trading_day

    with pytest.raises(DataUnavailableError, match="区间外"):
        resolve_next_zero_budget_trading_day(
            TARGET,
            _provider_factory=primary,
            _calendar_backup=lambda *_: (date(2026, 10, 1),),
        )
