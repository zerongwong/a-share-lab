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


def test_stock_master_uses_official_backup_only_when_baostock_is_unavailable():
    calls = []

    class official:
        provider = "sse_szse_official_via_akshare"
        last_evidence = "official-exchange-evidence"

        @staticmethod
        def fetch_cn_stock_symbols():
            calls.append("official")
            return ("600000.SH",)

    with httpx.Client() as client:
        adapter = FreeEodMetadataFallback(
            primary(), backup(), stock_master_backup=official(), client=client
        )
        assert adapter.fetch_cn_stock_symbols() == ("600000.SH",)
    assert calls == ["official"]
    assert adapter.metadata_sources["fetch_cn_stock_symbols"] == "official-exchange-evidence"


def test_stock_master_quality_failure_never_silently_uses_official_backup():
    calls = []

    class official:
        @staticmethod
        def fetch_cn_stock_symbols():
            calls.append("official")
            return ("600000.SH",)

    with httpx.Client() as client:
        adapter = FreeEodMetadataFallback(
            primary(DataQualityError), backup(), stock_master_backup=official(), client=client
        )
        with pytest.raises(DataQualityError):
            adapter.fetch_cn_stock_symbols()
    assert calls == []


def test_stock_master_reports_all_free_sources_unavailable_without_falling_through():
    class unavailable_official:
        @staticmethod
        def fetch_cn_stock_symbols():
            raise DataUnavailableError("private upstream detail")

    with httpx.Client() as client:
        adapter = FreeEodMetadataFallback(
            primary(), backup(), stock_master_backup=unavailable_official(), client=client
        )
        with pytest.raises(DataUnavailableError, match="所有免费来源") as caught:
            adapter.fetch_cn_stock_symbols()
    assert "private upstream" not in str(caught.value)


def complete_stock_master() -> tuple[str, ...]:
    sh = (f"{600_000 + value:06d}.SH" for value in range(2_600))
    sz = (f"{value:06d}.SZ" for value in range(2_600))
    return tuple(sorted((*sh, *sz)))


class StaticStockMaster:
    provider = "synthetic_stock_master"
    last_evidence = "synthetic-independent-evidence"

    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.calls = 0

    def fetch_cn_stock_symbols(self):
        self.calls += 1
        if self.error is not None:
            raise self.error("private provider details")
        return self.value


def completeness_adapter(primary_master, official_master, *, expected=None):
    return FreeEodMetadataFallback(
        primary_master,
        backup(),
        stock_master_backup=official_master,
        expected_stock_symbols=expected,
        require_stock_master_completeness_evidence=True,
        client=SimpleNamespace(close=lambda: None),
    )


def test_well_formed_baostock_truncation_is_rejected_against_current_official_master():
    complete = complete_stock_master()
    truncated = complete[:-1]
    baostock = StaticStockMaster(truncated)
    official = StaticStockMaster(complete)
    adapter = completeness_adapter(baostock, official, expected=complete)

    with pytest.raises(DataQualityError, match="不一致"):
        adapter.fetch_cn_stock_symbols()

    assert baostock.calls == official.calls == 1
    assert "fetch_cn_stock_symbols" not in adapter.metadata_sources


def test_quality_mismatch_does_not_switch_to_the_official_result():
    complete = complete_stock_master()
    candidate = tuple(sorted((*complete[:-1], "603000.SH")))
    adapter = completeness_adapter(
        StaticStockMaster(candidate),
        StaticStockMaster(complete),
        expected=complete,
    )

    with pytest.raises(DataQualityError):
        adapter.fetch_cn_stock_symbols()


def test_recent_anchor_can_cover_temporary_official_unavailability_without_allowing_shrink():
    complete = complete_stock_master()
    official = StaticStockMaster(error=DataUnavailableError)
    accepted = completeness_adapter(StaticStockMaster(complete), official, expected=complete)

    assert accepted.fetch_cn_stock_symbols() == complete
    assert accepted.metadata_sources["fetch_cn_stock_symbols"] == (
        "baostock:checked_against_last_verified_master"
    )

    rejected = completeness_adapter(
        StaticStockMaster(complete[:-1]),
        StaticStockMaster(error=DataUnavailableError),
        expected=complete,
    )
    with pytest.raises(DataQualityError, match="未经独立确认的缩减"):
        rejected.fetch_cn_stock_symbols()


def test_first_master_requires_exact_baostock_and_official_agreement():
    complete = complete_stock_master()
    official = StaticStockMaster(complete)
    adapter = completeness_adapter(StaticStockMaster(complete), official)

    assert adapter.fetch_cn_stock_symbols() == complete
    assert adapter.metadata_sources["fetch_cn_stock_symbols"] == (
        "baostock:crosschecked:synthetic-independent-evidence"
    )

    mismatch = completeness_adapter(StaticStockMaster(complete[:-1]), StaticStockMaster(complete))
    with pytest.raises(DataQualityError, match="不一致"):
        mismatch.fetch_cn_stock_symbols()


def test_first_master_never_publishes_a_single_official_list():
    official = StaticStockMaster(complete_stock_master())
    adapter = completeness_adapter(StaticStockMaster(error=DataUnavailableError), official)

    with pytest.raises(DataUnavailableError, match="单一官方名单"):
        adapter.fetch_cn_stock_symbols()

    assert official.calls == 0


def test_official_quality_failure_is_not_downgraded_to_anchor_only_success():
    complete = complete_stock_master()
    adapter = completeness_adapter(
        StaticStockMaster(complete),
        StaticStockMaster(error=DataQualityError),
        expected=complete,
    )

    with pytest.raises(DataQualityError, match="private provider details"):
        adapter.fetch_cn_stock_symbols()


def test_two_current_sources_can_confirm_a_small_real_membership_change():
    old = complete_stock_master()
    current = tuple(sorted((*old[:-1], "603000.SH")))
    adapter = completeness_adapter(
        StaticStockMaster(current), StaticStockMaster(current), expected=old
    )

    assert adapter.fetch_cn_stock_symbols() == current


def test_two_matching_but_obviously_short_lists_are_not_complete_evidence():
    short = complete_stock_master()[:4_999]
    adapter = completeness_adapter(StaticStockMaster(short), StaticStockMaster(short))

    with pytest.raises(DataQualityError, match="数量不足"):
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
