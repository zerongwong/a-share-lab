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
        with pytest.raises(
            DataUnavailableError, match="reason=all_configured_sources_unavailable"
        ) as caught:
            adapter.fetch_cn_stock_symbols()
    assert "private upstream" not in str(caught.value)
    assert adapter.metadata_diagnostics["fetch_cn_stock_symbols"] == {
        "version": "stock-master-consensus-v2",
        "status": "unavailable",
        "reason": "all_configured_sources_unavailable",
        "available_sources": (),
        "unavailable_sources": ("baostock", "official_exchange"),
    }


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


def completeness_adapter(primary_master, official_master, *, expected=None, metadata_master=None):
    if metadata_master is None:
        metadata_master = StaticStockMaster(error=DataUnavailableError)
    return FreeEodMetadataFallback(
        primary_master,
        metadata_master,
        stock_master_backup=official_master,
        expected_stock_symbols=expected,
        require_stock_master_completeness_evidence=True,
        client=SimpleNamespace(close=lambda: None),
    )


def test_bounded_one_source_timing_lag_is_resolved_by_two_of_three_vote():
    complete = complete_stock_master()
    lagging = complete[:-1]
    baostock = StaticStockMaster(lagging)
    official = StaticStockMaster(complete)
    adapter = completeness_adapter(baostock, official, expected=complete)

    assert adapter.fetch_cn_stock_symbols() == complete

    assert baostock.calls == official.calls == 1
    assert adapter.metadata_diagnostics["fetch_cn_stock_symbols"]["decision"] == (
        "membership_vote_two_of_three"
    )
    assert adapter.metadata_diagnostics["fetch_cn_stock_symbols"]["pairwise_drift"] == {
        "baostock__official_exchange": 1,
        "baostock__last_verified": 1,
        "official_exchange__last_verified": 0,
    }


def test_anchor_breaks_a_bounded_current_source_membership_tie_deterministically():
    complete = complete_stock_master()
    candidate = tuple(sorted((*complete[:-1], "603000.SH")))
    adapter = completeness_adapter(
        StaticStockMaster(candidate),
        StaticStockMaster(complete),
        expected=complete,
    )

    assert adapter.fetch_cn_stock_symbols() == complete
    assert "private" not in adapter.metadata_sources["fetch_cn_stock_symbols"]


def test_recent_anchor_can_cover_temporary_official_unavailability_without_allowing_shrink():
    complete = complete_stock_master()
    official = StaticStockMaster(error=DataUnavailableError)
    accepted = completeness_adapter(StaticStockMaster(complete), official, expected=complete)

    assert accepted.fetch_cn_stock_symbols() == complete
    assert (
        "decision=unanimous_current_and_anchor"
        in accepted.metadata_sources["fetch_cn_stock_symbols"]
    )
    assert accepted.metadata_diagnostics["fetch_cn_stock_symbols"]["unavailable_sources"] == (
        "official_exchange",
        "tushare",
    )

    rejected = completeness_adapter(
        StaticStockMaster(complete[:-1]),
        StaticStockMaster(error=DataUnavailableError),
        expected=complete,
    )
    with pytest.raises(DataUnavailableError, match="single_current_source_ambiguous"):
        rejected.fetch_cn_stock_symbols()


def test_first_master_requires_exact_baostock_and_official_agreement():
    complete = complete_stock_master()
    official = StaticStockMaster(complete)
    adapter = completeness_adapter(StaticStockMaster(complete), official)

    assert adapter.fetch_cn_stock_symbols() == complete
    assert "decision=unanimous_two_current" in adapter.metadata_sources["fetch_cn_stock_symbols"]

    mismatch = completeness_adapter(StaticStockMaster(complete[:-1]), StaticStockMaster(complete))
    with pytest.raises(DataUnavailableError, match="initial_current_sources_ambiguous"):
        mismatch.fetch_cn_stock_symbols()


def test_first_master_never_publishes_a_single_official_list():
    official = StaticStockMaster(complete_stock_master())
    adapter = completeness_adapter(StaticStockMaster(error=DataUnavailableError), official)

    with pytest.raises(DataUnavailableError, match="initial_sync_needs_two_current_sources"):
        adapter.fetch_cn_stock_symbols()

    assert official.calls == 1


def test_official_quality_failure_is_not_downgraded_to_anchor_only_success():
    complete = complete_stock_master()
    adapter = completeness_adapter(
        StaticStockMaster(complete),
        StaticStockMaster(error=DataQualityError),
        expected=complete,
    )

    with pytest.raises(DataQualityError, match="official_exchange_quality_failure") as caught:
        adapter.fetch_cn_stock_symbols()
    assert "private provider details" not in str(caught.value)


def test_two_current_sources_can_confirm_a_small_real_membership_change():
    old = complete_stock_master()
    current = tuple(sorted((*old[:-1], "603000.SH")))
    adapter = completeness_adapter(
        StaticStockMaster(current), StaticStockMaster(current), expected=old
    )

    assert adapter.fetch_cn_stock_symbols() == current


def test_tushare_replaces_unavailable_baostock_in_three_evidence_vote():
    old = complete_stock_master()
    current = tuple(sorted((*old[:-1], "603000.SH")))
    baostock = StaticStockMaster(error=DataUnavailableError)
    official = StaticStockMaster(current)
    tushare = StaticStockMaster(current)
    adapter = completeness_adapter(
        baostock,
        official,
        expected=old,
        metadata_master=tushare,
    )

    assert adapter.fetch_cn_stock_symbols() == current
    diagnostic = adapter.metadata_diagnostics["fetch_cn_stock_symbols"]
    assert diagnostic["decision"] == "membership_vote_two_of_three"
    assert diagnostic["available_sources"] == ("official_exchange", "tushare")
    assert diagnostic["unavailable_sources"] == ("baostock",)
    assert baostock.calls == official.calls == tushare.calls == 1


def test_large_drift_is_rejected_even_when_other_evidence_could_outvote_it():
    complete = complete_stock_master()
    drifted = tuple(
        sorted((*complete[100:], *(f"{603_000 + value:06d}.SH" for value in range(100))))
    )
    adapter = completeness_adapter(
        StaticStockMaster(drifted),
        StaticStockMaster(complete),
        expected=complete,
    )

    with pytest.raises(DataQualityError, match="baostock_large_drift") as caught:
        adapter.fetch_cn_stock_symbols()

    assert "603000.SH" not in str(caught.value)
    assert adapter.metadata_diagnostics["fetch_cn_stock_symbols"]["status"] == ("quality_rejected")


def test_two_individually_bounded_but_mutually_divergent_sources_are_rejected():
    complete = complete_stock_master()
    first = tuple(sorted((*complete[25:], *(f"{603_000 + value:06d}.SH" for value in range(25)))))
    second = tuple(sorted((*complete[:-25], *(f"{603_100 + value:06d}.SH" for value in range(25)))))
    adapter = completeness_adapter(
        StaticStockMaster(first),
        StaticStockMaster(second),
        expected=complete,
    )

    with pytest.raises(DataQualityError, match="current_sources_large_drift"):
        adapter.fetch_cn_stock_symbols()


def test_primary_quality_failure_is_sanitized_and_never_falls_through():
    complete = complete_stock_master()
    baostock = StaticStockMaster(error=DataQualityError)
    official = StaticStockMaster(complete)
    tushare = StaticStockMaster(complete)
    adapter = completeness_adapter(
        baostock,
        official,
        expected=complete,
        metadata_master=tushare,
    )

    with pytest.raises(DataQualityError, match="baostock_quality_failure") as caught:
        adapter.fetch_cn_stock_symbols()

    assert "private provider details" not in str(caught.value)
    assert baostock.calls == 1
    assert official.calls == tushare.calls == 0


def test_two_matching_but_obviously_short_lists_are_not_complete_evidence():
    short = complete_stock_master()[:4_999]
    adapter = completeness_adapter(StaticStockMaster(short), StaticStockMaster(short))

    with pytest.raises(DataQualityError, match="baostock_truncated_or_invalid"):
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
