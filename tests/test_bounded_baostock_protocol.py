"""Malformed emitted evidence must not unlock a stale-calendar fallback."""

from datetime import date
from types import SimpleNamespace

import pytest

from ashare_lab.adapters.bounded_baostock import BoundedBaoStockEod
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError


def adapter(stdout, *, returncode=0):
    def runner(*_, **__):
        return SimpleNamespace(returncode=returncode, stdout=stdout)

    return BoundedBaoStockEod(runner=runner)


@pytest.mark.parametrize(
    "stdout",
    [
        "", "private provider text", "null", "[]", "{}",
        '{"error":"unexpected-private-error"}',
        '{"error":null}', '{"error":false}',
        '{"result":[],"error":"unavailable"}',
        '{"result":[],"extra":"private"}',
        '{"result":null}', '{"result":"2026-10-12"}',
    ],
)
def test_emitted_invalid_protocol_is_quality_failure_without_raw_output(stdout):
    with pytest.raises(DataQualityError) as caught:
        adapter(stdout).fetch_cn_stock_symbols()
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("stdout", ["", "  \n"])
def test_crashed_child_without_claimed_evidence_is_unavailable(stdout):
    with pytest.raises(DataUnavailableError):
        adapter(stdout, returncode=1).fetch_cn_stock_symbols()


@pytest.mark.parametrize("returncode", [0, 2])
def test_explicit_provider_unavailability_remains_unavailable(returncode):
    with pytest.raises(DataUnavailableError):
        adapter('{"error":"unavailable"}', returncode=returncode).fetch_cn_stock_symbols()


def test_result_claim_from_failed_process_is_not_trusted_or_hidden_by_fallback():
    with pytest.raises(DataQualityError):
        adapter('{"result":[]}', returncode=2).fetch_cn_stock_symbols()


@pytest.mark.parametrize("value", ['"bad"', '"20261012"', "null", "123", "true"])
def test_calendar_wire_date_failure_is_quality_failure(value):
    with pytest.raises(DataQualityError):
        adapter('{"result":[' + value + "]}").fetch_cn_trading_days(
            date(2026, 10, 10), date(2026, 10, 23)
        )


def test_valid_calendar_protocol_preserved():
    assert adapter('{"result":["2026-10-12","2026-10-13"]}').fetch_cn_trading_days(
        date(2026, 10, 10), date(2026, 10, 23)
    ) == (date(2026, 10, 12), date(2026, 10, 13))
