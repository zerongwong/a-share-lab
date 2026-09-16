from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from ashare_lab.adapters import cninfo_company_actions as parent
from ashare_lab.cli import cninfo_company_actions_read as child
from ashare_lab.ports.company_actions import (
    CNINFO_COMPANY_ACTION_METHOD_VERSION,
    COMPANY_ACTION_COVERAGE_START,
    CompanyActionEvidenceStatus,
)

AS_OF = date(2026, 9, 15)
NOW = datetime(2026, 9, 15, 8, 45, tzinfo=UTC)
SYMBOL = "600919"
HASH = "a" * 64


def _receipt(stream: str) -> dict[str, object]:
    return {
        "stream": stream,
        "complete": True,
        "record_count": 0 if stream != "identity" else 1,
        "response_hash": HASH,
        "reason_code": "STREAM_COMPLETE",
    }


def _wire_result(symbol: str = SYMBOL) -> dict[str, object]:
    return {
        "symbol": symbol,
        "status": "clear",
        "coverage_from": "1990-01-01",
        "coverage_through": AS_OF.isoformat(),
        "knowledge_time": NOW.isoformat(),
        "event_dates": [],
        "event_kinds": [],
        "stream_receipts": [
            _receipt("identity"),
            _receipt("dividend"),
            _receipt("allotment"),
            _receipt("share_change"),
        ],
        "response_hash": HASH,
        "reason_code": "ALL_STREAMS_COMPLETE_NO_EVENT",
        "method_version": CNINFO_COMPANY_ACTION_METHOD_VERSION,
    }


def _child_document(results: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "method_version": CNINFO_COMPANY_ACTION_METHOD_VERSION,
            "as_of": AS_OF.isoformat(),
            "results": results,
        }
    )


def test_parent_sends_only_symbols_and_public_cutoff(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def runner(command, **kwargs):
        captured.update({"command": command, **kwargs})
        return SimpleNamespace(returncode=0, stdout=_child_document([_wire_result()]))

    monkeypatch.setattr(parent, "_runner", runner)
    result = parent.read_cninfo_company_actions((SYMBOL,), AS_OF)

    assert result[0].status is CompanyActionEvidenceStatus.CLEAR
    assert json.loads(captured["input"]) == {
        "symbols": [SYMBOL],
        "as_of": AS_OF.isoformat(),
    }
    serialized = str(captured)
    for private_sentinel in (
        "cost_price",
        "entry_date",
        "stock_sleeve_weight",
        "account_weight",
        "position_key",
        "holding_version",
    ):
        assert private_sentinel not in serialized


def test_empty_symbols_do_not_spawn_a_child(monkeypatch) -> None:
    monkeypatch.setattr(
        parent,
        "_runner",
        lambda *_args, **_kwargs: pytest.fail("no child should be started"),
    )
    assert parent.read_cninfo_company_actions((), AS_OF) == ()


@pytest.mark.parametrize(
    "symbols",
    [
        ("600919",) * 2,
        ("60091",),
        ("sh.600919",),
        ("600919", "000001", "300001", "688001", "600000", "000002"),
    ],
)
def test_invalid_or_over_bound_symbols_are_rejected_before_child(monkeypatch, symbols) -> None:
    monkeypatch.setattr(
        parent,
        "_runner",
        lambda *_args, **_kwargs: pytest.fail("invalid input must not spawn a child"),
    )
    with pytest.raises(ValueError):
        parent.read_cninfo_company_actions(symbols, AS_OF)


def test_child_timeout_becomes_unknown_without_raw_error(monkeypatch) -> None:
    def runner(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("private command", 45, stderr="secret URL")

    monkeypatch.setattr(parent, "_runner", runner)
    result = parent.read_cninfo_company_actions((SYMBOL,), AS_OF)
    assert result[0].status is CompanyActionEvidenceStatus.UNKNOWN
    assert result[0].reason_code == "CHILD_DEADLINE_EXCEEDED"
    assert result[0].response_hash is None


def test_invalid_child_protocol_fails_closed_for_every_symbol(monkeypatch) -> None:
    monkeypatch.setattr(
        parent,
        "_runner",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"provider_url":"https://forbidden.example","results":[]}',
        ),
    )
    result = parent.read_cninfo_company_actions(("600919", "000001"), AS_OF)
    assert [item.status for item in result] == [
        CompanyActionEvidenceStatus.UNKNOWN,
        CompanyActionEvidenceStatus.UNKNOWN,
    ]
    assert {item.reason_code for item in result} == {"CHILD_PROTOCOL_INVALID"}


@pytest.mark.parametrize("mutation", ["method", "identity"])
def test_parent_rejects_untrusted_detected_child_result(monkeypatch, mutation) -> None:
    wire = _wire_result()
    wire.update(
        {
            "status": "detected",
            "event_dates": [AS_OF.isoformat()],
            "event_kinds": ["CASH_DIVIDEND"],
            "reason_code": "COMPANY_ACTION_DETECTED",
        }
    )
    if mutation == "method":
        wire["method_version"] = "unexpected-method"
    else:
        wire["stream_receipts"][0]["complete"] = False
        wire["stream_receipts"][0]["reason_code"] = "IDENTITY_MISMATCH"
    monkeypatch.setattr(
        parent,
        "_runner",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=_child_document([wire]),
        ),
    )

    result = parent.read_cninfo_company_actions((SYMBOL,), AS_OF)

    assert result[0].status is CompanyActionEvidenceStatus.UNKNOWN
    assert result[0].reason_code == "CHILD_PROTOCOL_INVALID"


class _FakeTransport:
    def __init__(self, documents):
        self.documents = documents
        self.calls: list[tuple[str, str, date]] = []

    def read(self, stream, symbol, as_of):
        self.calls.append((stream, symbol, as_of))
        value = self.documents[stream]
        if isinstance(value, Exception):
            raise value
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        return child._ProviderResponse(value, hashlib.sha256(raw).hexdigest())


def _complete_documents(symbol: str = SYMBOL):
    def envelope(records):
        return {
            "resultcode": 200,
            "resultmsg": "success",
            "count": len(records),
            "total": len(records),
            "records": records,
        }

    return {
        "identity": envelope([{"ASECCODE": symbol}]),
        "dividend": envelope([]),
        "allotment": envelope([]),
        "share_change": envelope([]),
    }


def _read_with(documents):
    transport = _FakeTransport(documents)
    result = child._read_symbols((SYMBOL,), AS_OF, transport=transport, clock=lambda: NOW)[0]
    return result, transport


def test_all_four_identity_bound_complete_empty_streams_are_clear() -> None:
    result, transport = _read_with(_complete_documents())
    assert result.status is CompanyActionEvidenceStatus.CLEAR
    assert result.coverage_from == COMPANY_ACTION_COVERAGE_START
    assert result.coverage_through == AS_OF
    assert result.event_dates == ()
    assert [item[0] for item in transport.calls] == [
        "identity",
        "dividend",
        "allotment",
        "share_change",
    ]
    assert all(receipt.complete for receipt in result.stream_receipts)


def test_empty_records_without_declared_count_are_unknown() -> None:
    documents = _complete_documents()
    documents["dividend"] = {
        "resultcode": 200,
        "resultmsg": "success",
        "records": [],
    }
    result, _ = _read_with(documents)
    assert result.status is CompanyActionEvidenceStatus.UNKNOWN
    assert result.reason_code == "DECLARED_COUNT_MISSING"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("resultcode", 500, "PROVIDER_RESULT_CODE_INVALID"),
        ("resultmsg", "partial", "PROVIDER_RESULT_MESSAGE_INVALID"),
        ("total", 1, "DECLARED_COUNT_MISMATCH"),
    ],
)
def test_empty_results_require_success_and_equal_count_total(field, value, reason) -> None:
    documents = _complete_documents()
    documents["dividend"][field] = value
    result, _ = _read_with(documents)
    assert result.status is CompanyActionEvidenceStatus.UNKNOWN
    assert result.reason_code == reason


def test_identity_mismatch_is_unknown_and_stops_additional_queries() -> None:
    documents = _complete_documents()
    documents["identity"] = {
        "resultcode": 200,
        "resultmsg": "success",
        "count": 1,
        "total": 1,
        "records": [{"ASECCODE": "600000"}],
    }
    result, transport = _read_with(documents)
    assert result.status is CompanyActionEvidenceStatus.UNKNOWN
    assert result.reason_code == "IDENTITY_MISMATCH"
    assert [item[0] for item in transport.calls] == ["identity"]


def test_dividend_bonus_and_transfer_are_detected_by_effective_date() -> None:
    documents = _complete_documents()
    documents["dividend"] = {
        "resultcode": 200,
        "resultmsg": "success",
        "count": 1,
        "total": 1,
        "records": [
            {
                "F010N": "1",
                "F011N": "2",
                "F012N": "0.5",
                "F020D": "2026-09-10 00:00:00",
            }
        ],
    }
    result, _ = _read_with(documents)
    assert result.status is CompanyActionEvidenceStatus.DETECTED
    assert set(zip(result.event_dates, result.event_kinds, strict=True)) == {
        (date(2026, 9, 10), "CASH_DIVIDEND"),
        (date(2026, 9, 10), "STOCK_DIVIDEND"),
        (date(2026, 9, 10), "CAPITAL_RESERVE_TRANSFER"),
    }


def test_dividend_record_with_all_ratios_missing_is_unknown() -> None:
    documents = _complete_documents()
    documents["dividend"] = {
        "resultcode": 200,
        "resultmsg": "success",
        "count": 1,
        "total": 1,
        "records": [
            {
                "F010N": None,
                "F011N": "--",
                "F012N": "",
                "F020D": None,
            }
        ],
    }

    result, _ = _read_with(documents)

    assert result.status is CompanyActionEvidenceStatus.UNKNOWN
    assert result.reason_code == "DIVIDEND_RECORD_INVALID"


def test_reliable_event_stays_detected_when_declared_count_is_incomplete() -> None:
    documents = _complete_documents()
    documents["dividend"] = {
        "resultcode": 200,
        "resultmsg": "success",
        "count": 2,
        "total": 2,
        "records": [{"F010N": 0, "F011N": 0, "F012N": 1, "F020D": "2026-09-15"}],
    }
    result, _ = _read_with(documents)
    assert result.status is CompanyActionEvidenceStatus.DETECTED
    assert result.event_kinds == ("CASH_DIVIDEND",)
    dividend = next(item for item in result.stream_receipts if item.stream == "dividend")
    assert dividend.complete is False
    assert dividend.reason_code == "DECLARED_COUNT_MISMATCH"


def test_allotment_requires_exact_record_identity() -> None:
    documents = _complete_documents()
    row = {"SECCODE": SYMBOL, "F012D": "2026-09-15"}
    documents["allotment"] = {
        "resultcode": 200,
        "resultmsg": "success",
        "count": 1,
        "total": 1,
        "records": [row],
    }
    result, _ = _read_with(documents)
    assert result.status is CompanyActionEvidenceStatus.DETECTED
    assert (date(2026, 9, 15), "RIGHTS_ISSUE") in set(
        zip(result.event_dates, result.event_kinds, strict=True)
    )


def test_allotment_without_effective_date_is_unknown() -> None:
    documents = _complete_documents()
    documents["allotment"] = {
        "resultcode": 200,
        "resultmsg": "success",
        "count": 1,
        "total": 1,
        "records": [{"SECCODE": SYMBOL, "F012D": None}],
    }
    result, _ = _read_with(documents)
    assert result.status is CompanyActionEvidenceStatus.UNKNOWN
    assert result.reason_code == "ALLOTMENT_RECORD_INVALID"


def test_unclassified_share_capital_rows_are_conservative_detected_events() -> None:
    documents = _complete_documents()
    documents["share_change"] = {
        "resultcode": 200,
        "resultmsg": "success",
        "count": 2,
        "total": 2,
        "records": [
            {
                "SECCODE": SYMBOL,
                "VARYDATE": "2026-09-01",
                "F002V": "定期记录",
                "F003N": "1000000",
            },
            {
                "SECCODE": SYMBOL,
                "VARYDATE": "2026-09-12",
                "F002V": "未知股本变动",
                "F003N": "1200000",
            },
        ],
    }
    result, _ = _read_with(documents)
    assert result.status is CompanyActionEvidenceStatus.DETECTED
    assert set(zip(result.event_dates, result.event_kinds, strict=True)) == {
        (date(2026, 9, 1), "OTHER_SHARE_CHANGE"),
        (date(2026, 9, 12), "OTHER_SHARE_CHANGE"),
    }


def test_single_unclassified_share_change_cannot_be_false_clear() -> None:
    documents = _complete_documents()
    documents["share_change"] = {
        "resultcode": 200,
        "resultmsg": "success",
        "count": 1,
        "total": 1,
        "records": [
            {
                "SECCODE": SYMBOL,
                "VARYDATE": "2026-09-12",
                "F002V": "未知股本变动",
                "F003N": "1200000",
            }
        ],
    }

    result, _ = _read_with(documents)

    assert result.status is CompanyActionEvidenceStatus.DETECTED
    assert result.event_dates == (date(2026, 9, 12),)
    assert result.event_kinds == ("OTHER_SHARE_CHANGE",)


def test_knowledge_time_is_taken_after_every_symbol_request() -> None:
    transport = _FakeTransport(_complete_documents())

    def clock():
        assert len(transport.calls) == 4
        return NOW

    result = child._read_symbols((SYMBOL,), AS_OF, transport=transport, clock=clock)[0]
    assert result.knowledge_time == NOW


def test_event_after_cutoff_does_not_invalidate_completed_interval() -> None:
    documents = _complete_documents()
    documents["dividend"] = {
        "resultcode": 200,
        "resultmsg": "success",
        "count": 1,
        "total": 1,
        "records": [{"F010N": 0, "F011N": 0, "F012N": 1, "F020D": "2026-09-16"}],
    }
    result, _ = _read_with(documents)
    assert result.status is CompanyActionEvidenceStatus.CLEAR
    assert result.event_dates == ()


def test_transport_disables_environment_and_sends_no_entry_date(monkeypatch) -> None:
    constructed: dict[str, object] = {}
    calls: list[dict[str, object]] = []

    class Response:
        status_code = 200
        is_redirect = False
        headers = {"content-type": "application/json", "content-length": "24"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def iter_bytes(self):
            yield b'{"count":0,"records":[]}'

    class Client:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

        def stream(self, method, path, **kwargs):
            calls.append({"method": method, "path": path, **kwargs})
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(child.httpx, "Client", Client)
    monkeypatch.setattr(child._CninfoHttpsTransport, "_get_enckey", lambda _self: "public-key")
    with child._CninfoHttpsTransport() as transport:
        for stream in ("identity", "dividend", "allotment", "share_change"):
            transport.read(stream, SYMBOL, AS_OF)

    assert constructed["base_url"] == "https://webapi.cninfo.com.cn"
    assert constructed["follow_redirects"] is False
    assert constructed["trust_env"] is False
    assert calls[0]["params"] == {"scode": SYMBOL}
    assert calls[1]["params"] == {"scode": SYMBOL}
    for call in calls[2:]:
        assert call["params"] == {
            "scode": SYMBOL,
            "sdate": "1990-01-01",
            "edate": AS_OF.isoformat(),
        }
    assert "entry_date" not in json.dumps(calls)
