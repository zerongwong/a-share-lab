"""The bounded evidence worker must finish some names before its deadline."""

from __future__ import annotations

import copy
import json
import subprocess
from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from ashare_lab.adapters import candidate_financial_evidence as adapter

_NOW = datetime(2026, 9, 22, 8, 40, tzinfo=ZoneInfo("Asia/Shanghai"))
_CUTOFF = date(2026, 9, 21)
_FIRST = "601298.SH"
_SECOND = "002292.SZ"


class _Response:
    content = b"{}"

    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def _capture_interleaved(monkeypatch):
    events, receipts = [], []

    def financial(_module, symbol, _knowledge_time):
        events.append(("financial", symbol))
        return {"profit": [{"netProfit": "1"}]}

    def execution(_module, symbol, _cutoff):
        events.append(("execution", symbol))
        return {"cutoff": _CUTOFF.isoformat(), "rows": [{"code": symbol}]}

    def announcements(_client, symbol, _knowledge_time, _identities):
        events.append(("announcements", symbol))
        return {"symbol": symbol, "complete": True, "items": [{"title": "regular report"}]}

    def master(_url):
        events.append(("master", ""))
        return _Response({"stockList": [{"code": "601298", "orgId": "a"}, {"code": "002292", "orgId": "b"}]})

    monkeypatch.setattr(adapter, "collect_financials", financial)
    monkeypatch.setattr(adapter, "collect_execution_status", execution)
    monkeypatch.setattr(adapter, "collect_announcements", announcements)
    adapter.collect_worker_batch(
        [_FIRST, _SECOND], cutoff=_CUTOFF, knowledge_time=_NOW, module=object(),
        client=SimpleNamespace(get=master), emit=lambda receipt: receipts.append(copy.deepcopy(receipt)),
    )
    return events, receipts


def test_first_official_manifest_finishes_before_next_candidate_financials(monkeypatch):
    events, receipts = _capture_interleaved(monkeypatch)
    assert events == [
        ("financial", _FIRST), ("execution", _FIRST), ("master", ""),
        ("announcements", _FIRST), ("financial", _SECOND),
        ("execution", _SECOND), ("announcements", _SECOND),
    ]
    first_complete = next(i for i, receipt in enumerate(receipts) if receipt["symbol"] == _FIRST and receipt["announcements"].get("complete"))
    second_started = next(i for i, receipt in enumerate(receipts) if receipt["symbol"] == _SECOND)
    assert first_complete < second_started


def test_deadline_preserves_first_complete_manifest_and_keeps_second_unknown(monkeypatch):
    _, receipts = _capture_interleaved(monkeypatch)
    second_started = next(i for i, receipt in enumerate(receipts) if receipt["symbol"] == _SECOND)
    partial_stdout = "\n".join(json.dumps(receipt) for receipt in receipts[:second_started + 1]) + "\n"

    def deadline(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=partial_stdout)

    result = adapter.read_candidate_evidence(
        [_FIRST, _SECOND], cutoff=_CUTOFF, knowledge_time=_NOW,
        timeout_seconds=1, runner=deadline,
    )
    assert result[_FIRST]["announcements"]["complete"] is True
    assert result[_SECOND]["announcements"]["error"] == "OFFICIAL_MANIFEST_PENDING"


def test_collected_manifest_binds_canonical_symbol_and_retains_full_pagination_check():
    published_ms = int(_NOW.timestamp() * 1000)
    posted = []

    def post(_url, **kwargs):
        posted.append(kwargs["data"])
        return _Response({
            "totalAnnouncement": 1,
            "announcements": [{
                "announcementId": "a1", "secCode": "601298", "announcementTime": published_ms,
                "adjunctUrl": "final.PDF", "announcementTitle": "2026年半年度报告",
            }],
            "hasMore": False,
        })

    manifest = adapter.collect_announcements(
        SimpleNamespace(post=post), _FIRST, _NOW, {"601298": "issuer"},
    )
    assert manifest["symbol"] == _FIRST
    assert manifest["complete"] is True
    assert manifest["record_count"] == manifest["total_provider_records"] == 1
    assert posted[0]["stock"] == "601298,issuer"
    assert adapter.collect_announcements(
        SimpleNamespace(post=post), _FIRST, _NOW, {},
    ) == {"symbol": _FIRST, "error": "OFFICIAL_ISSUER_IDENTITY_UNAVAILABLE"}
