from __future__ import annotations

import hashlib
from contextlib import contextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from ashare_lab.adapters import candidate_financial_evidence as adapter
from ashare_lab.adapters import official_document_evidence as documents

NOW = datetime(2026, 10, 11, 17, tzinfo=ZoneInfo("Asia/Shanghai"))
SYMBOL = "688029.SH"
URL = "https://static.cninfo.com.cn/finalpage/2026-08-20/123456.PDF"


class Response:
    content = b"{}"

    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def row(ident="123456", title="2026年半年度报告"):
    return {
        "announcementId": ident, "announcementTitle": title, "secCode": "688029",
        "orgId": "issuer", "announcementTime": int((NOW - timedelta(days=1)).timestamp() * 1000),
        "adjunctUrl": f"finalpage/2026-10-10/{ident}.PDF",
    }


def collect(rows):
    client = SimpleNamespace(post=lambda *a, **k: Response({
        "totalAnnouncement": len(rows), "announcements": rows, "hasMore": False,
    }))
    return adapter.collect_announcements(client, SYMBOL, NOW, {"688029": "issuer"})


def test_exact_duplicate_rows_do_not_prove_provider_coverage():
    manifest = collect([row(), row(), row("second")])
    assert manifest["error"] == "OFFICIAL_MANIFEST_DUPLICATE_ROW_UNVERIFIED"


def test_conflicting_duplicate_or_wrong_org_cannot_pass():
    assert collect([row(), row(title="changed")])["error"] == "OFFICIAL_MANIFEST_CONFLICTING_DUPLICATE"
    wrong = row()
    wrong["orgId"] = "different"
    assert collect([wrong])["error"] == "OFFICIAL_MANIFEST_IDENTITY_OR_PAGINATION_INVALID"


def test_repeated_page_is_not_treated_as_a_legitimate_duplicate_row(monkeypatch):
    pages = iter([
        {"totalAnnouncement": 2, "announcements": [row()], "hasMore": True},
        {"totalAnnouncement": 2, "announcements": [row()], "hasMore": False},
    ])
    client = SimpleNamespace(post=lambda *a, **k: Response(next(pages)))
    def unable_to_partition(*args):
        raise ValueError("OFFICIAL_MANIFEST_SINGLE_DAY_PAGINATION_UNVERIFIED")
    monkeypatch.setattr(adapter, "_partitioned_announcement_pages", unable_to_partition)
    assert adapter.collect_announcements(client, SYMBOL, NOW, {"688029": "issuer"})["error"] == "OFFICIAL_MANIFEST_SINGLE_DAY_PAGINATION_UNVERIFIED"


def test_disjoint_complete_date_windows_recover_unstable_pagination():
    rows = []
    for index in range(40):
        item = row(str(index))
        item["announcementTime"] = int((NOW - timedelta(days=index * 4 + 1)).timestamp() * 1000)
        rows.append(item)

    def post(*args, **kwargs):
        start, end = kwargs["data"]["seDate"].split("~")
        selected = [r for r in rows if start <= datetime.fromtimestamp(r["announcementTime"] / 1000, NOW.tzinfo).date().isoformat() <= end]
        return Response({"totalAnnouncement": len(selected), "announcements": selected[:30], "hasMore": len(selected) > 30})

    manifest = adapter.collect_announcements(SimpleNamespace(post=post), SYMBOL, NOW, {"688029": "issuer"}, _partitioned=True)
    assert manifest["complete"] is True
    assert manifest["record_count"] == manifest["raw_record_count"] == 40
    assert len(manifest["partition_windows"]) > 1


def test_future_duplicate_still_cannot_prove_coverage():
    future = row()
    future["announcementTime"] = int((NOW + timedelta(hours=1)).timestamp() * 1000)
    manifest = collect([future, future])
    assert manifest["error"] == "OFFICIAL_MANIFEST_DUPLICATE_ROW_UNVERIFIED"


def item(url=URL):
    return {"announcement_id": "123456", "title": "2026年半年度报告", "url": url, "published_at": (NOW - timedelta(days=2)).isoformat()}


class PdfClient:
    def __init__(self, binary=b"%PDF-1.7\npublic fixture"):
        self.binary, self.calls = binary, 0

    @contextmanager
    def stream(self, method, url, **kwargs):
        self.calls += 1
        assert kwargs["follow_redirects"] is False
        yield SimpleNamespace(
            raise_for_status=lambda: None, status_code=200, url=url, headers={},
            iter_bytes=lambda: iter([self.binary]),
        )


def extract(_binary, _timeout):
    return {"text": "公开正文财务报表" * 80, "page_count": 2, "extractor": "pypdf"}


def targeted(client, tmp_path, items=None, **kwargs):
    selected = items or [item()]
    return documents.collect_targeted_documents(
        client, {"symbol": SYMBOL, "complete": True, "items": selected},
        symbol=SYMBOL, financial_period="2026-06-30", knowledge_time=NOW,
        cache_dir=tmp_path, selector=lambda *_: selected, extractor=extract, **kwargs,
    )


def test_complete_pdf_is_hashed_cached_and_bound_to_manifest(tmp_path):
    client = PdfClient()
    first = targeted(client, tmp_path)[0]
    assert first["extraction_status"] == "complete"
    assert first["sha256"] == hashlib.sha256(client.binary).hexdigest()
    assert first["text_sha256"] == hashlib.sha256(first["text"].encode()).hexdigest()
    assert targeted(client, tmp_path)[0]["cache_hit"] is True
    assert client.calls == 1


def test_corrupt_cache_is_not_reused(tmp_path):
    client = PdfClient()
    targeted(client, tmp_path)
    next(tmp_path.glob("*.pdf")).write_bytes(b"tampered")
    assert targeted(client, tmp_path)[0]["extraction_status"] == "complete"
    assert client.calls == 2


def test_untrusted_url_and_future_publication_never_requested(tmp_path):
    client = PdfClient()
    bad = [item("https://evil.example/file.pdf"), {**item(), "published_at": (NOW + timedelta(hours=1)).isoformat()}]
    assert all(result["error"] == "OFFICIAL_DOCUMENT_IDENTITY_OR_TIME_INVALID" for result in targeted(client, tmp_path, bad))
    assert client.calls == 0


def test_deadline_marks_pending_documents_instead_of_truncating_to_pass(tmp_path):
    client = PdfClient()
    ticks = iter([0, 20])
    result = targeted(client, tmp_path, clock=lambda: next(ticks))[0]
    assert result["error"] == "OFFICIAL_DOCUMENT_DEADLINE_EXCEEDED"
    assert result["extraction_status"] != "complete"
    assert client.calls == 0


def test_non_pdf_and_oversize_are_unknown(tmp_path, monkeypatch):
    assert targeted(PdfClient(b"not a PDF"), tmp_path)[0]["error"] == "OFFICIAL_DOCUMENT_NOT_PDF"
    monkeypatch.setattr(documents, "MAX_PDF_BYTES", 3)
    assert targeted(PdfClient(), tmp_path)[0]["error"] == "OFFICIAL_DOCUMENT_TOO_LARGE"
