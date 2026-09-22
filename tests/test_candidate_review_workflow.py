from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from ashare_lab.adapters.candidate_financial_evidence import METHOD_VERSION, content_hash
from ashare_lab.cli.candidate_review import main
from ashare_lab.services.candidate_evidence import _CHECKLIST
from ashare_lab.services.candidate_review_workflow import (
    ReviewWorkflowError,
    create_review_template,
    download_review_document,
    list_review_status,
    register_review,
    validate_review,
)

SYMBOL = "601298.SH"
KNOWN = datetime(2026, 9, 22, 1, 0, tzinfo=UTC)
PDF = b"%PDF-1.4\nsynthetic unit-test document, not a real filing\n%%EOF"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path):
    cache, review = tmp_path / "cache", tmp_path / "reviews"
    items = [
        {
            "announcement_id": "1001",
            "title": "2026年半年度报告",
            "published_at": "2026-08-29T00:00:00+00:00",
            "url": "https://static.cninfo.com.cn/finalpage/2026-08-29/1001.PDF",
        },
        {
            "announcement_id": "1002",
            "title": "2025年度报告及审计报告",
            "published_at": "2026-04-20T00:00:00+00:00",
            "url": "https://static.cninfo.com.cn/finalpage/2026-04-20/1002.PDF",
        },
    ]
    receipt = {
        "symbol": SYMBOL,
        "method_version": METHOD_VERSION,
        "retrieved_at": (KNOWN - timedelta(minutes=5)).isoformat(),
        "financials": {"selected_period": "2026-06-30"},
        "announcements": {
            "complete": True,
            "items": items,
            "content_hash": content_hash(items),
            "coverage_through": (KNOWN - timedelta(minutes=10)).isoformat(),
        },
    }
    _write(cache / f"{SYMBOL}.json", receipt)
    return {"cache_dir": cache, "review_dir": review, "symbol": SYMBOL, "known_at": KNOWN}


def _authored_review(workspace):
    result = create_review_template(**workspace)
    draft = Path(result["draft_path"])
    data = json.loads(draft.read_text())
    data.update(
        {
            "status": "pass",
            "reviewer": "unit-test analyst",
            "reviewed_at": KNOWN.isoformat(),
            "reason": "explicit synthetic test review",
            "checklist": dict.fromkeys(_CHECKLIST, True),
        }
    )
    for item, role in zip(
        data["available_documents"], ("latest_financial_report", "annual_audit"), strict=True
    ):
        file = Path("documents") / f"{item['announcement_id']}.pdf"
        path = workspace["review_dir"] / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PDF)
        data["documents"].append(
            {
                "announcement_id": item["announcement_id"],
                "url": item["url"],
                "role": role,
                "file": str(file),
                "sha256": hashlib.sha256(PDF).hexdigest(),
                "findings": "synthetic fixture only: document was supplied for this unit test",
            }
        )
    for row in data["triage"]:
        row.update({"decision": "document_reviewed", "reason": "synthetic test review"})
    _write(draft, data)
    return draft


def test_template_is_pending_and_does_not_register_or_claim_document_read(workspace):
    result = create_review_template(**workspace)
    draft = Path(result["draft_path"])
    data = json.loads(draft.read_text())
    assert result["registered"] is False
    assert data["status"] == "pending"
    assert data["reviewed_at"] is None
    assert data["reviewer"] == data["reason"] == ""
    assert data["checklist"] == dict.fromkeys(_CHECKLIST, False)
    assert all(row["decision"] == "pending" and row["reason"] == "" for row in data["triage"])
    assert data["documents"] == []
    assert not (workspace["review_dir"] / f"{SYMBOL}.json").exists()
    assert validate_review(**workspace, review_path=draft)["gate"] == "unknown"
    with pytest.raises(ReviewWorkflowError, match="PENDING_OR_INVALID"):
        register_review(**workspace, review_path=draft, confirm_reviewed=True)


def test_template_never_overwrites_existing_draft(workspace):
    create_review_template(**workspace)
    with pytest.raises(ReviewWorkflowError, match="EXISTS_NOT_OVERWRITTEN"):
        create_review_template(**workspace)


def test_real_review_registration_requires_confirmation_and_preserves_history(workspace):
    draft = _authored_review(workspace)
    result = validate_review(**workspace, review_path=draft)
    assert result["gate"] == "pass" and result["registered"] is False
    with pytest.raises(ReviewWorkflowError, match="EXPLICIT_REAL_DOCUMENT"):
        register_review(**workspace, review_path=draft, confirm_reviewed=False)
    registered = register_review(**workspace, review_path=draft, confirm_reviewed=True)
    active = Path(registered["active_path"])
    assert active.is_file() and registered["gate"] == "pass"
    assert len(list((workspace["review_dir"] / "history" / SYMBOL).glob("*.json"))) == 1
    later = workspace | {"known_at": KNOWN + timedelta(minutes=1)}
    with pytest.raises(ReviewWorkflowError, match="EXPLICIT_REPLACEMENT"):
        register_review(**later, review_path=draft, confirm_reviewed=True)
    register_review(**later, review_path=draft, confirm_reviewed=True, replace_existing=True)
    assert len(list((workspace["review_dir"] / "history" / SYMBOL).glob("*.json"))) == 3
    assert json.loads(active.read_text())["status"] == "pass"


def test_manifest_changes_invalidate_previously_reviewed_artifact(workspace):
    draft = _authored_review(workspace)
    path = workspace["cache_dir"] / f"{SYMBOL}.json"
    receipt = json.loads(path.read_text())
    receipt["announcements"]["items"][0]["title"] += "（更正后）"
    receipt["announcements"]["content_hash"] = content_hash(receipt["announcements"]["items"])
    _write(path, receipt)
    result = validate_review(**workspace, review_path=draft)
    assert result["gate"] == "unknown"
    assert "OFFICIAL_REVIEW_CONTENT_CHANGED_REVIEW_REQUIRED" in result["reasons"]


def test_failed_activation_does_not_claim_registered_in_history(workspace, monkeypatch):
    draft = _authored_review(workspace)

    def reject_activation(self, target):
        raise OSError("unit-test activation failure")

    monkeypatch.setattr(Path, "replace", reject_activation)
    with pytest.raises(OSError, match="activation failure"):
        register_review(**workspace, review_path=draft, confirm_reviewed=True)
    assert not (workspace["review_dir"] / f"{SYMBOL}.json").exists()
    assert not list((workspace["review_dir"] / "history").rglob("*-registration.json"))


def test_registration_cannot_redirect_another_symbol_via_active_symlink(workspace):
    draft = _authored_review(workspace)
    other = workspace["review_dir"] / "600000.SH.json"
    _write(other, {"status": "pending", "symbol": "600000.SH"})
    active = workspace["review_dir"] / f"{SYMBOL}.json"
    active.symlink_to(other)
    with pytest.raises(ReviewWorkflowError, match="ACTIVE_REVIEW_SYMLINK"):
        register_review(
            **workspace, review_path=draft, confirm_reviewed=True, replace_existing=True
        )
    assert json.loads(other.read_text())["symbol"] == "600000.SH"


def test_stale_or_future_receipt_cannot_create_or_register_review(workspace):
    for shifted in (KNOWN + timedelta(days=1), KNOWN - timedelta(hours=1)):
        with pytest.raises(ReviewWorkflowError, match="STALE_OR_IDENTITY"):
            create_review_template(**(workspace | {"known_at": shifted}))


def test_draft_and_document_escape_or_hash_mismatch_fail_closed(workspace, tmp_path):
    draft = _authored_review(workspace)
    outside = tmp_path / "outside.json"
    outside.write_bytes(draft.read_bytes())
    with pytest.raises(ReviewWorkflowError, match="WITHIN_REVIEW"):
        validate_review(**workspace, review_path=outside)
    link = workspace["review_dir"] / "drafts" / "escape.json"
    link.symlink_to(outside)
    with pytest.raises(ReviewWorkflowError, match="WITHIN_REVIEW"):
        validate_review(**workspace, review_path=link)
    data = json.loads(draft.read_text())
    data["documents"][0]["sha256"] = "0" * 64
    _write(draft, data)
    result = validate_review(**workspace, review_path=draft)
    assert result["gate"] == "unknown"
    assert "OFFICIAL_REVIEW_DOCUMENT_HASH_MISMATCH" in result["reasons"]


def test_status_lists_pending_without_fabricating_pass(workspace):
    result = list_review_status(**workspace)
    assert result["auto_pass"] is False
    assert result["candidates"][0]["gate"] == "unknown"
    assert result["candidates"][0]["announcement_count"] == 2


def test_download_is_bounded_official_pdf_not_confirmation(workspace):
    seen = []

    def respond(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=PDF)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = download_review_document(**workspace, announcement_id="1001", client=client)
    assert seen == ["https://static.cninfo.com.cn/finalpage/2026-08-29/1001.PDF"]
    assert result["read_confirmed"] is result["registered"] is False
    assert result["role"] == result["findings"] == ""
    file = workspace["review_dir"] / result["file"]
    assert file.read_bytes() == PDF
    assert result["sha256"] == hashlib.sha256(PDF).hexdigest()


@pytest.mark.parametrize("case", ("redirect", "html", "too_large"))
def test_download_rejects_redirect_nonpdf_and_oversize(workspace, case):
    def respond(request):
        if case == "redirect":
            return httpx.Response(302, headers={"location": "https://untrusted.example/a.pdf"})
        if case == "too_large":
            return httpx.Response(200, headers={"content-length": "50000001"}, content=PDF)
        return httpx.Response(200, content=b"<html>not a PDF</html>")

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        pytest.raises(ReviewWorkflowError),
    ):
        download_review_document(**workspace, announcement_id="1001", client=client)
    assert not (workspace["review_dir"] / "documents").exists()


def test_untrusted_manifest_url_never_reaches_network(workspace):
    path = workspace["cache_dir"] / f"{SYMBOL}.json"
    receipt = json.loads(path.read_text())
    receipt["announcements"]["items"][0]["url"] = "https://static.cninfo.com.cn.evil.example/a.pdf"
    receipt["announcements"]["content_hash"] = content_hash(receipt["announcements"]["items"])
    _write(path, receipt)

    def never_called(request):
        pytest.fail("untrusted URL reached network")

    with (
        httpx.Client(transport=httpx.MockTransport(never_called)) as client,
        pytest.raises(ReviewWorkflowError, match="ONLY_MANIFEST_CNINFO"),
    ):
        download_review_document(**workspace, announcement_id="1001", client=client)


def test_cli_pending_flow_never_sets_pass(workspace, capsys):
    options = [
        "--cache-dir",
        str(workspace["cache_dir"]),
        "--review-dir",
        str(workspace["review_dir"]),
    ]
    assert main([*options, "status"], _now=KNOWN) == 0
    assert json.loads(capsys.readouterr().out)["auto_pass"] is False
    assert main([*options, "template", SYMBOL], _now=KNOWN) == 0
    draft = json.loads(capsys.readouterr().out)["draft_path"]
    assert main([*options, "validate", SYMBOL, "--file", draft], _now=KNOWN) == 3
    assert json.loads(capsys.readouterr().out)["gate"] == "unknown"
    assert (
        main([*options, "register", SYMBOL, "--file", draft, "--confirm-reviewed"], _now=KNOWN) == 2
    )
    assert json.loads(capsys.readouterr().out)["registered"] is False
