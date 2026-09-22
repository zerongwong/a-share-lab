"""Local, explicitly reviewed candidate-announcement workflow; never auto-pass.

Only the download operation makes a request, and only to the exact official
CNINFO PDF URL already present in a fresh, complete candidate manifest. Drafts
start pending with every checklist item false. Registration validates the live
receipt again and requires explicit acknowledgement of real document review.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx

from ashare_lab.adapters.candidate_financial_evidence import (
    MAX_ANNOUNCEMENT_PAGES,
    METHOD_VERSION,
    PAGE_SIZE,
    content_hash,
)
from ashare_lab.services.candidate_evidence import _CHECKLIST, evaluate_announcement_review

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_SYMBOL = re.compile(r"\d{6}\.(?:SH|SZ|BJ)\Z")
_ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
_JSON_LIMIT = 2_000_000
_PDF_LIMIT = 50_000_000


class ReviewWorkflowError(ValueError):
    """A safe, actionable local-workflow failure without provider secrets."""


def _symbol(value: str) -> str:
    if not isinstance(value, str) or not _SYMBOL.fullmatch(value):
        raise ReviewWorkflowError("CANONICAL_SYMBOL_REQUIRED")
    return value


def _aware(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("aware timestamp required")
    return parsed


def _json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > _JSON_LIMIT:
        raise ReviewWorkflowError("LOCAL_JSON_MISSING_OR_TOO_LARGE")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ReviewWorkflowError("LOCAL_JSON_INVALID") from exc
    if not isinstance(result, dict):
        raise ReviewWorkflowError("LOCAL_JSON_INVALID")
    return result


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ReviewWorkflowError("REVIEW_FILE_MUST_BE_WITHIN_REVIEW_DIRECTORY")
    return resolved


def _fresh_receipt(cache_dir: Path, symbol: str, known_at: datetime) -> dict:
    _symbol(symbol)
    if known_at.utcoffset() is None:
        raise ReviewWorkflowError("KNOWLEDGE_TIME_MUST_BE_AWARE")
    path = _inside(cache_dir, cache_dir / f"{symbol}.json")
    receipt = _json(path)
    try:
        fetched = _aware(receipt["retrieved_at"])
        if (
            receipt["symbol"] != symbol
            or receipt["method_version"] != METHOD_VERSION
            or not timedelta(0) <= known_at - fetched <= timedelta(hours=24)
            or fetched.astimezone(_SHANGHAI).date() != known_at.astimezone(_SHANGHAI).date()
        ):
            raise ReviewWorkflowError("CURRENT_RECEIPT_STALE_OR_IDENTITY_MISMATCH")
        manifest = receipt["announcements"]
        items = manifest["items"]
        through = _aware(manifest["coverage_through"])
        if (
            manifest.get("error")
            or manifest["complete"] is not True
            or not isinstance(items, list)
            or not 0 < len(items) <= MAX_ANNOUNCEMENT_PAGES * PAGE_SIZE
            or not timedelta(0) <= known_at - through <= timedelta(hours=24)
            or manifest["content_hash"] != content_hash(items)
        ):
            raise ReviewWorkflowError("CURRENT_MANIFEST_INCOMPLETE_STALE_OR_CHANGED")
        ids = [item["announcement_id"] for item in items]
        if (
            len(ids) != len(set(ids))
            or any(not isinstance(value, str) or not _ID.fullmatch(value) for value in ids)
            or any(_aware(item["published_at"]) > known_at for item in items)
        ):
            raise ReviewWorkflowError("CURRENT_MANIFEST_IDENTITY_OR_TIME_INVALID")
        if not isinstance(receipt.get("financials"), Mapping):
            raise ReviewWorkflowError("CURRENT_FINANCIAL_RECEIPT_UNAVAILABLE")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ReviewWorkflowError):
            raise
        raise ReviewWorkflowError("CURRENT_RECEIPT_SCHEMA_INVALID") from exc
    return receipt


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
    except FileExistsError as exc:
        raise ReviewWorkflowError("FILE_EXISTS_NOT_OVERWRITTEN") from exc


def create_review_template(
    *, cache_dir: Path, review_dir: Path, symbol: str, known_at: datetime
) -> dict:
    receipt = _fresh_receipt(cache_dir, symbol, known_at)
    manifest = receipt["announcements"]
    template = {
        "method_version": METHOD_VERSION,
        "symbol": symbol,
        "status": "pending",
        "reviewer": "",
        "reviewed_at": None,
        "reason": "",
        "manifest_hash": manifest["content_hash"],
        "financial_hash": content_hash(receipt["financials"]),
        "financial_period": receipt["financials"].get("selected_period"),
        "checklist": dict.fromkeys(sorted(_CHECKLIST), False),
        "triage": [
            {"announcement_id": item["announcement_id"], "decision": "pending", "reason": ""}
            for item in manifest["items"]
        ],
        "documents": [],
        "available_documents": manifest["items"],
        "created_at": known_at.isoformat(),
        "instructions": (
            "下载并研读实际正文后填写，不能批量将检查项置真。逐条说明公告是否重大，"
            "填写财报与审计文件的角色、相对文件路径、SHA256和具体发现；"
            "核对结构化财务数值与原文。未知项目保持pending，不自动生成pass。"
        ),
    }
    stamp = known_at.strftime("%Y%m%dT%H%M%S%f%z")
    path = _inside(review_dir, review_dir / "drafts" / f"{symbol}-{stamp}.json")
    _write_new(path, template)
    return {"symbol": symbol, "status": "pending", "draft_path": str(path), "registered": False}


def validate_review(
    *, cache_dir: Path, review_dir: Path, symbol: str, review_path: Path, known_at: datetime
) -> dict:
    receipt = _fresh_receipt(cache_dir, symbol, known_at)
    source = _inside(review_dir, review_path)
    review = _json(source)
    review_hash = content_hash(review)
    gate, reasons = evaluate_announcement_review(
        receipt["announcements"],
        symbol=symbol,
        financial_hash=content_hash(receipt["financials"]),
        financial_period=receipt["financials"].get("selected_period"),
        knowledge_time=known_at,
        review_dir=review_dir,
        review_path=source,
    )
    if content_hash(_json(source)) != review_hash or content_hash(
        _fresh_receipt(cache_dir, symbol, known_at)
    ) != content_hash(receipt):
        raise ReviewWorkflowError("REVIEW_OR_RECEIPT_CHANGED_DURING_VALIDATION")
    return {
        "symbol": symbol,
        "gate": gate,
        "reasons": list(reasons),
        "review_hash": review_hash,
        "manifest_hash": receipt["announcements"]["content_hash"],
        "financial_hash": content_hash(receipt["financials"]),
        "validated_at": known_at.isoformat(),
        "registered": False,
        "scope": "official_content_review_only_not_final_buy_permission",
    }


def register_review(
    *,
    cache_dir: Path,
    review_dir: Path,
    symbol: str,
    review_path: Path,
    known_at: datetime,
    confirm_reviewed: bool,
    replace_existing: bool = False,
) -> dict:
    if confirm_reviewed is not True:
        raise ReviewWorkflowError("EXPLICIT_REAL_DOCUMENT_REVIEW_CONFIRMATION_REQUIRED")
    source = _inside(review_dir, review_path)
    result = validate_review(
        cache_dir=cache_dir,
        review_dir=review_dir,
        symbol=symbol,
        review_path=source,
        known_at=known_at,
    )
    if result["gate"] not in {"pass", "veto"}:
        raise ReviewWorkflowError("PENDING_OR_INVALID_REVIEW_CANNOT_BE_REGISTERED")
    review = _json(source)
    if content_hash(review) != result["review_hash"]:
        raise ReviewWorkflowError("REVIEW_CHANGED_DURING_REGISTRATION")
    destination_path = review_dir / f"{symbol}.json"
    if destination_path.is_symlink():
        raise ReviewWorkflowError("ACTIVE_REVIEW_SYMLINK_CANNOT_BE_REPLACED")
    destination = _inside(review_dir, destination_path)
    if source == destination:
        raise ReviewWorkflowError("REGISTER_A_DRAFT_NOT_THE_ACTIVE_FILE")
    if destination.exists() and not replace_existing:
        raise ReviewWorkflowError("ACTIVE_REVIEW_EXISTS_EXPLICIT_REPLACEMENT_REQUIRED")
    stamp = known_at.strftime("%Y%m%dT%H%M%S%f%z")
    history = _inside(review_dir, review_dir / "history" / symbol)
    if destination.exists():
        previous = _json(destination)
        _write_new(history / f"{stamp}-{content_hash(previous)}-previous.json", previous)
    registration = {
        **result,
        "registered": True,
        "reviewer": review["reviewer"],
        "reviewed_at": review["reviewed_at"],
        "registered_at": known_at.isoformat(),
    }
    audit_path = history / f"{stamp}-{result['review_hash']}-registration.json"
    if audit_path.exists():
        raise ReviewWorkflowError("FILE_EXISTS_NOT_OVERWRITTEN")
    temporary = _inside(review_dir, review_dir / f".{symbol}-{stamp}.pending.json")
    _write_new(temporary, review)
    # A failed activation must not leave a receipt claiming successful registration.
    temporary.replace(destination)
    _write_new(audit_path, registration)
    return {**registration, "active_path": str(destination)}


def list_review_status(
    *, cache_dir: Path, review_dir: Path, known_at: datetime, symbol: str | None = None
) -> dict:
    paths = (
        [cache_dir / f"{_symbol(symbol)}.json"]
        if symbol is not None
        else sorted(cache_dir.glob("*.json"))
    )
    rows = []
    for path in paths[:256]:
        code = path.stem
        try:
            receipt = _fresh_receipt(cache_dir, code, known_at)
            active = review_dir / f"{code}.json"
            if active.is_file():
                result = validate_review(
                    cache_dir=cache_dir,
                    review_dir=review_dir,
                    symbol=code,
                    review_path=active,
                    known_at=known_at,
                )
                gate, reasons = result["gate"], result["reasons"]
            else:
                gate, reasons = "unknown", ["OFFICIAL_DOCUMENT_CONTENT_REVIEW_REQUIRED"]
            rows.append(
                {
                    "symbol": code,
                    "gate": gate,
                    "reasons": reasons,
                    "retrieved_at": receipt["retrieved_at"],
                    "financial_period": receipt["financials"].get("selected_period"),
                    "announcement_count": len(receipt["announcements"]["items"]),
                }
            )
        except ReviewWorkflowError as exc:
            rows.append({"symbol": code, "gate": "unknown", "reasons": [str(exc)]})
    return {"candidates": rows, "truncated": len(paths) > 256, "auto_pass": False}


def _official_pdf_url(value: str) -> str:
    try:
        url = urlsplit(value)
        valid = (
            url.scheme == "https"
            and url.hostname == "static.cninfo.com.cn"
            and url.port in (None, 443)
            and url.username is None
            and url.password is None
            and url.path.startswith("/finalpage/")
            and url.path.lower().endswith(".pdf")
            and not url.query
            and not url.fragment
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ReviewWorkflowError("ONLY_MANIFEST_CNINFO_HTTPS_PDF_URL_IS_ALLOWED")
    return value


def download_review_document(
    *,
    cache_dir: Path,
    review_dir: Path,
    symbol: str,
    announcement_id: str,
    known_at: datetime,
    client: httpx.Client | None = None,
) -> dict:
    receipt = _fresh_receipt(cache_dir, symbol, known_at)
    matches = [
        row
        for row in receipt["announcements"]["items"]
        if row["announcement_id"] == announcement_id
    ]
    if len(matches) != 1:
        raise ReviewWorkflowError("DOCUMENT_ID_NOT_IN_CURRENT_MANIFEST")
    item = matches[0]
    url = _official_pdf_url(item["url"])
    owned = client is None
    active = client or httpx.Client(timeout=5.0, follow_redirects=False)
    started = time.monotonic()
    try:
        with active.stream("GET", url, follow_redirects=False, timeout=5.0) as response:
            response.raise_for_status()
            if (
                response.headers.get("content-length")
                and int(response.headers["content-length"]) > _PDF_LIMIT
            ):
                raise ReviewWorkflowError("DOCUMENT_EXCEEDS_SIZE_LIMIT")
            binary = bytearray()
            for chunk in response.iter_bytes():
                if time.monotonic() - started > 20:
                    raise ReviewWorkflowError("DOCUMENT_DOWNLOAD_DEADLINE_EXCEEDED")
                binary.extend(chunk)
                if len(binary) > _PDF_LIMIT:
                    raise ReviewWorkflowError("DOCUMENT_EXCEEDS_SIZE_LIMIT")
    except (httpx.HTTPError, ValueError) as exc:
        if isinstance(exc, ReviewWorkflowError):
            raise
        raise ReviewWorkflowError("OFFICIAL_DOCUMENT_DOWNLOAD_FAILED") from exc
    finally:
        if owned:
            active.close()
    if not binary.startswith(b"%PDF-"):
        raise ReviewWorkflowError("DOWNLOADED_DOCUMENT_IS_NOT_PDF")
    digest = hashlib.sha256(binary).hexdigest()
    relative = Path("documents") / symbol / f"{announcement_id}-{digest}.pdf"
    path = _inside(review_dir, review_dir / relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if (
            path.stat().st_size > _PDF_LIMIT
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ReviewWorkflowError("EXISTING_DOCUMENT_HASH_MISMATCH")
    else:
        with path.open("xb") as stream:
            stream.write(binary)
    return {
        "announcement_id": announcement_id,
        "url": url,
        "file": relative.as_posix(),
        "sha256": digest,
        "title": item["title"],
        "role": "",
        "findings": "",
        "downloaded": True,
        "read_confirmed": False,
        "registered": False,
    }
