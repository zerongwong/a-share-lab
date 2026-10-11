"""Bounded, provenance-bound CNINFO PDF extraction for public candidates.

These receipts are evidence, not verdicts. No web text is ever executed or used
as instructions. A corrupt, scanned, oversized or truncated document remains
unknown. A cached PDF is reusable only when its binary and extracted text hashes
both match; the decision layer still checks its issuer, period and publication.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

MAX_PDF_BYTES = 15_000_000
MAX_DOCUMENTS = 16
MAX_PAGES = 700
MAX_TEXT_CHARS = 900_000
MAX_BATCH_TEXT_CHARS = 2_000_000
EXTRACTION_VERSION = "cninfo-pdf-text-v2-strict"
_SYMBOL = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")


def _hash(binary: bytes) -> str:
    return hashlib.sha256(binary).hexdigest()


def _official_url(url: object) -> bool:
    if not isinstance(url, str):
        return False
    parsed = urlparse(url)
    return (
        parsed.scheme == "https" and parsed.netloc == "static.cninfo.com.cn"
        and parsed.path.lower().endswith(".pdf") and not parsed.query and not parsed.fragment
        and ".." not in parsed.path.split("/")
    )


def extract_pdf(binary: bytes) -> dict:
    """All pages, no OCR guesses or truncation. Executed in a timed subprocess."""
    from pypdf import PdfReader

    if not binary.startswith(b"%PDF-") or len(binary) > MAX_PDF_BYTES:
        raise ValueError("PDF_INVALID_OR_OVERSIZED")
    # Do not silently repair missing objects and then label the result complete.
    reader = PdfReader(BytesIO(binary), strict=True)
    if reader.is_encrypted or not 0 < len(reader.pages) <= MAX_PAGES:
        raise ValueError("PDF_ENCRYPTED_OR_PAGE_LIMIT")
    pages, size = [], 0
    for page in reader.pages:
        text = page.extract_text() or ""
        # A mostly image-only page cannot be certified as fully read. Blank
        # vector cover pages are allowed, but the overall report needs text.
        if len(text.strip()) < 30 and len(page.images):
            raise ValueError("PDF_IMAGE_PAGE_REQUIRES_OCR")
        size += len(text)
        if size > MAX_TEXT_CHARS:
            raise ValueError("PDF_TEXT_LIMIT")
        pages.append(text)
    text = "\n\f\n".join(pages)
    if len(text.strip()) < 250:
        raise ValueError("PDF_TEXT_INSUFFICIENT")
    return {"text": text, "page_count": len(pages), "extractor": "pypdf"}


def _extract_bounded(binary: bytes, timeout: float) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", __name__, "--extract"], input=binary,
        capture_output=True, timeout=timeout, check=False,
    )
    if result.returncode != 0 or len(result.stdout) > MAX_TEXT_CHARS * 5:
        raise ValueError("PDF_TEXT_EXTRACTION_FAILED")
    payload = json.loads(result.stdout)
    if payload.get("error"):
        raise ValueError(payload["error"])
    return payload


def _cached_document(root: Path | None, url: str) -> tuple[dict, bytes] | None:
    if root is None:
        return None
    key = _hash(url.encode())
    try:
        text_path, pdf_path = root / f"{key}.json", root / f"{key}.pdf"
        if text_path.stat().st_size > MAX_TEXT_CHARS * 5 or pdf_path.stat().st_size > MAX_PDF_BYTES:
            return None
        value, binary = json.loads(text_path.read_text()), pdf_path.read_bytes()
        if (
            value.get("extraction_version") == EXTRACTION_VERSION
            and value.get("url") == url and value.get("extraction_status") == "complete"
            and binary.startswith(b"%PDF-") and value.get("sha256") == _hash(binary)
            and value.get("text_sha256") == _hash(value["text"].encode())
            and 0 < value["page_count"] <= MAX_PAGES and 250 <= len(value["text"]) <= MAX_TEXT_CHARS
        ):
            return value, binary
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def _save_cache(root: Path | None, value: dict, binary: bytes) -> None:
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    key = _hash(value["url"].encode())
    for suffix, data in (("pdf", binary), ("json", json.dumps(value, ensure_ascii=False).encode())):
        target = root / f"{key}.{suffix}"
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)


def _download(client, url: str, timeout: float) -> bytes:
    # Streaming size enforcement applies before buffering an attacker/provider
    # supplied body; redirects cannot escape the official static host.
    with client.stream("GET", url, timeout=timeout, follow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200 or not _official_url(str(response.url)):
            raise ValueError("OFFICIAL_DOCUMENT_HTTP_INVALID")
        declared = response.headers.get("content-length")
        if declared and int(declared) > MAX_PDF_BYTES:
            raise ValueError("OFFICIAL_DOCUMENT_TOO_LARGE")
        chunks, size = [], 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > MAX_PDF_BYTES:
                raise ValueError("OFFICIAL_DOCUMENT_TOO_LARGE")
            chunks.append(chunk)
        return b"".join(chunks)


def collect_targeted_documents(
    client, manifest: dict, *, symbol: str, financial_period: str | None,
    knowledge_time: datetime, cache_dir: Path | None = None,
    timeout_seconds: float = 18, selector=None, extractor=None, clock=time.monotonic,
) -> list[dict]:
    """Fetch only required current filings and substantive-risk documents.

    Per-issuer limits preserve independence: a slow issuer cannot consume an
    unbounded report run. Missing required documents are returned as failures;
    a later retry reuses successful complete PDF work from the private cache.
    """
    if not _SYMBOL.fullmatch(symbol) or manifest.get("symbol") != symbol or manifest.get("complete") is not True:
        return []
    if not 0 < timeout_seconds <= 45 or knowledge_time.utcoffset() is None:
        raise ValueError("invalid document deadline/time")
    if selector is None:
        from ashare_lab.services.material_risk_review import required_document_items
        selector = required_document_items
    selected = selector(manifest.get("items", []), financial_period, knowledge_time)
    deadline, total_text, results = clock() + timeout_seconds, 0, []
    attempted_downloads = 0
    root = Path(cache_dir) if cache_dir is not None else None
    for item in selected:
        receipt = {key: item.get(key) for key in ("announcement_id", "title", "url", "published_at")}
        receipt.update(symbol=symbol, extraction_status="unavailable")
        try:
            published = datetime.fromisoformat(item["published_at"])
            if published.utcoffset() is None or published > knowledge_time or not _official_url(item["url"]):
                raise ValueError("OFFICIAL_DOCUMENT_IDENTITY_OR_TIME_INVALID")
            cached = _cached_document(root, item["url"])
            if cached:
                value, binary = cached
                # Bind to the currently verified manifest; cache never supplies
                # issuer or period identity on its own.
                value = {**value, **receipt, "extraction_status": "complete", "cache_hit": True}
            else:
                if attempted_downloads >= MAX_DOCUMENTS:
                    raise ValueError("OFFICIAL_DOCUMENT_COUNT_LIMIT")
                remaining = deadline - clock()
                if remaining < 0.25:
                    raise ValueError("OFFICIAL_DOCUMENT_DEADLINE_EXCEEDED")
                attempted_downloads += 1
                binary = _download(client, item["url"], min(6.0, remaining))
                if not binary.startswith(b"%PDF-"):
                    raise ValueError("OFFICIAL_DOCUMENT_NOT_PDF")
                remaining = deadline - clock()
                if remaining < 0.25:
                    raise ValueError("OFFICIAL_DOCUMENT_DEADLINE_EXCEEDED")
                extracted = (extractor or _extract_bounded)(binary, min(6.0, remaining))
                text = extracted["text"]
                if not isinstance(text, str) or not 250 <= len(text) <= MAX_TEXT_CHARS:
                    raise ValueError("OFFICIAL_DOCUMENT_TEXT_INVALID")
                value = {
                    **receipt, **extracted, "sha256": _hash(binary), "text_sha256": _hash(text.encode()),
                    "retrieved_at": datetime.now(UTC).isoformat(), "extraction_status": "complete",
                    "extraction_complete": True, "extraction_version": EXTRACTION_VERSION,
                }
                _save_cache(root, value, binary)
            total_text += len(value["text"])
            if total_text > MAX_BATCH_TEXT_CHARS:
                raise ValueError("OFFICIAL_DOCUMENT_BATCH_TEXT_LIMIT")
            results.append(value)
        except Exception as exc:
            # Only known enum-like failures, never provider bodies/credentials.
            code = str(exc) if isinstance(exc, ValueError) and re.fullmatch(r"[A-Z_]+", str(exc)) else "OFFICIAL_DOCUMENT_FETCH_OR_PARSE_FAILED"
            results.append({**receipt, "error": code})
    return results


if __name__ == "__main__":
    try:
        payload = extract_pdf(sys.stdin.buffer.read(MAX_PDF_BYTES + 1))
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ValueError) and re.fullmatch(r"[A-Z_]+", str(exc)) else "OFFICIAL_DOCUMENT_EXTRACTION_INCOMPLETE"
        payload = {"error": reason}
    print(json.dumps(payload, ensure_ascii=False))
