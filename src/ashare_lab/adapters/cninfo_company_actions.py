"""Bounded parent adapter for CNINFO company-action evidence.

The parent passes only normalized symbols and a public calendar cutoff to a
disposable child process.  It never passes a holding object, entry date, cost,
units, account value, weight, or database identifier.  Provider errors are
collapsed to stable reason codes and never surface a raw URL or exception.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Final

from ashare_lab.ports.company_actions import (
    CNINFO_COMPANY_ACTION_METHOD_VERSION,
    COMPANY_ACTION_COVERAGE_START,
    CompanyActionEvidence,
    CompanyActionEvidenceStatus,
    CompanyActionStreamReceipt,
)

_SYMBOL: Final = re.compile(r"^[0-9]{6}$")
_HASH: Final = re.compile(r"^[0-9a-f]{64}$")
_MAX_SYMBOLS: Final = 5
_MAX_CHILD_BYTES: Final = 1_048_576
_EXPECTED_STREAMS: Final = frozenset({"identity", "dividend", "allotment", "share_change"})
_RESULT_KEYS: Final = {
    "symbol",
    "status",
    "coverage_from",
    "coverage_through",
    "knowledge_time",
    "event_dates",
    "event_kinds",
    "stream_receipts",
    "response_hash",
    "reason_code",
    "method_version",
}
_RECEIPT_KEYS: Final = {
    "stream",
    "complete",
    "record_count",
    "response_hash",
    "reason_code",
}

# Kept as a module binding so tests can prove the exact child request without
# enabling provider networking.
_runner = subprocess.run


class _WireError(ValueError):
    pass


def read_cninfo_company_actions(
    symbols: Sequence[str],
    as_of: date,
    timeout_seconds: float = 45,
) -> tuple[CompanyActionEvidence, ...]:
    """Read current CNINFO metadata for at most five exact A-share codes.

    Invalid caller input is rejected before any process or network activity.
    Operational/provider failures return one ``UNKNOWN`` item per requested
    symbol so downstream holding logic can fail closed without losing the rest
    of the evening report.
    """

    requested = _validate_request(symbols, as_of, timeout_seconds)
    if not requested:
        return ()
    request = {"symbols": list(requested), "as_of": as_of.isoformat()}
    try:
        completed = _runner(
            [sys.executable, "-m", "ashare_lab.cli.cninfo_company_actions_read"],
            input=json.dumps(request, ensure_ascii=True, separators=(",", ":")),
            text=True,
            capture_output=True,
            timeout=float(timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _unknown_all(requested, as_of, "CHILD_DEADLINE_EXCEEDED")
    except OSError:
        return _unknown_all(requested, as_of, "CHILD_PROCESS_UNAVAILABLE")

    stdout = getattr(completed, "stdout", None)
    if not isinstance(stdout, str) or len(stdout.encode("utf-8")) > _MAX_CHILD_BYTES:
        return _unknown_all(requested, as_of, "CHILD_RESPONSE_INVALID")
    if getattr(completed, "returncode", 1) != 0:
        return _unknown_all(requested, as_of, "CHILD_READ_FAILED")
    try:
        document = json.loads(stdout)
        return _parse_document(document, requested=requested, as_of=as_of)
    except (TypeError, ValueError, KeyError):
        return _unknown_all(requested, as_of, "CHILD_PROTOCOL_INVALID")


def _validate_request(
    symbols: Sequence[str], as_of: date, timeout_seconds: float
) -> tuple[str, ...]:
    if isinstance(symbols, (str, bytes)) or not isinstance(symbols, Sequence):
        raise TypeError("symbols must be a sequence of six-digit strings")
    requested = tuple(symbols)
    if len(requested) > _MAX_SYMBOLS:
        raise ValueError("CNINFO company-action reads are bounded to five symbols")
    if any(
        not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None for symbol in requested
    ):
        raise ValueError("CNINFO company-action symbols must be exact six-digit strings")
    if len(set(requested)) != len(requested):
        raise ValueError("CNINFO company-action symbols cannot repeat")
    if not isinstance(as_of, date) or isinstance(as_of, datetime):
        raise TypeError("as_of must be a date")
    if as_of < COMPANY_ACTION_COVERAGE_START:
        raise ValueError("as_of predates the fixed company-action coverage start")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise TypeError("timeout_seconds must be numeric")
    if not 0 < float(timeout_seconds) <= 120:
        raise ValueError("timeout_seconds must be in (0, 120]")
    return requested


def _parse_document(
    document: object, *, requested: tuple[str, ...], as_of: date
) -> tuple[CompanyActionEvidence, ...]:
    if not isinstance(document, dict) or set(document) != {
        "method_version",
        "as_of",
        "results",
    }:
        raise _WireError("invalid child envelope")
    if document["method_version"] != CNINFO_COMPANY_ACTION_METHOD_VERSION:
        raise _WireError("method mismatch")
    if document["as_of"] != as_of.isoformat():
        raise _WireError("cutoff mismatch")
    raw_results = document["results"]
    if not isinstance(raw_results, list) or len(raw_results) != len(requested):
        raise _WireError("result cardinality mismatch")
    parsed = tuple(_parse_result(item, as_of=as_of) for item in raw_results)
    if tuple(item.symbol for item in parsed) != requested:
        raise _WireError("result identity/order mismatch")
    return parsed


def _parse_result(item: object, *, as_of: date) -> CompanyActionEvidence:
    if not isinstance(item, dict) or set(item) != _RESULT_KEYS:
        raise _WireError("invalid result shape")
    if item["coverage_from"] != COMPANY_ACTION_COVERAGE_START.isoformat():
        raise _WireError("invalid coverage start")
    if item["coverage_through"] != as_of.isoformat():
        raise _WireError("invalid coverage cutoff")
    if item["method_version"] != CNINFO_COMPANY_ACTION_METHOD_VERSION:
        raise _WireError("invalid result method")
    knowledge_time = datetime.fromisoformat(_required_text(item["knowledge_time"]))
    status = CompanyActionEvidenceStatus(_required_text(item["status"]))
    dates = _date_tuple(item["event_dates"])
    kinds = _text_tuple(item["event_kinds"])
    raw_receipts = item["stream_receipts"]
    if not isinstance(raw_receipts, list):
        raise _WireError("invalid stream receipts")
    receipts = tuple(_parse_receipt(value) for value in raw_receipts)
    streams = frozenset(receipt.stream for receipt in receipts)
    if status is CompanyActionEvidenceStatus.CLEAR and streams != _EXPECTED_STREAMS:
        raise _WireError("CLEAR result lacks a required stream")
    if status is CompanyActionEvidenceStatus.DETECTED:
        identity = next(
            (receipt for receipt in receipts if receipt.stream == "identity"),
            None,
        )
        if identity is None or not identity.complete:
            raise _WireError("DETECTED result lacks complete identity evidence")
    response_hash = item["response_hash"]
    if response_hash is not None:
        response_hash = _required_hash(response_hash)
    if status is not CompanyActionEvidenceStatus.UNKNOWN and response_hash is None:
        raise _WireError("trusted result lacks response hash")
    return CompanyActionEvidence(
        symbol=_required_text(item["symbol"]),
        status=status,
        coverage_from=COMPANY_ACTION_COVERAGE_START,
        coverage_through=as_of,
        knowledge_time=knowledge_time,
        event_dates=dates,
        event_kinds=kinds,
        stream_receipts=receipts,
        response_hash=response_hash,
        reason_code=_required_text(item["reason_code"]),
        method_version=_required_text(item["method_version"]),
    )


def _parse_receipt(item: object) -> CompanyActionStreamReceipt:
    if not isinstance(item, dict) or set(item) != _RECEIPT_KEYS:
        raise _WireError("invalid receipt shape")
    complete = item["complete"]
    if not isinstance(complete, bool):
        raise _WireError("receipt complete must be boolean")
    count = item["record_count"]
    if count is not None and (isinstance(count, bool) or not isinstance(count, int)):
        raise _WireError("receipt count must be integer or null")
    response_hash = item["response_hash"]
    if response_hash is not None:
        response_hash = _required_hash(response_hash)
    return CompanyActionStreamReceipt(
        stream=_required_text(item["stream"]),
        complete=complete,
        record_count=count,
        response_hash=response_hash,
        reason_code=_required_text(item["reason_code"]),
    )


def _required_text(value: object) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise _WireError("expected non-empty ASCII text")
    return value


def _required_hash(value: object) -> str:
    result = _required_text(value)
    if _HASH.fullmatch(result) is None:
        raise _WireError("expected SHA-256 hex")
    return result


def _date_tuple(value: object) -> tuple[date, ...]:
    if not isinstance(value, list):
        raise _WireError("expected date list")
    return tuple(date.fromisoformat(_required_text(item)) for item in value)


def _text_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _WireError("expected text list")
    return tuple(_required_text(item) for item in value)


def _unknown_all(
    symbols: tuple[str, ...], as_of: date, reason_code: str
) -> tuple[CompanyActionEvidence, ...]:
    now = datetime.now(UTC)
    return tuple(
        CompanyActionEvidence(
            symbol=symbol,
            status=CompanyActionEvidenceStatus.UNKNOWN,
            coverage_from=COMPANY_ACTION_COVERAGE_START,
            coverage_through=as_of,
            knowledge_time=now,
            event_dates=(),
            event_kinds=(),
            stream_receipts=(),
            response_hash=None,
            reason_code=reason_code,
            method_version=CNINFO_COMPANY_ACTION_METHOD_VERSION,
        )
        for symbol in symbols
    )
