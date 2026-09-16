"""Private child protocol for bounded CNINFO company-action reads.

The only holding-derived network value is ``scode``.  The lower date bound is
the same public constant for every symbol, so a position's entry date never
leaves the machine.  This module emits only normalized evidence and hashes;
raw provider errors, response bodies and URLs never reach stdout.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final

import httpx

from ashare_lab.ports.company_actions import (
    CNINFO_COMPANY_ACTION_METHOD_VERSION,
    COMPANY_ACTION_COVERAGE_START,
    CompanyActionEvidence,
    CompanyActionEvidenceStatus,
    CompanyActionStreamReceipt,
)

_BASE: Final = "https://webapi.cninfo.com.cn"
_ENDPOINTS: Final = {
    "identity": "/api/sysapi/p_sysapi1133",
    "dividend": "/api/sysapi/p_sysapi1139",
    "allotment": "/api/stock/p_stock2232",
    "share_change": "/api/stock/p_stock2215",
}
_SYMBOL: Final = re.compile(r"^[0-9]{6}$")
_MAX_SYMBOLS: Final = 5
_MAX_REQUEST_BYTES: Final = 16_384
_MAX_RESPONSE_BYTES: Final = 4_194_304
_SINGLE_REQUEST_TIMEOUT_SECONDS: Final = 6.0


class _ProviderIssue(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class _ProviderResponse:
    document: Mapping[str, object]
    response_hash: str


@dataclass(frozen=True, slots=True)
class _ParsedStream:
    receipt: CompanyActionStreamReceipt
    events: tuple[tuple[date, str], ...] = ()


class _CninfoHttpsTransport:
    """Fixed-host transport with no redirects, environment proxy, or cookies."""

    def __init__(self) -> None:
        timeout = httpx.Timeout(
            _SINGLE_REQUEST_TIMEOUT_SECONDS,
            connect=3.0,
            write=3.0,
            pool=3.0,
        )
        self._client = httpx.Client(
            base_url=_BASE,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            headers={
                "Accept": "application/json",
                "Origin": _BASE,
                "Referer": f"{_BASE}/",
                "User-Agent": "a-share-lab-company-action-research/1",
                "X-Requested-With": "XMLHttpRequest",
            },
        )
        self._enckey: str | None = None

    def __enter__(self) -> _CninfoHttpsTransport:
        return self

    def __exit__(self, *_args: object) -> None:
        self._client.close()

    def read(self, stream: str, symbol: str, as_of: date) -> _ProviderResponse:
        path = _ENDPOINTS.get(stream)
        if path is None:
            raise _ProviderIssue("STREAM_NOT_ALLOWED")
        params = {"scode": symbol}
        if stream in {"allotment", "share_change"}:
            params.update(
                {
                    "sdate": COMPANY_ACTION_COVERAGE_START.isoformat(),
                    "edate": as_of.isoformat(),
                }
            )
        try:
            with self._client.stream(
                "POST",
                path,
                params=params,
                headers={"Accept-Enckey": self._get_enckey()},
            ) as response:
                if response.status_code != 200 or response.is_redirect:
                    raise _ProviderIssue("HTTP_RESPONSE_UNAVAILABLE")
                media_type = response.headers.get("content-type", "").lower()
                if "json" not in media_type:
                    raise _ProviderIssue("RESPONSE_MEDIA_TYPE_INVALID")
                declared = response.headers.get("content-length")
                if declared is not None:
                    try:
                        if int(declared) > _MAX_RESPONSE_BYTES:
                            raise _ProviderIssue("RESPONSE_TOO_LARGE")
                    except ValueError:
                        raise _ProviderIssue("RESPONSE_LENGTH_INVALID") from None
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > _MAX_RESPONSE_BYTES:
                        raise _ProviderIssue("RESPONSE_TOO_LARGE")
        except _ProviderIssue:
            raise
        except httpx.HTTPError:
            raise _ProviderIssue("NETWORK_UNAVAILABLE") from None
        try:
            document = json.loads(bytes(body))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _ProviderIssue("RESPONSE_JSON_INVALID") from None
        if not isinstance(document, dict):
            raise _ProviderIssue("RESPONSE_ENVELOPE_INVALID")
        canonical = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return _ProviderResponse(
            document=document, response_hash=hashlib.sha256(canonical).hexdigest()
        )

    def _get_enckey(self) -> str:
        if self._enckey is not None:
            return self._enckey
        try:
            import py_mini_racer
            from akshare.datasets import get_ths_js

            javascript = Path(get_ths_js("cninfo.js")).read_text(encoding="utf-8")
            runtime = py_mini_racer.MiniRacer()
            runtime.eval(javascript)
            value = runtime.call("getResCode1")
        except Exception:
            raise _ProviderIssue("ENCKEY_UNAVAILABLE") from None
        if not isinstance(value, str) or not value or len(value) > 512:
            raise _ProviderIssue("ENCKEY_UNAVAILABLE")
        self._enckey = value
        return value


def _read_symbols(
    symbols: tuple[str, ...],
    as_of: date,
    *,
    transport: object,
    clock=None,
) -> tuple[CompanyActionEvidence, ...]:
    resolved_clock = clock or (lambda: datetime.now(UTC))
    results: list[CompanyActionEvidence] = []
    for symbol in symbols:
        try:
            results.append(
                _read_symbol(
                    symbol,
                    as_of,
                    transport=transport,
                    clock=resolved_clock,
                )
            )
        except Exception:
            results.append(
                _unknown_evidence(
                    symbol,
                    as_of,
                    _knowledge_time(resolved_clock),
                    reason_code="SYMBOL_READ_FAILED",
                )
            )
    return tuple(results)


def _read_symbol(
    symbol: str,
    as_of: date,
    *,
    transport: object,
    clock,
) -> CompanyActionEvidence:
    identity = _fetch_stream(transport, "identity", symbol, as_of, _parse_identity)
    streams = [identity]
    if not identity.receipt.complete:
        return _build_evidence(symbol, as_of, _knowledge_time(clock), streams)
    streams.extend(
        (
            _fetch_stream(transport, "dividend", symbol, as_of, _parse_dividend),
            _fetch_stream(transport, "allotment", symbol, as_of, _parse_allotment),
            _fetch_stream(transport, "share_change", symbol, as_of, _parse_share_change),
        )
    )
    return _build_evidence(symbol, as_of, _knowledge_time(clock), streams)


def _fetch_stream(transport, stream, symbol, as_of, parser) -> _ParsedStream:
    try:
        response = transport.read(stream, symbol, as_of)
    except _ProviderIssue as exc:
        return _incomplete_stream(stream, exc.reason_code)
    except Exception:
        return _incomplete_stream(stream, "PROVIDER_CALL_FAILED")
    if not isinstance(response, _ProviderResponse):
        return _incomplete_stream(stream, "PROVIDER_RESPONSE_INVALID")
    try:
        return parser(response.document, response.response_hash, symbol, as_of)
    except Exception:
        return _incomplete_stream(stream, "STREAM_PARSE_FAILED", response.response_hash)


def _parse_identity(document, response_hash, symbol, _as_of) -> _ParsedStream:
    records, envelope_complete, reason = _records_and_count(document)
    complete = envelope_complete and len(records) == 1
    if complete:
        record = records[0]
        complete = isinstance(record, dict) and _profile_symbol(record) == symbol
        if not complete:
            reason = "IDENTITY_MISMATCH"
    elif reason == "STREAM_COMPLETE":
        reason = "IDENTITY_CARDINALITY_INVALID"
    return _parsed_stream("identity", complete, records, response_hash, reason)


def _parse_dividend(document, response_hash, symbol, as_of) -> _ParsedStream:
    records, envelope_complete, envelope_reason = _records_and_count(document)
    events: list[tuple[date, str]] = []
    rows_complete = True
    for record in records:
        if not isinstance(record, dict) or not {"F010N", "F011N", "F012N", "F020D"}.issubset(
            record
        ):
            rows_complete = False
            continue
        if not _optional_identity_matches(record, symbol):
            rows_complete = False
            continue
        raw_values = (record["F012N"], record["F010N"], record["F011N"])
        if not any(_has_explicit_numeric(value) for value in raw_values):
            rows_complete = False
            continue
        try:
            values = (
                (_nonnegative(record["F012N"]), "CASH_DIVIDEND"),
                (_nonnegative(record["F010N"]), "STOCK_DIVIDEND"),
                (_nonnegative(record["F011N"]), "CAPITAL_RESERVE_TRANSFER"),
            )
        except ValueError:
            rows_complete = False
            continue
        positive = tuple(kind for value, kind in values if value > 0)
        raw_date = record.get("F020D")
        if not positive:
            continue
        event_date = _optional_date(raw_date)
        if event_date is None:
            rows_complete = False
            continue
        if COMPANY_ACTION_COVERAGE_START <= event_date <= as_of:
            events.extend((event_date, kind) for kind in positive)
    complete = envelope_complete and rows_complete
    reason = (
        envelope_reason
        if not envelope_complete
        else ("STREAM_COMPLETE" if rows_complete else "DIVIDEND_RECORD_INVALID")
    )
    return _parsed_stream("dividend", complete, records, response_hash, reason, tuple(events))


def _parse_allotment(document, response_hash, symbol, as_of) -> _ParsedStream:
    records, envelope_complete, envelope_reason = _records_and_count(document)
    events: list[tuple[date, str]] = []
    rows_complete = True
    for record in records:
        if not isinstance(record, dict) or str(record.get("SECCODE", "")) != symbol:
            rows_complete = False
            continue
        event_date = _optional_date(record.get("F012D"))
        if event_date is None:
            rows_complete = False
            continue
        if event_date is not None:
            if not (COMPANY_ACTION_COVERAGE_START <= event_date <= as_of):
                rows_complete = False
                continue
            events.append((event_date, "RIGHTS_ISSUE"))
    complete = envelope_complete and rows_complete
    reason = (
        envelope_reason
        if not envelope_complete
        else ("STREAM_COMPLETE" if rows_complete else "ALLOTMENT_RECORD_INVALID")
    )
    return _parsed_stream("allotment", complete, records, response_hash, reason, tuple(events))


def _parse_share_change(document, response_hash, symbol, as_of) -> _ParsedStream:
    records, envelope_complete, envelope_reason = _records_and_count(document)
    events: list[tuple[date, str]] = []
    rows_complete = True
    parsed_rows: list[tuple[date, Decimal, str]] = []
    for record in records:
        if not isinstance(record, dict) or str(record.get("SECCODE", "")) != symbol:
            rows_complete = False
            continue
        event_date = _optional_date(record.get("VARYDATE"))
        reason_text = record.get("F002V")
        if event_date is None or not isinstance(reason_text, str) or not reason_text.strip():
            rows_complete = False
            continue
        try:
            total_shares = _positive(record.get("F003N"))
        except ValueError:
            rows_complete = False
            continue
        if not (COMPANY_ACTION_COVERAGE_START <= event_date <= as_of):
            rows_complete = False
            continue
        parsed_rows.append((event_date, total_shares, reason_text.strip()))
    parsed_rows.sort(key=lambda item: (item[0], item[1], item[2]))
    previous_total: Decimal | None = None
    for event_date, total_shares, reason_text in parsed_rows:
        kinds = _share_change_kinds(reason_text)
        changed = previous_total is not None and total_shares != previous_total
        if kinds:
            if COMPANY_ACTION_COVERAGE_START <= event_date <= as_of:
                events.extend((event_date, kind) for kind in kinds)
        elif (previous_total is None or changed) and (
            COMPANY_ACTION_COVERAGE_START <= event_date <= as_of
        ):
            # The public fixed-start query cannot prove that its first row is
            # merely a baseline.  Treat it, and every later unexplained total
            # share change, as a conservative event rather than false CLEAR.
            events.append((event_date, "OTHER_SHARE_CHANGE"))
        previous_total = total_shares
    complete = envelope_complete and rows_complete
    reason = (
        envelope_reason
        if not envelope_complete
        else ("STREAM_COMPLETE" if rows_complete else "SHARE_CHANGE_UNCLASSIFIED")
    )
    return _parsed_stream("share_change", complete, records, response_hash, reason, tuple(events))


def _build_evidence(symbol, as_of, knowledge_time, streams) -> CompanyActionEvidence:
    receipts = tuple(item.receipt for item in streams)
    events = sorted({event for item in streams for event in item.events})
    identity_complete = bool(receipts and receipts[0].stream == "identity" and receipts[0].complete)
    if identity_complete and events:
        status = CompanyActionEvidenceStatus.DETECTED
        reason_code = "COMPANY_ACTION_DETECTED"
    elif len(receipts) == 4 and all(receipt.complete for receipt in receipts):
        status = CompanyActionEvidenceStatus.CLEAR
        reason_code = "ALL_STREAMS_COMPLETE_NO_EVENT"
    else:
        status = CompanyActionEvidenceStatus.UNKNOWN
        events = []
        reason_code = next(
            (receipt.reason_code for receipt in receipts if not receipt.complete),
            "COMPANY_ACTION_EVIDENCE_INCOMPLETE",
        )
    response_hash = _aggregate_hash(receipts)
    return CompanyActionEvidence(
        symbol=symbol,
        status=status,
        coverage_from=COMPANY_ACTION_COVERAGE_START,
        coverage_through=as_of,
        knowledge_time=knowledge_time,
        event_dates=tuple(item[0] for item in events),
        event_kinds=tuple(item[1] for item in events),
        stream_receipts=receipts,
        response_hash=response_hash,
        reason_code=reason_code,
        method_version=CNINFO_COMPANY_ACTION_METHOD_VERSION,
    )


def _records_and_count(document: object) -> tuple[list[object], bool, str]:
    if not isinstance(document, Mapping):
        return [], False, "RESPONSE_ENVELOPE_INVALID"
    result_code = document.get("resultcode")
    code_is_success = (
        isinstance(result_code, int) and not isinstance(result_code, bool) and result_code == 200
    ) or (isinstance(result_code, str) and result_code == "200")
    if not code_is_success:
        return [], False, "PROVIDER_RESULT_CODE_INVALID"
    if str(document.get("resultmsg", "")).strip().lower() != "success":
        return [], False, "PROVIDER_RESULT_MESSAGE_INVALID"
    records = document.get("records")
    if not isinstance(records, list):
        return [], False, "RECORDS_MISSING"
    if "count" not in document or "total" not in document:
        return records, False, "DECLARED_COUNT_MISSING"
    try:
        count = _exact_nonnegative_integer(document["count"])
        total = _exact_nonnegative_integer(document["total"])
    except ValueError:
        return records, False, "DECLARED_COUNT_INVALID"
    if count != total or count != len(records):
        return records, False, "DECLARED_COUNT_MISMATCH"
    return records, True, "STREAM_COMPLETE"


def _profile_symbol(record: Mapping[str, object]) -> str | None:
    value = record.get("ASECCODE")
    return None if value is None else str(value)


def _optional_identity_matches(record: Mapping[str, object], symbol: str) -> bool:
    present = [str(record[key]) for key in ("ASECCODE", "SECCODE") if key in record]
    return not present or all(value == symbol for value in present)


def _share_change_kinds(reason_text: str) -> tuple[str, ...]:
    result: list[str] = []
    if "配股" in reason_text:
        result.append("RIGHTS_ISSUE")
    if "送股" in reason_text or "送红股" in reason_text:
        result.append("STOCK_DIVIDEND")
    if "转增" in reason_text:
        result.append("CAPITAL_RESERVE_TRANSFER")
    return tuple(result)


def _nonnegative(value: object) -> Decimal:
    if value in (None, "", "--", "-"):
        return Decimal(0)
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("invalid numeric field") from None
    if not result.is_finite() or result < 0:
        raise ValueError("invalid numeric field")
    return result


def _has_explicit_numeric(value: object) -> bool:
    return value not in (None, "", "--", "-")


def _positive(value: object) -> Decimal:
    result = _nonnegative(value)
    if result <= 0:
        raise ValueError("invalid positive numeric field")
    return result


def _exact_nonnegative_integer(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("invalid integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError("invalid integer") from None
    if result < 0 or str(result) != str(value).strip():
        raise ValueError("invalid integer")
    return result


def _optional_date(value: object) -> date | None:
    if not _has_value(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return bool(str(value).strip()) and str(value).strip().lower() not in {"nan", "nat", "none"}


def _parsed_stream(
    stream,
    complete,
    records,
    response_hash,
    reason,
    events=(),
) -> _ParsedStream:
    return _ParsedStream(
        receipt=CompanyActionStreamReceipt(
            stream=stream,
            complete=bool(complete),
            record_count=len(records),
            response_hash=response_hash,
            reason_code="STREAM_COMPLETE" if complete else reason,
        ),
        events=tuple(events),
    )


def _incomplete_stream(stream, reason, response_hash=None) -> _ParsedStream:
    return _ParsedStream(
        receipt=CompanyActionStreamReceipt(
            stream=stream,
            complete=False,
            record_count=None,
            response_hash=response_hash,
            reason_code=reason,
        )
    )


def _aggregate_hash(receipts: Sequence[CompanyActionStreamReceipt]) -> str | None:
    hashable = tuple(receipt for receipt in receipts if receipt.response_hash is not None)
    if not hashable:
        return None
    canonical = "\n".join(
        f"{receipt.stream}:{receipt.response_hash}"
        for receipt in sorted(hashable, key=lambda x: x.stream)
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _knowledge_time(clock) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("knowledge clock must be timezone-aware")
    return value


def _unknown_evidence(symbol, as_of, knowledge_time, *, reason_code):
    return CompanyActionEvidence(
        symbol=symbol,
        status=CompanyActionEvidenceStatus.UNKNOWN,
        coverage_from=COMPANY_ACTION_COVERAGE_START,
        coverage_through=as_of,
        knowledge_time=knowledge_time,
        event_dates=(),
        event_kinds=(),
        stream_receipts=(),
        response_hash=None,
        reason_code=reason_code,
        method_version=CNINFO_COMPANY_ACTION_METHOD_VERSION,
    )


def _to_wire(evidence: CompanyActionEvidence) -> dict[str, object]:
    return {
        "symbol": evidence.symbol,
        "status": evidence.status.value,
        "coverage_from": evidence.coverage_from.isoformat(),
        "coverage_through": evidence.coverage_through.isoformat(),
        "knowledge_time": evidence.knowledge_time.isoformat(),
        "event_dates": [value.isoformat() for value in evidence.event_dates],
        "event_kinds": list(evidence.event_kinds),
        "stream_receipts": [
            {
                "stream": receipt.stream,
                "complete": receipt.complete,
                "record_count": receipt.record_count,
                "response_hash": receipt.response_hash,
                "reason_code": receipt.reason_code,
            }
            for receipt in evidence.stream_receipts
        ],
        "response_hash": evidence.response_hash,
        "reason_code": evidence.reason_code,
        "method_version": evidence.method_version,
    }


def _read_request() -> tuple[tuple[str, ...], date]:
    raw = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
    if len(raw) > _MAX_REQUEST_BYTES:
        raise ValueError("request too large")
    document = json.loads(raw)
    if not isinstance(document, dict) or set(document) != {"symbols", "as_of"}:
        raise ValueError("request shape invalid")
    raw_symbols = document["symbols"]
    if not isinstance(raw_symbols, list) or len(raw_symbols) > _MAX_SYMBOLS:
        raise ValueError("symbol scope invalid")
    symbols = tuple(raw_symbols)
    if any(not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None for symbol in symbols):
        raise ValueError("symbol invalid")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbol repeated")
    as_of = date.fromisoformat(document["as_of"])
    if as_of < COMPANY_ACTION_COVERAGE_START:
        raise ValueError("cutoff invalid")
    return symbols, as_of


def main() -> int:
    try:
        symbols, as_of = _read_request()
        if symbols:
            with _CninfoHttpsTransport() as transport:
                results = _read_symbols(
                    symbols,
                    as_of,
                    transport=transport,
                )
        else:
            results = ()
        document = {
            "method_version": CNINFO_COMPANY_ACTION_METHOD_VERSION,
            "as_of": as_of.isoformat(),
            "results": [_to_wire(item) for item in results],
        }
        sys.stdout.write(json.dumps(document, ensure_ascii=True, separators=(",", ":")))
        return 0
    except Exception:
        # The parent treats any non-zero exit as UNKNOWN.  Never print the raw
        # exception, URL, response body or dynamic request header.
        sys.stdout.write('{"status":"company_action_read_failed"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
