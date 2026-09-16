"""Provider-neutral evidence contract for holding company actions.

``UNKNOWN`` is deliberately distinct from ``DETECTED``.  A network or
completeness failure must never be represented as a detected action, while an
empty provider payload is not clearance unless every required stream proves
that its response is complete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Final

COMPANY_ACTION_COVERAGE_START: Final = date(1990, 1, 1)
CNINFO_COMPANY_ACTION_METHOD_VERSION: Final = "cninfo-company-actions-v1"
_SYMBOL = re.compile(r"^[0-9]{6}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


class CompanyActionEvidenceStatus(StrEnum):
    """Three-state result; absence of evidence is never evidence of absence."""

    CLEAR = "clear"
    DETECTED = "detected"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CompanyActionStreamReceipt:
    """Bounded audit receipt for one provider stream, without its raw payload."""

    stream: str
    complete: bool
    record_count: int | None
    response_hash: str | None
    reason_code: str

    def __post_init__(self) -> None:
        if not self.stream or not self.stream.isascii():
            raise ValueError("company-action stream must be non-empty ASCII")
        if self.record_count is not None and self.record_count < 0:
            raise ValueError("company-action record_count cannot be negative")
        if self.response_hash is not None and _HASH.fullmatch(self.response_hash) is None:
            raise ValueError("company-action response_hash must be SHA-256 hex")
        if not self.reason_code or not self.reason_code.isascii():
            raise ValueError("company-action reason_code must be non-empty ASCII")


@dataclass(frozen=True, slots=True)
class CompanyActionEvidence:
    """Interval evidence returned by an isolated company-action reader.

    ``event_dates`` and ``event_kinds`` are parallel tuples.  The public
    coverage lower bound is intentionally provider-wide and reveals no holding
    entry date; the caller filters this evidence locally for the position's
    actual interval.
    """

    symbol: str
    status: CompanyActionEvidenceStatus
    coverage_from: date
    coverage_through: date
    knowledge_time: datetime
    event_dates: tuple[date, ...]
    event_kinds: tuple[str, ...]
    stream_receipts: tuple[CompanyActionStreamReceipt, ...]
    response_hash: str | None
    reason_code: str
    method_version: str

    def __post_init__(self) -> None:
        if _SYMBOL.fullmatch(self.symbol) is None:
            raise ValueError("company-action symbol must be exactly six digits")
        if self.coverage_from != COMPANY_ACTION_COVERAGE_START:
            raise ValueError("company-action coverage_from must use the public fixed start")
        if self.coverage_through < self.coverage_from:
            raise ValueError("company-action coverage interval is reversed")
        if self.knowledge_time.tzinfo is None or self.knowledge_time.utcoffset() is None:
            raise ValueError("company-action knowledge_time must be timezone-aware")
        if len(self.event_dates) != len(self.event_kinds):
            raise ValueError("company-action dates and kinds must be parallel")
        if any(
            event_date < self.coverage_from or event_date > self.coverage_through
            for event_date in self.event_dates
        ):
            raise ValueError("company-action event lies outside the evidence interval")
        if any(not kind or not kind.isascii() for kind in self.event_kinds):
            raise ValueError("company-action event kinds must be non-empty ASCII")
        if len({receipt.stream for receipt in self.stream_receipts}) != len(self.stream_receipts):
            raise ValueError("company-action stream receipts must be unique")
        if self.response_hash is not None and _HASH.fullmatch(self.response_hash) is None:
            raise ValueError("company-action response_hash must be SHA-256 hex")
        if not self.reason_code or not self.reason_code.isascii():
            raise ValueError("company-action reason_code must be non-empty ASCII")
        if not self.method_version or not self.method_version.isascii():
            raise ValueError("company-action method_version must be non-empty ASCII")
        if self.status is CompanyActionEvidenceStatus.CLEAR:
            if self.event_dates or not self.stream_receipts:
                raise ValueError("CLEAR evidence must have receipts and no events")
            if not all(receipt.complete for receipt in self.stream_receipts):
                raise ValueError("CLEAR evidence requires every stream to be complete")
        elif self.status is CompanyActionEvidenceStatus.DETECTED:
            if not self.event_dates:
                raise ValueError("DETECTED evidence requires at least one event")
            identity = next(
                (receipt for receipt in self.stream_receipts if receipt.stream == "identity"),
                None,
            )
            if identity is None or not identity.complete:
                raise ValueError("DETECTED evidence requires complete identity evidence")
        elif self.event_dates:
            raise ValueError("UNKNOWN evidence cannot carry a trusted event")
