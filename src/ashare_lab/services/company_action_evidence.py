"""Consent-gated, privacy-minimised company-action evidence for active holdings.

The external reader receives only a stable tuple of six-digit symbols and one
public coverage date.  Holding names, entry dates, costs, quantities, amounts,
weights and portfolio identifiers remain local.  Results are bound to the
exact holding revision, sanitised, and archived append-only; this module never
connects to a broker and never creates an order.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Any, Final
from uuid import uuid4

from ashare_lab import bootstrap
from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.ports.company_actions import (
    CNINFO_COMPANY_ACTION_METHOD_VERSION,
    CompanyActionEvidence,
    CompanyActionEvidenceStatus,
)
from ashare_lab.ports.market_data import normalize_symbol
from ashare_lab.services.company_action_lock import company_action_lock
from ashare_lab.services.holding_ledger import (
    ActiveHolding,
    ActiveHoldingPortfolio,
    get_active_holding_portfolio,
)
from ashare_lab.services.review_active_holdings import CompanyActionClearance

AUTHORIZATION_SCOPE: Final = "active_holding_symbols_only"
AUTHORIZED_FIELDS: Final = ("symbol",)
CONFIG_RELATIVE_PATH: Final = Path("scheduler/company-actions/config.json")
COMPANY_ACTION_EVIDENCE_METHOD_VERSION: Final = "local-cninfo-evidence-v2"
MAX_AUTHORIZED_HOLDINGS: Final = 8
FETCH_TIMEOUT_SECONDS: Final = 8
TOTAL_FETCH_BUDGET_SECONDS: Final = 45
MAX_PHASE_ATTEMPTS: Final = 3
UNKNOWN_RETRY_DELAY: Final = timedelta(minutes=15)
COMPANY_ACTION_EVIDENCE_TABLE: Final = "company_action_evidence_attempts"
_PHASES: Final = frozenset({"intraday", "eod"})
_REQUIRED_STREAMS: Final = frozenset({"identity", "dividend", "allotment", "share_change"})

EvidenceFetcher = Callable[..., Sequence[CompanyActionEvidence]]


@dataclass(frozen=True, slots=True)
class _AuthorizationSnapshot:
    path: Path
    fingerprint: str
    enabled: bool
    authorized_at: datetime


@dataclass(frozen=True, slots=True)
class _HoldingSnapshot:
    portfolio_id: str
    holding_version: int
    position_keys: tuple[str, ...]
    positions: tuple[ActiveHolding, ...]


@dataclass(frozen=True, slots=True)
class _SanitisedEvidence:
    portfolio_id: str
    position_key: str
    holding_version: int
    symbol: str
    as_of: date
    phase: str
    attempt: int
    status: CompanyActionEvidenceStatus
    coverage_from: date | None
    coverage_through: date | None
    knowledge_time: datetime
    events: tuple[dict[str, str], ...]
    provider_response_hash: str | None
    stream_receipts: tuple[dict[str, object], ...]
    reason_code: str
    provider_method_version: str
    local_method_version: str = COMPANY_ACTION_EVIDENCE_METHOD_VERSION


_CLEARANCE_FIELD_NAMES = frozenset(field.name for field in fields(CompanyActionClearance))
if "knowledge_time" not in _CLEARANCE_FIELD_NAMES:

    @dataclass(frozen=True, slots=True)
    class _EvidenceBackedClearance(CompanyActionClearance):
        """Compatibility bridge until all callers use knowledge-time clearance."""

        knowledge_time: datetime | None = None

else:
    _EvidenceBackedClearance = CompanyActionClearance


def company_action_config_path() -> Path:
    """Return the private, machine-local authorization document path."""

    return bootstrap.application_data_dir() / CONFIG_RELATIVE_PATH


def authorize_company_actions(
    *,
    confirmed: bool,
    config_path: Path | None = None,
) -> Path:
    """Persist the narrow grant; the explicit confirmation cannot be omitted."""

    if confirmed is not True:
        raise ValueError("company_action_authorization_requires_explicit_yes")
    path = _resolved_config_path(config_path)
    with company_action_lock(path, blocking=True) as acquired:
        if not acquired:  # pragma: no cover - a blocking lock either acquires or raises
            raise RuntimeError("company_action_authorization_lock_unavailable")
        existing = _read_config(path)
        if existing is not None and existing.enabled:
            # Re-running an idempotent authorize command must not manufacture a
            # new authorization event or disrupt an in-flight guarded read.
            return path
        _write_private_json(
            path,
            {
                "enabled": True,
                "scope": AUTHORIZATION_SCOPE,
                "authorized_fields": list(AUTHORIZED_FIELDS),
                "orders_enabled": False,
                "authorized_at": datetime.now(UTC).isoformat(),
            },
        )
    return path


def revoke_company_actions(
    *,
    confirmed: bool,
    config_path: Path | None = None,
) -> Path:
    """Disable future external reads without deleting the local audit trail."""

    if confirmed is not True:
        raise ValueError("company_action_revocation_requires_explicit_yes")
    path = _resolved_config_path(config_path)
    with company_action_lock(path, blocking=True) as acquired:
        if not acquired:  # pragma: no cover - a blocking lock either acquires or raises
            raise RuntimeError("company_action_authorization_lock_unavailable")
        existing = _read_config(path)
        _write_private_json(
            path,
            {
                "enabled": False,
                "scope": AUTHORIZATION_SCOPE,
                "authorized_fields": list(AUTHORIZED_FIELDS),
                "orders_enabled": False,
                "authorized_at": (
                    existing.authorized_at.isoformat()
                    if existing is not None
                    else datetime.now(UTC).isoformat()
                ),
            },
        )
    return path


def is_company_action_authorized(*, config_path: Path | None = None) -> bool:
    """Return true only for the exact, owner-private, symbol-only grant."""

    return _read_authorization(_resolved_config_path(config_path)) is not None


def refresh_and_load_company_action_clearances(
    repository: SQLiteRepository,
    as_of: date,
    reviewed_at: datetime,
    phase: str,
    fetcher: EvidenceFetcher | None = None,
    config_path: Path | None = None,
    allow_noncanonical_repository: bool = False,
) -> dict[str, CompanyActionClearance]:
    """Refresh current holdings and return only known CLEAR/DETECTED states.

    Every exception is contained at this boundary.  An empty mapping therefore
    means "no decision-grade automated evidence", never "all clear".  The
    caller must also consult :func:`is_company_action_authorized` before it
    considers any legacy manual metadata fallback.
    """

    resolved_config_path = _resolved_config_path(config_path)
    try:
        with company_action_lock(resolved_config_path, blocking=False) as acquired:
            if not acquired:
                return {}
            return _refresh_and_load(
                repository,
                as_of=as_of,
                reviewed_at=reviewed_at,
                phase=phase,
                fetcher=fetcher or _default_fetcher,
                config_path=resolved_config_path,
                allow_noncanonical_repository=allow_noncanonical_repository,
                explicitly_injected=fetcher is not None and config_path is not None,
            )
    except Exception:  # noqa: BLE001 - fail-closed service boundary
        return {}


def _refresh_and_load(
    repository: SQLiteRepository,
    *,
    as_of: date,
    reviewed_at: datetime,
    phase: str,
    fetcher: EvidenceFetcher,
    config_path: Path,
    allow_noncanonical_repository: bool,
    explicitly_injected: bool,
) -> dict[str, CompanyActionClearance]:
    if not isinstance(repository, SQLiteRepository):
        raise TypeError("repository must be SQLiteRepository")
    if not isinstance(as_of, date) or isinstance(as_of, datetime):
        raise TypeError("as_of must be a date")
    review_time = _aware_datetime(reviewed_at)
    normalized_phase = str(phase).strip().lower()
    if normalized_phase not in _PHASES or as_of > review_time.date():
        return {}

    if allow_noncanonical_repository is not True and not explicitly_injected:
        canonical_repository = (bootstrap.application_data_dir() / "research.db").resolve()
        if repository.db_path != canonical_repository:
            return {}

    authorization = _read_authorization(config_path)
    if authorization is None:
        return {}

    repository.initialize()
    portfolio = get_active_holding_portfolio(repository)
    snapshot = _holding_snapshot(portfolio, reviewed_at=review_time)
    if snapshot is None:
        # No active positions means no provider call and no synthetic evidence.
        return {}
    if any(position.entry_date > as_of for position in snapshot.positions):
        return {}

    cached = _cached_rows(
        repository,
        snapshot.positions,
        portfolio_id=snapshot.portfolio_id,
        holding_version=snapshot.holding_version,
        as_of=as_of,
        phase=normalized_phase,
    )
    if len(snapshot.positions) > MAX_AUTHORIZED_HOLDINGS:
        if _guard_unchanged(repository, snapshot, authorization, review_time):
            rows = tuple(
                _unknown_evidence(
                    position,
                    portfolio_id=snapshot.portfolio_id,
                    holding_version=snapshot.holding_version,
                    as_of=as_of,
                    phase=normalized_phase,
                    attempt=1,
                    knowledge_time=review_time,
                    reason_code="holding_limit_exceeded",
                )
                for position in snapshot.positions
                if position.position_key not in cached
            )
            _archive_evidence(repository, rows, snapshot=snapshot)
        return {}

    cache_read_at = datetime.now(UTC)
    known = _clearances_from_rows(cached, snapshot.positions, cache_read_at)
    missing = tuple(
        position
        for position in snapshot.positions
        if normalize_symbol(position.symbol) not in known
        and _cached_row_needs_refresh(cached.get(position.position_key), now=cache_read_at)
    )
    if not missing:
        return known
    if not _guard_unchanged(repository, snapshot, authorization, review_time):
        return {}

    symbols = tuple(normalize_symbol(position.symbol) for position in missing)
    if any(len(symbol) != 6 or not symbol.isdigit() for symbol in symbols):
        return {}
    deadline = monotonic() + TOTAL_FETCH_BUDGET_SECONDS
    rows: list[_SanitisedEvidence] = []
    for index, position in enumerate(missing):
        # Re-read both local authorities before disclosing each individual
        # symbol.  A change stops the batch and discards every fetched result.
        if not _guard_unchanged(repository, snapshot, authorization, review_time):
            return {}
        remaining_seconds = int(deadline - monotonic())
        if remaining_seconds <= 0:
            rows.extend(
                _unknown_evidence(
                    remainder,
                    portfolio_id=snapshot.portfolio_id,
                    holding_version=snapshot.holding_version,
                    as_of=as_of,
                    phase=normalized_phase,
                    attempt=_next_attempt(cached.get(remainder.position_key)),
                    knowledge_time=datetime.now(UTC),
                    reason_code="total_fetch_budget_exhausted",
                )
                for remainder in missing[index:]
            )
            break
        symbol = normalize_symbol(position.symbol)
        try:
            fetched = tuple(
                fetcher(
                    (symbol,),
                    as_of,
                    timeout_seconds=min(FETCH_TIMEOUT_SECONDS, remaining_seconds),
                )
            )
            if (
                len(fetched) != 1
                or not isinstance(fetched[0], CompanyActionEvidence)
                or normalize_symbol(fetched[0].symbol) != symbol
            ):
                raise ValueError("provider result does not match requested symbol")
            completed_at = datetime.now(UTC)
            if _aware_datetime(fetched[0].knowledge_time) > completed_at:
                raise ValueError("provider knowledge time is in the future")
            rows.append(
                _localise_evidence(
                    position,
                    fetched[0],
                    portfolio_id=snapshot.portfolio_id,
                    holding_version=snapshot.holding_version,
                    as_of=as_of,
                    phase=normalized_phase,
                    attempt=_next_attempt(cached.get(position.position_key)),
                )
            )
        except Exception:  # noqa: BLE001 - one symbol cannot erase other evidence
            rows.append(
                _unknown_evidence(
                    position,
                    portfolio_id=snapshot.portfolio_id,
                    holding_version=snapshot.holding_version,
                    as_of=as_of,
                    phase=normalized_phase,
                    attempt=_next_attempt(cached.get(position.position_key)),
                    knowledge_time=datetime.now(UTC),
                    reason_code="provider_exception_or_contract_failure",
                )
            )

    # Detect a last-call race too.  No provider result is archived when either
    # the holding revision or the exact grant changed during this batch.
    if not _guard_unchanged(repository, snapshot, authorization, review_time):
        return {}
    if not _archive_evidence(repository, rows, snapshot=snapshot):
        return {}
    if not _guard_unchanged(repository, snapshot, authorization, review_time):
        return {}

    combined = _cached_rows(
        repository,
        snapshot.positions,
        portfolio_id=snapshot.portfolio_id,
        holding_version=snapshot.holding_version,
        as_of=as_of,
        phase=normalized_phase,
    )
    completed_at = datetime.now(UTC)
    return _clearances_from_rows(combined, snapshot.positions, completed_at)


def _default_fetcher(
    symbols: Sequence[str],
    as_of: date,
    *,
    timeout_seconds: int = FETCH_TIMEOUT_SECONDS,
) -> Sequence[CompanyActionEvidence]:
    from ashare_lab.adapters.cninfo_company_actions import read_cninfo_company_actions

    return read_cninfo_company_actions(symbols, as_of, timeout_seconds=timeout_seconds)


def _holding_snapshot(
    portfolio: ActiveHoldingPortfolio | None,
    *,
    reviewed_at: datetime,
) -> _HoldingSnapshot | None:
    if portfolio is None or portfolio.status != "active" or portfolio.effective_at > reviewed_at:
        return None
    positions = tuple(position for position in portfolio.positions if position.status == "active")
    if not positions:
        return None
    keys = tuple(position.position_key for position in positions)
    if len(keys) != len(set(keys)):
        raise ValueError("active holding position keys must be unique")
    return _HoldingSnapshot(
        portfolio_id=portfolio.id,
        holding_version=portfolio.version,
        position_keys=keys,
        positions=positions,
    )


def _guard_unchanged(
    repository: SQLiteRepository,
    expected: _HoldingSnapshot,
    authorization: _AuthorizationSnapshot,
    reviewed_at: datetime,
) -> bool:
    current_authorization = _read_authorization(authorization.path)
    if current_authorization != authorization:
        return False
    current = _holding_snapshot(
        get_active_holding_portfolio(repository),
        reviewed_at=reviewed_at,
    )
    return current is not None and (
        current.portfolio_id,
        current.holding_version,
        current.position_keys,
    ) == (
        expected.portfolio_id,
        expected.holding_version,
        expected.position_keys,
    )


def _database_snapshot_matches(connection, expected: _HoldingSnapshot) -> bool:
    """Linearize the final archive against a concurrent holding-ledger change."""

    revision = connection.execute(
        """
        SELECT id, version, status
        FROM holding_portfolio_revisions
        ORDER BY version DESC LIMIT 1
        """
    ).fetchone()
    if revision is None or (
        str(revision["id"]),
        int(revision["version"]),
        str(revision["status"]),
    ) != (expected.portfolio_id, expected.holding_version, "active"):
        return False
    keys = tuple(
        str(row["position_key"])
        for row in connection.execute(
            """
            SELECT position_key
            FROM holding_positions
            WHERE revision_id=? AND status='active'
            ORDER BY symbol
            """,
            (expected.portfolio_id,),
        ).fetchall()
    )
    return keys == expected.position_keys


def _localise_evidence(
    position: ActiveHolding,
    evidence: CompanyActionEvidence,
    *,
    portfolio_id: str,
    holding_version: int,
    as_of: date,
    phase: str,
    attempt: int,
) -> _SanitisedEvidence:
    symbol = normalize_symbol(position.symbol)
    knowledge_time = _aware_datetime(evidence.knowledge_time)
    receipts = tuple(
        {
            "stream": receipt.stream,
            "complete": bool(receipt.complete),
            "record_count": receipt.record_count,
            "response_hash": receipt.response_hash,
            "reason_code": receipt.reason_code,
        }
        for receipt in sorted(evidence.stream_receipts, key=lambda item: item.stream)
    )
    receipt_streams = frozenset(str(receipt["stream"]) for receipt in receipts)
    complete = receipt_streams == _REQUIRED_STREAMS and all(
        bool(receipt["complete"]) for receipt in receipts
    )
    coverage_valid = (
        evidence.symbol == symbol
        and evidence.coverage_from <= position.entry_date
        and evidence.coverage_through == as_of
    )
    local_events = tuple(
        sorted(
            (
                {"date": event_date.isoformat(), "kind": event_kind}
                for event_date, event_kind in zip(
                    evidence.event_dates,
                    evidence.event_kinds,
                    strict=True,
                )
                if position.entry_date <= event_date <= as_of
            ),
            key=lambda item: (item["date"], item["kind"]),
        )
    )

    if coverage_valid and evidence.status is CompanyActionEvidenceStatus.DETECTED and local_events:
        status = CompanyActionEvidenceStatus.DETECTED
        reason_code = evidence.reason_code
    elif (
        coverage_valid
        and complete
        and (
            evidence.status is CompanyActionEvidenceStatus.CLEAR
            or (evidence.status is CompanyActionEvidenceStatus.DETECTED and not local_events)
        )
    ):
        # Provider events before this holding's entry are excluded locally.
        status = CompanyActionEvidenceStatus.CLEAR
        reason_code = "clear_after_local_entry_filter"
    else:
        status = CompanyActionEvidenceStatus.UNKNOWN
        reason_code = (
            evidence.reason_code
            if evidence.status is CompanyActionEvidenceStatus.UNKNOWN
            else "local_coverage_or_completeness_unknown"
        )
        local_events = ()

    return _SanitisedEvidence(
        portfolio_id=portfolio_id,
        position_key=position.position_key,
        holding_version=holding_version,
        symbol=symbol,
        as_of=as_of,
        phase=phase,
        attempt=attempt,
        status=status,
        coverage_from=(
            position.entry_date if status is not CompanyActionEvidenceStatus.UNKNOWN else None
        ),
        coverage_through=(as_of if status is not CompanyActionEvidenceStatus.UNKNOWN else None),
        knowledge_time=knowledge_time,
        events=local_events,
        provider_response_hash=evidence.response_hash,
        stream_receipts=receipts,
        reason_code=reason_code,
        provider_method_version=evidence.method_version,
    )


def _unknown_evidence(
    position: ActiveHolding,
    *,
    portfolio_id: str,
    holding_version: int,
    as_of: date,
    phase: str,
    attempt: int,
    knowledge_time: datetime,
    reason_code: str,
) -> _SanitisedEvidence:
    return _SanitisedEvidence(
        portfolio_id=portfolio_id,
        position_key=position.position_key,
        holding_version=holding_version,
        symbol=normalize_symbol(position.symbol),
        as_of=as_of,
        phase=phase,
        attempt=attempt,
        status=CompanyActionEvidenceStatus.UNKNOWN,
        coverage_from=None,
        coverage_through=None,
        knowledge_time=_aware_datetime(knowledge_time),
        events=(),
        provider_response_hash=None,
        stream_receipts=(),
        reason_code=reason_code,
        provider_method_version=CNINFO_COMPANY_ACTION_METHOD_VERSION,
    )


def _archive_evidence(
    repository: SQLiteRepository,
    rows: Sequence[_SanitisedEvidence],
    *,
    snapshot: _HoldingSnapshot,
) -> bool:
    if not rows:
        return True
    created_at = datetime.now(UTC).isoformat()
    with repository.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            if not _database_snapshot_matches(connection, snapshot):
                connection.rollback()
                return False
            for row in rows:
                canonical = _canonical_evidence(row)
                evidence_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                connection.execute(
                    """
                    INSERT INTO company_action_evidence_attempts
                    (evidence_id, portfolio_id, position_key, holding_version, symbol, as_of, phase,
                     attempt, status, coverage_from, coverage_through, knowledge_time, events_json,
                     provider_response_hash, stream_receipts_json, evidence_hash,
                     reason_code, provider_method_version, local_method_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"company-action:{evidence_hash}",
                        row.portfolio_id,
                        row.position_key,
                        row.holding_version,
                        row.symbol,
                        row.as_of.isoformat(),
                        row.phase,
                        row.attempt,
                        row.status.value,
                        None if row.coverage_from is None else row.coverage_from.isoformat(),
                        None if row.coverage_through is None else row.coverage_through.isoformat(),
                        row.knowledge_time.astimezone(UTC).isoformat(),
                        json.dumps(row.events, ensure_ascii=True, separators=(",", ":")),
                        row.provider_response_hash,
                        json.dumps(
                            row.stream_receipts,
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        evidence_hash,
                        row.reason_code,
                        row.provider_method_version,
                        row.local_method_version,
                        created_at,
                    ),
                )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise


def _canonical_evidence(row: _SanitisedEvidence) -> str:
    return json.dumps(
        {
            "portfolio_id": row.portfolio_id,
            "position_key": row.position_key,
            "holding_version": row.holding_version,
            "symbol": row.symbol,
            "as_of": row.as_of.isoformat(),
            "phase": row.phase,
            "attempt": row.attempt,
            "status": row.status.value,
            "coverage_from": (None if row.coverage_from is None else row.coverage_from.isoformat()),
            "coverage_through": (
                None if row.coverage_through is None else row.coverage_through.isoformat()
            ),
            "knowledge_time": row.knowledge_time.astimezone(UTC).isoformat(),
            "events": row.events,
            "provider_response_hash": row.provider_response_hash,
            "stream_receipts": row.stream_receipts,
            "reason_code": row.reason_code,
            "provider_method_version": row.provider_method_version,
            "local_method_version": row.local_method_version,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _cached_rows(
    repository: SQLiteRepository,
    positions: Sequence[ActiveHolding],
    *,
    portfolio_id: str,
    holding_version: int,
    as_of: date,
    phase: str,
) -> dict[str, dict[str, Any]]:
    if not positions:
        return {}
    keys = tuple(position.position_key for position in positions)
    placeholders = ",".join("?" for _ in keys)
    with repository.connection() as connection:
        records = connection.execute(
            f"""
            SELECT * FROM company_action_evidence_attempts
            WHERE portfolio_id=? AND holding_version=? AND as_of=? AND phase=?
              AND provider_method_version=? AND local_method_version=?
              AND position_key IN ({placeholders})
            ORDER BY attempt DESC
            """,  # noqa: S608 - placeholders are generated, values stay parameterised
            (
                portfolio_id,
                holding_version,
                as_of.isoformat(),
                phase,
                CNINFO_COMPANY_ACTION_METHOD_VERSION,
                COMPANY_ACTION_EVIDENCE_METHOD_VERSION,
                *keys,
            ),
        ).fetchall()
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest.setdefault(str(record["position_key"]), dict(record))
    return latest


def _cached_row_needs_refresh(record: dict[str, Any] | None, *, now: datetime) -> bool:
    """Retry UNKNOWN/corrupt evidence after a bounded delay, never more than three times."""

    if record is None:
        return True
    try:
        attempt = int(record["attempt"])
    except (KeyError, TypeError, ValueError):
        return False
    if attempt >= MAX_PHASE_ATTEMPTS:
        return False
    try:
        status = CompanyActionEvidenceStatus(str(record["status"]))
        knowledge_time = _aware_datetime(datetime.fromisoformat(str(record["knowledge_time"])))
    except (KeyError, TypeError, ValueError):
        return True
    if status is not CompanyActionEvidenceStatus.UNKNOWN or knowledge_time > now:
        return True
    return now - knowledge_time >= UNKNOWN_RETRY_DELAY


def _next_attempt(record: dict[str, Any] | None) -> int:
    if record is None:
        return 1
    attempt = int(record["attempt"]) + 1
    if attempt > MAX_PHASE_ATTEMPTS:
        raise ValueError("company-action retry budget exhausted")
    return attempt


def _clearances_from_rows(
    rows: dict[str, dict[str, Any]],
    positions: Sequence[ActiveHolding],
    reviewed_at: datetime,
) -> dict[str, CompanyActionClearance]:
    results: dict[str, CompanyActionClearance] = {}
    for position in positions:
        record = rows.get(position.position_key)
        if record is None:
            continue
        try:
            status = CompanyActionEvidenceStatus(str(record["status"]))
            if status is CompanyActionEvidenceStatus.UNKNOWN:
                continue
            if (
                str(record["provider_method_version"]) != CNINFO_COMPANY_ACTION_METHOD_VERSION
                or str(record["local_method_version"]) != COMPANY_ACTION_EVIDENCE_METHOD_VERSION
                or not 1 <= int(record["attempt"]) <= MAX_PHASE_ATTEMPTS
            ):
                continue
            symbol = normalize_symbol(str(record["symbol"]))
            if symbol != normalize_symbol(position.symbol):
                continue
            coverage_from = date.fromisoformat(str(record["coverage_from"]))
            coverage_through = date.fromisoformat(str(record["coverage_through"]))
            knowledge_time = _aware_datetime(datetime.fromisoformat(str(record["knowledge_time"])))
            evidence_as_of = date.fromisoformat(str(record["as_of"]))
            if (
                coverage_from != position.entry_date
                or coverage_through != evidence_as_of
                or knowledge_time > reviewed_at
            ):
                continue
            events = json.loads(str(record["events_json"]))
            receipts = json.loads(str(record["stream_receipts_json"]))
            if not isinstance(events, list) or not isinstance(receipts, list):
                continue
            if not _cached_hash_is_valid(record, events=events, receipts=receipts):
                continue
            local_event_dates = tuple(
                date.fromisoformat(str(event["date"]))
                for event in events
                if isinstance(event, dict) and "date" in event
            )
            if status is CompanyActionEvidenceStatus.DETECTED and not any(
                position.entry_date <= event_date <= coverage_through
                for event_date in local_event_dates
            ):
                continue
            if status is CompanyActionEvidenceStatus.CLEAR:
                receipt_streams = {
                    str(receipt.get("stream")) for receipt in receipts if isinstance(receipt, dict)
                }
                if (
                    local_event_dates
                    or receipt_streams != _REQUIRED_STREAMS
                    or not all(
                        isinstance(receipt, dict) and receipt.get("complete") is True
                        for receipt in receipts
                    )
                ):
                    continue
            results[symbol] = _build_clearance(
                symbol=symbol,
                through_date=coverage_through,
                clear=status is CompanyActionEvidenceStatus.CLEAR,
                evidence_id=str(record["evidence_id"]),
                from_date=coverage_from,
                knowledge_time=knowledge_time,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            # Corrupt or incomplete cached evidence is not decision-grade.
            continue
    return results


def _cached_hash_is_valid(
    record: dict[str, Any],
    *,
    events: list[object],
    receipts: list[object],
) -> bool:
    canonical = json.dumps(
        {
            "portfolio_id": str(record["portfolio_id"]),
            "position_key": str(record["position_key"]),
            "holding_version": int(record["holding_version"]),
            "symbol": str(record["symbol"]),
            "as_of": str(record["as_of"]),
            "phase": str(record["phase"]),
            "attempt": int(record["attempt"]),
            "status": str(record["status"]),
            "coverage_from": record["coverage_from"],
            "coverage_through": record["coverage_through"],
            "knowledge_time": str(record["knowledge_time"]),
            "events": events,
            "provider_response_hash": record["provider_response_hash"],
            "stream_receipts": receipts,
            "reason_code": str(record["reason_code"]),
            "provider_method_version": str(record["provider_method_version"]),
            "local_method_version": str(record["local_method_version"]),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    evidence_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return (
        evidence_hash == str(record["evidence_hash"])
        and str(record["evidence_id"]) == f"company-action:{evidence_hash}"
    )


def _build_clearance(
    *,
    symbol: str,
    through_date: date,
    clear: bool,
    evidence_id: str,
    from_date: date,
    knowledge_time: datetime,
) -> CompanyActionClearance:
    values: dict[str, object] = {
        "symbol": symbol,
        "through_date": through_date,
        "clear": clear,
        "source": "cninfo_official_read",
        "evidence_id": evidence_id,
        "from_date": from_date,
    }
    if "knowledge_time" in _CLEARANCE_FIELD_NAMES:
        values["knowledge_time"] = knowledge_time
        return CompanyActionClearance(**values)
    return _EvidenceBackedClearance(**values, knowledge_time=knowledge_time)


def _read_authorization(path: Path) -> _AuthorizationSnapshot | None:
    snapshot = _read_config(path)
    return snapshot if snapshot is not None and snapshot.enabled else None


def _read_config(path: Path) -> _AuthorizationSnapshot | None:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            return None
        if stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_size > 4096:
            return None
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        if not isinstance(document, dict):
            return None
        if set(document) != {
            "enabled",
            "scope",
            "authorized_fields",
            "orders_enabled",
            "authorized_at",
        }:
            return None
        authorized_at = _aware_datetime(datetime.fromisoformat(str(document["authorized_at"])))
        if (
            not isinstance(document["enabled"], bool)
            or document["scope"] != AUTHORIZATION_SCOPE
            or document["authorized_fields"] != list(AUTHORIZED_FIELDS)
            or document["orders_enabled"] is not False
        ):
            return None
        return _AuthorizationSnapshot(
            path=path,
            fingerprint=hashlib.sha256(raw).hexdigest(),
            enabled=document["enabled"],
            authorized_at=authorized_at,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return None


def _resolved_config_path(config_path: Path | None) -> Path:
    path = company_action_config_path() if config_path is None else Path(config_path).expanduser()
    return path.resolve()


def _write_private_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def _aware_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("company-action knowledge time must be timezone-aware")
    return value


__all__ = [
    "AUTHORIZED_FIELDS",
    "AUTHORIZATION_SCOPE",
    "authorize_company_actions",
    "company_action_config_path",
    "is_company_action_authorized",
    "refresh_and_load_company_action_clearances",
    "revoke_company_actions",
]
