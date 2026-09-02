"""Explicit, persistent current-holding ledger for daily research reviews.

Only a user-confirmed call to :func:`replace_active_holdings` or
:func:`clear_active_holdings` changes membership or the planned holding
horizon.  Model runs and daily reviews are deliberately read-only consumers of
this ledger.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from math import isclose, isfinite
from typing import Any, Final
from uuid import UUID, uuid4, uuid5

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.analytics.multi_timeframe import SUPPORTED_HOLDING_WEEKS
from ashare_lab.ports.market_data import normalize_symbol

HOLDING_LEDGER_METHOD_VERSION: Final = "holding-ledger-v0.1.0"
HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY: Final = "holding_summary_delivery_channels"
HOLDING_SUMMARY_DELIVERY_CHANNELS: Final = ("serverchan", "bark")
HOLDING_CHART_DELIVERY_CHANNELS_KEY: Final = "holding_chart_delivery_channels"
HOLDING_CHART_PUBLISHER_ID_KEY: Final = "holding_chart_publisher_id"
HOLDING_CHART_PUBLISHER_IDS: Final = ("cloudflare_r2",)
_HOLDING_REVISION_NAMESPACE = UUID("2f8f1a72-6e63-44e1-a87f-7be76e6575f5")


@dataclass(frozen=True, slots=True, kw_only=True)
class HoldingPositionInput:
    symbol: str
    name: str
    entry_date: date
    stock_sleeve_weight: float
    cost_price: float | None = None
    account_weight: float | None = None
    source: str = "user_confirmed"
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ActiveHolding:
    id: str
    revision_id: str
    position_key: str
    symbol: str
    name: str
    entry_date: date
    cost_price: float | None
    stock_sleeve_weight: float
    account_weight: float | None
    status: str
    source: str
    version: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ActiveHoldingPortfolio:
    id: str
    version: int
    holding_weeks: int
    effective_at: datetime
    source: str
    status: str
    method_version: str
    positions: tuple[ActiveHolding, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class HoldingKnowledgeContext:
    """Exact current holding revision frozen at one live knowledge time.

    Market-data cutoffs are deliberately absent.  A report may therefore use
    the latest user-confirmed holding/authorization revision while keeping its
    price evidence on an earlier verified completed close.
    """

    portfolio_id: str
    version: int
    known_at: datetime

    def __post_init__(self) -> None:
        if not str(self.portfolio_id).strip():
            raise ValueError("portfolio_id cannot be blank")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version <= 0:
            raise ValueError("holding version must be positive")
        if not isinstance(self.known_at, datetime) or self.known_at.tzinfo is None:
            raise ValueError("holding known_at must be timezone-aware")


def holding_knowledge_context(
    portfolio: ActiveHoldingPortfolio,
    *,
    known_at: datetime,
) -> HoldingKnowledgeContext:
    """Freeze one exact, already-read holding revision for a live run."""

    if not isinstance(portfolio, ActiveHoldingPortfolio):
        raise TypeError("portfolio must be an ActiveHoldingPortfolio")
    context = HoldingKnowledgeContext(
        portfolio_id=portfolio.id,
        version=portfolio.version,
        known_at=known_at,
    )
    if portfolio.effective_at > context.known_at:
        raise ValueError("holding revision is newer than the live knowledge time")
    return context


def resolve_current_holding_context(
    repository: SQLiteRepository,
    context: HoldingKnowledgeContext,
) -> ActiveHoldingPortfolio:
    """Re-read and require that a frozen revision is still exactly current."""

    if not isinstance(context, HoldingKnowledgeContext):
        raise TypeError("context must be a HoldingKnowledgeContext")
    current = get_active_holding_portfolio(repository)
    if current is None or (current.id, current.version) != (
        context.portfolio_id,
        context.version,
    ):
        raise ValueError("frozen holding revision is no longer current")
    if current.effective_at > context.known_at:
        raise ValueError("holding revision is newer than the live knowledge time")
    return current


def replace_active_holdings(
    repository: SQLiteRepository,
    positions: Iterable[HoldingPositionInput],
    *,
    holding_weeks: int,
    effective_at: datetime,
    source: str = "user_confirmed",
    change_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    expected_current_revision_id: str | None = None,
    expected_current_version: int | None = None,
) -> ActiveHoldingPortfolio:
    """Atomically replace the complete current holding set.

    Unknown cost prices and account weights remain ``None``.  They are never
    inferred from a recommendation, a current quote, or other positions.
    ``change_id`` can be reused by a caller to make an explicit submission
    idempotent.
    """

    if (expected_current_revision_id is None) != (expected_current_version is None):
        raise ValueError("Holding update CAS requires both current id and version")
    if expected_current_version is not None and expected_current_version <= 0:
        raise ValueError("Expected holding version must be positive")
    _validate_holding_weeks(holding_weeks)
    effective_at = _aware_datetime(effective_at)
    normalized_source = _nonblank(source, "source")
    normalized_metadata = _normalize_portfolio_metadata(metadata)
    supplied = tuple(positions)
    if not supplied:
        raise ValueError("Use clear_active_holdings for an empty holding set")
    normalized = tuple(_normalize_input(item, effective_at=effective_at) for item in supplied)
    symbols = [item.symbol for item in normalized]
    if len(symbols) != len(set(symbols)):
        raise ValueError("Current holdings cannot contain duplicate symbols")
    if not isclose(
        sum(item.stock_sleeve_weight for item in normalized),
        1.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Stock-sleeve weights must sum to 1.0")
    known_account_total = sum(
        item.account_weight for item in normalized if item.account_weight is not None
    )
    if known_account_total > 1.0 + 1e-9:
        raise ValueError("Known account weights cannot exceed 1.0")

    revision_id = _revision_id(change_id)
    existing = repository.get_holding_snapshot(revision_id)
    if existing is not None:
        result = _portfolio_from_snapshot(existing)
        _validate_idempotent_retry(
            result,
            normalized,
            holding_weeks=holding_weeks,
            effective_at=effective_at,
            source=normalized_source,
            metadata=normalized_metadata,
        )
        return result
    version = repository.next_holding_snapshot_version()
    revision = {
        "id": revision_id,
        "version": version,
        "holding_weeks": holding_weeks,
        "effective_at": effective_at,
        "source": normalized_source,
        "status": "active",
        "method_version": HOLDING_LEDGER_METHOD_VERSION,
        "metadata_json": normalized_metadata,
    }
    rows = []
    for item in normalized:
        position_key = _position_key(item.symbol, item.entry_date)
        rows.append(
            {
                "id": f"holding-position:{revision_id}:{item.symbol}",
                "revision_id": revision_id,
                "position_key": position_key,
                "symbol": item.symbol,
                "name": item.name,
                "entry_date": item.entry_date,
                "cost_price": item.cost_price,
                "stock_sleeve_weight": item.stock_sleeve_weight,
                "account_weight": item.account_weight,
                "status": "active",
                "source": item.source,
                "version": version,
                "metadata_json": dict(item.metadata or {}),
            }
        )
    repository.archive_holding_snapshot(
        revision,
        positions=rows,
        expected_current_revision_id=expected_current_revision_id,
        expected_current_version=expected_current_version,
    )
    result = get_active_holding_portfolio(repository)
    if result is None:
        raise RuntimeError("Holding snapshot was archived but could not be read back")
    return result


def clear_active_holdings(
    repository: SQLiteRepository,
    *,
    effective_at: datetime,
    source: str = "user_confirmed",
    holding_weeks: int | None = None,
    change_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ActiveHoldingPortfolio:
    """Record an explicit empty snapshot; daily EXIT research does not call it."""

    current = get_active_holding_portfolio(repository)
    resolved_weeks = holding_weeks or (current.holding_weeks if current is not None else None)
    if resolved_weeks is None:
        raise ValueError("holding_weeks is required when no earlier holding snapshot exists")
    _validate_holding_weeks(resolved_weeks)
    effective_at = _aware_datetime(effective_at)
    normalized_source = _nonblank(source, "source")
    normalized_metadata = _normalize_portfolio_metadata(metadata)
    revision_id = _revision_id(change_id)
    existing = repository.get_holding_snapshot(revision_id)
    if existing is not None:
        result = _portfolio_from_snapshot(existing)
        if (
            result.status != "cleared"
            or result.positions
            or result.holding_weeks != resolved_weeks
            or result.effective_at != effective_at.astimezone(result.effective_at.tzinfo)
            or result.source != normalized_source
            or dict(result.metadata) != normalized_metadata
        ):
            raise ValueError("change_id was already used for a different holding snapshot")
        return result
    revision = {
        "id": revision_id,
        "version": repository.next_holding_snapshot_version(),
        "holding_weeks": resolved_weeks,
        "effective_at": effective_at,
        "source": normalized_source,
        "status": "cleared",
        "method_version": HOLDING_LEDGER_METHOD_VERSION,
        "metadata_json": normalized_metadata,
    }
    repository.archive_holding_snapshot(revision, positions=())
    result = get_active_holding_portfolio(repository)
    if result is None:
        raise RuntimeError("Cleared holding snapshot could not be read back")
    return result


def get_active_holding_portfolio(
    repository: SQLiteRepository,
    *,
    as_of: date | None = None,
) -> ActiveHoldingPortfolio | None:
    snapshot = repository.get_current_holding_snapshot(as_of=as_of)
    if snapshot is None:
        return None
    return _portfolio_from_snapshot(snapshot)


def _portfolio_from_snapshot(snapshot: Mapping[str, Any]) -> ActiveHoldingPortfolio:
    revision = snapshot["revision"]
    positions = tuple(_holding_from_row(row) for row in snapshot["positions"])
    return ActiveHoldingPortfolio(
        id=str(revision["id"]),
        version=int(revision["version"]),
        holding_weeks=int(revision["holding_weeks"]),
        effective_at=_parse_datetime(revision["effective_at"]),
        source=str(revision["source"]),
        status=str(revision["status"]),
        method_version=str(revision["method_version"]),
        positions=positions,
        metadata=dict(revision["metadata_json"]),
    )


def _validate_idempotent_retry(
    existing: ActiveHoldingPortfolio,
    positions: tuple[HoldingPositionInput, ...],
    *,
    holding_weeks: int,
    effective_at: datetime,
    source: str,
    metadata: Mapping[str, Any],
) -> None:
    if (
        existing.holding_weeks != holding_weeks
        or existing.effective_at != effective_at.astimezone(existing.effective_at.tzinfo)
        or existing.source != source
        or dict(existing.metadata) != dict(metadata)
    ):
        raise ValueError("change_id was already used for a different holding snapshot")
    expected = {
        item.symbol: (
            item.name,
            item.entry_date,
            item.cost_price,
            item.stock_sleeve_weight,
            item.account_weight,
            item.source,
            dict(item.metadata or {}),
        )
        for item in positions
    }
    actual = {
        item.symbol: (
            item.name,
            item.entry_date,
            item.cost_price,
            item.stock_sleeve_weight,
            item.account_weight,
            item.source,
            dict(item.metadata),
        )
        for item in existing.positions
    }
    if actual != expected:
        raise ValueError("change_id was already used for different holding positions")


def list_active_holdings(repository: SQLiteRepository) -> tuple[ActiveHolding, ...]:
    portfolio = get_active_holding_portfolio(repository)
    if portfolio is None or portfolio.status != "active":
        return ()
    return tuple(item for item in portfolio.positions if item.status == "active")


def holding_summary_delivery_channels(
    portfolio: ActiveHoldingPortfolio | None,
) -> frozenset[str]:
    """Return only explicit, per-provider disclosure consent.

    The retired ``external_delivery_consent`` boolean is deliberately ignored:
    it neither names a destination nor authorizes disclosure to a new channel.
    """

    if portfolio is None or portfolio.status != "active" or not portfolio.positions:
        return frozenset()
    raw = portfolio.metadata.get(HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY, ())
    return frozenset(
        _normalize_delivery_channels(
            raw,
            field_name=HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY,
        )
    )


def holding_chart_delivery_channels(
    portfolio: ActiveHoldingPortfolio | None,
) -> frozenset[str]:
    """Return only the independently granted per-provider chart allow-list.

    A text-summary grant is deliberately insufficient.  Existing snapshots
    without this separate field therefore authorize no chart generation or
    delivery.
    """

    if portfolio is None or portfolio.status != "active" or not portfolio.positions:
        return frozenset()
    raw = portfolio.metadata.get(HOLDING_CHART_DELIVERY_CHANNELS_KEY, ())
    return frozenset(
        _normalize_delivery_channels(
            raw,
            field_name=HOLDING_CHART_DELIVERY_CHANNELS_KEY,
        )
    )


def holding_chart_publisher_id(
    portfolio: ActiveHoldingPortfolio | None,
) -> str | None:
    """Return one explicitly authorized private chart publisher, if any."""

    if portfolio is None or portfolio.status != "active" or not portfolio.positions:
        return None
    return _normalize_chart_publisher_id(portfolio.metadata.get(HOLDING_CHART_PUBLISHER_ID_KEY))


def _normalize_portfolio_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(metadata or {})
    # Preserve no ambiguous legacy consent marker in a newly confirmed
    # snapshot.  Existing snapshots may still contain it, but the reader above
    # never treats it as authorization.
    normalized.pop("external_delivery_consent", None)
    normalized[HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY] = list(
        _normalize_delivery_channels(
            normalized.get(HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY, ()),
            field_name=HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY,
        )
    )
    normalized[HOLDING_CHART_DELIVERY_CHANNELS_KEY] = list(
        _normalize_delivery_channels(
            normalized.get(HOLDING_CHART_DELIVERY_CHANNELS_KEY, ()),
            field_name=HOLDING_CHART_DELIVERY_CHANNELS_KEY,
        )
    )
    normalized[HOLDING_CHART_PUBLISHER_ID_KEY] = _normalize_chart_publisher_id(
        normalized.get(HOLDING_CHART_PUBLISHER_ID_KEY)
    )
    return normalized


def _normalize_delivery_channels(
    value: object,
    *,
    field_name: str,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bool)) or not isinstance(value, (list, tuple, set)):
        raise ValueError(f"{field_name} must be a list of serverchan/bark")
    supplied: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must contain only provider names")
        channel = item.strip().lower()
        if channel not in HOLDING_SUMMARY_DELIVERY_CHANNELS:
            raise ValueError(f"Unsupported {field_name} channel")
        supplied.add(channel)
    return tuple(channel for channel in HOLDING_SUMMARY_DELIVERY_CHANNELS if channel in supplied)


def _normalize_chart_publisher_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{HOLDING_CHART_PUBLISHER_ID_KEY} must be a supported provider name")
    normalized = value.strip().lower()
    if normalized not in HOLDING_CHART_PUBLISHER_IDS:
        raise ValueError("Unsupported holding-chart publisher")
    return normalized


def _normalize_input(
    item: HoldingPositionInput,
    *,
    effective_at: datetime,
) -> HoldingPositionInput:
    if not isinstance(item, HoldingPositionInput):
        raise TypeError("positions must contain HoldingPositionInput values")
    symbol = normalize_symbol(item.symbol)
    name = _nonblank(item.name, "name")
    if not isinstance(item.entry_date, date):
        raise TypeError("entry_date must be a date")
    if item.entry_date > effective_at.date():
        raise ValueError("entry_date cannot be after the explicit holding update")
    stock_weight = _positive_ratio(item.stock_sleeve_weight, "stock_sleeve_weight")
    cost = _positive_optional(item.cost_price, "cost_price")
    account = (
        None
        if item.account_weight is None
        else _positive_ratio(item.account_weight, "account_weight")
    )
    return HoldingPositionInput(
        symbol=symbol,
        name=name,
        entry_date=item.entry_date,
        cost_price=cost,
        stock_sleeve_weight=stock_weight,
        account_weight=account,
        source=_nonblank(item.source, "position source"),
        metadata=dict(item.metadata or {}),
    )


def _holding_from_row(row: Mapping[str, Any]) -> ActiveHolding:
    return ActiveHolding(
        id=str(row["id"]),
        revision_id=str(row["revision_id"]),
        position_key=str(row["position_key"]),
        symbol=str(row["symbol"]),
        name=str(row["name"]),
        entry_date=date.fromisoformat(str(row["entry_date"])),
        cost_price=(None if row["cost_price"] is None else float(row["cost_price"])),
        stock_sleeve_weight=float(row["stock_sleeve_weight"]),
        account_weight=(None if row["account_weight"] is None else float(row["account_weight"])),
        status=str(row["status"]),
        source=str(row["source"]),
        version=int(row["version"]),
        metadata=dict(row["metadata_json"]),
    )


def _revision_id(change_id: str | None) -> str:
    if change_id is None:
        return f"holding-revision:{uuid4()}"
    normalized = _nonblank(change_id, "change_id")
    return f"holding-revision:{uuid5(_HOLDING_REVISION_NAMESPACE, normalized)}"


def _position_key(symbol: str, entry_date: date) -> str:
    digest = hashlib.sha256(f"{symbol}|{entry_date.isoformat()}".encode()).hexdigest()[:24]
    return f"holding:{symbol}:{digest}"


def _validate_holding_weeks(value: int) -> None:
    if isinstance(value, bool) or value not in SUPPORTED_HOLDING_WEEKS:
        raise ValueError("holding_weeks must be one of 1, 2, 4, 13, 26 or 52")


def _positive_optional(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    converted = float(value)
    if not isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{field_name} must be positive when provided")
    return converted


def _positive_ratio(value: float, field_name: str) -> float:
    converted = float(value)
    if not isfinite(converted) or not 0.0 < converted <= 1.0:
        raise ValueError(f"{field_name} must be in (0, 1]")
    return converted


def _nonblank(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be blank")
    return normalized


def _aware_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("effective_at must be a timezone-aware datetime")
    return value


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed
