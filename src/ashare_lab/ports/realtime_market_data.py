"""Provider-neutral contracts for research-grade real-time A-share snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RealtimeBatch:
    """One auditable provider response.

    ``records`` deliberately keeps the provider payload intact.  Feature code
    must map explicitly named fields and must never guess an unknown unit.
    """

    dataset: str
    records: tuple[dict[str, Any], ...]
    requested_symbols: tuple[str, ...]
    received_symbols: tuple[str, ...]
    fetched_at: datetime
    trace_ids: tuple[str, ...]
    provider: str

    @property
    def coverage_ratio(self) -> float:
        requested = set(self.requested_symbols)
        if not requested:
            return 1.0
        return len(requested.intersection(self.received_symbols)) / len(requested)


@dataclass(frozen=True, slots=True)
class CurrentFundamentalsSnapshot:
    """A provider's *current* symbol-fundamentals snapshot.

    ``fetched_at`` is the local observation time, not a financial disclosure
    date.  The Infoway symbol-info endpoint does not provide the publication
    timeline needed to reconstruct what was known on a historical date, so the
    payload must not be joined into a point-in-time backtest.
    """

    records: tuple[dict[str, Any], ...]
    requested_symbols: tuple[str, ...]
    received_symbols: tuple[str, ...]
    fetched_at: datetime
    trace_ids: tuple[str, ...]
    provider: str
    dataset: str = "symbol_fundamentals"
    snapshot_scope: str = "current"
    is_point_in_time_history: bool = False
    historical_backtest_eligible: bool = False

    @property
    def coverage_ratio(self) -> float:
        requested = set(self.requested_symbols)
        if not requested:
            return 1.0
        return len(requested.intersection(self.received_symbols)) / len(requested)


@runtime_checkable
class RealtimeMarketDataPort(Protocol):
    """Minimal real-time surface used by the tail-session research scanner."""

    def fetch_symbol_list(self) -> tuple[dict[str, Any], ...]:
        """Return the provider's current ``STOCK_CN`` security list."""

    def fetch_symbol_fundamentals(
        self,
        symbols: Sequence[str],
    ) -> CurrentFundamentalsSnapshot:
        """Return current metadata/EPS/BPS snapshots, never PIT history."""

    def fetch_recent_klines(
        self,
        symbols: Sequence[str],
        *,
        interval: int = 1,
        count: int = 2,
    ) -> RealtimeBatch:
        """Return recent bars for up to the provider-approved batch size."""

    def fetch_latest_trades(self, symbols: Sequence[str]) -> RealtimeBatch:
        """Return the latest transaction snapshot for each requested symbol."""

    def fetch_depth(self, symbols: Sequence[str]) -> RealtimeBatch:
        """Return the latest order-book snapshot for each requested symbol."""
