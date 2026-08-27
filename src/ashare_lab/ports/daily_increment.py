"""Provider-neutral contract for one completed A-share daily increment.

The long-history research database remains immutable.  Implementations of this
port fetch a small, provider-isolated end-of-day overlay that may be appended
only after the caller's data-quality and licensing gates have passed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

AssetKind = Literal["stocks", "indices"]


@dataclass(frozen=True, slots=True)
class DailyIncrementBatch:
    """One auditable, normalized response for a completed trading session.

    ``frame`` contains ``symbol`` followed by the canonical columns from
    :mod:`ashare_lab.ports.market_data`.  Symbols in the frame are normalized
    six-digit A-share codes.  ``requested_symbols`` and ``received_symbols``
    intentionally preserve the provider's market suffix so coverage is checked
    before any identity information is discarded.
    """

    frame: pd.DataFrame
    target_date: date
    requested_symbols: tuple[str, ...]
    received_symbols: tuple[str, ...]
    fetched_at: datetime
    trace_ids: tuple[str, ...]
    provider: str
    cutoff_timestamp: int

    @property
    def coverage_ratio(self) -> float:
        requested = set(self.requested_symbols)
        if not requested:
            return 1.0
        return len(requested.intersection(self.received_symbols)) / len(requested)


@runtime_checkable
class DailyIncrementPort(Protocol):
    """Minimal surface for a full-market, unadjusted daily overlay."""

    provider: str

    def fetch_cn_trading_days(self, start: date, end: date) -> tuple[date, ...]:
        """Return verified CN full and half trading sessions, inclusive."""

    def fetch_cn_stock_symbols(self) -> tuple[str, ...]:
        """Return current SSE/SZSE stock identifiers in provider format."""

    def fetch_daily_increment(
        self,
        symbols: Sequence[str],
        target_date: date,
        *,
        cutoff_timestamp: int | None = None,
        asset_kind: AssetKind = "stocks",
    ) -> DailyIncrementBatch:
        """Return completed stock or index bars without inferring their kind."""
