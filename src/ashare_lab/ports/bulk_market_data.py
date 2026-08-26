"""Contracts for licensed, bulk A-share reference and daily-market data.

The interactive UI must never fan out thousands of network calls.  A provider
adapter implements this port, a background ingestion service writes a local
cache, and research screens read only that cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

from ashare_lab.domain.data_sources import SourceId

CANONICAL_SECURITY_MASTER_COLUMNS = (
    "symbol",
    "name",
    "exchange",
    "board",
    "list_date",
    "delist_date",
    "industry",
    "is_st",
    "is_delisting",
    "is_suspended",
    "source",
    "retrieved_at",
)


@runtime_checkable
class BulkMarketDataPort(Protocol):
    """A licensed adapter capable of date-major full-market retrieval.

    ``fetch_security_master`` returns one point-in-time universe snapshot using
    :data:`CANONICAL_SECURITY_MASTER_COLUMNS`.

    ``fetch_trade_calendar`` returns the provider's open A-share sessions.
    ``fetch_daily_for_trade_date`` then returns one complete market cross
    section containing ``symbol`` followed by the canonical daily columns from
    :mod:`ashare_lab.ports.market_data`.

    Date-major retrieval is the safe default for providers such as Tushare:
    five sessions require five market calls, not ``number_of_stocks * 5``.
    The adapter, not the UI, remains responsible for request-size, rate-limit
    and entitlement rules.  iFinD/Choice adapters may also implement this
    contract even when their SDK has additional multi-code functions.
    """

    source_id: SourceId

    def fetch_security_master(self, as_of: date) -> pd.DataFrame:
        """Return the A-share security master known at ``as_of``."""

    def fetch_trade_calendar(self, start: date, end: date) -> Sequence[date]:
        """Return unique open sessions in ascending order, inclusive."""

    def fetch_daily_for_trade_date(
        self,
        trade_date: date,
        *,
        adjust: str = "none",
    ) -> pd.DataFrame:
        """Return one complete canonical A-share market cross section."""


@runtime_checkable
class SymbolBatchMarketDataPort(Protocol):
    """Optional capability for vendors that efficiently accept many codes."""

    source_id: SourceId

    def fetch_security_master(self, as_of: date) -> pd.DataFrame:
        """Return the A-share security master known at ``as_of``."""

    def fetch_daily_batch(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        *,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """Return canonical daily bars for a provider-approved code batch."""
