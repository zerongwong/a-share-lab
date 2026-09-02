"""Production assembly for loading and reviewing only current holdings.

The loader joins the immutable, unadjusted CSMAR baseline to the continuous
verified unadjusted overlay after that baseline.  It does no networking and
never loads or screens the full stock universe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, Protocol

import pandas as pd

from ashare_lab.adapters.csmar_local import CSMARParquetMarketData
from ashare_lab.adapters.market_overlay_store import MarketOverlayStore
from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.analytics.multi_timeframe import horizon_contract
from ashare_lab.domain.data_sources import DEFAULT_MARKET_OVERLAY_SOURCE_ID
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.services.holding_ledger import (
    HoldingKnowledgeContext,
    get_active_holding_portfolio,
    holding_knowledge_context,
    resolve_current_holding_context,
)
from ashare_lab.services.review_active_holdings import (
    CompanyActionClearance,
    HoldingTreeReviewSummary,
    review_active_holdings,
)

ACTIVE_HOLDING_LOAD_METHOD_VERSION: Final = "active-holding-hybrid-loader-v0.1.0"


class _BaselineMarket(Protocol):
    def latest_trade_date(self, *, on_or_before: date | None = None) -> date: ...

    def fetch_daily(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        adjust: str = "none",
    ) -> pd.DataFrame: ...


class _OverlayMarket(Protocol):
    def read_verified_manifest(self, *, source_id: str | None = None) -> pd.DataFrame: ...

    def verified_dates_from(
        self,
        *,
        source_id: str,
        baseline_cutoff: date,
        through_date: date | None = None,
    ) -> tuple[date, ...]: ...

    def read_verified_daily(
        self,
        trade_date: date,
        *,
        source_id: str,
        asset_kind: str,
    ) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class ActiveHoldingHistoryLoad:
    portfolio_id: str
    holding_version: int
    holding_weeks: int
    histories: dict[str, pd.DataFrame]
    baseline_cutoff: date
    verified_overlay_dates: tuple[date, ...]
    data_cutoff: date
    unavailable_symbols: tuple[str, ...]
    sources: tuple[str, ...]
    adjustment: str = "none"
    verified_close: bool = True
    method_version: str = ACTIVE_HOLDING_LOAD_METHOD_VERSION


def load_active_holding_histories(
    repository: SQLiteRepository,
    *,
    dataset_root: str | Path,
    overlay_root: str | Path,
    as_of: date,
    overlay_source_id: str = DEFAULT_MARKET_OVERLAY_SOURCE_ID,
    _baseline_market: _BaselineMarket | None = None,
    _overlay_store: _OverlayMarket | None = None,
    _holding_context: HoldingKnowledgeContext | None = None,
) -> ActiveHoldingHistoryLoad | None:
    """Load the minimum verified history required by the current holding set."""

    if not isinstance(as_of, date):
        raise TypeError("as_of must be a date")
    portfolio = (
        get_active_holding_portfolio(repository, as_of=as_of)
        if _holding_context is None
        else resolve_current_holding_context(repository, _holding_context)
    )
    if portfolio is None or portfolio.status != "active" or not portfolio.positions:
        return None
    source_id = overlay_source_id.strip().lower()
    if not source_id:
        raise ValueError("overlay_source_id cannot be blank")
    baseline = _baseline_market or CSMARParquetMarketData(dataset_root)
    overlay = _overlay_store or MarketOverlayStore(overlay_root)
    baseline_cutoff = baseline.latest_trade_date(on_or_before=as_of)
    if baseline_cutoff > as_of:
        raise DataQualityError("CSMAR baseline cutoff is after the holding review date")

    manifest = overlay.read_verified_manifest(source_id=source_id)
    if not manifest.empty and "trade_date" not in manifest.columns:
        raise DataQualityError("Verified overlay manifest has no trade_date")
    eligible_manifest = (
        manifest.loc[
            pd.to_datetime(manifest["trade_date"], errors="coerce").dt.date <= as_of
        ].copy()
        if not manifest.empty
        else manifest
    )
    if (
        not eligible_manifest.empty
        and "adjustment" in eligible_manifest
        and not bool((eligible_manifest["adjustment"].astype(str) == "none").all())
    ):
        raise DataQualityError("Holding review overlay must use unadjusted prices")
    manifest_latest = (
        None
        if eligible_manifest.empty
        else pd.Timestamp(eligible_manifest["trade_date"].max()).date()
    )
    overlay_dates = overlay.verified_dates_from(
        source_id=source_id,
        baseline_cutoff=baseline_cutoff,
        through_date=as_of,
    )
    if (
        manifest_latest is not None
        and manifest_latest > baseline_cutoff
        and (not overlay_dates or overlay_dates[-1] != manifest_latest)
    ):
        raise DataQualityError(
            "Latest verified overlay is not on one continuous chain after the CSMAR baseline"
        )
    data_cutoff = overlay_dates[-1] if overlay_dates else baseline_cutoff
    contract = horizon_contract(portfolio.holding_weeks)
    requested_sessions = contract.minimum_daily_sessions + 80
    start = baseline_cutoff - timedelta(days=requested_sessions * 2 + 30)

    histories: dict[str, pd.DataFrame] = {}
    unavailable: list[str] = []
    for holding in portfolio.positions:
        try:
            history = baseline.fetch_daily(
                holding.symbol,
                start,
                baseline_cutoff,
                adjust="none",
            ).copy()
        except (DataUnavailableError, OSError, ValueError):
            unavailable.append(holding.symbol)
            continue
        pieces = [history]
        for overlay_date in overlay_dates:
            cross_section = overlay.read_verified_daily(
                overlay_date,
                source_id=source_id,
                asset_kind="stocks",
            )
            selected = cross_section.loc[
                cross_section["symbol"].astype(str) == holding.symbol
            ].copy()
            if not selected.empty:
                pieces.append(selected.drop(columns=["symbol"], errors="ignore"))
        merged = pd.concat(pieces, ignore_index=True)
        merged["trade_date"] = pd.to_datetime(merged["trade_date"]).dt.normalize()
        merged = merged.loc[merged["trade_date"].dt.date <= data_cutoff]
        if bool(merged["trade_date"].duplicated().any()):
            raise DataQualityError(
                f"{holding.symbol} has overlapping CSMAR and overlay observations"
            )
        merged = merged.sort_values("trade_date").reset_index(drop=True)
        if pd.Timestamp(merged.iloc[-1]["trade_date"]).date() != data_cutoff:
            unavailable.append(holding.symbol)
        history_sources = ("CSMAR", source_id) if overlay_dates else ("CSMAR",)
        merged.attrs.update(
            {
                "adjustment": "none",
                "sources": history_sources,
                "verified_overlay_dates": tuple(value.isoformat() for value in overlay_dates),
                "data_cutoff": data_cutoff.isoformat(),
                "method_version": ACTIVE_HOLDING_LOAD_METHOD_VERSION,
            }
        )
        histories[holding.symbol] = merged

    if not histories and portfolio.positions:
        raise DataUnavailableError("No current holding has a readable CSMAR history")
    sources = ("CSMAR", source_id) if overlay_dates else ("CSMAR",)
    return ActiveHoldingHistoryLoad(
        portfolio_id=portfolio.id,
        holding_version=portfolio.version,
        holding_weeks=portfolio.holding_weeks,
        histories=histories,
        baseline_cutoff=baseline_cutoff,
        verified_overlay_dates=overlay_dates,
        data_cutoff=data_cutoff,
        unavailable_symbols=tuple(sorted(unavailable)),
        sources=sources,
    )


def run_active_holding_review(
    repository: SQLiteRepository,
    *,
    dataset_root: str | Path,
    overlay_root: str | Path,
    as_of: date,
    overlay_source_id: str = DEFAULT_MARKET_OVERLAY_SOURCE_ID,
    reviewed_at: datetime | None = None,
    persist: bool = True,
    _baseline_market: _BaselineMarket | None = None,
    _overlay_store: _OverlayMarket | None = None,
    holding_context: HoldingKnowledgeContext | None = None,
) -> HoldingTreeReviewSummary:
    """Load verified local evidence and run the holding-management core."""

    resolved_reviewed_at = (
        holding_context.known_at
        if reviewed_at is None and holding_context is not None
        else reviewed_at
    )
    loaded = load_active_holding_histories(
        repository,
        dataset_root=dataset_root,
        overlay_root=overlay_root,
        as_of=as_of,
        overlay_source_id=overlay_source_id,
        _baseline_market=_baseline_market,
        _overlay_store=_overlay_store,
        _holding_context=holding_context,
    )
    if loaded is None:
        return review_active_holdings(
            repository,
            {},
            as_of=as_of,
            verified_data_cutoff=as_of,
            reviewed_at=resolved_reviewed_at,
            persist=persist,
            holding_context=holding_context,
        )
    clearances = _local_company_action_clearances(
        repository,
        cutoff=loaded.data_cutoff,
        holding_context=holding_context,
    )
    return review_active_holdings(
        repository,
        loaded.histories,
        as_of=loaded.data_cutoff,
        verified_data_cutoff=loaded.data_cutoff,
        verified_close=loaded.verified_close,
        reviewed_at=resolved_reviewed_at,
        persist=persist,
        company_action_clear_by_symbol=clearances,
        holding_context=holding_context,
    )


def build_evening_holding_review(
    repository: SQLiteRepository,
    *,
    dataset_root: str | Path,
    overlay_root: str | Path,
    decision_date: date,
    overlay_source_id: str = DEFAULT_MARKET_OVERLAY_SOURCE_ID,
    reviewed_at: datetime | None = None,
    persist: bool = True,
    _baseline_market: _BaselineMarket | None = None,
    _overlay_store: _OverlayMarket | None = None,
    holding_context: HoldingKnowledgeContext | None = None,
) -> HoldingTreeReviewSummary:
    """Stable, side-effect-bounded evening-digest integration entrypoint."""

    resolved_reviewed_at = reviewed_at or datetime.now(UTC)
    context = holding_context
    if context is None:
        current = get_active_holding_portfolio(repository)
        if current is not None:
            context = holding_knowledge_context(
                current,
                known_at=resolved_reviewed_at,
            )
    return run_active_holding_review(
        repository,
        dataset_root=dataset_root,
        overlay_root=overlay_root,
        as_of=decision_date,
        overlay_source_id=overlay_source_id,
        reviewed_at=resolved_reviewed_at,
        persist=persist,
        _baseline_market=_baseline_market,
        _overlay_store=_overlay_store,
        holding_context=context,
    )


def _local_company_action_clearances(
    repository: SQLiteRepository,
    *,
    cutoff: date,
    holding_context: HoldingKnowledgeContext | None = None,
) -> dict[str, CompanyActionClearance]:
    """Read only explicit dated local evidence; never infer from prev_close."""

    portfolio = (
        get_active_holding_portfolio(repository, as_of=cutoff)
        if holding_context is None
        else resolve_current_holding_context(repository, holding_context)
    )
    if portfolio is None:
        return {}
    clearances: dict[str, CompanyActionClearance] = {}
    for holding in portfolio.positions:
        metadata = dict(holding.metadata)
        explicit = metadata.get("company_action_clear")
        through_raw = metadata.get("company_action_clear_through")
        source = str(metadata.get("company_action_evidence_source", "")).strip()
        evidence_id = str(metadata.get("company_action_evidence_id", "")).strip()
        if not isinstance(explicit, bool) or through_raw is None or not source or not evidence_id:
            continue
        try:
            through_date = date.fromisoformat(str(through_raw))
        except ValueError:
            continue
        if through_date < holding.entry_date or through_date > cutoff:
            # A pre-entry or future-dated assertion is not evidence for this
            # point-in-time holding review.
            continue
        clearances[holding.symbol] = CompanyActionClearance(
            symbol=holding.symbol,
            through_date=through_date,
            clear=explicit,
            source=source,
            evidence_id=evidence_id,
        )
    return clearances
