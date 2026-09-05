"""Record an explicitly supplied, local-only actual continuous-account snapshot.

``run_continuous_performance(repository, as_of=...)`` reads only the current
user holding revision and its metadata['continuous_valuation_snapshots'][date].
It does not fetch prices, infer recommendations as fills, alter holdings, or
send account performance. Absence of a snapshot is a zero-write inputs_missing
result: an unattended daily call must not create a broken unavailable NAV chain.

Each snapshot must include as_of and data_cutoff (the exact ISO close date),
holding_revision_id and holding_version (the current explicit revision),
mode='actual', method_version='continuous-signal-v1', and a stable portfolio_id
identifying this continuous accounting stream, NOT a legacy cohort or changing
holding revision. Required inputs are position_units, close_prices,
cash_before_fees_and_dividends, fees, dividend_cash, external_flow, and evidence.
Evidence follows continuous_strategy_journal's full-interval contract and must
explicitly contain user_confirmed=True. Units must cover exactly the current
holding symbols. The valuation id is deterministically derived, never supplied
by a recommendation. All numbers and private evidence remain in the local DB.

This version cannot produce unattended actual NAV until the caller supplies
confirmed trades/units, full cash and fee information, verified close prices,
cash-flow history, and complete corporate-action evidence. An explicit unknown
value may be journaled as unavailable by the accounting engine; the runner
cannot repair that evidence. This is bookkeeping plumbing, not a completed
backtest or proof that the trading strategy is profitable.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from ashare_lab.ports.market_data import normalize_symbol
from ashare_lab.services.continuous_strategy_journal import (
    CONTINUOUS_METHOD_VERSION,
    record_continuous_valuation,
)
from ashare_lab.services.holding_ledger import get_active_holding_portfolio

_REQUIRED = (
    "position_units",
    "close_prices",
    "cash_before_fees_and_dividends",
    "fees",
    "dividend_cash",
    "external_flow",
    "evidence",
)


def run_continuous_performance(repository, *, as_of: date) -> dict[str, Any]:
    """Return status/counters only; never leak symbols, units, cash, or returns."""

    if not isinstance(as_of, date) or isinstance(as_of, datetime):
        raise TypeError("as_of must be a calendar date")
    try:
        portfolio = get_active_holding_portfolio(repository)
        if portfolio is None:
            return _summary(as_of, "inputs_missing", "no_explicit_holding_revision")
        snapshots = portfolio.metadata.get("continuous_valuation_snapshots")
        if snapshots is None:
            return _summary(as_of, "inputs_missing", "explicit_valuation_snapshot_missing")
        if not isinstance(snapshots, Mapping):
            return _summary(as_of, "inputs_invalid", "snapshot_collection_invalid")
        snapshot = snapshots.get(as_of.isoformat())
        if snapshot is None:
            return _summary(as_of, "inputs_missing", "explicit_valuation_snapshot_missing")
        if not isinstance(snapshot, Mapping):
            return _summary(as_of, "inputs_invalid", "snapshot_object_invalid", seen=1)
        validated = _validate_snapshot(snapshot, portfolio, as_of)
        valuation_id = str(
            uuid5(
                NAMESPACE_URL,
                f"a-share-lab:continuous-actual-valuation:{CONTINUOUS_METHOD_VERSION}:"
                f"{validated['portfolio_id']}:{as_of.isoformat()}",
            )
        )
        with repository.connection() as connection:
            table_exists = connection.execute(
                """SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'continuous_strategy_valuations'"""
            ).fetchone()
            existing = bool(
                table_exists
                and connection.execute(
                    "SELECT 1 FROM continuous_strategy_valuations WHERE valuation_id = ?",
                    (valuation_id,),
                ).fetchone()
            )
            result = record_continuous_valuation(
                connection,
                valuation_id=valuation_id,
                portfolio_id=validated["portfolio_id"],
                as_of=as_of,
                mode="actual",
                position_units=validated["position_units"],
                close_prices=validated["close_prices"],
                cash_before_fees_and_dividends=snapshot["cash_before_fees_and_dividends"],
                fees=snapshot["fees"],
                dividend_cash=snapshot["dividend_cash"],
                evidence=validated["evidence"],
                external_flow=snapshot["external_flow"],
                method_version=CONTINUOUS_METHOD_VERSION,
            )
        ready = result["status"] == "ready"
        return _summary(
            as_of,
            "already_recorded" if existing else "recorded" if ready else "unavailable",
            "explicit_snapshot_accounting_complete" if ready else "accounting_evidence_incomplete",
            seen=1,
            recorded=0 if existing else 1,
            nav_available=int(ready),
        )
    except (ValueError, TypeError, KeyError):
        # Never copy an exception: validators may include a symbol or private
        # snapshot content in their message.
        return _summary(
            as_of, "inputs_invalid", "snapshot_validation_or_immutable_conflict", seen=1
        )
    except (sqlite3.Error, OSError):
        return _summary(as_of, "error_retry_later", "local_accounting_storage_unavailable")


def _validate_snapshot(snapshot: Mapping, portfolio, as_of: date) -> dict[str, Any]:
    if any(key not in snapshot for key in _REQUIRED):
        raise ValueError("explicit accounting fields missing")
    if (
        snapshot.get("as_of") != as_of.isoformat()
        or snapshot.get("data_cutoff") != as_of.isoformat()
    ):
        raise ValueError("exact verified snapshot cutoff required")
    if (
        snapshot.get("holding_revision_id") != portfolio.id
        or isinstance(snapshot.get("holding_version"), bool)
        or snapshot.get("holding_version") != portfolio.version
    ):
        raise ValueError("snapshot must match current user holding revision")
    if (
        snapshot.get("mode") != "actual"
        or snapshot.get("method_version") != CONTINUOUS_METHOD_VERSION
    ):
        raise ValueError("explicit continuous actual method required")
    portfolio_id = snapshot.get("portfolio_id")
    if (
        not isinstance(portfolio_id, str)
        or not portfolio_id.strip()
        or portfolio_id != portfolio_id.strip()
    ):
        raise ValueError("stable explicit continuous portfolio id required")
    evidence = snapshot["evidence"]
    if not isinstance(evidence, Mapping) or evidence.get("user_confirmed") is not True:
        raise ValueError("actual accounting requires user confirmation")
    units = _normalized_symbols(snapshot["position_units"])
    prices = _normalized_symbols(snapshot["close_prices"])
    current_symbols = {normalize_symbol(row.symbol) for row in portfolio.positions}
    if set(units) != current_symbols:
        raise ValueError("snapshot units must cover exactly the current holding membership")
    return {
        "portfolio_id": portfolio_id,
        "position_units": units,
        "close_prices": prices,
        "evidence": {
            **evidence,
            "holding_revision_id": portfolio.id,
            "holding_version": portfolio.version,
        },
    }


def _normalized_symbols(values: Any) -> dict:
    if not isinstance(values, Mapping):
        raise TypeError("stock inputs must be mappings")
    normalized = {}
    for symbol, value in values.items():
        canonical = normalize_symbol(symbol)
        if canonical in normalized:
            raise ValueError("duplicate stock aliases in explicit snapshot")
        normalized[canonical] = value
    return normalized


def _summary(as_of, status, reason_code, *, seen=0, recorded=0, nav_available=0):
    return {
        "status": status,
        "reason_code": reason_code,
        "as_of": as_of.isoformat(),
        "mode": "actual",
        "snapshots_seen": seen,
        "valuations_recorded": recorded,
        "nav_available_count": nav_available,
        "performance_scope": "explicit_snapshot_local_only",
        "external_delivery_allowed": False,
        "orders_enabled": False,
    }
