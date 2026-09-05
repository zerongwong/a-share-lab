"""Local-only immutable journal for a signal-exit continuous strategy.

This module is deliberately independent of the legacy six-horizon archives and
the user's holding ledger. A decision is not a trade. It neither infers fills
from recommendations nor changes actual holdings, sends notifications, or orders.

``record_continuous_valuation`` accepts a *complete closing snapshot*, not a
trade instruction. Units include all open positions after any confirmed trades
and corporate actions. ``cash_before_fees_and_dividends`` is that snapshot's
cash before ONLY the supplied ``fees`` and ``dividend_cash`` adjustments; all
earlier adjustments must already be included. Do not supply net cash and then
deduct the same fees again. Net assets equal marked open positions plus cash
minus fees plus dividend cash. Both adjustments must be explicit, even if zero.
The caller must establish the evidence flags; this pure local accounting layer
cannot independently certify a price source, a corporate action, or a fill.

Evidence requires positions_complete, cash_complete, trades_complete,
fees_complete, dividends_complete, corporate_actions_complete, prices_verified,
external_flows_complete, nonempty evidence_ids, and covered_from/covered_through
ISO dates spanning the preceding snapshot through the current one. Actual mode
additionally requires user_confirmed. Shadow mode always remains hypothetical.
Unknown evidence yields unavailable and null returns, never invented zero P&L.

V1 supports only zero-external-flow return chains. ``external_flow`` is the net
external flow SINCE the previous snapshot, not a cumulative balance. A nonzero
or unknown flow invalidates normalized NAV/returns; an independently complete
snapshot may still disclose net assets locally. Any unavailable valuation or
flow breaks this stream's return chain permanently. A deliberately new
portfolio_id is needed for a fresh baseline; no silent rebasing is performed.
This conservative limitation avoids treating deposits as investment profit or
pretending to know the timing of flows for time-weighted return calculations.

The first fully evidenced positive snapshot establishes NAV=1, not a profitable
trade. Returns before that inception snapshot are unknown; to measure inception
trading costs, record the funded cash baseline BEFORE those first trades.
Subsequent NAV includes unrealized losses/gains, cash, fees, and verified
dividend cash. Closed-trade-only selection is not used. Methods and actual vs.
shadow streams cannot be mixed. Backdated inserts and same-day replacements are
rejected. Each input/result is hashed; SQLite triggers prevent UPDATE, DELETE,
and INSERT OR REPLACE. Database administrators can remove triggers, so this is
application-level immutability, not tamper-proof external custody.

Writes use savepoints and never commit/rollback an existing caller transaction.
Only the two continuous_* tables and their own indexes/triggers are created.
Private accounting results are local-only and not approved for external delivery.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Literal
from uuid import uuid4

CONTINUOUS_METHOD_VERSION = "continuous-signal-v1"
CONTINUOUS_VALUATION_METHOD = "continuous-zero-flow-nav-v1"
_EVIDENCE_FLAGS = (
    "positions_complete",
    "cash_complete",
    "trades_complete",
    "fees_complete",
    "dividends_complete",
    "corporate_actions_complete",
    "prices_verified",
    "external_flows_complete",
)


@contextmanager
def _savepoint(connection: sqlite3.Connection):
    name = f"continuous_journal_{uuid4().hex}"
    connection.execute(f"SAVEPOINT {name}")
    try:
        yield
    except BaseException:
        connection.execute(f"ROLLBACK TO {name}")
        connection.execute(f"RELEASE {name}")
        raise
    else:
        connection.execute(f"RELEASE {name}")


def ensure_continuous_strategy_schema(connection: sqlite3.Connection) -> None:
    """Create independent tables without touching or committing legacy data."""

    with _savepoint(connection):
        connection.execute(
            """CREATE TABLE IF NOT EXISTS continuous_strategy_decisions (
                decision_id TEXT PRIMARY KEY NOT NULL,
                as_of TEXT NOT NULL,
                method_version TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_hash TEXT NOT NULL
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS continuous_strategy_valuations (
                valuation_id TEXT PRIMARY KEY NOT NULL,
                portfolio_id TEXT NOT NULL,
                mode TEXT NOT NULL CHECK (mode IN ('actual', 'shadow')),
                as_of TEXT NOT NULL,
                method_version TEXT NOT NULL,
                input_json TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                result_json TEXT NOT NULL,
                record_hash TEXT NOT NULL,
                UNIQUE (portfolio_id, mode, as_of)
            )"""
        )
        for table, identifier in (
            ("continuous_strategy_decisions", "decision_id"),
            ("continuous_strategy_valuations", "valuation_id"),
        ):
            for operation in ("UPDATE", "DELETE"):
                connection.execute(
                    f"""CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()}
                    BEFORE {operation} ON {table}
                    BEGIN SELECT RAISE(ABORT, 'continuous journal is immutable'); END"""
                )
            # REPLACE can otherwise bypass DELETE triggers when recursive
            # triggers are off (SQLite's default).
            conflict = f"{identifier} = NEW.{identifier}"
            if table == "continuous_strategy_valuations":
                conflict += (
                    " OR (portfolio_id = NEW.portfolio_id AND mode = NEW.mode"
                    " AND as_of = NEW.as_of)"
                )
            connection.execute(
                f"""CREATE TRIGGER IF NOT EXISTS {table}_no_replace
                BEFORE INSERT ON {table}
                WHEN EXISTS (SELECT 1 FROM {table} WHERE {conflict})
                BEGIN SELECT RAISE(ABORT, 'continuous journal is immutable'); END"""
            )


def archive_continuous_decision(
    connection: sqlite3.Connection,
    decision_id: str,
    as_of: date,
    payload: Mapping[str, Any],
    method_version: str = CONTINUOUS_METHOD_VERSION,
) -> dict[str, Any]:
    """Archive one new decision; identical retries are reads, conflicts fail."""

    if not isinstance(payload, Mapping):
        raise TypeError("decision payload must be a JSON object")
    envelope = {
        "decision_id": _nonblank(decision_id, "decision_id"),
        "as_of": _date(as_of, "as_of").isoformat(),
        "method_version": _nonblank(method_version, "method_version"),
        "payload": _json_value(payload),
        "record_nature": "decision_not_execution",
        "external_delivery_allowed": False,
        "orders_enabled": False,
    }
    serialized = _canonical(envelope)
    digest = _sha(serialized)
    with _savepoint(connection):
        ensure_continuous_strategy_schema(connection)
        row = connection.execute(
            """SELECT payload_json, payload_hash FROM continuous_strategy_decisions
            WHERE decision_id = ?""",
            (decision_id,),
        ).fetchone()
        if row is not None:
            if _sha(row[0]) != row[1]:
                raise ValueError("continuous decision integrity check failed")
            if row[0] != serialized or row[1] != digest:
                raise ValueError("decision_id already has different immutable content")
            return json.loads(row[0])
        connection.execute(
            """INSERT INTO continuous_strategy_decisions
            (decision_id, as_of, method_version, payload_json, payload_hash)
            VALUES (?, ?, ?, ?, ?)""",
            (decision_id, envelope["as_of"], method_version, serialized, digest),
        )
    return envelope


def record_continuous_valuation(
    connection: sqlite3.Connection,
    *,
    valuation_id: str,
    portfolio_id: str,
    as_of: date,
    mode: Literal["actual", "shadow"],
    position_units: Mapping[str, float],
    close_prices: Mapping[str, float | None],
    cash_before_fees_and_dividends: float | None,
    fees: float | None,
    dividend_cash: float | None,
    evidence: Mapping[str, Any],
    external_flow: float | None = None,
    method_version: str = CONTINUOUS_METHOD_VERSION,
) -> dict[str, Any]:
    """Archive an explicit full snapshot; return null P&L when evidence fails.

    No external flow is assumed by default: callers must explicitly pass 0.0
    and attest complete flow evidence for a valid no-flow return observation.
    """

    _nonblank(valuation_id, "valuation_id")
    _nonblank(portfolio_id, "portfolio_id")
    _nonblank(method_version, "method_version")
    cutoff = _date(as_of, "as_of")
    if mode not in ("actual", "shadow"):
        raise ValueError("mode must explicitly be actual or shadow")
    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a JSON object")
    units = _numeric_mapping(position_units, "position_units", optional=False)
    prices = _numeric_mapping(close_prices, "close_prices", optional=True)
    cash = _amount(cash_before_fees_and_dividends, "cash_before_fees_and_dividends")
    fee_amount = _amount(fees, "fees")
    distribution = _amount(dividend_cash, "dividend_cash")
    flow = _amount(external_flow, "external_flow", nonnegative=False)
    normalized_evidence = _json_value(evidence)
    for flag in (*_EVIDENCE_FLAGS, "user_confirmed"):
        if flag in normalized_evidence and not isinstance(normalized_evidence[flag], bool):
            raise TypeError(f"evidence {flag} must be a boolean")
    inputs = {
        "valuation_id": valuation_id,
        "portfolio_id": portfolio_id,
        "as_of": cutoff.isoformat(),
        "mode": mode,
        "method_version": method_version,
        "position_units": units,
        "close_prices": prices,
        "cash_before_fees_and_dividends": cash,
        "fees": fee_amount,
        "dividend_cash": distribution,
        "external_flow": flow,
        "evidence": normalized_evidence,
    }
    serialized = _canonical(inputs)
    with _savepoint(connection):
        ensure_continuous_strategy_schema(connection)
        existing = connection.execute(
            """SELECT input_json, input_hash, result_json, record_hash
            FROM continuous_strategy_valuations WHERE valuation_id = ?""",
            (valuation_id,),
        ).fetchone()
        if existing is not None:
            result = _verified_result(existing)
            if existing[0] != serialized:
                raise ValueError("valuation_id already has different immutable content")
            return result
        preceding = connection.execute(
            """SELECT input_json, input_hash, result_json, record_hash
            FROM continuous_strategy_valuations WHERE portfolio_id = ? AND mode = ?
            ORDER BY as_of DESC LIMIT 1""",
            (portfolio_id, mode),
        ).fetchone()
        previous = None if preceding is None else _verified_result(preceding)
        if previous is not None:
            if previous["as_of"] >= cutoff.isoformat():
                raise ValueError("continuous snapshots must be strictly chronological")
            if previous["method_version"] != method_version:
                raise ValueError("a continuous stream cannot change method_version")
        result = _value_snapshot(inputs, previous)
        result["previous_record_hash"] = None if preceding is None else preceding[3]
        encoded_result = _canonical(result)
        connection.execute(
            """INSERT INTO continuous_strategy_valuations
            (valuation_id, portfolio_id, mode, as_of, method_version,
             input_json, input_hash, result_json, record_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                valuation_id,
                portfolio_id,
                mode,
                cutoff.isoformat(),
                method_version,
                serialized,
                _sha(serialized),
                encoded_result,
                _sha(serialized + "\n" + encoded_result),
            ),
        )
    return result


def _value_snapshot(inputs: dict, previous: dict | None) -> dict[str, Any]:
    evidence = inputs["evidence"]
    reasons = [f"{flag}_not_verified" for flag in _EVIDENCE_FLAGS if not evidence.get(flag, False)]
    if inputs["mode"] == "actual" and not evidence.get("user_confirmed", False):
        reasons.append("actual_snapshot_not_user_confirmed")
    ids = evidence.get("evidence_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(x, str) or not x.strip() for x in ids)
    ):
        reasons.append("evidence_ids_missing")
    interval_start = inputs["as_of"] if previous is None else previous["as_of"]
    try:
        covered_from = _date(evidence.get("covered_from"), "covered_from").isoformat()
        covered_through = _date(evidence.get("covered_through"), "covered_through").isoformat()
        if covered_from > interval_start or covered_through < inputs["as_of"]:
            reasons.append("evidence_does_not_cover_valuation_interval")
    except (TypeError, ValueError):
        reasons.append("evidence_coverage_unknown")
    values: list[float] = []
    for symbol, units in inputs["position_units"].items():
        if units == 0.0:
            continue
        price = inputs["close_prices"].get(symbol)
        if price is None or price <= 0.0:
            reasons.append(f"close_price_missing_or_nonpositive:{symbol}")
        else:
            values.append(units * price)
    for field in ("cash_before_fees_and_dividends", "fees", "dividend_cash"):
        if inputs[field] is None:
            reasons.append(f"{field}_unknown")
    market_value = cash_after = net_assets = None
    if not reasons:
        try:
            market_value = math.fsum(values)
            cash_after = math.fsum(
                (inputs["cash_before_fees_and_dividends"], -inputs["fees"], inputs["dividend_cash"])
            )
            net_assets = math.fsum((market_value, cash_after))
            if not all(math.isfinite(x) for x in (market_value, cash_after, net_assets)):
                raise ValueError("nonfinite valuation")
            if cash_after < 0.0:
                raise ValueError("financing is not supported")
        except (ValueError, OverflowError):
            reasons.append("invalid_net_assets_or_financed_cash")
            market_value = cash_after = net_assets = None
    snapshot_complete = not reasons
    if inputs["external_flow"] is None:
        reasons.append("external_flow_unknown")
    elif inputs["external_flow"] != 0.0:
        reasons.append("external_flow_not_supported_by_zero_flow_nav_v1")
    if previous is not None and previous["status"] != "ready":
        reasons.append("previous_return_chain_unavailable_no_silent_rebase")
    if previous is None and net_assets is not None and net_assets <= 0.0:
        reasons.append("positive_initial_net_assets_required")
    if previous is not None and previous["net_assets"] == 0.0:
        reasons.append("previous_net_assets_zero_return_undefined")
    nav = cumulative_return = period_return = baseline = None
    if not reasons:
        baseline = net_assets if previous is None else previous["baseline_net_assets"]
        nav = net_assets / baseline
        cumulative_return = nav - 1.0
        period_return = None if previous is None else net_assets / previous["net_assets"] - 1.0
        if not all(math.isfinite(x) for x in (nav, cumulative_return)) or (
            period_return is not None and not math.isfinite(period_return)
        ):
            reasons.append("nonfinite_normalized_return")
            nav = cumulative_return = period_return = baseline = None
    return {
        "valuation_id": inputs["valuation_id"],
        "portfolio_id": inputs["portfolio_id"],
        "as_of": inputs["as_of"],
        "mode": inputs["mode"],
        "method_version": inputs["method_version"],
        "valuation_method_version": CONTINUOUS_VALUATION_METHOD,
        "status": "unavailable" if reasons else "ready",
        "snapshot_status": "ready" if snapshot_complete else "unavailable",
        "reasons": reasons,
        "open_position_market_value": market_value,
        "cash_after_fees_and_dividends": cash_after,
        "net_assets": net_assets,
        "baseline_net_assets": baseline,
        "normalized_nav": nav,
        "cumulative_return": cumulative_return,
        "previous_snapshot_return": period_return,
        "previous_valuation_id": None if previous is None else previous["valuation_id"],
        "performance_nature": (
            "actual_user_confirmed_accounting"
            if inputs["mode"] == "actual"
            else "shadow_simulation_not_actual_account"
        ),
        "external_delivery_allowed": False,
        "orders_enabled": False,
    }


def _verified_result(row: Any) -> dict[str, Any]:
    if _sha(row[0]) != row[1] or _sha(row[0] + "\n" + row[2]) != row[3]:
        raise ValueError("continuous valuation integrity check failed")
    return json.loads(row[2])


def _numeric_mapping(value: Mapping, label: str, *, optional: bool) -> dict:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result = {}
    for key, amount in value.items():
        _nonblank(key, f"{label} key")
        result[key] = _amount(amount, f"{label}:{key}")
        if result[key] is None and not optional:
            raise ValueError(f"{label}:{key} cannot be unknown")
    return result


def _amount(value: Any, label: str, *, nonnegative: bool = True) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number or None")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(normalized) or (nonnegative and normalized < 0.0):
        raise ValueError(f"{label} must be finite and nonnegative")
    return normalized


def _nonblank(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonblank unpadded string")
    return value


def _date(value: Any, label: str) -> date:
    if isinstance(value, datetime):
        raise TypeError(f"{label} must be a date, not datetime")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        parsed = date.fromisoformat(value)
        if parsed.isoformat() == value:
            return parsed
    raise TypeError(f"{label} must be a date or ISO calendar date")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError("value is not strictly JSON serializable")


def _canonical(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
