"""Replay provider-accepted recommendations independently of actual holdings.

The lightweight ingestion path performs local work only.  The daily replay
reads verified unadjusted overlays and requests public company-action evidence
by stock code; no holding, cost, size, weight or account identifier is supplied
to the provider.  This service neither sends notifications nor changes rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from ashare_lab.adapters.market_overlay_store import MarketOverlayStore
from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.bootstrap import application_data_dir, project_root
from ashare_lab.domain.data_sources import DEFAULT_MARKET_OVERLAY_SOURCE_ID
from ashare_lab.ports.market_data import normalize_symbol

TRACKING_START = date(2026, 10, 8)
_STREAMS = {"identity", "dividend", "allotment", "share_change"}


def _accepted_reports(connection, *, report_id=None, through=None):
    required = {
        "recommendation_reports", "recommendation_delivery_events", "continuous_strategy_decisions"
    }
    present = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if not required <= present:
        return [], 0
    query = """
        SELECT r.id, r.plan_for_date, r.common_cutoff, r.method_version,
               d.as_of, d.method_version, d.payload_json, d.payload_hash,
               MIN(e.attempted_at)
        FROM recommendation_reports r
        JOIN continuous_strategy_decisions d ON d.decision_id = r.id
        JOIN recommendation_delivery_events e ON e.report_id = r.id
        WHERE r.archive_nature = 'original' AND r.plan_for_date >= ?
          AND e.provider_status = 'provider_accepted'
          AND e.delivery_kind = 'evening_provider_accepted'
    """
    params: list[Any] = [TRACKING_START.isoformat()]
    if report_id is not None:
        query += " AND r.id = ?"
        params.append(report_id)
    if through is not None:
        query += " AND r.plan_for_date <= ?"
        params.append(through.isoformat())
    query += " GROUP BY r.id ORDER BY r.plan_for_date, MIN(e.attempted_at), r.id"
    reports, rejected = [], 0
    for row in connection.execute(query, params):
        try:
            identifier, plan, cutoff, method, decision_cutoff, decision_method, raw, digest, sent = row
            if hashlib.sha256(raw.encode("utf-8")).hexdigest() != digest:
                raise ValueError("decision hash mismatch")
            document = json.loads(raw)
            payload = document["payload"]
            if (
                document["decision_id"] != identifier
                or document["as_of"] != cutoff
                or decision_cutoff != cutoff
                or decision_method != method
                or document["method_version"] != method
                or payload.get("plan_for_date") != plan
                or payload.get("mode") != "continuous"
            ):
                raise ValueError("archive identity mismatch")
            published = datetime.fromisoformat(sent.replace("Z", "+00:00"))
            if published.tzinfo is None or published.utcoffset() is None:
                raise ValueError("publication time must be aware")
            entries = _qualified_entries(payload)
            reports.append({
                "report_id": identifier,
                "plan_date": date.fromisoformat(plan),
                "published_at": published,
                "cutoff": date.fromisoformat(cutoff),
                "method_version": method,
                "entries": entries,
            })
        except (ValueError, TypeError, KeyError, AttributeError):
            rejected += 1
    return reports, rejected


def _qualified_entries(payload):
    """Observation lists are never promoted to recommendation fills."""
    model = payload.get("market_model_portfolio") or {}
    if not isinstance(model, Mapping):
        raise ValueError("market plan must be an object")
    result = {}
    for collection in (model.get("entries", []), payload.get("entries", [])):
        if not isinstance(collection, list):
            raise ValueError("entries must be a list")
        for entry in collection:
            if not isinstance(entry, Mapping) or entry.get("entry_qualified") is not True:
                continue
            symbol = normalize_symbol(entry["symbol"])
            normalized = {**entry, "symbol": symbol}
            prior = result.get(symbol)
            if prior is not None and prior.get("entry_rule") != normalized.get("entry_rule"):
                raise ValueError("conflicting conditions for one published symbol")
            result.setdefault(symbol, normalized)
    return list(result.values())


def ingest_published_recommendations(repository, *, report_id=None, through=None):
    """Idempotently register accepted original reports, including empty ones."""
    from ashare_lab.services.recommendation_tracking import ensure_tracking_schema, ingest_report

    with repository.connection() as connection:
        ensure_tracking_schema(connection)
        reports, rejected = _accepted_reports(connection, report_id=report_id, through=through)
        recorded = sum(bool(ingest_report(connection, **report)) for report in reports)
    return {
        "reports_seen": len(reports),
        "reports_ingested": recorded,
        "entries_seen": sum(len(report["entries"]) for report in reports),
        "archive_rejected": rejected,
    }


def _verified_chain(store, *, source_id, start, through, symbols):
    manifest = store.read_verified_manifest(source_id=source_id)
    required = {"trade_date", "previous_trade_date", "adjustment"}
    if not required <= set(manifest.columns):
        raise ValueError("verified session manifest incomplete")
    rows = manifest.copy()
    rows["trade_date"] = pd.to_datetime(rows["trade_date"], errors="raise").dt.date
    rows["previous_trade_date"] = pd.to_datetime(
        rows["previous_trade_date"], errors="raise"
    ).dt.date
    rows = rows.loc[(rows.trade_date >= start) & (rows.trade_date <= through)]
    if rows.trade_date.duplicated().any() or not rows.adjustment.eq("none").all():
        raise ValueError("ambiguous or adjusted session manifest")
    rows = rows.sort_values("trade_date")
    if rows.empty or rows.iloc[0].trade_date != start:
        return [], {}, "anchor_session_missing"
    dates, bars = [], {symbol: {} for symbol in symbols}
    previous = None
    reason = "verified_chain_complete"
    for row in rows.itertuples(index=False):
        day = row.trade_date
        if row.previous_trade_date >= day or (
            previous is not None and row.previous_trade_date != previous
        ):
            reason = "verified_session_chain_gap"
            break
        try:
            frame = store.read_verified_daily(day, source_id=source_id, asset_kind="stocks")
            if not {"symbol", "trade_date", "open", "high", "low", "close"} <= set(frame.columns):
                raise ValueError("daily OHLC evidence incomplete")
            frame = frame.copy()
            frame["symbol"] = frame.symbol.map(normalize_symbol)
            frame = frame.loc[frame.symbol.isin(symbols)]
            if frame.symbol.duplicated().any():
                raise ValueError("duplicate symbol in daily evidence")
            day_bars = {}
            for record in frame.to_dict("records"):
                if pd.Timestamp(record["trade_date"]).date() != day:
                    raise ValueError("daily evidence cutoff mismatch")
                values = {}
                for name in (
                    "open", "high", "low", "close", "prev_close", "volume_shares", "amount_cny",
                    "volume", "amount",
                ):
                    value = record.get(name)
                    if value is not None:
                        try:
                            number = float(value)
                        except (ValueError, TypeError):
                            continue
                        if math.isfinite(number):
                            values[name] = number
                day_bars[record["symbol"]] = values
        except Exception:
            reason = "verified_daily_evidence_unavailable"
            break
        for symbol, values in day_bars.items():
            bars[symbol][day] = values
        dates.append(day)
        previous = day
    if dates and dates[-1] < through and reason == "verified_chain_complete":
        reason = "latest_available_verified_session"
    return dates, bars, reason


def _company_clearances(
    symbols, through, *, reader=None, budget_seconds=45, on_attempt=None,
):
    if reader is None:
        from ashare_lab.adapters.cninfo_company_actions import read_cninfo_company_actions
        reader = read_cninfo_company_actions
    result = {
        symbol: {
            "status": "unknown", "coverage_from": None, "coverage_through": None,
            "event_dates": [], "complete": False, "source": "cninfo",
            "reason": "company_action_evidence_unavailable",
        }
        for symbol in symbols
    }
    deadline = time.monotonic() + budget_seconds
    for offset in range(0, len(symbols), 5):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        batch = tuple(symbols[offset:offset + 5])
        if on_attempt is not None:
            on_attempt(batch)
        try:
            values = reader(batch, through, timeout_seconds=min(10.0, remaining))
            if len(values) != len(batch) or {row.symbol for row in values} != set(batch):
                continue
            for row in values:
                complete = (
                    {receipt.stream for receipt in row.stream_receipts} == _STREAMS
                    and all(receipt.complete for receipt in row.stream_receipts)
                    and row.response_hash is not None
                )
                result[row.symbol] = {
                    "status": row.status.value,
                    "coverage_from": row.coverage_from.isoformat(),
                    "coverage_through": row.coverage_through.isoformat(),
                    "event_dates": [day.isoformat() for day in row.event_dates],
                    "complete": complete,
                    "source": "cninfo",
                    "reason": row.reason_code,
                    "response_hash": row.response_hash,
                    "knowledge_time": row.knowledge_time.isoformat(),
                }
        except Exception:
            continue
    return result


def _terminal_coverage_requirements(previous, reports, sessions):
    """Freeze provider work only for fully represented, final research episodes."""
    current_ids = {}
    for report in reports:
        for entry in report["entries"]:
            current_ids.setdefault(entry["symbol"], set()).add(report["report_id"])
    episodes_by_symbol = {}
    for episode in previous.get("episodes", []):
        episodes_by_symbol.setdefault(episode["symbol"], []).append(episode)
    requirements = {}
    for symbol, ids in current_ids.items():
        episodes = episodes_by_symbol.get(symbol, [])
        represented = {rid for episode in episodes for rid in episode.get("report_ids", [])}
        if not episodes or represented != ids or any(
            episode["status"] not in {"closed", "unfilled"} for episode in episodes
        ):
            continue
        dates = []
        for episode in episodes:
            if episode["status"] == "closed":
                day = episode.get("exit_date")
            elif episode.get("reason") in {
                "late_publication_no_retroactive_entry", "daily_entry_condition_not_confirmed",
            }:
                day = episode["plan_date"]
            else:
                plan_date = date.fromisoformat(episode["plan_date"])
                following = next((day for day in sessions if day > plan_date), None)
                day = following.isoformat() if following else previous.get("through")
            if day is None:
                break
            dates.append(date.fromisoformat(day))
        if len(dates) == len(episodes):
            requirements[symbol] = max(dates)
    return requirements


def _cached_company_clearances(
    repository, symbols, through, *, reader=None, terminal_requirements=None,
):
    """Retain verified historical coverage without extending it past its cutoff."""
    cached, attempts = {}, {}
    terminal_requirements = terminal_requirements or {}
    with repository.connection() as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS tracking_company_action_cache (
            symbol TEXT NOT NULL, through_date TEXT NOT NULL,
            evidence_json TEXT NOT NULL, evidence_hash TEXT NOT NULL,
            PRIMARY KEY (symbol, through_date))""")
        connection.execute("""CREATE TABLE IF NOT EXISTS tracking_company_action_attempts (
            symbol TEXT PRIMARY KEY NOT NULL, last_attempt INTEGER NOT NULL)""")
        for symbol in symbols:
            for row in connection.execute(
                """SELECT evidence_json, evidence_hash, through_date
                   FROM tracking_company_action_cache
                   WHERE symbol=? AND through_date<=? ORDER BY through_date DESC""",
                (symbol, through.isoformat()),
            ):
                if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
                    continue
                try:
                    evidence = json.loads(row[0])
                    if (
                        evidence.get("complete") is True
                        and evidence.get("source") == "cninfo"
                        and evidence.get("coverage_through") == row[2]
                        and evidence.get("status") in {"clear", "detected"}
                    ):
                        cached[symbol] = evidence
                        break
                except (ValueError, TypeError, AttributeError):
                    continue
            attempt = connection.execute(
                "SELECT last_attempt FROM tracking_company_action_attempts WHERE symbol=?", (symbol,)
            ).fetchone()
            attempts[symbol] = 0 if attempt is None else attempt[0]
    missing = [
        symbol for symbol in symbols
        if symbol not in cached or date.fromisoformat(cached[symbol]["coverage_through"])
        < terminal_requirements.get(symbol, through)
    ]
    # Persistent least-recently-attempted ordering prevents a failing prefix
    # from using every bounded run. Oldest coverage resolves equal priorities.
    missing.sort(key=lambda symbol: (
        attempts[symbol], cached.get(symbol, {}).get("coverage_through") or "", symbol,
    ))
    if missing:
        def mark_attempt(batch):
            with repository.connection() as connection:
                sequence = connection.execute(
                    "SELECT COALESCE(MAX(last_attempt), 0) + 1 FROM tracking_company_action_attempts"
                ).fetchone()[0]
                connection.executemany(
                    """INSERT INTO tracking_company_action_attempts VALUES (?, ?)
                       ON CONFLICT(symbol) DO UPDATE SET last_attempt=excluded.last_attempt""",
                    [(symbol, sequence) for symbol in batch],
                )

        fetched = _company_clearances(missing, through, reader=reader, on_attempt=mark_attempt)
        with repository.connection() as connection:
            for symbol, evidence in fetched.items():
                if evidence["complete"] and evidence["status"] in {"clear", "detected"}:
                    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
                    connection.execute(
                        """INSERT INTO tracking_company_action_cache
                           (symbol, through_date, evidence_json, evidence_hash) VALUES (?, ?, ?, ?)
                           ON CONFLICT(symbol, through_date) DO UPDATE SET
                             evidence_json=excluded.evidence_json,
                             evidence_hash=excluded.evidence_hash""",
                        (symbol, through.isoformat(), encoded,
                         hashlib.sha256(encoded.encode()).hexdigest()),
                    )
        for symbol, evidence in fetched.items():
            # A failed refresh cannot erase already verified historical facts.
            # The old coverage_through is preserved, so later intervals still
            # fail evidence checks in the replay engine.
            if evidence["complete"] or symbol not in cached:
                cached[symbol] = evidence
    return cached


def _joint_terminal_coverage_requirements(previous, reference, reports, sessions):
    """A completed v1 trade must not cut short a still-open price reference."""
    legacy = _terminal_coverage_requirements(previous, reports, sessions)
    by_symbol = {}
    for episode in reference.get("episodes", []):
        by_symbol.setdefault(episode["symbol"], []).append(episode)
    required_ids = {}
    for report in reports:
        for entry in report["entries"]:
            required_ids.setdefault(entry["symbol"], set()).add(report["report_id"])
    result = {}
    for symbol, legacy_day in legacy.items():
        episodes = by_symbol.get(symbol, [])
        ids = {rid for episode in episodes for rid in episode.get("report_ids", [])}
        if not episodes or ids != required_ids.get(symbol) or any(
            episode["status"] not in {"closed", "excluded_late_publication"}
            for episode in episodes
        ):
            continue
        reference_day = max(date.fromisoformat(
            episode["exit_date"] if episode["status"] == "closed" else episode["plan_date"]
        ) for episode in episodes)
        result[symbol] = max(legacy_day, reference_day)
    return result


def run_recommendation_tracking(
    repository, *, overlay_store, through: date, source_id=DEFAULT_MARKET_OVERLAY_SOURCE_ID,
    company_action_reader=None, run_at=None,
):
    """Ingest every original recommendation and replay only a verified chain."""
    from ashare_lab.services.recommendation_reference import reference_summary, replay_reference
    from ashare_lab.services.recommendation_tracking import replay_tracking, tracking_summary

    run_at = run_at or datetime.now(UTC)
    counts = ingest_published_recommendations(repository, through=through)
    with repository.connection() as connection:
        reports, _ = _accepted_reports(connection, through=through)
        previous_summary = tracking_summary(connection)
        previous_reference = reference_summary(connection)
    symbols = sorted({entry["symbol"] for report in reports for entry in report["entries"]})
    outcome = {**counts, "status": "no_qualified_recommendations", "latest_verified_date": None}
    if not symbols:
        with repository.connection() as connection:
            outcome["replay"] = replay_tracking(
                connection, sessions=[], bars={}, through=through, clearances={},
                run_at=run_at,
            )
            outcome["reference_replay"] = replay_reference(
                connection, sessions=[], bars={}, through=through, clearances={}, run_at=run_at,
            )
    else:
        try:
            start = min(report["cutoff"] for report in reports if report["entries"])
            sessions, bars, reason = _verified_chain(
                overlay_store, source_id=source_id, start=start, through=through, symbols=symbols
            )
            outcome["evidence_status"] = reason
            if sessions:
                clearances = _cached_company_clearances(
                    repository, symbols, sessions[-1], reader=company_action_reader,
                    terminal_requirements=_joint_terminal_coverage_requirements(
                        previous_summary, previous_reference, reports, sessions
                    ),
                )
                with repository.connection() as connection:
                    outcome["replay"] = replay_tracking(
                        connection, sessions=sessions, bars=bars, through=sessions[-1],
                        clearances=clearances, run_at=run_at,
                    )
                    outcome["reference_replay"] = replay_reference(
                        connection, sessions=sessions, bars=bars, through=sessions[-1],
                        clearances=clearances, run_at=run_at,
                    )
                outcome["latest_verified_date"] = sessions[-1].isoformat()
                outcome["status"] = "evaluated"
            else:
                outcome["status"] = "awaiting_verified_prices"
        except Exception:
            outcome["status"] = "evidence_or_replay_unavailable"
    with repository.connection() as connection:
        outcome["summary"] = tracking_summary(connection)
        outcome["reference_summary"] = reference_summary(connection)
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description="独立推荐影子追踪；不发送消息、不修改实际持仓")
    parser.add_argument("--through", type=date.fromisoformat,
                        default=datetime.now(ZoneInfo("Asia/Shanghai")).date())
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--backfill", action="store_true", help="重放自2026-10-08起正式推荐；幂等")
    parser.add_argument("--database", type=Path, default=application_data_dir() / "research.db")
    parser.add_argument("--overlay-root", type=Path,
                        default=application_data_dir() / "cache" / "market_overlay")
    args = parser.parse_args(argv)
    repository = SQLiteRepository(args.database, project_root() / "migrations")
    repository.initialize()
    if args.status:
        from ashare_lab.services.recommendation_reference import reference_summary
        from ashare_lab.services.recommendation_tracking import (
            ensure_tracking_schema,
            tracking_summary,
        )
        with repository.connection() as connection:
            ensure_tracking_schema(connection)
            result = tracking_summary(connection)
            result["reference_summary"] = reference_summary(connection)
    else:
        result = run_recommendation_tracking(
            repository, overlay_store=MarketOverlayStore(args.overlay_root), through=args.through,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
