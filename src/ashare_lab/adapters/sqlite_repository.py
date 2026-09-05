"""SQLite persistence for immutable research predictions and later outcomes.

This database is an audit log, not a LangGraph checkpoint.  Completed research
runs are inserted atomically and never edited.  When the forecast horizon has
elapsed, observed returns are written to the separate ``outcomes`` table.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from math import isclose
from pathlib import Path
from typing import Any

JsonRecord = Mapping[str, Any] | Any

_RECOMMENDATION_HORIZONS = {
    5: ("1w", 1),
    10: ("2w", 2),
    20: ("1m", 4),
    60: ("3m", 13),
    120: ("6m", 26),
    252: ("1y", 52),
}


class SQLiteRepository:
    """Small, dependency-free repository for a single-user local application."""

    def __init__(
        self,
        db_path: str | Path,
        migrations_dir: str | Path | None = None,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.migrations_dir = (
            Path(migrations_dir).expanduser().resolve()
            if migrations_dir is not None
            else Path(__file__).resolve().parents[3] / "migrations"
        )
        self.timeout_seconds = timeout_seconds

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Open a configured connection and close it deterministically."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.db_path,
            timeout=self.timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """Apply all SQL migrations in filename order; safe to call repeatedly."""

        migration_files = sorted(self.migrations_dir.glob("[0-9][0-9][0-9]_*.sql"))
        if not migration_files:
            raise FileNotFoundError(f"No migrations found in {self.migrations_dir}")

        with self.connection() as connection:
            for migration_file in migration_files:
                connection.executescript(migration_file.read_text(encoding="utf-8"))

    def archive_run(
        self,
        run: JsonRecord,
        *,
        data_snapshots: Iterable[JsonRecord] = (),
        analyses: Iterable[JsonRecord] = (),
        scenarios: Iterable[JsonRecord] = (),
        portfolio_sets: Iterable[JsonRecord] = (),
        portfolio_members: Iterable[JsonRecord] = (),
        evidence: Iterable[JsonRecord] = (),
    ) -> None:
        """Insert one complete prediction bundle in a single transaction.

        All identifiers are supplied by the application service.  This makes
        relationships explicit and lets a caller hash/sign a bundle before it
        reaches persistence.  Any invalid child row rolls back the whole run.
        """

        self.initialize()
        run_row = _record(run)
        collections = {
            "data_snapshots": tuple(_record(item) for item in data_snapshots),
            "stock_analyses": tuple(_record(item) for item in analyses),
            "scenarios": tuple(_record(item) for item in scenarios),
            "portfolio_sets": tuple(_record(item) for item in portfolio_sets),
            "portfolio_members": tuple(_record(item) for item in portfolio_members),
            "evidence": tuple(_record(item) for item in evidence),
        }

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._insert_run(connection, run_row)
                for table, rows in collections.items():
                    for row in rows:
                        self._insert_child(connection, table, row)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        self.initialize()
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return _decode_row(row, json_columns={"warning_json"})

    def list_runs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return the newest immutable research runs for the local review page."""

        if limit <= 0 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        self.initialize()
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_decode_row(row, json_columns={"warning_json"}) for row in rows]

    def list_stock_analyses(self, run_id: str) -> list[dict[str, Any]]:
        self.initialize()
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM stock_analyses
                WHERE run_id = ?
                ORDER BY symbol, horizon_sessions
                """,
                (run_id,),
            ).fetchall()
        return [_decode_row(row, json_columns={"rationale_json"}) for row in rows]

    def list_scenarios(self, analysis_id: str) -> list[dict[str, Any]]:
        self.initialize()
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM scenarios
                WHERE analysis_id = ?
                ORDER BY CASE label WHEN 'up' THEN 1 WHEN 'sideways' THEN 2 ELSE 3 END
                """,
                (analysis_id,),
            ).fetchall()
        return [_decode_row(row) for row in rows]

    def archive_recommendation_report(
        self,
        report: JsonRecord,
        *,
        batches: Iterable[JsonRecord] = (),
        members: Iterable[JsonRecord] = (),
        delivery_events: Iterable[JsonRecord] = (),
    ) -> None:
        """Atomically archive one immutable multi-horizon recommendation report.

        Recommendation rows describe what the program said at the time.  They
        are deliberately separate from mutable maturity observations, so a
        later return calculation can never rewrite the original cohort.
        """

        self.initialize()
        report_row = dict(_record(report))
        report_id = str(_required(report_row, "id"))
        archive_nature = str(_required(report_row, "archive_nature"))
        raw_batches = tuple(dict(_record(item)) for item in batches)
        raw_members = tuple(dict(_record(item)) for item in members)
        raw_events = tuple(dict(_record(item)) for item in delivery_events)

        batch_ids = [str(_required(row, "id")) for row in raw_batches]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("Recommendation batch identifiers must be unique")
        member_ids = [str(_required(row, "id")) for row in raw_members]
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("Recommendation member identifiers must be unique")

        member_counts = dict.fromkeys(batch_ids, 0)
        for row in raw_members:
            batch_id = str(_required(row, "batch_id"))
            if batch_id not in member_counts:
                raise ValueError(f"Recommendation member references unknown batch: {batch_id}")
            member_counts[batch_id] += 1

        batch_rows: list[dict[str, Any]] = []
        for row in raw_batches:
            if str(_required(row, "report_id")) != report_id:
                raise ValueError("Every recommendation batch must belong to the archived report")
            batch_id = str(_required(row, "id"))
            normalized = _normalize_recommendation_batch(
                row,
                archive_nature=archive_nature,
                actual_member_count=member_counts[batch_id],
            )
            declared_count = normalized["member_count"]
            if declared_count != member_counts[batch_id]:
                raise ValueError(
                    f"Batch {batch_id} declares {declared_count} members but archives "
                    f"{member_counts[batch_id]}"
                )
            batch_rows.append(normalized)

        batch_by_id = {row["id"]: row for row in batch_rows}
        member_rows = [
            _normalize_recommendation_member(row, batch=batch_by_id[str(row["batch_id"])])
            for row in raw_members
        ]
        event_rows = []
        for row in raw_events:
            normalized = _normalize_recommendation_delivery_event(row)
            if normalized["report_id"] != report_id:
                raise ValueError("Every delivery event must belong to the archived report")
            event_batch_id = normalized["batch_id"]
            if event_batch_id is not None and event_batch_id not in batch_by_id:
                raise ValueError("A delivery event cannot reference a batch from another report")
            event_rows.append(normalized)

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT content_hash FROM recommendation_reports WHERE id = ?",
                    (report_id,),
                ).fetchone()
                if existing is not None:
                    if existing["content_hash"] != _required(report_row, "content_hash"):
                        raise ValueError(
                            "An archived recommendation report id cannot be reused "
                            "for different content"
                        )
                    stored_batch_ids = {
                        item["id"]
                        for item in connection.execute(
                            "SELECT id FROM recommendation_batches WHERE report_id = ?",
                            (report_id,),
                        ).fetchall()
                    }
                    stored_member_ids = {
                        item["id"]
                        for item in connection.execute(
                            """
                            SELECT m.id FROM recommendation_members m
                            JOIN recommendation_batches b ON b.id = m.batch_id
                            WHERE b.report_id = ?
                            """,
                            (report_id,),
                        ).fetchall()
                    }
                    if stored_batch_ids != set(batch_ids) or stored_member_ids != set(member_ids):
                        raise ValueError(
                            "Idempotent recommendation archive retry does not match "
                            "the stored bundle"
                        )
                    stored_event_ids = {
                        item["id"]
                        for item in connection.execute(
                            """
                            SELECT id FROM recommendation_delivery_events
                            WHERE report_id = ?
                            """,
                            (report_id,),
                        ).fetchall()
                    }
                    for row in event_rows:
                        if row["id"] not in stored_event_ids:
                            self._insert_recommendation_delivery_event(connection, row)
                    connection.commit()
                    return
                self._insert_recommendation_report(connection, report_row)
                for row in batch_rows:
                    self._insert_recommendation_batch(connection, row)
                for row in member_rows:
                    self._insert_recommendation_member(connection, row)
                for row in event_rows:
                    self._insert_recommendation_delivery_event(connection, row)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def get_recommendation_report(self, report_id: str) -> dict[str, Any] | None:
        self.initialize()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM recommendation_reports WHERE id = ?", (report_id,)
            ).fetchone()
        return _decode_row(row, json_columns={"metadata_json"})

    def get_recommendation_report_by_content_hash(self, content_hash: str) -> dict[str, Any] | None:
        self.initialize()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM recommendation_reports WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        return _decode_row(row, json_columns={"metadata_json"})

    def list_recommendation_reports(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if limit <= 0 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        self.initialize()
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM recommendation_reports
                ORDER BY plan_for_date DESC, created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_decode_row(row, json_columns={"metadata_json"}) for row in rows]

    def get_recommendation_batch(self, batch_id: str) -> dict[str, Any] | None:
        self.initialize()
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT b.*, r.decision_date, r.common_cutoff, r.plan_for_date,
                       r.archive_nature,
                       EXISTS (
                           SELECT 1 FROM recommendation_delivery_events d
                           WHERE d.report_id = r.id
                             AND d.provider_status IN ('provider_accepted', 'accepted')
                       ) AS delivery_accepted
                FROM recommendation_batches b
                JOIN recommendation_reports r ON r.id = b.report_id
                WHERE b.id = ?
                """,
                (batch_id,),
            ).fetchone()
        return _decode_row(row, json_columns={"metadata_json"})

    def list_recommendation_batches(self, report_id: str) -> list[dict[str, Any]]:
        self.initialize()
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT b.*, r.decision_date, r.common_cutoff, r.plan_for_date,
                       r.archive_nature,
                       EXISTS (
                           SELECT 1 FROM recommendation_delivery_events d
                           WHERE d.report_id = r.id
                             AND d.provider_status IN ('provider_accepted', 'accepted')
                       ) AS delivery_accepted
                FROM recommendation_batches b
                JOIN recommendation_reports r ON r.id = b.report_id
                WHERE b.report_id = ?
                ORDER BY b.holding_sessions
                """,
                (report_id,),
            ).fetchall()
        return [_decode_row(row, json_columns={"metadata_json"}) for row in rows]

    def list_recommendation_batches_pending_settlement(
        self,
        *,
        as_of: date | str | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidate cohorts; the caller still verifies trading sessions.

        A calendar date alone cannot prove that 5/10/20/60/120/252 market
        sessions have elapsed.  This method only bounds cohorts by plan date;
        the settlement service must count verified exchange sessions.
        """

        self.initialize()
        parameters: tuple[Any, ...] = ()
        date_clause = ""
        if as_of is not None:
            date_clause = "AND r.plan_for_date <= ?"
            parameters = (_iso_text(as_of),)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT b.*, r.decision_date, r.common_cutoff, r.plan_for_date,
                       r.archive_nature,
                       br.status AS result_status,
                       EXISTS (
                           SELECT 1 FROM recommendation_delivery_events d
                           WHERE d.report_id = r.id
                             AND d.provider_status IN ('provider_accepted', 'accepted')
                       ) AS delivery_accepted
                FROM recommendation_batches b
                JOIN recommendation_reports r ON r.id = b.report_id
                LEFT JOIN recommendation_batch_results br ON br.batch_id = b.id
                WHERE b.status = 'pending'
                  AND (br.id IS NULL OR br.status IN ('pending', 'needs_review'))
                  {date_clause}
                ORDER BY r.plan_for_date, b.holding_sessions
                """,
                parameters,
            ).fetchall()
        return [_decode_row(row, json_columns={"metadata_json"}) for row in rows]

    def list_recommendation_members(self, batch_id: str) -> list[dict[str, Any]]:
        self.initialize()
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM recommendation_members
                WHERE batch_id = ?
                ORDER BY rank
                """,
                (batch_id,),
            ).fetchall()
        return [
            _member_result_aliases(
                _decode_row(row, json_columns={"entry_plan_json", "metadata_json"})
            )
            for row in rows
        ]

    def record_recommendation_delivery_event(self, event: JsonRecord) -> None:
        """Append one immutable provider submission/acceptance audit event."""

        self.initialize()
        row = _normalize_recommendation_delivery_event(dict(_record(event)))
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if row["batch_id"] is not None:
                    owner = connection.execute(
                        "SELECT report_id FROM recommendation_batches WHERE id = ?",
                        (row["batch_id"],),
                    ).fetchone()
                    if owner is None or owner["report_id"] != row["report_id"]:
                        raise ValueError("Delivery event batch does not belong to its report")
                self._insert_recommendation_delivery_event(connection, row)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def list_recommendation_delivery_events(self, report_id: str) -> list[dict[str, Any]]:
        self.initialize()
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM recommendation_delivery_events
                WHERE report_id = ?
                ORDER BY attempted_at, id
                """,
                (report_id,),
            ).fetchall()
        return [_decode_row(row, json_columns={"detail_json"}) for row in rows]

    def record_recommendation_member_result(self, result: JsonRecord) -> None:
        """Idempotently insert or refresh one member's maturity observation."""

        self.initialize()
        row = dict(_record(result))
        member_id = str(_required(row, "member_id"))
        evaluated_at = _iso_text(row.get("evaluated_at") or datetime.now(UTC))
        updated_at = _iso_text(row.get("updated_at") or datetime.now(UTC))
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO recommendation_member_results (
                    id, member_id, evaluated_at, status, entry_date, entry_price,
                    maturity_date, maturity_close, realized_return,
                    holding_sessions_observed, max_drawdown, max_runup,
                    reason_code, company_action_clear, data_cutoff,
                    method_version, details_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(member_id) DO UPDATE SET
                    evaluated_at = excluded.evaluated_at,
                    status = excluded.status,
                    entry_date = excluded.entry_date,
                    entry_price = excluded.entry_price,
                    maturity_date = excluded.maturity_date,
                    maturity_close = excluded.maturity_close,
                    realized_return = excluded.realized_return,
                    holding_sessions_observed = excluded.holding_sessions_observed,
                    max_drawdown = excluded.max_drawdown,
                    max_runup = excluded.max_runup,
                    reason_code = excluded.reason_code,
                    company_action_clear = excluded.company_action_clear,
                    data_cutoff = excluded.data_cutoff,
                    method_version = excluded.method_version,
                    details_json = excluded.details_json,
                    updated_at = excluded.updated_at
                """,
                (
                    row.get("id") or f"recommendation-member-result:{member_id}",
                    member_id,
                    evaluated_at,
                    _normalize_member_result_status(_required(row, "status")),
                    _iso_optional(row.get("entry_date")),
                    row.get("entry_price"),
                    _iso_optional(_first_present(row, "maturity_date", "due_date")),
                    _first_present(row, "maturity_close", "due_close"),
                    _first_present(
                        row,
                        "realized_return",
                        "unadjusted_price_return",
                        "return_pct",
                    ),
                    row.get("holding_sessions_observed"),
                    row.get("max_drawdown"),
                    row.get("max_runup"),
                    row.get("reason_code"),
                    _optional_bool_int(row.get("company_action_clear")),
                    _iso_optional(row.get("data_cutoff")),
                    _required(row, "method_version"),
                    _json_text(_first_present(row, "details_json", "detail_json", default={})),
                    updated_at,
                ),
            )

    def record_recommendation_settlement(
        self,
        *,
        batch_result: JsonRecord,
        member_results: Iterable[JsonRecord],
    ) -> None:
        """Commit every member, its aggregate and an immutable revision together."""

        self.initialize()
        batch = dict(_record(batch_result))
        members = tuple(dict(_record(item)) for item in member_results)
        batch_id = str(_required(batch, "batch_id"))
        supplied_ids = [str(_required(item, "member_id")) for item in members]
        if len(supplied_ids) != len(set(supplied_ids)):
            raise ValueError("settlement member ids must be unique")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                expected_ids = {
                    str(row["id"])
                    for row in connection.execute(
                        "SELECT id FROM recommendation_members WHERE batch_id = ?", (batch_id,)
                    )
                }
                if not expected_ids or set(supplied_ids) != expected_ids:
                    raise ValueError("atomic settlement must cover exactly its archived members")
                prior = connection.execute(
                    "SELECT * FROM recommendation_batch_results WHERE batch_id = ?", (batch_id,)
                ).fetchone()
                recorded = connection.execute(
                    "SELECT 1 FROM recommendation_settlement_history WHERE batch_id = ? LIMIT 1",
                    (batch_id,),
                ).fetchone()
                if prior is not None and recorded is None:
                    # Capture pre-migration observations before their first refresh.
                    prior_members = connection.execute(
                        """SELECT mr.* FROM recommendation_member_results mr
                           JOIN recommendation_members m ON m.id = mr.member_id
                           WHERE m.batch_id = ? ORDER BY m.rank""",
                        (batch_id,),
                    ).fetchall()
                    prior_snapshot = _json_text(
                        {"batch": dict(prior), "members": [dict(row) for row in prior_members]}
                    )
                    connection.execute(
                        """INSERT OR IGNORE INTO recommendation_settlement_history
                           (id, batch_id, evaluated_at, method_version, snapshot_json)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            f"settlement-legacy:{hashlib.sha256(prior_snapshot.encode()).hexdigest()}",
                            batch_id,
                            prior["evaluated_at"],
                            prior["method_version"],
                            prior_snapshot,
                        ),
                    )
                # Reuse the exact adapters while binding every write to this transaction.
                writer = _SettlementTransactionRepository(connection)
                for member in members:
                    writer.record_recommendation_member_result(member)
                writer.record_recommendation_batch_result(batch)
                snapshot = {
                    "batch": batch,
                    "members": sorted(members, key=lambda r: r["member_id"]),
                }
                canonical = {
                    "batch": {
                        k: v for k, v in batch.items() if k not in {"evaluated_at", "updated_at"}
                    },
                    "members": [
                        {k: v for k, v in row.items() if k not in {"evaluated_at", "updated_at"}}
                        for row in snapshot["members"]
                    ],
                }
                digest = hashlib.sha256(_json_text(canonical).encode()).hexdigest()
                connection.execute(
                    """INSERT OR IGNORE INTO recommendation_settlement_history
                       (id, batch_id, evaluated_at, method_version, snapshot_json)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        f"settlement:{digest}",
                        batch_id,
                        _iso_text(batch.get("evaluated_at") or datetime.now(UTC)),
                        _required(batch, "method_version"),
                        _json_text(snapshot),
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def get_recommendation_member_result(self, member_id: str) -> dict[str, Any] | None:
        self.initialize()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM recommendation_member_results WHERE member_id = ?",
                (member_id,),
            ).fetchone()
        return _decode_row(row, json_columns={"details_json"})

    def list_recommendation_member_results(self, batch_id: str) -> list[dict[str, Any]]:
        self.initialize()
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT mr.* FROM recommendation_member_results mr
                JOIN recommendation_members m ON m.id = mr.member_id
                WHERE m.batch_id = ?
                ORDER BY m.rank
                """,
                (batch_id,),
            ).fetchall()
        return [_decode_row(row, json_columns={"details_json"}) for row in rows]

    def record_recommendation_batch_result(self, result: JsonRecord) -> None:
        """Idempotently insert or refresh one horizon-level maturity result."""

        self.initialize()
        row = dict(_record(result))
        batch_id = str(_required(row, "batch_id"))
        evaluated_at = _iso_text(row.get("evaluated_at") or datetime.now(UTC))
        updated_at = _iso_text(row.get("updated_at") or datetime.now(UTC))
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO recommendation_batch_results (
                    id, batch_id, evaluated_at, status, maturity_date,
                    stock_sleeve_return, account_return,
                    entered_stock_sleeve_weight, entered_account_weight,
                    cash_weight, resolved_member_count, total_member_count,
                    reason_code, data_cutoff, method_version, details_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    evaluated_at = excluded.evaluated_at,
                    status = excluded.status,
                    maturity_date = excluded.maturity_date,
                    stock_sleeve_return = excluded.stock_sleeve_return,
                    account_return = excluded.account_return,
                    entered_stock_sleeve_weight = excluded.entered_stock_sleeve_weight,
                    entered_account_weight = excluded.entered_account_weight,
                    cash_weight = excluded.cash_weight,
                    resolved_member_count = excluded.resolved_member_count,
                    total_member_count = excluded.total_member_count,
                    reason_code = excluded.reason_code,
                    data_cutoff = excluded.data_cutoff,
                    method_version = excluded.method_version,
                    details_json = excluded.details_json,
                    updated_at = excluded.updated_at
                """,
                (
                    row.get("id") or f"recommendation-batch-result:{batch_id}",
                    batch_id,
                    evaluated_at,
                    _normalize_batch_result_status(_required(row, "status")),
                    _iso_optional(_first_present(row, "maturity_date", "due_date")),
                    row.get("stock_sleeve_return"),
                    row.get("account_return"),
                    _first_present(
                        row,
                        "entered_stock_sleeve_weight",
                        "entered_weight",
                    ),
                    row.get("entered_account_weight"),
                    _first_present(row, "cash_weight", "stock_sleeve_cash_weight"),
                    row.get("resolved_member_count"),
                    row.get("total_member_count"),
                    row.get("reason_code"),
                    _iso_optional(row.get("data_cutoff")),
                    _required(row, "method_version"),
                    _json_text(_first_present(row, "details_json", "detail_json", default={})),
                    updated_at,
                ),
            )

    def get_recommendation_batch_result(self, batch_id: str) -> dict[str, Any] | None:
        self.initialize()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM recommendation_batch_results WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
        return _decode_row(row, json_columns={"details_json"})

    def get_recommendation_batch_performance(
        self,
        batch_id: str,
    ) -> dict[str, Any] | None:
        """Return one archived cohort, its result, and rank-ordered members."""

        batch = self.get_recommendation_batch(batch_id)
        if batch is None:
            return None
        result = self.get_recommendation_batch_result(batch_id)
        members = self.list_recommendation_members(batch_id)
        result_by_member_id = {
            item["member_id"]: item for item in self.list_recommendation_member_results(batch_id)
        }
        return {
            "batch": batch,
            "result": result,
            "members": [
                {
                    "recommendation": member,
                    "result": result_by_member_id.get(member["id"]),
                }
                for member in members
            ],
        }

    def list_maturity_results_pending_notification(
        self,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return mature cohorts not yet accepted by a notification provider."""

        if limit <= 0 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        self.initialize()
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT br.*, b.report_id, b.horizon_key, b.holding_weeks,
                       b.holding_sessions, b.label, b.evaluation_mode,
                       b.cohort_nature, b.stock_exposure, b.cash_weight,
                       r.decision_date, r.common_cutoff, r.plan_for_date,
                       r.archive_nature
                FROM recommendation_batch_results br
                JOIN recommendation_batches b ON b.id = br.batch_id
                JOIN recommendation_reports r ON r.id = b.report_id
                WHERE br.status IN ('resolved', 'no_entries', 'partial', 'needs_review')
                  AND br.maturity_date IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM recommendation_delivery_events d
                      WHERE d.report_id = r.id
                        AND d.batch_id = b.id
                        AND d.delivery_kind = 'maturity_provider_accepted'
                        AND (
                            (
                                json_extract(d.detail_json, '$.result_status') = br.status
                                AND json_extract(
                                    d.detail_json,
                                    '$.result_method_version'
                                ) = br.method_version
                            )
                            OR (
                                json_extract(d.detail_json, '$.result_status') IS NULL
                                AND datetime(d.attempted_at) >= datetime(br.updated_at)
                            )
                        )
                  )
                ORDER BY br.evaluated_at, b.holding_sessions
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_decode_row(row, json_columns={"details_json"}) for row in rows]

    def record_outcome(self, outcome: JsonRecord) -> None:
        """Insert or refresh an observation without altering its prediction."""

        self.initialize()
        row = _record(outcome)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO outcomes (
                        id, analysis_id, observed_at, realized_return,
                        max_drawdown, max_runup, relative_return, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(analysis_id, observed_at) DO UPDATE SET
                        realized_return = excluded.realized_return,
                        max_drawdown = excluded.max_drawdown,
                        max_runup = excluded.max_runup,
                        relative_return = excluded.relative_return,
                        status = excluded.status
                    """,
                    (
                        _required(row, "id"),
                        _required(row, "analysis_id"),
                        _iso_text(_required(row, "observed_at")),
                        row.get("realized_return"),
                        row.get("max_drawdown"),
                        row.get("max_runup"),
                        row.get("relative_return"),
                        _required(row, "status"),
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def get_outcomes(self, analysis_id: str) -> list[dict[str, Any]]:
        self.initialize()
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM outcomes WHERE analysis_id = ? ORDER BY observed_at",
                (analysis_id,),
            ).fetchall()
        return [_decode_row(row) for row in rows]

    def save_position(self, position: JsonRecord) -> None:
        """Upsert the user's current local position; positions are mutable state."""

        self.initialize()
        row = _record(position)
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO positions (id, symbol, shares, cost_price, as_of)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    id = excluded.id,
                    shares = excluded.shares,
                    cost_price = excluded.cost_price,
                    as_of = excluded.as_of
                """,
                (
                    _required(row, "id"),
                    _required(row, "symbol"),
                    _required(row, "shares"),
                    _required(row, "cost_price"),
                    _iso_text(_required(row, "as_of")),
                ),
            )

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        self.initialize()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM positions WHERE symbol = ?", (symbol,)
            ).fetchone()
        return _decode_row(row)

    def archive_holding_snapshot(
        self,
        revision: JsonRecord,
        *,
        positions: Iterable[JsonRecord] = (),
        expected_current_revision_id: str | None = None,
        expected_current_version: int | None = None,
    ) -> None:
        """Append one explicit user-confirmed holding snapshot atomically.

        Reviews never call this method.  Consequently, the newest revision
        remains the user's current holding statement until another explicit
        snapshot is archived.
        """

        if (expected_current_revision_id is None) != (expected_current_version is None):
            raise ValueError("Holding snapshot CAS requires both current id and version")
        if expected_current_version is not None and expected_current_version <= 0:
            raise ValueError("Expected holding snapshot version must be positive")
        self.initialize()
        revision_row = dict(_record(revision))
        revision_id = str(_required(revision_row, "id"))
        version = int(_required(revision_row, "version"))
        status = str(_required(revision_row, "status"))
        raw_positions = tuple(dict(_record(item)) for item in positions)
        symbols = [str(_required(row, "symbol")) for row in raw_positions]
        keys = [str(_required(row, "position_key")) for row in raw_positions]
        if len(symbols) != len(set(symbols)):
            raise ValueError("A holding snapshot cannot contain a duplicate symbol")
        if len(keys) != len(set(keys)):
            raise ValueError("A holding snapshot cannot contain a duplicate position key")
        if status == "active" and not raw_positions:
            raise ValueError("An active holding snapshot must contain at least one position")
        if status == "cleared" and raw_positions:
            raise ValueError("A cleared holding snapshot cannot contain positions")
        if status not in {"active", "cleared"}:
            raise ValueError("Holding snapshot status must be active or cleared")

        for row in raw_positions:
            if str(_required(row, "revision_id")) != revision_id:
                raise ValueError("Every holding position must belong to its snapshot")
            if int(_required(row, "version")) != version:
                raise ValueError("Holding position version must match its snapshot")
            if str(_required(row, "status")) != "active":
                raise ValueError("A current holding snapshot may contain only active positions")
        if raw_positions:
            stock_sleeve = sum(
                float(_required(row, "stock_sleeve_weight")) for row in raw_positions
            )
            if not isclose(stock_sleeve, 1.0, abs_tol=1e-9):
                raise ValueError("Active stock-sleeve weights must sum to 1.0")
            account_weights = [row.get("account_weight") for row in raw_positions]
            if sum(float(value) for value in account_weights if value is not None) > 1.0 + 1e-9:
                raise ValueError("Holding account weights cannot exceed 1.0")

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT version FROM holding_portfolio_revisions WHERE id = ?",
                    (revision_id,),
                ).fetchone()
                latest = connection.execute(
                    """
                    SELECT id, version FROM holding_portfolio_revisions
                    ORDER BY version DESC LIMIT 1
                    """
                ).fetchone()
                if existing is not None:
                    if int(existing["version"]) != version:
                        raise ValueError("A holding revision id cannot be reused")
                    stored = {
                        row["id"]
                        for row in connection.execute(
                            "SELECT id FROM holding_positions WHERE revision_id = ?",
                            (revision_id,),
                        ).fetchall()
                    }
                    supplied = {str(_required(row, "id")) for row in raw_positions}
                    if stored != supplied:
                        raise ValueError("Idempotent holding snapshot retry does not match")
                    if expected_current_revision_id is not None and (
                        latest is None
                        or str(latest["id"]) != revision_id
                        or int(latest["version"]) != version
                    ):
                        raise ValueError("Current holding snapshot changed; reload before saving")
                    connection.commit()
                    return
                if expected_current_revision_id is not None and (
                    latest is None
                    or str(latest["id"]) != expected_current_revision_id
                    or int(latest["version"]) != expected_current_version
                ):
                    raise ValueError("Current holding snapshot changed; reload before saving")
                expected_version = (0 if latest is None else int(latest["version"])) + 1
                if version != expected_version:
                    raise ValueError(
                        f"Holding snapshot version must be {expected_version}, got {version}"
                    )
                connection.execute(
                    """
                    INSERT INTO holding_portfolio_revisions (
                        id, version, holding_weeks, effective_at, source,
                        status, method_version, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        revision_id,
                        version,
                        _required(revision_row, "holding_weeks"),
                        _iso_text(_required(revision_row, "effective_at")),
                        _required(revision_row, "source"),
                        status,
                        _required(revision_row, "method_version"),
                        _json_text(revision_row.get("metadata_json", {})),
                    ),
                )
                for row in raw_positions:
                    connection.execute(
                        """
                        INSERT INTO holding_positions (
                            id, revision_id, position_key, symbol, name,
                            entry_date, cost_price, stock_sleeve_weight,
                            account_weight, status, source, version, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            _required(row, "id"),
                            revision_id,
                            _required(row, "position_key"),
                            _required(row, "symbol"),
                            _required(row, "name"),
                            _iso_text(_required(row, "entry_date")),
                            row.get("cost_price"),
                            _required(row, "stock_sleeve_weight"),
                            row.get("account_weight"),
                            "active",
                            _required(row, "source"),
                            version,
                            _json_text(row.get("metadata_json", {})),
                        ),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def next_holding_snapshot_version(self) -> int:
        self.initialize()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 AS version FROM holding_portfolio_revisions"
            ).fetchone()
        return int(row["version"])

    def get_current_holding_snapshot(
        self,
        *,
        as_of: date | None = None,
    ) -> dict[str, Any] | None:
        """Return the newest explicit snapshot knowable by ``as_of``.

        With no cutoff this preserves the live/current read path.  Historical
        holding reviews pass their completed-close date so a later explicit
        user statement cannot leak into an earlier replay.
        """

        self.initialize()
        if as_of is not None and (not isinstance(as_of, date) or isinstance(as_of, datetime)):
            raise TypeError("holding snapshot as_of must be a date")
        with self.connection() as connection:
            if as_of is None:
                revision = connection.execute(
                    """
                    SELECT * FROM holding_portfolio_revisions
                    ORDER BY version DESC LIMIT 1
                    """
                ).fetchone()
            else:
                revision = connection.execute(
                    """
                    SELECT * FROM holding_portfolio_revisions
                    WHERE substr(effective_at, 1, 10) <= ?
                    ORDER BY version DESC LIMIT 1
                    """,
                    (as_of.isoformat(),),
                ).fetchone()
            if revision is None:
                return None
            positions = connection.execute(
                """
                SELECT * FROM holding_positions
                WHERE revision_id = ? ORDER BY symbol
                """,
                (revision["id"],),
            ).fetchall()
        return {
            "revision": _decode_row(revision, json_columns={"metadata_json"}),
            "positions": [_decode_row(row, json_columns={"metadata_json"}) for row in positions],
        }

    def get_holding_snapshot(self, revision_id: str) -> dict[str, Any] | None:
        """Return one explicit revision, including a superseded revision."""

        self.initialize()
        with self.connection() as connection:
            revision = connection.execute(
                "SELECT * FROM holding_portfolio_revisions WHERE id = ?",
                (revision_id,),
            ).fetchone()
            if revision is None:
                return None
            positions = connection.execute(
                """
                SELECT * FROM holding_positions
                WHERE revision_id = ? ORDER BY symbol
                """,
                (revision_id,),
            ).fetchall()
        return {
            "revision": _decode_row(revision, json_columns={"metadata_json"}),
            "positions": [_decode_row(row, json_columns={"metadata_json"}) for row in positions],
        }

    def list_active_holdings(self) -> list[dict[str, Any]]:
        snapshot = self.get_current_holding_snapshot()
        if snapshot is None or snapshot["revision"]["status"] != "active":
            return []
        return [row for row in snapshot["positions"] if row["status"] == "active"]

    def get_holding_protective_stop(self, position_key: str) -> dict[str, Any] | None:
        self.initialize()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM holding_protective_stops WHERE position_key = ?",
                (position_key,),
            ).fetchone()
        return _decode_row(row, json_columns={"details_json"})

    def record_holding_review(
        self,
        review: JsonRecord,
        *,
        stop_state: JsonRecord | None = None,
    ) -> None:
        """Atomically append a daily review and ratchet its remembered stop."""

        self.initialize()
        review_row = dict(_record(review))
        stop_row = None if stop_state is None else dict(_record(stop_state))
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                owner = connection.execute(
                    """
                    SELECT p.revision_id, p.position_key
                    FROM holding_positions p WHERE p.id = ?
                    """,
                    (_required(review_row, "position_id"),),
                ).fetchone()
                if owner is None:
                    raise ValueError("Holding review references an unknown position")
                if owner["revision_id"] != _required(review_row, "revision_id"):
                    raise ValueError("Holding review position belongs to another snapshot")
                if owner["position_key"] != _required(review_row, "position_key"):
                    raise ValueError("Holding review position key mismatch")

                if stop_row is not None:
                    if stop_row.get("position_key") != review_row.get("position_key"):
                        raise ValueError("Holding stop and review must reference the same position")
                    existing = connection.execute(
                        """
                        SELECT effective_stop, data_cutoff
                        FROM holding_protective_stops WHERE position_key = ?
                        """,
                        (stop_row["position_key"],),
                    ).fetchone()
                    proposed_cutoff = _iso_text(_required(stop_row, "data_cutoff"))
                    if existing is not None and proposed_cutoff < str(existing["data_cutoff"]):
                        raise ValueError("Holding protection data cutoff cannot move backwards")
                    if (
                        existing is not None
                        and float(_required(stop_row, "effective_stop"))
                        < float(existing["effective_stop"]) - 1e-9
                    ):
                        raise ValueError("Effective holding protection stop cannot move down")
                    connection.execute(
                        """
                        INSERT INTO holding_protective_stops (
                            position_key, symbol, entry_date, effective_stop,
                            candidate_stop, previous_stop, data_cutoff,
                            source_timeframe, evidence_date, holding_version,
                            method_version, details_json, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(position_key) DO UPDATE SET
                            effective_stop = excluded.effective_stop,
                            candidate_stop = excluded.candidate_stop,
                            previous_stop = excluded.previous_stop,
                            data_cutoff = excluded.data_cutoff,
                            source_timeframe = excluded.source_timeframe,
                            evidence_date = excluded.evidence_date,
                            holding_version = excluded.holding_version,
                            method_version = excluded.method_version,
                            details_json = excluded.details_json,
                            updated_at = excluded.updated_at
                        """,
                        (
                            stop_row["position_key"],
                            _required(stop_row, "symbol"),
                            _iso_text(_required(stop_row, "entry_date")),
                            _required(stop_row, "effective_stop"),
                            _required(stop_row, "candidate_stop"),
                            stop_row.get("previous_stop"),
                            proposed_cutoff,
                            _required(stop_row, "source_timeframe"),
                            _iso_text(_required(stop_row, "evidence_date")),
                            _required(stop_row, "holding_version"),
                            _required(stop_row, "method_version"),
                            _json_text(stop_row.get("details_json", {})),
                            _iso_text(stop_row.get("updated_at") or datetime.now(UTC)),
                        ),
                    )

                connection.execute(
                    """
                    INSERT INTO holding_review_events (
                        id, revision_id, position_id, position_key, symbol,
                        name, holding_weeks, holding_version, reviewed_at,
                        data_cutoff, status, holding_action, latest_close,
                        candidate_stop, previous_stop, effective_stop,
                        close_below_stop, source_timeframe, evidence_date,
                        company_action_clear, company_action_evidence_id,
                        company_action_evidence_source, company_action_clear_through,
                        decision_layer, candidate_rank_used, next_session_only,
                        auto_order_allowed, reason_json, evidence_hash,
                        method_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(
                        position_key, holding_version, data_cutoff,
                        method_version, evidence_hash
                    )
                    DO NOTHING
                    """,
                    (
                        _required(review_row, "id"),
                        _required(review_row, "revision_id"),
                        _required(review_row, "position_id"),
                        _required(review_row, "position_key"),
                        _required(review_row, "symbol"),
                        _required(review_row, "name"),
                        _required(review_row, "holding_weeks"),
                        _required(review_row, "holding_version"),
                        _iso_text(_required(review_row, "reviewed_at")),
                        _iso_text(_required(review_row, "data_cutoff")),
                        _required(review_row, "status"),
                        _required(review_row, "holding_action"),
                        review_row.get("latest_close"),
                        review_row.get("candidate_stop"),
                        review_row.get("previous_stop"),
                        review_row.get("effective_stop"),
                        _optional_bool_int(review_row.get("close_below_stop")),
                        review_row.get("source_timeframe"),
                        _iso_optional(review_row.get("evidence_date")),
                        _optional_bool_int(review_row.get("company_action_clear")),
                        review_row.get("company_action_evidence_id"),
                        review_row.get("company_action_evidence_source"),
                        _iso_optional(review_row.get("company_action_clear_through")),
                        "holding_management",
                        0,
                        1,
                        0,
                        _json_text(review_row.get("reason_json", [])),
                        _required(review_row, "evidence_hash"),
                        _required(review_row, "method_version"),
                        _iso_text(review_row.get("created_at") or datetime.now(UTC)),
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def list_holding_reviews(
        self,
        *,
        data_cutoff: date | str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit <= 0 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        self.initialize()
        clause = ""
        parameters: tuple[Any, ...]
        if data_cutoff is None:
            parameters = (limit,)
        else:
            clause = "WHERE data_cutoff = ?"
            parameters = (_iso_text(data_cutoff), limit)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM holding_review_events
                {clause}
                ORDER BY data_cutoff DESC, symbol
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [_decode_row(row, json_columns={"reason_json"}) for row in rows]

    def archive_holding_shadow_events(
        self,
        events: Iterable[JsonRecord],
    ) -> int:
        """Append local-only stop experiments without touching production state.

        The isolation flags are literals supplied by the repository rather than
        trusted caller input.  An idempotent retry of the same evidence is a
        no-op; a reused primary identifier with different evidence still fails.
        """

        self.initialize()
        rows = tuple(dict(_record(item)) for item in events)
        if not rows:
            return 0
        inserted = 0
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for row in rows:
                    owner = connection.execute(
                        """
                        SELECT revision_id, position_key, symbol, version
                        FROM holding_positions WHERE id = ?
                        """,
                        (_required(row, "position_id"),),
                    ).fetchone()
                    if owner is None:
                        raise ValueError("Holding shadow event references an unknown position")
                    if owner["revision_id"] != _required(row, "revision_id"):
                        raise ValueError(
                            "Holding shadow event position belongs to another snapshot"
                        )
                    if owner["position_key"] != _required(row, "position_key"):
                        raise ValueError("Holding shadow event position key mismatch")
                    if owner["symbol"] != _required(row, "symbol"):
                        raise ValueError("Holding shadow event symbol mismatch")
                    if int(owner["version"]) != int(_required(row, "holding_version")):
                        raise ValueError("Holding shadow event version mismatch")
                    cursor = connection.execute(
                        """
                        INSERT INTO holding_stop_shadow_events (
                            id, revision_id, position_id, position_key, symbol,
                            holding_weeks, holding_version, entry_date, data_cutoff,
                            evaluated_at, archive_nature, variant_key, status,
                            source_timeframe, baseline_kind, baseline_date,
                            confirmation_date, baseline_price, latest_close,
                            latest_low, candidate_stop, previous_shadow_stop,
                            effective_shadow_stop, effective_from_date,
                            next_effective_shadow_stop,
                            latest_intraday_touch_observed,
                            intraday_touch_observed, intraday_touch_date,
                            latest_close_breach_observed,
                            close_breach_observed, close_breach_date,
                            company_action_clear,
                            company_action_evidence_id,
                            company_action_evidence_source,
                            company_action_covered_from,
                            company_action_clear_through,
                            company_action_knowledge_time, evaluation_eligible,
                            decision_layer, production_decision_input,
                            external_delivery_allowed, auto_order_allowed,
                            parameters_json, reason_json, input_data_hash, evidence_hash,
                            parameter_hash, method_version, created_at
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        ON CONFLICT(
                            position_key, holding_weeks, data_cutoff,
                            variant_key, method_version, parameter_hash, evidence_hash
                        ) DO NOTHING
                        """,
                        (
                            _required(row, "id"),
                            _required(row, "revision_id"),
                            _required(row, "position_id"),
                            _required(row, "position_key"),
                            _required(row, "symbol"),
                            _required(row, "holding_weeks"),
                            _required(row, "holding_version"),
                            _iso_text(_required(row, "entry_date")),
                            _iso_text(_required(row, "data_cutoff")),
                            _iso_text(_required(row, "evaluated_at")),
                            row.get("archive_nature", "live_shadow"),
                            _required(row, "variant_key"),
                            _required(row, "status"),
                            row.get("source_timeframe"),
                            row.get("baseline_kind"),
                            _iso_optional(row.get("baseline_date")),
                            _iso_optional(row.get("confirmation_date")),
                            row.get("baseline_price"),
                            row.get("latest_close"),
                            row.get("latest_low"),
                            row.get("candidate_stop"),
                            row.get("previous_shadow_stop"),
                            row.get("effective_shadow_stop"),
                            _iso_optional(row.get("effective_from_date")),
                            row.get("next_effective_shadow_stop"),
                            _optional_bool_int(row.get("latest_intraday_touch_observed")),
                            _optional_bool_int(row.get("intraday_touch_observed")),
                            _iso_optional(row.get("intraday_touch_date")),
                            _optional_bool_int(row.get("latest_close_breach_observed")),
                            _optional_bool_int(row.get("close_breach_observed")),
                            _iso_optional(row.get("close_breach_date")),
                            _optional_bool_int(row.get("company_action_clear")),
                            row.get("company_action_evidence_id"),
                            row.get("company_action_evidence_source"),
                            _iso_optional(row.get("company_action_covered_from")),
                            _iso_optional(row.get("company_action_clear_through")),
                            _iso_optional(row.get("company_action_knowledge_time")),
                            1 if bool(row.get("evaluation_eligible", False)) else 0,
                            "shadow_research_only",
                            0,
                            0,
                            0,
                            _json_text(row.get("parameters_json", {})),
                            _json_text(row.get("reason_json", [])),
                            _required(row, "input_data_hash"),
                            _required(row, "evidence_hash"),
                            _required(row, "parameter_hash"),
                            _required(row, "method_version"),
                            _iso_text(row.get("created_at") or datetime.now(UTC)),
                        ),
                    )
                    inserted += max(0, int(cursor.rowcount))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return inserted

    def list_holding_shadow_events(
        self,
        *,
        data_cutoff: date | str | None = None,
        position_key: str | None = None,
        variant_key: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read private shadow observations; no notification path calls this."""

        if limit <= 0 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        self.initialize()
        clauses: list[str] = []
        parameters: list[Any] = []
        if data_cutoff is not None:
            clauses.append("data_cutoff = ?")
            parameters.append(_iso_text(data_cutoff))
        if position_key is not None:
            clauses.append("position_key = ?")
            parameters.append(position_key)
        if variant_key is not None:
            clauses.append("variant_key = ?")
            parameters.append(variant_key)
        where = "" if not clauses else f"WHERE {' AND '.join(clauses)}"
        parameters.append(limit)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM holding_stop_shadow_events
                {where}
                ORDER BY data_cutoff DESC, position_key, variant_key, created_at DESC
                LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()
        return [_decode_row(row, json_columns={"parameters_json", "reason_json"}) for row in rows]

    def get_latest_holding_shadow_state(
        self,
        *,
        position_key: str,
        holding_weeks: int,
        variant_key: str,
        method_version: str,
        parameter_hash: str,
        before_cutoff: date | str,
    ) -> dict[str, Any] | None:
        """Return only an earlier compatible shadow line, never future evidence."""

        self.initialize()
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM holding_stop_shadow_events
                WHERE position_key = ?
                  AND holding_weeks = ?
                  AND variant_key = ?
                  AND method_version = ?
                  AND parameter_hash = ?
                  AND data_cutoff < ?
                  AND effective_shadow_stop IS NOT NULL
                ORDER BY data_cutoff DESC, created_at DESC
                LIMIT 1
                """,
                (
                    position_key,
                    holding_weeks,
                    variant_key,
                    method_version,
                    parameter_hash,
                    _iso_text(before_cutoff),
                ),
            ).fetchone()
        return _decode_row(row, json_columns={"parameters_json", "reason_json"})

    @staticmethod
    def _insert_recommendation_report(
        connection: sqlite3.Connection,
        row: Mapping[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO recommendation_reports (
                id, content_hash, archive_nature, decision_date, plan_for_date,
                common_cutoff, method_version, cycle_label, entry_strictness,
                max_stock_exposure, minimum_cash_weight, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _required(row, "id"),
                _required(row, "content_hash"),
                _required(row, "archive_nature"),
                _iso_text(_required(row, "decision_date")),
                _iso_text(_required(row, "plan_for_date")),
                _iso_text(_required(row, "common_cutoff")),
                _required(row, "method_version"),
                _required(row, "cycle_label"),
                _required(row, "entry_strictness"),
                _required(row, "max_stock_exposure"),
                _required(row, "minimum_cash_weight"),
                _iso_text(row.get("created_at") or datetime.now(UTC)),
                _json_text(row.get("metadata_json", {})),
            ),
        )

    @staticmethod
    def _insert_recommendation_batch(
        connection: sqlite3.Connection,
        row: Mapping[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO recommendation_batches (
                id, report_id, horizon_key, holding_weeks, holding_sessions,
                label, data_cutoff, source_status, evaluation_mode,
                cohort_nature, stock_exposure, cash_weight,
                anchor_session_date, member_count, status, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["report_id"],
                row["horizon_key"],
                row["holding_weeks"],
                row["holding_sessions"],
                row["label"],
                _iso_optional(row.get("data_cutoff")),
                row["source_status"],
                row["evaluation_mode"],
                row["cohort_nature"],
                row["stock_exposure"],
                row["cash_weight"],
                _iso_optional(row.get("anchor_session_date")),
                row["member_count"],
                row["status"],
                _json_text(row.get("metadata_json", {})),
            ),
        )

    @staticmethod
    def _insert_recommendation_member(
        connection: sqlite3.Connection,
        row: Mapping[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO recommendation_members (
                id, batch_id, rank, symbol, name, action, allocation_nature,
                stock_sleeve_weight, account_weight, price_nature, plan_kind,
                price_low, price_high, trigger_price, reference_price,
                observation_anchor, confirmation_rule, invalidation_price,
                plan_cutoff, plan_sessions, plan_method_version,
                price_condition, evidence_pending, primary_timeframe,
                primary_structure, entry_plan_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["batch_id"],
                row["rank"],
                row["symbol"],
                row["name"],
                row["action"],
                row["allocation_nature"],
                row.get("stock_sleeve_weight"),
                row.get("account_weight"),
                row["price_nature"],
                row.get("plan_kind"),
                row.get("price_low"),
                row.get("price_high"),
                row.get("trigger_price"),
                row.get("reference_price"),
                row["observation_anchor"],
                row.get("confirmation_rule"),
                row.get("invalidation_price"),
                _iso_optional(row.get("plan_cutoff")),
                row.get("plan_sessions"),
                row.get("plan_method_version"),
                row["price_condition"],
                int(bool(row.get("evidence_pending", False))),
                row.get("primary_timeframe"),
                row.get("primary_structure"),
                _json_text(row.get("entry_plan_json", {})),
                _json_text(row.get("metadata_json", {})),
            ),
        )

    @staticmethod
    def _insert_recommendation_delivery_event(
        connection: sqlite3.Connection,
        row: Mapping[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO recommendation_delivery_events (
                id, report_id, batch_id, delivery_kind, channel,
                attempted_at, provider_status, provider_receipt_id, detail_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["report_id"],
                row.get("batch_id"),
                row["delivery_kind"],
                row["channel"],
                _iso_text(row["attempted_at"]),
                row["provider_status"],
                row.get("provider_receipt_id"),
                _json_text(row.get("detail_json", {})),
            ),
        )

    @staticmethod
    def _insert_run(connection: sqlite3.Connection, row: Mapping[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO runs (
                id, run_type, as_of, data_cutoff, created_at, strategy_version,
                model_id, config_hash, data_hash, status, warning_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _required(row, "id"),
                _required(row, "run_type"),
                _iso_text(_required(row, "as_of")),
                _iso_text(_required(row, "data_cutoff")),
                _iso_text(row.get("created_at") or datetime.now(UTC)),
                _required(row, "strategy_version"),
                row.get("model_id"),
                _required(row, "config_hash"),
                _required(row, "data_hash"),
                _required(row, "status"),
                _json_text(row.get("warning_json", [])),
            ),
        )

    @staticmethod
    def _insert_child(
        connection: sqlite3.Connection,
        table: str,
        row: Mapping[str, Any],
    ) -> None:
        if table == "data_snapshots":
            connection.execute(
                """
                INSERT INTO data_snapshots (
                    id, run_id, source, dataset, symbol, first_at, last_at,
                    row_count, adjustment, unit_json, checksum, retrieved_at, is_stale
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _required(row, "id"),
                    _required(row, "run_id"),
                    _required(row, "source"),
                    _required(row, "dataset"),
                    row.get("symbol"),
                    _iso_optional(row.get("first_at")),
                    _iso_optional(row.get("last_at")),
                    _required(row, "row_count"),
                    _required(row, "adjustment"),
                    _json_text(row.get("unit_json", {})),
                    _required(row, "checksum"),
                    _iso_text(_required(row, "retrieved_at")),
                    int(bool(row.get("is_stale", False))),
                ),
            )
            return

        if table == "stock_analyses":
            connection.execute(
                """
                INSERT INTO stock_analyses (
                    id, run_id, symbol, horizon_sessions, trend_state,
                    action_for_empty, action_for_holder, entry_low, entry_high,
                    add_above, reduce_low, reduce_high, invalidation, confidence,
                    rationale_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _required(row, "id"),
                    _required(row, "run_id"),
                    _required(row, "symbol"),
                    _required(row, "horizon_sessions"),
                    _required(row, "trend_state"),
                    _required(row, "action_for_empty"),
                    _required(row, "action_for_holder"),
                    row.get("entry_low"),
                    row.get("entry_high"),
                    row.get("add_above"),
                    row.get("reduce_low"),
                    row.get("reduce_high"),
                    row.get("invalidation"),
                    _required(row, "confidence"),
                    _json_text(row.get("rationale_json", {})),
                ),
            )
            return

        if table == "scenarios":
            connection.execute(
                """
                INSERT INTO scenarios (
                    id, analysis_id, label, probability_low, probability_mid,
                    probability_high, return_p10, return_p50, return_p90,
                    sample_n, method, calibration_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _required(row, "id"),
                    _required(row, "analysis_id"),
                    _required(row, "label"),
                    row.get("probability_low"),
                    row.get("probability_mid"),
                    row.get("probability_high"),
                    row.get("return_p10"),
                    row.get("return_p50"),
                    row.get("return_p90"),
                    row.get("sample_n", 0),
                    _required(row, "method"),
                    row.get("calibration_version"),
                ),
            )
            return

        if table == "portfolio_sets":
            connection.execute(
                """
                INSERT INTO portfolio_sets (
                    id, run_id, risk_profile, cash_weight, borrowed_weight,
                    expected_return, expected_vol, expected_max_drawdown,
                    sharpe, metric_window
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _required(row, "id"),
                    _required(row, "run_id"),
                    _required(row, "risk_profile"),
                    _required(row, "cash_weight"),
                    row.get("borrowed_weight", 0.0),
                    row.get("expected_return"),
                    row.get("expected_vol"),
                    row.get("expected_max_drawdown"),
                    row.get("sharpe"),
                    _required(row, "metric_window"),
                ),
            )
            return

        if table == "portfolio_members":
            connection.execute(
                """
                INSERT INTO portfolio_members (portfolio_id, symbol, weight, rank, reason_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    _required(row, "portfolio_id"),
                    _required(row, "symbol"),
                    _required(row, "weight"),
                    _required(row, "rank"),
                    _json_text(row.get("reason_json", {})),
                ),
            )
            return

        if table == "evidence":
            connection.execute(
                """
                INSERT INTO evidence (
                    id, run_id, symbol, evidence_type, source, title,
                    published_at, retrieved_at, url, content_hash, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _required(row, "id"),
                    _required(row, "run_id"),
                    row.get("symbol"),
                    _required(row, "evidence_type"),
                    _required(row, "source"),
                    row.get("title"),
                    _iso_optional(row.get("published_at")),
                    _iso_text(_required(row, "retrieved_at")),
                    row.get("url"),
                    _required(row, "content_hash"),
                    row.get("summary"),
                ),
            )
            return

        raise ValueError(f"Unsupported archive table: {table}")


def _normalize_recommendation_batch(
    row: Mapping[str, Any],
    *,
    archive_nature: str,
    actual_member_count: int,
) -> dict[str, Any]:
    sessions = int(_required(row, "holding_sessions"))
    try:
        expected_key, expected_weeks = _RECOMMENDATION_HORIZONS[sessions]
    except KeyError as exc:
        raise ValueError(f"Unsupported recommendation horizon: {sessions} sessions") from exc

    weeks = int(_first_present(row, "holding_weeks", default=expected_weeks))
    horizon_key = str(_first_present(row, "horizon_key", default=expected_key))
    if (horizon_key, weeks) != (expected_key, expected_weeks):
        raise ValueError(
            f"Horizon {sessions} sessions must use {expected_key}/{expected_weeks} weeks"
        )

    allocation_nature = str(
        _first_present(row, "allocation_nature", "cohort_nature", default="unavailable")
    )
    cohort_nature = row.get("cohort_nature")
    if cohort_nature is None:
        cohort_nature = {
            "action_research": "action_qualified",
            "risk_qualified_research": "risk_qualified",
            "observation_only": "observation_only",
            "unavailable": "unavailable",
            "action_qualified": "action_qualified",
            "risk_qualified": "risk_qualified",
        }.get(allocation_nature, "unavailable")
    cohort_nature = str(cohort_nature)

    evaluation_mode = row.get("evaluation_mode")
    if evaluation_mode is None:
        if cohort_nature == "unavailable":
            evaluation_mode = "unavailable"
        elif archive_nature == "reconstructed":
            evaluation_mode = "reconstructed_observation"
        elif cohort_nature == "action_qualified":
            evaluation_mode = "action_simulation"
        else:
            evaluation_mode = "observation_simulation"

    declared_status = row.get("status")
    status = (
        "unavailable"
        if cohort_nature == "unavailable" or declared_status == "unavailable"
        else "pending"
    )
    declared_count = int(row.get("member_count", actual_member_count))
    return {
        "id": str(_required(row, "id")),
        "report_id": str(_required(row, "report_id")),
        "horizon_key": horizon_key,
        "holding_weeks": weeks,
        "holding_sessions": sessions,
        "label": str(_required(row, "label")),
        "data_cutoff": row.get("data_cutoff"),
        "source_status": str(_required(row, "source_status")),
        "evaluation_mode": str(evaluation_mode),
        "cohort_nature": cohort_nature,
        "stock_exposure": float(
            _first_present(row, "stock_exposure", "action_stock_exposure", default=0.0)
        ),
        "cash_weight": float(_first_present(row, "cash_weight", "action_cash_weight", default=1.0)),
        "anchor_session_date": row.get("anchor_session_date"),
        "member_count": declared_count,
        "status": status,
        "metadata_json": row.get("metadata_json", {}),
    }


def _normalize_recommendation_member(
    row: Mapping[str, Any],
    *,
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    reference_price = _first_present(
        row,
        "reference_price",
        "entry_reference_price",
        "evaluation_price",
    )
    observation_anchor = row.get("observation_anchor")
    if observation_anchor is None:
        if batch["evaluation_mode"] in {
            "observation_simulation",
            "reconstructed_observation",
        }:
            observation_anchor = "plan_session_close"
        else:
            observation_anchor = "none"
    return {
        "id": str(_required(row, "id")),
        "batch_id": str(_required(row, "batch_id")),
        "rank": int(_required(row, "rank")),
        "symbol": str(_required(row, "symbol")),
        "name": str(_required(row, "name")),
        "action": str(_required(row, "action")),
        "allocation_nature": str(_required(row, "allocation_nature")),
        "stock_sleeve_weight": _first_present(
            row,
            "stock_sleeve_weight",
            "operational_stock_sleeve_weight",
            "sleeve_weight",
        ),
        "account_weight": _first_present(
            row,
            "account_weight",
            "operational_account_weight",
        ),
        "price_nature": str(_required(row, "price_nature")),
        "plan_kind": row.get("plan_kind"),
        "price_low": _first_present(row, "price_low", "entry_low"),
        "price_high": _first_present(row, "price_high", "entry_high"),
        "trigger_price": row.get("trigger_price"),
        "reference_price": reference_price,
        "observation_anchor": str(observation_anchor),
        "confirmation_rule": row.get("confirmation_rule"),
        "invalidation_price": _first_present(
            row,
            "invalidation_price",
            "invalidation",
        ),
        "plan_cutoff": row.get("plan_cutoff"),
        "plan_sessions": row.get("plan_sessions"),
        "plan_method_version": row.get("plan_method_version"),
        "price_condition": str(_required(row, "price_condition")),
        "evidence_pending": bool(row.get("evidence_pending", False)),
        "primary_timeframe": row.get("primary_timeframe"),
        "primary_structure": row.get("primary_structure"),
        "entry_plan_json": _first_present(
            row,
            "entry_plan_json",
            "entry_rule_json",
            default={},
        ),
        "metadata_json": row.get("metadata_json", {}),
    }


def _normalize_recommendation_delivery_event(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(_required(row, "id")),
        "report_id": str(_required(row, "report_id")),
        "batch_id": row.get("batch_id"),
        "delivery_kind": str(
            _required(
                {
                    "delivery_kind": _first_present(
                        row,
                        "delivery_kind",
                        "event_kind",
                    )
                },
                "delivery_kind",
            )
        ),
        "channel": str(_required(row, "channel")),
        "attempted_at": _required(row, "attempted_at"),
        "provider_status": str(
            _required(
                {
                    "provider_status": _first_present(
                        row,
                        "provider_status",
                        "status",
                    )
                },
                "provider_status",
            )
        ),
        "provider_receipt_id": row.get("provider_receipt_id"),
        "detail_json": _first_present(row, "detail_json", "details_json", default={}),
    }


def _member_result_aliases(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        raise ValueError("Expected a recommendation member row")
    if row.get("observation_anchor") == "none":
        row["observation_anchor"] = None
    row["member_id"] = row["id"]
    row["sleeve_weight"] = row["stock_sleeve_weight"]
    row["operational_stock_sleeve_weight"] = row["stock_sleeve_weight"]
    row["operational_account_weight"] = row["account_weight"]
    row["entry_reference_price"] = row["reference_price"]
    return row


def _first_present(
    row: Mapping[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return default


def _optional_bool_int(value: Any) -> int | None:
    if value is None:
        return None
    if value in (True, 1):
        return 1
    if value in (False, 0):
        return 0
    raise ValueError(f"Expected an optional boolean value, got {value!r}")


class _SettlementTransactionRepository(SQLiteRepository):
    """Internal adapter sharing an already-open outer settlement transaction."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._settlement_connection = connection

    def initialize(self) -> None:
        pass

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        yield self._settlement_connection


def _normalize_member_result_status(value: Any) -> str:
    text = str(value)
    if text in {"pending", "pending_maturity_session"}:
        return "pending"
    if text in {"resolved", "settled"}:
        return "resolved"
    if text in {
        "not_entered",
        "expired_untriggered",
        "entry_price_limit_exceeded",
        "entry_invalidated_before_fill",
    }:
        return "not_entered"
    if text in {"unavailable", "archive_ineligible"}:
        return "unavailable"
    if text == "needs_review" or text in {
        "entry_rule_incomplete",
        "source_adjustment_invalid",
        "plan_session_missing",
        "plan_price_missing",
        "plan_price_invalid",
        "plan_session_suspended",
        "entry_price_missing",
        "entry_price_invalid",
        "entry_session_suspended",
        "maturity_price_missing",
        "maturity_price_invalid",
        "maturity_session_suspended",
        "corporate_action_evidence_unknown",
        "corporate_action_detected",
    }:
        return "needs_review"
    raise ValueError(f"Unsupported recommendation member result status: {value!r}")


def _normalize_batch_result_status(value: Any) -> str:
    text = str(value)
    normalized = {
        "pending": "pending",
        "partial": "partial",
        "settled_partial_entry": "partial",
        "resolved": "resolved",
        "settled": "resolved",
        "no_entry": "no_entries",
        "no_entries": "no_entries",
        "unavailable": "unavailable",
        "needs_review": "needs_review",
        "data_quality_failure": "needs_review",
    }.get(text)
    if normalized is None:
        raise ValueError(f"Unsupported recommendation batch result status: {value!r}")
    return normalized


def _record(value: JsonRecord) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="python")
    raise TypeError(f"Expected a mapping, dataclass, or Pydantic model; got {type(value)!r}")


def _required(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required field: {key}")
    if isinstance(value, Decimal):
        return float(value)
    return value


def _iso_text(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("Datetime values must include a timezone")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"Expected an ISO date/datetime value, got {value!r}")


def _iso_optional(value: Any) -> str | None:
    return None if value is None else _iso_text(value)


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        # Validate callers that already serialized the value.
        json.loads(value)
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode_row(
    row: sqlite3.Row | None,
    *,
    json_columns: set[str] | None = None,
) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    for column in json_columns or set():
        value = result.get(column)
        if value is not None:
            result[column] = json.loads(value)
    return result
