"""SQLite persistence for immutable research predictions and later outcomes.

This database is an audit log, not a LangGraph checkpoint.  Completed research
runs are inserted atomically and never edited.  When the forecast horizon has
elapsed, observed returns are written to the separate ``outcomes`` table.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

JsonRecord = Mapping[str, Any] | Any


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
