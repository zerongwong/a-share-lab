from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ashare_lab.adapters.sqlite_repository import SQLiteRepository


def _legacy_database(db_path: Path, migrations: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.executescript((migrations / "001_init.sql").read_text(encoding="utf-8"))
        connection.executescript(
            (migrations / "003_active_holding_tree.sql").read_text(encoding="utf-8")
        )
        connection.execute(
            """
            INSERT INTO holding_protective_stops (
                position_key, symbol, entry_date, effective_stop,
                candidate_stop, previous_stop, data_cutoff,
                source_timeframe, evidence_date, holding_version,
                method_version, details_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-position",
                "600000",
                "2026-09-16",
                9.20,
                9.20,
                None,
                "2026-09-16",
                "daily",
                "2026-09-16",
                1,
                "holding-stop-v1",
                '{"origin":"legacy"}',
                "2026-09-16T15:30:00+08:00",
            ),
        )


def _stop_row(position_key: str) -> tuple[object, ...]:
    return (
        position_key,
        "600000",
        "2026-09-16",
        9.50,
        9.50,
        None,
        "2026-09-17",
        "daily",
        "2026-09-17",
        2,
        "holding-stop-v2",
        '{"origin":"new-revision"}',
        "2026-09-17T15:30:00+08:00",
    )


def test_migration_preserves_old_stop_and_allows_fresh_position_identity(
    tmp_path: Path,
) -> None:
    migrations = Path(__file__).resolve().parents[1] / "migrations"
    db_path = tmp_path / "legacy.db"
    _legacy_database(db_path, migrations)

    repository = SQLiteRepository(db_path, migrations)
    repository.initialize()
    repository.initialize()

    with repository.connection() as connection:
        legacy = connection.execute(
            "SELECT * FROM holding_protective_stops WHERE position_key = ?",
            ("legacy-position",),
        ).fetchone()
        assert legacy is not None
        assert legacy["effective_stop"] == pytest.approx(9.20)
        assert legacy["details_json"] == '{"origin":"legacy"}'

        connection.execute(
            """
            INSERT INTO holding_protective_stops (
                position_key, symbol, entry_date, effective_stop,
                candidate_stop, previous_stop, data_cutoff,
                source_timeframe, evidence_date, holding_version,
                method_version, details_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _stop_row("fresh-position"),
        )

        rows = connection.execute(
            """
            SELECT position_key FROM holding_protective_stops
            WHERE symbol = ? AND entry_date = ? ORDER BY position_key
            """,
            ("600000", "2026-09-16"),
        ).fetchall()
        assert [row["position_key"] for row in rows] == [
            "fresh-position",
            "legacy-position",
        ]
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE version = 10"
            ).fetchone()[0]
            == 1
        )


def test_migration_recreates_stop_ratchet_guards(tmp_path: Path) -> None:
    migrations = Path(__file__).resolve().parents[1] / "migrations"
    db_path = tmp_path / "legacy.db"
    _legacy_database(db_path, migrations)
    repository = SQLiteRepository(db_path, migrations)
    repository.initialize()

    with repository.connection() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="cannot move down"):
            connection.execute(
                """
                UPDATE holding_protective_stops SET effective_stop = 9.10
                WHERE position_key = 'legacy-position'
                """
            )
        with pytest.raises(sqlite3.IntegrityError, match="cannot move backwards"):
            connection.execute(
                """
                UPDATE holding_protective_stops SET data_cutoff = '2026-09-15'
                WHERE position_key = 'legacy-position'
                """
            )
