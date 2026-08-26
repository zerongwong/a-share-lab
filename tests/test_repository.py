from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ashare_lab.adapters.sqlite_repository import SQLiteRepository


@pytest.fixture
def repository(tmp_path: Path) -> SQLiteRepository:
    migrations = Path(__file__).resolve().parents[1] / "migrations"
    repo = SQLiteRepository(tmp_path / "research.db", migrations)
    repo.initialize()
    return repo


def _run(run_id: str = "run-1") -> dict:
    return {
        "id": run_id,
        "run_type": "stock_analysis",
        "as_of": "2026-08-21",
        "data_cutoff": "2026-08-21T15:00:00+08:00",
        "created_at": datetime(2026, 8, 21, 8, 0, tzinfo=UTC),
        "strategy_version": "edwards-magee+livermore/0.1.0",
        "model_id": "qwen3.6:35b",
        "config_hash": "config-sha256",
        "data_hash": "data-sha256",
        "status": "completed",
        "warning_json": ["AKShare prototype source"],
    }


def _analysis(run_id: str = "run-1", analysis_id: str = "analysis-1") -> dict:
    return {
        "id": analysis_id,
        "run_id": run_id,
        "symbol": "600150.SH",
        "horizon_sessions": 20,
        "trend_state": "primary_down_secondary_rebound",
        "action_for_empty": "wait_for_confirmation",
        "action_for_holder": "hold_small_or_reduce_on_failed_breakout",
        "entry_low": 32.7,
        "entry_high": 33.1,
        "add_above": 34.88,
        "reduce_low": 35.5,
        "reduce_high": 35.6,
        "invalidation": 31.98,
        "confidence": "medium",
        "rationale_json": {"rule": "close_above_resistance_with_volume"},
    }


def test_initialize_is_idempotent_and_records_version(repository: SQLiteRepository) -> None:
    repository.initialize()
    with repository.connection() as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert [row["version"] for row in versions] == [1]


def test_archive_bundle_round_trip(repository: SQLiteRepository) -> None:
    repository.archive_run(
        _run(),
        data_snapshots=[
            {
                "id": "snapshot-1",
                "run_id": "run-1",
                "source": "akshare",
                "dataset": "daily_ohlcv",
                "symbol": "600150.SH",
                "first_at": "2021-08-23",
                "last_at": "2026-08-21",
                "row_count": 1215,
                "adjustment": "none",
                "unit_json": {"volume": "shares", "amount": "CNY"},
                "checksum": "snapshot-sha256",
                "retrieved_at": "2026-08-21T15:05:00+08:00",
                "is_stale": False,
            }
        ],
        analyses=[_analysis()],
        scenarios=[
            {
                "id": "scenario-up",
                "analysis_id": "analysis-1",
                "label": "up",
                "probability_low": 0.32,
                "probability_mid": 0.41,
                "probability_high": 0.50,
                "return_p10": -0.04,
                "return_p50": 0.025,
                "return_p90": 0.11,
                "sample_n": 84,
                "method": "historical-neighbours-v1",
                "calibration_version": "walk-forward-2026-08",
            }
        ],
        portfolio_sets=[
            {
                "id": "portfolio-1",
                "run_id": "run-1",
                "risk_profile": "conservative",
                "cash_weight": 0.2,
                "borrowed_weight": 0.0,
                "expected_return": 0.08,
                "expected_vol": 0.13,
                "expected_max_drawdown": -0.1,
                "sharpe": 0.55,
                "metric_window": "walk-forward-3y",
            }
        ],
        portfolio_members=[
            {
                "portfolio_id": "portfolio-1",
                "symbol": "600150.SH",
                "weight": 0.3,
                "rank": 1,
                "reason_json": {"relative_strength": "high"},
            }
        ],
        evidence=[
            {
                "id": "evidence-1",
                "run_id": "run-1",
                "symbol": "600150.SH",
                "evidence_type": "market_snapshot",
                "source": "akshare",
                "title": "2026-08-21 verified close",
                "published_at": "2026-08-21T15:00:00+08:00",
                "retrieved_at": "2026-08-21T15:05:00+08:00",
                "url": None,
                "content_hash": "evidence-sha256",
                "summary": "Close 33.68 CNY; volume normalized to shares.",
            }
        ],
    )

    run = repository.get_run("run-1")
    assert run is not None
    assert run["warning_json"] == ["AKShare prototype source"]

    analyses = repository.list_stock_analyses("run-1")
    assert len(analyses) == 1
    assert analyses[0]["rationale_json"]["rule"] == "close_above_resistance_with_volume"

    scenarios = repository.list_scenarios("analysis-1")
    assert scenarios[0]["probability_mid"] == pytest.approx(0.41)


def test_archive_is_atomic_on_invalid_child(repository: SQLiteRepository) -> None:
    invalid_scenario = {
        "id": "scenario-orphan",
        "analysis_id": "missing-analysis",
        "label": "up",
        "sample_n": 0,
        "method": "unavailable",
    }

    with pytest.raises(sqlite3.IntegrityError):
        repository.archive_run(_run("rolled-back"), scenarios=[invalid_scenario])

    assert repository.get_run("rolled-back") is None


def test_prediction_rows_are_immutable(repository: SQLiteRepository) -> None:
    repository.archive_run(_run(), analyses=[_analysis()])

    with repository.connection() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "UPDATE stock_analyses SET action_for_empty = 'buy_now' WHERE id = ?",
            ("analysis-1",),
        )

    assert repository.list_stock_analyses("run-1")[0]["action_for_empty"] == (
        "wait_for_confirmation"
    )

    with repository.connection() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute("DELETE FROM stock_analyses WHERE id = ?", ("analysis-1",))


def test_outcome_can_be_resolved_without_mutating_prediction(
    repository: SQLiteRepository,
) -> None:
    repository.archive_run(_run(), analyses=[_analysis()])
    pending = {
        "id": "outcome-1",
        "analysis_id": "analysis-1",
        "observed_at": "2026-09-18",
        "status": "pending",
    }
    repository.record_outcome(pending)

    resolved = {
        **pending,
        "realized_return": 0.036,
        "max_drawdown": -0.041,
        "max_runup": 0.072,
        "relative_return": 0.012,
        "status": "resolved",
    }
    repository.record_outcome(resolved)

    outcomes = repository.get_outcomes("analysis-1")
    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "resolved"
    assert outcomes[0]["realized_return"] == pytest.approx(0.036)
    assert repository.list_stock_analyses("run-1")[0]["confidence"] == "medium"


def test_position_is_mutable_local_state(repository: SQLiteRepository) -> None:
    repository.save_position(
        {
            "id": "position-1",
            "symbol": "600150.SH",
            "shares": 1000,
            "cost_price": 34.2,
            "as_of": "2026-08-21",
        }
    )
    repository.save_position(
        {
            "id": "position-2",
            "symbol": "600150.SH",
            "shares": 800,
            "cost_price": 33.9,
            "as_of": "2026-08-24",
        }
    )

    position = repository.get_position("600150.SH")
    assert position is not None
    assert position["id"] == "position-2"
    assert position["shares"] == pytest.approx(800)
    assert position["as_of"] == "2026-08-24"
