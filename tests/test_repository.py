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
    assert [row["version"] for row in versions] == [1, 2, 3, 4]


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


def _recommendation_report() -> dict:
    return {
        "id": "report-20260828",
        "content_hash": "digest-content-sha256",
        "archive_nature": "original",
        "decision_date": "2026-08-27",
        "plan_for_date": "2026-08-28",
        "common_cutoff": "2026-08-27",
        "method_version": "evening-six-horizon-digest-v0.4.0",
        "cycle_label": "震荡防守",
        "entry_strictness": "加强确认",
        "max_stock_exposure": 0.3,
        "minimum_cash_weight": 0.7,
        "created_at": "2026-08-27T21:00:00+08:00",
        "metadata_json": {"raw_data_exposed": False, "orders_enabled": False},
    }


def _recommendation_batch() -> dict:
    return {
        "id": "batch-20260828-1w",
        "report_id": "report-20260828",
        "holding_weeks": 1,
        "holding_sessions": 5,
        "label": "1周",
        "data_cutoff": "2026-08-27",
        "source_status": "ready",
        "allocation_nature": "action_research",
        "action_stock_exposure": 0.3,
        "action_cash_weight": 0.7,
        "member_count": 1,
        "status": "pending",
    }


def _recommendation_member() -> dict:
    return {
        "id": "member-20260828-1w-1",
        "batch_id": "batch-20260828-1w",
        "rank": 1,
        "symbol": "601919.SH",
        "name": "中远海控",
        "action": "条件介入研究",
        "allocation_nature": "action_research",
        "operational_stock_sleeve_weight": 1.0,
        "operational_account_weight": 0.3,
        "price_nature": "conditional_entry",
        "plan_kind": "breakout_or_pullback",
        "price_low": 14.8,
        "price_high": 15.1,
        "trigger_price": 15.3,
        "evaluation_price": 15.0,
        "confirmation_rule": "plan_day_close_confirms_then_next_session_open",
        "invalidation_price": 14.2,
        "plan_cutoff": "2026-08-27",
        "plan_sessions": 5,
        "plan_method_version": "conditional-entry-v1",
        "price_condition": "回踩14.80–15.10或放量站上15.30",
        "evidence_pending": False,
        "primary_timeframe": "daily",
        "primary_structure": "near_breakout",
        "entry_rule_json": {
            "kind": "reclaim_close_confirmation",
            "trigger_price": 15.3,
            "trigger_valid_sessions": 1,
        },
    }


def _accepted_delivery_event(
    *,
    event_id: str,
    delivery_kind: str,
    batch_id=None,
    attempted_at: str = "2026-08-27T21:01:00+08:00",
    detail_json: dict | None = None,
) -> dict:
    return {
        "id": event_id,
        "report_id": "report-20260828",
        "batch_id": batch_id,
        "delivery_kind": delivery_kind,
        "channel": "serverchan",
        "attempted_at": attempted_at,
        "provider_status": "provider_accepted",
        "provider_receipt_id": "receipt-redacted",
        "detail_json": detail_json or {"delivery_confirmed": False},
    }


def test_recommendation_archive_is_atomic_immutable_and_queryable(
    repository: SQLiteRepository,
) -> None:
    repository.archive_recommendation_report(
        _recommendation_report(),
        batches=[_recommendation_batch()],
        members=[_recommendation_member()],
        delivery_events=[
            _accepted_delivery_event(
                event_id="delivery-evening-1",
                delivery_kind="evening_provider_accepted",
            )
        ],
    )
    repository.archive_recommendation_report(
        {**_recommendation_report(), "created_at": "2026-08-27T21:05:00+08:00"},
        batches=[_recommendation_batch()],
        members=[_recommendation_member()],
        delivery_events=[
            _accepted_delivery_event(
                event_id="delivery-evening-1",
                delivery_kind="evening_provider_accepted",
            )
        ],
    )

    report = repository.get_recommendation_report("report-20260828")
    assert report is not None
    assert report["metadata_json"]["orders_enabled"] is False
    assert repository.get_recommendation_report_by_content_hash("digest-content-sha256") == report

    batches = repository.list_recommendation_batches("report-20260828")
    assert len(batches) == 1
    assert batches[0]["horizon_key"] == "1w"
    assert batches[0]["evaluation_mode"] == "action_simulation"
    assert batches[0]["cohort_nature"] == "action_qualified"
    assert batches[0]["delivery_accepted"] == 1

    members = repository.list_recommendation_members("batch-20260828-1w")
    assert members[0]["entry_plan_json"]["trigger_valid_sessions"] == 1
    assert members[0]["sleeve_weight"] == pytest.approx(1.0)
    assert members[0]["entry_reference_price"] == pytest.approx(15.0)
    assert members[0]["observation_anchor"] is None

    from ashare_lab.services.settle_recommendation_performance import (
        archived_batch_from_mapping,
        archived_member_from_mapping,
        archived_report_from_mapping,
    )

    assert archived_report_from_mapping(batches[0]).plan_for_date.isoformat() == "2026-08-28"
    assert archived_batch_from_mapping(batches[0]).holding_sessions == 5
    assert archived_member_from_mapping(members[0]).entry_plan is not None

    pending = repository.list_recommendation_batches_pending_settlement(as_of="2026-08-31")
    assert [row["id"] for row in pending] == ["batch-20260828-1w"]

    with repository.connection() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "UPDATE recommendation_members SET symbol = '000001.SZ' WHERE id = ?",
            ("member-20260828-1w-1",),
        )


def test_recommendation_archive_rejects_incomplete_bundle_atomically(
    repository: SQLiteRepository,
) -> None:
    batch = {**_recommendation_batch(), "member_count": 2}
    with pytest.raises(ValueError, match="declares 2 members"):
        repository.archive_recommendation_report(
            _recommendation_report(),
            batches=[batch],
            members=[_recommendation_member()],
        )
    assert repository.get_recommendation_report("report-20260828") is None


def test_recommendation_maturity_results_are_idempotent_and_notified_once(
    repository: SQLiteRepository,
) -> None:
    repository.archive_recommendation_report(
        _recommendation_report(),
        batches=[_recommendation_batch()],
        members=[_recommendation_member()],
    )
    pending_member = {
        "member_id": "member-20260828-1w-1",
        "evaluated_at": "2026-09-03T16:00:00+08:00",
        "status": "pending",
        "holding_sessions_observed": 4,
        "reason_code": "horizon_not_elapsed",
        "data_cutoff": "2026-09-03",
        "method_version": "maturity-v1",
    }
    repository.record_recommendation_member_result(pending_member)
    repository.record_recommendation_member_result(
        {
            **pending_member,
            "evaluated_at": "2026-09-04T16:00:00+08:00",
            "status": "resolved",
            "entry_date": "2026-08-31",
            "entry_price": 15.2,
            "due_date": "2026-09-04",
            "due_close": 15.96,
            "return_pct": 0.05,
            "holding_sessions_observed": 5,
            "reason_code": "maturity_close_verified",
            "company_action_clear": True,
            "data_cutoff": "2026-09-04",
        }
    )
    member_result = repository.get_recommendation_member_result("member-20260828-1w-1")
    assert member_result is not None
    assert member_result["status"] == "resolved"
    assert member_result["realized_return"] == pytest.approx(0.05)
    assert member_result["maturity_date"] == "2026-09-04"

    batch_result = {
        "batch_id": "batch-20260828-1w",
        "evaluated_at": "2026-09-04T16:01:00+08:00",
        "status": "resolved",
        "due_date": "2026-09-04",
        "stock_sleeve_return": 0.05,
        "account_return": 0.015,
        "entered_weight": 1.0,
        "entered_account_weight": 0.3,
        "cash_weight": 0.7,
        "resolved_member_count": 1,
        "total_member_count": 1,
        "reason_code": "all_entered_members_resolved",
        "data_cutoff": "2026-09-04",
        "method_version": "maturity-v1",
    }
    repository.record_recommendation_batch_result(batch_result)
    repository.record_recommendation_batch_result(batch_result)
    assert repository.get_recommendation_batch_result("batch-20260828-1w")[
        "account_return"
    ] == pytest.approx(0.015)
    assert len(repository.list_recommendation_member_results("batch-20260828-1w")) == 1
    performance = repository.get_recommendation_batch_performance("batch-20260828-1w")
    assert performance is not None
    assert performance["batch"]["horizon_key"] == "1w"
    assert performance["members"][0]["result"]["status"] == "resolved"

    due_notifications = repository.list_maturity_results_pending_notification()
    assert [row["batch_id"] for row in due_notifications] == ["batch-20260828-1w"]

    repository.record_recommendation_delivery_event(
        _accepted_delivery_event(
            event_id="delivery-maturity-1",
            delivery_kind="maturity_provider_accepted",
            batch_id="batch-20260828-1w",
            attempted_at="2026-09-04T16:02:00+08:00",
            detail_json={
                "result_status": "resolved",
                "result_method_version": "maturity-v1",
            },
        )
    )
    assert repository.list_maturity_results_pending_notification() == []
    assert repository.list_recommendation_batches_pending_settlement() == []


def test_needs_review_without_maturity_date_is_never_notified(
    repository: SQLiteRepository,
) -> None:
    repository.archive_recommendation_report(
        _recommendation_report(),
        batches=[_recommendation_batch()],
        members=[_recommendation_member()],
    )
    repository.record_recommendation_batch_result(
        {
            "batch_id": "batch-20260828-1w",
            "evaluated_at": "2026-08-31T16:00:00+08:00",
            "status": "needs_review",
            "maturity_date": None,
            "resolved_member_count": 0,
            "total_member_count": 1,
            "reason_code": "pre_maturity_archive_review",
            "data_cutoff": "2026-08-31",
            "method_version": "maturity-v1",
        }
    )

    assert repository.list_maturity_results_pending_notification() == []


def test_needs_review_and_final_result_notifications_are_separately_idempotent(
    repository: SQLiteRepository,
) -> None:
    repository.archive_recommendation_report(
        _recommendation_report(),
        batches=[_recommendation_batch()],
        members=[_recommendation_member()],
    )
    needs_review = {
        "batch_id": "batch-20260828-1w",
        "evaluated_at": "2026-09-04T16:00:00+08:00",
        "updated_at": "2026-09-04T16:00:00+08:00",
        "status": "needs_review",
        "maturity_date": "2026-09-04",
        "resolved_member_count": 1,
        "total_member_count": 1,
        "reason_code": "corporate_action_evidence_unknown",
        "data_cutoff": "2026-09-04",
        "method_version": "maturity-v1",
    }
    repository.record_recommendation_batch_result(needs_review)
    assert [row["batch_id"] for row in repository.list_maturity_results_pending_notification()] == [
        "batch-20260828-1w"
    ]
    assert [row["id"] for row in repository.list_recommendation_batches_pending_settlement()] == [
        "batch-20260828-1w"
    ]

    repository.record_recommendation_delivery_event(
        _accepted_delivery_event(
            event_id="delivery-maturity-review",
            delivery_kind="maturity_provider_accepted",
            batch_id="batch-20260828-1w",
            attempted_at="2026-09-04T16:01:00+08:00",
            detail_json={
                "result_status": "needs_review",
                "result_method_version": "maturity-v1",
            },
        )
    )
    assert repository.list_maturity_results_pending_notification() == []

    repository.record_recommendation_batch_result(
        {
            **needs_review,
            "evaluated_at": "2026-09-05T16:00:00+08:00",
            "updated_at": "2026-09-05T16:00:00+08:00",
            "status": "resolved",
            "stock_sleeve_return": 0.04,
            "account_return": 0.02,
            "reason_code": "settled",
            "data_cutoff": "2026-09-05",
        }
    )
    assert [row["batch_id"] for row in repository.list_maturity_results_pending_notification()] == [
        "batch-20260828-1w"
    ]
    assert repository.list_recommendation_batches_pending_settlement() == []

    repository.record_recommendation_delivery_event(
        _accepted_delivery_event(
            event_id="delivery-maturity-final",
            delivery_kind="maturity_provider_accepted",
            batch_id="batch-20260828-1w",
            attempted_at="2026-09-05T16:01:00+08:00",
            detail_json={
                "result_status": "resolved",
                "result_method_version": "maturity-v1",
            },
        )
    )
    assert repository.list_maturity_results_pending_notification() == []


def test_partial_entry_result_is_terminal_for_settlement(
    repository: SQLiteRepository,
) -> None:
    repository.archive_recommendation_report(
        _recommendation_report(),
        batches=[_recommendation_batch()],
        members=[_recommendation_member()],
    )
    repository.record_recommendation_batch_result(
        {
            "batch_id": "batch-20260828-1w",
            "evaluated_at": "2026-09-04T16:00:00+08:00",
            "updated_at": "2026-09-04T16:00:00+08:00",
            "status": "settled_partial_entry",
            "maturity_date": "2026-09-04",
            "stock_sleeve_return": 0.04,
            "account_return": 0.02,
            "resolved_member_count": 1,
            "total_member_count": 1,
            "reason_code": "settled_partial_entry",
            "data_cutoff": "2026-09-04",
            "method_version": "maturity-v1",
        }
    )

    assert repository.list_recommendation_batches_pending_settlement() == []
