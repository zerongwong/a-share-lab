from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.analytics.magee_shadow import (
    MAGEE_SHADOW_METHOD_VERSION,
    MageeShadowVariant,
)
from ashare_lab.services.holding_ledger import HoldingPositionInput, replace_active_holdings
from ashare_lab.services.run_active_holding_review import ActiveHoldingHistoryLoad
from ashare_lab.services.run_holding_stop_shadows import (
    HoldingStopShadowRunStatus,
    run_holding_stop_shadows,
)


@pytest.fixture
def repository(tmp_path: Path) -> SQLiteRepository:
    result = SQLiteRepository(
        tmp_path / "research.db",
        Path(__file__).resolve().parents[1] / "migrations",
    )
    result.initialize()
    return result


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("2026-08-03", 10.50, 11.00, 10.00, 10.60),
            ("2026-08-04", 11.20, 11.50, 11.01, 11.30),
            ("2026-08-05", 11.30, 11.60, 11.02, 11.40),
            ("2026-08-06", 11.40, 11.90, 11.03, 11.50),
            ("2026-08-07", 11.50, 12.00, 11.10, 11.60),
        ],
        columns=("trade_date", "open", "high", "low", "close"),
    )


def _set_holding(repository: SQLiteRepository):
    return replace_active_holdings(
        repository,
        (
            HoldingPositionInput(
                symbol="600919",
                name="江苏银行",
                entry_date=date(2026, 8, 3),
                stock_sleeve_weight=1.0,
                metadata={
                    "company_action_clear": True,
                    "company_action_evidence_id": "ca:600919:2026-08-07",
                    "company_action_evidence_source": "test_point_in_time_feed",
                    "company_action_covered_from": "2026-08-03",
                    "company_action_clear_through": "2026-08-07",
                    "company_action_knowledge_time": "2026-08-07T16:00:00+08:00",
                },
            ),
        ),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 3, 21, 0, tzinfo=UTC),
        change_id="shadow-test-holding",
    )


def _loader(portfolio, frame: pd.DataFrame):
    def load(_repository, **_kwargs):
        return ActiveHoldingHistoryLoad(
            portfolio_id=portfolio.id,
            holding_version=portfolio.version,
            holding_weeks=portfolio.holding_weeks,
            histories={"600919": frame},
            baseline_cutoff=date(2026, 8, 3),
            verified_overlay_dates=(date(2026, 8, 7),),
            data_cutoff=date(2026, 8, 7),
            unavailable_symbols=(),
            sources=("CSMAR", "infoway"),
        )

    return load


def test_runner_archives_two_immutable_local_only_variants_idempotently(
    repository: SQLiteRepository,
    tmp_path: Path,
) -> None:
    portfolio = _set_holding(repository)
    arguments = {
        "dataset_root": tmp_path / "csmar",
        "overlay_root": tmp_path / "overlay",
        "as_of": date(2026, 8, 7),
        "evaluated_at": datetime(2026, 8, 7, 22, 0, tzinfo=UTC),
        "_history_loader": _loader(portfolio, _history()),
    }

    first = run_holding_stop_shadows(repository, **arguments)
    second = run_holding_stop_shadows(repository, **arguments)

    # Official point-in-time session-chain evidence is not yet integrated, so
    # raw shadows are archived but deliberately remain ineligible for comparison.
    assert first.status is HoldingStopShadowRunStatus.DATA_NOT_READY
    assert first.observation_count == 2
    assert first.inserted_count == 2
    assert first.variant_count == 2
    assert second.inserted_count == 0

    rows = repository.list_holding_shadow_events(data_cutoff="2026-08-07")
    assert len(rows) == 2
    assert {row["variant_key"] for row in rows} == {
        MageeShadowVariant.THREE_DAY_ESCAPE_6PCT.value,
        MageeShadowVariant.NEW_HIGH_3PCT_6PCT.value,
    }
    assert {row["status"] for row in rows} == {"needs_review"}
    assert {row["method_version"] for row in rows} == {MAGEE_SHADOW_METHOD_VERSION}
    assert all(row["decision_layer"] == "shadow_research_only" for row in rows)
    assert all(row["production_decision_input"] == 0 for row in rows)
    assert all(row["external_delivery_allowed"] == 0 for row in rows)
    assert all(row["auto_order_allowed"] == 0 for row in rows)
    assert all(row["evaluation_eligible"] == 0 for row in rows)
    assert all(row["company_action_clear"] == 1 for row in rows)
    assert all(row["company_action_covered_from"] == "2026-08-03" for row in rows)
    assert all(row["company_action_clear_through"] == "2026-08-07" for row in rows)
    assert all(row["parameter_hash"] and row["input_data_hash"] for row in rows)
    assert all(
        "official_session_chain_unverified_shadow_not_comparison_eligible" in row["reason_json"]
        for row in rows
    )

    with repository.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM holding_review_events").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM holding_protective_stops").fetchone()[0] == 0
        )
        ddl = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("holding_stop_shadow_events",),
        ).fetchone()[0]
    assert "production_decision_input = 0" in ddl
    assert "external_delivery_allowed = 0" in ddl
    assert "auto_order_allowed = 0" in ddl

    row = next(
        item
        for item in rows
        if item["variant_key"] == MageeShadowVariant.THREE_DAY_ESCAPE_6PCT.value
    )
    prior = repository.get_latest_holding_shadow_state(
        position_key=row["position_key"],
        holding_weeks=4,
        variant_key=row["variant_key"],
        method_version=row["method_version"],
        parameter_hash=row["parameter_hash"],
        before_cutoff="2026-08-08",
    )
    assert prior is not None
    assert prior["id"] == row["id"]

    with repository.connection() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "UPDATE holding_stop_shadow_events SET status = 'ready' WHERE id = ?",
            (row["id"],),
        )
    with repository.connection() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute("DELETE FROM holding_stop_shadow_events WHERE id = ?", (row["id"],))


def test_no_holding_is_a_local_noop(
    repository: SQLiteRepository,
    tmp_path: Path,
) -> None:
    summary = run_holding_stop_shadows(
        repository,
        dataset_root=tmp_path / "csmar",
        overlay_root=tmp_path / "overlay",
        as_of=date(2026, 8, 7),
        _history_loader=lambda *_args, **_kwargs: None,
    )

    assert summary.status is HoldingStopShadowRunStatus.NO_HOLDINGS
    assert summary.observation_count == 0
    assert repository.list_holding_shadow_events() == []


def test_changed_ohlc_input_creates_distinct_evidence_even_when_result_is_same(
    repository: SQLiteRepository,
    tmp_path: Path,
) -> None:
    portfolio = _set_holding(repository)
    original = _history()
    changed = _history()
    changed.loc[0, "open"] = 10.55
    common = {
        "dataset_root": tmp_path / "csmar",
        "overlay_root": tmp_path / "overlay",
        "as_of": date(2026, 8, 7),
        "evaluated_at": datetime(2026, 8, 7, 22, 0, tzinfo=UTC),
    }

    first = run_holding_stop_shadows(
        repository,
        **common,
        _history_loader=_loader(portfolio, original),
    )
    second = run_holding_stop_shadows(
        repository,
        **common,
        _history_loader=_loader(portfolio, changed),
    )

    assert first.inserted_count == 2
    assert second.inserted_count == 2
    rows = repository.list_holding_shadow_events(data_cutoff="2026-08-07")
    assert len(rows) == 4
    for variant in MageeShadowVariant:
        variant_rows = [row for row in rows if row["variant_key"] == variant.value]
        assert len({row["input_data_hash"] for row in variant_rows}) == 2
        assert len({row["evidence_hash"] for row in variant_rows}) == 2


def test_incomplete_company_action_interval_is_archived_needs_review(
    repository: SQLiteRepository,
    tmp_path: Path,
) -> None:
    portfolio = replace_active_holdings(
        repository,
        (
            HoldingPositionInput(
                symbol="600919",
                name="江苏银行",
                entry_date=date(2026, 8, 3),
                stock_sleeve_weight=1.0,
                metadata={
                    "company_action_clear": True,
                    "company_action_evidence_id": "incomplete-evidence",
                    "company_action_evidence_source": "test_point_in_time_feed",
                    "company_action_clear_through": "2026-08-07",
                },
            ),
        ),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 3, 21, 0, tzinfo=UTC),
        change_id="shadow-incomplete-company-action",
    )

    summary = run_holding_stop_shadows(
        repository,
        dataset_root=tmp_path / "csmar",
        overlay_root=tmp_path / "overlay",
        as_of=date(2026, 8, 7),
        evaluated_at=datetime(2026, 8, 7, 22, 0, tzinfo=UTC),
        _history_loader=_loader(portfolio, _history()),
    )

    assert summary.observation_count == 2
    rows = repository.list_holding_shadow_events(data_cutoff="2026-08-07")
    assert {row["status"] for row in rows} == {"needs_review"}
    assert all(row["evaluation_eligible"] == 0 for row in rows)
    assert all(
        "company_action_interval_or_knowledge_coverage_incomplete" in row["reason_json"]
        for row in rows
    )
