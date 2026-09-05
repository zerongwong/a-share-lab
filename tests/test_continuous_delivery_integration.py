"""Synthetic continuous report -> local archive integration; never sends data.

Uses the real repository methods and all project migrations against in-memory
SQLite. No user holding, credential, market-data file, or private NAV is read.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.cli.evening_digest import _archive_original_digest
from ashare_lab.services.archive_recommendation_report import build_recommendation_archive_bundle
from ashare_lab.services.build_evening_digest import (
    EveningDigestCandidate,
    EveningPeriodDigest,
    EveningResearchDigest,
    render_evening_digest_markdown,
)
from ashare_lab.services.continuous_strategy_journal import archive_continuous_decision
from ashare_lab.services.review_active_holdings import (
    HoldingAction,
    HoldingReviewRowStatus,
    HoldingReviewSummaryStatus,
    HoldingTreeReviewRow,
    HoldingTreeReviewSummary,
)

CUTOFF = date(2026, 9, 4)
PLAN_DATE = date(2026, 9, 7)


class _MemoryRepository(SQLiteRepository):
    def __init__(self):
        self.db = sqlite3.connect(":memory:", isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        migrations = Path(__file__).resolve().parents[1] / "migrations"
        for path in sorted(migrations.glob("[0-9][0-9][0-9]_*.sql")):
            self.db.executescript(path.read_text(encoding="utf-8"))

    def initialize(self):
        pass

    @contextmanager
    def connection(self):
        yield self.db


@pytest.fixture
def repository():
    repo = _MemoryRepository()
    yield repo
    repo.db.close()


def _legacy_digest():
    candidate = EveningDigestCandidate(
        rank=1,
        symbol="600099",
        name="旧观察样本",
        action="wait_confirmation",
        allocation_nature="observation_only",
        stock_sleeve_weight=1.0,
        account_weight=None,
        price_condition="观察突破≥10.00+量",
        price_nature="observation_only",
        evidence_pending=True,
        price_plan_evaluation_price=10.0,
        price_plan_cutoff=CUTOFF,
    )
    periods = tuple(
        EveningPeriodDigest(
            holding_weeks=weeks,
            holding_sessions=sessions,
            label=label,
            data_cutoff=CUTOFF,
            source_status="research_candidates",
            action_nature="仅观察",
            risk_nature="未合格",
            action_stock_exposure=0.0,
            action_cash_weight=1.0,
            candidates=(candidate,),
            performance_nature="observation_only",
        )
        for weeks, sessions, label in (
            (1, 5, "1周"),
            (2, 10, "2周"),
            (4, 20, "1个月"),
            (13, 60, "3个月"),
            (26, 120, "6个月"),
            (52, 252, "1年"),
        )
    )
    return EveningResearchDigest(
        common_cutoff=CUTOFF,
        decision_date=CUTOFF,
        plan_for_date=PLAN_DATE,
        cycle_label="中期下行｜短线修复反弹",
        entry_strictness="defensive",
        max_stock_exposure=0.3,
        minimum_cash_weight=0.7,
        cycle_rule_agreement=0.875,
        method_version="legacy-test-method",
        periods=periods,
    )


def _continuous_digest(*, holding_based=False):
    plan = {
        "mode": "continuous",
        "method_version": "continuous-signal-v1",
        "planned_exit_date": None,
        "signal_profile": "daily_weekly_v1",
        "risk_observation_sessions": 20,
        "holding_based": holding_based,
        "holding_identity": ["synthetic-private-ledger-id", 5] if holding_based else None,
        "entries": [
            {
                "symbol": f"60000{index}",
                "name": f"连续合成{index}",
                "account_weight": weight,
                "entry_qualified": True,
                "entry_label": "确认≥10.01，买≤10.30+量",
                "protection_line": 9.4812,
                "maximum_entry_price": 10.30,
            }
            for index, weight in enumerate((0.12, 0.06, 0.06, 0.06))
        ],
        "cash_weight": 0.7,
        "status_note": "合成补位预案，须确认卖出后再复核。" if holding_based else "合成初建方案。",
        "joint_evaluation": {"data_cutoff": "2026-09-04", "private_marker": "NOT_FOR_DELIVERY"},
    }
    # The transport may still carry its legacy analytical window. It must not
    # escape into a six-group display or a new fixed-maturity cohort.
    return replace(_legacy_digest(), method_version="continuous-signal-v1", continuous_plan=plan)


def _holding_review():
    row = HoldingTreeReviewRow(
        symbol="600088",
        name="合成私有持仓",
        holding_weeks=4,
        holding_version=5,
        position_key="synthetic-private-position-id",
        status=HoldingReviewRowStatus.READY,
        action=HoldingAction.EXIT,
        latest_close=9.0,
        cost_price=987654.32,
        stock_sleeve_weight=0.4,
        account_weight=0.12,
        candidate_stop=9.2,
        previous_stop=9.1,
        effective_stop=9.2,
        stop_raised=True,
        close_below_stop=True,
        source_timeframe="daily",
        evidence_date=CUTOFF,
        slow_direction="down",
        primary_structure="failed",
        daily_execution="failed",
        reasons=("complete_close_confirmed_below_effective_stop",),
    )
    return HoldingTreeReviewSummary(
        status=HoldingReviewSummaryStatus.READY,
        portfolio_id="synthetic-private-ledger-id",
        holding_version=5,
        holding_weeks=4,
        reviewed_at=datetime(2026, 9, 4, 13, tzinfo=UTC),
        data_cutoff=CUTOFF,
        rows=(row,),
    )


def _table_rows(repository, table):
    return [tuple(row) for row in repository.db.execute(f"SELECT * FROM {table} ORDER BY 1")]


def test_real_evening_renderer_outputs_one_account_weighted_plan_with_price_bounds():
    body = render_evening_digest_markdown(_continuous_digest())
    assert body.count("## 🩵 条件新买") == 1
    assert body.count("总资金12%") == 1
    assert body.count("总资金6%") == 3
    assert "计划现金：总资金70%" in body
    assert body.count("买≤10.30+量｜保护9.4812") == 4
    assert "仅信号退出，不设到期卖出" in body
    for forbidden in ("六期限", "1个月", "3个月", "旧观察样本", "股票仓内", "NOT_FOR_DELIVERY"):
        assert forbidden not in body


def test_holding_based_details_cannot_escape_without_holding_authorization():
    digest = _continuous_digest(holding_based=True)
    body = render_evening_digest_markdown(digest, _holding_review(), include_holding_summary=False)
    assert "持仓授权待核验" in body
    assert "未核定（不代表空仓）" in body
    for forbidden in (
        "合成私有持仓",
        "600088",
        "连续合成",
        "10.30",
        "9.4812",
        "总资金12%",
        "合成补位预案",
        "987654.32",
        "synthetic-private-ledger-id",
        "NOT_FOR_DELIVERY",
    ):
        assert forbidden not in body


def test_authorized_holdings_keep_exit_priority_but_remove_legacy_horizon_and_cost():
    body = render_evening_digest_markdown(
        _continuous_digest(holding_based=True), _holding_review(), include_holding_summary=True
    )
    assert "合成私有持仓(600088)" in body
    assert "卖出建议" in body
    assert "保护线9.20" in body
    assert body.index("卖出建议") < body.index("条件新买")
    assert "买≤10.30+量｜保护9.4812" in body
    for forbidden in ("1个月", "一个月", "六期限", "987654.32", "NOT_FOR_DELIVERY"):
        assert forbidden not in body


def test_cli_archives_continuous_journal_and_delivery_header_but_no_legacy_batches(repository):
    digest = _continuous_digest()
    first = _archive_original_digest(digest, repository)
    assert not first.batches
    assert not first.members
    assert repository.list_recommendation_batches(first.report_id) == []
    row = repository.db.execute(
        "SELECT payload_json FROM continuous_strategy_decisions WHERE decision_id = ?",
        (first.report_id,),
    ).fetchone()
    envelope = json.loads(row[0])
    assert envelope["payload"]["mode"] == "continuous"
    assert envelope["payload"]["planned_exit_date"] is None
    assert envelope["payload"]["plan_for_date"] == "2026-09-07"
    assert envelope["payload"]["entries"][0]["account_weight"] == 0.12
    assert envelope["record_nature"] == "decision_not_execution"
    assert envelope["external_delivery_allowed"] is False
    header_before = _table_rows(repository, "recommendation_reports")
    journal_before = _table_rows(repository, "continuous_strategy_decisions")
    second = _archive_original_digest(digest, repository)
    assert second.report_id == first.report_id
    assert second.content_hash == first.content_hash
    assert _table_rows(repository, "recommendation_reports") == header_before
    assert _table_rows(repository, "continuous_strategy_decisions") == journal_before
    assert _table_rows(repository, "continuous_strategy_valuations") == []


def test_new_journal_and_continuous_plan_do_not_rewrite_legacy_hashes_or_cohorts(repository):
    legacy = _legacy_digest()
    legacy_bundle = _archive_original_digest(legacy, repository)
    before = {
        table: _table_rows(repository, table)
        for table in ("recommendation_batches", "recommendation_members")
    }
    assert len(before["recommendation_batches"]) == 6
    assert len(before["recommendation_members"]) == 6
    header = repository.get_recommendation_report(legacy_bundle.report_id)
    _archive_original_digest(_continuous_digest(), repository)
    assert repository.get_recommendation_report(legacy_bundle.report_id) == header
    assert {table: _table_rows(repository, table) for table in before} == before
    assert build_recommendation_archive_bundle(legacy).content_hash == legacy_bundle.content_hash
    replay = _archive_original_digest(legacy, repository)
    assert replay.report_id == legacy_bundle.report_id
    assert replay.content_hash == legacy_bundle.content_hash


def test_different_continuous_payload_same_date_gets_different_content_address(repository):
    digest = _continuous_digest()
    first = _archive_original_digest(digest, repository)
    changed = {**digest.continuous_plan, "status_note": "独立的新决策，不覆盖原档。"}
    second = _archive_original_digest(replace(digest, continuous_plan=changed), repository)
    assert second.report_id != first.report_id
    assert second.content_hash != first.content_hash
    assert len(_table_rows(repository, "continuous_strategy_decisions")) == 2
    assert _table_rows(repository, "recommendation_batches") == []
    with pytest.raises(ValueError, match="different immutable content"):
        archive_continuous_decision(repository.db, first.report_id, CUTOFF, changed)


def test_undated_plan_or_nonfinite_continuous_content_never_archives_or_succeeds(repository):
    digest = _continuous_digest()
    with pytest.raises(ValueError, match="plan_for_date"):
        _archive_original_digest(replace(digest, plan_for_date=None), repository)
    invalid = {**digest.continuous_plan, "cash_weight": float("nan")}
    with pytest.raises(ValueError):
        _archive_original_digest(replace(digest, continuous_plan=invalid), repository)
    assert _table_rows(repository, "recommendation_reports") == []


def test_direct_continuous_bundle_cannot_create_fixed_maturity_batches():
    with pytest.raises(ValueError, match="fixed-maturity"):
        build_recommendation_archive_bundle(_continuous_digest())
