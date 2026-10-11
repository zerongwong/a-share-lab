"""Publication order, not report IDs or ISO text order, freezes an episode."""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime

import pytest

from ashare_lab.services.recommendation_tracking import ingest_report, replay_tracking

PLAN = date(2026, 10, 8)
NEXT = date(2026, 10, 9)
SYMBOL = "600001"


def _report(connection, report_id, published_at, *, cap=10.5, plan=PLAN, cutoff=date(2026, 9, 30)):
    ingest_report(
        connection,
        report_id=report_id,
        plan_date=plan,
        published_at=datetime.fromisoformat(published_at),
        cutoff=cutoff,
        method_version="continuous-signal-v4",
        entries=[
            {
                "symbol": SYMBOL,
                "name": "合成排序样本",
                "entry_qualified": True,
                "account_weight": 0.1,
                "entry_rule": {
                    "schema_version": "recommendation-entry-rule-v1",
                    "kind": "reclaim_close_confirmation",
                    "trigger_price": 10.2,
                    "maximum_entry_price": cap,
                    "structural_invalidation_price": 9.7,
                },
            }
        ],
    )


def _replay(connection):
    return replay_tracking(
        connection,
        sessions=[PLAN, NEXT],
        bars={
            SYMBOL: {
                PLAN: {
                    "open": 10.0,
                    "high": 10.6,
                    "low": 9.9,
                    "close": 10.3,
                    "volume_shares": 1000.0,
                },
                NEXT: {
                    "open": 10.4,
                    "high": 10.7,
                    "low": 10.1,
                    "close": 10.6,
                    "volume_shares": 1000.0,
                },
            }
        },
        through=NEXT,
        clearances={
            SYMBOL: {
                "status": "clear",
                "complete": True,
                "coverage_from": "1990-01-01",
                "coverage_through": NEXT.isoformat(),
                "event_dates": [],
            }
        },
        run_at=datetime(2026, 10, 10, tzinfo=UTC),
    )


@pytest.mark.parametrize("reverse_insertion", [False, True])
def test_earlier_valid_same_day_report_survives_later_after_open_report(reverse_insertion):
    connection = sqlite3.connect(":memory:")
    try:
        reports = [
            ("z-earlier", "2026-10-08T08:40:00+08:00", 10.5),
            ("a-later", "2026-10-08T10:00:00+08:00", 10.25),
        ]
        for report_id, published_at, cap in reversed(reports) if reverse_insertion else reports:
            _report(connection, report_id, published_at, cap=cap)
        result = _replay(connection)
        assert result["mentions"] == 2
        assert len(result["episodes"]) == 1
        episode = result["episodes"][0]
        assert episode["report_ids"] == ["z-earlier", "a-later"]
        assert episode["entry_price"] == pytest.approx(10.4)
        assert episode["entry_date"] == NEXT.isoformat()
        assert episode["entry_rule"]["maximum_entry_price"] == 10.5
    finally:
        connection.close()


@pytest.mark.parametrize(
    "later_timestamp",
    [
        "2026-10-08T00:50:00+00:00",
        "2026-10-07T20:50:00-04:00",
        "2026-10-08T09:50:00+09:00",
    ],
)
def test_offsets_are_compared_as_instants_not_iso_timestamp_strings(later_timestamp):
    connection = sqlite3.connect(":memory:")
    try:
        # The smaller ID and (in two cases) smaller timestamp text are later.
        _report(connection, "a-later", later_timestamp, cap=10.25)
        _report(connection, "z-earlier", "2026-10-08T08:40:00+08:00")
        result = _replay(connection)
        assert len(result["episodes"]) == 1
        episode = result["episodes"][0]
        assert episode["report_ids"] == ["z-earlier", "a-later"]
        assert episode["entry_price"] == pytest.approx(10.4)
        assert episode["status"] == "open"
    finally:
        connection.close()


def test_equivalent_timestamp_offsets_use_report_id_only_as_tie_break():
    connection = sqlite3.connect(":memory:")
    try:
        _report(connection, "b-tied", "2026-10-08T08:40:00+08:00", cap=10.25)
        _report(connection, "a-tied", "2026-10-08T00:40:00+00:00")
        result = _replay(connection)
        assert result["mentions"] == 2
        assert len(result["episodes"]) == 1
        episode = result["episodes"][0]
        assert episode["report_ids"] == ["a-tied", "b-tied"]
        assert episode["entry_price"] == pytest.approx(10.4)
    finally:
        connection.close()


@pytest.mark.parametrize("old_status", ["evidence_missing", "corporate_action_review_required"])
def test_unentered_blocked_plan_does_not_swallow_later_independent_recommendation(old_status):
    connection = sqlite3.connect(":memory:")
    later_plan, later_entry = date(2026, 10, 12), date(2026, 10, 13)
    try:
        _report(connection, "old", "2026-10-08T08:40:00+08:00")
        _report(connection, "new", "2026-10-12T08:40:00+08:00", plan=later_plan, cutoff=NEXT)
        bar = {"open": 10.0, "high": 10.6, "low": 9.9, "close": 10.3, "volume_shares": 1000.0}
        history = {day: dict(bar) for day in [PLAN, NEXT, later_plan, later_entry]}
        history[later_entry].update(open=10.4, low=10.1, high=10.7, close=10.6)
        evidence = {
            "status": "clear",
            "complete": True,
            "coverage_from": "1990-01-01",
            "coverage_through": later_entry.isoformat(),
            "event_dates": [],
        }
        if old_status == "evidence_missing":
            del history[PLAN]
        else:
            # The old price basis is unresolved, but the new recommendation
            # was made from a cutoff after this historical company action.
            evidence.update(status="detected", event_dates=[PLAN.isoformat()])
        result = replay_tracking(
            connection,
            sessions=[PLAN, NEXT, later_plan, later_entry],
            bars={SYMBOL: history},
            through=later_entry,
            clearances={SYMBOL: evidence},
            run_at=datetime(2026, 10, 14, tzinfo=UTC),
        )
        assert result["mentions"] == 2
        assert len(result["episodes"]) == 2
        old, new = result["episodes"]
        assert old["status"] == old_status
        assert old["entry_price"] is None
        assert old["report_ids"] == ["old"]
        assert new["status"] == "open"
        assert new["entry_date"] == later_entry.isoformat()
        assert new["entry_price"] == pytest.approx(10.4)
        assert new["report_ids"] == ["new"]
    finally:
        connection.close()
