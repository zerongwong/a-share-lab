"""V5 cannot rewrite/absorb an active V4 recommendation of the same security."""

import sqlite3
from datetime import datetime

import pytest
from test_recommendation_tracking import (
    ENTRY,
    MONDAY,
    PLAN,
    SESSIONS,
    SHANGHAI,
    _bars,
    _clearances,
    _entry,
)

from ashare_lab.services.recommendation_reference import replay_reference
from ashare_lab.services.recommendation_tracking import ingest_report, replay_tracking


def _ingest(connection, version, plan, *, rule_version=None):
    entry = _entry()
    entry["entry_rule"]["signal_method_version"] = rule_version or version
    return ingest_report(
        connection, report_id=version, plan_date=plan,
        published_at=datetime.combine(plan, datetime.min.time(), tzinfo=SHANGHAI).replace(hour=8),
        cutoff=PLAN.replace(day=7), method_version=version, entries=[entry],
    )


@pytest.mark.parametrize("replay,table", [
    (replay_tracking, "tracking_runs"), (replay_reference, "tracking_reference_runs"),
])
def test_cross_method_same_symbol_is_separate_and_old_run_immutable(replay, table):
    connection = sqlite3.connect(":memory:")
    _ingest(connection, "continuous-signal-v4", PLAN)
    inputs = {"sessions": SESSIONS, "bars": _bars(), "through": MONDAY,
              "clearances": _clearances(), "run_at": datetime(2026, 10, 15, 16, tzinfo=SHANGHAI)}
    old = replay(connection, **inputs)
    stored = connection.execute(f"SELECT * FROM {table}").fetchall()
    assert replay(connection, **inputs) == old
    _ingest(connection, "continuous-signal-v5", ENTRY)
    new = replay(connection, **inputs)
    assert new["method_version"] != old["method_version"]
    assert len(new["episodes"]) == 2
    assert new["episodes"][0]["episode_id"] == old["episodes"][0]["episode_id"]
    assert [e["report_ids"] for e in new["episodes"]] == [
        ["continuous-signal-v4"], ["continuous-signal-v5"],
    ]
    assert {e["recommendation_method"] for e in new["episodes"]} == {
        "continuous-signal-v4", "continuous-signal-v5",
    }
    assert new["input_hash"] != old["input_hash"]
    assert connection.execute(f"SELECT * FROM {table}").fetchall()[:1] == stored
    assert replay(connection, **inputs) == new
    connection.close()


def test_outer_report_cannot_relabel_entry_method():
    connection = sqlite3.connect(":memory:")
    with pytest.raises(ValueError, match="conflicts"):
        _ingest(connection, "continuous-signal-v5", PLAN, rule_version="continuous-signal-v4")
    connection.close()
