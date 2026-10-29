"""Synthetic observations distinguish statistical marks from executable trades."""

from __future__ import annotations

import copy
import sqlite3
from datetime import UTC, date, datetime

import pytest

from ashare_lab.services.recommendation_reference import (
    REFERENCE_METHOD,
    reference_summary,
    replay_reference,
)
from ashare_lab.services.recommendation_tracking import ingest_report, replay_tracking

PLAN, NEXT, MON, TUE = [date.fromisoformat(s) for s in (
    "2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13",
)]
SESSIONS = [PLAN, NEXT, MON, TUE]
SYMBOL = "600001"
RUN_AT = datetime(2026, 10, 14, 10, tzinfo=UTC)


@pytest.fixture
def connection():
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


def _report(conn, *, identifier="r1", plan=PLAN, published=None, empty=False, stop=9.3):
    entry = {
        "symbol": SYMBOL, "name": "合成样本", "entry_qualified": True,
        "account_weight": 0.15,
        "entry_rule": {
            "schema_version": "recommendation-entry-rule-v1",
            "kind": "healthy_pullback_range", "price_low": 9.9, "price_high": 10.1,
            "maximum_entry_price": 10.3, "structural_invalidation_price": stop,
        },
    }
    ingest_report(
        conn, report_id=identifier, plan_date=plan,
        published_at=datetime.fromisoformat(published or f"{plan}T08:30:00+08:00"),
        cutoff=date(2026, 9, 30), method_version="continuous-signal-v4",
        entries=[] if empty else [entry],
    )


def _bar(op=10.0, low=9.9, close=10.1, high=10.2, volume=1000):
    return {"open": op, "high": high, "low": low, "close": close, "volume_shares": volume}


def _inputs(through=TUE):
    return {
        "sessions": SESSIONS, "bars": {SYMBOL: {d: _bar() for d in SESSIONS}},
        "through": through, "run_at": RUN_AT,
        "clearances": {SYMBOL: {"complete": True, "status": "clear",
                                "coverage_from": "2026-09-30", "coverage_through": str(TUE),
                                "event_dates": []}},
    }


def test_open_reference_is_separate_from_v1_confirmation_then_next_open(connection):
    _report(connection)
    inputs = _inputs()
    inputs["bars"][SYMBOL][NEXT] = _bar(op=10.2, low=10.1, close=10.3, high=10.4)
    legacy = replay_tracking(connection, **inputs)
    legacy_rows = connection.execute("SELECT * FROM tracking_runs").fetchall()
    reference = replay_reference(connection, **inputs)
    row = reference["episodes"][0]
    assert legacy["episodes"][0]["entry_date"] == str(NEXT)
    assert row["entry_date"] == str(PLAN)
    assert row["entry_price"] == 10.0
    assert row["name"] == "合成样本"
    assert row["execution_eligible"] is False
    assert row["entry_condition_assumed"] is True
    assert reference["method_version"] == REFERENCE_METHOD
    assert connection.execute("SELECT * FROM tracking_runs").fetchall() == legacy_rows


def test_first_structural_exit_day_close_and_inclusive_days(connection):
    _report(connection)
    inputs = _inputs()
    inputs["bars"][SYMBOL][MON] = _bar(op=9.5, low=9.21, close=9.25, high=9.6)
    inputs["bars"][SYMBOL][TUE] = _bar(op=8.0, low=7.0, close=7.5, high=8.1)
    row = replay_reference(connection, **inputs)["episodes"][0]
    assert row["status"] == "closed"
    assert row["exit_trigger_date"] == row["exit_date"] == str(MON)
    assert row["exit_price"] == 9.25
    assert row["profit_per_share"] == -0.75
    assert row["trading_days"] == 3
    assert row["calendar_days"] == 5
    assert row["return_fraction"] == pytest.approx(-0.075)
    assert row["daily_average_return"] == pytest.approx(-0.025)
    assert row["daily_average_profit_per_share"] == -0.25


def test_intraday_cost_stop_latches_even_when_trigger_day_close_recovers(connection):
    _report(connection, stop=8.0)
    inputs = _inputs()
    inputs["bars"][SYMBOL][NEXT] = _bar(low=9.1, close=10.1)
    row = replay_reference(connection, **inputs)["episodes"][0]
    assert row["exit_date"] == str(NEXT)
    assert row["exit_price"] == 10.1
    assert row["reason"] == "cost_8pct_stop_signal_close_reference"
    assert row["return_fraction"] > 0
    assert row["execution_eligible"] is False


def test_same_day_stop_keeps_t1_warning_and_close_mark_loss_not_clamped(connection):
    _report(connection, stop=8.0)
    inputs = _inputs()
    inputs["bars"][SYMBOL][PLAN] = _bar(low=8.4, close=8.5)
    result = replay_reference(connection, **inputs)
    row = result["episodes"][0]
    assert row["exit_date"] == str(PLAN)
    assert row["trading_days"] == 1
    assert row["return_fraction"] == pytest.approx(-0.15)
    assert "same_day_exit_not_executable_under_t1" in row["execution_limitations"]
    assert result["diagnostics"]["losing_references"] == 1
    assert result["diagnostics"]["research_flags"]


@pytest.mark.parametrize("published", ["2026-10-08T09:30:00+08:00", "2026-10-08T14:00:00+08:00"])
def test_late_publication_is_not_a_retroactive_buy(connection, published):
    _report(connection, published=published)
    row = replay_reference(connection, **_inputs())["episodes"][0]
    assert row["status"] == "excluded_late_publication"
    assert row["entry_price"] is None
    assert row["daily_average_return"] is None


def test_report_unknown_at_run_time_is_not_retroactively_included(connection):
    _report(connection, published="2026-10-08T18:30:00+08:00")
    inputs = _inputs(through=PLAN)
    inputs["run_at"] = datetime.fromisoformat("2026-10-08T16:00:00+08:00")
    assert replay_reference(connection, **inputs)["reports"] == 0


@pytest.mark.parametrize("clock", ["2026-10-07T16:00:00+08:00", "2026-10-08T14:59:59+08:00"])
def test_future_or_intraday_cutoff_is_rejected_before_reference_archive_write(connection, clock):
    _report(connection)
    inputs = _inputs(through=PLAN)
    inputs["run_at"] = datetime.fromisoformat(clock)
    with pytest.raises(ValueError, match="completed session"):
        replay_reference(connection, **inputs)
    assert connection.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name='tracking_reference_runs'"
    ).fetchone()[0] == 0


def test_exact_completed_close_boundary_resolves_shanghai_timezone(connection):
    _report(connection)
    inputs = _inputs(through=PLAN)
    inputs["run_at"] = datetime(2026, 10, 8, 7, 0, tzinfo=UTC)
    result = replay_reference(connection, **inputs)
    assert result["episodes"][0]["last_valued_date"] == str(PLAN)


def test_repeated_exit_day_mentions_do_not_reset_then_later_recommendation_restarts(connection):
    _report(connection)
    _report(connection, identifier="r2", plan=MON)
    _report(connection, identifier="r3", plan=TUE)
    inputs = _inputs()
    inputs["bars"][SYMBOL][MON] = _bar(op=9.5, low=9.21, close=9.25, high=9.6)
    result = replay_reference(connection, **inputs)
    old, new = result["episodes"]
    assert old["report_ids"] == ["r1", "r2"]
    assert old["entry_date"] == str(PLAN)
    assert new["report_ids"] == ["r3"]
    assert new["entry_date"] == str(TUE)
    assert old["episode_id"] != new["episode_id"]


def test_open_outside_original_conditions_is_flagged_not_hidden(connection):
    _report(connection)
    inputs = _inputs()
    inputs["bars"][SYMBOL][PLAN] = _bar(op=10.5, low=10.4, close=10.6, high=10.7)
    row = replay_reference(connection, **inputs)["episodes"][0]
    assert row["entry_price"] == 10.5
    assert "open_above_archived_buy_cap" in row["entry_reference_limitations"]
    assert "open_outside_archived_pullback_range" in row["entry_reference_limitations"]
    assert row["execution_eligible"] is False


def test_zero_reports_and_untriggered_v1_remain_in_separate_denominators(connection):
    _report(connection)
    _report(connection, identifier="empty", plan=NEXT, empty=True)
    inputs = _inputs()
    inputs["bars"][SYMBOL][PLAN] = _bar(close=10.0)
    v1 = replay_tracking(connection, **inputs)
    ref = replay_reference(connection, **inputs)
    assert v1["episodes"][0]["status"] == "unfilled"
    assert ref["episodes"][0]["status"] == "open"
    assert ref["zero_entry_reports"] == v1["zero_entry_reports"] == 1
    assert ref["diagnostics"]["completed_references"] == 0


@pytest.mark.parametrize("defect,expected", [
    ("bar", "evidence_missing"), ("volume", "evidence_missing"),
    ("actions", "corporate_action_review_required"),
    ("detected", "corporate_action_review_required"),
])
def test_missing_evidence_never_becomes_a_completed_result(connection, defect, expected):
    _report(connection)
    inputs = _inputs()
    if defect == "bar":
        del inputs["bars"][SYMBOL][NEXT]
    elif defect == "volume":
        inputs["bars"][SYMBOL][NEXT]["volume_shares"] = 0
    elif defect == "actions":
        inputs["clearances"] = {}
    else:
        inputs["clearances"][SYMBOL].update(status="detected", event_dates=[str(NEXT)])
    result = replay_reference(connection, **inputs)
    assert result["episodes"][0]["status"] == expected
    assert result["episodes"][0]["return_fraction"] is None
    assert result["diagnostics"]["completed_references"] == 0


def test_idempotence_immutable_runs_repaired_evidence_and_real_holdings_untouched(connection):
    connection.execute("CREATE TABLE actual_holdings (symbol TEXT, shares INTEGER)")
    connection.execute("INSERT INTO actual_holdings VALUES ('600999', 123)")
    _report(connection)
    inputs = _inputs()
    first = replay_reference(connection, **inputs)
    replay_reference(connection, **inputs)
    assert connection.execute("SELECT COUNT(*) FROM tracking_reference_runs").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute("DELETE FROM tracking_reference_runs")
    changed = copy.deepcopy(inputs)
    changed["bars"][SYMBOL][NEXT] = _bar(low=9.1, close=9.2)
    second = replay_reference(connection, **changed)
    assert second["input_hash"] != first["input_hash"]
    assert connection.execute("SELECT COUNT(*) FROM tracking_reference_runs").fetchone()[0] == 2
    assert reference_summary(connection) == second
    assert connection.execute("SELECT * FROM actual_holdings").fetchall() == [("600999", 123)]
