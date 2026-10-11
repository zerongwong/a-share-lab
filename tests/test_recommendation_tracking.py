"""Behavioral acceptance tests for the independent recommendation shadow ledger.

The scenarios deliberately use invented prices and securities.  They verify
execution chronology and evidence handling, not a strategy's profitability.
"""

from __future__ import annotations

import copy
import sqlite3
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from ashare_lab.services.recommendation_tracking import (
    ensure_tracking_schema,
    ingest_report,
    replay_tracking,
    tracking_summary,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
PLAN = date(2026, 10, 8)
ENTRY = date(2026, 10, 9)
MONDAY = date(2026, 10, 12)
TUESDAY = date(2026, 10, 13)
WEDNESDAY = date(2026, 10, 14)
SESSIONS = [PLAN, ENTRY, MONDAY, TUESDAY, WEDNESDAY]
SYMBOL = "600001"


@pytest.fixture
def connection():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_tracking_schema(conn)
    yield conn
    conn.close()


def _bar(open=10.0, high=10.6, low=9.9, close=10.3, volume=1200.0, amount=12500.0):
    return {
        "open": open,
        "high": high,
        "low": low,
        "close": close,
        "volume_shares": volume,
        "amount_cny": amount,
    }


def _entry(*, symbol=SYMBOL, kind="healthy_pullback_range", stop=9.7):
    return {
        "symbol": symbol,
        "name": "测试股份",
        "entry_qualified": True,
        "account_weight": 0.2,
        "protection_line": stop,
        "entry_rule": {
            "schema_version": "recommendation-entry-rule-v1",
            "kind": kind,
            "data_cutoff": "2026-09-30",
            "price_low": 9.9,
            "price_high": 10.2,
            "trigger_price": 10.2,
            "maximum_entry_price": 10.5,
            "structural_invalidation_price": stop,
            "confirmation_activity_metric": "volume_shares",
            "confirmation_activity_min": 1000.0,
        },
    }


def _ingest(conn, *, report_id="report-1", plan=PLAN, entries=None, hour=8, minute=40):
    return ingest_report(
        conn,
        report_id=report_id,
        plan_date=plan,
        published_at=datetime(plan.year, plan.month, plan.day, hour, minute, tzinfo=SHANGHAI),
        cutoff=date(2026, 9, 30),
        method_version="continuous-signal-v4",
        entries=[_entry()] if entries is None else entries,
    )


def _bars():
    return {
        SYMBOL: {
            PLAN: _bar(),
            ENTRY: _bar(open=10.4, high=10.7, low=10.1, close=10.6),
            MONDAY: _bar(open=10.6, high=10.9, low=10.3, close=10.8),
            TUESDAY: _bar(open=10.8, high=11.0, low=10.5, close=10.9),
            WEDNESDAY: _bar(open=10.9, high=11.1, low=10.7, close=11.0),
        }
    }


def _clearances():
    return {
        SYMBOL: {
            "status": "clear",
            "complete": True,
            "coverage_from": "2026-09-30",
            "coverage_through": "2026-11-30",
            "event_dates": [],
        }
    }


def _replay(conn, *, through=MONDAY, bars=None, clearances=None):
    return replay_tracking(
        conn,
        sessions=SESSIONS,
        bars=_bars() if bars is None else bars,
        through=through,
        clearances=_clearances() if clearances is None else clearances,
        run_at=datetime(2026, 10, 15, 16, tzinfo=SHANGHAI),
    )


def _episode(conn):
    episodes = tracking_summary(conn)["episodes"]
    assert len(episodes) == 1
    return episodes[0]


def test_missing_volume_is_unknown_not_an_untriggered_signal(connection):
    _ingest(connection)
    bars = _bars()
    del bars[SYMBOL][PLAN]["volume_shares"]
    _replay(connection, bars=bars)
    row = _episode(connection)
    assert row["status"] == "evidence_missing"
    assert row["entry_price"] is None


def test_schema_is_additive_and_does_not_mutate_legacy_tables():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE user_holdings (name TEXT)")
    conn.execute("INSERT INTO user_holdings VALUES ('retained')")
    ensure_tracking_schema(conn)
    ensure_tracking_schema(conn)
    tables = {
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert all(name == "user_holdings" or name.startswith("tracking_") for name in tables)
    assert conn.execute("SELECT name FROM user_holdings").fetchall() == [("retained",)]
    conn.close()


def test_zero_recommendation_report_is_still_archived(connection):
    _ingest(connection, entries=[])
    summary = tracking_summary(connection)
    assert summary["reports"] == 1
    assert summary["mentions"] == 0
    assert summary["episodes"] == []


def test_exact_report_retry_is_idempotent_and_different_content_cannot_overwrite(connection):
    _ingest(connection)
    _ingest(connection)
    assert tracking_summary(connection)["reports"] == 1
    assert tracking_summary(connection)["mentions"] == 1
    changed = _entry(stop=9.8)
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        _ingest(connection, entries=[changed])
    assert tracking_summary(connection)["reports"] == 1


def test_retest_confirmation_uses_next_session_open_not_same_day_low(connection):
    _ingest(connection)
    _replay(connection, through=ENTRY)
    row = _episode(connection)
    assert row["entry_date"] == ENTRY.isoformat()
    assert row["entry_price"] == pytest.approx(10.4)
    assert row["entry_price"] != 9.9
    assert row["exit_date"] is None
    assert row["exit_price"] is None


@pytest.mark.parametrize(
    "plan_bar",
    [
        _bar(open=10.4, high=10.7, low=10.3, close=10.5),  # Never retests the range.
        _bar(open=10.4, close=10.3),  # Retests, but has not turned upward.
        _bar(open=10.0, close=10.1),  # Still below the frozen range ceiling.
    ],
)
def test_unconfirmed_pullback_does_not_create_a_trade(connection, plan_bar):
    _ingest(connection)
    bars = _bars()
    bars[SYMBOL][PLAN] = plan_bar
    _replay(connection, bars=bars)
    row = _episode(connection)
    assert row["status"] == "unfilled"
    assert row["entry_price"] is None
    assert row["return_fraction"] is None


def test_late_publication_cannot_backfill_plan_day_confirmation(connection):
    _ingest(connection, hour=10)
    _replay(connection)
    row = _episode(connection)
    assert row["entry_date"] is None
    assert row["entry_price"] is None
    assert row["return_fraction"] is None


@pytest.mark.parametrize(
    "open_price,low,high,close",
    [(10.6, 10.5, 10.9, 10.8), (9.7, 9.6, 10.0, 9.9), (9.6, 9.5, 9.9, 9.8)],
)
def test_frozen_price_cap_and_structure_line_bound_simulated_entry(
    connection, open_price, low, high, close
):
    _ingest(connection)
    bars = _bars()
    bars[SYMBOL][ENTRY] = _bar(open=open_price, low=low, high=high, close=close)
    _replay(connection, bars=bars)
    row = _episode(connection)
    assert row["entry_price"] is None
    assert row["return_fraction"] is None


def test_distant_structural_line_does_not_delay_eight_percent_cost_stop(connection):
    _ingest(connection, entries=[_entry(stop=9.0)])
    bars = _bars()
    bars[SYMBOL][MONDAY] = _bar(open=10.5, high=10.8, low=9.5, close=10.6)
    _replay(connection, bars=bars)
    row = _episode(connection)
    assert row["entry_price"] == pytest.approx(10.4)
    assert row["exit_price"] == pytest.approx(10.4 * 0.92)
    assert row["return_fraction"] == pytest.approx(-0.08)


@pytest.mark.parametrize("volume,high,low", [(0.0, 10.7, 10.1), (2000.0, 10.4, 10.4)])
def test_no_volume_or_locked_one_price_bar_cannot_fill_entry(connection, volume, high, low):
    _ingest(connection)
    bars = _bars()
    bars[SYMBOL][ENTRY] = _bar(open=10.4, high=high, low=low, close=10.4, volume=volume)
    _replay(connection, bars=bars)
    row = _episode(connection)
    assert row["entry_price"] is None
    assert row["return_fraction"] is None


@pytest.mark.parametrize("activity,should_enter", [(999.0, False), (1000.0, True)])
def test_volume_breakout_replays_archived_activity_threshold(connection, activity, should_enter):
    _ingest(connection, entries=[_entry(kind="volume_breakout_close_confirmation")])
    bars = _bars()
    bars[SYMBOL][PLAN] = _bar(volume=activity)
    _replay(connection, bars=bars)
    row = _episode(connection)
    assert (row["entry_price"] is not None) is should_enter


def test_repeated_recommendations_share_open_episode_and_preserve_cost(connection):
    _ingest(connection)
    _replay(connection, through=ENTRY)
    original = _episode(connection)
    _ingest(connection, report_id="report-2", plan=MONDAY)
    _replay(connection, through=TUESDAY)
    row = _episode(connection)
    assert row["episode_id"] == original["episode_id"]
    assert row["entry_date"] == ENTRY.isoformat()
    assert row["entry_price"] == pytest.approx(10.4)
    assert set(row["report_ids"]) == {"report-1", "report-2"}
    assert tracking_summary(connection)["mentions"] == 2


def test_new_published_plan_can_start_new_episode_after_prior_plan_was_unfilled(connection):
    _ingest(connection)
    bars = _bars()
    bars[SYMBOL][PLAN] = _bar(open=10.4, high=10.7, low=10.3, close=10.5)
    _replay(connection, through=ENTRY, bars=bars)
    original = _episode(connection)
    assert original["status"] == "unfilled"
    _ingest(connection, report_id="report-2", plan=MONDAY)
    bars[SYMBOL][MONDAY] = _bar()
    bars[SYMBOL][TUESDAY] = _bar(open=10.4, high=10.7, low=10.1, close=10.6)
    _replay(connection, through=TUESDAY, bars=bars)
    rows = tracking_summary(connection)["episodes"]
    assert len(rows) == 2
    assert len({row["episode_id"] for row in rows}) == 2
    filled = [row for row in rows if row["entry_price"] is not None]
    assert len(filled) == 1
    assert filled[0]["entry_date"] == TUESDAY.isoformat()
    assert original["episode_id"] != filled[0]["episode_id"]


def test_old_missing_rule_does_not_swallow_new_complete_recommendation(connection):
    historical = _entry()
    historical["entry_rule"] = {}
    _ingest(connection, entries=[historical])
    _replay(connection, through=ENTRY)
    assert _episode(connection)["status"] == "rule_missing"
    _ingest(connection, report_id="complete-new-plan", plan=MONDAY)
    bars = _bars()
    bars[SYMBOL][MONDAY] = _bar()
    bars[SYMBOL][TUESDAY] = _bar(open=10.4, high=10.7, low=10.1, close=10.6)
    result = _replay(connection, through=TUESDAY, bars=bars)
    assert len(result["episodes"]) == 2
    old, new = result["episodes"]
    assert old["status"] == "rule_missing"
    assert old["entry_price"] is None
    assert old["report_ids"] == ["report-1"]
    assert new["report_ids"] == ["complete-new-plan"]
    assert new["entry_date"] == TUESDAY.isoformat()
    assert new["entry_price"] == pytest.approx(10.4)


def test_expired_old_next_open_cap_does_not_swallow_same_day_new_plan(connection):
    _ingest(connection)
    revised = _entry(stop=10.0)
    revised["entry_rule"].update(price_low=10.4, price_high=10.6, maximum_entry_price=11.0)
    _ingest(connection, report_id="new-higher-range", plan=ENTRY, entries=[revised])
    bars = _bars()
    bars[SYMBOL][ENTRY] = _bar(open=10.6, high=10.9, low=10.4, close=10.7)
    bars[SYMBOL][MONDAY] = _bar(open=10.8, high=11.0, low=10.6, close=10.9)
    result = _replay(connection, bars=bars)
    assert len(result["episodes"]) == 2
    old, new = result["episodes"]
    assert old["status"] == "unfilled"
    assert old["entry_price"] is None
    assert old["report_ids"] == ["report-1"]
    assert new["report_ids"] == ["new-higher-range"]
    assert new["entry_date"] == MONDAY.isoformat()
    assert new["entry_price"] == pytest.approx(10.8)


def test_old_pending_exit_fills_before_same_day_new_recommendation_is_linked(connection):
    _ingest(connection)
    replacement = _entry(stop=9.5)
    replacement["entry_rule"].update(price_low=9.7, price_high=10.0, maximum_entry_price=10.3)
    _ingest(connection, report_id="fresh-after-exit", plan=TUESDAY, entries=[replacement])
    bars = _bars()
    bars[SYMBOL][MONDAY] = _bar(open=10.3, high=10.4, low=9.6, close=9.65)
    bars[SYMBOL][TUESDAY] = _bar(open=9.8, high=10.3, low=9.7, close=10.1)
    bars[SYMBOL][WEDNESDAY] = _bar(open=10.2, high=10.5, low=10.0, close=10.4)
    result = _replay(connection, through=WEDNESDAY, bars=bars)
    assert len(result["episodes"]) == 2
    old, new = result["episodes"]
    assert old["status"] == "closed"
    assert old["exit_date"] == TUESDAY.isoformat()
    assert old["exit_price"] == pytest.approx(9.8)
    assert old["report_ids"] == ["report-1"]
    assert new["report_ids"] == ["fresh-after-exit"]
    assert new["entry_date"] == WEDNESDAY.isoformat()
    assert new["entry_price"] == pytest.approx(10.2)


def test_touch_of_eight_percent_line_is_not_erased_by_close_recovery(connection):
    _ingest(connection)
    bars = _bars()
    bars[SYMBOL][MONDAY] = _bar(open=10.5, high=10.8, low=9.5, close=10.6)
    _replay(connection, bars=bars)
    row = _episode(connection)
    assert row["entry_price"] == pytest.approx(10.4)
    assert row["exit_date"] == MONDAY.isoformat()
    assert row["exit_price"] == pytest.approx(10.4 * 0.92)
    assert row["return_fraction"] == pytest.approx(-0.08)


def test_gap_through_hard_stop_uses_worse_open_and_can_lose_more_than_eight_percent(connection):
    _ingest(connection)
    bars = _bars()
    bars[SYMBOL][MONDAY] = _bar(open=9.3, high=9.6, low=9.2, close=9.4)
    _replay(connection, bars=bars)
    row = _episode(connection)
    assert row["exit_price"] == pytest.approx(9.3)
    assert row["return_fraction"] == pytest.approx(9.3 / 10.4 - 1)
    assert row["return_fraction"] < -0.08


def test_entry_day_hard_stop_waits_for_next_tradable_session_due_to_t_plus_one(connection):
    _ingest(connection)
    bars = _bars()
    bars[SYMBOL][ENTRY] = _bar(open=10.4, high=10.7, low=9.4, close=10.5)
    bars[SYMBOL][MONDAY] = _bar(open=9.8, high=10.0, low=9.7, close=9.9)
    _replay(connection, through=ENTRY, bars=bars)
    assert _episode(connection)["exit_date"] is None
    _replay(connection, through=MONDAY, bars=bars)
    row = _episode(connection)
    assert row["exit_date"] == MONDAY.isoformat()
    assert row["exit_price"] == pytest.approx(9.8)


def test_structure_close_breach_exits_next_open_without_waiting_for_rebound(connection):
    _ingest(connection)
    bars = _bars()
    bars[SYMBOL][MONDAY] = _bar(open=10.3, high=10.4, low=9.6, close=9.65)
    bars[SYMBOL][TUESDAY] = _bar(open=9.8, high=10.0, low=9.7, close=9.9)
    _replay(connection, through=MONDAY, bars=bars)
    assert _episode(connection)["exit_date"] is None
    _replay(connection, through=TUESDAY, bars=bars)
    row = _episode(connection)
    assert row["exit_date"] == TUESDAY.isoformat()
    assert row["exit_price"] == pytest.approx(9.8)


def test_locked_price_bar_delays_triggered_exit_until_a_tradable_session(connection):
    _ingest(connection)
    bars = _bars()
    bars[SYMBOL][MONDAY] = _bar(open=9.3, high=9.3, low=9.3, close=9.3)
    bars[SYMBOL][TUESDAY] = _bar(open=9.2, high=9.4, low=9.0, close=9.3)
    _replay(connection, through=MONDAY, bars=bars)
    assert _episode(connection)["exit_date"] is None
    _replay(connection, through=TUESDAY, bars=bars)
    row = _episode(connection)
    assert row["exit_date"] == TUESDAY.isoformat()
    assert row["exit_price"] == pytest.approx(9.2)


def test_missing_bar_blocks_later_execution_and_is_not_inferred_to_be_suspension(connection):
    _ingest(connection)
    bars = _bars()
    del bars[SYMBOL][MONDAY]
    bars[SYMBOL][TUESDAY] = _bar(open=9.3, high=9.6, low=9.2, close=9.4)
    _replay(connection, through=TUESDAY, bars=bars)
    row = _episode(connection)
    assert row["status"] == "evidence_missing"
    assert row["exit_date"] is None
    assert row["exit_price"] is None
    assert row["return_fraction"] is None


@pytest.mark.parametrize("evidence_kind", ["missing", "incomplete", "detected"])
def test_company_action_uncertainty_does_not_become_a_verified_return(connection, evidence_kind):
    _ingest(connection)
    clearances = _clearances()
    if evidence_kind == "missing":
        clearances = {}
    elif evidence_kind == "incomplete":
        clearances[SYMBOL]["complete"] = False
    else:
        clearances[SYMBOL]["status"] = "detected"
        clearances[SYMBOL]["event_dates"] = [MONDAY.isoformat()]
    _replay(connection, clearances=clearances)
    row = _episode(connection)
    assert row["status"] == "corporate_action_review_required"
    assert row["return_fraction"] is None
    assert row["profit_per_share"] is None
    assert row["daily_compound_return"] is None


def test_replay_is_idempotent_and_future_bars_cannot_change_past_results(connection):
    _ingest(connection)
    _replay(connection, through=ENTRY)
    first = copy.deepcopy(_episode(connection))
    bars = _bars()
    bars[SYMBOL][MONDAY] = _bar(open=1.0, high=2.0, low=0.5, close=1.0)
    _replay(connection, through=ENTRY, bars=bars)
    repeated = _episode(connection)
    assert repeated == first
    assert tracking_summary(connection)["reports"] == 1
    assert tracking_summary(connection)["mentions"] == 1


def test_exited_trade_reports_per_share_profit_duration_and_both_daily_return_definitions(connection):
    _ingest(connection)
    bars = _bars()
    bars[SYMBOL][MONDAY] = _bar(open=10.5, high=10.8, low=9.5, close=10.6)
    _replay(connection, bars=bars)
    row = _episode(connection)
    result = 0.92 - 1
    assert row["profit_per_share"] == pytest.approx(10.4 * result)
    assert row["trading_days"] == 2  # Friday entry and Monday exit; weekend is excluded.
    assert row["calendar_days"] == 4  # Inclusive exposure dates, separately labelled.
    assert row["daily_average_return"] == pytest.approx(result / 2)
    assert row["daily_compound_return"] == pytest.approx((1 + result) ** 0.5 - 1)


@pytest.mark.parametrize("table", ["tracking_reports", "tracking_runs"])
@pytest.mark.parametrize("operation", ["UPDATE", "DELETE", "INSERT OR REPLACE"])
def test_recommendation_and_run_records_reject_mutation(connection, table, operation):
    _ingest(connection)
    _replay(connection)
    if operation == "UPDATE":
        column = "plan_date" if table == "tracking_reports" else "through_date"
        statement = f"UPDATE {table} SET {column} = '2026-10-10'"
    elif operation == "DELETE":
        statement = f"DELETE FROM {table}"
    else:
        statement = f"INSERT OR REPLACE INTO {table} SELECT * FROM {table}"
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(statement)


def test_confirmed_weekly_pivot_ratchets_stop_without_reapplying_original_entry_cap(connection):
    _ingest(connection)
    last = date(2026, 11, 20)
    days = [
        PLAN + timedelta(days=index)
        for index in range((last - PLAN).days + 1)
        if (PLAN + timedelta(days=index)).weekday() < 5
    ]
    bars = _bars()
    # Completed weekly lows: 9.9, 12, 11.5, 11, 12, 13, 14. The Oct 30 low
    # is confirmed only after the Nov 13 close. It raises protection above the
    # old 10.50 entry ceiling, which is irrelevant to an already-open position.
    week_lows = {42: 12.0, 43: 11.5, 44: 11.0, 45: 12.0, 46: 13.0, 47: 14.0}
    for day in days:
        if day <= ENTRY:
            continue
        low = week_lows[day.isocalendar().week]
        bars[SYMBOL][day] = _bar(open=low + 0.2, high=low + 0.5, low=low, close=low + 0.3)
    result = replay_tracking(
        connection,
        sessions=days,
        bars=bars,
        through=last,
        clearances=_clearances(),
        run_at=datetime(2026, 11, 21, 16, tzinfo=SHANGHAI),
    )
    row = result["episodes"][0]
    raises = [event for event in row["events"] if event["kind"] == "raise_structural_stop_next_session"]
    assert raises
    assert raises[0]["date"] >= "2026-11-13"
    assert row["stop_price"] > 10.5
    assert row["status"] == "open"
    assert row["entry_price"] == pytest.approx(10.4)
    assert row["exit_price"] is None
    assert row["last_valued_date"] == last.isoformat()
