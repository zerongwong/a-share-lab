from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.analytics.cost_stop import observe_cost_stop
from ashare_lab.services import review_active_holdings as service
from ashare_lab.services.build_evening_digest import _holding_review_lines
from ashare_lab.services.holding_ledger import HoldingPositionInput, replace_active_holdings
from ashare_lab.services.review_active_holdings import (
    CompanyActionClearance,
    HoldingAction,
    HoldingReviewRowStatus,
)

ENTRY = date(2026, 8, 28)
CUTOFF = date(2026, 9, 1)


@pytest.fixture
def repository(tmp_path: Path, monkeypatch):
    from test_continuous_holding_stop import _candidate

    repo = SQLiteRepository(
        tmp_path / "cost-stop.db", Path(__file__).resolve().parents[1] / "migrations"
    )
    repo.initialize()
    _register(repo)
    monkeypatch.setattr(service, "_candidate_stop", lambda *_a, **_k: _candidate(5.0))
    monkeypatch.setattr(
        service,
        "_holding_action",
        lambda *_a, **_k: (
            HoldingAction.HOLD,
            ("strong_or_intact_holding_retained_without_rank_churn",),
        ),
    )
    return repo


def _register(repo, cost=10.0, entry=ENTRY, effective=ENTRY):
    return replace_active_holdings(
        repo,
        [
            HoldingPositionInput(
                symbol="600919",
                name="示例股票",
                entry_date=entry,
                cost_price=cost,
                stock_sleeve_weight=1.0,
            )
        ],
        holding_weeks=4,
        effective_at=datetime.combine(effective, datetime.min.time(), UTC),
    )


def _history(close=9.5, low=9.3, cutoff=CUTOFF):
    from test_review_active_holdings import _history as make_history

    frame = make_history(end=cutoff.isoformat())
    frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = [close, close + 0.1, low, close]
    return frame


def _review(
    repo,
    frame,
    *,
    cutoff=CUTOFF,
    clear=True,
    start=ENTRY,
    through=None,
    persist=True,
    verified=True,
):
    return service.review_active_holdings(
        repo,
        {} if frame is None else {"600919": frame},
        as_of=cutoff,
        verified_data_cutoff=cutoff,
        verified_close=verified,
        reviewed_at=datetime.combine(cutoff, datetime.max.time(), UTC),
        continuous_profile=True,
        persist=persist,
        company_action_clear_by_symbol={}
        if clear is None
        else {
            "600919": CompanyActionClearance(
                symbol="600919",
                through_date=through or cutoff,
                from_date=start,
                clear=clear,
                source="synthetic_test",
                evidence_id="synthetic-only",
            )
        },
    )


@pytest.mark.parametrize("price,triggered", [(9.21, False), (9.20, True), (9.19, True)])
def test_inclusive_cost_boundary_and_no_orders_or_ledger_change(repository, price, triggered):
    row = _review(repository, _history(price, price)).rows[0]
    assert (row.action is HoldingAction.EXIT) is triggered
    assert row.cost_stop == 9.2
    assert row.effective_stop == 9.2
    assert row.urgent is triggered
    assert row.holding_version == 1
    assert not row.auto_order_allowed and not row.replacement_requested
    assert len(repository.list_active_holdings()) == 1


def test_low_touch_and_rebound_still_alerts_and_survives_next_day(repository):
    first = _review(repository, _history(close=9.5, low=9.19)).rows[0]
    assert first.action is HoldingAction.EXIT
    assert first.close_below_stop is False
    assert first.cost_stop_touched_on == CUTOFF
    next_date = date(2026, 9, 2)
    # The recovered input no longer even contains the old breached bar.
    second = _review(repository, _history(10.1, 10.0, next_date), cutoff=next_date).rows[0]
    assert second.action is HoldingAction.EXIT
    assert second.cost_stop_touched_on == CUTOFF


def test_missing_data_does_not_silence_previously_triggered_alert(repository):
    _review(repository, _history(9.1, 9.0))
    failed = _review(repository, None).rows[0]
    assert failed.action is HoldingAction.REVIEW and failed.urgent
    assert "cost_stop_pending_breach:2026-09-01" in failed.reasons


def test_removing_cost_does_not_silence_a_previously_triggered_alert(repository):
    _review(repository, _history(9.1, 9.0))
    _register(repository, cost=None)
    row = _review(repository, _history(10.0, 9.9)).rows[0]
    assert row.action is HoldingAction.REVIEW and row.urgent


@pytest.mark.parametrize(
    "clear,start,through",
    [
        (None, ENTRY, CUTOFF),
        (False, ENTRY, CUTOFF),
        (True, None, CUTOFF),
        (True, date(2026, 8, 31), CUTOFF),
        (True, ENTRY, ENTRY),
    ],
)
def test_corporate_action_uncertainty_is_urgent_not_a_silent_hold(
    repository, clear, start, through
):
    summary = _review(repository, _history(9.1, 9.0), clear=clear, start=start, through=through)
    row = summary.rows[0]
    assert row.action is HoldingAction.REVIEW and row.urgent
    assert row.status is HoldingReviewRowStatus.DATA_NOT_READY
    assert repository.get_holding_protective_stop(row.position_key) is None
    text = "\n".join(_holding_review_lines(summary, name_bytes=36, reason_bytes=20))
    assert "8%成本止损" in text and "除权/分红待核验" in text
    assert "10.0" not in text  # Do not expose the raw cost.


@pytest.mark.parametrize("bad_technical", ["candidate", "short_history", "volume"])
def test_cost_guard_does_not_depend_on_technical_model_readiness(
    repository, monkeypatch, bad_technical
):
    history = _history(9.1, 9.0)
    if bad_technical == "candidate":

        def unavailable(*_a, **_k):
            raise ValueError("synthetic_atr_missing")

        monkeypatch.setattr(service, "_candidate_stop", unavailable)
    elif bad_technical == "short_history":
        history = history.tail(3)
    else:
        history.loc[history.index[-1], "amount_cny"] = -1
    summary = _review(repository, history)
    row = summary.rows[0]
    assert row.action is HoldingAction.EXIT and row.urgent
    assert row.status is HoldingReviewRowStatus.READY
    assert row.company_action_clear_from == ENTRY
    assert repository.get_holding_protective_stop(row.position_key)["effective_stop"] == 9.2
    replay = _review(repository, history).rows[0]
    assert replay.cost_stop_touched_on == row.cost_stop_touched_on
    assert len(repository.list_holding_reviews()) == 1


@pytest.mark.parametrize("bad_data", ["stale", "unverified", "adjusted", "invalid_price"])
def test_invalid_or_noncomparable_price_is_never_confirmed_as_exit(repository, bad_data):
    history = _history(9.1, 9.0)
    if bad_data == "stale":
        history = history.iloc[:-1]
    elif bad_data == "adjusted":
        history.attrs["adjustment"] = "qfq"
    elif bad_data == "invalid_price":
        history.loc[history.index[-1], "low"] = -1
    row = _review(repository, history, verified=bad_data != "unverified").rows[0]
    assert row.action is HoldingAction.REVIEW
    assert repository.get_holding_protective_stop(row.position_key) is None


def test_entry_day_low_is_not_assumed_to_be_after_purchase(repository):
    row = _review(repository, _history(9.5, 9.1, ENTRY), cutoff=ENTRY).rows[0]
    assert row.action is HoldingAction.HOLD
    assert row.cost_stop_touched_on is None


def test_future_touch_not_used(repository):
    history = _history()
    future = history.tail(1).copy()
    future["trade_date"] = pd.Timestamp("2026-09-02")
    future[["open", "high", "low", "close"]] = [8.0, 8.1, 7.9, 8.0]
    row = _review(repository, pd.concat([history, future], ignore_index=True)).rows[0]
    assert row.cost_stop_touched_on is None


def test_no_rounding_down_and_existing_tighter_stop_is_preserved(repository, monkeypatch):
    from test_continuous_holding_stop import _candidate

    _register(repository, cost=10.006)
    row = _review(repository, _history(close=9.205, low=9.205)).rows[0]
    assert row.cost_stop == pytest.approx(9.20552)
    assert row.action is HoldingAction.EXIT
    monkeypatch.setattr(service, "_candidate_stop", lambda *_a, **_k: _candidate(9.8))
    row = _review(repository, _history(close=10.0, low=9.9)).rows[0]
    assert row.effective_stop == 9.8
    assert row.action is HoldingAction.EXIT  # Already-triggered loss is latched.


def test_new_confirmed_position_does_not_inherit_old_exit(repository):
    first = _review(repository, _history(9.1, 9.0)).rows[0]
    new_date = date(2026, 9, 2)
    _register(repository, entry=new_date, effective=new_date)
    second = _review(repository, _history(10.1, 10.0, new_date), cutoff=new_date).rows[0]
    assert first.position_key != second.position_key
    assert second.action is HoldingAction.HOLD


def test_new_average_cost_does_not_rewrite_earlier_price_observations():
    daily = pd.DataFrame(
        [
            {"trade_date": ENTRY, "close": 9.5, "low": 9.1},
            {"trade_date": CUTOFF, "close": 11.0, "low": 9.5},
        ]
    )
    row = observe_cost_stop(
        daily, cost_price=11.0, entry_date=ENTRY, remembered_line=9.2, cost_update_date=CUTOFF
    )
    assert row.line == 10.12
    assert row.touched_on is None
    replay = observe_cost_stop(
        daily,
        cost_price=11.0,
        entry_date=ENTRY,
        remembered_line=row.line,
        remembered_history=row.line_history,
        cost_update_date=CUTOFF,
    )
    assert replay.touched_on is None
    assert replay.line_history == row.line_history


def test_average_down_does_not_loosen_old_cost_limit():
    daily = pd.DataFrame([{"trade_date": CUTOFF, "close": 9.1, "low": 9.0}])
    row = observe_cost_stop(daily, cost_price=9.0, entry_date=ENTRY, remembered_line=9.2)
    assert row.line == 9.2 and row.touched_on == CUTOFF


def test_compact_report_names_cost_stop_without_disclosing_cost(repository):
    summary = _review(repository, _history(9.1, 9.0))
    text = "\n".join(_holding_review_lines(summary, name_bytes=36, reason_bytes=90))
    assert "卖出建议" in text and "8%成本止损" in text and "保护线9.20" in text
    assert "10.00" not in text


def test_structural_exit_can_happen_before_cost_eight_percent(repository, monkeypatch):
    from test_continuous_holding_stop import _candidate

    monkeypatch.setattr(service, "_candidate_stop", lambda *_a, **_k: _candidate(9.8))
    monkeypatch.setattr(
        service,
        "_holding_action",
        lambda *_a, **k: (
            HoldingAction.EXIT if k["close_below_stop"] else HoldingAction.HOLD,
            ("complete_close_confirmed_below_effective_stop",),
        ),
    )
    row = _review(repository, _history(9.7, 9.6)).rows[0]
    assert row.action is HoldingAction.EXIT
    assert row.effective_stop == 9.8
    assert row.cost_stop_touched_on is None


def test_pure_cost_check_cannot_use_a_cost_increase_after_price_cutoff():
    daily = pd.DataFrame([{"trade_date": CUTOFF, "close": 9.5, "low": 9.3}])
    with pytest.raises(ValueError, match="cost_update_after_verified_price_cutoff"):
        observe_cost_stop(
            daily,
            cost_price=11.0,
            entry_date=ENTRY,
            remembered_line=9.2,
            cost_update_date=date(2026, 9, 2),
        )
