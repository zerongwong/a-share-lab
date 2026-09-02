from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.services.holding_ledger import HoldingPositionInput, replace_active_holdings
from ashare_lab.services.review_active_holdings import (
    CompanyActionClearance,
    HoldingAction,
    HoldingReviewRowStatus,
    HoldingReviewSummaryStatus,
    review_active_holdings,
)


@pytest.fixture
def repository(tmp_path: Path) -> SQLiteRepository:
    repo = SQLiteRepository(
        tmp_path / "research.db",
        Path(__file__).resolve().parents[1] / "migrations",
    )
    repo.initialize()
    return repo


def _history(*, periods: int = 900, end: str = "2026-08-28") -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=periods)
    index = np.arange(periods, dtype=float)
    close = 12.0 * np.exp(0.0008 * index + 0.025 * np.sin(index / 13.0))
    open_price = close * (1.0 - 0.002 * np.sin(index / 7.0))
    high = np.maximum(open_price, close) * 1.012
    low = np.minimum(open_price, close) * 0.988
    amount = np.full(periods, 80_000_000.0)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "amount_cny": amount,
        }
    )


def _set_one(repository: SQLiteRepository, *, holding_weeks: int) -> None:
    replace_active_holdings(
        repository,
        (
            HoldingPositionInput(
                symbol="600919",
                name="江苏银行",
                entry_date=date(2026, 8, 28),
                cost_price=None,
                stock_sleeve_weight=1.0,
                account_weight=None,
            ),
        ),
        holding_weeks=holding_weeks,
        effective_at=datetime(2026, 8, 28, 21, 0, tzinfo=UTC),
    )


def _clearance(
    through: str = "2026-08-28", *, clear: bool = True
) -> dict[str, CompanyActionClearance]:
    return {
        "600919": CompanyActionClearance(
            symbol="600919",
            through_date=date.fromisoformat(through),
            clear=clear,
            source="test_independent_company_action_feed",
            evidence_id=f"ca:600919:{through}:{clear}",
        )
    }


@pytest.mark.parametrize(
    ("holding_weeks", "expected_source"),
    (
        (1, "daily"),
        (2, "daily"),
        (4, "daily"),
        (13, "weekly_completed"),
        (26, "weekly_completed"),
        (52, "weekly_completed"),
    ),
)
def test_horizon_uses_contract_primary_timeframe_and_not_candidate_rank(
    repository: SQLiteRepository,
    holding_weeks: int,
    expected_source: str,
) -> None:
    _set_one(repository, holding_weeks=holding_weeks)

    result = review_active_holdings(
        repository,
        {"600919": _history()},
        as_of=date(2026, 8, 28),
        verified_data_cutoff=date(2026, 8, 28),
        reviewed_at=datetime(2026, 8, 28, 22, 0, tzinfo=UTC),
        company_action_clear_by_symbol=_clearance(),
    )

    assert result.status is HoldingReviewSummaryStatus.READY
    row = result.rows[0]
    assert row.status is HoldingReviewRowStatus.READY
    assert row.action is not HoldingAction.REVIEW
    assert row.source_timeframe == expected_source
    assert row.decision_layer == "holding_management"
    assert row.candidate_rank_used is False
    assert row.auto_order_allowed is False
    assert row.replacement_requested is False
    assert row.cost_price is None


def test_effective_stop_is_remembered_idempotent_and_never_moves_down(
    repository: SQLiteRepository,
) -> None:
    _set_one(repository, holding_weeks=4)
    history = _history(periods=320)

    first = review_active_holdings(
        repository,
        {"600919": history},
        as_of="2026-08-28",
        reviewed_at=datetime(2026, 8, 28, 22, 0, tzinfo=UTC),
        company_action_clear_by_symbol=_clearance(),
    )
    second = review_active_holdings(
        repository,
        {"600919": history},
        as_of="2026-08-28",
        reviewed_at=datetime(2026, 8, 28, 22, 5, tzinfo=UTC),
        company_action_clear_by_symbol=_clearance(),
    )

    assert second.rows[0].effective_stop == first.rows[0].effective_stop
    assert second.rows[0].action == first.rows[0].action
    assert len(repository.list_holding_reviews(data_cutoff="2026-08-28")) == 1
    stop = repository.get_holding_protective_stop(first.rows[0].position_key)
    assert stop is not None
    with repository.connection() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "UPDATE holding_protective_stops SET effective_stop = ? WHERE position_key = ?",
            (float(stop["effective_stop"]) - 0.01, first.rows[0].position_key),
        )


def test_out_of_order_replay_fails_closed_and_cannot_move_stop_cutoff_back(
    repository: SQLiteRepository,
) -> None:
    _set_one(repository, holding_weeks=4)
    later_history = _history(periods=320, end="2026-09-01")
    later = review_active_holdings(
        repository,
        {"600919": later_history},
        as_of="2026-09-01",
        company_action_clear_by_symbol=_clearance("2026-09-01"),
    )
    stored_before = repository.get_holding_protective_stop(later.rows[0].position_key)
    assert stored_before is not None
    assert stored_before["data_cutoff"] == "2026-09-01"

    replay = review_active_holdings(
        repository,
        {"600919": later_history},
        as_of="2026-08-28",
        company_action_clear_by_symbol=_clearance(),
    )

    row = replay.rows[0]
    assert replay.status is HoldingReviewSummaryStatus.DATA_NOT_READY
    assert row.action is HoldingAction.REVIEW
    assert row.effective_stop is None
    assert "protective_stop_cutoff_after_review_rejected" in row.reasons
    assert "future_protective_stop_not_used" in row.reasons
    assert repository.get_holding_protective_stop(row.position_key) == stored_before

    review_record = repository.list_holding_reviews(data_cutoff="2026-09-01")[0]
    with pytest.raises(ValueError, match="cutoff cannot move backwards"):
        repository.record_holding_review(
            review_record,
            stop_state={**stored_before, "data_cutoff": "2026-08-28"},
        )
    with (
        repository.connection() as connection,
        pytest.raises(
            sqlite3.IntegrityError,
            match="cutoff cannot move backwards",
        ),
    ):
        connection.execute(
            "UPDATE holding_protective_stops SET data_cutoff = ? WHERE position_key = ?",
            ("2026-08-28", row.position_key),
        )


def test_replay_uses_snapshot_effective_by_cutoff_not_newest_statement(
    repository: SQLiteRepository,
) -> None:
    _set_one(repository, holding_weeks=4)
    replace_active_holdings(
        repository,
        (
            HoldingPositionInput(
                symbol="601919",
                name="中远海控",
                entry_date=date(2026, 8, 31),
                stock_sleeve_weight=1.0,
            ),
        ),
        holding_weeks=13,
        effective_at=datetime(2026, 8, 31, 21, 0, tzinfo=UTC),
    )

    replay = review_active_holdings(
        repository,
        {"600919": _history(periods=320)},
        as_of="2026-08-28",
        persist=False,
        company_action_clear_by_symbol=_clearance(),
    )

    assert replay.holding_version == 1
    assert replay.holding_weeks == 4
    assert [row.symbol for row in replay.rows] == ["600919"]


def test_completed_close_below_remembered_stop_is_exit_for_next_session_only(
    repository: SQLiteRepository,
) -> None:
    _set_one(repository, holding_weeks=4)
    history = _history(periods=320)
    first = review_active_holdings(
        repository,
        {"600919": history},
        as_of="2026-08-28",
        persist=True,
        company_action_clear_by_symbol=_clearance(),
    )
    stop = first.rows[0].effective_stop
    assert stop is not None
    next_date = pd.Timestamp("2026-08-31")
    broken_close = stop * 0.90
    broken = pd.concat(
        [
            history,
            pd.DataFrame(
                {
                    "trade_date": [next_date],
                    "open": [broken_close * 1.01],
                    "high": [broken_close * 1.02],
                    "low": [broken_close * 0.98],
                    "close": [broken_close],
                    "amount_cny": [80_000_000.0],
                }
            ),
        ],
        ignore_index=True,
    )

    result = review_active_holdings(
        repository,
        {"600919": broken},
        as_of="2026-08-31",
        company_action_clear_by_symbol=_clearance("2026-08-31"),
    )

    row = result.rows[0]
    assert row.action is HoldingAction.EXIT
    assert row.close_below_stop is True
    assert row.next_session_only is True
    assert row.auto_order_allowed is False
    # A research EXIT does not mutate the user's explicit holding membership.
    assert repository.list_active_holdings()[0]["symbol"] == "600919"


def test_unverified_or_missing_close_fails_closed_without_moving_stop(
    repository: SQLiteRepository,
) -> None:
    _set_one(repository, holding_weeks=4)

    result = review_active_holdings(
        repository,
        {},
        as_of="2026-08-28",
        verified_close=False,
    )

    assert result.status is HoldingReviewSummaryStatus.DATA_NOT_READY
    assert result.rows[0].action is HoldingAction.REVIEW
    assert repository.get_holding_protective_stop(result.rows[0].position_key) is None


def test_future_prices_cannot_change_same_cutoff_review(repository: SQLiteRepository) -> None:
    _set_one(repository, holding_weeks=13)
    history = _history()
    future = history.copy()
    extra_dates = pd.bdate_range("2026-08-31", periods=5)
    future_rows = history.tail(5).copy()
    future_rows["trade_date"] = extra_dates
    future_rows[["open", "high", "low", "close"]] *= 100.0
    future = pd.concat([future, future_rows], ignore_index=True)

    expected = review_active_holdings(
        repository,
        {"600919": history},
        as_of="2026-08-28",
        persist=False,
        company_action_clear_by_symbol=_clearance(),
    ).rows[0]
    actual = review_active_holdings(
        repository,
        {"600919": future},
        as_of="2026-08-28",
        persist=False,
        company_action_clear_by_symbol=_clearance(),
    ).rows[0]

    assert actual == expected


def test_pre_entry_pivot_is_not_reused_as_a_new_trailing_ratchet(
    repository: SQLiteRepository,
) -> None:
    _set_one(repository, holding_weeks=4)
    history = _history(periods=320)
    first = review_active_holdings(
        repository,
        {"600919": history},
        as_of="2026-08-28",
        company_action_clear_by_symbol=_clearance(),
    ).rows[0]
    extra_dates = pd.bdate_range("2026-08-31", periods=2)
    last = float(history.iloc[-1]["close"])
    extra_close = np.array([last * 1.01, last * 1.02])
    extended = pd.concat(
        [
            history,
            pd.DataFrame(
                {
                    "trade_date": extra_dates,
                    "open": extra_close * 0.999,
                    "high": extra_close * 1.01,
                    "low": extra_close * 0.99,
                    "close": extra_close,
                    "amount_cny": np.full(2, 80_000_000.0),
                }
            ),
        ],
        ignore_index=True,
    )

    later = review_active_holdings(
        repository,
        {"600919": extended},
        as_of="2026-09-01",
        company_action_clear_by_symbol=_clearance("2026-09-01"),
    ).rows[0]

    assert later.candidate_stop == first.candidate_stop
    assert later.effective_stop == first.effective_stop
    stop = repository.get_holding_protective_stop(first.position_key)
    assert stop is not None
    assert stop["details_json"]["support_kind"] == "entry_cutoff_primary_structure_floor"


def test_missing_company_action_evidence_allows_hold_but_does_not_advance_stop(
    repository: SQLiteRepository,
) -> None:
    _set_one(repository, holding_weeks=4)

    result = review_active_holdings(
        repository,
        {"600919": _history(periods=320)},
        as_of="2026-08-28",
    )

    row = result.rows[0]
    assert row.status is HoldingReviewRowStatus.READY
    assert row.action is HoldingAction.HOLD
    assert row.candidate_stop is not None
    assert repository.get_holding_protective_stop(row.position_key) is None
    assert "candidate_stop_not_persisted_without_company_action_clearance" in row.reasons


def test_destructive_signal_without_clearance_becomes_urgent_review_then_clear_can_exit(
    repository: SQLiteRepository,
) -> None:
    _set_one(repository, holding_weeks=4)
    history = _history(periods=320)
    established = review_active_holdings(
        repository,
        {"600919": history},
        as_of="2026-08-28",
        company_action_clear_by_symbol=_clearance(),
    ).rows[0]
    assert established.effective_stop is not None
    stored_before = repository.get_holding_protective_stop(established.position_key)
    assert stored_before is not None
    broken_close = established.effective_stop * 0.90
    broken = pd.concat(
        [
            history,
            pd.DataFrame(
                {
                    "trade_date": [pd.Timestamp("2026-08-31")],
                    "open": [broken_close * 1.01],
                    "high": [broken_close * 1.02],
                    "low": [broken_close * 0.98],
                    "close": [broken_close],
                    "amount_cny": [80_000_000.0],
                }
            ),
        ],
        ignore_index=True,
    )

    blocked = review_active_holdings(
        repository,
        {"600919": broken},
        as_of="2026-08-31",
    ).rows[0]

    assert blocked.action is HoldingAction.REVIEW
    assert blocked.status is HoldingReviewRowStatus.DATA_NOT_READY
    assert blocked.urgent is True
    stored_blocked = repository.get_holding_protective_stop(blocked.position_key)
    assert stored_blocked is not None
    assert stored_blocked["effective_stop"] == stored_before["effective_stop"]

    cleared = review_active_holdings(
        repository,
        {"600919": broken},
        as_of="2026-08-31",
        company_action_clear_by_symbol=_clearance("2026-08-31"),
    ).rows[0]
    assert cleared.action is HoldingAction.EXIT
    assert cleared.status is HoldingReviewRowStatus.READY


def test_detected_company_action_blocks_even_an_apparent_hold(
    repository: SQLiteRepository,
) -> None:
    _set_one(repository, holding_weeks=4)

    row = review_active_holdings(
        repository,
        {"600919": _history(periods=320)},
        as_of="2026-08-28",
        company_action_clear_by_symbol=_clearance(clear=False),
    ).rows[0]

    assert row.action is HoldingAction.REVIEW
    assert row.status is HoldingReviewRowStatus.DATA_NOT_READY
    assert row.urgent is True
    assert repository.get_holding_protective_stop(row.position_key) is None
