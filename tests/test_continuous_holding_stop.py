from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.analytics import continuous_signals, medium_term_stage
from ashare_lab.services import review_active_holdings as service
from ashare_lab.services.review_active_holdings import (
    CompanyActionClearance,
    HoldingAction,
    HoldingReviewRowStatus,
)

ENTRY = date(2026, 8, 28)
CUTOFF = date(2026, 9, 1)


@pytest.fixture
def repository(tmp_path: Path):
    repository = SQLiteRepository(
        tmp_path / "continuous-stop.db", Path(__file__).resolve().parents[1] / "migrations"
    )
    repository.initialize()
    return repository


def _clearance(*, start=ENTRY, through=CUTOFF):
    return {
        "600919": CompanyActionClearance(
            symbol="600919",
            from_date=start,
            through_date=through,
            clear=True,
            source="synthetic_interval_clearance",
            evidence_id="same-evidence-revised-coverage",
        )
    }


def _candidate(stop):
    return service._CandidateStop(
        stop=stop,
        support=stop + 0.5,
        atr14=1.0,
        support_kind="synthetic_confirmed_reaction",
        source_timeframe="daily",
        evidence_date=ENTRY,
        atr_cutoff=ENTRY,
    )


def _review(repository, history, *, cutoff=CUTOFF, clearance=None, continuous=True, persist=True):
    return service.review_active_holdings(
        repository,
        {"600919": history},
        as_of=cutoff,
        verified_data_cutoff=cutoff,
        reviewed_at=datetime(2026, 9, 1, 22, tzinfo=UTC),
        persist=persist,
        company_action_clear_by_symbol=_clearance() if clearance is None else clearance,
        continuous_profile=continuous,
    )


def test_continuous_production_stop_profile_does_not_follow_legacy_holding_deadline(repository):
    from test_review_active_holdings import _history, _set_one

    history = _history(end=CUTOFF.isoformat())
    evidence = []
    for legacy_weeks in (1, 2, 4, 13, 26, 52):
        _set_one(repository, holding_weeks=legacy_weeks)
        row = _review(repository, history, persist=False).rows[0]
        assert row.status is HoldingReviewRowStatus.READY
        assert row.source_timeframe == "daily"
        assert row.holding_weeks == legacy_weeks  # Ledger is not rewritten.
        assert row.method_version.endswith("+continuous-v1")
        assert "signal_profile:continuous_daily_weekly_v1;no_expiry" in row.reasons
        evidence.append(
            (
                row.candidate_stop,
                row.effective_stop,
                row.action,
                row.primary_structure,
                row.daily_execution,
            )
        )
    assert all(item == evidence[0] for item in evidence)


def test_same_cutoff_profile_switch_never_lowers_previously_persisted_line(repository, monkeypatch):
    from test_review_active_holdings import _history, _set_one

    _set_one(repository, holding_weeks=13)
    history = _history(end=CUTOFF.isoformat())
    monkeypatch.setattr(service, "_candidate_stop", lambda *_a, **_k: _candidate(15.0))
    legacy = _review(repository, history, continuous=False).rows[0]
    assert legacy.effective_stop == 15.0
    monkeypatch.setattr(service, "_candidate_stop", lambda *_a, **_k: _candidate(10.0))
    continuous = _review(repository, history).rows[0]
    assert continuous.status is HoldingReviewRowStatus.READY
    assert continuous.candidate_stop == 10.0
    assert continuous.effective_stop == 15.0
    assert repository.get_holding_protective_stop(continuous.position_key)["effective_stop"] == 15.0
    replay = _review(repository, history).rows[0]
    assert replay.effective_stop == 15.0
    assert repository.get_holding_protective_stop(replay.position_key)["effective_stop"] == 15.0


@pytest.mark.parametrize("action", [HoldingAction.EXIT, HoldingAction.TIGHTEN])
@pytest.mark.parametrize("start", [None, date(2026, 8, 29)])
def test_incomplete_interval_blocks_exit_or_tighten_and_does_not_write_candidate_stop(
    repository, monkeypatch, action, start
):
    from test_review_active_holdings import _history, _set_one

    _set_one(repository, holding_weeks=4)
    history = _history(end=CUTOFF.isoformat())
    monkeypatch.setattr(service, "_candidate_stop", lambda *_a, **_k: _candidate(10.0))
    seed = _review(repository, history, cutoff=ENTRY).rows[0]
    before = repository.get_holding_protective_stop(seed.position_key)
    monkeypatch.setattr(service, "_candidate_stop", lambda *_a, **_k: _candidate(11.0))
    if action is HoldingAction.EXIT:
        history = history.copy()
        history.loc[history.index[-1], ["open", "high", "low", "close"]] = [9.0, 9.1, 8.9, 9.0]
    row = _review(repository, history, clearance=_clearance(start=start)).rows[0]
    assert row.action is HoldingAction.REVIEW
    assert row.status is HoldingReviewRowStatus.DATA_NOT_READY
    assert row.urgent
    assert any(
        reason.startswith(f"company_action_evidence_blocks_{action.value}:")
        for reason in row.reasons
    )
    assert repository.get_holding_protective_stop(seed.position_key) == before


@pytest.mark.parametrize("action", [HoldingAction.EXIT, HoldingAction.TIGHTEN])
def test_complete_start_and_end_coverage_allows_confirmed_action_and_stop_write(
    repository, monkeypatch, action
):
    from test_review_active_holdings import _history, _set_one

    _set_one(repository, holding_weeks=4)
    history = _history(end=CUTOFF.isoformat())
    monkeypatch.setattr(service, "_candidate_stop", lambda *_a, **_k: _candidate(10.0))
    seed = _review(repository, history, cutoff=ENTRY).rows[0]
    monkeypatch.setattr(service, "_candidate_stop", lambda *_a, **_k: _candidate(11.0))
    if action is HoldingAction.EXIT:
        history = history.copy()
        history.loc[history.index[-1], ["open", "high", "low", "close"]] = [9.0, 9.1, 8.9, 9.0]
    row = _review(repository, history).rows[0]
    assert row.action is action
    assert row.status is HoldingReviewRowStatus.READY
    assert row.company_action_clear_from == ENTRY
    assert row.company_action_clear_through == CUTOFF
    assert repository.get_holding_protective_stop(seed.position_key)["effective_stop"] == 11.0
    assert not row.auto_order_allowed
    assert not row.replacement_requested


def test_missing_coverage_end_does_not_allow_action_despite_valid_start(repository, monkeypatch):
    from test_review_active_holdings import _history, _set_one

    _set_one(repository, holding_weeks=4)
    history = _history(end=CUTOFF.isoformat())
    monkeypatch.setattr(service, "_candidate_stop", lambda *_a, **_k: _candidate(100.0))
    row = _review(repository, history, clearance=_clearance(through=ENTRY)).rows[0]
    assert row.action is HoldingAction.REVIEW
    assert repository.get_holding_protective_stop(row.position_key) is None


def test_maturity_or_candidate_rank_never_reapplies_initial_entry_gate_to_intact_holding(
    repository, monkeypatch
):
    from test_review_active_holdings import _history, _set_one

    _set_one(repository, holding_weeks=52)
    monkeypatch.setattr(
        continuous_signals,
        "assess_continuous_entry",
        lambda *_a, **_k: pytest.fail("new-entry gate must not manage old holdings"),
    )
    monkeypatch.setattr(
        medium_term_stage,
        "assess_medium_term_stage",
        lambda *_a, **_k: pytest.fail("maturity gate must not force a held position to exit"),
    )
    monkeypatch.setattr(service, "_candidate_stop", lambda *_a, **_k: _candidate(10.0))
    row = _review(repository, _history(end=CUTOFF.isoformat())).rows[0]
    assert row.action is HoldingAction.HOLD
    assert row.candidate_rank_used is False
    assert "daily_rank_decline_does_not_trigger_replacement" in row.reasons


def test_same_day_interval_evidence_upgrade_gets_distinct_immutable_review_record(
    repository, monkeypatch
):
    from test_review_active_holdings import _history, _set_one

    _set_one(repository, holding_weeks=4)
    history = _history(end=CUTOFF.isoformat())
    monkeypatch.setattr(service, "_candidate_stop", lambda *_a, **_k: _candidate(100.0))
    absent_start = _clearance(start=None)
    first = _review(repository, history, clearance=absent_start).rows[0]
    assert first.action is HoldingAction.REVIEW
    full = {"600919": replace(absent_start["600919"], from_date=ENTRY)}
    second = _review(repository, history, clearance=full).rows[0]
    assert second.action is HoldingAction.EXIT
    records = repository.list_holding_reviews(data_cutoff=CUTOFF)
    assert len(records) == 2
    assert {row["holding_action"] for row in records} == {"review", "exit"}
    assert len({row["evidence_hash"] for row in records}) == 2
