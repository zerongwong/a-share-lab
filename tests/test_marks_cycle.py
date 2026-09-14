from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta

import pytest

from ashare_lab.analytics.marks_cycle import DIMENSIONS, assess_marks_cycle
from ashare_lab.services.marks_cycle_shadow import import_evidence, run_marks_cycle_shadow

NOW = datetime(2026, 9, 14, 12, tzinfo=UTC)
CUTOFF = date(2026, 9, 14)


class Repo:
    def __init__(self, path):
        self.path = path

    @contextmanager
    def connection(self):
        with sqlite3.connect(self.path) as connection:
            yield connection


def inputs(values=(50, 50, 50, 50), now=NOW):
    return [
        {
            "dimension": dimension,
            "value": value,
            "observed_through": now.date().isoformat(),
            "published_at": (now - timedelta(hours=2)).isoformat(),
            "retrieved_at": (now - timedelta(hours=1)).isoformat(),
            "publisher": "合成来源",
            "source_url": "https://example.org/synthetic",
            "metric_definition": "合成参考百分位/比例",
            "reference_window": "合成历史样本，不用于投资",
            "sample_size": 100,
        }
        for dimension, value in zip(DIMENSIONS, values, strict=True)
    ]


def evidence(values=(50, 50, 50, 50), now=NOW):
    return [
        {"id": str(i), "recorded_at": now.isoformat(), **obs}
        for i, obs in enumerate(inputs(values, now))
    ]


@pytest.mark.parametrize(
    "values,state,ceiling",
    [
        ((90, 90, 10, 90), "overheated_defense", 0.3),
        ((10, 10, 90, 10), "credit_stress_defense", 0.3),
        ((10, 10, 50, 50), "opportunity_watch", 0.5),
        ((50, 60, 30, 70), "recovery_offense", 0.8),
        ((50, 50, 50, 50), "mixed_baseline", 0.5),
    ],
)
def test_states_separate_cheapness_greed_and_credit_stress(values, state, ceiling):
    result = assess_marks_cycle(
        evidence(values), price_cutoff=CUTOFF, known_at=NOW, incumbent_cap=0.6
    )
    assert result["state"] == state
    assert result["shadow_cap"] == min(0.6, ceiling)
    assert result["minimum_shadow_cash"] == 1 - result["shadow_cap"]
    assert result["production_decision_input"] is False
    assert result["external_delivery_allowed"] is False
    assert result["confidence_probability"] is None


def test_missing_and_stale_evidence_is_unavailable_not_neutral():
    for observations in ([], evidence()[:3], evidence(now=NOW - timedelta(days=130))):
        result = assess_marks_cycle(
            observations, price_cutoff=CUTOFF, known_at=NOW, incumbent_cap=0.8
        )
        assert result["state"] == "data_not_ready"
        assert result["shadow_cap"] is None
        assert result["missing_dimensions"]


@pytest.mark.parametrize("field", ["published_at", "retrieved_at", "recorded_at"])
def test_future_knowledge_cannot_enter_a_historical_assessment(field):
    observations = evidence()
    observations[0][field] = (NOW + timedelta(seconds=1)).isoformat()
    result = assess_marks_cycle(observations, price_cutoff=CUTOFF, known_at=NOW, incumbent_cap=0.8)
    assert result["state"] == "data_not_ready"


def test_weekly_freeze_but_extreme_risk_can_tighten_immediately():
    previous = assess_marks_cycle(evidence(), price_cutoff=CUTOFF, known_at=NOW, incumbent_cap=0.8)
    tomorrow = NOW + timedelta(days=1)
    result = assess_marks_cycle(
        evidence((50, 60, 30, 70), tomorrow),
        price_cutoff=tomorrow.date(),
        known_at=tomorrow,
        incumbent_cap=0.8,
        previous=previous,
    )
    assert result["state"] == "mixed_baseline"
    result = assess_marks_cycle(
        evidence((90, 90, 30, 70), tomorrow),
        price_cutoff=tomorrow.date(),
        known_at=tomorrow,
        incumbent_cap=0.8,
        previous=previous,
    )
    assert result["state"] == "overheated_defense"


def test_shadow_never_increases_incumbent_risk_ceiling():
    result = assess_marks_cycle(
        evidence((50, 60, 30, 70)), price_cutoff=CUTOFF, known_at=NOW, incumbent_cap=0.2
    )
    assert result["shadow_cap"] == 0.2


def test_local_journal_is_immutable_and_import_time_is_not_backdated(tmp_path):
    repo = Repo(tmp_path / "test.db")
    saved = import_evidence(repo, inputs(), now=NOW)
    assert len(saved) == 4
    assert all(o["recorded_at"] == NOW.isoformat() for o in saved)
    result = run_marks_cycle_shadow(repo, price_cutoff=CUTOFF, known_at=NOW, incumbent_cap=0.8)
    assert result["state"] == "mixed_baseline"
    assert (
        run_marks_cycle_shadow(repo, price_cutoff=CUTOFF, known_at=NOW, incumbent_cap=0.8) == result
    )
    past = run_marks_cycle_shadow(
        repo, price_cutoff=CUTOFF, known_at=NOW - timedelta(seconds=1), incumbent_cap=0.8
    )
    assert past["state"] == "data_not_ready"
    with repo.connection() as conn:
        assert conn.execute("select count(*) from marks_cycle_runs").fetchone()[0] == 2
        for sql in ("DELETE FROM marks_cycle_runs", "UPDATE marks_cycle_evidence SET payload='{}'"):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(sql)


@pytest.mark.parametrize(
    "change",
    [
        {"value": True},
        {"value": float("nan")},
        {"value": 101},
        {"sample_size": 1},
        {"source_url": "http://example.org"},
        {"source_url": "https://user:secret@example.org"},
        {"published_at": (NOW + timedelta(days=1)).isoformat()},
        {"recorded_at": "2000-01-01"},
    ],
)
def test_reject_bad_evidence_without_partial_import(tmp_path, change):
    repo = Repo(tmp_path / "test.db")
    observations = inputs()
    observations[-1].update(change)
    with pytest.raises(ValueError):
        import_evidence(repo, observations, now=NOW)
