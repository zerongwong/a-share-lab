from types import SimpleNamespace

import pandas as pd
import pytest

from ashare_lab.analytics.continuous_signals import (
    CONTINUOUS_METHOD_VERSION,
    CONTINUOUS_SIGNAL_CONTRACT,
    assess_continuous_entry,
    continuous_entry_stage_rank_score,
)
from ashare_lab.analytics.medium_term_stage import MediumTermStage, assess_medium_term_stage
from ashare_lab.analytics.multi_timeframe import ExecutionState, StructureState, horizon_contract


def _admit(
    *,
    stage=MediumTermStage.EARLY_UPTREND,
    frozen=False,
    structure=StructureState.BREAKOUT,
    execution=ExecutionState.READY_BREAKOUT,
    candidate_qualified=True,
    execution_ready=True,
    stage_metrics=None,
):
    metrics = {
        "distance_ma20": 0.0,
        "distance_ma60": 0.0,
        "return_60": 0.0,
        "return_120": 0.0,
    }
    metrics.update(stage_metrics or {})
    return assess_continuous_entry(
        SimpleNamespace(stage=stage, hard_freeze_new_entry=frozen, **metrics),
        SimpleNamespace(
            candidate_qualified=candidate_qualified,
            execution_ready=execution_ready,
            structure=SimpleNamespace(state=structure),
            execution=SimpleNamespace(state=execution),
        ),
    )


@pytest.mark.parametrize(
    "structure,execution",
    [
        (StructureState.BREAKOUT, ExecutionState.READY_BREAKOUT),
        (StructureState.HEALTHY_PULLBACK, ExecutionState.READY_PULLBACK),
    ],
)
@pytest.mark.parametrize(
    "stage",
    [
        MediumTermStage.RANGE,
        MediumTermStage.EARLY_UPTREND,
        MediumTermStage.ORDERLY_UPTREND,
    ],
)
def test_confirmed_non_extended_breakout_and_healthy_retest_can_enter(
    structure, execution, stage
):
    admission = _admit(stage=stage, structure=structure, execution=execution)
    assert admission.qualified
    assert admission.reasons == ()
    assert admission.method_version == CONTINUOUS_METHOD_VERSION


@pytest.mark.parametrize(
    "stage",
    [
        stage
        for stage in MediumTermStage
        if stage
        not in {
            MediumTermStage.RANGE,
            MediumTermStage.EARLY_UPTREND,
            MediumTermStage.ORDERLY_UPTREND,
        }
    ],
)
def test_non_uptrend_or_extended_stage_remains_ineligible(stage):
    assert not _admit(stage=stage).qualified


@pytest.mark.parametrize(
    "metric,limit",
    [
        ("distance_ma20", 0.10),
        ("distance_ma60", 0.18),
        ("return_60", 0.50),
        ("return_120", 0.85),
    ],
)
def test_range_reversal_cannot_bypass_existing_extension_limits(metric, limit):
    assert _admit(stage=MediumTermStage.RANGE, stage_metrics={metric: limit}).qualified
    admission = _admit(
        stage=MediumTermStage.RANGE, stage_metrics={metric: limit + 0.0001}
    )
    assert not admission.qualified
    assert "entry_extension_exceeds_existing_limits" in admission.reasons


def test_mixed_ma_reversal_with_actual_extended_history_is_rejected():
    stage = assess_medium_term_stage(pd.Series([20.0] * 61 + [10.0] * 40 + [14.0] * 20))
    assert stage.stage is MediumTermStage.RANGE
    assert not stage.hard_freeze_new_entry
    assert stage.distance_ma60 > 0.18
    admission = assess_continuous_entry(
        stage,
        SimpleNamespace(
            candidate_qualified=True,
            execution_ready=True,
            structure=SimpleNamespace(state=StructureState.BREAKOUT),
            execution=SimpleNamespace(state=ExecutionState.READY_BREAKOUT),
        ),
    )
    assert not admission.qualified
    assert "entry_extension_exceeds_existing_limits" in admission.reasons


def test_early_location_is_a_rank_bonus_not_an_orderly_trend_veto():
    early = SimpleNamespace(stage=MediumTermStage.EARLY_UPTREND, hard_freeze_new_entry=False)
    base_reversal = SimpleNamespace(stage=MediumTermStage.RANGE, hard_freeze_new_entry=False)
    orderly = SimpleNamespace(stage=MediumTermStage.ORDERLY_UPTREND, hard_freeze_new_entry=False)
    frozen = SimpleNamespace(stage=MediumTermStage.EARLY_UPTREND, hard_freeze_new_entry=True)
    assert (
        continuous_entry_stage_rank_score(early)
        > continuous_entry_stage_rank_score(base_reversal)
        > continuous_entry_stage_rank_score(orderly)
    )
    assert continuous_entry_stage_rank_score(orderly) > 0.0
    assert continuous_entry_stage_rank_score(frozen) == 0.0


@pytest.mark.parametrize(
    "structure",
    [
        StructureState.NEAR_BREAKOUT,
        StructureState.BASE,
        StructureState.FAILED,
        StructureState.RECLAIM_WAIT,
    ],
)
def test_near_breakout_and_reclaim_wait_remain_ineligible_new_entries(structure):
    admission = _admit(structure=structure)
    assert not admission.qualified
    assert "confirmed_base_breakout_or_healthy_retest_required" in admission.reasons


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frozen": True},
        {"candidate_qualified": False},
        {"execution_ready": False},
        {"execution": ExecutionState.EXTENDED},
        {"execution": ExecutionState.WAIT_CONFIRMATION},
    ],
)
def test_every_hard_admission_gate_is_required(kwargs):
    assert not _admit(**kwargs).qualified


def test_continuous_profile_is_independently_named_without_mutating_legacy_contract():
    legacy = horizon_contract(4)
    assert CONTINUOUS_SIGNAL_CONTRACT is not legacy
    assert CONTINUOUS_SIGNAL_CONTRACT.label == "continuous_daily_weekly_v3"
    assert legacy.label != CONTINUOUS_SIGNAL_CONTRACT.label
