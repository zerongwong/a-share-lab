from types import SimpleNamespace

import pytest

from ashare_lab.analytics.continuous_signals import (
    CONTINUOUS_METHOD_VERSION,
    CONTINUOUS_SIGNAL_CONTRACT,
    assess_continuous_entry,
)
from ashare_lab.analytics.medium_term_stage import MediumTermStage
from ashare_lab.analytics.multi_timeframe import ExecutionState, StructureState, horizon_contract


def _admit(
    *,
    stage=MediumTermStage.EARLY_UPTREND,
    frozen=False,
    structure=StructureState.BREAKOUT,
    execution=ExecutionState.READY_BREAKOUT,
    candidate_qualified=True,
    execution_ready=True,
):
    return assess_continuous_entry(
        SimpleNamespace(stage=stage, hard_freeze_new_entry=frozen),
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
def test_confirmed_early_breakout_and_healthy_retest_can_enter(structure, execution):
    admission = _admit(structure=structure, execution=execution)
    assert admission.qualified
    assert admission.reasons == ()
    assert admission.method_version == CONTINUOUS_METHOD_VERSION


@pytest.mark.parametrize(
    "stage", [stage for stage in MediumTermStage if stage is not MediumTermStage.EARLY_UPTREND]
)
def test_orderly_or_extended_trend_is_not_relabelled_as_early_entry(stage):
    assert not _admit(stage=stage).qualified


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
    assert CONTINUOUS_SIGNAL_CONTRACT.label == "continuous_daily_weekly_v1"
    assert legacy.label != CONTINUOUS_SIGNAL_CONTRACT.label
