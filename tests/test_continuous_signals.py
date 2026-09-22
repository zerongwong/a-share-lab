from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ashare_lab.analytics.continuous_signals import (
    CONTINUOUS_METHOD_VERSION,
    CONTINUOUS_SIGNAL_CONTRACT,
    assess_continuous_entry,
    continuous_entry_stage_rank_score,
)
from ashare_lab.analytics.medium_term_stage import MediumTermStage, assess_medium_term_stage
from ashare_lab.analytics.multi_timeframe import (
    BarTimeframe,
    ExecutionState,
    StructureState,
    assess_multi_timeframe,
    horizon_contract,
)


def _admit(
    *,
    stage=MediumTermStage.EARLY_UPTREND,
    frozen=False,
    structure=StructureState.BREAKOUT,
    execution=ExecutionState.READY_BREAKOUT,
    candidate_qualified=True,
    execution_ready=True,
    stage_metrics=None,
    structure_timeframe=BarTimeframe.WEEKLY,
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
            structure=SimpleNamespace(state=structure, timeframe=structure_timeframe),
            slow_direction=SimpleNamespace(timeframe=BarTimeframe.WEEKLY, qualified=True),
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
def test_confirmed_non_extended_breakout_and_healthy_retest_can_enter(structure, execution, stage):
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
    admission = _admit(stage=MediumTermStage.RANGE, stage_metrics={metric: limit + 0.0001})
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
            structure=SimpleNamespace(state=StructureState.BREAKOUT, timeframe=BarTimeframe.WEEKLY),
            slow_direction=SimpleNamespace(timeframe=BarTimeframe.WEEKLY, qualified=True),
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
    assert CONTINUOUS_SIGNAL_CONTRACT.label == "continuous_weekly_breakout_daily_execution_v4"
    assert CONTINUOUS_SIGNAL_CONTRACT.structure_timeframe is BarTimeframe.WEEKLY
    assert CONTINUOUS_SIGNAL_CONTRACT.execution_uses_structure
    assert legacy.structure_timeframe is BarTimeframe.DAILY
    assert not legacy.execution_uses_structure
    assert legacy.label != CONTINUOUS_SIGNAL_CONTRACT.label


def test_daily_breakout_cannot_be_passed_off_as_weekly_confirmation():
    result = _admit(structure_timeframe=BarTimeframe.DAILY)
    assert not result.qualified
    assert "completed_weekly_direction_and_structure_not_qualified" in result.reasons


def _weekly_setup():
    dates = pd.bdate_range(end="2026-09-18", periods=160)
    close = np.concatenate([np.linspace(9.0, 10.0, 100), np.full(60, 10.0)])
    close[-5:] = [10.0, 10.22, 10.24, 10.26, 10.28]
    volume = np.full(len(dates), 100.0)
    # Weekly activity confirms the breakout, but no individual daily breakout
    # has increased activity: Monday is high activity before Tuesday penetrates.
    volume[-5] = 300.0
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": close,
            "high": close + 0.04,
            "low": close - 0.04,
            "close": close,
            "volume_shares": volume,
        }
    )


def _weekly_assessment(frame, as_of="2026-09-18"):
    return assess_multi_timeframe(
        frame, as_of=as_of, holding_weeks=4, signal_contract=CONTINUOUS_SIGNAL_CONTRACT
    )


def test_completed_weekly_breakout_needs_no_separate_daily_breakout():
    frame = _weekly_setup()
    result = _weekly_assessment(frame)
    legacy = assess_multi_timeframe(frame, as_of="2026-09-18", holding_weeks=4)
    assert result.structure.state is StructureState.BREAKOUT
    assert result.structure.timeframe is BarTimeframe.WEEKLY
    assert result.structure_bar_cutoff == result.weekly_cutoff == pd.Timestamp("2026-09-18")
    assert result.candidate_qualified
    assert result.execution_ready
    assert result.execution.reference_timeframe is BarTimeframe.WEEKLY
    assert result.execution.breakout_line == result.structure.breakout_line
    assert "independent_daily_breakout_not_required" in result.execution.reasons
    assert not legacy.execution_ready
    assert assess_continuous_entry(assess_medium_term_stage(frame["close"]), result).qualified
    payload = asdict(result)
    assert payload["structure"]["timeframe"] == payload["execution"]["reference_timeframe"]


def test_daily_penetration_in_incomplete_week_does_not_confirm_weekly_breakout():
    frame = _weekly_setup()
    result = _weekly_assessment(frame, as_of="2026-09-16")
    assert result.incomplete_week_excluded
    assert result.weekly_cutoff == result.structure_bar_cutoff == pd.Timestamp("2026-09-11")
    assert result.structure.state is not StructureState.BREAKOUT
    assert not result.candidate_qualified
    assert not result.execution_ready
    assert not assess_continuous_entry(assess_medium_term_stage(frame["close"]), result).qualified


def test_post_confirmation_daily_retest_uses_weekly_line_and_blocks_breached_path():
    frame = _weekly_setup()
    following = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-09-21", "2026-09-22"]),
            "open": [10.15, 10.12],
            "high": [10.18, 10.16],
            "low": [10.04, 10.02],
            "close": [10.14, 10.12],
            "volume_shares": [100.0, 100.0],
        }
    )
    frame = pd.concat([frame, following], ignore_index=True)
    result = _weekly_assessment(frame, as_of="2026-09-22")
    assert result.incomplete_week_excluded
    assert result.execution.state is ExecutionState.READY_PULLBACK
    assert result.execution_ready
    assert result.execution.sessions_since_breakout == 2
    broken = frame.copy()
    broken.loc[broken["trade_date"] == pd.Timestamp("2026-09-21"), "low"] = 9.70
    failed = _weekly_assessment(broken, as_of="2026-09-22")
    assert failed.structure.state is StructureState.BREAKOUT
    assert failed.execution.state is ExecutionState.WAIT_RECLAIM
    assert not failed.execution_ready


def test_future_week_changes_cannot_modify_frozen_weekly_assessment():
    frame = _weekly_setup()
    before = _weekly_assessment(frame, as_of="2026-09-16")
    frame.loc[
        frame["trade_date"] > pd.Timestamp("2026-09-16"), ["open", "high", "low", "close"]
    ] *= 2
    after = _weekly_assessment(frame, as_of="2026-09-16")
    assert asdict(before) == asdict(after)


def test_completed_weekly_healthy_retest_retains_confirmed_weekly_reference():
    frame = _weekly_setup()
    following = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2026-09-21", periods=5),
            "open": [10.15] * 5,
            "high": [10.20] * 5,
            "low": [10.02] * 5,
            "close": [10.15] * 5,
            "volume_shares": [100.0] * 5,
        }
    )
    result = _weekly_assessment(pd.concat([frame, following]), as_of="2026-09-25")
    assert result.structure.state is StructureState.HEALTHY_PULLBACK
    assert result.structure.days_or_bars_since_breakout == 1
    assert result.structure_bar_cutoff == pd.Timestamp("2026-09-25")
    assert result.execution.state is ExecutionState.READY_PULLBACK
    assert result.execution.reference_timeframe is BarTimeframe.WEEKLY
    assert result.execution.sessions_since_breakout == 5
    assert result.execution_ready


def test_confirmed_weekly_breakout_does_not_override_daily_extension_guard():
    frame = _weekly_setup()
    following = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-09-21"]),
            "open": [12.0],
            "high": [12.1],
            "low": [11.9],
            "close": [12.0],
            "volume_shares": [100.0],
        }
    )
    result = _weekly_assessment(pd.concat([frame, following]), as_of="2026-09-21")
    assert result.structure.state is StructureState.BREAKOUT
    assert result.execution.state is ExecutionState.EXTENDED
    assert not result.execution_ready
