"""Frozen entry/monitoring profile, independent from an intended exit date.

The numerical rules are conservative research hypotheses, not a validated
predictor of a future 'main wave'.  Weekly structure and daily execution are
explicitly frozen observation windows; no expiry exists.
An entry gate must never be reapplied to an intact existing holding.
"""

from dataclasses import dataclass

from ashare_lab.analytics.medium_term_stage import (
    MediumTermStage,
    MediumTermStageAssessment,
    entry_extension_exceeded,
)
from ashare_lab.analytics.multi_timeframe import (
    BarTimeframe,
    ExecutionState,
    HorizonContract,
    MultiTimeframeAssessment,
    StructureState,
)

CONTINUOUS_METHOD_VERSION = "continuous-signal-v4"
CONTINUOUS_SIGNAL_CONTRACT = HorizonContract(
    4,  # legacy transport discriminator ONLY; not a holding deadline
    "continuous_weekly_breakout_daily_execution_v4",
    BarTimeframe.WEEKLY,
    8,
    26,
    BarTimeframe.WEEKLY,
    13,
    4,
    8,
    0.22,
    40,
    10,
    20,
    60,
    20,
    140,
    execution_uses_structure=True,
)
RISK_OBSERVATION_SESSIONS = 20

# Existing v1-v3 holdings retain their original monitoring windows.  A new
# entry-method version never retroactively changes their source timeframe.
LEGACY_CONTINUOUS_HOLDING_CONTRACT = HorizonContract(
    4,
    "continuous_daily_weekly_v3",
    BarTimeframe.WEEKLY,
    8,
    26,
    BarTimeframe.DAILY,
    60,
    10,
    30,
    0.18,
    40,
    10,
    20,
    60,
    20,
    140,
)


@dataclass(frozen=True, slots=True)
class ContinuousEntryAdmission:
    qualified: bool
    reasons: tuple[str, ...]
    method_version: str = CONTINUOUS_METHOD_VERSION


def assess_continuous_entry(
    stage: MediumTermStageAssessment,
    timeframe: MultiTimeframeAssessment,
) -> ContinuousEntryAdmission:
    """Require completed breakout evidence while rejecting weak or late stages.

    V4 admits a range/base reversal only when completed weekly bars prove both
    upward direction and a base breakout or healthy retest.  Daily bars assess
    the buying location and whether that weekly structure remains intact; they
    cannot manufacture a missing weekly breakout.  Early and orderly uptrends
    are also admitted.
    Location is therefore a ranking preference rather than an eligibility
    veto.  Downtrends, insufficient history, extended/parabolic formations and
    explicitly frozen formations remain ineligible.
    """
    reasons: list[str] = []
    if (
        stage.stage
        not in {
            MediumTermStage.RANGE,
            MediumTermStage.EARLY_UPTREND,
            MediumTermStage.ORDERLY_UPTREND,
        }
        or stage.hard_freeze_new_entry
    ):
        reasons.append("downtrend_extended_or_unavailable_stage")
    # RANGE describes mixed MA ordering, not proof of an unextended entry.
    # Preserve the existing limits even when a sharp reversal is not yet
    # classified as an ordered uptrend by the shared legacy stage classifier.
    if entry_extension_exceeded(
        distance_ma20=stage.distance_ma20,
        distance_ma60=stage.distance_ma60,
        return_60=stage.return_60,
        return_120=stage.return_120,
    ):
        reasons.append("entry_extension_exceeds_existing_limits")
    if (
        not timeframe.candidate_qualified
        or timeframe.structure.timeframe is not BarTimeframe.WEEKLY
        or timeframe.slow_direction.timeframe is not BarTimeframe.WEEKLY
        or not timeframe.slow_direction.qualified
    ):
        reasons.append("completed_weekly_direction_and_structure_not_qualified")
    if timeframe.structure.state not in {StructureState.BREAKOUT, StructureState.HEALTHY_PULLBACK}:
        reasons.append("confirmed_base_breakout_or_healthy_retest_required")
    if not timeframe.execution_ready or timeframe.execution.state not in {
        ExecutionState.READY_BREAKOUT,
        ExecutionState.READY_PULLBACK,
    }:
        reasons.append("daily_execution_of_confirmed_weekly_structure_required")
    return ContinuousEntryAdmission(not reasons, tuple(reasons))


def continuous_entry_stage_rank_score(stage: MediumTermStageAssessment) -> float:
    """Prefer an earlier valid location without turning it into a hard gate."""

    if stage.stage is MediumTermStage.EARLY_UPTREND and not stage.hard_freeze_new_entry:
        return 1.0
    if stage.stage is MediumTermStage.RANGE and not stage.hard_freeze_new_entry:
        # The separate contract has already proved a weekly confirmed breakout;
        # this is a base reversal, not admission of an arbitrary range.
        return 0.8
    if stage.stage is MediumTermStage.ORDERLY_UPTREND and not stage.hard_freeze_new_entry:
        return 0.6
    return 0.0
