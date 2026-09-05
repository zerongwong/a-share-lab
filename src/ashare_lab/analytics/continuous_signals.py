"""Frozen entry/monitoring profile, independent from an intended exit date.

The numerical rules are conservative research hypotheses, not a validated
predictor of a future 'main wave'.  The legacy four-week parameter values are
explicitly frozen here as daily/weekly observation windows; no expiry exists.
An entry gate must never be reapplied to an intact existing holding.
"""

from dataclasses import dataclass

from ashare_lab.analytics.medium_term_stage import MediumTermStage, MediumTermStageAssessment
from ashare_lab.analytics.multi_timeframe import (
    BarTimeframe,
    ExecutionState,
    HorizonContract,
    MultiTimeframeAssessment,
    StructureState,
)

CONTINUOUS_METHOD_VERSION = "continuous-signal-v1"
CONTINUOUS_SIGNAL_CONTRACT = HorizonContract(
    4,  # legacy transport discriminator ONLY; not a holding deadline
    "continuous_daily_weekly_v1",
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
RISK_OBSERVATION_SESSIONS = 20


@dataclass(frozen=True, slots=True)
class ContinuousEntryAdmission:
    qualified: bool
    reasons: tuple[str, ...]
    method_version: str = CONTINUOUS_METHOD_VERSION


def assess_continuous_entry(
    stage: MediumTermStageAssessment,
    timeframe: MultiTimeframeAssessment,
) -> ContinuousEntryAdmission:
    """Require observable early, non-extended strength AND confirmed structure.

    EARLY_UPTREND uses the frozen stage guard: ordered positive trend with
    120-session gain <=15%, no extension/vertical acceleration.  This does not
    claim an absolute bottom, first-ever breakout or inevitable future profit.
    """
    reasons: list[str] = []
    if stage.stage is not MediumTermStage.EARLY_UPTREND or stage.hard_freeze_new_entry:
        reasons.append("early_non_extended_uptrend_not_confirmed")
    if not timeframe.candidate_qualified:
        reasons.append("daily_weekly_structure_not_qualified")
    if timeframe.structure.state not in {StructureState.BREAKOUT, StructureState.HEALTHY_PULLBACK}:
        reasons.append("confirmed_base_breakout_or_healthy_retest_required")
    if not timeframe.execution_ready or timeframe.execution.state not in {
        ExecutionState.READY_BREAKOUT,
        ExecutionState.READY_PULLBACK,
    }:
        reasons.append("daily_close_entry_confirmation_required")
    return ContinuousEntryAdmission(not reasons, tuple(reasons))
