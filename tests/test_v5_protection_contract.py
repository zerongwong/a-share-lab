"""Point-in-time boundaries for the v5 entry / frozen-v4 protection split."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from ashare_lab.analytics.continuous_signals import (
    CONTINUOUS_SIGNAL_CONTRACT,
    PREVIOUS_CONTINUOUS_SIGNAL_CONTRACT,
    assess_continuous_entry,
)
from ashare_lab.analytics.entry_readiness import EntryPattern
from ashare_lab.analytics.medium_term_stage import assess_medium_term_stage
from ashare_lab.analytics.multi_timeframe import (
    ExecutionState,
    assess_multi_timeframe,
    build_completed_timeframes,
)
from ashare_lab.services.build_midterm_portfolio import (
    ConditionalEntryPlanKind,
    _build_horizon_price_observation_plan,
)
from ashare_lab.services.review_active_holdings import _candidate_stop


def _daily_setup(*, retest: bool) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-09-16", periods=160)
    close = np.full(160, 10.0)
    activity = np.full(160, 100.0)
    event = -3 if retest else -1
    close[event] = 10.20
    activity[event] = 150.0
    if retest:
        close[-2:] = [10.08, 10.10]
    frame = pd.DataFrame(
        {
            "trade_date": dates,
            "open": close,
            "high": close + 0.04,
            "low": close - 0.04,
            "close": close,
            "volume_shares": activity,
        }
    )
    if retest:
        frame.loc[len(frame) - 1, "open"] = 10.07
    return frame


@pytest.mark.parametrize(
    ("retest", "entry_pattern", "execution_state", "plan_kind"),
    (
        (
            False,
            EntryPattern.VOLUME_BREAKOUT,
            ExecutionState.READY_BREAKOUT,
            ConditionalEntryPlanKind.RECLAIM,
        ),
        (
            True,
            EntryPattern.HEALTHY_PULLBACK,
            ExecutionState.READY_PULLBACK,
            ConditionalEntryPlanKind.HEALTHY_PULLBACK,
        ),
    ),
)
def test_v5_price_plan_uses_same_frozen_v4_protection_as_holding_review(
    retest: bool,
    entry_pattern: EntryPattern,
    execution_state: ExecutionState,
    plan_kind: ConditionalEntryPlanKind,
) -> None:
    frame = _daily_setup(retest=retest)
    cutoff = pd.Timestamp(frame.iloc[-1]["trade_date"]).normalize()
    entry_assessment = assess_multi_timeframe(
        frame,
        as_of=cutoff,
        holding_weeks=4,
        signal_contract=CONTINUOUS_SIGNAL_CONTRACT,
    )
    protection_assessment = assess_multi_timeframe(
        frame,
        as_of=cutoff,
        holding_weeks=4,
        signal_contract=PREVIOUS_CONTINUOUS_SIGNAL_CONTRACT,
    )

    assert entry_assessment.execution.state is execution_state
    plan = _build_horizon_price_observation_plan(
        frame,
        cutoff=cutoff,
        holding_weeks=4,
        timeframe=entry_assessment,
        entry_pattern=entry_pattern,
        continuous_policy=True,
    )
    expected = _candidate_stop(
        build_completed_timeframes(frame, as_of=cutoff),
        protection_assessment,
        entry_date=cutoff.date(),
    )

    assert plan is not None
    assert plan.kind is plan_kind
    assert plan.primary_structure_timeframe == "daily"
    assert plan.invalidation_source_timeframe == "weekly_completed"
    assert plan.invalidation_price == expected.stop
    assert plan.initial_protection_support == expected.support
    assert plan.initial_protection_atr == expected.atr14
    assert plan.initial_protection_evidence_date == expected.evidence_date.isoformat()
    assert plan.initial_protection_atr_cutoff == expected.atr_cutoff.isoformat()
    assert plan.initial_protection_method_version.endswith("+continuous-weekly-v4")
    assert plan.effective_initial_protection_price == max(
        expected.stop,
        plan.planned_cost_stop_price,
    )


def test_future_price_and_activity_mutations_cannot_change_prior_cutoff_result() -> None:
    frame = _daily_setup(retest=True)
    cutoff = pd.Timestamp(frame.iloc[-1]["trade_date"]).normalize()
    frame["amount_cny"] = frame["close"] * frame["volume_shares"]
    future = frame.iloc[[-1]].copy()
    future["trade_date"] = cutoff + pd.offsets.BDay(1)
    with_future = pd.concat((frame, future), ignore_index=True)
    before = assess_multi_timeframe(
        with_future,
        as_of=cutoff,
        holding_weeks=4,
        signal_contract=CONTINUOUS_SIGNAL_CONTRACT,
    )

    mutated = with_future.copy()
    future_index = mutated.index[-1]
    mutated.loc[future_index, "volume_shares"] = -1.0
    mutated["amount_cny"] = mutated["amount_cny"].astype(object)
    mutated.loc[future_index, "amount_cny"] = "future-provider-corruption"
    mutated.loc[future_index, ["open", "high", "low", "close"]] = [99.0, 1.0, 100.0, 0.0]
    after = assess_multi_timeframe(
        mutated,
        as_of=cutoff,
        holding_weeks=4,
        signal_contract=CONTINUOUS_SIGNAL_CONTRACT,
    )

    assert asdict(after) == asdict(before)


def test_v5_assessment_is_deterministic_for_identical_point_in_time_input() -> None:
    frame = _daily_setup(retest=True)
    cutoff = pd.Timestamp(frame.iloc[-1]["trade_date"]).normalize()

    first = assess_multi_timeframe(
        frame,
        as_of=cutoff,
        holding_weeks=4,
        signal_contract=CONTINUOUS_SIGNAL_CONTRACT,
    )
    second = assess_multi_timeframe(
        frame.copy(deep=True),
        as_of=cutoff,
        holding_weeks=4,
        signal_contract=CONTINUOUS_SIGNAL_CONTRACT,
    )

    assert asdict(second) == asdict(first)


def test_v5_breakout_without_a_complete_activity_series_fails_closed() -> None:
    frame = _daily_setup(retest=False).drop(columns="volume_shares")
    cutoff = pd.Timestamp(frame.iloc[-1]["trade_date"]).normalize()
    assessment = assess_multi_timeframe(
        frame,
        as_of=cutoff,
        holding_weeks=4,
        signal_contract=CONTINUOUS_SIGNAL_CONTRACT,
    )

    admission = assess_continuous_entry(
        assess_medium_term_stage(frame["close"]),
        assessment,
    )
    assert not assessment.candidate_qualified
    assert not assessment.execution_ready
    assert assessment.execution.state is ExecutionState.WAIT_CONFIRMATION
    assert not admission.qualified
