"""Synthetic point-in-time contract checks, not investment-performance proof."""

from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from ashare_lab.analytics.continuous_signals import (
    CONTINUOUS_SIGNAL_CONTRACT,
    PREVIOUS_CONTINUOUS_SIGNAL_CONTRACT,
    assess_continuous_entry,
)
from ashare_lab.analytics.medium_term_stage import assess_medium_term_stage
from ashare_lab.analytics.multi_timeframe import (
    BarTimeframe,
    ExecutionState,
    StructureState,
    TrendDirection,
    _assess_direction,
    assess_multi_timeframe,
)


def _daily_setup(*, retest=False):
    dates = pd.bdate_range(end="2026-09-16", periods=160)
    close = np.full(160, 10.0)
    activity = np.full(160, 100.0)
    event = -3 if retest else -1
    close[event] = 10.20
    activity[event] = 150.0
    if retest:
        close[-2:] = [10.08, 10.10]
    frame = pd.DataFrame(
        {"trade_date": dates, "open": close, "high": close + 0.04,
         "low": close - 0.04, "close": close, "volume_shares": activity}
    )
    if retest:
        frame.loc[len(frame) - 1, "open"] = 10.07
    return frame


def _assess(frame, *, previous=False, as_of="2026-09-16"):
    return assess_multi_timeframe(
        frame, as_of=as_of, holding_weeks=4,
        signal_contract=(PREVIOUS_CONTINUOUS_SIGNAL_CONTRACT if previous
                         else CONTINUOUS_SIGNAL_CONTRACT),
    )


def test_v5_daily_breakout_can_confirm_before_weekly_breakout():
    frame = _daily_setup()
    current, previous = _assess(frame), _assess(frame, previous=True)
    assert current.incomplete_week_excluded
    assert current.weekly_cutoff == pd.Timestamp("2026-09-11")
    assert current.slow_direction.direction is TrendDirection.MIXED
    assert current.slow_direction.qualified  # not falsely relabelled UP
    assert current.structure.timeframe is BarTimeframe.DAILY
    assert current.structure_bar_cutoff == pd.Timestamp("2026-09-16")
    assert current.structure.state is StructureState.BREAKOUT
    assert current.execution.state is ExecutionState.READY_BREAKOUT
    assert current.execution.breakout_line == current.structure.breakout_line
    assert current.execution.reference_timeframe is BarTimeframe.DAILY
    assert current.candidate_qualified and current.execution_ready
    assert assess_continuous_entry(assess_medium_term_stage(frame["close"]), current).qualified
    assert not previous.candidate_qualified
    assert not previous.execution_ready


def test_daily_retest_must_stabilize_after_the_original_breakout():
    frame = _daily_setup(retest=True)
    result = _assess(frame)
    assert result.structure.state is StructureState.HEALTHY_PULLBACK
    assert result.execution.state is ExecutionState.READY_PULLBACK
    assert result.execution.sessions_since_breakout == 2
    assert result.execution.breakout_line == result.structure.breakout_line
    assert "daily_close_not_below_open_or_previous_close" in result.execution.reasons
    assert result.execution_ready


@pytest.mark.parametrize("change", ["red_candle", "lower_close"])
def test_touch_and_reclaim_without_stabilization_remains_wait(change):
    frame = _daily_setup(retest=True)
    if change == "red_candle":
        frame.loc[len(frame) - 1, "open"] = 10.12
    else:
        frame.loc[len(frame) - 2, ["open", "high", "low", "close"]] = [10.12, 10.16, 10.08, 10.12]
    result = _assess(frame)
    assert result.structure.state is StructureState.HEALTHY_PULLBACK
    assert result.execution.state is ExecutionState.WAIT_CONFIRMATION
    assert not result.execution_ready
    assert not assess_continuous_entry(assess_medium_term_stage(frame["close"]), result).qualified


def test_intervening_support_break_cannot_be_erased_by_rebound():
    frame = _daily_setup(retest=True)
    frame.loc[len(frame) - 2, "low"] = 9.70
    result = _assess(frame)
    assert result.execution.state is ExecutionState.WAIT_RECLAIM
    assert not result.candidate_qualified
    assert not result.execution_ready


def test_new_breakout_is_not_mislabelled_a_retest_or_excluded():
    result = _assess(_daily_setup())
    assert result.execution.ready
    assert result.execution.state is ExecutionState.READY_BREAKOUT
    assert result.execution.sessions_since_breakout == 0
    assert not result.execution.latest_retest_touched


def test_support_bounce_without_original_breakout_is_not_ready():
    frame = _daily_setup(retest=True)
    # The price shape alone cannot replace the original activity confirmation.
    frame["volume_shares"] = 100.0
    result = _assess(frame)
    assert result.structure.days_or_bars_since_breakout is None
    assert result.execution.state is ExecutionState.WAIT_CONFIRMATION
    assert not result.execution_ready


def test_rising_above_the_line_without_touching_support_is_not_a_retest():
    frame = _daily_setup(retest=True)
    frame.loc[len(frame) - 2, ["open", "high", "low", "close"]] = [10.30, 10.34, 10.28, 10.30]
    frame.loc[len(frame) - 1, ["open", "high", "low", "close"]] = [10.32, 10.40, 10.30, 10.36]
    result = _assess(frame)
    assert not result.structure.latest_retest_touched
    assert result.execution.state is ExecutionState.WAIT_CONFIRMATION
    assert not result.execution_ready


def test_new_breakout_does_not_disable_existing_overextension_guard():
    frame = _daily_setup()
    frame.loc[len(frame) - 1, ["open", "high", "low", "close"]] = [12.0, 12.04, 11.96, 12.0]
    result = _assess(frame)
    assert result.structure.state is StructureState.BREAKOUT
    assert result.execution.state is ExecutionState.EXTENDED
    assert not result.execution_ready
    assert not assess_continuous_entry(assess_medium_term_stage(frame["close"]), result).qualified


@pytest.mark.parametrize(
    "prices,expected,qualified",
    [(np.linspace(20.0, 10.0, 30), TrendDirection.DOWN, False),
     (np.full(30, 10.0), TrendDirection.MIXED, True),
     (np.linspace(10.0, 20.0, 30), TrendDirection.UP, True),
     (np.full(12, 10.0), TrendDirection.INSUFFICIENT, False)],
)
def test_weekly_filter_direction_contract(prices, expected, qualified):
    result = _assess_direction(pd.DataFrame({"close": prices}), CONTINUOUS_SIGNAL_CONTRACT)
    assert result.direction is expected
    assert result.qualified is qualified
    old = _assess_direction(pd.DataFrame({"close": prices}), PREVIOUS_CONTINUOUS_SIGNAL_CONTRACT)
    assert old.qualified is (expected is TrendDirection.UP)


def test_incomplete_current_week_cannot_rewrite_weekly_background():
    frame = _daily_setup(retest=True)
    before = _assess(frame)
    frame.loc[frame.trade_date > pd.Timestamp("2026-09-11"), ["open", "high", "low", "close"]] *= 2
    after = _assess(frame)
    assert asdict(before.slow_direction) == asdict(after.slow_direction)
    assert before.weekly_cutoff == after.weekly_cutoff


def test_future_bars_cannot_change_daily_confirmation():
    frame = _daily_setup(retest=True)
    before = _assess(frame, as_of="2026-09-15")
    frame.loc[frame.trade_date > pd.Timestamp("2026-09-15"), ["open", "high", "low", "close"]] *= 2
    after = _assess(frame, as_of="2026-09-15")
    assert asdict(before) == asdict(after)


def test_v4_contract_values_are_frozen_independently_of_v5():
    old = PREVIOUS_CONTINUOUS_SIGNAL_CONTRACT
    assert old.label == "continuous_weekly_breakout_daily_execution_v4"
    assert old.structure_timeframe is BarTimeframe.WEEKLY
    assert (old.structure_lookback_bars, old.structure_recent_breakout_bars,
            old.structure_base_bars, old.structure_max_base_width) == (13, 4, 8, 0.22)
    assert not old.slow_direction_filter_only
    assert not old.require_stabilized_retest
    assert old == replace(old)  # ordinary immutable dataclass, not a live alias
