from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from ashare_lab.analytics.multi_timeframe import (
    HORIZON_CONTRACTS,
    MULTI_TIMEFRAME_IMPLEMENTATION_STATUS,
    MULTI_TIMEFRAME_METHOD_VERSION,
    SUPPORTED_HOLDING_WEEKS,
    BarTimeframe,
    ExecutionState,
    StructureState,
    TrendDirection,
    _assess_execution,
    _assess_structure,
    assess_all_horizons,
    assess_multi_timeframe,
    build_completed_timeframes,
    horizon_contract,
)


def _history(
    *,
    end: str = "2026-07-31",
    periods: int = 820,
    speed: float = 0.00075,
    phase: float = 0.0,
) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=periods)
    index = np.arange(periods, dtype=float)
    close = 10.0 * np.exp(
        speed * index
        + 0.035 * np.sin(index / 31.0 + phase)
        + 0.012 * np.sin(index / 7.0 + phase / 2.0)
    )
    open_price = close * (1.0 - 0.001 * np.sin(index / 5.0))
    high = np.maximum(open_price, close) * 1.012
    low = np.minimum(open_price, close) * 0.988
    activity = 1_000_000.0 * (1.0 + 0.12 * np.sin(index / 13.0))
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume_shares": activity,
            "amount_cny": activity * close,
        }
    )


def _simple_calendar_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2026-06-01", "2026-09-04")
    close = np.linspace(10.0, 12.0, len(dates))
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": close - 0.02,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume_shares": np.full(len(dates), 100.0),
        }
    )


def _assessment_snapshot(result: object) -> dict[str, object]:
    raw = asdict(result)
    # The contract is intentionally part of the comparison.  If a future bar
    # can change any audit field at the same cutoff, this dictionary changes.
    return raw


def test_weekly_bar_is_excluded_midweek_and_included_on_friday() -> None:
    frame = _simple_calendar_frame()

    wednesday = build_completed_timeframes(frame, as_of="2026-08-26")
    friday = build_completed_timeframes(frame, as_of="2026-08-28")

    assert wednesday.incomplete_week_excluded
    assert wednesday.weekly_cutoff == pd.Timestamp("2026-08-21")
    assert pd.Timestamp(wednesday.weekly.iloc[-1]["period_end"]) == pd.Timestamp("2026-08-21")
    assert pd.Timestamp(wednesday.weekly.iloc[-1]["trade_date"]) == pd.Timestamp("2026-08-21")
    assert friday.incomplete_week_excluded is False
    assert friday.weekly_cutoff == pd.Timestamp("2026-08-28")
    assert pd.Timestamp(friday.weekly.iloc[-1]["period_end"]) == pd.Timestamp("2026-08-28")
    assert pd.Timestamp(friday.weekly.iloc[-1]["trade_date"]) == pd.Timestamp("2026-08-28")
    assert friday.weekly.iloc[-1]["source_rows"] == 5


def test_monthly_bar_is_excluded_midmonth_and_included_at_business_month_end() -> None:
    frame = _simple_calendar_frame()

    midmonth = build_completed_timeframes(frame, as_of="2026-08-28")
    month_end = build_completed_timeframes(frame, as_of="2026-08-31")

    assert midmonth.incomplete_month_excluded
    assert midmonth.monthly_cutoff == pd.Timestamp("2026-07-31")
    assert pd.Timestamp(midmonth.monthly.iloc[-1]["period_end"]) == pd.Timestamp("2026-07-31")
    assert month_end.incomplete_month_excluded is False
    assert month_end.monthly_cutoff == pd.Timestamp("2026-08-31")
    assert pd.Timestamp(month_end.monthly.iloc[-1]["period_end"]) == pd.Timestamp("2026-08-31")
    assert pd.Timestamp(month_end.monthly.iloc[-1]["trade_date"]) == pd.Timestamp("2026-08-31")


def test_last_business_day_closes_month_when_calendar_month_end_is_weekend() -> None:
    dates = pd.bdate_range("2025-11-03", "2026-02-06")
    close = np.linspace(10.0, 11.0, len(dates))
    frame = pd.DataFrame(
        {
            "trade_date": dates,
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "amount_cny": np.full(len(dates), 100.0),
        }
    )

    result = build_completed_timeframes(frame, as_of="2026-01-30")

    assert pd.Timestamp(result.monthly.iloc[-1]["period_end"]) == pd.Timestamp("2026-01-31")
    assert pd.Timestamp(result.monthly.iloc[-1]["trade_date"]) == pd.Timestamp("2026-01-30")


@pytest.mark.parametrize("holding_weeks", SUPPORTED_HOLDING_WEEKS)
def test_future_daily_bars_cannot_change_any_assessment_field(holding_weeks: int) -> None:
    full = _history()
    cutoff = pd.Timestamp(full.iloc[650]["trade_date"])
    future = full.copy()
    future.loc[future["trade_date"] > cutoff, ["open", "high", "low", "close"]] *= 100.0
    future.loc[future["trade_date"] > cutoff, ["volume_shares", "amount_cny"]] *= 1_000.0
    truncated = full.loc[full["trade_date"] <= cutoff].copy()

    clean_result = assess_multi_timeframe(
        truncated,
        as_of=cutoff,
        holding_weeks=holding_weeks,
    )
    future_result = assess_multi_timeframe(
        future,
        as_of=cutoff,
        holding_weeks=holding_weeks,
    )

    assert _assessment_snapshot(future_result) == _assessment_snapshot(clean_result)


@pytest.mark.parametrize(
    ("column", "future_value"),
    (
        ("amount_cny", np.nan),
        ("volume_shares", np.inf),
        ("amount_cny", -1.0),
    ),
)
def test_future_activity_corruption_cannot_change_same_cutoff_assessment(
    column: str,
    future_value: float,
) -> None:
    full = _history()
    cutoff = pd.Timestamp(full.iloc[650]["trade_date"])
    truncated = full.loc[full["trade_date"] <= cutoff].copy()
    corrupted = full.copy()
    corrupted.loc[corrupted["trade_date"] > cutoff, column] = future_value

    expected = assess_multi_timeframe(
        truncated,
        as_of=cutoff,
        holding_weeks=13,
    )
    actual = assess_multi_timeframe(
        corrupted,
        as_of=cutoff,
        holding_weeks=13,
    )

    assert _assessment_snapshot(actual) == _assessment_snapshot(expected)


@pytest.mark.parametrize(
    ("column", "future_value"),
    (
        ("close", np.nan),
        ("high", -1.0),
    ),
)
def test_future_price_corruption_cannot_change_same_cutoff_assessment(
    column: str,
    future_value: float,
) -> None:
    full = _history()
    cutoff = pd.Timestamp(full.iloc[650]["trade_date"])
    truncated = full.loc[full["trade_date"] <= cutoff].copy()
    corrupted = full.copy()
    corrupted.loc[corrupted["trade_date"] > cutoff, column] = future_value

    expected = assess_multi_timeframe(
        truncated,
        as_of=cutoff,
        holding_weeks=13,
    )
    actual = assess_multi_timeframe(
        corrupted,
        as_of=cutoff,
        holding_weeks=13,
    )

    assert _assessment_snapshot(actual) == _assessment_snapshot(expected)


def test_future_duplicate_trade_date_is_ignored_after_point_in_time_cutoff() -> None:
    full = _history()
    cutoff = pd.Timestamp(full.iloc[650]["trade_date"])
    truncated = full.loc[full["trade_date"] <= cutoff].copy()
    future_row = full.loc[full["trade_date"] > cutoff].iloc[[0]].copy()
    with_future_duplicate = pd.concat([full, future_row], ignore_index=True)

    expected = assess_multi_timeframe(
        truncated,
        as_of=cutoff,
        holding_weeks=13,
    )
    actual = assess_multi_timeframe(
        with_future_duplicate,
        as_of=cutoff,
        holding_weeks=13,
    )

    assert _assessment_snapshot(actual) == _assessment_snapshot(expected)


def test_six_horizons_have_distinct_contracts_and_computed_evidence() -> None:
    frame = _history()
    benchmark = _history(speed=0.00035, phase=1.2)

    results = assess_all_horizons(
        frame,
        as_of="2026-07-31",
        benchmark_frame=benchmark,
    )

    assert tuple(result.holding_weeks for result in results) == SUPPORTED_HOLDING_WEEKS
    assert len({contract.audit_signature for contract in HORIZON_CONTRACTS.values()}) == 6
    assert len({result.score for result in results}) >= 4
    assert (
        len(
            {
                (
                    result.slow_direction.timeframe,
                    result.slow_direction.bars_required,
                    result.structure.timeframe,
                    result.structure.lookback_bars,
                    result.execution.moving_average,
                    result.relative_strength.sessions,
                )
                for result in results
            }
        )
        == 6
    )
    assert all(result.method_version == MULTI_TIMEFRAME_METHOD_VERSION for result in results)
    assert all(
        result.implementation_status == MULTI_TIMEFRAME_IMPLEMENTATION_STATUS for result in results
    )


def test_contracts_encode_slow_middle_and_fast_timeframes() -> None:
    one_week = horizon_contract(1)
    one_month = horizon_contract(4)
    three_months = horizon_contract(13)
    one_year = horizon_contract(52)

    assert one_week.slow_timeframe is BarTimeframe.WEEKLY
    assert one_week.structure_timeframe is BarTimeframe.DAILY
    assert one_month.structure_lookback_bars == 60
    assert three_months.slow_timeframe is BarTimeframe.MONTHLY
    assert three_months.structure_timeframe is BarTimeframe.WEEKLY
    assert one_year.slow_timeframe is BarTimeframe.MONTHLY
    assert one_year.structure_timeframe is BarTimeframe.WEEKLY
    assert all(contract.execution_ma_sessions > 0 for contract in HORIZON_CONTRACTS.values())


def test_one_year_contract_audits_monthly_ma12_ma24_and_daily_ma252() -> None:
    frame = _history(periods=900, speed=0.0010)

    result = assess_multi_timeframe(
        frame,
        as_of="2026-07-31",
        holding_weeks=52,
    )

    assert result.contract.slow_timeframe is BarTimeframe.MONTHLY
    assert result.slow_direction.bars_required == 24
    assert result.slow_direction.bars_available >= 24
    assert result.slow_direction.fast_average is not None
    assert result.slow_direction.slow_average is not None
    assert result.slow_direction.fast_average > result.slow_direction.slow_average
    assert result.slow_direction.direction is TrendDirection.UP
    assert result.contract.daily_anchor_sessions == 252
    assert result.daily_anchor_average == pytest.approx(frame["close"].tail(252).mean())
    assert result.above_daily_anchor is True
    assert "one_year_contract_uses_completed_monthly_ma12_ma24_and_daily_ma252" in result.reasons


def test_one_year_contract_fails_closed_without_24_completed_months() -> None:
    frame = _history(periods=400)

    result = assess_multi_timeframe(
        frame,
        as_of="2026-07-31",
        holding_weeks=52,
    )

    assert result.slow_direction.direction is TrendDirection.INSUFFICIENT
    assert result.slow_direction.bars_available < 24
    assert result.candidate_qualified is False
    assert "minimum_24_monthly_completed_bars_required" in result.slow_direction.reasons


def test_benchmark_relative_strength_uses_same_dates_and_horizon_window() -> None:
    security = _history(speed=0.0010)
    full_benchmark = _history(speed=0.0002, phase=0.5)
    benchmark = full_benchmark.loc[:, ["trade_date", "close"]].rename(
        columns={"trade_date": "date"}
    )

    result = assess_multi_timeframe(
        security,
        as_of="2026-07-31",
        holding_weeks=13,
        benchmark_frame=benchmark,
    )

    assert result.relative_strength.available
    assert result.relative_strength.sessions == 60
    assert result.relative_strength.excess_return is not None
    assert result.relative_strength.excess_return > 0
    assert result.relative_strength.score is not None
    assert result.relative_strength.score > 0.5


def test_wholly_unavailable_optional_volume_uses_valid_amount_without_fabrication() -> None:
    frame = _history()
    frame["volume_shares"] = np.nan

    result = assess_multi_timeframe(
        frame,
        as_of="2026-07-31",
        holding_weeks=13,
    )

    assert result.data_cutoff == pd.Timestamp("2026-07-31")
    assert result.structure.activity_ratio is not None
    assert result.structure.activity_ratio >= 0.0


def test_partially_available_volume_uses_complete_amount_without_fabrication() -> None:
    frame = _history()
    frame.loc[frame.index[:-2], "volume_shares"] = np.nan

    result = assess_multi_timeframe(
        frame,
        as_of="2026-07-31",
        holding_weeks=13,
    )

    assert result.data_cutoff == pd.Timestamp("2026-07-31")
    assert result.structure.activity_ratio is not None


def test_partially_available_amount_uses_complete_volume_without_fabrication() -> None:
    frame = _history()
    frame.loc[frame.index[:-2], "amount_cny"] = np.nan

    result = assess_multi_timeframe(
        frame,
        as_of="2026-07-31",
        holding_weeks=13,
    )

    assert result.data_cutoff == pd.Timestamp("2026-07-31")
    assert result.structure.activity_ratio is not None


def test_partial_activity_without_complete_alternative_fails_closed() -> None:
    frame = _history()
    frame.loc[frame.index[:-2], ["amount_cny", "volume_shares"]] = np.nan

    with pytest.raises(
        ValueError,
        match="partial_activity_without_complete_alternative",
    ):
        assess_multi_timeframe(
            frame,
            as_of="2026-07-31",
            holding_weeks=13,
        )


def test_future_benchmark_prices_cannot_change_relative_strength() -> None:
    security = _history(speed=0.0010)
    benchmark = _history(speed=0.0002, phase=0.5).loc[:, ["trade_date", "close"]]
    cutoff = pd.Timestamp(benchmark.iloc[650]["trade_date"])
    truncated_benchmark = benchmark.loc[benchmark["trade_date"] <= cutoff].copy()
    future_benchmark = benchmark.copy()
    future_benchmark.loc[future_benchmark["trade_date"] > cutoff, "close"] *= 1_000.0

    expected = assess_multi_timeframe(
        security,
        as_of=cutoff,
        holding_weeks=13,
        benchmark_frame=truncated_benchmark,
    )
    actual = assess_multi_timeframe(
        security,
        as_of=cutoff,
        holding_weeks=13,
        benchmark_frame=future_benchmark,
    )

    assert actual.relative_strength == expected.relative_strength
    assert actual.score == expected.score


def test_candidate_membership_is_separate_from_daily_execution() -> None:
    frame = _history(speed=0.0009)

    result = assess_multi_timeframe(
        frame,
        as_of="2026-07-31",
        holding_weeks=13,
    )

    assert isinstance(result.candidate_qualified, bool)
    assert isinstance(result.execution_ready, bool)
    if result.execution_ready:
        assert result.candidate_qualified
        assert result.execution.ready


def test_horizon_contract_rejects_unsupported_period() -> None:
    with pytest.raises(ValueError, match="1, 2, 4, 13, 26 or 52"):
        horizon_contract(8)


def _retest_history(*, breach: bool = False, touching: bool = True, reclaimed: bool = True):
    frame = pd.DataFrame(
        {
            "open": [9.90] * 150,
            "high": [10.0] * 150,
            "low": [9.80] * 150,
            "close": [9.90] * 150,
            "amount_cny": [100.0] * 150,
        }
    )
    frame.loc[147] = [10.0, 10.10, 9.98, 10.05, 180.0]
    frame.loc[148] = [10.01, 10.08, 9.70 if breach else 9.98, 10.03, 90.0]
    if touching:
        frame.loc[149] = [10.01, 10.06, 9.95, 10.03 if reclaimed else 9.99, 90.0]
    else:
        frame.loc[149] = [10.40, 10.50, 10.35, 10.45, 90.0]
    return frame


@pytest.mark.parametrize("holding_weeks", SUPPORTED_HOLDING_WEEKS)
def test_healthy_retest_requires_full_path_support_touch_and_reclaim(holding_weeks: int) -> None:
    contract = horizon_contract(holding_weeks)
    frame = _retest_history()
    structure = _assess_structure(frame, contract)
    execution = _assess_execution(frame, contract)
    assert structure.state is StructureState.HEALTHY_PULLBACK
    assert execution.state is ExecutionState.READY_PULLBACK
    assert structure.post_breakout_floor_held is True
    assert execution.latest_retest_touched and execution.latest_retest_reclaimed


@pytest.mark.parametrize("holding_weeks", SUPPORTED_HOLDING_WEEKS)
def test_intervening_breach_is_never_relabelled_healthy_after_recovery(holding_weeks: int) -> None:
    contract = horizon_contract(holding_weeks)
    frame = _retest_history(breach=True)
    structure = _assess_structure(frame, contract)
    execution = _assess_execution(frame, contract)
    assert structure.state is StructureState.RECLAIM_WAIT
    assert not structure.qualified
    assert execution.state is ExecutionState.WAIT_RECLAIM
    assert not execution.ready


@pytest.mark.parametrize("holding_weeks", SUPPORTED_HOLDING_WEEKS)
@pytest.mark.parametrize("conditions", ({"touching": False}, {"reclaimed": False}))
def test_holding_above_line_without_retest_or_without_reclaim_is_not_ready(
    holding_weeks: int, conditions: dict[str, bool]
) -> None:
    frame = _retest_history(**conditions)
    contract = horizon_contract(holding_weeks)
    assert _assess_structure(frame, contract).state is not StructureState.HEALTHY_PULLBACK
    assert not _assess_execution(frame, contract).ready


@pytest.mark.parametrize("holding_weeks", SUPPORTED_HOLDING_WEEKS)
def test_volume_breakout_cannot_bypass_the_pre_event_base_width_contract(
    holding_weeks: int,
) -> None:
    frame = _retest_history().iloc[:148].copy()
    frame.loc[frame.index[:-1], "low"] = 4.0
    contract = horizon_contract(holding_weeks)
    assert _assess_structure(frame, contract).state is not StructureState.BREAKOUT
    assert not _assess_execution(frame, contract).ready


@pytest.mark.parametrize("holding_weeks", SUPPORTED_HOLDING_WEEKS)
def test_compact_pre_event_base_still_allows_a_confirmed_breakout(holding_weeks: int) -> None:
    frame = _retest_history().iloc[:148].copy()
    contract = horizon_contract(holding_weeks)
    assert _assess_structure(frame, contract).state is StructureState.BREAKOUT
    assert _assess_execution(frame, contract).state is ExecutionState.READY_BREAKOUT
