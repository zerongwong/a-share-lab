"""No-lookahead multi-timeframe evidence for medium-term A-share research.

The module separates three questions that must not be collapsed into one daily
breakout score:

* the slower, completed-bar timeframe decides the primary direction;
* the holding-period-specific middle timeframe describes the base/breakout;
* complete daily bars provide the execution state for every horizon.

Weekly and monthly bars are formed only after their period is complete.  A
Friday cutoff can close a weekly bar and the last business day of a month can
close a monthly bar.  During a week or month, the partial aggregate is excluded.
This is deliberately conservative when an exchange calendar is unavailable.

Every score is a deterministic research rank in ``[0, 1]``.  It is not a
probability, expected return, or permission to trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd

MULTI_TIMEFRAME_METHOD_VERSION: Final = "multi-timeframe-core-v0.3.0"
MULTI_TIMEFRAME_IMPLEMENTATION_STATUS: Final = "analytics_core_only"

_PRICE_COLUMNS: Final = ("open", "high", "low", "close")
_REQUIRED_COLUMNS: Final = ("trade_date", *_PRICE_COLUMNS)
_ACTIVITY_COLUMNS: Final = ("amount_cny", "volume_shares")


class MultiTimeframeDataError(ValueError):
    """Raised when daily history cannot support an auditable assessment."""


class BarTimeframe(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly_completed"
    MONTHLY = "monthly_completed"


class TrendDirection(StrEnum):
    INSUFFICIENT = "insufficient"
    DOWN = "down"
    MIXED = "mixed"
    UP = "up"


class StructureState(StrEnum):
    INSUFFICIENT = "insufficient"
    FAILED = "failed"
    TREND_CONTINUATION = "trend_continuation_without_entry_structure"
    BASE = "base_not_yet_near_breakout"
    NEAR_BREAKOUT = "near_breakout"
    HEALTHY_PULLBACK = "healthy_post_breakout_pullback"
    BREAKOUT = "volume_confirmed_breakout"
    RECLAIM_WAIT = "broken_breakout_wait_for_new_structure"


class ExecutionState(StrEnum):
    INSUFFICIENT = "insufficient"
    FAILED = "failed"
    EXTENDED = "extended_do_not_chase"
    WAIT_CONFIRMATION = "wait_for_daily_confirmation"
    READY_PULLBACK = "daily_healthy_pullback_ready"
    READY_BREAKOUT = "daily_volume_breakout_ready"
    WAIT_RECLAIM = "broken_breakout_wait_for_reclaim_confirmation"


@dataclass(frozen=True, slots=True)
class HorizonContract:
    """One explicit six-horizon contract.

    ``slow_*`` controls the primary direction, ``structure_*`` controls the
    Edwards--Magee base/breakout evidence, and ``execution_*`` controls the
    complete-daily-bar entry state.  The daily anchor prevents a slow aggregate
    from hiding a broken current trend; the one-year contract explicitly uses
    MA252 as well as 12/24 completed monthly averages.
    """

    holding_weeks: int
    label: str
    slow_timeframe: BarTimeframe
    slow_fast_bars: int
    slow_slow_bars: int
    structure_timeframe: BarTimeframe
    structure_lookback_bars: int
    structure_recent_breakout_bars: int
    structure_base_bars: int
    structure_max_base_width: float
    execution_breakout_sessions: int
    execution_recent_sessions: int
    execution_ma_sessions: int
    daily_anchor_sessions: int
    relative_strength_sessions: int
    minimum_daily_sessions: int

    @property
    def audit_signature(self) -> tuple[object, ...]:
        """Stable parameters proving that horizons are not aliases."""

        return (
            self.slow_timeframe,
            self.slow_fast_bars,
            self.slow_slow_bars,
            self.structure_timeframe,
            self.structure_lookback_bars,
            self.structure_recent_breakout_bars,
            self.structure_base_bars,
            self.execution_breakout_sessions,
            self.execution_recent_sessions,
            self.execution_ma_sessions,
            self.daily_anchor_sessions,
            self.relative_strength_sessions,
            self.minimum_daily_sessions,
        )


_CONTRACTS = {
    1: HorizonContract(
        1,
        "1_week",
        BarTimeframe.WEEKLY,
        4,
        13,
        BarTimeframe.DAILY,
        20,
        5,
        12,
        0.14,
        20,
        5,
        10,
        20,
        5,
        80,
    ),
    2: HorizonContract(
        2,
        "2_weeks",
        BarTimeframe.WEEKLY,
        4,
        13,
        BarTimeframe.DAILY,
        30,
        8,
        18,
        0.16,
        30,
        8,
        10,
        20,
        10,
        95,
    ),
    4: HorizonContract(
        4,
        "1_month",
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
    ),
    13: HorizonContract(
        13,
        "3_months",
        BarTimeframe.MONTHLY,
        3,
        6,
        BarTimeframe.WEEKLY,
        13,
        4,
        8,
        0.22,
        60,
        15,
        20,
        120,
        60,
        180,
    ),
    26: HorizonContract(
        26,
        "6_months",
        BarTimeframe.MONTHLY,
        6,
        12,
        BarTimeframe.WEEKLY,
        26,
        6,
        13,
        0.28,
        90,
        20,
        40,
        120,
        120,
        280,
    ),
    52: HorizonContract(
        52,
        "1_year",
        BarTimeframe.MONTHLY,
        12,
        24,
        BarTimeframe.WEEKLY,
        52,
        8,
        26,
        0.40,
        120,
        25,
        50,
        252,
        252,
        540,
    ),
}

HORIZON_CONTRACTS: Final = MappingProxyType(_CONTRACTS)
SUPPORTED_HOLDING_WEEKS: Final = tuple(_CONTRACTS)


@dataclass(frozen=True, slots=True)
class CompletedTimeframes:
    as_of: pd.Timestamp
    data_cutoff: pd.Timestamp
    weekly_cutoff: pd.Timestamp | None
    monthly_cutoff: pd.Timestamp | None
    daily: pd.DataFrame
    weekly: pd.DataFrame
    monthly: pd.DataFrame
    incomplete_week_excluded: bool
    incomplete_month_excluded: bool

    def for_timeframe(self, timeframe: BarTimeframe) -> pd.DataFrame:
        if timeframe is BarTimeframe.DAILY:
            return self.daily
        if timeframe is BarTimeframe.WEEKLY:
            return self.weekly
        if timeframe is BarTimeframe.MONTHLY:
            return self.monthly
        raise ValueError(f"unsupported timeframe: {timeframe}")


@dataclass(frozen=True, slots=True)
class DirectionAssessment:
    timeframe: BarTimeframe
    direction: TrendDirection
    qualified: bool
    bars_available: int
    bars_required: int
    latest_close: float | None
    fast_average: float | None
    slow_average: float | None
    return_over_slow_window: float | None
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructureAssessment:
    timeframe: BarTimeframe
    state: StructureState
    qualified: bool
    bars_available: int
    lookback_bars: int
    breakout_line: float | None
    days_or_bars_since_breakout: int | None
    activity_ratio: float | None
    base_width: float | None
    score: float
    reasons: tuple[str, ...]
    base_qualified_at_breakout: bool | None = None
    post_breakout_floor_held: bool | None = None
    latest_retest_touched: bool = False
    latest_retest_reclaimed: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionAssessment:
    state: ExecutionState
    ready: bool
    breakout_line: float | None
    sessions_since_breakout: int | None
    activity_ratio: float | None
    moving_average: float | None
    distance_to_average: float | None
    score: float
    reasons: tuple[str, ...]
    base_qualified_at_breakout: bool | None = None
    post_breakout_floor_held: bool | None = None
    latest_retest_touched: bool = False
    latest_retest_reclaimed: bool = False


@dataclass(frozen=True, slots=True)
class RelativeStrengthAssessment:
    sessions: int
    available: bool
    security_return: float | None
    benchmark_return: float | None
    excess_return: float | None
    score: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MultiTimeframeAssessment:
    method_version: str
    implementation_status: str
    holding_weeks: int
    contract: HorizonContract
    as_of: pd.Timestamp
    data_cutoff: pd.Timestamp
    weekly_cutoff: pd.Timestamp | None
    monthly_cutoff: pd.Timestamp | None
    candidate_qualified: bool
    execution_ready: bool
    score: float
    slow_direction: DirectionAssessment
    structure: StructureAssessment
    execution: ExecutionAssessment
    relative_strength: RelativeStrengthAssessment
    daily_anchor_average: float | None
    above_daily_anchor: bool | None
    incomplete_week_excluded: bool
    incomplete_month_excluded: bool
    reasons: tuple[str, ...]
    slow_bar_cutoff: pd.Timestamp | None = None
    structure_bar_cutoff: pd.Timestamp | None = None


def horizon_contract(holding_weeks: int) -> HorizonContract:
    """Return the immutable contract for one supported holding horizon."""

    try:
        return HORIZON_CONTRACTS[int(holding_weeks)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("holding_weeks must be one of 1, 2, 4, 13, 26 or 52") from exc


def build_completed_timeframes(
    frame: pd.DataFrame,
    *,
    as_of: object,
) -> CompletedTimeframes:
    """Return daily data plus completed weekly and monthly aggregates.

    The input is assumed to contain complete daily bars.  Rows later than
    ``as_of`` are ignored before any price calculation.  A partial current week
    or month is not allowed to leak into its higher-timeframe aggregate.
    """

    cutoff = _normalize_as_of(as_of)
    daily = _prepare_daily(frame, cutoff)
    data_cutoff = pd.Timestamp(daily.iloc[-1]["trade_date"])
    effective_cutoff = min(cutoff, data_cutoff)

    week_end = daily["trade_date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    last_complete_friday = pd.offsets.Week(weekday=4).rollback(effective_cutoff)
    weekly, incomplete_week = _aggregate_periods(
        daily,
        period_end=week_end,
        completed_period_end=pd.Timestamp(last_complete_friday).normalize(),
    )

    month_end = daily["trade_date"].dt.to_period("M").dt.end_time.dt.normalize()
    last_business_month_end = pd.offsets.BMonthEnd().rollback(effective_cutoff)
    completed_month_end = pd.Timestamp(last_business_month_end) + pd.offsets.MonthEnd(0)
    monthly, incomplete_month = _aggregate_periods(
        daily,
        period_end=month_end,
        completed_period_end=pd.Timestamp(completed_month_end).normalize(),
    )
    return CompletedTimeframes(
        as_of=cutoff,
        data_cutoff=data_cutoff,
        weekly_cutoff=(None if weekly.empty else pd.Timestamp(weekly.iloc[-1]["period_end"])),
        monthly_cutoff=(None if monthly.empty else pd.Timestamp(monthly.iloc[-1]["period_end"])),
        daily=daily,
        weekly=weekly,
        monthly=monthly,
        incomplete_week_excluded=incomplete_week,
        incomplete_month_excluded=incomplete_month,
    )


def assess_multi_timeframe(
    frame: pd.DataFrame,
    *,
    as_of: object,
    holding_weeks: int,
    benchmark_frame: pd.DataFrame | None = None,
    signal_contract: HorizonContract | None = None,
) -> MultiTimeframeAssessment:
    """Assess slow direction, middle structure, and daily execution.

    Candidate membership and entry readiness remain separate.  A security may
    have a qualified slow/middle structure while its daily execution state says
    to wait.  This mirrors the research contract and prevents a marketable
    looking daily candle from overriding a broken primary trend.
    """

    contract = signal_contract or horizon_contract(holding_weeks)
    bars = build_completed_timeframes(frame, as_of=as_of)
    return _assess_completed_timeframes(
        bars,
        contract=contract,
        benchmark_frame=benchmark_frame,
    )


def assess_all_horizons(
    frame: pd.DataFrame,
    *,
    as_of: object,
    benchmark_frame: pd.DataFrame | None = None,
) -> tuple[MultiTimeframeAssessment, ...]:
    """Return all six contracts after preparing the source bars only once."""

    bars = build_completed_timeframes(frame, as_of=as_of)
    return tuple(
        _assess_completed_timeframes(
            bars,
            contract=HORIZON_CONTRACTS[holding_weeks],
            benchmark_frame=benchmark_frame,
        )
        for holding_weeks in SUPPORTED_HOLDING_WEEKS
    )


def _assess_completed_timeframes(
    bars: CompletedTimeframes,
    *,
    contract: HorizonContract,
    benchmark_frame: pd.DataFrame | None,
) -> MultiTimeframeAssessment:
    slow_frame = bars.for_timeframe(contract.slow_timeframe)
    structure_frame = bars.for_timeframe(contract.structure_timeframe)
    slow = _assess_direction(slow_frame, contract)
    structure = _assess_structure(
        structure_frame,
        contract,
    )
    execution = _assess_execution(bars.daily, contract)
    anchor = _daily_anchor(bars.daily, contract.daily_anchor_sessions)
    latest_close = float(bars.daily.iloc[-1]["close"])
    above_anchor = None if anchor is None else latest_close >= anchor
    relative_strength = _assess_relative_strength(
        bars.daily,
        benchmark_frame,
        as_of=bars.as_of,
        sessions=contract.relative_strength_sessions,
    )

    reasons: list[str] = [
        f"contract:{contract.label}",
        f"slow_direction:{contract.slow_timeframe.value}",
        f"middle_structure:{contract.structure_timeframe.value}",
        "fast_execution:complete_daily_bars",
        f"daily_anchor:ma{contract.daily_anchor_sessions}",
    ]
    minimum_history_ok = len(bars.daily) >= contract.minimum_daily_sessions
    if not minimum_history_ok:
        reasons.append(f"minimum_{contract.minimum_daily_sessions}_daily_sessions_required")
    if anchor is None:
        reasons.append(f"daily_ma{contract.daily_anchor_sessions}_unavailable")
    elif above_anchor:
        reasons.append(f"latest_close_at_or_above_daily_ma{contract.daily_anchor_sessions}")
    else:
        reasons.append(f"latest_close_below_daily_ma{contract.daily_anchor_sessions}")
    if contract.holding_weeks == 52:
        reasons.append("one_year_contract_uses_completed_monthly_ma12_ma24_and_daily_ma252")

    candidate_qualified = bool(
        minimum_history_ok and slow.qualified and structure.qualified and above_anchor is True
    )
    if not candidate_qualified:
        reasons.append("candidate_structure_not_qualified_for_this_horizon")
    execution_ready = candidate_qualified and execution.ready
    if candidate_qualified and not execution.ready:
        reasons.append("candidate_qualified_but_daily_execution_waits")

    score_components: list[tuple[float, float]] = [
        (0.40, slow.score),
        (0.40, structure.score),
        (0.15, execution.score),
    ]
    if relative_strength.available and relative_strength.score is not None:
        score_components.append((0.05, relative_strength.score))
    else:
        # Relative strength is optional evidence.  Its absent weight is moved
        # to structure rather than silently assigning a neutral score.
        score_components[1] = (0.45, structure.score)
    score = sum(weight * value for weight, value in score_components)

    return MultiTimeframeAssessment(
        method_version=MULTI_TIMEFRAME_METHOD_VERSION,
        implementation_status=MULTI_TIMEFRAME_IMPLEMENTATION_STATUS,
        holding_weeks=contract.holding_weeks,
        contract=contract,
        as_of=bars.as_of,
        data_cutoff=bars.data_cutoff,
        weekly_cutoff=bars.weekly_cutoff,
        monthly_cutoff=bars.monthly_cutoff,
        candidate_qualified=candidate_qualified,
        execution_ready=execution_ready,
        score=round(float(np.clip(score, 0.0, 1.0)), 6),
        slow_direction=slow,
        structure=structure,
        execution=execution,
        relative_strength=relative_strength,
        daily_anchor_average=anchor,
        above_daily_anchor=above_anchor,
        incomplete_week_excluded=bars.incomplete_week_excluded,
        incomplete_month_excluded=bars.incomplete_month_excluded,
        reasons=tuple(reasons),
        slow_bar_cutoff=_last_bar_cutoff(slow_frame),
        structure_bar_cutoff=_last_bar_cutoff(structure_frame),
    )


def _normalize_as_of(value: object) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise MultiTimeframeDataError("invalid_as_of") from exc
    if pd.isna(result):
        raise MultiTimeframeDataError("invalid_as_of")
    if result.tzinfo is not None:
        result = result.tz_localize(None)
    return result.normalize()


def _last_bar_cutoff(frame: pd.DataFrame) -> pd.Timestamp | None:
    if frame.empty or "trade_date" not in frame.columns:
        return None
    return pd.Timestamp(frame.iloc[-1]["trade_date"]).normalize()


def _prepare_daily(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise MultiTimeframeDataError("frame_must_be_dataframe")
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise MultiTimeframeDataError("missing_required_columns:" + ",".join(missing))

    # Establish the point-in-time slice before inspecting optional activity
    # fields.  Whether a future row is missing, negative or non-finite must not
    # change which activity source is used at an earlier research cutoff.
    source_columns = [
        *_REQUIRED_COLUMNS,
        *(column for column in _ACTIVITY_COLUMNS if column in frame.columns),
    ]
    prepared = frame.loc[:, source_columns].copy()
    parsed = pd.to_datetime(prepared["trade_date"], errors="coerce")
    if bool(parsed.isna().any()):
        # An invalid date cannot safely be classified as before or after the
        # cutoff, so it remains a fail-closed data error.
        raise MultiTimeframeDataError("invalid_trade_date")
    try:
        parsed = parsed.dt.tz_localize(None)
    except (AttributeError, TypeError) as exc:
        raise MultiTimeframeDataError("invalid_trade_date_timezone") from exc
    prepared["trade_date"] = parsed.dt.normalize()
    prepared = prepared.loc[prepared["trade_date"] <= cutoff].copy()
    if prepared.empty:
        raise MultiTimeframeDataError("no_complete_daily_bars_at_or_before_as_of")
    if bool(prepared["trade_date"].duplicated().any()):
        raise MultiTimeframeDataError("duplicate_trade_date")
    prepared = prepared.sort_values("trade_date").reset_index(drop=True)

    selected_columns = [*_REQUIRED_COLUMNS]
    activity_columns: list[str] = []
    partial_activity_columns: list[str] = []
    for column in _ACTIVITY_COLUMNS:
        if column not in prepared.columns:
            continue
        converted = pd.to_numeric(prepared[column], errors="coerce")
        if bool(converted.isna().all()):
            # CSMAR daily bars intentionally keep full-day volume null when the
            # licensed extract only supplies after-hours volume.  A wholly
            # unavailable optional field is omitted; amount can still provide
            # auditable activity confirmation.  Partial corruption remains a
            # hard error below.
            continue
        finite_values = converted.dropna().to_numpy(dtype=float)
        if not bool(np.isfinite(finite_values).all()) or bool((finite_values < 0).any()):
            raise MultiTimeframeDataError(f"invalid_{column}")
        if bool(converted.isna().any()):
            partial_activity_columns.append(column)
            continue
        activity_columns.append(column)
        selected_columns.append(column)
    if partial_activity_columns and not activity_columns:
        raise MultiTimeframeDataError(
            "partial_activity_without_complete_alternative:" + ",".join(partial_activity_columns)
        )
    prepared = prepared.loc[:, selected_columns].copy()

    numeric_columns = [*_PRICE_COLUMNS]
    numeric_columns.extend(activity_columns)
    numeric = prepared.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not bool(np.isfinite(numeric.to_numpy(dtype=float)).all()):
        raise MultiTimeframeDataError("non_finite_ohlcv")
    if bool((numeric.loc[:, _PRICE_COLUMNS] <= 0).any().any()):
        raise MultiTimeframeDataError("non_positive_ohlc")
    for column in _ACTIVITY_COLUMNS:
        if column in numeric and bool((numeric[column] < 0).any()):
            raise MultiTimeframeDataError(f"negative_{column}")
    if bool(
        (
            (numeric["high"] < numeric[["open", "close"]].max(axis=1))
            | (numeric["low"] > numeric[["open", "close"]].min(axis=1))
            | (numeric["high"] < numeric["low"])
        ).any()
    ):
        raise MultiTimeframeDataError("invalid_ohlc_relationship")
    for column in numeric_columns:
        prepared[column] = numeric[column].to_numpy(dtype=float)
    return prepared


def _aggregate_periods(
    daily: pd.DataFrame,
    *,
    period_end: pd.Series,
    completed_period_end: pd.Timestamp,
) -> tuple[pd.DataFrame, bool]:
    grouped_source = daily.assign(_period_end=period_end.to_numpy())
    incomplete_excluded = bool((grouped_source["_period_end"] > completed_period_end).any())
    complete = grouped_source.loc[grouped_source["_period_end"] <= completed_period_end].copy()
    output_columns = [
        "period_start",
        "period_end",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "source_rows",
        *(column for column in _ACTIVITY_COLUMNS if column in daily.columns),
    ]
    if complete.empty:
        return pd.DataFrame(columns=output_columns), incomplete_excluded

    aggregations: dict[str, tuple[str, str]] = {
        "period_start": ("trade_date", "min"),
        "trade_date": ("trade_date", "max"),
        "open": ("open", "first"),
        "high": ("high", "max"),
        "low": ("low", "min"),
        "close": ("close", "last"),
        "source_rows": ("trade_date", "size"),
    }
    for column in _ACTIVITY_COLUMNS:
        if column in complete.columns:
            aggregations[column] = (column, "sum")
    result = (
        complete.groupby("_period_end", sort=True)
        .agg(**aggregations)
        .reset_index()
        .rename(columns={"_period_end": "period_end"})
    )
    return result.loc[:, output_columns].reset_index(drop=True), incomplete_excluded


def _assess_direction(
    frame: pd.DataFrame,
    contract: HorizonContract,
) -> DirectionAssessment:
    required = contract.slow_slow_bars
    available = len(frame)
    if available < required:
        return DirectionAssessment(
            timeframe=contract.slow_timeframe,
            direction=TrendDirection.INSUFFICIENT,
            qualified=False,
            bars_available=available,
            bars_required=required,
            latest_close=None,
            fast_average=None,
            slow_average=None,
            return_over_slow_window=None,
            score=0.0,
            reasons=(f"minimum_{required}_{contract.slow_timeframe.value}_bars_required",),
        )

    close = frame["close"].astype(float)
    latest = float(close.iloc[-1])
    fast_average = float(close.tail(contract.slow_fast_bars).mean())
    slow_average = float(close.tail(contract.slow_slow_bars).mean())
    first = float(close.iloc[-contract.slow_slow_bars])
    period_return = latest / first - 1.0
    if latest > fast_average > slow_average and period_return > 0:
        direction = TrendDirection.UP
        score = 0.72 + min(0.18, max(0.0, latest / slow_average - 1.0))
        score += min(0.10, max(0.0, period_return) / 2.0)
        reasons = (
            "latest_close_above_fast_above_slow_average",
            "slow_window_return_positive",
        )
    elif latest < fast_average < slow_average and period_return < 0:
        direction = TrendDirection.DOWN
        score = 0.08
        reasons = (
            "latest_close_below_fast_below_slow_average",
            "slow_window_return_negative",
        )
    else:
        direction = TrendDirection.MIXED
        score = 0.42 if latest >= slow_average else 0.25
        reasons = ("slow_timeframe_average_order_is_mixed",)
    return DirectionAssessment(
        timeframe=contract.slow_timeframe,
        direction=direction,
        qualified=direction is TrendDirection.UP,
        bars_available=available,
        bars_required=required,
        latest_close=latest,
        fast_average=fast_average,
        slow_average=slow_average,
        return_over_slow_window=period_return,
        score=round(float(np.clip(score, 0.0, 1.0)), 6),
        reasons=reasons,
    )


def _activity_series(frame: pd.DataFrame) -> pd.Series | None:
    for column in _ACTIVITY_COLUMNS:
        if column in frame.columns:
            activity = frame[column].astype(float)
            if bool((activity > 0).any()):
                return activity
    return None


def _breakout_evidence(
    frame: pd.DataFrame,
    *,
    lookback: int,
    base_bars: int,
    maximum_base_width: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    prior_high = frame["high"].astype(float).shift(1).rolling(lookback).max()
    activity = _activity_series(frame)
    if activity is None:
        ratio = pd.Series(np.nan, index=frame.index, dtype=float)
    else:
        activity_lookback = max(3, min(20, lookback))
        reference = activity.shift(1).rolling(activity_lookback).median()
        ratio = activity / reference.replace(0.0, np.nan)
    # Apply the contract to the base preceding *each* possible breakout, not
    # merely to today's observation.  A wide, vertical run-up cannot become a
    # qualified breakout by bypassing the later compressed-base branch.
    base_high = frame["high"].astype(float).shift(1).rolling(base_bars).max()
    base_low = frame["low"].astype(float).shift(1).rolling(base_bars).min()
    compressed = (base_high / base_low - 1.0) <= maximum_base_width
    event = (
        (frame["close"].astype(float) >= prior_high * 1.002)
        & (ratio >= 1.05)
        & prior_high.notna()
        & compressed
    )
    return prior_high, ratio, event


def _retest_evidence(
    frame: pd.DataFrame, event_position: int | None, event_line: float | None
) -> tuple[bool | None, bool, bool]:
    """Require an unbroken post-breakout path and a genuine latest-bar retest.

    The existing two-percent tolerance is retained.  Closing above a line
    without touching its neighbourhood is not a pullback; recovering after an
    intervening breach is a separate, unconfirmed reclaim state.
    """

    if event_position is None or event_line is None or event_position == len(frame) - 1:
        return None, False, False
    path = frame.iloc[event_position + 1 :]
    intact = bool((path["low"].astype(float) >= event_line * 0.98).all())
    latest = frame.iloc[-1]
    touched = bool(
        float(latest["low"]) <= event_line * 1.02 and float(latest["high"]) >= event_line * 0.98
    )
    reclaimed = float(latest["close"]) >= event_line
    return intact, touched, reclaimed


def _assess_structure(
    frame: pd.DataFrame,
    contract: HorizonContract,
) -> StructureAssessment:
    lookback = contract.structure_lookback_bars
    minimum = lookback + 1
    available = len(frame)
    if available < minimum:
        return StructureAssessment(
            timeframe=contract.structure_timeframe,
            state=StructureState.INSUFFICIENT,
            qualified=False,
            bars_available=available,
            lookback_bars=lookback,
            breakout_line=None,
            days_or_bars_since_breakout=None,
            activity_ratio=None,
            base_width=None,
            score=0.0,
            reasons=(f"minimum_{minimum}_{contract.structure_timeframe.value}_bars_required",),
        )

    prior_high, activity_ratio, breakout_event = _breakout_evidence(
        frame,
        lookback=lookback,
        base_bars=contract.structure_base_bars,
        maximum_base_width=contract.structure_max_base_width,
    )
    latest_close = float(frame.iloc[-1]["close"])
    latest_line = float(prior_high.iloc[-1])
    latest_ratio = (
        float(activity_ratio.iloc[-1]) if isfinite(float(activity_ratio.iloc[-1])) else None
    )
    recent = breakout_event.tail(contract.structure_recent_breakout_bars)
    positions = np.flatnonzero(recent.to_numpy())
    event_position = None
    event_line = None
    bars_since = None
    if len(positions):
        event_position = len(frame) - len(recent) + int(positions[-1])
        event_line = float(prior_high.iloc[event_position])
        bars_since = len(frame) - 1 - event_position

    # The current bar may confirm a setup, but it must not create its own base
    # width or floor.  All structural references therefore end one bar earlier.
    base = frame.iloc[-contract.structure_base_bars - 1 : -1]
    base_low = float(base["low"].min())
    base_high = float(base["high"].max())
    base_mid = float(base["close"].median())
    base_width = base_high / base_low - 1.0
    compressed = base_width <= contract.structure_max_base_width
    trend_average = float(frame["close"].tail(max(3, lookback // 2)).mean())
    path_intact, touched, reclaimed = _retest_evidence(frame, event_position, event_line)

    if latest_close < base_low * 0.98:
        state = StructureState.FAILED
        qualified = False
        score = 0.0
        reasons = ("latest_close_broke_below_recent_structure_floor",)
    elif bool(breakout_event.iloc[-1]):
        state = StructureState.BREAKOUT
        qualified = True
        score = min(1.0, 0.90 + 0.05 * min(2.0, latest_ratio or 1.0))
        reasons = (
            "close_above_shifted_prior_high",
            "activity_at_least_1_05_of_shifted_median",
            "pre_breakout_base_width_within_contract",
        )
    elif path_intact is False:
        state = StructureState.RECLAIM_WAIT
        qualified = False
        score = 0.30
        reasons = ("post_breakout_path_breached_2pct_floor_wait_for_new_structure",)
    elif path_intact and touched and reclaimed:
        state = StructureState.HEALTHY_PULLBACK
        qualified = True
        score = 0.88
        reasons = (
            "recent_volume_breakout_found",
            "entire_post_breakout_path_holds_2pct_floor",
            "latest_bar_touches_line_neighbourhood_and_closes_at_or_above_line",
        )
    elif latest_close >= latest_line * 0.97 and compressed:
        state = StructureState.NEAR_BREAKOUT
        qualified = True
        score = 0.76
        reasons = (
            "latest_close_within_3pct_of_shifted_prior_high",
            "recent_base_width_within_contract",
        )
    elif compressed and latest_close >= base_mid:
        state = StructureState.BASE
        qualified = False
        score = 0.56
        reasons = ("base_present_but_not_yet_near_breakout",)
    else:
        state = StructureState.TREND_CONTINUATION
        qualified = False
        score = 0.48 if latest_close >= trend_average else 0.30
        reasons = ("trend_present_without_contract_entry_structure",)

    return StructureAssessment(
        timeframe=contract.structure_timeframe,
        state=state,
        qualified=qualified,
        bars_available=available,
        lookback_bars=lookback,
        breakout_line=event_line if event_line is not None else latest_line,
        days_or_bars_since_breakout=bars_since,
        activity_ratio=latest_ratio,
        base_width=base_width,
        score=round(float(np.clip(score, 0.0, 1.0)), 6),
        reasons=reasons,
        base_qualified_at_breakout=True if event_position is not None else None,
        post_breakout_floor_held=path_intact,
        latest_retest_touched=touched,
        latest_retest_reclaimed=reclaimed,
    )


def _assess_execution(
    daily: pd.DataFrame,
    contract: HorizonContract,
) -> ExecutionAssessment:
    lookback = contract.execution_breakout_sessions
    minimum = max(lookback + 1, contract.execution_ma_sessions)
    if len(daily) < minimum:
        return ExecutionAssessment(
            state=ExecutionState.INSUFFICIENT,
            ready=False,
            breakout_line=None,
            sessions_since_breakout=None,
            activity_ratio=None,
            moving_average=None,
            distance_to_average=None,
            score=0.0,
            reasons=(f"minimum_{minimum}_complete_daily_sessions_required",),
        )

    prior_high, activity_ratio, event = _breakout_evidence(
        daily,
        lookback=lookback,
        base_bars=min(lookback, contract.execution_ma_sessions),
        maximum_base_width=contract.structure_max_base_width,
    )
    latest_close = float(daily.iloc[-1]["close"])
    latest_line = float(prior_high.iloc[-1])
    moving_average = float(daily["close"].tail(contract.execution_ma_sessions).mean())
    distance = latest_close / moving_average - 1.0
    latest_ratio = (
        float(activity_ratio.iloc[-1]) if isfinite(float(activity_ratio.iloc[-1])) else None
    )
    recent = event.tail(contract.execution_recent_sessions)
    positions = np.flatnonzero(recent.to_numpy())
    event_line = None
    event_position = None
    sessions_since = None
    if len(positions):
        event_position = len(daily) - len(recent) + int(positions[-1])
        event_line = float(prior_high.iloc[event_position])
        sessions_since = len(daily) - 1 - event_position
    path_intact, touched, reclaimed = _retest_evidence(daily, event_position, event_line)

    if latest_close < moving_average * 0.97:
        state = ExecutionState.FAILED
        ready = False
        score = 0.05
        reasons = ("latest_close_below_execution_average_by_more_than_3pct",)
    elif distance > 0.15:
        state = ExecutionState.EXTENDED
        ready = False
        score = 0.15
        reasons = ("latest_close_more_than_15pct_above_execution_average",)
    elif bool(event.iloc[-1]):
        state = ExecutionState.READY_BREAKOUT
        ready = True
        score = 0.95
        reasons = (
            "daily_close_above_shifted_prior_high",
            "daily_activity_confirmation_present",
            "pre_breakout_base_width_within_contract",
        )
    elif path_intact is False:
        state = ExecutionState.WAIT_RECLAIM
        ready = False
        score = 0.30
        reasons = ("post_breakout_path_breached_2pct_floor_reclaim_is_not_healthy_pullback",)
    elif path_intact and touched and reclaimed:
        state = ExecutionState.READY_PULLBACK
        ready = True
        score = 0.88
        reasons = (
            "recent_daily_breakout_present",
            "entire_post_breakout_path_holds_2pct_floor",
            "latest_bar_touches_line_neighbourhood_and_closes_at_or_above_line",
        )
    else:
        state = ExecutionState.WAIT_CONFIRMATION
        ready = False
        score = 0.58 if latest_close >= latest_line * 0.98 else 0.42
        reasons = ("wait_for_complete_daily_breakout_or_healthy_retest",)

    return ExecutionAssessment(
        state=state,
        ready=ready,
        breakout_line=event_line if event_line is not None else latest_line,
        sessions_since_breakout=sessions_since,
        activity_ratio=latest_ratio,
        moving_average=moving_average,
        distance_to_average=distance,
        score=score,
        reasons=reasons,
        base_qualified_at_breakout=True if event_position is not None else None,
        post_breakout_floor_held=path_intact,
        latest_retest_touched=touched,
        latest_retest_reclaimed=reclaimed,
    )


def _daily_anchor(daily: pd.DataFrame, sessions: int) -> float | None:
    if len(daily) < sessions:
        return None
    result = float(daily["close"].tail(sessions).mean())
    return result if isfinite(result) and result > 0 else None


def _assess_relative_strength(
    daily: pd.DataFrame,
    benchmark_frame: pd.DataFrame | None,
    *,
    as_of: pd.Timestamp,
    sessions: int,
) -> RelativeStrengthAssessment:
    if benchmark_frame is None:
        return RelativeStrengthAssessment(
            sessions=sessions,
            available=False,
            security_return=None,
            benchmark_return=None,
            excess_return=None,
            score=None,
            reasons=("benchmark_not_supplied",),
        )
    try:
        benchmark = _prepare_benchmark_close(benchmark_frame, as_of)
    except MultiTimeframeDataError as exc:
        return RelativeStrengthAssessment(
            sessions=sessions,
            available=False,
            security_return=None,
            benchmark_return=None,
            excess_return=None,
            score=None,
            reasons=(f"benchmark_unavailable:{exc}",),
        )
    aligned = (
        daily.loc[:, ["trade_date", "close"]]
        .rename(columns={"close": "security_close"})
        .merge(
            benchmark.loc[:, ["trade_date", "close"]].rename(columns={"close": "benchmark_close"}),
            on="trade_date",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("trade_date")
    )
    if len(aligned) <= sessions:
        return RelativeStrengthAssessment(
            sessions=sessions,
            available=False,
            security_return=None,
            benchmark_return=None,
            excess_return=None,
            score=None,
            reasons=(f"minimum_{sessions + 1}_aligned_sessions_required",),
        )
    security_return = float(
        aligned.iloc[-1]["security_close"] / aligned.iloc[-sessions - 1]["security_close"] - 1.0
    )
    benchmark_return = float(
        aligned.iloc[-1]["benchmark_close"] / aligned.iloc[-sessions - 1]["benchmark_close"] - 1.0
    )
    excess = security_return - benchmark_return
    score = float(np.clip(0.5 + excess / 0.30, 0.0, 1.0))
    return RelativeStrengthAssessment(
        sessions=sessions,
        available=True,
        security_return=security_return,
        benchmark_return=benchmark_return,
        excess_return=excess,
        score=round(score, 6),
        reasons=("same-date_aligned_holding-horizon_relative_return",),
    )


def _prepare_benchmark_close(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Normalize a read-only benchmark that may expose close-only history."""

    if not isinstance(frame, pd.DataFrame) or "close" not in frame.columns:
        raise MultiTimeframeDataError("benchmark_close_required")
    if "trade_date" in frame.columns:
        raw_dates = frame["trade_date"]
    elif "date" in frame.columns:
        raw_dates = frame["date"]
    else:
        raw_dates = frame.index
    dates = pd.to_datetime(raw_dates, errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    if bool(pd.isna(dates).any()):
        raise MultiTimeframeDataError("invalid_benchmark_date")
    try:
        clean_dates = pd.DatetimeIndex(dates).tz_localize(None).normalize()
    except (AttributeError, TypeError) as exc:
        raise MultiTimeframeDataError("invalid_benchmark_timezone") from exc
    result = pd.DataFrame(
        {
            "trade_date": clean_dates,
            "close": close.to_numpy(dtype=float),
        }
    )
    result = result.loc[result["trade_date"] <= cutoff].sort_values("trade_date")
    values = result["close"].to_numpy(dtype=float)
    if result.empty:
        raise MultiTimeframeDataError("no_benchmark_bars_at_or_before_as_of")
    if bool(result["trade_date"].duplicated().any()):
        raise MultiTimeframeDataError("duplicate_benchmark_trade_date")
    if bool(result["close"].isna().any()):
        raise MultiTimeframeDataError("invalid_benchmark_close")
    if not bool(np.isfinite(values).all()) or bool((values <= 0).any()):
        raise MultiTimeframeDataError("non_positive_or_non_finite_benchmark_close")
    return result.reset_index(drop=True)
