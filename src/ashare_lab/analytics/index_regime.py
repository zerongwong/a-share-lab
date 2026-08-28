"""Deterministic core-index evidence for portfolio-level risk posture.

This module deliberately produces no security-level factor.  Its result is a
cycle input built from multiple core A-share indices observed on one common
cutoff and one common recent-session calendar.  Risk-off tightens exposure and
entry confirmation; it does not suppress an otherwise valid stock screen.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

_MINIMUM_OBSERVATIONS = 121
_ANNUAL_SESSIONS = 252


class IndexRegimeState(StrEnum):
    """Portfolio-level core-index confirmation state."""

    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class CoreIndexMetrics:
    """Transparent inputs used to form the aggregate regime gate."""

    index_code: str
    above_ma20: bool
    above_ma60: bool
    above_ma120: bool
    return_20: float
    return_60: float
    annualized_volatility_60: float
    max_drawdown_120: float


@dataclass(frozen=True, slots=True)
class IndexRegimeAssessment:
    """Aggregate confirmation; ``score`` must never be added to a stock score."""

    state: IndexRegimeState
    score: float | None
    cutoff: pd.Timestamp | None
    eligible_indices: int
    required_indices: int
    breadth_above_ma20: float | None = None
    breadth_above_ma60: float | None = None
    breadth_above_ma120: float | None = None
    median_return_20: float | None = None
    median_return_60: float | None = None
    median_annualized_volatility_60: float | None = None
    median_max_drawdown_120: float | None = None
    worst_max_drawdown_120: float | None = None
    index_metrics: tuple[CoreIndexMetrics, ...] = ()
    reason: str = ""
    method: str = (
        "组合级核心指数确认；共同截止日与共同121交易日；"
        "MA20/60/120、20/60日收益、60日年化波动、120日最大回撤"
    )


@dataclass(frozen=True, slots=True)
class _PreparedIndex:
    index_code: str
    close: pd.Series
    declared_common_cutoff: pd.Timestamp | None


def _normalise_stamp(value: object) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _unavailable(
    *,
    reason: str,
    cutoff: pd.Timestamp | None,
    eligible_indices: int,
    required_indices: int,
) -> IndexRegimeAssessment:
    return IndexRegimeAssessment(
        state=IndexRegimeState.UNAVAILABLE,
        score=None,
        cutoff=cutoff,
        eligible_indices=eligible_indices,
        required_indices=required_indices,
        reason=reason,
    )


def _prepare_index(index_code: str, frame: pd.DataFrame) -> _PreparedIndex | str:
    if not isinstance(frame, pd.DataFrame):
        return f"{index_code}:not_a_dataframe"
    missing = {"trade_date", "close"}.difference(frame.columns)
    if missing:
        return f"{index_code}:missing_columns_{'_'.join(sorted(missing))}"
    if frame.empty:
        return f"{index_code}:empty_history"

    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    closes = pd.to_numeric(frame["close"], errors="coerce")
    if bool(dates.isna().any()) or bool(closes.isna().any()):
        return f"{index_code}:invalid_date_or_close"
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    dates = dates.dt.normalize()
    values = closes.to_numpy(dtype=float)
    if not bool(np.isfinite(values).all()) or bool((values <= 0.0).any()):
        return f"{index_code}:non_positive_or_non_finite_close"
    if bool(dates.duplicated().any()):
        return f"{index_code}:duplicate_trade_date"

    declared_cutoff: pd.Timestamp | None = None
    if "common_cutoff_date" in frame:
        declared = pd.to_datetime(frame["common_cutoff_date"], errors="coerce")
        if bool(declared.isna().any()):
            return f"{index_code}:invalid_common_cutoff_metadata"
        if declared.dt.tz is not None:
            declared = declared.dt.tz_localize(None)
        unique_declared = declared.dt.normalize().drop_duplicates()
        if len(unique_declared) != 1:
            return f"{index_code}:inconsistent_common_cutoff_metadata"
        declared_cutoff = pd.Timestamp(unique_declared.iloc[0])

    if "historical_backtest_eligible" in frame:
        eligible = frame["historical_backtest_eligible"]
        if bool(eligible.isna().any()) or not bool(eligible.astype(bool).all()):
            return f"{index_code}:not_historical_backtest_eligible"

    close = pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float).sort_index()
    return _PreparedIndex(
        index_code=str(index_code),
        close=close,
        declared_common_cutoff=declared_cutoff,
    )


def _scaled(value: float, lower: float, upper: float) -> float:
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def _inverse_scaled(value: float, lower: float, upper: float) -> float:
    return float(1.0 - np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def _maximum_drawdown(close: pd.Series) -> float:
    equity = close / float(close.iloc[0])
    return float((equity / equity.cummax() - 1.0).min())


def assess_index_regime(
    histories: Mapping[str, pd.DataFrame],
    cutoff: object | None = None,
    *,
    minimum_indices: int = 3,
) -> IndexRegimeAssessment:
    """Confirm the core-index environment without creating a stock factor.

    If ``cutoff`` is omitted, the latest date available in every supplied
    history is used.  All supplied histories are mandatory: a malformed,
    stale, short, or calendar-misaligned core index makes the result
    ``unavailable`` instead of silently reducing the confirmation basket.
    """

    if minimum_indices < 2:
        raise ValueError("minimum_indices must be at least 2")
    if not isinstance(histories, Mapping) or len(histories) < minimum_indices:
        count = len(histories) if isinstance(histories, Mapping) else 0
        return _unavailable(
            reason=f"insufficient_core_indices:{count}_of_{minimum_indices}",
            cutoff=_normalise_stamp(cutoff) if cutoff is not None else None,
            eligible_indices=0,
            required_indices=max(count, minimum_indices),
        )

    prepared: list[_PreparedIndex] = []
    for raw_code in sorted(histories, key=str):
        code = str(raw_code)
        result = _prepare_index(code, histories[raw_code])
        if isinstance(result, str):
            return _unavailable(
                reason=result,
                cutoff=_normalise_stamp(cutoff) if cutoff is not None else None,
                eligible_indices=len(prepared),
                required_indices=len(histories),
            )
        prepared.append(result)

    metadata_cutoffs = {
        item.declared_common_cutoff for item in prepared if item.declared_common_cutoff is not None
    }
    if len(metadata_cutoffs) > 1:
        return _unavailable(
            reason="core_indices_have_different_common_cutoff_metadata",
            cutoff=_normalise_stamp(cutoff) if cutoff is not None else None,
            eligible_indices=0,
            required_indices=len(prepared),
        )

    declared_cutoff = next(iter(metadata_cutoffs), None)
    if cutoff is not None:
        common_cutoff = _normalise_stamp(cutoff)
        if common_cutoff is None:
            return _unavailable(
                reason="invalid_requested_cutoff",
                cutoff=None,
                eligible_indices=0,
                required_indices=len(prepared),
            )
    elif declared_cutoff is not None:
        common_cutoff = declared_cutoff
    else:
        shared_dates = set(prepared[0].close.index)
        for item in prepared[1:]:
            shared_dates.intersection_update(item.close.index)
        if not shared_dates:
            return _unavailable(
                reason="no_common_trade_date",
                cutoff=None,
                eligible_indices=0,
                required_indices=len(prepared),
            )
        common_cutoff = max(shared_dates)

    if declared_cutoff is not None and common_cutoff > declared_cutoff:
        return _unavailable(
            reason="requested_cutoff_after_declared_common_cutoff",
            cutoff=common_cutoff,
            eligible_indices=0,
            required_indices=len(prepared),
        )

    trimmed: dict[str, pd.Series] = {}
    for item in prepared:
        close = item.close.loc[item.close.index <= common_cutoff]
        if close.empty or close.index[-1] != common_cutoff:
            return _unavailable(
                reason=f"{item.index_code}:missing_common_cutoff_observation",
                cutoff=common_cutoff,
                eligible_indices=len(trimmed),
                required_indices=len(prepared),
            )
        if len(close) < _MINIMUM_OBSERVATIONS:
            return _unavailable(
                reason=f"{item.index_code}:only_{len(close)}_observations",
                cutoff=common_cutoff,
                eligible_indices=len(trimmed),
                required_indices=len(prepared),
            )
        trimmed[item.index_code] = close

    reference_calendar = next(iter(trimmed.values())).index[-_MINIMUM_OBSERVATIONS:]
    for code, close in trimmed.items():
        if not close.index[-_MINIMUM_OBSERVATIONS:].equals(reference_calendar):
            return _unavailable(
                reason=f"{code}:recent_session_calendar_mismatch",
                cutoff=common_cutoff,
                eligible_indices=0,
                required_indices=len(prepared),
            )

    metrics: list[CoreIndexMetrics] = []
    for code, close in trimmed.items():
        recent = close.iloc[-_MINIMUM_OBSERVATIONS:]
        latest = float(recent.iloc[-1])
        daily_returns = recent.pct_change(fill_method=None).dropna()
        volatility = float(daily_returns.tail(60).std(ddof=1) * np.sqrt(_ANNUAL_SESSIONS))
        metrics.append(
            CoreIndexMetrics(
                index_code=code,
                above_ma20=latest > float(recent.tail(20).mean()),
                above_ma60=latest > float(recent.tail(60).mean()),
                above_ma120=latest > float(recent.tail(120).mean()),
                return_20=float(latest / recent.iloc[-21] - 1.0),
                return_60=float(latest / recent.iloc[-61] - 1.0),
                annualized_volatility_60=volatility,
                max_drawdown_120=_maximum_drawdown(recent.tail(120)),
            )
        )

    breadth20 = float(np.mean([item.above_ma20 for item in metrics]))
    breadth60 = float(np.mean([item.above_ma60 for item in metrics]))
    breadth120 = float(np.mean([item.above_ma120 for item in metrics]))
    median20 = float(np.median([item.return_20 for item in metrics]))
    median60 = float(np.median([item.return_60 for item in metrics]))
    median_volatility = float(np.median([item.annualized_volatility_60 for item in metrics]))
    drawdowns = [item.max_drawdown_120 for item in metrics]
    median_drawdown = float(np.median(drawdowns))
    worst_drawdown = float(min(drawdowns))

    score = float(
        0.14 * breadth20
        + 0.18 * breadth60
        + 0.18 * breadth120
        + 0.12 * _scaled(median20, -0.08, 0.08)
        + 0.13 * _scaled(median60, -0.15, 0.15)
        + 0.12 * _inverse_scaled(median_volatility, 0.10, 0.35)
        + 0.13 * _inverse_scaled(abs(median_drawdown), 0.05, 0.25)
    )

    stressed = (
        (breadth60 <= 1.0 / 3.0 and breadth120 <= 1.0 / 3.0 and median60 < 0.0)
        or median60 <= -0.12
        or median_drawdown <= -0.18
        or worst_drawdown <= -0.25
    )
    confirmed = (
        score >= 0.65
        and breadth60 >= 2.0 / 3.0
        and breadth120 >= 2.0 / 3.0
        and median20 > 0.0
        and median60 > 0.0
        and median_drawdown > -0.15
        and worst_drawdown > -0.20
    )
    if stressed or score < 0.34:
        state = IndexRegimeState.RISK_OFF
        reason = "core_index_trend_or_downside_risk_requires_defensive_posture"
    elif confirmed:
        state = IndexRegimeState.RISK_ON
        reason = "multiple_core_indices_confirm_trend_with_controlled_volatility_and_drawdown"
    else:
        state = IndexRegimeState.NEUTRAL
        reason = "core_index_evidence_is_mixed_or_not_strong_enough_for_risk_on_confirmation"

    return IndexRegimeAssessment(
        state=state,
        score=score,
        cutoff=common_cutoff,
        eligible_indices=len(metrics),
        required_indices=len(prepared),
        breadth_above_ma20=breadth20,
        breadth_above_ma60=breadth60,
        breadth_above_ma120=breadth120,
        median_return_20=median20,
        median_return_60=median60,
        median_annualized_volatility_60=median_volatility,
        median_max_drawdown_120=median_drawdown,
        worst_max_drawdown_120=worst_drawdown,
        index_metrics=tuple(metrics),
        reason=reason,
    )
