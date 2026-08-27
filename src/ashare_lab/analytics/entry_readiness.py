"""Deterministic price-and-turnover hard gate for medium-term entry candidates.

The assessment is intentionally narrow: it recognizes only a volume-confirmed
60-session closing breakout, or a still-healthy retest of such a breakout.  The
returned score is a deterministic ranking aid, never a probability or forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from math import isfinite

import numpy as np
import pandas as pd

from ashare_lab.analytics.indicators import atr
from ashare_lab.analytics.medium_term_stage import (
    MediumTermStage,
    assess_medium_term_stage,
)

_REQUIRED_COLUMNS = ("trade_date", "open", "high", "low", "close", "amount_cny")
_PRICE_COLUMNS = ("open", "high", "low", "close")
_ANALYSIS_SESSIONS = 121
_BREAKOUT_LOOKBACK = 60
_RECENT_BREAKOUT_SESSIONS = 10
_RETEST_LOOKBACK_SESSIONS = 30
_AMOUNT_LOOKBACK = 20
_BREAKOUT_ATR_BUFFER = 0.10
_MINIMUM_BREAKOUT_AMOUNT_RATIO = 1.10
_MAXIMUM_MA20_DISTANCE_RATIO = 0.12
_MAXIMUM_MA20_DISTANCE_ATR = 3.0


class EntryPattern(StrEnum):
    """Recognized entry structures; ``NO_SIGNAL`` is the fail-closed state."""

    NO_SIGNAL = "no_signal"
    VOLUME_BREAKOUT = "volume_confirmed_breakout"
    HEALTHY_PULLBACK = "healthy_breakout_pullback"
    BREAKOUT_RECLAIM = "breakout_line_reclaim"


@dataclass(frozen=True, slots=True)
class EntryReadinessAssessment:
    ready: bool
    pattern: EntryPattern
    score: float
    stage: MediumTermStage
    data_cutoff: date | None
    breakout_line: float | None
    days_since_breakout: int | None
    breakout_amount_ratio: float | None
    distance_ma20_ratio: float | None
    distance_ma20_atr: float | None
    reasons: tuple[str, ...]


def assess_entry_readiness(
    frame: pd.DataFrame,
    *,
    as_of: date | datetime | pd.Timestamp | str,
) -> EntryReadinessAssessment:
    """Assess one candidate using only complete daily bars known by ``as_of``.

    A passing candidate must be in an early or orderly uptrend, remain close to
    MA20, and have a volume-confirmed close above the shifted prior 60-session
    high.  A breakout older than ten sessions is accepted only after a healthy
    pullback or a controlled reclaim of the original breakout line.
    """

    cutoff = _normalize_cutoff(as_of)
    if cutoff is None:
        return _failure("invalid_as_of")
    if not isinstance(frame, pd.DataFrame):
        return _failure("frame_must_be_dataframe", data_cutoff=cutoff.date())

    missing = tuple(column for column in _REQUIRED_COLUMNS if column not in frame.columns)
    if missing:
        return _failure(
            "missing_required_columns:" + ",".join(missing),
            data_cutoff=cutoff.date(),
        )

    prepared = frame.loc[:, _REQUIRED_COLUMNS].copy()
    parsed_dates = pd.to_datetime(prepared["trade_date"], errors="coerce")
    if bool(parsed_dates.isna().any()):
        return _failure("invalid_trade_date", data_cutoff=cutoff.date())
    try:
        parsed_dates = parsed_dates.dt.tz_localize(None)
    except (AttributeError, TypeError):
        return _failure("invalid_trade_date_timezone", data_cutoff=cutoff.date())
    prepared["trade_date"] = parsed_dates.dt.normalize()
    prepared = prepared.loc[prepared["trade_date"] <= cutoff].copy()
    if prepared.empty:
        return _failure("no_bars_at_or_before_as_of", data_cutoff=cutoff.date())
    if bool(prepared["trade_date"].duplicated().any()):
        return _failure("duplicate_trade_date", data_cutoff=cutoff.date())

    prepared = prepared.sort_values("trade_date").tail(_ANALYSIS_SESSIONS).reset_index(drop=True)
    actual_cutoff = pd.Timestamp(prepared.iloc[-1]["trade_date"]).date()
    if len(prepared) < _ANALYSIS_SESSIONS:
        return _failure("minimum_121_complete_sessions_required", data_cutoff=actual_cutoff)

    numeric_columns = (*_PRICE_COLUMNS, "amount_cny")
    numeric = prepared.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not bool(np.isfinite(values).all()):
        return _failure("non_finite_ohlc_or_amount", data_cutoff=actual_cutoff)
    if bool((numeric.loc[:, _PRICE_COLUMNS] <= 0).any().any()):
        return _failure("non_positive_ohlc", data_cutoff=actual_cutoff)
    if bool((numeric["amount_cny"] <= 0).any()):
        return _failure("non_positive_amount_cny", data_cutoff=actual_cutoff)
    if bool(
        (
            (numeric["high"] < numeric[["open", "close"]].max(axis=1))
            | (numeric["low"] > numeric[["open", "close"]].min(axis=1))
            | (numeric["high"] < numeric["low"])
        ).any()
    ):
        return _failure("invalid_ohlc_relationship", data_cutoff=actual_cutoff)
    for column in numeric_columns:
        prepared[column] = numeric[column].to_numpy(dtype=float)

    try:
        stage = assess_medium_term_stage(prepared["close"])
    except (TypeError, ValueError):
        return _failure("medium_term_stage_unavailable", data_cutoff=actual_cutoff)
    if stage.stage not in {MediumTermStage.EARLY_UPTREND, MediumTermStage.ORDERLY_UPTREND}:
        return _failure(
            f"stage_not_entry_ready:{stage.stage.value}",
            stage=stage.stage,
            data_cutoff=actual_cutoff,
        )

    atr14 = atr(prepared, period=14)
    latest_atr = float(atr14.iloc[-1])
    ma20 = float(prepared["close"].tail(20).mean())
    latest_close = float(prepared.iloc[-1]["close"])
    if not isfinite(latest_atr) or latest_atr <= 0 or not isfinite(ma20) or ma20 <= 0:
        return _failure(
            "ma20_or_atr_unavailable",
            stage=stage.stage,
            data_cutoff=actual_cutoff,
        )
    distance_ratio = float(latest_close / ma20 - 1.0)
    distance_atr = float((latest_close - ma20) / latest_atr)
    distance_failures: list[str] = []
    if distance_ratio > _MAXIMUM_MA20_DISTANCE_RATIO:
        distance_failures.append("distance_ma20_ratio_exceeds_12pct")
    if distance_atr > _MAXIMUM_MA20_DISTANCE_ATR:
        distance_failures.append("distance_ma20_atr_exceeds_3")
    if distance_failures:
        return _failure(
            *distance_failures,
            stage=stage.stage,
            data_cutoff=actual_cutoff,
            distance_ma20_ratio=distance_ratio,
            distance_ma20_atr=distance_atr,
        )

    # Both reference series are shifted.  The current bar can confirm a break,
    # but neither its high nor its turnover can influence its own thresholds.
    prior_high_60 = prepared["high"].shift(1).rolling(_BREAKOUT_LOOKBACK).max()
    prior_amount_median20 = prepared["amount_cny"].shift(1).rolling(_AMOUNT_LOOKBACK).median()
    amount_ratio = prepared["amount_cny"] / prior_amount_median20
    breakout_threshold = prior_high_60 + _BREAKOUT_ATR_BUFFER * atr14
    qualified = (
        (prepared["close"] >= breakout_threshold)
        & (amount_ratio >= _MINIMUM_BREAKOUT_AMOUNT_RATIO)
        & prior_high_60.notna()
        & prior_amount_median20.notna()
        & atr14.notna()
    )
    recent_candidates = qualified.tail(_RETEST_LOOKBACK_SESSIONS)
    candidate_positions = np.flatnonzero(recent_candidates.to_numpy())
    if not len(candidate_positions):
        return _failure(
            "no_volume_confirmed_breakout_within_30_sessions",
            stage=stage.stage,
            data_cutoff=actual_cutoff,
            distance_ma20_ratio=distance_ratio,
            distance_ma20_atr=distance_atr,
        )

    event_position = len(prepared) - len(recent_candidates) + int(candidate_positions[-1])
    days_since_breakout = len(prepared) - 1 - event_position
    breakout_line = float(prior_high_60.iloc[event_position])
    event_atr = float(atr14.iloc[event_position])
    event_amount_ratio = float(amount_ratio.iloc[event_position])
    if not all(isfinite(value) and value > 0 for value in (breakout_line, event_atr, event_amount_ratio)):
        return _failure(
            "breakout_evidence_unavailable",
            stage=stage.stage,
            data_cutoff=actual_cutoff,
            distance_ma20_ratio=distance_ratio,
            distance_ma20_atr=distance_atr,
        )

    post_breakout = prepared.iloc[event_position + 1 :]
    if not post_breakout.empty:
        structural_floor = breakout_line - event_atr
        if bool((post_breakout["close"] < structural_floor).any()):
            return _failure(
                "breakout_structure_failed_below_one_atr",
                stage=stage.stage,
                data_cutoff=actual_cutoff,
                breakout_line=breakout_line,
                days_since_breakout=days_since_breakout,
                breakout_amount_ratio=event_amount_ratio,
                distance_ma20_ratio=distance_ratio,
                distance_ma20_atr=distance_atr,
            )
    if latest_close < breakout_line:
        return _failure(
            "latest_close_below_breakout_line",
            stage=stage.stage,
            data_cutoff=actual_cutoff,
            breakout_line=breakout_line,
            days_since_breakout=days_since_breakout,
            breakout_amount_ratio=event_amount_ratio,
            distance_ma20_ratio=distance_ratio,
            distance_ma20_atr=distance_atr,
        )

    pattern = _classify_pattern(
        prepared,
        atr14,
        event_position=event_position,
        breakout_line=breakout_line,
        event_atr=event_atr,
        days_since_breakout=days_since_breakout,
    )
    if pattern is EntryPattern.NO_SIGNAL:
        return _failure(
            "breakout_lacks_healthy_retest_or_recent_confirmation",
            stage=stage.stage,
            data_cutoff=actual_cutoff,
            breakout_line=breakout_line,
            days_since_breakout=days_since_breakout,
            breakout_amount_ratio=event_amount_ratio,
            distance_ma20_ratio=distance_ratio,
            distance_ma20_atr=distance_atr,
        )

    score = _deterministic_score(
        pattern=pattern,
        stage_quality=stage.quality_score,
        amount_ratio=event_amount_ratio,
        distance_ratio=distance_ratio,
        distance_atr=distance_atr,
    )
    return EntryReadinessAssessment(
        ready=True,
        pattern=pattern,
        score=score,
        stage=stage.stage,
        data_cutoff=actual_cutoff,
        breakout_line=breakout_line,
        days_since_breakout=days_since_breakout,
        breakout_amount_ratio=event_amount_ratio,
        distance_ma20_ratio=distance_ratio,
        distance_ma20_atr=distance_atr,
        reasons=(
            f"stage_allowed:{stage.stage.value}",
            "shifted_prior_60_high_close_confirmed",
            "breakout_amount_ratio_at_least_1_10",
            "ma20_distance_within_12pct_and_3atr",
            f"pattern_confirmed:{pattern.value}",
            "score_is_deterministic_not_probability",
        ),
    )


def _classify_pattern(
    prepared: pd.DataFrame,
    atr14: pd.Series,
    *,
    event_position: int,
    breakout_line: float,
    event_atr: float,
    days_since_breakout: int,
) -> EntryPattern:
    if days_since_breakout == 0:
        return EntryPattern.VOLUME_BREAKOUT

    post = prepared.iloc[event_position + 1 :]
    post_atr = atr14.iloc[event_position + 1 :]
    previous_close = prepared["close"].shift(1).iloc[event_position + 1 :]
    reclaimed = (
        (post["close"] >= breakout_line + 0.05 * post_atr)
        & (previous_close < breakout_line)
    )
    if bool(reclaimed.any()) and float(post.iloc[-1]["close"]) >= breakout_line:
        return EntryPattern.BREAKOUT_RECLAIM

    touched = float(post["low"].min()) <= breakout_line + 0.50 * event_atr
    held = float(post["close"].min()) >= breakout_line - 0.50 * event_atr
    latest_near_line = (
        float(post.iloc[-1]["close"])
        <= breakout_line + 1.50 * float(post_atr.iloc[-1])
    )
    if touched and held and latest_near_line:
        return EntryPattern.HEALTHY_PULLBACK
    if days_since_breakout < _RECENT_BREAKOUT_SESSIONS:
        return EntryPattern.VOLUME_BREAKOUT
    return EntryPattern.NO_SIGNAL


def _deterministic_score(
    *,
    pattern: EntryPattern,
    stage_quality: float,
    amount_ratio: float,
    distance_ratio: float,
    distance_atr: float,
) -> float:
    pattern_quality = {
        EntryPattern.VOLUME_BREAKOUT: 0.85,
        EntryPattern.HEALTHY_PULLBACK: 0.95,
        EntryPattern.BREAKOUT_RECLAIM: 0.88,
    }[pattern]
    volume_quality = min(1.0, amount_ratio / 1.50)
    distance_pressure = max(
        max(0.0, distance_ratio) / _MAXIMUM_MA20_DISTANCE_RATIO,
        max(0.0, distance_atr) / _MAXIMUM_MA20_DISTANCE_ATR,
    )
    distance_quality = max(0.0, 1.0 - distance_pressure)
    score = (
        0.40 * stage_quality
        + 0.35 * pattern_quality
        + 0.15 * volume_quality
        + 0.10 * distance_quality
    )
    return round(min(1.0, max(0.0, score)), 4)


def _normalize_cutoff(value: object) -> pd.Timestamp | None:
    try:
        cutoff = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(cutoff):
        return None
    if cutoff.tz is not None:
        cutoff = cutoff.tz_localize(None)
    return cutoff.normalize()


def _failure(
    *reasons: str,
    stage: MediumTermStage = MediumTermStage.INSUFFICIENT,
    data_cutoff: date | None = None,
    breakout_line: float | None = None,
    days_since_breakout: int | None = None,
    breakout_amount_ratio: float | None = None,
    distance_ma20_ratio: float | None = None,
    distance_ma20_atr: float | None = None,
) -> EntryReadinessAssessment:
    return EntryReadinessAssessment(
        ready=False,
        pattern=EntryPattern.NO_SIGNAL,
        score=0.0,
        stage=stage,
        data_cutoff=data_cutoff,
        breakout_line=breakout_line,
        days_since_breakout=days_since_breakout,
        breakout_amount_ratio=breakout_amount_ratio,
        distance_ma20_ratio=distance_ma20_ratio,
        distance_ma20_atr=distance_ma20_atr,
        reasons=tuple(reasons),
    )
