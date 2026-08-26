"""Deterministic multi-horizon trend-stage guard for medium-term entries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

import numpy as np
import pandas as pd


class MediumTermStage(StrEnum):
    INSUFFICIENT = "insufficient"
    DOWNTREND = "downtrend"
    RANGE = "range"
    EARLY_UPTREND = "early_uptrend"
    ORDERLY_UPTREND = "orderly_uptrend"
    EXTENDED = "extended"
    PARABOLIC = "parabolic"


@dataclass(frozen=True, slots=True)
class MediumTermStageAssessment:
    stage: MediumTermStage
    quality_score: float
    hard_freeze_new_entry: bool
    reasons: tuple[str, ...]
    return_5: float | None
    return_20: float | None
    return_60: float | None
    return_120: float | None
    distance_ma20: float | None
    distance_ma60: float | None
    distance_ma120: float | None


def _clean_close(close: pd.Series) -> pd.Series:
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series")
    clean = pd.to_numeric(close, errors="coerce").dropna().astype(float)
    if clean.empty or not bool(np.isfinite(clean.to_numpy()).all()) or bool((clean <= 0).any()):
        raise ValueError("close must contain finite positive prices")
    return clean


def _return(clean: pd.Series, sessions: int) -> float | None:
    if len(clean) <= sessions:
        return None
    return float(clean.iloc[-1] / clean.iloc[-sessions - 1] - 1.0)


def _distance_to_average(clean: pd.Series, sessions: int) -> float | None:
    if len(clean) < sessions:
        return None
    average = float(clean.tail(sessions).mean())
    return float(clean.iloc[-1] / average - 1.0) if average > 0 else None


def assess_medium_term_stage(close: pd.Series) -> MediumTermStageAssessment:
    """Classify entry maturity across 5/20/60/120 sessions.

    The hard gate targets late acceleration, not ordinary strength.  It is a
    formation-date entry guard and must not be interpreted as a sell signal for
    an existing position.
    """

    clean = _clean_close(close)
    r5, r20, r60, r120 = (_return(clean, sessions) for sessions in (5, 20, 60, 120))
    d20, d60, d120 = (_distance_to_average(clean, sessions) for sessions in (20, 60, 120))
    values = (r20, r60, r120, d20, d60, d120)
    if len(clean) < 121 or any(value is None or not isfinite(value) for value in values):
        return MediumTermStageAssessment(
            stage=MediumTermStage.INSUFFICIENT,
            quality_score=0.0,
            hard_freeze_new_entry=False,
            reasons=("minimum_121_sessions_required",),
            return_5=r5,
            return_20=r20,
            return_60=r60,
            return_120=r120,
            distance_ma20=d20,
            distance_ma60=d60,
            distance_ma120=d120,
        )

    assert all(value is not None for value in values)
    reasons: list[str] = []
    if r5 is not None and r5 > 0.25 and r20 > 0.35:
        reasons.append("five_session_vertical_acceleration")
    if r20 > 0.35 and d20 > 0.15:
        reasons.append("twenty_session_price_ma_extension")
    if r60 > 0.70 and d60 > 0.28 and r20 > 0.12:
        reasons.append("sixty_session_late_acceleration")
    if r120 > 1.20 and d120 > 0.45 and r20 > 0.20:
        reasons.append("one_twenty_session_parabolic_extension")

    hard_freeze = bool(reasons)
    latest = float(clean.iloc[-1])
    ma20 = float(clean.tail(20).mean())
    ma60 = float(clean.tail(60).mean())
    ma120 = float(clean.tail(120).mean())
    ordered_uptrend = latest > ma20 > ma60 > ma120 and r20 > 0 and r60 > 0
    ordered_downtrend = latest < ma20 < ma60 < ma120 and r20 < 0 and r60 < 0
    extended = d20 > 0.10 or d60 > 0.18 or r60 > 0.50 or r120 > 0.85

    if hard_freeze:
        stage = MediumTermStage.PARABOLIC
        quality = 0.0
    elif ordered_downtrend:
        stage = MediumTermStage.DOWNTREND
        quality = 0.05
    elif ordered_uptrend and extended:
        stage = MediumTermStage.EXTENDED
        quality = 0.20
        reasons.append("trend_intact_but_entry_extension_is_high")
    elif ordered_uptrend and r120 <= 0.15:
        stage = MediumTermStage.EARLY_UPTREND
        quality = 0.85
    elif ordered_uptrend:
        stage = MediumTermStage.ORDERLY_UPTREND
        quality = 1.0
    else:
        stage = MediumTermStage.RANGE
        quality = 0.35

    return MediumTermStageAssessment(
        stage=stage,
        quality_score=quality,
        hard_freeze_new_entry=hard_freeze,
        reasons=tuple(reasons),
        return_5=r5,
        return_20=r20,
        return_60=r60,
        return_120=r120,
        distance_ma20=d20,
        distance_ma60=d60,
        distance_ma120=d120,
    )
