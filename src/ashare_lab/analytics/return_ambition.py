"""Translate annual return ambitions into honest holding-period evidence checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from ashare_lab.analytics.portfolio_statistics import (
    HISTORICAL_OVERLAP_NOTICE,
    WilsonInterval,
    wilson_score_interval,
)

ANNUAL_RETURN_AMBITION_PCTS = tuple(range(20, 201, 20))
TRADING_SESSIONS_PER_WEEK = 5


class ReturnAmbitionStatus(StrEnum):
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNSUPPORTED = "UNSUPPORTED"
    STRETCH = "STRETCH"
    HISTORICALLY_SUPPORTED = "HISTORICALLY_SUPPORTED"


@dataclass(frozen=True, slots=True)
class ReturnAmbitionAssessment:
    status: ReturnAmbitionStatus
    annual_return_ambition_pct: int
    holding_weeks: int
    horizon_sessions: int
    horizon_return_hurdle: float
    sample_n: int
    effective_non_overlapping_n: int
    return_p10: float | None = None
    return_p50: float | None = None
    return_p90: float | None = None
    historical_hit_rate: float | None = None
    hit_rate_interval: WilsonInterval | None = None
    reason: str = ""
    method: str = "历史重叠持有期收益窗口；非walk-forward、非样本外、未经概率校准"
    is_out_of_sample: bool = False
    is_forecast_probability: bool = False
    is_promise: bool = False
    disclaimer: str = HISTORICAL_OVERLAP_NOTICE


def validate_annual_return_ambition(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("annual return ambition must be an integer percentage")
    if value not in ANNUAL_RETURN_AMBITION_PCTS:
        allowed = ", ".join(str(item) for item in ANNUAL_RETURN_AMBITION_PCTS)
        raise ValueError(f"annual return ambition must be one of: {allowed}")
    return value


def validate_holding_weeks(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("holding_weeks must be an integer")
    if not 1 <= value <= 52:
        raise ValueError("holding_weeks must be between 1 and 52")
    return value


def horizon_return_hurdle(annual_return_ambition_pct: int, holding_weeks: int) -> float:
    """Convert an annual ambition into an equivalent compounded horizon hurdle."""

    ambition = validate_annual_return_ambition(annual_return_ambition_pct) / 100.0
    weeks = validate_holding_weeks(holding_weeks)
    return float((1.0 + ambition) ** (weeks / 52.0) - 1.0)


def _clean_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    clean = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    values = clean.to_numpy()
    if not bool(np.isfinite(values).all()):
        raise ValueError("returns must contain only finite values after missing data is removed")
    if bool((values <= -1.0).any()):
        raise ValueError("a simple return cannot be less than or equal to -100%")
    return clean


def _historical_horizon_returns(returns: pd.Series, horizon_sessions: int) -> pd.Series:
    values = (1.0 + returns).rolling(horizon_sessions).apply(np.prod, raw=True) - 1.0
    return values.dropna()


def assess_return_ambition(
    returns: pd.Series,
    *,
    annual_return_ambition_pct: int,
    holding_weeks: int,
    minimum_samples: int = 30,
    confidence: float = 0.90,
) -> ReturnAmbitionAssessment:
    """Assess historical support without claiming a future achievement probability."""

    ambition = validate_annual_return_ambition(annual_return_ambition_pct)
    weeks = validate_holding_weeks(holding_weeks)
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    clean = _clean_returns(returns)
    horizon_sessions = weeks * TRADING_SESSIONS_PER_WEEK
    hurdle = horizon_return_hurdle(ambition, weeks)
    samples = _historical_horizon_returns(clean, horizon_sessions)
    sample_n = len(samples)
    effective_n = len(clean) // horizon_sessions
    if sample_n < minimum_samples:
        return ReturnAmbitionAssessment(
            status=ReturnAmbitionStatus.INSUFFICIENT_EVIDENCE,
            annual_return_ambition_pct=ambition,
            holding_weeks=weeks,
            horizon_sessions=horizon_sessions,
            horizon_return_hurdle=hurdle,
            sample_n=sample_n,
            effective_non_overlapping_n=effective_n,
            reason=(
                f"历史重叠窗口仅{sample_n}个，低于最低要求{minimum_samples}；不评估目标支持度。"
            ),
        )

    p10, p50, p90 = (float(value) for value in samples.quantile([0.10, 0.50, 0.90]))
    successes = int((samples >= hurdle).sum())
    interval = wilson_score_interval(successes, sample_n, confidence=confidence)

    if hurdle > p90 or interval.upper < 0.20:
        status = ReturnAmbitionStatus.UNSUPPORTED
        reason = "持有期门槛高于历史P90，或历史命中率区间上界低于20%。"
    elif hurdle > p50 or interval.lower < 0.35:
        status = ReturnAmbitionStatus.STRETCH
        reason = "历史样本偶尔达到该门槛，但不足以视为常态情景。"
    else:
        status = ReturnAmbitionStatus.HISTORICALLY_SUPPORTED
        reason = "历史重叠样本对该门槛有一定支持；这仍不是样本外预测或收益承诺。"

    return ReturnAmbitionAssessment(
        status=status,
        annual_return_ambition_pct=ambition,
        holding_weeks=weeks,
        horizon_sessions=horizon_sessions,
        horizon_return_hurdle=hurdle,
        sample_n=sample_n,
        effective_non_overlapping_n=effective_n,
        return_p10=p10,
        return_p50=p50,
        return_p90=p90,
        historical_hit_rate=interval.estimate,
        hit_rate_interval=interval,
        reason=reason,
    )
