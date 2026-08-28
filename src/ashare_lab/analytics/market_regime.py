"""Point-in-time A-share breadth evidence for portfolio-level risk posture."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd


class MarketRegimeState(StrEnum):
    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class MarketRegimeAssessment:
    state: MarketRegimeState
    score: float | None
    cutoff: pd.Timestamp
    eligible_symbols: int
    breadth_above_ma20: float | None = None
    breadth_above_ma60: float | None = None
    breadth_above_ma120: float | None = None
    median_return_20: float | None = None
    median_return_60: float | None = None
    reason: str = ""
    method: str = "共同截止日全市场价格宽度；只使用截止日及之前的日线"


def _as_close(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series | None:
    if not isinstance(frame, pd.DataFrame) or "close" not in frame:
        return None
    if "trade_date" in frame:
        dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    elif "date" in frame:
        dates = pd.to_datetime(frame["date"], errors="coerce")
    else:
        dates = pd.to_datetime(frame.index, errors="coerce")
    values = pd.to_numeric(frame["close"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates).dropna().sort_index()
    series = series.loc[series.index <= cutoff]
    if len(series) < 121 or series.index[-1].normalize() != cutoff:
        return None
    if bool((series <= 0).any()) or not bool(np.isfinite(series.to_numpy()).all()):
        return None
    return series


def _scale_return(value: float, lower: float, upper: float) -> float:
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def assess_market_regime(
    histories: Mapping[str, pd.DataFrame],
    cutoff: object,
    *,
    minimum_symbols: int = 4,
) -> MarketRegimeAssessment:
    """Measure market participation; do not use one index as a market proxy."""

    stamp = pd.Timestamp(cutoff)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    stamp = stamp.normalize()
    rows: list[tuple[float, float, float, float, float]] = []
    for frame in histories.values():
        close = _as_close(frame, stamp)
        if close is None:
            continue
        latest = float(close.iloc[-1])
        rows.append(
            (
                float(latest > close.tail(20).mean()),
                float(latest > close.tail(60).mean()),
                float(latest > close.tail(120).mean()),
                float(latest / close.iloc[-21] - 1.0),
                float(latest / close.iloc[-61] - 1.0),
            )
        )
    if len(rows) < minimum_symbols:
        return MarketRegimeAssessment(
            state=MarketRegimeState.UNAVAILABLE,
            score=None,
            cutoff=stamp,
            eligible_symbols=len(rows),
            reason=f"only_{len(rows)}_symbols_with_121_session_common_cutoff_history",
        )

    values = np.asarray(rows, dtype=float)
    breadth20, breadth60, breadth120 = (float(values[:, i].mean()) for i in range(3))
    median20 = float(np.median(values[:, 3]))
    median60 = float(np.median(values[:, 4]))
    score = float(
        0.25 * breadth20
        + 0.25 * breadth60
        + 0.20 * breadth120
        + 0.15 * _scale_return(median20, -0.08, 0.08)
        + 0.15 * _scale_return(median60, -0.15, 0.15)
    )
    if score >= 0.65:
        state = MarketRegimeState.RISK_ON
        reason = "broad_participation_and_multihorizon_trend_support_new_research_entries"
    elif score >= 0.35:
        state = MarketRegimeState.NEUTRAL
        reason = "mixed_market_participation_requires_selectivity_and_cash_reserve"
    else:
        state = MarketRegimeState.RISK_OFF
        reason = "weak_breadth_requires_defensive_exposure_and_stricter_entry_confirmation"
    return MarketRegimeAssessment(
        state=state,
        score=score,
        cutoff=stamp,
        eligible_symbols=len(rows),
        breadth_above_ma20=breadth20,
        breadth_above_ma60=breadth60,
        breadth_above_ma120=breadth120,
        median_return_20=median20,
        median_return_60=median60,
        reason=reason,
    )
