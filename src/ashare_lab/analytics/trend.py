from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ashare_lab.domain.enums import TrendState


@dataclass(frozen=True)
class SwingPoint:
    index: int
    trade_date: pd.Timestamp
    confirmed_at: pd.Timestamp
    kind: str
    price: float


def confirmed_swings(
    frame: pd.DataFrame,
    *,
    left: int = 2,
    right: int = 2,
) -> list[SwingPoint]:
    """Return pivots only after the right-hand confirmation bars exist."""
    ordered = frame.sort_values("trade_date").reset_index(drop=True)
    points: list[SwingPoint] = []
    for index in range(left, len(ordered) - right):
        window = ordered.iloc[index - left : index + right + 1]
        row = ordered.iloc[index]
        confirmed_at = pd.Timestamp(ordered.iloc[index + right]["trade_date"])
        if row["high"] == window["high"].max() and int((window["high"] == row["high"]).sum()) == 1:
            points.append(
                SwingPoint(
                    index, pd.Timestamp(row["trade_date"]), confirmed_at, "high", float(row["high"])
                )
            )
        if row["low"] == window["low"].min() and int((window["low"] == row["low"]).sum()) == 1:
            points.append(
                SwingPoint(
                    index, pd.Timestamp(row["trade_date"]), confirmed_at, "low", float(row["low"])
                )
            )
    return points


def classify_trend(frame: pd.DataFrame) -> TrendState:
    if len(frame) < 60:
        return TrendState.INSUFFICIENT
    latest = frame.iloc[-1]
    required = ("close", "ma20", "ma60")
    if any(pd.isna(latest.get(key)) for key in required):
        return TrendState.INSUFFICIENT

    recent = confirmed_swings(frame.tail(80), left=2, right=2)
    highs = [point.price for point in recent if point.kind == "high"][-2:]
    lows = [point.price for point in recent if point.kind == "low"][-2:]
    higher_structure = (
        len(highs) == 2 and len(lows) == 2 and highs[-1] > highs[-2] and lows[-1] > lows[-2]
    )
    lower_structure = (
        len(highs) == 2 and len(lows) == 2 and highs[-1] < highs[-2] and lows[-1] < lows[-2]
    )

    if latest["close"] > latest["ma20"] > latest["ma60"] and higher_structure:
        return TrendState.UP
    if latest["close"] < latest["ma20"] < latest["ma60"] and lower_structure:
        return TrendState.DOWN
    return TrendState.RANGE


def breakout_evidence(frame: pd.DataFrame) -> dict[str, object]:
    latest = frame.iloc[-1]
    atr_value = float(latest.get("atr14", 0) or 0)
    prior_high = float(latest.get("prior_high_60", float("nan")))
    close = float(latest["close"])
    threshold = prior_high + 0.10 * atr_value
    volume_confirmed = False
    if "volume_shares" in frame and "volume_median20" in frame:
        average = latest.get("volume_median20")
        volume_confirmed = bool(
            pd.notna(average) and average > 0 and latest["volume_shares"] >= 1.2 * average
        )
    return {
        "is_breakout": bool(pd.notna(prior_high) and close > threshold and volume_confirmed),
        "prior_high": prior_high,
        "threshold": threshold,
        "volume_confirmed": volume_confirmed,
    }
