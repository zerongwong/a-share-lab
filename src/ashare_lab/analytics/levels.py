from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import pandas as pd

from ashare_lab.analytics.trend import confirmed_swings


@dataclass(frozen=True)
class HorizonLevels:
    horizon: str
    sessions: int
    reference_price: float
    pullback_entry_low: float
    pullback_entry_high: float
    entry_trigger: float
    breakout_trigger: float
    breakout_confirmation_rule: str
    reduce_low: float
    reduce_high: float
    second_reduce_low: float
    second_reduce_high: float
    first_reduce_fraction: float
    invalidation: float
    stop_execution_rule: str
    measured_move_target: float
    reward_risk_ratio: float | None
    time_stop_sessions: int
    level_method: str
    level_evidence_dates: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _nearest_levels(frame: pd.DataFrame, lookback: int) -> tuple[float, float, tuple[str, ...]]:
    window = frame.tail(min(lookback, len(frame)))
    current = float(window.iloc[-1]["close"])
    points = confirmed_swings(window, left=2, right=2)
    supports = [point for point in points if point.kind == "low" and point.price < current]
    resistances = [point for point in points if point.kind == "high" and point.price > current]

    if supports:
        support_point = max(supports, key=lambda point: point.price)
        support = support_point.price
        support_date = support_point.trade_date
    else:
        support_index = window["low"].astype(float).idxmin()
        support = float(window.loc[support_index, "low"])
        support_date = pd.Timestamp(window.loc[support_index, "trade_date"])

    if resistances:
        resistance_point = min(resistances, key=lambda point: point.price)
        resistance = resistance_point.price
        resistance_date = resistance_point.trade_date
    else:
        resistance_index = window["high"].astype(float).idxmax()
        resistance = float(window.loc[resistance_index, "high"])
        resistance_date = pd.Timestamp(window.loc[resistance_index, "trade_date"])

    evidence = tuple(
        sorted(
            {
                pd.Timestamp(support_date).date().isoformat(),
                pd.Timestamp(resistance_date).date().isoformat(),
            }
        )
    )
    return support, resistance, evidence


def build_horizon_levels(frame: pd.DataFrame) -> list[HorizonLevels]:
    latest = frame.iloc[-1]
    reference = float(latest["close"])
    atr_value = float(latest["atr14"])
    if not isfinite(atr_value) or atr_value <= 0:
        atr_value = reference * 0.02
    definitions = (
        ("一周", 5, 30),
        ("一月", 20, 60),
        ("三个月", 60, 120),
        ("六个月", 120, 250),
        ("一年", 252, 250),
    )
    results: list[HorizonLevels] = []
    for label, sessions, lookback in definitions:
        support, resistance, evidence_dates = _nearest_levels(frame, lookback)
        if resistance <= support:
            resistance = max(reference, support) + 1.5 * atr_value
        entry_low = support
        entry_high = min(reference, support + 0.50 * atr_value)
        entry_trigger = min(resistance, entry_high + 0.25 * atr_value)
        breakout_buffer = max(0.25 * atr_value, 0.005 * resistance)
        breakout_trigger = resistance + breakout_buffer
        stop_buffer = max(0.50 * atr_value, 0.01 * support)
        invalidation = support - stop_buffer
        pattern_height = max(resistance - support, 1.5 * atr_value)
        measured_target = resistance + pattern_height
        entry_reference = (entry_low + entry_high) / 2
        planned_risk = entry_reference - invalidation
        planned_reward = measured_target - entry_reference
        reward_risk = planned_reward / planned_risk if planned_risk > 0 else None
        results.append(
            HorizonLevels(
                horizon=label,
                sessions=sessions,
                reference_price=round(reference, 2),
                pullback_entry_low=round(entry_low, 2),
                pullback_entry_high=round(entry_high, 2),
                entry_trigger=round(entry_trigger, 2),
                breakout_trigger=round(breakout_trigger, 2),
                breakout_confirmation_rule=(
                    "以收盘价确认突破，且当日成交量不低于20日中位数的1.2倍；"
                    "盘中瞬间越线不算有效突破"
                ),
                reduce_low=round(resistance, 2),
                reduce_high=round(resistance + 0.50 * atr_value, 2),
                second_reduce_low=round(max(resistance, measured_target - 0.25 * atr_value), 2),
                second_reduce_high=round(measured_target + 0.25 * atr_value, 2),
                first_reduce_fraction=0.50,
                invalidation=round(invalidation, 2),
                stop_execution_rule=(
                    "收盘确认跌破结构失效位后，在下一可成交价格执行退出；"
                    "受T+1、停牌或跌停影响时不保证按失效价成交"
                ),
                measured_move_target=round(measured_target, 2),
                reward_risk_ratio=(round(reward_risk, 2) if reward_risk is not None else None),
                time_stop_sessions=sessions,
                level_method=(
                    "艾德华兹–麦吉趋势/支撑阻力/收盘确认/量价确认的规则化转译；"
                    "目标位采用区间高度测量，ATR仅作波动缓冲"
                ),
                level_evidence_dates=evidence_dates,
            )
        )
    return results
