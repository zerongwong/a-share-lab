from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ashare_lab.domain.enums import TrendState


@dataclass(frozen=True)
class LivermorePlan:
    pivotal_buy_above: float
    add_only_above: float
    invalidation_below: float
    may_open_probe: bool
    may_add_to_position: bool
    note: str


def build_livermore_plan(
    frame: pd.DataFrame,
    trend: TrendState,
    *,
    current_position_profitable: bool = False,
) -> LivermorePlan:
    latest = frame.iloc[-1]
    atr_value = float(latest["atr14"])
    resistance = float(latest["prior_high_60"])
    support = float(latest["prior_low_20"])
    pivot = resistance + 0.25 * atr_value
    add_above = pivot + 0.50 * atr_value
    invalidation = support - 0.75 * atr_value
    trend_confirmed = trend == TrendState.UP
    may_probe = trend_confirmed and float(latest["close"]) >= pivot
    may_add = may_probe and current_position_profitable and float(latest["close"]) >= add_above
    return LivermorePlan(
        pivotal_buy_above=round(pivot, 2),
        add_only_above=round(add_above, 2),
        invalidation_below=round(invalidation, 2),
        may_open_probe=may_probe,
        may_add_to_position=may_add,
        note="只在趋势与关键点共同确认后试仓；只向盈利仓加码，绝不因下跌摊低成本。",
    )
