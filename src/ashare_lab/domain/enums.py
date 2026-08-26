from __future__ import annotations

from enum import StrEnum


class TrendState(StrEnum):
    UP = "上升趋势"
    DOWN = "下降趋势"
    RANGE = "区间震荡"
    INSUFFICIENT = "数据不足"


class DataQuality(StrEnum):
    LIVE = "实时/最新"
    CACHED = "缓存"
    SAMPLE = "离线样例"
    UNAVAILABLE = "不可用"


class RiskProfile(StrEnum):
    CONSERVATIVE = "稳健型"
    BALANCED = "平衡型"
    AGGRESSIVE = "激进型"
