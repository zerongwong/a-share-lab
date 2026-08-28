"""Transparent price-cycle classification and portfolio risk posture.

This module deliberately does not rank securities, fetch data, or decide that
an entry must exist.  It translates the existing whole-market breadth and
core-index assessments into a price-cycle label and a portfolio-level risk
policy.  A risk-off market therefore tightens exposure and confirmation rules
without stopping the search for three-to-five research candidates.

The thresholds are an explicit research specification.  They have not yet
passed point-in-time walk-forward validation and must not be described as an
optimal policy or as a forecast of a market turning point.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

import pandas as pd

from ashare_lab.analytics.index_regime import IndexRegimeAssessment, IndexRegimeState
from ashare_lab.analytics.market_regime import MarketRegimeAssessment, MarketRegimeState

METHOD_VERSION = "price-cycle-policy-v0.1.0"
VALIDATION_STATUS = "pending_point_in_time_walk_forward_validation"


class PriceCycleState(StrEnum):
    """Observable price-cycle states; these are not economic-cycle forecasts."""

    UPTREND_EXPANSION = "uptrend_expansion"
    UPTREND_PULLBACK = "uptrend_pullback"
    TRANSITION_RECOVERY = "transition_recovery"
    DOWNTREND_REPAIR = "downtrend_repair"
    DOWNTREND_PRESSURE = "downtrend_pressure"
    UNAVAILABLE = "unavailable"


class EntryStrictness(StrEnum):
    """How much confirmation an entry needs under the observed price cycle."""

    STANDARD = "standard"
    TIGHT = "tight"
    DEFENSIVE = "defensive"
    EXCEPTION_ONLY = "exception_only"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class CycleRiskPolicy:
    """Portfolio ceilings, never target allocations or promises to invest."""

    max_stock_exposure: float
    max_annual_downside_volatility: float
    max_rolling_drawdown_60_p90: float
    max_es95_5d: float
    max_down_period_correlation: float
    max_position_downside_risk_contribution: float
    industry_weight_limit: float
    entry_strictness: EntryStrictness
    entry_requirements: tuple[str, ...]
    borrowed_weight: float = 0.0

    @property
    def minimum_cash_weight(self) -> float:
        """Minimum total-account cash implied by the exposure ceiling."""

        return 1.0 - self.max_stock_exposure


@dataclass(frozen=True, slots=True)
class PriceCycleAssessment:
    """One deterministic price-cycle assessment and its risk posture."""

    state: PriceCycleState
    label: str
    confidence: float | None
    cutoff: pd.Timestamp | None
    evidence: tuple[str, ...]
    research_continues: bool
    entry_list_may_be_empty: bool
    policy: CycleRiskPolicy
    method_version: str = METHOD_VERSION
    validation_status: str = VALIDATION_STATUS
    method: str = (
        "核心指数MA60/MA120、60日收益和120日回撤判定中期方向；"
        "全市场与核心指数MA20宽度及20日收益四项投票判定短期修复；"
        "周期只调节风险姿态，不参与个股打分"
    )


_STANDARD_ENTRY = (
    "保留现有放量突破、健康回踩或重新站回突破线规则",
    "突破成交额不低于过去20日中位数1.10倍",
    "距MA20不超过12%且不超过3ATR",
)
_TIGHT_ENTRY = (
    "优先健康回踩或重新站回突破线，不追单日突然加速",
    "突破成交额不低于过去20日中位数1.20倍",
    "距MA20不超过8%且不超过2ATR",
)
_DEFENSIVE_ENTRY = (
    "60日绝对收益为正且60日相对强度位于当日数据合格全市场前10%",
    "收盘价高于MA20且MA20高于MA60，下行捕获率不高于0.80",
    "成交额不低于过去20日中位数1.30倍",
    "距MA20不超过6%且不超过1.5ATR",
)
_EXCEPTION_ENTRY = (
    *_DEFENSIVE_ENTRY,
    "仅接受健康回踩或重新站回突破线，允许可介入清单为空",
)


_POLICIES: dict[PriceCycleState, CycleRiskPolicy] = {
    PriceCycleState.UPTREND_EXPANSION: CycleRiskPolicy(
        max_stock_exposure=0.80,
        max_annual_downside_volatility=0.18,
        max_rolling_drawdown_60_p90=0.12,
        max_es95_5d=0.08,
        max_down_period_correlation=0.75,
        max_position_downside_risk_contribution=0.45,
        # At most half of the stock sleeve may sit in one industry.
        industry_weight_limit=0.40,
        entry_strictness=EntryStrictness.STANDARD,
        entry_requirements=_STANDARD_ENTRY,
    ),
    PriceCycleState.UPTREND_PULLBACK: CycleRiskPolicy(
        max_stock_exposure=0.60,
        max_annual_downside_volatility=0.15,
        max_rolling_drawdown_60_p90=0.10,
        max_es95_5d=0.06,
        max_down_period_correlation=0.65,
        max_position_downside_risk_contribution=0.40,
        industry_weight_limit=0.30,
        entry_strictness=EntryStrictness.TIGHT,
        entry_requirements=_TIGHT_ENTRY,
    ),
    PriceCycleState.TRANSITION_RECOVERY: CycleRiskPolicy(
        max_stock_exposure=0.50,
        max_annual_downside_volatility=0.14,
        max_rolling_drawdown_60_p90=0.09,
        max_es95_5d=0.06,
        max_down_period_correlation=0.65,
        max_position_downside_risk_contribution=0.40,
        industry_weight_limit=0.25,
        entry_strictness=EntryStrictness.TIGHT,
        entry_requirements=_TIGHT_ENTRY,
    ),
    PriceCycleState.DOWNTREND_REPAIR: CycleRiskPolicy(
        max_stock_exposure=0.30,
        max_annual_downside_volatility=0.12,
        max_rolling_drawdown_60_p90=0.08,
        max_es95_5d=0.05,
        max_down_period_correlation=0.60,
        max_position_downside_risk_contribution=0.35,
        industry_weight_limit=0.15,
        entry_strictness=EntryStrictness.DEFENSIVE,
        entry_requirements=_DEFENSIVE_ENTRY,
    ),
    PriceCycleState.DOWNTREND_PRESSURE: CycleRiskPolicy(
        max_stock_exposure=0.20,
        max_annual_downside_volatility=0.10,
        max_rolling_drawdown_60_p90=0.07,
        max_es95_5d=0.04,
        max_down_period_correlation=0.55,
        max_position_downside_risk_contribution=0.35,
        industry_weight_limit=0.10,
        entry_strictness=EntryStrictness.EXCEPTION_ONLY,
        entry_requirements=_EXCEPTION_ENTRY,
    ),
    PriceCycleState.UNAVAILABLE: CycleRiskPolicy(
        max_stock_exposure=0.0,
        max_annual_downside_volatility=0.0,
        max_rolling_drawdown_60_p90=0.0,
        max_es95_5d=0.0,
        max_down_period_correlation=0.0,
        max_position_downside_risk_contribution=0.0,
        industry_weight_limit=0.0,
        entry_strictness=EntryStrictness.UNAVAILABLE,
        entry_requirements=("数据完整并通过共同截止日核验后再研究",),
    ),
}

_LABELS = {
    PriceCycleState.UPTREND_EXPANSION: "中期上行｜短线增强",
    PriceCycleState.UPTREND_PULLBACK: "中期上行｜短线回撤或分化",
    PriceCycleState.TRANSITION_RECOVERY: "中期过渡｜复苏尝试或证据混合",
    PriceCycleState.DOWNTREND_REPAIR: "中期下行｜短线修复反弹",
    PriceCycleState.DOWNTREND_PRESSURE: "中期下行｜短线压力",
    PriceCycleState.UNAVAILABLE: "价格周期数据不可用",
}


def assess_price_cycle(
    market: MarketRegimeAssessment | None,
    indices: IndexRegimeAssessment | None,
) -> PriceCycleAssessment:
    """Classify one price cycle without blocking research in a risk-off market.

    ``confidence`` is the agreement ratio of the transparent rules supporting
    the selected label.  It is not a probability of future market direction.
    """

    unavailable_reason = _unavailable_reason(market, indices)
    if unavailable_reason is not None:
        return _unavailable(unavailable_reason)

    assert market is not None and indices is not None
    values = _required_values(market, indices)
    if values is None:
        return _unavailable("周期判定所需指标缺失、越界或非有限值")
    if _normalise_cutoff(market.cutoff) != _normalise_cutoff(indices.cutoff):
        return _unavailable("全市场与核心指数共同截止日不一致")

    down_votes = (
        values["index_breadth60"] <= 1.0 / 3.0,
        values["index_breadth120"] <= 1.0 / 3.0,
        values["index_return60"] < 0.0,
        values["median_drawdown120"] <= -0.18,
        values["worst_drawdown120"] <= -0.25,
    )
    up_votes = (
        values["index_breadth60"] >= 2.0 / 3.0,
        values["index_breadth120"] >= 2.0 / 3.0,
        values["index_return60"] > 0.0,
        values["median_drawdown120"] > -0.15,
        values["worst_drawdown120"] > -0.20,
    )
    downtrend = all(down_votes[:3]) or down_votes[3] or down_votes[4]
    uptrend = all(up_votes)

    short_votes = (
        values["market_breadth20"] >= 0.50,
        values["market_return20"] > 0.0,
        values["index_return20"] > 0.0,
        values["index_breadth20"] >= 0.50,
    )
    positive_short_votes = sum(short_votes)
    short_strengthening = positive_short_votes >= 3

    if uptrend:
        state = (
            PriceCycleState.UPTREND_EXPANSION
            if short_strengthening
            else PriceCycleState.UPTREND_PULLBACK
        )
        direction_support = sum(up_votes) / len(up_votes)
    elif downtrend:
        state = (
            PriceCycleState.DOWNTREND_REPAIR
            if short_strengthening
            else PriceCycleState.DOWNTREND_PRESSURE
        )
        direction_support = max(0.50, sum(down_votes) / len(down_votes))
    else:
        state = PriceCycleState.TRANSITION_RECOVERY
        direction_support = 0.50

    short_support = (
        positive_short_votes / len(short_votes)
        if state in {PriceCycleState.UPTREND_EXPANSION, PriceCycleState.DOWNTREND_REPAIR}
        else (len(short_votes) - positive_short_votes) / len(short_votes)
    )
    if state is PriceCycleState.TRANSITION_RECOVERY:
        short_support = max(positive_short_votes, len(short_votes) - positive_short_votes) / len(
            short_votes
        )
    confidence = round((direction_support + short_support) / 2.0, 3)

    evidence = (
        f"全市场MA20宽度{values['market_breadth20']:.1%}，"
        f"20日收益中位数{values['market_return20']:.1%}",
        f"核心指数MA20/MA60/MA120宽度分别为{values['index_breadth20']:.1%}/"
        f"{values['index_breadth60']:.1%}/{values['index_breadth120']:.1%}",
        f"核心指数20日/60日收益中位数分别为{values['index_return20']:.1%}/"
        f"{values['index_return60']:.1%}",
        f"核心指数60日年化波动中位数{values['index_volatility60']:.1%}，"
        f"120日回撤中位数{values['median_drawdown120']:.1%}，"
        f"最差{values['worst_drawdown120']:.1%}",
        f"短期增强投票{positive_short_votes}/4；confidence是规则一致度，不是预测概率",
        "研究继续生成3至5只相对领先候选；可介入清单允许为空",
    )
    return PriceCycleAssessment(
        state=state,
        label=_LABELS[state],
        confidence=confidence,
        cutoff=_normalise_cutoff(market.cutoff),
        evidence=evidence,
        research_continues=True,
        entry_list_may_be_empty=True,
        policy=_POLICIES[state],
    )


def _unavailable_reason(
    market: MarketRegimeAssessment | None,
    indices: IndexRegimeAssessment | None,
) -> str | None:
    if market is None:
        return "缺少全市场宽度评估"
    if indices is None:
        return "缺少核心指数评估"
    if market.state is MarketRegimeState.UNAVAILABLE:
        return f"全市场宽度不可用：{market.reason or '未提供原因'}"
    if indices.state is IndexRegimeState.UNAVAILABLE:
        return f"核心指数不可用：{indices.reason or '未提供原因'}"
    return None


def _required_values(
    market: MarketRegimeAssessment,
    indices: IndexRegimeAssessment,
) -> dict[str, float] | None:
    raw = {
        "market_breadth20": market.breadth_above_ma20,
        "market_return20": market.median_return_20,
        "index_breadth20": indices.breadth_above_ma20,
        "index_breadth60": indices.breadth_above_ma60,
        "index_breadth120": indices.breadth_above_ma120,
        "index_return20": indices.median_return_20,
        "index_return60": indices.median_return_60,
        "index_volatility60": indices.median_annualized_volatility_60,
        "median_drawdown120": indices.median_max_drawdown_120,
        "worst_drawdown120": indices.worst_max_drawdown_120,
    }
    if any(value is None or isinstance(value, bool) for value in raw.values()):
        return None
    values = {key: float(value) for key, value in raw.items() if value is not None}
    if not all(isfinite(value) for value in values.values()):
        return None
    for key in (
        "market_breadth20",
        "index_breadth20",
        "index_breadth60",
        "index_breadth120",
    ):
        if not 0.0 <= values[key] <= 1.0:
            return None
    if values["index_volatility60"] < 0.0:
        return None
    if values["median_drawdown120"] > 0.0 or values["worst_drawdown120"] > 0.0:
        return None
    if values["worst_drawdown120"] > values["median_drawdown120"]:
        return None
    return values


def _normalise_cutoff(value: object) -> pd.Timestamp | None:
    try:
        cutoff = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(cutoff):
        return None
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    return cutoff.normalize()


def _unavailable(reason: str) -> PriceCycleAssessment:
    return PriceCycleAssessment(
        state=PriceCycleState.UNAVAILABLE,
        label=_LABELS[PriceCycleState.UNAVAILABLE],
        confidence=None,
        cutoff=None,
        evidence=(reason,),
        research_continues=False,
        entry_list_may_be_empty=True,
        policy=_POLICIES[PriceCycleState.UNAVAILABLE],
    )
