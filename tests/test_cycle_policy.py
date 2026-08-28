from __future__ import annotations

import pandas as pd
import pytest

from ashare_lab.analytics.cycle_policy import (
    METHOD_VERSION,
    VALIDATION_STATUS,
    EntryStrictness,
    PriceCycleState,
    assess_price_cycle,
)
from ashare_lab.analytics.index_regime import IndexRegimeAssessment, IndexRegimeState
from ashare_lab.analytics.market_regime import MarketRegimeAssessment, MarketRegimeState

CUTOFF = pd.Timestamp("2026-08-26")


def _market(
    *,
    state: MarketRegimeState = MarketRegimeState.NEUTRAL,
    breadth20: float | None = 0.55,
    return20: float | None = 0.04,
    cutoff: pd.Timestamp = CUTOFF,
) -> MarketRegimeAssessment:
    return MarketRegimeAssessment(
        state=state,
        score=0.45,
        cutoff=cutoff,
        eligible_symbols=4_843,
        breadth_above_ma20=breadth20,
        breadth_above_ma60=0.48,
        breadth_above_ma120=0.20,
        median_return_20=return20,
        median_return_60=-0.08,
        reason="fixture",
    )


def _indices(
    *,
    state: IndexRegimeState = IndexRegimeState.RISK_OFF,
    breadth20: float | None = 1.0 / 6.0,
    breadth60: float | None = 0.0,
    breadth120: float | None = 0.0,
    return20: float | None = 0.02,
    return60: float | None = -0.08,
    volatility60: float | None = 0.35,
    median_drawdown: float | None = -0.19,
    worst_drawdown: float | None = -0.26,
    cutoff: pd.Timestamp = CUTOFF,
) -> IndexRegimeAssessment:
    return IndexRegimeAssessment(
        state=state,
        score=0.16,
        cutoff=cutoff,
        eligible_indices=6,
        required_indices=6,
        breadth_above_ma20=breadth20,
        breadth_above_ma60=breadth60,
        breadth_above_ma120=breadth120,
        median_return_20=return20,
        median_return_60=return60,
        median_annualized_volatility_60=volatility60,
        median_max_drawdown_120=median_drawdown,
        worst_max_drawdown_120=worst_drawdown,
        reason="fixture",
    )


def test_current_2026_08_26_is_downtrend_repair_without_stopping_research() -> None:
    market = _market(breadth20=0.5447036960561635, return20=0.05639097744360888)
    indices = _indices(
        breadth20=1.0 / 6.0,
        breadth60=0.0,
        breadth120=0.0,
        return20=0.017672082690934232,
        return60=-0.08319893525811534,
        volatility60=0.35195908564725176,
        median_drawdown=-0.18958814527986712,
        worst_drawdown=-0.2578622681804552,
    )

    result = assess_price_cycle(market, indices)

    assert result.state is PriceCycleState.DOWNTREND_REPAIR
    assert result.label == "中期下行｜短线修复反弹"
    assert result.confidence == pytest.approx(0.875)
    assert result.research_continues is True
    assert result.entry_list_may_be_empty is True
    assert result.policy.max_stock_exposure == pytest.approx(0.30)
    assert result.policy.minimum_cash_weight == pytest.approx(0.70)
    assert result.policy.entry_strictness is EntryStrictness.DEFENSIVE
    assert result.policy.max_annual_downside_volatility == pytest.approx(0.12)
    assert result.policy.industry_weight_limit == pytest.approx(0.15)
    assert any("3/4" in item for item in result.evidence)


def test_pure_weak_sample_is_downtrend_pressure_but_research_still_continues() -> None:
    result = assess_price_cycle(
        _market(state=MarketRegimeState.RISK_OFF, breadth20=0.20, return20=-0.10),
        _indices(breadth20=0.0, return20=-0.08),
    )

    assert result.state is PriceCycleState.DOWNTREND_PRESSURE
    assert result.research_continues is True
    assert result.policy.max_stock_exposure == pytest.approx(0.20)
    assert result.policy.industry_weight_limit == pytest.approx(0.10)
    assert result.policy.entry_strictness is EntryStrictness.EXCEPTION_ONLY
    assert "允许可介入清单为空" in result.policy.entry_requirements[-1]


def test_confirmed_uptrend_with_short_strength_is_expansion() -> None:
    result = assess_price_cycle(
        _market(state=MarketRegimeState.RISK_ON, breadth20=0.75, return20=0.07),
        _indices(
            state=IndexRegimeState.RISK_ON,
            breadth20=0.75,
            breadth60=0.80,
            breadth120=0.75,
            return20=0.05,
            return60=0.12,
            volatility60=0.18,
            median_drawdown=-0.08,
            worst_drawdown=-0.12,
        ),
    )

    assert result.state is PriceCycleState.UPTREND_EXPANSION
    assert result.policy.max_stock_exposure == pytest.approx(0.80)
    assert result.policy.entry_strictness is EntryStrictness.STANDARD


def test_confirmed_uptrend_with_short_weakness_is_pullback() -> None:
    result = assess_price_cycle(
        _market(state=MarketRegimeState.NEUTRAL, breadth20=0.35, return20=-0.02),
        _indices(
            state=IndexRegimeState.RISK_ON,
            breadth20=0.25,
            breadth60=0.80,
            breadth120=0.75,
            return20=-0.01,
            return60=0.10,
            volatility60=0.18,
            median_drawdown=-0.08,
            worst_drawdown=-0.12,
        ),
    )

    assert result.state is PriceCycleState.UPTREND_PULLBACK
    assert result.policy.max_stock_exposure == pytest.approx(0.60)
    assert result.policy.entry_strictness is EntryStrictness.TIGHT


def test_mixed_primary_evidence_is_transition_recovery() -> None:
    result = assess_price_cycle(
        _market(),
        _indices(
            state=IndexRegimeState.NEUTRAL,
            breadth20=0.50,
            breadth60=0.50,
            breadth120=0.50,
            return20=0.01,
            return60=0.01,
            volatility60=0.22,
            median_drawdown=-0.12,
            worst_drawdown=-0.17,
        ),
    )

    assert result.state is PriceCycleState.TRANSITION_RECOVERY
    assert result.policy.max_stock_exposure == pytest.approx(0.50)


@pytest.mark.parametrize("unavailable_layer", ["market", "indices"])
def test_unavailable_data_is_the_only_fail_closed_cycle(unavailable_layer: str) -> None:
    market = _market()
    indices = _indices()
    if unavailable_layer == "market":
        market = _market(state=MarketRegimeState.UNAVAILABLE)
    else:
        indices = _indices(state=IndexRegimeState.UNAVAILABLE)

    result = assess_price_cycle(market, indices)

    assert result.state is PriceCycleState.UNAVAILABLE
    assert result.research_continues is False
    assert result.policy.max_stock_exposure == 0.0
    assert result.policy.entry_strictness is EntryStrictness.UNAVAILABLE


def test_mismatched_cutoffs_or_missing_metrics_fail_closed() -> None:
    mismatched = assess_price_cycle(
        _market(),
        _indices(cutoff=CUTOFF - pd.Timedelta(days=1)),
    )
    missing = assess_price_cycle(_market(), _indices(return60=None))

    assert mismatched.state is PriceCycleState.UNAVAILABLE
    assert "共同截止日不一致" in mismatched.evidence[0]
    assert missing.state is PriceCycleState.UNAVAILABLE


def test_policy_discloses_version_and_validation_status() -> None:
    result = assess_price_cycle(_market(), _indices())

    assert result.method_version == METHOD_VERSION
    assert result.validation_status == VALIDATION_STATUS
    assert "walk_forward" in result.validation_status
