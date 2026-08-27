from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from ashare_lab.analytics.adaptive_portfolio import (
    AdaptiveCandidate,
    AdaptivePortfolioDataError,
    AdaptiveRiskBudget,
    evaluate_adaptive_portfolio,
    optimize_adaptive_portfolio,
)


def _candidate_set(
    count: int,
    *,
    periods: int = 260,
    industries: tuple[str, ...] | None = None,
) -> tuple[AdaptiveCandidate, ...]:
    dates = pd.bdate_range("2025-01-02", periods=periods)
    common = np.random.default_rng(910).normal(0.0, 0.004, periods)
    labels = industries or tuple(f"行业{index}" for index in range(count))
    candidates = []
    for index in range(count):
        idiosyncratic = np.random.default_rng(100 + index).normal(
            0.0, 0.003 + index * 0.0005, periods
        )
        returns = 0.00035 + (0.55 + 0.05 * index) * common + idiosyncratic
        candidates.append(
            AdaptiveCandidate(
                symbol=f"{index + 1:06d}.SZ",
                industry=labels[index],
                signal_score=index / max(1, count - 1),
                returns=pd.Series(returns, index=dates),
            )
        )
    return tuple(candidates)


def _lax_budget(**overrides: object) -> AdaptiveRiskBudget:
    values: dict[str, object] = {
        "max_annual_downside_volatility": 5.0,
        "max_rolling_drawdown_60_p90": 1.0,
        "max_es95_5d": 1.0,
        "max_down_period_correlation": 1.0,
        "max_position_downside_risk_contribution": 1.0,
    }
    values.update(overrides)
    return AdaptiveRiskBudget(**values)


@pytest.mark.parametrize(
    ("count", "exposure", "cash", "lower", "upper"),
    (
        (3, 0.70, 0.30, 0.15, 0.35),
        (4, 0.80, 0.20, 0.10, 0.30),
        (5, 0.85, 0.15, 0.08, 0.25),
    ),
)
def test_optimizer_uses_count_specific_exposure_and_position_limits(
    count: int,
    exposure: float,
    cash: float,
    lower: float,
    upper: float,
) -> None:
    result = optimize_adaptive_portfolio(_candidate_set(count), budget=_lax_budget())

    weights = [position.weight for position in result.positions]
    assert len(weights) == count
    assert sum(weights) == pytest.approx(exposure)
    assert min(weights) >= lower - 1e-12
    assert max(weights) <= upper + 1e-12
    assert result.stock_exposure == exposure
    assert result.cash_weight == pytest.approx(cash)
    assert result.borrowed_weight == 0.0
    assert max(dict(result.industry_weights).values()) <= 0.40 + 1e-12


def test_equal_risk_candidates_receive_a_bounded_signal_tilt() -> None:
    dates = pd.bdate_range("2025-01-02", periods=260)
    shared_returns = pd.Series(
        np.random.default_rng(7).normal(0.0003, 0.006, len(dates)), index=dates
    )
    candidates = tuple(
        AdaptiveCandidate(
            symbol=f"00000{index + 1}.SZ",
            industry=f"行业{index}",
            signal_score=score,
            returns=shared_returns.copy(),
        )
        for index, score in enumerate((0.0, 0.33, 0.67, 1.0))
    )

    result = optimize_adaptive_portfolio(candidates, budget=_lax_budget())
    weights = [position.weight for position in result.positions]

    assert weights == sorted(weights)
    assert weights[-1] > weights[0]
    assert weights[0] >= 0.10
    assert weights[-1] <= 0.30


def test_industry_projection_caps_a_high_priority_group_at_forty_percent() -> None:
    candidates = list(
        _candidate_set(4, industries=("科技", "科技", "金融", "消费"))
    )
    for index in (0, 1):
        original = candidates[index]
        candidates[index] = AdaptiveCandidate(
            symbol=original.symbol,
            industry=original.industry,
            signal_score=1.0,
            returns=original.returns * 0.25,
        )

    result = optimize_adaptive_portfolio(candidates, budget=_lax_budget())

    industry_weights = dict(result.industry_weights)
    assert industry_weights["科技"] == pytest.approx(0.40)
    assert max(industry_weights.values()) <= 0.40 + 1e-12
    assert result.risk_budget.industry_concentration_passed


def test_infeasible_industry_mix_fails_closed() -> None:
    candidates = _candidate_set(
        5,
        industries=("科技", "科技", "科技", "金融", "金融"),
    )

    with pytest.raises(AdaptivePortfolioDataError, match="allocation bounds"):
        optimize_adaptive_portfolio(candidates, budget=_lax_budget())


def test_metrics_match_the_documented_formulas() -> None:
    candidates = _candidate_set(4, periods=260)
    weights_by_symbol = {candidate.symbol: 0.20 for candidate in candidates}
    budget = _lax_budget(holding_period_cost_rate=0.001, lcb_confidence=0.90)

    result = evaluate_adaptive_portfolio(candidates, weights_by_symbol, budget=budget)
    ordered = tuple(sorted(candidates, key=lambda item: item.symbol))
    returns = np.column_stack([item.returns.to_numpy() for item in ordered])
    weights = np.full(4, 0.20)
    daily_portfolio = returns @ weights

    expected_downside_volatility = (
        np.sqrt(np.mean(np.square(np.minimum(daily_portfolio, 0.0)))) * np.sqrt(252)
    )
    drawdowns = []
    for start in range(len(daily_portfolio) - 60 + 1):
        equity = np.concatenate(
            ([1.0], np.cumprod(1.0 + daily_portfolio[start : start + 60]))
        )
        drawdowns.append(max(0.0, -float((equity / np.maximum.accumulate(equity) - 1).min())))

    five_day_blocks = returns.reshape(52, 5, 4)
    five_day_returns = (np.prod(1.0 + five_day_blocks, axis=1) - 1.0) @ weights
    five_day_cutoff = np.quantile(five_day_returns, 0.05)
    expected_es = max(0.0, -float(five_day_returns[five_day_returns <= five_day_cutoff].mean()))

    holding_blocks = returns.reshape(13, 20, 4)
    holding_returns = (np.prod(1.0 + holding_blocks, axis=1) - 1.0) @ weights - 0.001
    expected_lcb = float(
        holding_returns.mean()
        - NormalDist().inv_cdf(0.90)
        * holding_returns.std(ddof=1)
        / math.sqrt(len(holding_returns))
    )

    downside = np.minimum(returns, 0.0)
    second_moment = downside.T @ downside / len(downside)
    marginal = second_moment @ weights
    contributions = weights * marginal / float(weights @ marginal)

    metrics = result.metrics
    assert metrics.annual_downside_volatility == pytest.approx(expected_downside_volatility)
    assert metrics.rolling_max_drawdown_60_p90 == pytest.approx(np.quantile(drawdowns, 0.90))
    assert metrics.es95_5d == pytest.approx(expected_es)
    assert metrics.max_position_downside_risk_contribution == pytest.approx(
        contributions.max()
    )
    assert metrics.holding_period_return_mean == pytest.approx(holding_returns.mean())
    assert metrics.holding_period_return_lcb == pytest.approx(expected_lcb)
    assert sum(position.downside_risk_contribution for position in result.positions) == pytest.approx(
        1.0
    )


def test_cost_deduction_lowers_mean_and_lcb_one_for_one() -> None:
    candidates = _candidate_set(4)
    weights = {candidate.symbol: 0.20 for candidate in candidates}
    without_cost = evaluate_adaptive_portfolio(
        candidates, weights, budget=_lax_budget(holding_period_cost_rate=0.0)
    )
    with_cost = evaluate_adaptive_portfolio(
        candidates, weights, budget=_lax_budget(holding_period_cost_rate=0.0125)
    )

    assert with_cost.metrics.holding_period_return_mean == pytest.approx(
        without_cost.metrics.holding_period_return_mean - 0.0125
    )
    assert with_cost.metrics.holding_period_return_lcb == pytest.approx(
        without_cost.metrics.holding_period_return_lcb - 0.0125
    )


def test_risk_budget_reports_each_breach_instead_of_loosening_limits() -> None:
    candidates = _candidate_set(4)
    weights = {candidate.symbol: 0.20 for candidate in candidates}
    budget = AdaptiveRiskBudget(
        max_annual_downside_volatility=0.0001,
        max_rolling_drawdown_60_p90=0.0001,
        max_es95_5d=0.0001,
        max_down_period_correlation=-0.99,
        max_position_downside_risk_contribution=0.20,
        industry_weight_limit=0.19,
    )

    result = evaluate_adaptive_portfolio(candidates, weights, budget=budget)

    assert not result.risk_budget.passed
    assert "annual_downside_volatility" in result.risk_budget.violations
    assert "position_downside_risk_contribution" in result.risk_budget.violations
    assert "industry_concentration" in result.risk_budget.violations
    assert not result.risk_budget.industry_concentration_passed


def test_optimizer_is_deterministic_and_invariant_to_candidate_order() -> None:
    candidates = _candidate_set(4)

    forward = optimize_adaptive_portfolio(candidates, budget=_lax_budget())
    reversed_result = optimize_adaptive_portfolio(tuple(reversed(candidates)), budget=_lax_budget())

    assert forward == reversed_result


@pytest.mark.parametrize("count", (2, 6))
def test_candidate_count_outside_three_to_five_fails_closed(count: int) -> None:
    with pytest.raises(AdaptivePortfolioDataError, match="three, four, or five"):
        optimize_adaptive_portfolio(_candidate_set(count), budget=_lax_budget())


def test_short_missing_or_misaligned_returns_fail_closed() -> None:
    short = _candidate_set(4, periods=100)
    with pytest.raises(AdaptivePortfolioDataError, match="at least 160"):
        optimize_adaptive_portfolio(short, budget=_lax_budget())

    missing = list(_candidate_set(4))
    damaged = missing[0].returns.copy()
    damaged.iloc[5] = np.nan
    missing[0] = AdaptiveCandidate(
        missing[0].symbol,
        missing[0].industry,
        missing[0].signal_score,
        damaged,
    )
    with pytest.raises(AdaptivePortfolioDataError, match="missing data"):
        optimize_adaptive_portfolio(missing, budget=_lax_budget())

    misaligned = list(_candidate_set(4))
    shifted = misaligned[0].returns.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    misaligned[0] = AdaptiveCandidate(
        misaligned[0].symbol,
        misaligned[0].industry,
        misaligned[0].signal_score,
        shifted,
    )
    with pytest.raises(AdaptivePortfolioDataError, match="indices must match exactly"):
        optimize_adaptive_portfolio(misaligned, budget=_lax_budget())


def test_degenerate_downside_history_fails_closed() -> None:
    dates = pd.bdate_range("2025-01-02", periods=260)
    candidates = tuple(
        AdaptiveCandidate(
            symbol=f"00000{index + 1}.SZ",
            industry=f"行业{index}",
            signal_score=0.5,
            returns=pd.Series(np.full(len(dates), 0.001), index=dates),
        )
        for index in range(4)
    )

    with pytest.raises(AdaptivePortfolioDataError, match="non-zero historical downside"):
        optimize_adaptive_portfolio(candidates, budget=_lax_budget())


def test_explicit_weights_must_match_symbols_bounds_and_exposure() -> None:
    candidates = _candidate_set(4)

    with pytest.raises(AdaptivePortfolioDataError, match="exactly match"):
        evaluate_adaptive_portfolio(
            candidates,
            {candidate.symbol: 0.20 for candidate in candidates[:-1]},
            budget=_lax_budget(),
        )
    with pytest.raises(AdaptivePortfolioDataError, match="must be in"):
        evaluate_adaptive_portfolio(
            candidates,
            {
                candidates[0].symbol: 0.35,
                candidates[1].symbol: 0.15,
                candidates[2].symbol: 0.15,
                candidates[3].symbol: 0.15,
            },
            budget=_lax_budget(),
        )
    with pytest.raises(AdaptivePortfolioDataError, match="must sum"):
        evaluate_adaptive_portfolio(
            candidates,
            {candidate.symbol: 0.19 for candidate in candidates},
            budget=_lax_budget(),
        )
