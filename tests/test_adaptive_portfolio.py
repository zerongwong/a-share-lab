from __future__ import annotations

import math
from dataclasses import fields
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from ashare_lab.analytics.adaptive_portfolio import (
    AdaptiveCandidate,
    AdaptivePortfolioDataError,
    AdaptiveRiskBudget,
    evaluate_adaptive_portfolio,
    evaluate_operational_adaptive_portfolio,
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
    ("count", "exposure", "cash", "sleeve_lower", "sleeve_upper"),
    (
        (3, 0.70, 0.30, 0.20, 0.50),
        (4, 0.80, 0.20, 0.10, 0.40),
        (5, 0.85, 0.15, 0.10, 0.30),
    ),
)
def test_optimizer_uses_count_specific_exposure_and_operational_sleeve_limits(
    count: int,
    exposure: float,
    cash: float,
    sleeve_lower: float,
    sleeve_upper: float,
) -> None:
    result = optimize_adaptive_portfolio(_candidate_set(count), budget=_lax_budget())

    weights = [position.weight for position in result.positions]
    sleeve_weights = [weight / exposure for weight in weights]
    assert len(weights) == count
    assert sum(weights) == pytest.approx(exposure)
    assert min(sleeve_weights) >= sleeve_lower - 1e-12
    assert max(sleeve_weights) <= sleeve_upper + 1e-12
    assert all(math.isclose(weight * 10, round(weight * 10)) for weight in sleeve_weights)
    assert sum(sleeve_weights) == pytest.approx(1.0)
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


def test_cycle_exposure_cap_scales_positions_and_keeps_remainder_in_cash() -> None:
    result = optimize_adaptive_portfolio(
        _candidate_set(4),
        budget=_lax_budget(maximum_stock_exposure=0.30),
    )

    assert result.stock_exposure == pytest.approx(0.30)
    assert result.cash_weight == pytest.approx(0.70)
    assert sum(item.weight for item in result.positions) == pytest.approx(0.30)
    sleeve_weights = [item.weight / result.stock_exposure for item in result.positions]
    assert all(0.10 <= weight <= 0.40 for weight in sleeve_weights)
    assert all(math.isclose(weight * 10, round(weight * 10)) for weight in sleeve_weights)


@pytest.mark.parametrize("count", (3, 5))
def test_cycle_exposure_cap_scales_three_and_five_stock_operational_sleeves(
    count: int,
) -> None:
    result = optimize_adaptive_portfolio(
        _candidate_set(count),
        budget=_lax_budget(maximum_stock_exposure=0.30),
    )

    assert result.stock_exposure == pytest.approx(0.30)
    assert result.cash_weight == pytest.approx(0.70)
    sleeve_weights = [item.weight / result.stock_exposure for item in result.positions]
    lower, upper = {3: (0.20, 0.50), 5: (0.10, 0.30)}[count]
    assert sum(sleeve_weights) == pytest.approx(1.0)
    assert all(lower <= weight <= upper for weight in sleeve_weights)
    assert all(math.isclose(weight * 10, round(weight * 10)) for weight in sleeve_weights)


@pytest.mark.parametrize("value", (0.0, -0.1, 0.86, float("inf"), True))
def test_invalid_cycle_exposure_cap_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="maximum_stock_exposure"):
        _lax_budget(maximum_stock_exposure=value)


def test_industry_projection_caps_a_high_priority_group_at_forty_percent() -> None:
    candidates = list(_candidate_set(4, industries=("科技", "科技", "金融", "消费")))
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


def test_no_industry_feasible_ten_percent_grid_fails_without_fallback() -> None:
    candidates = _candidate_set(4)

    with pytest.raises(AdaptivePortfolioDataError, match="10% stock-sleeve grid"):
        optimize_adaptive_portfolio(
            candidates,
            budget=_lax_budget(industry_weight_limit=0.21),
        )


def test_evaluation_keeps_continuous_target_and_audits_grid_method() -> None:
    candidates = _candidate_set(4)
    target = {
        candidates[0].symbol: 0.22,
        candidates[1].symbol: 0.21,
        candidates[2].symbol: 0.19,
        candidates[3].symbol: 0.18,
    }

    result = evaluate_adaptive_portfolio(candidates, target, budget=_lax_budget())

    assert dict(result.exact_target_weights) == pytest.approx(target)
    assert result.stock_sleeve_weight_step == pytest.approx(0.10)
    assert result.weight_quantization_method_version == ("exhaustive-stock-sleeve-grid-v1.0.0")
    operation = {position.symbol: position.weight for position in result.positions}
    assert operation != pytest.approx(target)
    assert sum(operation.values()) == pytest.approx(result.stock_exposure)
    assert all(
        math.isclose(
            weight / result.stock_exposure * 10, round(weight / result.stock_exposure * 10)
        )
        for weight in operation.values()
    )


def test_direct_operational_evaluation_never_calls_quantizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _candidate_set(4)
    operation = {
        candidates[0].symbol: 0.24,
        candidates[1].symbol: 0.24,
        candidates[2].symbol: 0.16,
        candidates[3].symbol: 0.16,
    }

    def fail_if_called(*args: object, **kwargs: object) -> tuple[float, ...]:
        raise AssertionError("operational evaluation must not quantize")

    monkeypatch.setattr(
        "ashare_lab.analytics.adaptive_portfolio.quantize_stock_sleeve_weights",
        fail_if_called,
    )
    result = evaluate_operational_adaptive_portfolio(
        candidates,
        operation,
        budget=_lax_budget(),
    )

    assert {position.symbol: position.weight for position in result.positions} == pytest.approx(
        operation
    )
    assert dict(result.exact_target_weights) == pytest.approx(operation)
    assert "without quantization" in result.method
    assert result.weight_quantization_method_version == (
        "direct-operational-stock-sleeve-validation-v1.0.0"
    )


def test_direct_and_continuous_evaluators_recompute_identical_metrics_for_same_grid() -> None:
    candidates = _candidate_set(4)
    operation = {
        candidates[0].symbol: 0.24,
        candidates[1].symbol: 0.24,
        candidates[2].symbol: 0.16,
        candidates[3].symbol: 0.16,
    }
    budget = _lax_budget(
        holding_period_sessions=120,
        holding_period_cost_rate=0.001,
        minimum_holding_period_samples=2,
    )

    continuous = evaluate_adaptive_portfolio(candidates, operation, budget=budget)
    direct = evaluate_operational_adaptive_portfolio(candidates, operation, budget=budget)

    for direct_position, continuous_position in zip(
        direct.positions, continuous.positions, strict=True
    ):
        assert direct_position.symbol == continuous_position.symbol
        assert direct_position.industry == continuous_position.industry
        assert direct_position.signal_score == continuous_position.signal_score
        assert direct_position.weight == pytest.approx(continuous_position.weight)
        assert direct_position.annual_downside_volatility == pytest.approx(
            continuous_position.annual_downside_volatility
        )
        assert direct_position.downside_risk_contribution == pytest.approx(
            continuous_position.downside_risk_contribution
        )
    assert dict(direct.industry_weights) == pytest.approx(dict(continuous.industry_weights))
    for field in fields(direct.metrics):
        direct_value = getattr(direct.metrics, field.name)
        continuous_value = getattr(continuous.metrics, field.name)
        if isinstance(direct_value, float):
            assert direct_value == pytest.approx(continuous_value)
        else:
            assert direct_value == continuous_value
    assert direct.risk_budget == continuous.risk_budget
    assert direct.method != continuous.method


@pytest.mark.parametrize(
    ("count", "accepted_sleeve", "rejected_sleeve"),
    (
        (3, (0.50, 0.30, 0.20), (0.60, 0.20, 0.20)),
        (4, (0.40, 0.20, 0.20, 0.20), (0.50, 0.20, 0.20, 0.10)),
        (5, (0.30, 0.20, 0.20, 0.20, 0.10), (0.40, 0.20, 0.20, 0.10, 0.10)),
    ),
)
def test_direct_operational_grid_enforces_count_specific_boundaries(
    count: int,
    accepted_sleeve: tuple[float, ...],
    rejected_sleeve: tuple[float, ...],
) -> None:
    candidates = _candidate_set(count)
    exposure = {3: 0.70, 4: 0.80, 5: 0.85}[count]
    accepted = {
        candidate.symbol: exposure * sleeve
        for candidate, sleeve in zip(candidates, accepted_sleeve, strict=True)
    }
    rejected = {
        candidate.symbol: exposure * sleeve
        for candidate, sleeve in zip(candidates, rejected_sleeve, strict=True)
    }

    result = evaluate_operational_adaptive_portfolio(
        candidates,
        accepted,
        budget=_lax_budget(),
    )
    assert [position.weight / exposure for position in result.positions] == pytest.approx(
        accepted_sleeve
    )
    with pytest.raises(AdaptivePortfolioDataError, match="operational sleeve weight must be in"):
        evaluate_operational_adaptive_portfolio(candidates, rejected, budget=_lax_budget())


def test_direct_operational_grid_rejects_non_grid_and_wrong_exposure() -> None:
    candidates = _candidate_set(4)

    with pytest.raises(AdaptivePortfolioDataError, match="exact 10% increments"):
        evaluate_operational_adaptive_portfolio(
            candidates,
            {candidate.symbol: 0.20 for candidate in candidates},
            budget=_lax_budget(),
        )
    with pytest.raises(AdaptivePortfolioDataError, match="must sum to exposure"):
        evaluate_operational_adaptive_portfolio(
            candidates,
            {candidate.symbol: 0.16 for candidate in candidates},
            budget=_lax_budget(),
        )


def test_direct_operational_grid_enforces_total_account_industry_cap_at_boundary() -> None:
    candidates = _candidate_set(4, industries=("科技", "科技", "金融", "消费"))
    at_cap = dict(
        zip(
            (candidate.symbol for candidate in candidates),
            (0.24, 0.16, 0.24, 0.16),
            strict=True,
        )
    )
    above_cap = dict(
        zip(
            (candidate.symbol for candidate in candidates),
            (0.24, 0.24, 0.16, 0.16),
            strict=True,
        )
    )

    result = evaluate_operational_adaptive_portfolio(candidates, at_cap, budget=_lax_budget())
    assert dict(result.industry_weights)["科技"] == pytest.approx(0.40)
    with pytest.raises(AdaptivePortfolioDataError, match="industry cap"):
        evaluate_operational_adaptive_portfolio(candidates, above_cap, budget=_lax_budget())


def test_metrics_match_the_documented_formulas() -> None:
    candidates = _candidate_set(4, periods=260)
    weights_by_symbol = {candidate.symbol: 0.20 for candidate in candidates}
    budget = _lax_budget(holding_period_cost_rate=0.001, lcb_confidence=0.90)

    result = evaluate_adaptive_portfolio(candidates, weights_by_symbol, budget=budget)
    ordered = tuple(sorted(candidates, key=lambda item: item.symbol))
    returns = np.column_stack([item.returns.to_numpy() for item in ordered])
    weights = np.asarray([position.weight for position in result.positions])
    exact_target = np.asarray([weight for _symbol, weight in result.exact_target_weights])
    assert not np.array_equal(weights, exact_target)
    daily_portfolio = returns @ weights

    expected_downside_volatility = np.sqrt(
        np.mean(np.square(np.minimum(daily_portfolio, 0.0)))
    ) * np.sqrt(252)
    drawdowns = []
    for start in range(len(daily_portfolio) - 60 + 1):
        equity = np.concatenate(([1.0], np.cumprod(1.0 + daily_portfolio[start : start + 60])))
        drawdowns.append(max(0.0, -float((equity / np.maximum.accumulate(equity) - 1).min())))

    five_day_blocks = returns.reshape(52, 5, 4)
    five_day_returns = (np.prod(1.0 + five_day_blocks, axis=1) - 1.0) @ weights
    five_day_cutoff = np.quantile(five_day_returns, 0.05)
    expected_es = max(0.0, -float(five_day_returns[five_day_returns <= five_day_cutoff].mean()))

    holding_blocks = returns.reshape(13, 20, 4)
    holding_returns = (np.prod(1.0 + holding_blocks, axis=1) - 1.0) @ weights - 0.001
    expected_lcb = float(
        holding_returns.mean()
        - NormalDist().inv_cdf(0.90) * holding_returns.std(ddof=1) / math.sqrt(len(holding_returns))
    )

    downside = np.minimum(returns, 0.0)
    second_moment = downside.T @ downside / len(downside)
    marginal = second_moment @ weights
    contributions = weights * marginal / float(weights @ marginal)

    metrics = result.metrics
    assert metrics.annual_downside_volatility == pytest.approx(expected_downside_volatility)
    assert metrics.rolling_max_drawdown_60_p90 == pytest.approx(np.quantile(drawdowns, 0.90))
    assert metrics.es95_5d == pytest.approx(expected_es)
    assert metrics.max_position_downside_risk_contribution == pytest.approx(contributions.max())
    assert metrics.holding_period_return_mean == pytest.approx(holding_returns.mean())
    assert metrics.holding_period_return_lcb == pytest.approx(expected_lcb)
    assert sum(
        position.downside_risk_contribution for position in result.positions
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("holding_weeks", "holding_sessions"),
    ((1, 5), (2, 10), (4, 20), (13, 60), (26, 120), (52, 252)),
)
def test_horizon_rolling_drawdown_uses_each_holding_window(
    holding_weeks: int,
    holding_sessions: int,
) -> None:
    periods = 600
    candidates = _candidate_set(4, periods=periods)
    weights_by_symbol = {candidate.symbol: 0.20 for candidate in candidates}
    result = evaluate_adaptive_portfolio(
        candidates,
        weights_by_symbol,
        budget=_lax_budget(
            holding_period_sessions=holding_sessions,
            minimum_holding_period_samples=2,
        ),
    )

    ordered = tuple(sorted(candidates, key=lambda item: item.symbol))
    returns = np.column_stack([item.returns.to_numpy() for item in ordered])
    weights = np.asarray([position.weight for position in result.positions])
    daily_portfolio = returns @ weights
    expected_drawdowns = []
    for start in range(periods - holding_sessions + 1):
        equity = np.concatenate(
            (
                [1.0],
                np.cumprod(1.0 + daily_portfolio[start : start + holding_sessions]),
            )
        )
        expected_drawdowns.append(
            max(0.0, -float((equity / np.maximum.accumulate(equity) - 1.0).min()))
        )

    assert holding_weeks in (1, 2, 4, 13, 26, 52)
    assert result.metrics.horizon_rolling_drawdown_window_sessions == holding_sessions
    assert result.metrics.horizon_rolling_drawdown_window_count == (periods - holding_sessions + 1)
    assert result.metrics.horizon_rolling_max_drawdown_p90 == pytest.approx(
        np.quantile(expected_drawdowns, 0.90)
    )
    assert result.risk_budget.horizon_rolling_drawdown_window_sessions == holding_sessions
    assert result.risk_budget.horizon_rolling_drawdown_limit == pytest.approx(1.0)
    assert result.risk_budget.rolling_drawdown_passed == (
        result.risk_budget.horizon_rolling_drawdown_passed
    )


def test_legacy_60_day_drawdown_is_report_only_while_horizon_budget_binds() -> None:
    periods = 800
    dates = pd.bdate_range("2023-01-02", periods=periods)
    cycle = np.asarray([-0.01] * 30 + [0.004] * 70, dtype=float)
    shared = np.resize(cycle, periods)
    candidates = tuple(
        AdaptiveCandidate(
            symbol=f"00000{index + 1}.SZ",
            industry=f"行业{index}",
            signal_score=index / 3,
            returns=pd.Series(shared, index=dates),
        )
        for index in range(4)
    )
    weights = {candidate.symbol: 0.20 for candidate in candidates}

    short = evaluate_adaptive_portfolio(
        candidates,
        weights,
        budget=_lax_budget(
            max_rolling_drawdown_60_p90=0.06,
            holding_period_sessions=5,
        ),
    )
    long = evaluate_adaptive_portfolio(
        candidates,
        weights,
        budget=_lax_budget(
            max_rolling_drawdown_60_p90=0.06,
            holding_period_sessions=252,
            minimum_holding_period_samples=2,
        ),
    )

    assert short.metrics.rolling_max_drawdown_60_p90 > 0.06
    assert short.metrics.horizon_rolling_max_drawdown_p90 < 0.06
    assert short.risk_budget.horizon_rolling_drawdown_passed
    assert "rolling_drawdown_60_p90" not in short.risk_budget.violations
    assert "horizon_rolling_drawdown_p90" not in short.risk_budget.violations
    assert long.metrics.rolling_max_drawdown_60_p90 == pytest.approx(
        short.metrics.rolling_max_drawdown_60_p90
    )
    assert long.metrics.horizon_rolling_max_drawdown_p90 > 0.06
    assert not long.risk_budget.horizon_rolling_drawdown_passed
    assert "horizon_rolling_drawdown_p90" in long.risk_budget.violations


def test_explicit_horizon_drawdown_limit_overrides_legacy_fallback() -> None:
    candidates = _candidate_set(4)
    weights = {candidate.symbol: 0.20 for candidate in candidates}
    result = evaluate_adaptive_portfolio(
        candidates,
        weights,
        budget=_lax_budget(
            max_rolling_drawdown_60_p90=1.0,
            max_horizon_rolling_drawdown_p90=0.000001,
        ),
    )

    assert result.risk_budget.horizon_rolling_drawdown_limit == pytest.approx(0.000001)
    assert not result.risk_budget.horizon_rolling_drawdown_passed
    assert not result.risk_budget.rolling_drawdown_passed
    assert "horizon_rolling_drawdown_p90" in result.risk_budget.violations


def test_appended_horizon_budget_field_preserves_legacy_positional_construction() -> None:
    budget = AdaptiveRiskBudget(
        0.18,
        0.12,
        0.08,
        0.75,
        0.45,
        0.40,
        20,
        0.0,
        0.90,
        160,
        20,
        8,
        None,
    )

    assert budget.max_rolling_drawdown_60_p90 == pytest.approx(0.12)
    assert budget.max_horizon_rolling_drawdown_p90 is None


@pytest.mark.parametrize("value", (0.0, -0.1, 1.01, float("inf"), True))
def test_invalid_horizon_drawdown_limit_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="max_horizon_rolling_drawdown_p90"):
        _lax_budget(max_horizon_rolling_drawdown_p90=value)


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


def test_risk_budget_reports_metric_breaches_without_loosening_limits() -> None:
    candidates = _candidate_set(4)
    weights = {candidate.symbol: 0.20 for candidate in candidates}
    budget = AdaptiveRiskBudget(
        max_annual_downside_volatility=0.0001,
        max_rolling_drawdown_60_p90=0.0001,
        max_es95_5d=0.0001,
        max_down_period_correlation=-0.99,
        max_position_downside_risk_contribution=0.20,
        industry_weight_limit=0.40,
    )

    result = evaluate_adaptive_portfolio(candidates, weights, budget=budget)

    assert not result.risk_budget.passed
    assert "annual_downside_volatility" in result.risk_budget.violations
    assert "position_downside_risk_contribution" in result.risk_budget.violations
    assert "industry_concentration" not in result.risk_budget.violations
    assert result.risk_budget.industry_concentration_passed
    assert [position.weight for position in result.positions] == pytest.approx(
        (0.24, 0.24, 0.16, 0.16)
    )
    assert "no farther-grid risk rescue" in result.method


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
