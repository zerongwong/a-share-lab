from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ashare_lab.analytics.portfolio_statistics import (
    HISTORICAL_OVERLAP_NOTICE,
    calculate_portfolio_statistics,
    rolling_max_drawdown_magnitudes,
    wilson_score_interval,
)


def test_rolling_drawdown_includes_window_starting_equity() -> None:
    returns = pd.Series([0.10, -0.20, 0.05, 0.01])
    result = rolling_max_drawdown_magnitudes(returns, 2)

    assert len(result) == 3
    assert result.iloc[0] == pytest.approx(0.20)
    assert result.iloc[1] == pytest.approx(0.20)
    assert result.between(0.0, 1.0).all()


def test_wilson_interval_is_bounded_and_contains_estimate() -> None:
    interval = wilson_score_interval(20, 100, confidence=0.90)

    assert interval.estimate == pytest.approx(0.20)
    assert 0.0 <= interval.lower <= interval.estimate
    assert interval.estimate <= interval.upper <= 1.0
    assert interval.successes == 20
    assert interval.sample_n == 100


def test_portfolio_statistics_add_calmar_and_window_drawdown_probabilities() -> None:
    rng = np.random.default_rng(42)
    returns = pd.Series(0.0005 + rng.normal(0.0, 0.008, 320))
    result = calculate_portfolio_statistics(
        returns,
        drawdown_budget=0.05,
        minimum_window_samples=30,
    )

    assert result.historical_calmar is not None
    assert result.historical_cagr is not None
    assert result.historical_max_drawdown is not None
    assert result.historical_calmar == pytest.approx(
        result.historical_cagr / abs(result.historical_max_drawdown)
    )
    assert {item.window_sessions for item in result.drawdown_windows} == {20, 40, 60}
    for item in result.drawdown_windows:
        assert item.available
        assert item.drawdown_magnitude_p10 <= item.drawdown_magnitude_p50
        assert item.drawdown_magnitude_p50 <= item.drawdown_magnitude_p90
        assert item.breach_probability is not None
        assert item.breach_interval is not None
        assert item.breach_probability == pytest.approx(item.breach_interval.estimate)
        assert not item.is_out_of_sample
        assert not item.is_forecast_probability
        assert not item.is_promise


def test_insufficient_window_is_unavailable_not_filled_with_fake_statistics() -> None:
    returns = pd.Series(np.full(70, 0.001))
    result = calculate_portfolio_statistics(
        returns,
        drawdown_budget=0.05,
        window_sessions=(60,),
        minimum_window_samples=30,
    )

    distribution = result.drawdown_windows[0]
    assert not distribution.available
    assert distribution.sample_n == 11
    assert distribution.breach_probability is None
    assert distribution.breach_interval is None
    assert distribution.disclaimer == HISTORICAL_OVERLAP_NOTICE


def test_statistics_reject_impossible_returns_and_invalid_wilson_counts() -> None:
    with pytest.raises(ValueError, match="-100%"):
        calculate_portfolio_statistics(pd.Series([0.01, -1.0]), drawdown_budget=0.10)
    with pytest.raises(ValueError, match="between zero"):
        wilson_score_interval(11, 10)
    with pytest.raises(ValueError, match="below one"):
        calculate_portfolio_statistics(pd.Series([0.01] * 30), drawdown_budget=1.0)


def test_zero_drawdown_does_not_invent_infinite_calmar() -> None:
    result = calculate_portfolio_statistics(
        pd.Series([0.001] * 100),
        drawdown_budget=0.05,
        minimum_window_samples=1,
    )
    assert result.historical_calmar is None
    assert not math.isinf(result.historical_cagr)
