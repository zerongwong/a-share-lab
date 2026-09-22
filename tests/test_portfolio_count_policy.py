from __future__ import annotations

import pytest

from ashare_lab.analytics.portfolio_count_policy import (
    CONTINUOUS_COUNT_POLICY_VERSION,
    continuous_count_preference,
    continuous_count_state,
    continuous_portfolio_selection_key,
)


@pytest.mark.parametrize(
    ("count", "state"),
    (
        (0, "cash"),
        (1, "concentrated"),
        (2, "concentrated"),
        (3, "diversified"),
        (4, "diversified"),
        (5, "diversified"),
    ),
)
def test_continuous_count_state(count: int, state: str) -> None:
    assert continuous_count_state(count) == state


@pytest.mark.parametrize("count", (-1, 6, 8, 9))
def test_continuous_count_state_rejects_out_of_range(count: int) -> None:
    with pytest.raises(ValueError):
        continuous_count_state(count)


def test_v4_compares_all_feasible_counts_not_formed_first() -> None:
    assert CONTINUOUS_COUNT_POLICY_VERSION == "continuous-count-policy-v4.0.0"
    order = continuous_count_preference(0.80)

    assert order[0] == 5
    assert set(order) == {1, 2, 3, 4, 5}


def test_lower_cycle_exposure_moves_preference_without_mandating_slot_filling() -> None:
    order = continuous_count_preference(0.30)

    assert order[0] == 2
    assert set(order) == {1, 2, 3, 4, 5}
    assert continuous_count_preference(0.15)[0] == 1


def _selection_key(count: int, *, lcb: float = 0.02, **overrides) -> tuple:
    values = {
        "historical_return_lcb": lcb,
        "annual_downside_volatility": 0.06,
        "horizon_drawdown": 0.04,
        "es95_5d": 0.01,
        "max_down_period_correlation": 0.40 if count > 1 else None,
        "max_position_downside_risk_contribution": 0.40 if count > 2 else None,
        "symbols": tuple(f"STOCK{index}" for index in range(count)),
        "maximum_stock_exposure": 0.80,
    }
    return continuous_portfolio_selection_key(**(values | overrides))


@pytest.mark.parametrize("count", (1, 2))
def test_stronger_small_allocation_beats_every_qualified_larger_count(count: int) -> None:
    # Every input is assumed to have passed its risk gates. Count does not
    # override the documented total-account historical return-LCB objective.
    small = _selection_key(count, lcb=0.04)
    for larger in (3, 4, 5):
        assert small < _selection_key(larger, lcb=0.03)


def test_larger_allocation_still_wins_when_historical_objective_is_stronger() -> None:
    assert _selection_key(5, lcb=0.04) < _selection_key(1, lcb=0.03)


def test_joint_risk_precedes_exposure_linked_count_tiebreak() -> None:
    assert _selection_key(2, annual_downside_volatility=0.05) < _selection_key(5)
    assert _selection_key(3, max_down_period_correlation=0.30) < _selection_key(5)
    assert _selection_key(3, max_position_downside_risk_contribution=0.30) < _selection_key(5)


def test_count_preference_applies_only_after_return_and_risk_ties() -> None:
    assert _selection_key(5) < _selection_key(4)
    assert _selection_key(3, maximum_stock_exposure=0.45) < _selection_key(
        5, maximum_stock_exposure=0.45
    )


@pytest.mark.parametrize("count", (0, 6))
def test_selection_key_rejects_cash_and_over_five_equity_inputs(count: int) -> None:
    # Cash has a separate zero-return baseline; it is not a fabricated stock set.
    with pytest.raises(ValueError, match="one and five"):
        _selection_key(count)


def test_selection_key_preserves_missing_concentration_not_zero() -> None:
    key = _selection_key(1)
    assert key[4] == (True, 0.0)
    assert key[5] == (True, 0.0)
    assert key[4] != _selection_key(2, max_down_period_correlation=0.0)[4]


@pytest.mark.parametrize("metric", ("historical_return_lcb", "max_down_period_correlation"))
def test_selection_key_rejects_nonfinite_metric(metric: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        _selection_key(3, **{metric: float("nan")})
