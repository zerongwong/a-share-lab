from __future__ import annotations

import pytest

from ashare_lab.analytics.portfolio_count_policy import (
    continuous_count_preference,
    continuous_count_state,
)


@pytest.mark.parametrize(
    ("count", "state"),
    (
        (0, "cash"),
        (1, "concentrated_transition"),
        (2, "concentrated_transition"),
        (3, "preferred_formed"),
        (4, "preferred_formed"),
        (5, "preferred_formed"),
    ),
)
def test_continuous_count_state(count: int, state: str) -> None:
    assert continuous_count_state(count) == state


@pytest.mark.parametrize("count", (-1, 6, 8, 9))
def test_continuous_count_state_rejects_out_of_range(count: int) -> None:
    with pytest.raises(ValueError):
        continuous_count_state(count)


def test_count_preference_targets_three_to_five_with_smaller_fallbacks() -> None:
    order = continuous_count_preference(0.80)

    assert order[0] == 5
    assert set(order[:3]) == {3, 4, 5}
    assert set(order[3:]) == {1, 2}


def test_lower_cycle_exposure_moves_preference_without_mandating_slot_filling() -> None:
    order = continuous_count_preference(0.30)

    assert order[0] == 3
    assert set(order[:3]) == {3, 4, 5}
    assert set(order[3:]) == {1, 2}
