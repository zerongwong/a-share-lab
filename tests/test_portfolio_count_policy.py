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
        (3, "concentrated_transition"),
        (4, "formed"),
        (5, "preferred_formed"),
        (6, "preferred_formed"),
        (8, "formed"),
    ),
)
def test_continuous_count_state(count: int, state: str) -> None:
    assert continuous_count_state(count) == state


@pytest.mark.parametrize("count", (-1, 9))
def test_continuous_count_state_rejects_out_of_range(count: int) -> None:
    with pytest.raises(ValueError):
        continuous_count_state(count)


def test_count_preference_centres_on_five_or_six_without_excluding_other_counts() -> None:
    order = continuous_count_preference(0.80)

    assert order[:2] == (5, 6)
    assert set(order) == set(range(1, 9))


def test_lower_cycle_exposure_moves_preference_without_mandating_slot_filling() -> None:
    order = continuous_count_preference(0.30)

    assert order[0] == 2
    assert set(order) == set(range(1, 9))
