from __future__ import annotations

import math

import pytest

from ashare_lab.analytics.weight_quantization import quantize_stock_sleeve_weights


def _squared_error(actual: tuple[float, ...], target: tuple[float, ...]) -> float:
    return sum((left - right) ** 2 for left, right in zip(actual, target, strict=True))


def test_exact_ten_percent_allocation_is_unchanged() -> None:
    assert quantize_stock_sleeve_weights((0.50, 0.30, 0.20)) == (0.50, 0.30, 0.20)


def test_five_security_allocation_uses_global_nearest_units() -> None:
    exact = (0.34, 0.26, 0.18, 0.12, 0.10)

    quantized = quantize_stock_sleeve_weights(exact)

    assert quantized == (0.30, 0.30, 0.20, 0.10, 0.10)
    assert sum(quantized) == pytest.approx(1.0)
    assert all(weight >= 0.10 for weight in quantized)
    assert all(math.isclose(weight * 10, round(weight * 10)) for weight in quantized)


def test_exact_error_tie_favours_earlier_input_position() -> None:
    assert quantize_stock_sleeve_weights((0.50, 0.25, 0.25)) == (0.50, 0.30, 0.20)


def test_small_positive_weight_still_receives_ten_percent_minimum() -> None:
    assert quantize_stock_sleeve_weights((0.90, 0.09, 0.01)) == (0.80, 0.10, 0.10)


def test_input_order_is_preserved_instead_of_sorting_by_weight() -> None:
    forward = quantize_stock_sleeve_weights((0.46, 0.34, 0.20))
    reverse = quantize_stock_sleeve_weights((0.20, 0.34, 0.46))

    assert forward == (0.50, 0.30, 0.20)
    assert reverse == tuple(reversed(forward))


def test_selected_allocation_has_no_higher_error_than_any_feasible_three_name_set() -> None:
    exact = (0.381, 0.333, 0.286)
    observed = quantize_stock_sleeve_weights(exact)
    observed_error = _squared_error(observed, exact)

    for first in range(1, 9):
        for second in range(1, 10 - first):
            third = 10 - first - second
            if third < 1:
                continue
            candidate = (first / 10, second / 10, third / 10)
            assert observed_error <= _squared_error(candidate, exact) + 1e-15


@pytest.mark.parametrize(
    "weights",
    [
        (0.6, 0.4),
        (0.20, 0.20, 0.15, 0.15, 0.15, 0.15),
        (0.50, 0.50, 0.0),
        (0.60, 0.50, -0.10),
        (0.50, 0.30, float("nan")),
        (0.50, 0.30, float("inf")),
        (0.50, 0.30, True),
        (0.50, 0.30, "0.20"),
        (0.50, 0.30, 0.19),
        (0.50, 0.30, 0.21),
    ],
)
def test_invalid_inputs_fail_closed(weights: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        quantize_stock_sleeve_weights(weights)  # type: ignore[arg-type]


def test_tiny_binary_sum_error_is_accepted() -> None:
    weights = (0.1, 0.2, 0.7000000000000001)

    assert quantize_stock_sleeve_weights(weights) == (0.10, 0.20, 0.70)


def test_explicit_alternative_step_is_supported_but_does_not_change_default() -> None:
    assert quantize_stock_sleeve_weights((0.50, 0.30, 0.20), step=0.20) == (
        0.60,
        0.20,
        0.20,
    )
    assert quantize_stock_sleeve_weights((0.50, 0.30, 0.20)) == (0.50, 0.30, 0.20)


@pytest.mark.parametrize("step", [0.0, -0.1, float("nan"), float("inf"), True, 0.3])
def test_invalid_step_fails_closed(step: object) -> None:
    with pytest.raises(ValueError):
        quantize_stock_sleeve_weights((0.50, 0.30, 0.20), step=step)  # type: ignore[arg-type]


def test_step_must_leave_one_unit_for_every_security() -> None:
    with pytest.raises(ValueError, match="one unit per security"):
        quantize_stock_sleeve_weights((0.20, 0.20, 0.20, 0.20, 0.20), step=0.25)


def test_per_security_bounds_are_enforced_inside_global_search() -> None:
    result = quantize_stock_sleeve_weights(
        (0.70, 0.20, 0.10),
        minimum_weight=0.20,
        maximum_weight=0.50,
    )

    assert result == (0.50, 0.30, 0.20)


def test_group_cap_selects_nearest_feasible_grid_not_unconstrained_rounding() -> None:
    result = quantize_stock_sleeve_weights(
        (0.40, 0.30, 0.20, 0.10),
        minimum_weight=0.10,
        maximum_weight=0.40,
        group_labels=("科技", "科技", "金融", "消费"),
        maximum_group_weight=0.50,
    )

    assert result == (0.30, 0.20, 0.30, 0.20)
    assert result[0] + result[1] <= 0.50


def test_structurally_infeasible_grid_fails_closed() -> None:
    with pytest.raises(ValueError, match="no feasible"):
        quantize_stock_sleeve_weights(
            (0.25, 0.25, 0.25, 0.25),
            minimum_weight=0.10,
            maximum_weight=0.40,
            group_labels=("甲", "乙", "丙", "丁"),
            maximum_group_weight=0.20,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"minimum_weight": 0.15}, "integer multiple"),
        ({"maximum_weight": 0.35}, "integer multiple"),
        ({"minimum_weight": 0.40, "maximum_weight": 0.30}, "cannot exceed"),
        ({"group_labels": ("甲", "乙", "丙")}, "require maximum_group_weight"),
        ({"maximum_group_weight": 0.40}, "requires group_labels"),
        (
            {"group_labels": ("甲", "乙"), "maximum_group_weight": 0.40},
            "must match weights",
        ),
    ],
)
def test_invalid_structural_constraints_fail_closed(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        quantize_stock_sleeve_weights(  # type: ignore[arg-type]
            (0.40, 0.30, 0.30),
            **kwargs,
        )
