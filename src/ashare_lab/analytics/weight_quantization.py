"""Deterministically quantize stock-sleeve weights to integer step units.

The helper is intentionally independent from portfolio construction and UI
code.  By default it searches every positive integer composition of ten units,
so the
returned allocation is the global minimum of squared weight error under these
constraints:

* three to five securities;
* one unit (ten percent by default) minimum per security;
* ten units in total under the default step;
* original input order is preserved.

When more than one allocation has exactly the same squared error, the earlier
input position receives the larger unit count.  This makes ties reproducible
without sorting or otherwise changing security order.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from decimal import Decimal
from numbers import Real

DEFAULT_QUANTIZATION_STEP = Decimal("0.10")
WEIGHT_QUANTIZATION_METHOD_VERSION = "exhaustive-stock-sleeve-grid-v1.0.0"
MINIMUM_UNITS_PER_SECURITY = 1
_SUM_TOLERANCE = Decimal("1e-9")
_INTEGER_TOLERANCE = Decimal("1e-12")


def quantize_stock_sleeve_weights(
    weights: Sequence[float],
    *,
    step: float = 0.10,
    minimum_weight: float | None = None,
    maximum_weight: float | None = None,
    group_labels: Sequence[str] | None = None,
    maximum_group_weight: float | None = None,
) -> tuple[float, ...]:
    """Return the closest integer-step allocation in the original order.

    Closeness is the sum of squared differences from the supplied exact stock-
    sleeve weights.  Inputs must contain three to five finite, strictly
    positive real values whose sum is one (within floating-point tolerance).
    ``step`` defaults to ten percent and must be a finite positive reciprocal
    of an integer that leaves at least one unit per security.  Optional minimum,
    maximum, and group constraints are part of the exhaustive search rather
    than checks applied after rounding.  Invalid, internally inconsistent, or
    structurally infeasible inputs raise :class:`ValueError`.
    """

    exact = _validated_weights(weights)
    step_decimal, total_units = _validated_step(step, security_count=len(exact))
    minimum_units = _validated_bound_units(
        minimum_weight,
        default_units=MINIMUM_UNITS_PER_SECURITY,
        step=step_decimal,
        total_units=total_units,
        field="minimum_weight",
    )
    maximum_units = _validated_bound_units(
        maximum_weight,
        default_units=total_units,
        step=step_decimal,
        total_units=total_units,
        field="maximum_weight",
    )
    if minimum_units > maximum_units:
        raise ValueError("minimum_weight cannot exceed maximum_weight")
    if minimum_units * len(exact) > total_units or maximum_units * len(exact) < total_units:
        raise ValueError("per-security bounds cannot sum to 1.0")
    labels, group_cap = _validated_group_constraint(
        group_labels,
        maximum_group_weight,
        count=len(exact),
    )
    best_units: tuple[int, ...] | None = None
    best_key: tuple[Decimal, tuple[int, ...]] | None = None

    for units in _positive_unit_compositions(total_units, len(exact)):
        if any(unit < minimum_units or unit > maximum_units for unit in units):
            continue
        if labels is not None and group_cap is not None:
            group_units: dict[str, int] = {}
            for label, unit in zip(labels, units, strict=True):
                group_units[label] = group_units.get(label, 0) + unit
            if any(
                Decimal(unit_count) * step_decimal > group_cap + _SUM_TOLERANCE
                for unit_count in group_units.values()
            ):
                continue
        squared_error = sum(
            (Decimal(unit) * step_decimal - weight) ** 2
            for unit, weight in zip(units, exact, strict=True)
        )
        # On an exact error tie, favour the earlier input position.  Negative
        # unit counts turn the normal ascending tuple comparison into a stable
        # earlier-position-first rule.
        key = (squared_error, tuple(-unit for unit in units))
        if best_key is None or key < best_key:
            best_key = key
            best_units = units

    if best_units is None:  # Defensive: valid 3--5-name inputs always have a composition.
        raise ValueError("no feasible integer-step allocation")
    return tuple(float(Decimal(unit) * step_decimal) for unit in best_units)


def _validated_weights(weights: Sequence[float]) -> tuple[Decimal, ...]:
    if isinstance(weights, (str, bytes)):
        raise ValueError("weights must be a sequence of real numbers")
    try:
        values = tuple(weights)
    except TypeError as exc:
        raise ValueError("weights must be a sequence of real numbers") from exc
    if not 3 <= len(values) <= 5:
        raise ValueError("weights must contain three to five securities")

    exact: list[Decimal] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"weight at index {index} must be a finite real number")
        weight = Decimal(str(value))
        if not weight.is_finite() or weight <= 0 or weight > 1:
            raise ValueError(f"weight at index {index} must be in (0, 1]")
        exact.append(weight)

    total = sum(exact, start=Decimal(0))
    if abs(total - Decimal(1)) > _SUM_TOLERANCE:
        raise ValueError("weights must sum to 1.0")
    return tuple(exact)


def _validated_step(step: object, *, security_count: int) -> tuple[Decimal, int]:
    if isinstance(step, bool) or not isinstance(step, Real):
        raise ValueError("step must be a finite positive real number")
    step_decimal = Decimal(str(step))
    if not step_decimal.is_finite() or not 0 < step_decimal <= 1:
        raise ValueError("step must be in (0, 1]")
    reciprocal = Decimal(1) / step_decimal
    total_units = int(reciprocal.to_integral_value())
    if abs(reciprocal - Decimal(total_units)) > _INTEGER_TOLERANCE:
        raise ValueError("step must divide 1.0 into an integer number of units")
    if total_units < security_count * MINIMUM_UNITS_PER_SECURITY:
        raise ValueError("step leaves fewer than one unit per security")
    return step_decimal, total_units


def _validated_bound_units(
    value: object,
    *,
    default_units: int,
    step: Decimal,
    total_units: int,
    field: str,
) -> int:
    if value is None:
        return default_units
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real number")
    bound = Decimal(str(value))
    if not bound.is_finite() or not 0 < bound <= 1:
        raise ValueError(f"{field} must be in (0, 1]")
    units = bound / step
    rounded_units = int(units.to_integral_value())
    if abs(units - Decimal(rounded_units)) > _INTEGER_TOLERANCE:
        raise ValueError(f"{field} must be an integer multiple of step")
    if not MINIMUM_UNITS_PER_SECURITY <= rounded_units <= total_units:
        raise ValueError(f"{field} is outside the feasible unit range")
    return rounded_units


def _validated_group_constraint(
    labels: Sequence[str] | None,
    maximum_group_weight: object,
    *,
    count: int,
) -> tuple[tuple[str, ...] | None, Decimal | None]:
    if labels is None:
        if maximum_group_weight is not None:
            raise ValueError("maximum_group_weight requires group_labels")
        return None, None
    if isinstance(labels, (str, bytes)):
        raise ValueError("group_labels must be a sequence")
    try:
        normalized = tuple(str(label).strip() for label in labels)
    except TypeError as exc:
        raise ValueError("group_labels must be a sequence") from exc
    if len(normalized) != count or any(not label for label in normalized):
        raise ValueError("group_labels must match weights and be non-blank")
    if maximum_group_weight is None:
        raise ValueError("group_labels require maximum_group_weight")
    if isinstance(maximum_group_weight, bool) or not isinstance(maximum_group_weight, Real):
        raise ValueError("maximum_group_weight must be a finite real number")
    group_cap = Decimal(str(maximum_group_weight))
    if not group_cap.is_finite() or not 0 < group_cap <= 1:
        raise ValueError("maximum_group_weight must be in (0, 1]")
    return normalized, group_cap


def _positive_unit_compositions(total: int, count: int) -> Iterator[tuple[int, ...]]:
    """Yield all ordered ``count``-part positive integer compositions."""

    if count == 1:
        if total >= MINIMUM_UNITS_PER_SECURITY:
            yield (total,)
        return
    maximum_first = total - MINIMUM_UNITS_PER_SECURITY * (count - 1)
    for first in range(MINIMUM_UNITS_PER_SECURITY, maximum_first + 1):
        for remainder in _positive_unit_compositions(total - first, count - 1):
            yield (first, *remainder)
