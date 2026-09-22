"""Versioned holding-count policy for the continuous A-share strategy.

The legacy fixed-horizon research path keeps its historical three-to-five
portfolio contract.  This module applies only to ``continuous-signal-v3``.
It separates a valid empty account from concentrated transition states and
formed portfolios; no count is a permission to weaken an entry or risk gate.
"""

from __future__ import annotations

import math

CONTINUOUS_COUNT_POLICY_VERSION = "continuous-count-policy-v3.0.0"

MIN_CONTINUOUS_HOLDINGS = 1
MAX_CONTINUOUS_HOLDINGS = 5
FORMED_PORTFOLIO_MIN_HOLDINGS = 3
PREFERRED_HOLDING_COUNTS = (3, 4, 5)

NORMAL_NEW_ACCOUNT_WEIGHT = 0.15
MAX_NEW_ACCOUNT_WEIGHT = 0.20
MIN_MEAN_ACCOUNT_WEIGHT = 0.08

# Count-specific ceilings are total-account stock exposures before the market
# cycle overlay. One or two names remain low-exposure transition states only
# when no qualified three-to-five-name construction is available. The five-name
# upper bound never authorizes relaxing an entry or risk gate to fill a slot.
CONTINUOUS_POSITION_LIMITS: dict[int, tuple[float, float, float]] = {
    1: (0.15, 0.15, 0.15),
    2: (0.30, 0.10, 0.20),
    3: (0.45, 0.10, 0.20),
    4: (0.60, 0.10, 0.20),
    5: (0.75, 0.10, 0.20),
}

# Bounds remain fractions of the stock sleeve because the established
# operational interface uses a ten-point sleeve grid.  The account hard cap is
# applied independently after the cycle overlay.
CONTINUOUS_OPERATION_STOCK_SLEEVE_LIMITS: dict[int, tuple[float, float]] = {
    1: (1.00, 1.00),
    2: (0.40, 0.60),
    3: (0.20, 0.40),
    4: (0.20, 0.30),
    5: (0.20, 0.20),
}


def continuous_count_state(count: int) -> str:
    """Return the user-facing state for an exact live holding count."""

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("holding count must be an integer")
    if count == 0:
        return "cash"
    if not MIN_CONTINUOUS_HOLDINGS <= count <= MAX_CONTINUOUS_HOLDINGS:
        raise ValueError("continuous holding count must be between zero and five")
    if count < FORMED_PORTFOLIO_MIN_HOLDINGS:
        return "concentrated_transition"
    if count in PREFERRED_HOLDING_COUNTS:
        return "preferred_formed"
    return "formed"


def continuous_count_preference(maximum_stock_exposure: float | None) -> tuple[int, ...]:
    """Order counts by the exposure-linked 15% account-weight centre.

    Formed three-to-five-name sets come before one-to-two-name transitions.
    Inside each group this is a tie-break preference, never a reason to weaken
    an entry or risk gate. The selector still compares historical return and
    joint risk among all eligible formed sets before considering a transition.
    """

    supplied = 0.80 if maximum_stock_exposure is None else float(maximum_stock_exposure)
    if not math.isfinite(supplied) or supplied <= 0.0:
        raise ValueError("maximum stock exposure must be positive and finite")
    exposure = min(0.80, supplied)
    target = exposure / NORMAL_NEW_ACCOUNT_WEIGHT
    counts = tuple(range(MIN_CONTINUOUS_HOLDINGS, MAX_CONTINUOUS_HOLDINGS + 1))
    return tuple(
        sorted(
            counts,
            key=lambda count: (
                count < FORMED_PORTFOLIO_MIN_HOLDINGS,
                exposure / count < MIN_MEAN_ACCOUNT_WEIGHT - 1e-12,
                abs(count - target),
                0 if count in PREFERRED_HOLDING_COUNTS else 1,
                count,
            ),
        )
    )


__all__ = [
    "CONTINUOUS_COUNT_POLICY_VERSION",
    "CONTINUOUS_OPERATION_STOCK_SLEEVE_LIMITS",
    "CONTINUOUS_POSITION_LIMITS",
    "FORMED_PORTFOLIO_MIN_HOLDINGS",
    "MAX_CONTINUOUS_HOLDINGS",
    "MAX_NEW_ACCOUNT_WEIGHT",
    "MIN_CONTINUOUS_HOLDINGS",
    "NORMAL_NEW_ACCOUNT_WEIGHT",
    "PREFERRED_HOLDING_COUNTS",
    "continuous_count_preference",
    "continuous_count_state",
]
