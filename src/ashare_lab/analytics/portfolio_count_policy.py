"""Versioned holding-count policy for the continuous A-share strategy.

The legacy fixed-horizon research path keeps its historical three-to-five
portfolio contract.  This module applies only to the continuous strategy.
It treats cash and one-to-five-name allocations as genuine alternatives. No
count is a permission to weaken an entry or risk gate or add inferior names.
"""

from __future__ import annotations

import math

CONTINUOUS_COUNT_POLICY_VERSION = "continuous-count-policy-v4.0.0"

MIN_CONTINUOUS_HOLDINGS = 1
MAX_CONTINUOUS_HOLDINGS = 5
RISK_CONTRIBUTION_MIN_HOLDINGS = 3

NORMAL_NEW_ACCOUNT_WEIGHT = 0.15
MAX_NEW_ACCOUNT_WEIGHT = 0.20
MIN_MEAN_ACCOUNT_WEIGHT = 0.08

# Count-specific ceilings are total-account stock exposures before the market
# cycle overlay. One or two names remain low-exposure allocations but compete
# directly with larger feasible sets. The five-name upper bound never authorizes
# relaxing an entry or risk gate to fill a slot.
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
    return "concentrated" if count < RISK_CONTRIBUTION_MIN_HOLDINGS else "diversified"


def continuous_count_preference(maximum_stock_exposure: float | None) -> tuple[int, ...]:
    """Order counts by the exposure-linked 15% account-weight centre.

    This is only a deterministic tie-break after historical return and joint
    risk, never a reason to prefer a worse portfolio or weaken an entry gate.
    One and two names do not wait for larger sets to fail.
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
                exposure / count < MIN_MEAN_ACCOUNT_WEIGHT - 1e-12,
                abs(count - target),
                count,
            ),
        )
    )


def continuous_portfolio_selection_key(
    *,
    historical_return_lcb: float,
    annual_downside_volatility: float,
    horizon_drawdown: float,
    es95_5d: float,
    max_down_period_correlation: float | None,
    max_position_downside_risk_contribution: float | None,
    symbols: tuple[str, ...],
    maximum_stock_exposure: float | None = None,
) -> tuple:
    """Rank already-feasible sets, independently of count, using frozen history.

    The objective is the total-account historical net-return lower confidence
    bound (LCB), not a forecast, future Sharpe or global market optimum. Equal
    LCBs prefer smaller downside volatility, drawdown and tail loss, followed
    by applicable concentration metrics, then count and stable symbol order.
    Inapplicable concentration metrics remain missing, not a fabricated zero;
    when all preceding components tie, an observed value precedes missing.

    The caller must apply every eligibility/risk gate and compare with cash
    (zero historical price return) before using this ordering. It must not pass
    an unqualified set or force a nonpositive-LCB purchase through this helper.
    """

    count = len(symbols)
    if not MIN_CONTINUOUS_HOLDINGS <= count <= MAX_CONTINUOUS_HOLDINGS:
        raise ValueError("selection requires between one and five symbols")
    if len(set(symbols)) != count or any(not symbol for symbol in symbols):
        raise ValueError("selection symbols must be unique and non-blank")
    values = (historical_return_lcb, annual_downside_volatility, horizon_drawdown, es95_5d)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("selection metrics must be finite")

    def applicable_metric(value: float | None) -> tuple[bool, float]:
        if value is None:
            return (True, 0.0)
        if not math.isfinite(value):
            raise ValueError("selection metrics must be finite when applicable")
        return (False, value)

    count_order = continuous_count_preference(maximum_stock_exposure)
    return (
        -historical_return_lcb,
        annual_downside_volatility,
        horizon_drawdown,
        es95_5d,
        applicable_metric(max_down_period_correlation),
        applicable_metric(max_position_downside_risk_contribution),
        count_order.index(count),
        tuple(sorted(symbols)),
    )


__all__ = [
    "CONTINUOUS_COUNT_POLICY_VERSION",
    "CONTINUOUS_OPERATION_STOCK_SLEEVE_LIMITS",
    "CONTINUOUS_POSITION_LIMITS",
    "MAX_CONTINUOUS_HOLDINGS",
    "MAX_NEW_ACCOUNT_WEIGHT",
    "MIN_CONTINUOUS_HOLDINGS",
    "NORMAL_NEW_ACCOUNT_WEIGHT",
    "RISK_CONTRIBUTION_MIN_HOLDINGS",
    "continuous_count_preference",
    "continuous_count_state",
    "continuous_portfolio_selection_key",
]
