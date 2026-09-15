"""Cost-based loss guard on verified, comparable completed daily bars.

This is a user-selected risk limit, not an optimized parameter or an execution
guarantee. It is independent of technical-pattern eligibility/history length.
Corporate-action clearance is enforced by the holding-review caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import pandas as pd

COST_STOP_METHOD_VERSION = "confirmed-cost-loss-8pct-v1"
COST_STOP_RETAINED_FRACTION = Decimal("0.92")


@dataclass(frozen=True, slots=True)
class CostStopObservation:
    line: float
    touched_on: date | None
    latest_close: float
    close_loss_fraction: float
    line_history: tuple[tuple[str, float], ...]


def observe_cost_stop(
    daily: pd.DataFrame,
    *,
    cost_price: float,
    entry_date: date,
    remembered_line: float | None = None,
    remembered_touch: date | None = None,
    cost_update_date: date | None = None,
    remembered_history: list | tuple | None = None,
) -> CostStopObservation:
    """Use an inclusive, unrounded 8% boundary; retain earlier breaches.

    The entry-day low may precede the fill, so use only its close. Subsequent
    complete days use the low, including a touch followed by a recovery. Never
    reset a raised cost line or a triggered alert after an average-cost update.
    A genuinely new position must have a new confirmed position identity.
    ``daily`` must already be validated, cutoff-clipped, unadjusted OHLC.
    """

    cost = Decimal(str(cost_price))
    if not cost.is_finite() or cost <= 0:
        raise ValueError("confirmed_cost_must_be_finite_and_positive")
    line = cost * COST_STOP_RETAINED_FRACTION
    if remembered_line is not None:
        old_line = Decimal(str(remembered_line))
        if not old_line.is_finite() or old_line <= 0:
            raise ValueError("remembered_cost_line_invalid")
        line = max(line, old_line)
    close = Decimal(str(daily.iloc[-1]["close"]))
    touched_on = remembered_touch
    history = [
        (date.fromisoformat(day), Decimal(str(value))) for day, value in (remembered_history or ())
    ]
    if not history:
        history = [
            (entry_date, Decimal(str(remembered_line)) if remembered_line is not None else line)
        ]
    if line > history[-1][1]:
        if cost_update_date is None:
            raise ValueError("raised_cost_basis_requires_confirmed_update_date")
        history.append((max(entry_date, cost_update_date), line))
    if history[0][0] != entry_date or any(
        not value.is_finite() or value <= 0 for _, value in history
    ):
        raise ValueError("cost_stop_history_invalid")
    if any(
        history[i][0] > history[i + 1][0] or history[i][1] > history[i + 1][1]
        for i in range(len(history) - 1)
    ):
        raise ValueError("cost_stop_history_cannot_move_backwards")
    if history[-1][0] > pd.Timestamp(daily.iloc[-1]["trade_date"]).date():
        raise ValueError("cost_update_after_verified_price_cutoff")
    for bar in daily.itertuples(index=False):
        bar_date = pd.Timestamp(bar.trade_date).date()
        if bar_date < entry_date:
            continue
        # Do not apply a newly raised cost basis to lows before the confirmed
        # update, nor to a low that may precede that day's additional fill.
        active_day, comparison_line = [(day, value) for day, value in history if day <= bar_date][
            -1
        ]
        observation = bar.close if bar_date == active_day else bar.low
        if Decimal(str(observation)) <= comparison_line:
            touched_on = min(touched_on, bar_date) if touched_on else bar_date
    return CostStopObservation(
        line=float(line),
        touched_on=touched_on,
        latest_close=float(close),
        close_loss_fraction=float(close / cost - 1),
        line_history=tuple((day.isoformat(), float(value)) for day, value in history),
    )
