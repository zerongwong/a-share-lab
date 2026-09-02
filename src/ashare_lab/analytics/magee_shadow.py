"""Pure, point-in-time Edwards--Magee stop shadows.

The engines in this module are deliberately isolated from holding actions,
repositories, charts, and notifications.  They provide two daily-bar research
counterfactuals that can be compared with the production holding review:

* a three-day escape baseline with a six-percent protective distance; and
* a three-percent new-high ratchet with the same protective distance.

Only bars on or after the actual position entry date and on or before ``as_of``
are observed.  A baseline confirmed on one bar becomes effective on the next
observed trading bar, so the result never uses same-bar or future knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import ROUND_FLOOR, Decimal
from enum import StrEnum
from math import isfinite
from typing import Final

import pandas as pd

MAGEE_SHADOW_METHOD_VERSION: Final = "magee-shadow-daily-v0.1.0"
_STOP_RETAINED_FRACTION: Final = Decimal("0.94")
_NEW_HIGH_MULTIPLIER: Final = Decimal("1.03")
_CENT: Final = Decimal("0.01")
_REQUIRED_COLUMNS: Final = ("trade_date", "open", "high", "low", "close")


class MageeShadowVariant(StrEnum):
    THREE_DAY_ESCAPE_6PCT = "three_day_escape_6pct"
    NEW_HIGH_3PCT_6PCT = "new_high_3pct_6pct"


class MageeShadowEventKind(StrEnum):
    THREE_DAY_ESCAPE = "three_day_escape"
    NEW_HIGH_3PCT = "new_high_3pct"


class MageeShadowDataError(ValueError):
    """Raised when an observed bar cannot support a fail-closed shadow result."""


@dataclass(frozen=True, slots=True)
class MageeShadowEvent:
    kind: MageeShadowEventKind
    baseline_date: date
    confirmed_on: date
    effective_on: date | None
    baseline_price: float
    candidate_stop: float
    previous_stop: float | None
    effective_stop: float | None
    raised: bool | None
    accepted_high: float | None = None

    @property
    def pending(self) -> bool:
        return self.effective_on is None


@dataclass(frozen=True, slots=True)
class MageeShadowResult:
    method_version: str
    variant: MageeShadowVariant
    entry_date: date
    as_of: date
    data_cutoff: date | None
    latest_close: float | None
    latest_low: float | None
    effective_stop: float | None
    pending_stop: float | None
    next_effective_stop: float | None
    # These two fields describe only the latest observed completed bar.
    close_below_stop: bool | None
    low_touched_stop: bool | None
    # The cumulative fields retain an earlier breach even after price recovers.
    ever_close_below_stop: bool | None
    ever_low_touched_stop: bool | None
    first_close_breach_on: date | None
    first_low_touch_on: date | None
    events: tuple[MageeShadowEvent, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ObservedBar:
    trade_date: date
    open: float
    high: float
    low: float
    close: float


@dataclass(slots=True)
class _EscapeCandidate:
    trade_date: date
    high: float
    low: float
    escape_count: int = 0


def evaluate_magee_shadow(
    frame: pd.DataFrame,
    *,
    entry_date: date | datetime | str | pd.Timestamp,
    as_of: date | datetime | str | pd.Timestamp,
    variant: MageeShadowVariant | str,
) -> MageeShadowResult:
    """Evaluate one research-only stop shadow from completed daily bars.

    The primary breach is a completed close strictly below the line.  An
    intraday low at or below the line is retained only as a diagnostic; neither
    value is a broker order or a production holding action.
    """

    entry = _as_date(entry_date, field="entry_date")
    cutoff = _as_date(as_of, field="as_of")
    if entry > cutoff:
        raise MageeShadowDataError("entry_date_after_as_of")
    try:
        selected_variant = MageeShadowVariant(variant)
    except ValueError as exc:
        raise MageeShadowDataError(f"unsupported_magee_shadow_variant:{variant}") from exc

    bars = _observed_bars(frame, entry_date=entry, as_of=cutoff)
    if not bars:
        return MageeShadowResult(
            method_version=MAGEE_SHADOW_METHOD_VERSION,
            variant=selected_variant,
            entry_date=entry,
            as_of=cutoff,
            data_cutoff=None,
            latest_close=None,
            latest_low=None,
            effective_stop=None,
            pending_stop=None,
            next_effective_stop=None,
            close_below_stop=None,
            low_touched_stop=None,
            ever_close_below_stop=None,
            ever_low_touched_stop=None,
            first_close_breach_on=None,
            first_low_touch_on=None,
            events=(),
            reasons=(
                "research_shadow_only_no_holding_action",
                "no_observed_daily_bar_on_or_after_entry",
            ),
        )

    if selected_variant is MageeShadowVariant.THREE_DAY_ESCAPE_6PCT:
        events, effective_stop = _three_day_escape_events(bars)
    else:
        events, effective_stop = _new_high_events(bars)

    pending_stop = events[-1].candidate_stop if events and events[-1].pending else None
    next_effective_stop = (
        max(effective_stop or pending_stop, pending_stop)
        if pending_stop is not None
        else effective_stop
    )
    latest = bars[-1]
    close_below = None if effective_stop is None else latest.close < effective_stop
    low_touched = None if effective_stop is None else latest.low <= effective_stop
    (
        ever_close_below,
        ever_low_touched,
        first_close_breach_on,
        first_low_touch_on,
    ) = _breach_history(bars, events)
    reasons = [
        "research_shadow_only_no_holding_action",
        "daily_completed_bars_entry_date_through_as_of_only",
        "confirmed_line_effective_on_next_observed_trading_bar",
        "effective_stop_never_moves_down",
        "latest_complete_close_below_line_is_primary_current_breach",
        "latest_intraday_low_touch_is_current_diagnostic_only",
        "first_and_cumulative_breach_history_is_retained",
    ]
    if pending_stop is not None:
        reasons.append("confirmed_baseline_waiting_for_next_observed_bar")

    return MageeShadowResult(
        method_version=MAGEE_SHADOW_METHOD_VERSION,
        variant=selected_variant,
        entry_date=entry,
        as_of=cutoff,
        data_cutoff=latest.trade_date,
        latest_close=latest.close,
        latest_low=latest.low,
        effective_stop=effective_stop,
        pending_stop=pending_stop,
        next_effective_stop=next_effective_stop,
        close_below_stop=close_below,
        low_touched_stop=low_touched,
        ever_close_below_stop=ever_close_below,
        ever_low_touched_stop=ever_low_touched,
        first_close_breach_on=first_close_breach_on,
        first_low_touch_on=first_low_touch_on,
        events=tuple(events),
        reasons=tuple(reasons),
    )


def _three_day_escape_events(
    bars: tuple[_ObservedBar, ...],
) -> tuple[list[MageeShadowEvent], float | None]:
    events: list[MageeShadowEvent] = []
    pending_index: int | None = None
    effective_stop: float | None = None
    candidate = _EscapeCandidate(bars[0].trade_date, bars[0].high, bars[0].low)
    previous_bar = bars[0]

    for index, bar in enumerate(bars):
        if index > 0 and pending_index is not None:
            effective_stop = _activate_event(events, pending_index, bar.trade_date, effective_stop)
            pending_index = None

        if index == 0:
            continue

        if candidate is None:
            if bar.low < previous_bar.low:
                candidate = _EscapeCandidate(bar.trade_date, bar.high, bar.low)
        elif bar.low < candidate.low:
            candidate = _EscapeCandidate(bar.trade_date, bar.high, bar.low)
        elif bar.low > candidate.high:
            candidate.escape_count += 1
            if candidate.escape_count == 3:
                events.append(
                    MageeShadowEvent(
                        kind=MageeShadowEventKind.THREE_DAY_ESCAPE,
                        baseline_date=candidate.trade_date,
                        confirmed_on=bar.trade_date,
                        effective_on=None,
                        baseline_price=candidate.low,
                        candidate_stop=_six_percent_below(candidate.low),
                        previous_stop=None,
                        effective_stop=None,
                        raised=None,
                    )
                )
                pending_index = len(events) - 1
                candidate = None
        else:
            candidate.escape_count = 0
        previous_bar = bar

    return events, effective_stop


def _new_high_events(
    bars: tuple[_ObservedBar, ...],
) -> tuple[list[MageeShadowEvent], float | None]:
    events: list[MageeShadowEvent] = []
    pending_index: int | None = None
    effective_stop: float | None = None
    accepted_high = bars[0].high

    for index, bar in enumerate(bars):
        if index > 0 and pending_index is not None:
            effective_stop = _activate_event(events, pending_index, bar.trade_date, effective_stop)
            pending_index = None

        if index == 0:
            continue
        if _decimal(bar.high) < _decimal(accepted_high) * _NEW_HIGH_MULTIPLIER:
            continue

        accepted_high = bar.high
        events.append(
            MageeShadowEvent(
                kind=MageeShadowEventKind.NEW_HIGH_3PCT,
                baseline_date=bar.trade_date,
                confirmed_on=bar.trade_date,
                effective_on=None,
                baseline_price=bar.low,
                candidate_stop=_six_percent_below(bar.low),
                previous_stop=None,
                effective_stop=None,
                raised=None,
                accepted_high=accepted_high,
            )
        )
        pending_index = len(events) - 1

    return events, effective_stop


def _activate_event(
    events: list[MageeShadowEvent],
    event_index: int,
    effective_on: date,
    previous_stop: float | None,
) -> float:
    event = events[event_index]
    effective_stop = max(previous_stop or event.candidate_stop, event.candidate_stop)
    events[event_index] = replace(
        event,
        effective_on=effective_on,
        previous_stop=previous_stop,
        effective_stop=effective_stop,
        raised=previous_stop is None or effective_stop > previous_stop,
    )
    return effective_stop


def _breach_history(
    bars: tuple[_ObservedBar, ...],
    events: list[MageeShadowEvent],
) -> tuple[bool | None, bool | None, date | None, date | None]:
    active_stop: float | None = None
    event_by_effective_date = {
        event.effective_on: event
        for event in events
        if event.effective_on is not None and event.effective_stop is not None
    }
    line_was_active = False
    first_close_breach_on: date | None = None
    first_low_touch_on: date | None = None
    for bar in bars:
        event = event_by_effective_date.get(bar.trade_date)
        if event is not None:
            active_stop = event.effective_stop
        if active_stop is None:
            continue
        line_was_active = True
        if first_close_breach_on is None and bar.close < active_stop:
            first_close_breach_on = bar.trade_date
        if first_low_touch_on is None and bar.low <= active_stop:
            first_low_touch_on = bar.trade_date
    if not line_was_active:
        return None, None, None, None
    return (
        first_close_breach_on is not None,
        first_low_touch_on is not None,
        first_close_breach_on,
        first_low_touch_on,
    )


def _observed_bars(
    frame: pd.DataFrame,
    *,
    entry_date: date,
    as_of: date,
) -> tuple[_ObservedBar, ...]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise MageeShadowDataError(f"daily_bar_columns_missing:{','.join(missing)}")

    dated = frame.copy()
    parsed_dates = pd.to_datetime(dated["trade_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise MageeShadowDataError("daily_bar_trade_date_invalid")
    if parsed_dates.dt.tz is not None:
        parsed_dates = parsed_dates.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    dated["_trade_date"] = parsed_dates.dt.date
    observed = dated.loc[
        (dated["_trade_date"] >= entry_date) & (dated["_trade_date"] <= as_of)
    ].copy()
    if observed.empty:
        return ()
    if observed["_trade_date"].duplicated().any():
        raise MageeShadowDataError("duplicate_observed_daily_bar")
    observed = observed.sort_values("_trade_date").reset_index(drop=True)

    for column in ("open", "high", "low", "close"):
        observed[column] = pd.to_numeric(observed[column], errors="coerce")
        if observed[column].isna().any() or not all(
            isfinite(float(value)) and float(value) > 0.0 for value in observed[column]
        ):
            raise MageeShadowDataError(f"observed_daily_bar_{column}_invalid")

    bars: list[_ObservedBar] = []
    values = observed.loc[:, ["_trade_date", "open", "high", "low", "close"]]
    for trade_date, raw_open, raw_high, raw_low, raw_close in values.itertuples(
        index=False,
        name=None,
    ):
        open_price = float(raw_open)
        high = float(raw_high)
        low = float(raw_low)
        close = float(raw_close)
        if high < max(open_price, low, close) or low > min(open_price, high, close):
            raise MageeShadowDataError("observed_daily_bar_ohlc_inconsistent")
        bars.append(
            _ObservedBar(
                trade_date=trade_date,
                open=open_price,
                high=high,
                low=low,
                close=close,
            )
        )
    return tuple(bars)


def _six_percent_below(baseline: float) -> float:
    stop = (_decimal(baseline) * _STOP_RETAINED_FRACTION).quantize(
        _CENT,
        rounding=ROUND_FLOOR,
    )
    if stop <= 0:
        raise MageeShadowDataError("magee_candidate_stop_not_positive")
    return float(stop)


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _as_date(value: date | datetime | str | pd.Timestamp, *, field: str) -> date:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise MageeShadowDataError(f"{field}_invalid") from exc
    if pd.isna(timestamp):
        raise MageeShadowDataError(f"{field}_invalid")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return timestamp.date()
