from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from ashare_lab.analytics.magee_shadow import (
    MAGEE_SHADOW_METHOD_VERSION,
    MageeShadowDataError,
    MageeShadowVariant,
    evaluate_magee_shadow,
)


def _frame(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=("trade_date", "open", "high", "low", "close"))


def test_three_strict_escape_bars_confirm_at_close_and_activate_next_bar() -> None:
    frame = _frame(
        [
            ("2026-08-03", 10.5, 11.0, 10.0, 10.6),
            ("2026-08-04", 11.2, 11.5, 11.01, 11.3),
            ("2026-08-05", 11.3, 11.6, 11.02, 11.4),
            ("2026-08-06", 11.4, 11.7, 11.03, 11.5),
            ("2026-08-07", 11.5, 11.8, 11.10, 11.6),
        ]
    )

    pending = evaluate_magee_shadow(
        frame,
        entry_date="2026-08-03",
        as_of="2026-08-06",
        variant="three_day_escape_6pct",
    )
    assert pending.effective_stop is None
    assert pending.pending_stop == 9.40
    assert pending.next_effective_stop == 9.40
    assert pending.events[0].confirmed_on == date(2026, 8, 6)
    assert pending.events[0].effective_on is None

    active = evaluate_magee_shadow(
        frame,
        entry_date="2026-08-03",
        as_of="2026-08-07",
        variant=MageeShadowVariant.THREE_DAY_ESCAPE_6PCT,
    )
    assert active.method_version == MAGEE_SHADOW_METHOD_VERSION
    assert active.effective_stop == 9.40
    assert active.events[0].effective_on == date(2026, 8, 7)
    assert active.events[0].raised is True


def test_escape_requires_strictly_above_and_non_escape_resets_counter() -> None:
    frame = _frame(
        [
            ("2026-08-03", 10.5, 11.0, 10.0, 10.6),
            ("2026-08-04", 11.2, 11.5, 11.01, 11.3),
            ("2026-08-05", 11.1, 11.4, 11.00, 11.2),
            ("2026-08-06", 11.2, 11.5, 11.01, 11.3),
            ("2026-08-07", 11.3, 11.6, 11.02, 11.4),
            ("2026-08-10", 11.4, 11.7, 11.03, 11.5),
        ]
    )
    result = evaluate_magee_shadow(
        frame,
        entry_date="2026-08-03",
        as_of="2026-08-10",
        variant="three_day_escape_6pct",
    )

    assert len(result.events) == 1
    assert result.events[0].confirmed_on == date(2026, 8, 10)


def test_lower_low_replaces_escape_candidate() -> None:
    frame = _frame(
        [
            ("2026-08-03", 10.5, 11.0, 10.0, 10.6),
            ("2026-08-04", 11.2, 11.5, 11.10, 11.3),
            ("2026-08-05", 10.0, 10.5, 9.50, 10.2),
            ("2026-08-06", 10.7, 10.9, 10.51, 10.8),
            ("2026-08-07", 10.8, 11.0, 10.60, 10.9),
            ("2026-08-10", 10.9, 11.1, 10.70, 11.0),
        ]
    )
    result = evaluate_magee_shadow(
        frame,
        entry_date="2026-08-03",
        as_of="2026-08-10",
        variant="three_day_escape_6pct",
    )

    assert result.events[0].baseline_date == date(2026, 8, 5)
    assert result.events[0].baseline_price == 9.50
    assert result.events[0].candidate_stop == 8.93


def test_new_high_uses_last_accepted_high_and_three_percent_is_inclusive() -> None:
    frame = _frame(
        [
            ("2026-08-03", 9.5, 10.00, 9.00, 9.8),
            ("2026-08-04", 10.0, 10.29, 9.70, 10.1),
            ("2026-08-05", 10.1, 10.30, 9.80, 10.2),
            ("2026-08-06", 10.3, 10.50, 10.00, 10.4),
            ("2026-08-07", 10.5, 10.60, 10.10, 10.5),
            ("2026-08-10", 10.6, 10.61, 10.20, 10.6),
        ]
    )
    result = evaluate_magee_shadow(
        frame,
        entry_date="2026-08-03",
        as_of="2026-08-10",
        variant="new_high_3pct_6pct",
    )

    assert [event.accepted_high for event in result.events] == [10.30, 10.61]
    assert result.events[0].baseline_date == date(2026, 8, 5)
    assert result.events[0].candidate_stop == 9.21
    assert result.events[0].effective_on == date(2026, 8, 6)
    assert result.events[1].effective_on is None


def test_new_high_stop_ratchets_and_never_moves_down() -> None:
    frame = _frame(
        [
            ("2026-08-03", 9.5, 10.00, 9.00, 9.8),
            ("2026-08-04", 10.1, 10.30, 9.80, 10.2),
            ("2026-08-05", 10.2, 10.40, 10.00, 10.3),
            ("2026-08-06", 10.0, 10.61, 9.00, 10.4),
            ("2026-08-07", 10.4, 10.70, 10.20, 10.5),
        ]
    )
    result = evaluate_magee_shadow(
        frame,
        entry_date="2026-08-03",
        as_of="2026-08-07",
        variant="new_high_3pct_6pct",
    )

    assert result.events[0].effective_stop == 9.21
    assert result.events[1].candidate_stop == 8.46
    assert result.events[1].effective_stop == 9.21
    assert result.events[1].raised is False
    assert result.effective_stop == 9.21


@pytest.mark.parametrize(
    ("latest_low", "latest_close", "expected_close", "expected_touch"),
    ((9.20, 9.39, True, True), (9.39, 9.50, False, True)),
)
def test_close_breach_is_primary_and_low_touch_is_diagnostic(
    latest_low: float,
    latest_close: float,
    expected_close: bool,
    expected_touch: bool,
) -> None:
    frame = _frame(
        [
            ("2026-08-03", 10.5, 11.0, 10.0, 10.6),
            ("2026-08-04", 11.2, 11.5, 11.01, 11.3),
            ("2026-08-05", 11.3, 11.6, 11.02, 11.4),
            ("2026-08-06", 11.4, 11.7, 11.03, 11.5),
            ("2026-08-07", 9.60, 9.80, latest_low, latest_close),
        ]
    )
    result = evaluate_magee_shadow(
        frame,
        entry_date="2026-08-03",
        as_of="2026-08-07",
        variant="three_day_escape_6pct",
    )

    assert result.close_below_stop is expected_close
    assert result.low_touched_stop is expected_touch


def test_pre_entry_extremes_and_future_dirty_ohlc_do_not_affect_result() -> None:
    observed = _frame(
        [
            ("2026-08-03", 10.5, 11.0, 10.0, 10.6),
            ("2026-08-04", 11.2, 11.5, 11.01, 11.3),
            ("2026-08-05", 11.3, 11.6, 11.02, 11.4),
            ("2026-08-06", 11.4, 11.7, 11.03, 11.5),
        ]
    )
    contaminated = pd.concat(
        [
            _frame([("2026-07-31", 50.0, 100.0, 1.0, 50.0)]),
            observed,
            _frame([("2026-08-07", -1.0, -2.0, -3.0, -4.0)]),
        ],
        ignore_index=True,
    )

    clean = evaluate_magee_shadow(
        observed,
        entry_date="2026-08-03",
        as_of="2026-08-06",
        variant="three_day_escape_6pct",
    )
    dirty = evaluate_magee_shadow(
        contaminated,
        entry_date="2026-08-03",
        as_of="2026-08-06",
        variant="three_day_escape_6pct",
    )
    assert dirty == clean


def test_earlier_breach_dates_remain_after_latest_bar_recovers() -> None:
    frame = _frame(
        [
            ("2026-08-03", 10.5, 11.0, 10.0, 10.6),
            ("2026-08-04", 11.2, 11.5, 11.01, 11.3),
            ("2026-08-05", 11.3, 11.6, 11.02, 11.4),
            ("2026-08-06", 11.4, 11.7, 11.03, 11.5),
            ("2026-08-07", 9.60, 9.80, 9.20, 9.30),
            ("2026-08-10", 10.0, 10.4, 9.80, 10.2),
        ]
    )
    result = evaluate_magee_shadow(
        frame,
        entry_date="2026-08-03",
        as_of="2026-08-10",
        variant="three_day_escape_6pct",
    )

    assert result.close_below_stop is False
    assert result.low_touched_stop is False
    assert result.ever_close_below_stop is True
    assert result.ever_low_touched_stop is True
    assert result.first_close_breach_on == date(2026, 8, 7)
    assert result.first_low_touch_on == date(2026, 8, 7)


def test_invalid_observed_ohlc_fails_closed() -> None:
    frame = _frame([("2026-08-03", 10.0, 9.0, 8.0, 10.0)])
    with pytest.raises(MageeShadowDataError, match="ohlc_inconsistent"):
        evaluate_magee_shadow(
            frame,
            entry_date="2026-08-03",
            as_of="2026-08-03",
            variant="three_day_escape_6pct",
        )
