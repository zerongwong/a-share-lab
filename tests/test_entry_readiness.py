from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_lab.analytics.entry_readiness import (
    EntryPattern,
    assess_entry_readiness,
)
from ashare_lab.analytics.medium_term_stage import MediumTermStage


def _frame(
    close: np.ndarray,
    *,
    amount: np.ndarray | None = None,
    spread: float = 0.15,
) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    if amount is None:
        amount = np.full(len(close), 100.0)
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=len(close), freq="D"),
            "open": close - min(spread / 2, 0.05),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "amount_cny": np.asarray(amount, dtype=float),
        }
    )


def _fresh_breakout_frame(*, amount_ratio: float = 1.30, spread: float = 0.15) -> pd.DataFrame:
    close = np.concatenate(
        [
            np.linspace(10.0, 15.5, 155),
            np.linspace(15.4, 15.6, 20),
            np.array([16.2, 16.25, 16.3, 16.35, 16.4]),
        ]
    )
    amount = np.full(len(close), 100.0)
    amount[175] = 100.0 * amount_ratio
    return _frame(close, amount=amount, spread=spread)


def test_recent_volume_breakout_uses_shifted_prior_high() -> None:
    frame = _fresh_breakout_frame()
    result = assess_entry_readiness(frame, as_of=frame.iloc[-1]["trade_date"])

    assert result.ready
    assert result.pattern == EntryPattern.VOLUME_BREAKOUT
    assert result.stage == MediumTermStage.ORDERLY_UPTREND
    assert result.days_since_breakout == 4
    assert result.breakout_line is not None
    # The breakout day's own high is above its close.  Recognition therefore
    # proves that the 60-session reference excludes the current bar.
    assert result.breakout_line < frame.iloc[175]["close"] < frame.iloc[175]["high"]
    assert result.breakout_amount_ratio == pytest.approx(1.30)
    assert 0.0 < result.score <= 1.0
    assert "shifted_prior_60_high_close_confirmed" in result.reasons


def test_healthy_pullback_after_volume_breakout_is_recognized() -> None:
    close = np.concatenate(
        [
            np.linspace(10.0, 15.5, 150),
            np.linspace(15.4, 15.6, 20),
            np.array([16.2, 16.0, 15.85, 15.95, 16.05, 16.1, 16.12, 16.14, 16.16, 16.18]),
        ]
    )
    amount = np.full(len(close), 100.0)
    amount[170] = 140.0
    frame = _frame(close, amount=amount)

    result = assess_entry_readiness(frame, as_of=frame.iloc[-1]["trade_date"])

    assert result.ready
    assert result.pattern == EntryPattern.HEALTHY_PULLBACK
    assert result.days_since_breakout == 9
    assert result.breakout_amount_ratio == pytest.approx(1.40)


def test_controlled_reclaim_of_breakout_line_is_recognized() -> None:
    close = np.concatenate(
        [
            np.linspace(10.0, 15.5, 150),
            np.linspace(15.4, 15.6, 20),
            np.array([16.2, 15.9, 15.65, 15.72, 15.82, 16.0, 16.1, 16.2, 16.3, 16.35]),
        ]
    )
    amount = np.full(len(close), 100.0)
    amount[170] = 150.0
    frame = _frame(close, amount=amount)

    result = assess_entry_readiness(frame, as_of=frame.iloc[-1]["trade_date"])

    assert result.ready
    assert result.pattern == EntryPattern.BREAKOUT_RECLAIM
    assert result.breakout_line == pytest.approx(15.75)
    assert result.days_since_breakout == 9


def test_cutoff_excludes_a_valid_breakout_that_occurs_later() -> None:
    frame = _fresh_breakout_frame()
    before_breakout = pd.Timestamp(frame.iloc[174]["trade_date"])

    earlier = assess_entry_readiness(frame, as_of=before_breakout)
    current = assess_entry_readiness(frame, as_of=frame.iloc[-1]["trade_date"])

    assert not earlier.ready
    assert earlier.data_cutoff == before_breakout.date()
    assert earlier.reasons == ("no_volume_confirmed_breakout_within_30_sessions",)
    assert current.ready


def test_breakout_turnover_must_be_at_least_prior_twenty_day_median_times_1_10() -> None:
    frame = _fresh_breakout_frame(amount_ratio=1.09)

    result = assess_entry_readiness(frame, as_of=frame.iloc[-1]["trade_date"])

    assert not result.ready
    assert result.pattern == EntryPattern.NO_SIGNAL
    assert result.reasons == ("no_volume_confirmed_breakout_within_30_sessions",)


@pytest.mark.parametrize(
    ("close", "expected_stage"),
    [
        (np.linspace(30.0, 10.0, 180), MediumTermStage.DOWNTREND),
        (np.full(180, 15.0), MediumTermStage.RANGE),
        (
            np.concatenate([np.linspace(10.0, 15.0, 140), np.linspace(15.0, 20.0, 40)]),
            MediumTermStage.EXTENDED,
        ),
        (
            np.concatenate([np.linspace(10.0, 12.0, 255), np.linspace(12.2, 16.5, 5)]),
            MediumTermStage.PARABOLIC,
        ),
    ],
)
def test_disallowed_medium_term_stages_fail_closed(
    close: np.ndarray,
    expected_stage: MediumTermStage,
) -> None:
    frame = _frame(close)

    result = assess_entry_readiness(frame, as_of=frame.iloc[-1]["trade_date"])

    assert not result.ready
    assert result.score == 0.0
    assert result.stage == expected_stage
    assert result.reasons == (f"stage_not_entry_ready:{expected_stage.value}",)


def test_missing_or_non_finite_market_fields_fail_closed() -> None:
    missing_amount = _fresh_breakout_frame().drop(columns="amount_cny")
    missing_result = assess_entry_readiness(
        missing_amount,
        as_of=missing_amount.iloc[-1]["trade_date"],
    )
    assert not missing_result.ready
    assert missing_result.reasons == ("missing_required_columns:amount_cny",)

    non_finite = _fresh_breakout_frame()
    non_finite.loc[non_finite.index[-3], "low"] = np.nan
    non_finite_result = assess_entry_readiness(
        non_finite,
        as_of=non_finite.iloc[-1]["trade_date"],
    )
    assert not non_finite_result.ready
    assert non_finite_result.reasons == ("non_finite_ohlc_or_amount",)


def test_price_too_far_above_ma20_in_atr_units_fails_closed() -> None:
    frame = _fresh_breakout_frame(spread=0.01)
    frame.loc[175:, ["open", "high", "low", "close"]] += 0.25

    result = assess_entry_readiness(frame, as_of=frame.iloc[-1]["trade_date"])

    assert not result.ready
    assert result.score == 0.0
    assert "distance_ma20_atr_exceeds_3" in result.reasons
