from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_lab.analytics.levels import build_horizon_levels


def _structured_history() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=280)
    trend = np.linspace(20.0, 31.0, len(dates))
    wave = 1.8 * np.sin(np.linspace(0.0, 18.0 * np.pi, len(dates)))
    close = trend + wave
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": close - 0.10,
            "high": close + 0.55,
            "low": close - 0.55,
            "close": close,
            "atr14": np.full(len(dates), 0.90),
        }
    )


def test_edwards_magee_levels_require_confirmation_and_define_two_exits() -> None:
    frame = _structured_history()
    levels = build_horizon_levels(frame)

    assert [item.sessions for item in levels] == [5, 20, 60, 120, 252]
    cutoff = frame.iloc[-1]["trade_date"].date().isoformat()
    for item in levels:
        assert item.invalidation < item.pullback_entry_low
        assert item.pullback_entry_low <= item.entry_trigger < item.breakout_trigger
        assert item.breakout_trigger > item.reduce_low
        assert item.second_reduce_low >= item.reduce_low
        assert item.second_reduce_high >= item.second_reduce_low
        assert item.measured_move_target >= item.reduce_low
        assert item.first_reduce_fraction == 0.50
        assert item.reward_risk_ratio is not None and item.reward_risk_ratio > 0
        assert "收盘价确认" in item.breakout_confirmation_rule
        assert "1.2倍" in item.breakout_confirmation_rule
        assert "下一可成交价格" in item.stop_execution_rule
        assert all(day <= cutoff for day in item.level_evidence_dates)


def test_zero_atr_uses_a_bounded_price_based_fallback() -> None:
    frame = _structured_history()
    frame["atr14"] = 0.0

    item = build_horizon_levels(frame)[0]

    assert item.breakout_trigger > item.reduce_low
    assert item.invalidation < item.pullback_entry_low
