import numpy as np
import pandas as pd

from ashare_lab.analytics.indicators import enrich_indicators
from ashare_lab.analytics.trend import confirmed_swings


def test_swing_is_only_known_after_confirmation_bar():
    high = [1, 2, 5, 2, 1]
    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-01-01", periods=5),
            "open": np.ones(5),
            "high": high,
            "low": np.zeros(5),
            "close": np.ones(5),
        }
    )
    point = [p for p in confirmed_swings(frame, left=2, right=2) if p.kind == "high"][0]
    assert point.trade_date == frame.iloc[2]["trade_date"]
    assert point.confirmed_at == frame.iloc[4]["trade_date"]


def test_prior_breakout_level_excludes_current_bar():
    close = pd.Series(np.arange(1, 70, dtype=float))
    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=69, freq="B"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume_shares": 100,
        }
    )
    result = enrich_indicators(frame)
    assert result.iloc[-1]["prior_high_60"] == frame.iloc[-61:-1]["high"].max()
