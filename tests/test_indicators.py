import numpy as np
import pandas as pd

from ashare_lab.analytics.indicators import atr, enrich_indicators, rsi


def make_frame(rows: int = 80) -> pd.DataFrame:
    close = pd.Series(np.linspace(10, 20, rows))
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=rows, freq="B"),
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume_shares": 1_000_000,
        }
    )


def test_wilder_rsi_is_bounded_and_rising_series_reaches_100():
    result = rsi(make_frame()["close"])
    assert result.dropna().between(0, 100).all()
    assert result.iloc[-1] == 100


def test_atr_and_prior_high_do_not_use_future_rows():
    frame = make_frame()
    enriched = enrich_indicators(frame)
    assert atr(frame).iloc[-1] > 0
    expected = frame.iloc[-61:-1]["high"].max()
    assert enriched.iloc[-1]["prior_high_60"] == expected
