from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_lab.analytics.market_regime import MarketRegimeState, assess_market_regime

DATES = pd.bdate_range("2025-01-01", periods=180)


def _universe(start: float, end: float) -> dict[str, pd.DataFrame]:
    return {
        f"{index:06d}": pd.DataFrame(
            {"trade_date": DATES, "close": np.linspace(start + index, end + index, len(DATES))}
        )
        for index in range(8)
    }


def test_broad_uptrend_is_risk_on() -> None:
    result = assess_market_regime(_universe(50.0, 100.0), DATES[-1])
    assert result.state == MarketRegimeState.RISK_ON
    assert result.breadth_above_ma120 == 1.0


def test_broad_downtrend_is_risk_off() -> None:
    result = assess_market_regime(_universe(100.0, 50.0), DATES[-1])
    assert result.state == MarketRegimeState.RISK_OFF
    assert result.breadth_above_ma60 == 0.0


def test_future_bars_cannot_change_earlier_regime() -> None:
    histories = _universe(50.0, 100.0)
    earlier = assess_market_regime(histories, DATES[-1])
    future_dates = pd.bdate_range(DATES[-1] + pd.Timedelta(days=1), periods=10)
    for symbol, frame in histories.items():
        future = pd.DataFrame({"trade_date": future_dates, "close": np.linspace(10.0, 1.0, 10)})
        histories[symbol] = pd.concat([frame, future], ignore_index=True)
    repeated = assess_market_regime(histories, DATES[-1])
    assert repeated == earlier
