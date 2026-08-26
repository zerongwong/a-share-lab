from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_lab.analytics.index_regime import IndexRegimeState, assess_index_regime

DATES = pd.bdate_range("2025-01-02", periods=180)


def _frame(close: np.ndarray, *, dates: pd.DatetimeIndex = DATES) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index_code": "fixture",
            "trade_date": dates,
            "close": close,
            "historical_backtest_eligible": True,
            "common_cutoff_date": dates[-1],
        }
    )


def _basket(close: np.ndarray) -> dict[str, pd.DataFrame]:
    return {
        code: _frame(close * scale)
        for code, scale in {
            "000001": 1.0,
            "000300": 1.2,
            "000905": 0.8,
        }.items()
    }


def test_orderly_core_index_uptrend_is_risk_on() -> None:
    close = np.linspace(100.0, 145.0, len(DATES))
    result = assess_index_regime(_basket(close), DATES[-1])

    assert result.state == IndexRegimeState.RISK_ON
    assert result.cutoff == DATES[-1]
    assert result.breadth_above_ma120 == 1.0
    assert result.median_return_60 is not None and result.median_return_60 > 0.0
    assert result.median_annualized_volatility_60 is not None
    assert result.index_metrics[0].index_code == "000001"


def test_broad_core_index_downtrend_is_risk_off() -> None:
    close = np.linspace(145.0, 90.0, len(DATES))
    result = assess_index_regime(_basket(close), DATES[-1])

    assert result.state == IndexRegimeState.RISK_OFF
    assert result.breadth_above_ma60 == 0.0
    assert result.median_max_drawdown_120 is not None
    assert result.median_max_drawdown_120 < 0.0


def test_mixed_core_indices_are_neutral() -> None:
    histories = {
        "000001": _frame(np.linspace(100.0, 135.0, len(DATES))),
        "000300": _frame(np.linspace(135.0, 100.0, len(DATES))),
        "000905": _frame(np.full(len(DATES), 110.0)),
    }

    result = assess_index_regime(histories, DATES[-1])

    assert result.state == IndexRegimeState.NEUTRAL
    assert result.score is not None


def test_future_rows_do_not_change_requested_common_cutoff() -> None:
    histories = _basket(np.linspace(100.0, 145.0, len(DATES)))
    earlier = assess_index_regime(histories, DATES[-1])
    future_dates = pd.bdate_range(DATES[-1] + pd.Timedelta(days=1), periods=5)
    for code, frame in histories.items():
        future = pd.DataFrame(
            {
                "index_code": code,
                "trade_date": future_dates,
                "close": np.linspace(50.0, 10.0, len(future_dates)),
                "historical_backtest_eligible": True,
                "common_cutoff_date": DATES[-1],
            }
        )
        histories[code] = pd.concat([frame, future], ignore_index=True)

    assert assess_index_regime(histories, DATES[-1]) == earlier


def test_missing_cutoff_observation_returns_unavailable() -> None:
    histories = _basket(np.linspace(100.0, 145.0, len(DATES)))
    histories["000300"] = histories["000300"].iloc[:-1]

    result = assess_index_regime(histories, DATES[-1])

    assert result.state == IndexRegimeState.UNAVAILABLE
    assert result.score is None
    assert "missing_common_cutoff_observation" in result.reason


def test_omitted_cutoff_does_not_hide_missing_declared_cutoff_row() -> None:
    histories = _basket(np.linspace(100.0, 145.0, len(DATES)))
    histories["000300"] = histories["000300"].iloc[:-1]

    result = assess_index_regime(histories)

    assert result.state == IndexRegimeState.UNAVAILABLE
    assert result.cutoff == DATES[-1]
    assert "missing_common_cutoff_observation" in result.reason


def test_recent_session_gap_returns_unavailable_instead_of_silent_alignment() -> None:
    histories = _basket(np.linspace(100.0, 145.0, len(DATES)))
    histories["000905"] = histories["000905"].drop(index=150).reset_index(drop=True)

    result = assess_index_regime(histories, DATES[-1])

    assert result.state == IndexRegimeState.UNAVAILABLE
    assert "recent_session_calendar_mismatch" in result.reason


def test_short_or_ineligible_history_returns_unavailable() -> None:
    short_dates = DATES[-100:]
    histories = {
        code: _frame(np.linspace(100.0, 120.0, len(short_dates)), dates=short_dates)
        for code in ("000001", "000300", "000905")
    }
    short = assess_index_regime(histories, short_dates[-1])
    assert short.state == IndexRegimeState.UNAVAILABLE
    assert "only_100_observations" in short.reason

    histories = _basket(np.linspace(100.0, 145.0, len(DATES)))
    histories["000300"].loc[:, "historical_backtest_eligible"] = False
    ineligible = assess_index_regime(histories, DATES[-1])
    assert ineligible.state == IndexRegimeState.UNAVAILABLE
    assert "not_historical_backtest_eligible" in ineligible.reason


def test_different_declared_common_cutoffs_return_unavailable() -> None:
    histories = _basket(np.linspace(100.0, 145.0, len(DATES)))
    histories["000905"].loc[:, "common_cutoff_date"] = DATES[-2]

    result = assess_index_regime(histories, DATES[-2])

    assert result.state == IndexRegimeState.UNAVAILABLE
    assert result.reason == "core_indices_have_different_common_cutoff_metadata"


def test_insufficient_core_index_count_returns_unavailable() -> None:
    histories = _basket(np.linspace(100.0, 145.0, len(DATES)))

    result = assess_index_regime({"000001": histories["000001"]}, DATES[-1])

    assert result.state == IndexRegimeState.UNAVAILABLE
    assert result.eligible_indices == 0
    assert result.required_indices == 3
