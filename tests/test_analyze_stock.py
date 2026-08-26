from datetime import date

import numpy as np
import pandas as pd

from ashare_lab.services.analyze_stock import analyze_stock


class FakeProvider:
    def fetch_daily(self, symbol, start, end, *, adjust="qfq"):
        rows = 520
        trend = np.linspace(10, 22, rows)
        wave = np.sin(np.linspace(0, 24, rows))
        close = trend + wave
        frame = pd.DataFrame(
            {
                "trade_date": pd.date_range(end=end, periods=rows, freq="B"),
                "open": close - 0.1,
                "high": close + 0.35,
                "low": close - 0.35,
                "close": close,
                "prev_close": pd.Series(close).shift(1),
                "volume_shares": 2_000_000 + np.arange(rows) * 100,
                "amount_cny": 30_000_000,
                "turnover_pct": 1.0,
                "source": "fixture",
                "retrieved_at": "2026-08-24T12:00:00+08:00",
            }
        )
        frame.attrs.update(source="fixture", retrieved_at="2026-08-24T12:00:00+08:00")
        return frame


def test_analysis_has_five_horizons_and_no_future_bar():
    result = analyze_stock(FakeProvider(), "600150.SS", date(2026, 8, 21))
    assert result["symbol"] == "600150"
    assert result["data_cutoff"] <= "2026-08-21"
    assert [row["sessions"] for row in result["levels"]] == [5, 20, 60, 120, 252]
    assert all(row["invalidation"] < row["reference_price"] for row in result["levels"])
    assert "加码" in result["livermore"]["note"]
