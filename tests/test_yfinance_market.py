from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from ashare_lab.adapters.yfinance_market import YFinanceMarketData, yahoo_symbol


def test_yahoo_symbol_maps_a_share_exchanges_explicitly():
    assert yahoo_symbol("600150") == "600150.SS"
    assert yahoo_symbol("000001") == "000001.SZ"
    assert yahoo_symbol("430047") == "430047.BJ"


class FakeYahoo:
    @staticmethod
    def download(*args, **kwargs):
        index = pd.to_datetime(["2026-08-21", "2026-08-24"])
        columns = pd.MultiIndex.from_tuples(
            [
                (name, "600150.SS")
                for name in ("Open", "High", "Low", "Close", "Adj Close", "Volume")
            ]
        )
        return pd.DataFrame(
            [[10, 11, 9, 10.5, 10.5, 1000], [11, 12, 10, 11.5, 11.5, 1200]],
            index=index,
            columns=columns,
        ).rename_axis("Date")


class FailingYahoo:
    @staticmethod
    def download(*args, **kwargs):
        raise RuntimeError("rate limited")


def test_current_intraday_bar_is_not_labelled_as_completed_close():
    provider = YFinanceMarketData(
        module_loader=lambda: FakeYahoo,
        cache_dir=None,
        clock=lambda: datetime(2026, 8, 24, 5, 40, tzinfo=UTC),  # 13:40 Shanghai
    )
    result = provider.fetch_daily("600150", date(2026, 8, 1), date(2026, 8, 24))
    assert result.iloc[-1]["trade_date"].date() == date(2026, 8, 21)
    assert result.attrs["current_session_excluded"] is True


def test_yahoo_cache_round_trip_allows_missing_optional_amount_and_turnover(
    tmp_path: Path,
):
    def clock() -> datetime:
        return datetime(2026, 8, 24, 8, 0, tzinfo=UTC)

    live = YFinanceMarketData(
        module_loader=lambda: FakeYahoo,
        cache_dir=tmp_path,
        clock=clock,
    )
    live.fetch_daily("600150", date(2026, 8, 1), date(2026, 8, 21))

    fallback = YFinanceMarketData(
        module_loader=lambda: FailingYahoo,
        cache_dir=tmp_path,
        clock=clock,
    )
    result = fallback.fetch_daily("600150", date(2026, 8, 1), date(2026, 8, 21))

    assert len(result) == 1
    assert result.iloc[-1]["trade_date"].date() == date(2026, 8, 21)
    assert result["amount_cny"].isna().all()
    assert result["turnover_pct"].isna().all()
    assert result.attrs["data_quality"] == "cached"
