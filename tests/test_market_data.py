from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ashare_lab.adapters.akshare_market import AKShareMarketData
from ashare_lab.adapters.csv_market import CSVMarketData
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS, MarketDataPort

FIXED_NOW = datetime(2026, 8, 24, 5, 0, tzinfo=UTC)


def _akshare_frame(*, include_future: bool = False) -> pd.DataFrame:
    rows = {
        "日期": ["2026-08-19", "2026-08-20", "2026-08-21"],
        "开盘": [33.40, 33.20, 32.83],
        "收盘": [33.10, 32.99, 33.68],
        "最高": [33.70, 33.50, 33.79],
        "最低": [32.90, 32.80, 32.59],
        "成交量": [500_000, 600_000, 760_207],  # lots (手)
        "成交额": [1.66e9, 1.98e9, 2.54e9],
        "换手率": [1.12, 1.31, 1.69],
    }
    if include_future:
        rows["日期"].append("2026-08-24")
        rows["开盘"].append(99.0)
        rows["收盘"].append(99.0)
        rows["最高"].append(99.0)
        rows["最低"].append(99.0)
        rows["成交量"].append(1)
        rows["成交额"].append(99.0)
        rows["换手率"].append(0.01)
    return pd.DataFrame(rows)


def test_akshare_is_lazy_and_satisfies_market_data_port(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetcher(**kwargs):
        calls.append(kwargs["symbol"])
        return _akshare_frame()

    def loader():
        calls.append("import")
        return SimpleNamespace(stock_zh_a_hist=fetcher)

    adapter = AKShareMarketData(
        tmp_path,
        module_loader=loader,
        clock=lambda: FIXED_NOW,
    )
    assert isinstance(adapter, MarketDataPort)
    assert calls == []  # constructing the adapter does not import AKShare

    result = adapter.fetch_daily("600150.SH", date(2026, 8, 20), date(2026, 8, 21), adjust="qfq")

    assert calls == ["import", "600150"]
    assert tuple(result.columns) == CANONICAL_DAILY_COLUMNS
    assert result["trade_date"].dt.date.tolist() == [date(2026, 8, 20), date(2026, 8, 21)]
    assert result.loc[1, "volume_shares"] == 76_020_700
    assert result.loc[0, "prev_close"] == pytest.approx(33.10)
    assert set(result["source"]) == {"akshare"}
    assert result.attrs["data_quality"] == "live"
    assert result.attrs["is_cache_fallback"] is False


def test_akshare_filters_any_rows_after_as_of(tmp_path: Path) -> None:
    fake = SimpleNamespace(stock_zh_a_hist=lambda **_: _akshare_frame(include_future=True))
    adapter = AKShareMarketData(
        tmp_path,
        module_loader=lambda: fake,
        clock=lambda: FIXED_NOW,
    )
    result = adapter.fetch_daily("600150", date(2026, 8, 19), date(2026, 8, 21), adjust="none")

    assert result["trade_date"].max().date() == date(2026, 8, 21)
    assert 99.0 not in result["close"].tolist()
    assert result.attrs["lookahead_rows_dropped"] == 1


def test_live_failure_uses_only_explicitly_marked_same_source_cache(tmp_path: Path) -> None:
    live = AKShareMarketData(
        tmp_path,
        module_loader=lambda: SimpleNamespace(stock_zh_a_hist=lambda **_: _akshare_frame()),
        clock=lambda: FIXED_NOW,
    )
    live.fetch_daily("600150", date(2026, 8, 19), date(2026, 8, 21))

    def failing_loader():
        raise RuntimeError("network unavailable")

    fallback = AKShareMarketData(
        tmp_path,
        module_loader=failing_loader,
        clock=lambda: FIXED_NOW,
    )
    result = fallback.fetch_daily("600150", date(2026, 8, 20), date(2026, 8, 21))

    assert set(result["source"]) == {"akshare:cache"}
    assert result.attrs["data_quality"] == "cached"
    assert result.attrs["is_cache_fallback"] is True
    assert result.attrs["warning"].startswith("LIVE_FETCH_FAILED_USING_EXPLICIT_CACHE")
    # A canonical cache already stores shares; it must not be multiplied again.
    assert result.loc[1, "volume_shares"] == 76_020_700


def test_live_failure_without_cache_is_loud(tmp_path: Path) -> None:
    adapter = AKShareMarketData(
        tmp_path,
        module_loader=lambda: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    with pytest.raises(DataUnavailableError, match="没有同源缓存"):
        adapter.fetch_daily("600150", date(2026, 8, 20), date(2026, 8, 21))


def test_cache_fallback_can_be_disabled(tmp_path: Path) -> None:
    adapter = AKShareMarketData(
        tmp_path,
        allow_cache_fallback=False,
        module_loader=lambda: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    with pytest.raises(DataUnavailableError, match="缓存降级已关闭"):
        adapter.fetch_daily("600150", date(2026, 8, 20), date(2026, 8, 21))


def test_csv_offline_adapter_filters_symbol_date_and_converts_lots(tmp_path: Path) -> None:
    csv_path = tmp_path / "daily.csv"
    frame = _akshare_frame(include_future=True)
    frame["代码"] = ["600150", "600150", "600150", "600150"]
    frame.to_csv(csv_path, index=False)

    adapter = CSVMarketData(
        csv_path,
        adjustment="none",
        volume_unit="lots",
        clock=lambda: FIXED_NOW,
    )
    assert isinstance(adapter, MarketDataPort)
    result = adapter.fetch_daily("600150.SH", date(2026, 8, 20), date(2026, 8, 21), adjust="none")

    assert tuple(result.columns) == CANONICAL_DAILY_COLUMNS
    assert result["trade_date"].max().date() == date(2026, 8, 21)
    assert result.loc[1, "volume_shares"] == 76_020_700
    assert set(result["source"]) == {"csv:offline"}
    assert result.attrs["data_quality"] == "offline"
    assert "OFFLINE_CSV_DATA" in result.attrs["warning"]


def test_csv_does_not_silently_change_adjustment_or_symbol(tmp_path: Path) -> None:
    csv_path = tmp_path / "daily.csv"
    frame = _akshare_frame()
    frame["代码"] = "600150"
    frame.to_csv(csv_path, index=False)
    adapter = CSVMarketData(csv_path, adjustment="qfq", volume_unit="lots")

    with pytest.raises(DataUnavailableError, match="复权口径不匹配"):
        adapter.fetch_daily("600150", date(2026, 8, 20), date(2026, 8, 21), adjust="none")
    with pytest.raises(DataUnavailableError, match="没有股票"):
        adapter.fetch_daily("000001", date(2026, 8, 20), date(2026, 8, 21), adjust="qfq")


def test_bad_price_geometry_is_rejected(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    frame = _akshare_frame()
    frame.loc[1, "最高"] = 1.0
    frame.to_csv(csv_path, index=False)

    adapter = CSVMarketData(csv_path, adjustment="qfq", volume_unit="lots")
    with pytest.raises(DataQualityError, match="最高价"):
        adapter.fetch_daily("600150", date(2026, 8, 19), date(2026, 8, 21))
