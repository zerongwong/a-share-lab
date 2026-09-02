from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from ashare_lab.adapters.tushare_daily import (
    TUSHARE_DAILY_ROW_LIMIT,
    TUSHARE_DAILY_SOURCE,
    TushareDailyClient,
    normalize_tushare_daily,
)
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS

TARGET = date(2026, 9, 1)
SHANGHAI = ZoneInfo("Asia/Shanghai")
RETRIEVED = datetime(2026, 9, 1, 21, 5, tzinfo=SHANGHAI)


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["600000.SH", "000001.SZ", "830799.BJ"],
            "trade_date": ["20260901", "20260901", "20260901"],
            "open": [10.2, 20.5, 5.1],
            "high": [11.0, 22.0, 5.5],
            "low": [10.0, 20.0, 5.0],
            "close": [10.5, 21.0, 5.4],
            "pre_close": [10.1, 20.4, 5.0],
            "change": [0.4, 0.6, 0.4],
            "pct_chg": [3.96, 2.94, 8.0],
            "vol": [1_000.0, 2_000.5, 100.0],  # 手
            "amount": [1_050.0, 4_201.05, 52.0],  # 千元
        }
    )


class _Client:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[dict[str, str]] = []

    def daily(self, **kwargs: str) -> pd.DataFrame:
        self.calls.append(kwargs)
        return self.frame.copy()


def test_fetches_full_trade_date_once_and_strictly_normalizes_units() -> None:
    client = _Client(_raw())
    adapter = TushareDailyClient(client=client, clock=lambda: RETRIEVED)

    result = adapter.fetch_daily(TARGET)
    frame = result.frame

    assert client.calls == [{"trade_date": "20260901"}]
    assert list(frame.columns) == ["symbol", *CANONICAL_DAILY_COLUMNS]
    assert frame["symbol"].tolist() == ["000001", "600000", "830799"]
    assert set(frame["source"]) == {TUSHARE_DAILY_SOURCE}
    assert set(frame["retrieved_at"]) == {"2026-09-01T13:05:00Z"}
    assert frame["trade_date"].dt.date.tolist() == [TARGET, TARGET, TARGET]
    by_symbol = frame.set_index("symbol")
    assert by_symbol.loc["600000", "prev_close"] == pytest.approx(10.1)
    assert by_symbol.loc["600000", "volume_shares"] == 100_000
    assert by_symbol.loc["000001", "volume_shares"] == 200_050
    assert by_symbol.loc["000001", "amount_cny"] == pytest.approx(4_201_050.0)
    assert frame["turnover_pct"].isna().all()
    assert frame.attrs["adjustment"] == "none"
    assert frame.attrs["unit_contract"] == "vol=100_shares;amount=1000_cny"
    assert result.requested_trade_date == TARGET
    assert result.received_trade_dates == (TARGET,)
    assert result.received_symbols == ("000001", "600000", "830799")
    assert result.fetched_at.isoformat() == "2026-09-01T13:05:00+00:00"
    assert result.trace_ids == ()
    assert not hasattr(result, "requested_symbols")


def test_token_is_only_passed_to_factory_and_not_retained() -> None:
    seen: list[str] = []
    client = _Client(_raw())

    def factory(token: str) -> object:
        seen.append(token)
        return client

    adapter = TushareDailyClient(
        "private-token",
        client_factory=factory,
        clock=lambda: RETRIEVED,
    )

    assert seen == ["private-token"]
    assert not hasattr(adapter, "token")
    assert not hasattr(adapter, "_token")
    assert len(adapter.fetch_daily(TARGET).frame) == 3


def test_rejects_provider_row_limit_before_accepting_a_possibly_truncated_day() -> None:
    raw = pd.concat([_raw().iloc[[0]]] * TUSHARE_DAILY_ROW_LIMIT, ignore_index=True)
    client = _Client(raw)

    with pytest.raises(DataQualityError, match="达到 6000 行接口上限"):
        TushareDailyClient(client=client, clock=lambda: RETRIEVED).fetch_daily(TARGET)

    assert client.calls == [{"trade_date": "20260901"}]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda frame: frame.drop(columns="pre_close"), "缺少字段"),
        (lambda frame: frame.assign(ts_code=["bad", "000001.SZ", "830799.BJ"]), "无效 A 股代码"),
        (lambda frame: frame.assign(trade_date="20260902"), "目标交易日以外"),
        (lambda frame: frame.assign(pre_close=0), "pre_close 必须为正数"),
        (lambda frame: frame.assign(change=9.9), "昨收与涨跌额"),
        (lambda frame: frame.assign(pct_chg=99.0), "昨收与涨跌幅"),
        (lambda frame: frame.assign(high=1), "最高价"),
        (lambda frame: frame.assign(vol=[1.234, 2_000.5, 100]), "不是整数股"),
        (lambda frame: frame.assign(amount=1), "均价超出"),
    ],
)
def test_rejects_ambiguous_or_invalid_daily_rows(mutate, match: str) -> None:
    with pytest.raises(DataQualityError, match=match):
        normalize_tushare_daily(mutate(_raw()), target_date=TARGET, retrieved_at=RETRIEVED)


def test_rejects_duplicate_symbols_and_naive_retrieval_time() -> None:
    duplicate = pd.concat([_raw(), _raw().iloc[[0]]], ignore_index=True)
    with pytest.raises(DataQualityError, match="重复证券代码"):
        normalize_tushare_daily(duplicate, target_date=TARGET, retrieved_at=RETRIEVED)

    with pytest.raises(DataQualityError, match="带时区"):
        normalize_tushare_daily(
            _raw(),
            target_date=TARGET,
            retrieved_at=datetime(2026, 9, 1, 21, 5),
        )


def test_provider_failures_are_redacted() -> None:
    secret = "token-should-never-escape"
    raw_message = "upstream request rejected"

    def fail(**_: str) -> pd.DataFrame:
        raise RuntimeError(f"{raw_message}; token={secret}")

    adapter = TushareDailyClient(
        client=SimpleNamespace(daily=fail),
        clock=lambda: RETRIEVED,
    )
    with pytest.raises(DataUnavailableError) as captured:
        adapter.fetch_daily(TARGET)

    assert secret not in str(captured.value)
    assert raw_message not in str(captured.value)
    assert captured.value.__suppress_context__ is True


def test_empty_response_is_unavailable_not_a_valid_zero_stock_session() -> None:
    client = _Client(pd.DataFrame())
    with pytest.raises(DataUnavailableError, match="没有返回日线数据"):
        TushareDailyClient(client=client, clock=lambda: RETRIEVED).fetch_daily(TARGET)
