from __future__ import annotations

from datetime import UTC, date, datetime, time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from ashare_lab.adapters.baostock_eod import (
    BAOSTOCK_CORE_INDEX_PROVIDER_CODES,
    BAOSTOCK_CORE_INDEX_SYMBOLS,
    BAOSTOCK_INDEX_UNIT_CONTRACT_VERSION,
    BAOSTOCK_INDEX_UNIT_RESOLUTION_METHOD_VERSION,
    BaoStockEodMarketData,
)
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS

TARGET = date(2026, 9, 1)
NOW = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)  # 21:30 Asia/Shanghai
CN = ZoneInfo("Asia/Shanghai")
CALENDAR_FIELDS = ("calendar_date", "is_trading_day")
STOCK_FIELDS = ("code", "code_name", "ipoDate", "outDate", "type", "status")
INDEX_FIELDS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
)


class FakeResult:
    def __init__(
        self,
        fields: tuple[str, ...],
        rows: list[list[object]],
        *,
        error_code: str = "0",
    ) -> None:
        self.fields = list(fields)
        self.error_code = error_code
        self.error_msg = "success" if error_code == "0" else "fake failure"
        self._rows = rows
        self._position = 0

    def next(self) -> bool:
        if self._position >= len(self._rows):
            return False
        self._position += 1
        return True

    def get_row_data(self) -> list[object]:
        return self._rows[self._position - 1]


class FakeBaoStock:
    def __init__(
        self,
        *,
        calendar: FakeResult | None = None,
        stocks: FakeResult | None = None,
        index_rows: dict[str, FakeResult] | None = None,
        login_error: str = "0",
    ) -> None:
        self.calendar = calendar or _calendar_result((TARGET, "1"))
        self.stocks = stocks or _stock_result(
            [
                ["sh.600000", "浦发银行", "1999-11-10", "", "1", "1"],
                ["sz.000001", "平安银行", "1991-04-03", "", "1", "1"],
            ]
        )
        self.index_rows = index_rows or {
            provider: _index_result(provider, offset)
            for offset, provider in enumerate(BAOSTOCK_CORE_INDEX_PROVIDER_CODES.values())
        }
        self.login_error = login_error
        self.login_calls = 0
        self.logout_calls = 0
        self.calendar_calls: list[dict[str, str]] = []
        self.stock_calls = 0
        self.index_calls: list[dict[str, str]] = []

    def login(self) -> SimpleNamespace:
        self.login_calls += 1
        return SimpleNamespace(error_code=self.login_error, error_msg="fake")

    def logout(self) -> SimpleNamespace:
        self.logout_calls += 1
        return SimpleNamespace(error_code="0", error_msg="success")

    def query_trade_dates(self, **kwargs: str) -> FakeResult:
        self.calendar_calls.append(kwargs)
        return self.calendar

    def query_stock_basic(self) -> FakeResult:
        self.stock_calls += 1
        return self.stocks

    def query_history_k_data_plus(
        self,
        code: str,
        fields: str,
        **kwargs: str,
    ) -> FakeResult:
        self.index_calls.append({"code": code, "fields": fields, **kwargs})
        return self.index_rows[code]


def _calendar_result(*rows: tuple[date, str]) -> FakeResult:
    return FakeResult(
        CALENDAR_FIELDS,
        [[value.isoformat(), flag] for value, flag in rows],
    )


def _stock_result(rows: list[list[object]]) -> FakeResult:
    return FakeResult(STOCK_FIELDS, rows)


def _index_result(
    provider_code: str,
    offset: int = 0,
    *,
    row_override: dict[int, object] | None = None,
    fields: tuple[str, ...] = INDEX_FIELDS,
) -> FakeResult:
    base = 100 + offset
    row: list[object] = [
        TARGET.isoformat(),
        provider_code,
        f"{base}.00",
        f"{base + 2}.00",
        f"{base - 1}.00",
        f"{base + 1}.00",
        f"{base - 0.5:.2f}",
        "1000000",
        "100500000",
    ]
    for position, value in (row_override or {}).items():
        row[position] = value
    return FakeResult(fields, [row])


def _adapter(fake: FakeBaoStock, *, now: datetime = NOW) -> BaoStockEodMarketData:
    return BaoStockEodMarketData(module_loader=lambda: fake, clock=lambda: now)


def test_dependency_is_lazy_and_injectable() -> None:
    calls = 0

    def loader() -> object:
        nonlocal calls
        calls += 1
        raise ImportError("not installed")

    adapter = BaoStockEodMarketData(module_loader=loader)
    assert calls == 0

    with pytest.raises(DataUnavailableError, match="可选依赖"):
        adapter.fetch_cn_trading_days(TARGET, TARGET)
    assert calls == 1


def test_calendar_requires_complete_natural_day_range_and_returns_open_days() -> None:
    fake = FakeBaoStock(
        calendar=_calendar_result(
            (date(2026, 8, 29), "0"),
            (date(2026, 8, 30), "0"),
            (date(2026, 8, 31), "1"),
            (TARGET, "1"),
        )
    )

    result = _adapter(fake).fetch_cn_trading_days(date(2026, 8, 29), TARGET)

    assert result == (date(2026, 8, 31), TARGET)
    assert fake.calendar_calls == [
        {"start_date": "2026-08-29", "end_date": "2026-09-01"}
    ]
    assert fake.login_calls == 1
    assert fake.logout_calls == 1


@pytest.mark.parametrize(
    "calendar, match",
    [
        (_calendar_result((TARGET, "1"), (TARGET, "1")), "重复"),
        (_calendar_result((TARGET, "yes")), "无效交易日标记"),
        (
            FakeResult(("calendar_date",), [[TARGET.isoformat()]]),
            "字段与请求合同不一致",
        ),
    ],
)
def test_calendar_rejects_partial_duplicate_flag_or_schema(
    calendar: FakeResult,
    match: str,
) -> None:
    fake = FakeBaoStock(calendar=calendar)
    with pytest.raises(DataQualityError, match=match):
        _adapter(fake).fetch_cn_trading_days(TARGET, TARGET)


def test_calendar_rejects_incomplete_natural_day_coverage() -> None:
    fake = FakeBaoStock(calendar=_calendar_result((TARGET, "1")))
    with pytest.raises(DataQualityError, match="完整覆盖"):
        _adapter(fake).fetch_cn_trading_days(date(2026, 8, 31), TARGET)


def test_calendar_rejects_reversed_range_before_loading_provider() -> None:
    calls = 0

    def loader() -> object:
        nonlocal calls
        calls += 1
        return FakeBaoStock()

    adapter = BaoStockEodMarketData(module_loader=loader)
    with pytest.raises(ValueError, match="start"):
        adapter.fetch_cn_trading_days(TARGET, date(2026, 8, 31))
    assert calls == 0


def test_current_symbol_list_keeps_suspended_a_shares_and_excludes_other_products() -> None:
    fake = FakeBaoStock(
        stocks=_stock_result(
            [
                ["sh.600000", "浦发银行", "1999-11-10", "", "1", "1"],
                ["sh.688001", "科创测试", "2020-01-01", "", "1", "1"],
                ["sz.000001", "平安银行", "1991-04-03", "", "1", "1"],
                # Status is listing status, not today's trade status.  A listed
                # suspended name therefore remains in this universe.
                ["sz.300001", "创业测试", "2010-01-01", "", "1", "1"],
                ["sh.900901", "沪B", "1992-01-01", "", "1", "1"],
                ["sz.200002", "深B", "1992-01-01", "", "1", "1"],
                ["bj.430047", "北交测试", "2021-01-01", "", "1", "1"],
                ["sh.000001", "上证指数", "1991-01-01", "", "2", "1"],
                ["sh.510300", "沪深300ETF", "2012-01-01", "", "5", "1"],
                ["sh.600001", "已退市", "1999-01-01", "2020-01-01", "1", "0"],
            ]
        )
    )

    symbols = _adapter(fake).fetch_cn_stock_symbols()

    assert symbols == ("000001.SZ", "300001.SZ", "600000.SH", "688001.SH")
    assert fake.stock_calls == 1
    assert fake.login_calls == fake.logout_calls == 1


@pytest.mark.parametrize(
    "rows, match",
    [
        (
            [
                ["sh.600000", "A", "1999-01-01", "", "1", "1"],
                ["sh.600000", "A", "1999-01-01", "", "1", "1"],
            ],
            "重复",
        ),
        ([ ["xx.600000", "A", "1999-01-01", "", "1", "1"] ], "无法识别"),
        ([ ["sh.500001", "A", "1999-01-01", "", "1", "1"] ], "新前缀"),
        ([ ["sh.600000", "A", "1999-01-01", "", "1", "active"] ], "type或status"),
    ],
)
def test_symbol_list_rejects_duplicate_unknown_code_or_status(
    rows: list[list[object]],
    match: str,
) -> None:
    with pytest.raises(DataQualityError, match=match):
        _adapter(FakeBaoStock(stocks=_stock_result(rows))).fetch_cn_stock_symbols()


def test_core_indices_are_complete_unadjusted_and_keep_documented_units() -> None:
    fake = FakeBaoStock()

    batch = _adapter(fake).fetch_core_index_daily(TARGET)

    assert batch.provider == "baostock"
    assert batch.target_date == TARGET
    assert batch.requested_symbols == BAOSTOCK_CORE_INDEX_SYMBOLS
    assert batch.received_symbols == BAOSTOCK_CORE_INDEX_SYMBOLS
    assert batch.coverage_ratio == 1.0
    assert batch.fetched_at == NOW
    assert batch.unit_contract_version == BAOSTOCK_INDEX_UNIT_CONTRACT_VERSION
    assert (
        batch.unit_resolution_method_version
        == BAOSTOCK_INDEX_UNIT_RESOLUTION_METHOD_VERSION
    )
    assert batch.amount_multiplier_to_cny == "1"
    assert batch.trace_ids == ()
    assert list(batch.frame.columns) == ["symbol", *CANONICAL_DAILY_COLUMNS]
    assert batch.frame["symbol"].tolist() == sorted(
        symbol[:6] for symbol in BAOSTOCK_CORE_INDEX_SYMBOLS
    )
    assert set(batch.frame["trade_date"]) == {TARGET}
    assert set(batch.frame["source"]) == {"baostock:eod_unadjusted:indices"}
    assert set(batch.frame["retrieved_at"]) == {"2026-09-01T13:30:00Z"}
    assert (batch.frame["volume_shares"] == 1_000_000).all()
    assert (batch.frame["amount_cny"] == 100_500_000).all()
    assert batch.frame["turnover_pct"].isna().all()
    assert all(isinstance(value, date) for value in batch.frame["trade_date"])

    assert fake.login_calls == fake.logout_calls == 1
    assert len(fake.calendar_calls) == 1
    assert len(fake.index_calls) == 6
    assert [call["code"] for call in fake.index_calls] == list(
        BAOSTOCK_CORE_INDEX_PROVIDER_CODES.values()
    )
    assert all(call["fields"] == ",".join(INDEX_FIELDS) for call in fake.index_calls)
    assert all(call["start_date"] == TARGET.isoformat() for call in fake.index_calls)
    assert all(call["end_date"] == TARGET.isoformat() for call in fake.index_calls)
    assert all(call["frequency"] == "d" for call in fake.index_calls)
    assert all(call["adjustflag"] == "3" for call in fake.index_calls)


@pytest.mark.parametrize(
    "row_override, match",
    [
        ({0: "2026-08-31"}, "日期"),
        ({1: "sh.000016"}, "证券代码"),
        ({3: "98.00"}, "开高低收"),
        ({7: "0"}, "volume"),
        ({8: "NaN"}, "amount"),
    ],
)
def test_core_index_rejects_wrong_identity_ohlc_or_units(
    row_override: dict[int, object],
    match: str,
) -> None:
    first = next(iter(BAOSTOCK_CORE_INDEX_PROVIDER_CODES.values()))
    fake = FakeBaoStock()
    fake.index_rows[first] = _index_result(first, row_override=row_override)

    with pytest.raises(DataQualityError, match=match):
        _adapter(fake).fetch_core_index_daily(TARGET)


def test_core_index_rejects_missing_field_or_missing_index() -> None:
    first = next(iter(BAOSTOCK_CORE_INDEX_PROVIDER_CODES.values()))
    missing_field = FakeBaoStock()
    missing_field.index_rows[first] = _index_result(
        first,
        fields=INDEX_FIELDS[:-1],
    )
    with pytest.raises(DataQualityError, match="字段与请求合同不一致"):
        _adapter(missing_field).fetch_core_index_daily(TARGET)

    missing_row = FakeBaoStock()
    missing_row.index_rows[first] = FakeResult(INDEX_FIELDS, [])
    with pytest.raises(DataUnavailableError, match="没有返回"):
        _adapter(missing_row).fetch_core_index_daily(TARGET)


def test_core_index_rejects_non_session_and_intraday_before_provider_queries() -> None:
    closed = FakeBaoStock(calendar=_calendar_result((TARGET, "0")))
    with pytest.raises(DataQualityError, match="未确认"):
        _adapter(closed).fetch_core_index_daily(TARGET)
    assert closed.index_calls == []

    intraday_now = datetime.combine(TARGET, time(14, 59), tzinfo=CN)
    intraday = FakeBaoStock()
    with pytest.raises(DataQualityError, match="尚未收盘"):
        _adapter(intraday, now=intraday_now).fetch_core_index_daily(TARGET)
    assert intraday.login_calls == 0


def test_login_failure_is_normalized_and_does_not_query() -> None:
    fake = FakeBaoStock(login_error="10001011")
    with pytest.raises(DataUnavailableError, match="登录返回失败"):
        _adapter(fake).fetch_cn_trading_days(TARGET, TARGET)
    assert fake.calendar_calls == []
    assert fake.logout_calls == 0


def test_core_index_cutoff_must_be_same_day_after_close() -> None:
    before_close = int(datetime.combine(TARGET, time(14, 59), tzinfo=CN).timestamp())
    with pytest.raises(ValueError, match="at or after CN close"):
        _adapter(FakeBaoStock()).fetch_core_index_daily(
            TARGET,
            cutoff_timestamp=before_close,
        )


def test_result_iterator_shape_is_strict() -> None:
    class BadIteratorResult(FakeResult):
        def next(self) -> object:
            return "yes"

    fake = FakeBaoStock(
        calendar=BadIteratorResult(CALENDAR_FIELDS, [[TARGET.isoformat(), "1"]])
    )
    with pytest.raises(DataQualityError, match="不是布尔值"):
        _adapter(fake).fetch_cn_trading_days(TARGET, TARGET)


def test_frame_contains_only_finite_numeric_core_values() -> None:
    frame = _adapter(FakeBaoStock()).fetch_core_index_daily(TARGET).frame
    numeric = frame[["open", "high", "low", "close", "prev_close", "volume_shares", "amount_cny"]]
    assert numeric.notna().all().all()
