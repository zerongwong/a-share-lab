from __future__ import annotations

from datetime import UTC, date, datetime, time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from ashare_lab.adapters.akshare_eod_verifier import AKShareVerificationStatus
from ashare_lab.adapters.baostock_eod import BAOSTOCK_CORE_INDEX_SYMBOLS
from ashare_lab.adapters.market_overlay_store import MarketOverlayStore
from ashare_lab.adapters.tushare_daily import TushareDailyFetch
from ashare_lab.adapters.zero_budget_eod import (
    ZERO_BUDGET_AMOUNT_MULTIPLIER_TO_CNY,
    ZERO_BUDGET_EOD_PROVIDER,
    ZERO_BUDGET_INDEX_SOURCE,
    ZERO_BUDGET_STOCK_SOURCE,
    ZERO_BUDGET_UNIT_CONTRACT_VERSION,
    ZERO_BUDGET_UNIT_RESOLUTION_METHOD_VERSION,
    ZeroBudgetEodMarketData,
)
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.daily_increment import DailyIncrementBatch, DailyIncrementPort
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS
from ashare_lab.services.sync_daily_overlay import sync_daily_overlay_range

TARGET = date(2026, 9, 1)
BASELINE = date(2026, 8, 31)
NOW = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)
CN = ZoneInfo("Asia/Shanghai")
STOCKS = ("000001.SZ", "600000.SH")


def _cutoff() -> int:
    return int(datetime.combine(TARGET, time(23, 59, 59), tzinfo=CN).timestamp())


def _stock_values(symbol: str) -> dict[str, float]:
    offset = float(int(symbol) % 100)
    low = 10.0 + offset
    volume = 100_000.0
    return {
        "open": low + 0.2,
        "high": low + 1.0,
        "low": low,
        "close": low + 0.5,
        "prev_close": low + 0.1,
        "volume_shares": volume,
        "amount_cny": (low + 0.5) * volume,
    }


def _stock_frame(
    symbols: tuple[str, ...] = ("000001", "600000", "830799"),
    *,
    trade_date: date = TARGET,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": trade_date,
                **_stock_values(symbol),
                "turnover_pct": float("nan"),
                "source": "tushare:daily_unadjusted:stocks",
                "retrieved_at": NOW.isoformat().replace("+00:00", "Z"),
            }
            for symbol in symbols
        ],
        columns=["symbol", *CANONICAL_DAILY_COLUMNS],
    )


def _index_frame(*, trade_date: date = TARGET, source: str = "baostock:eod_unadjusted:indices") -> pd.DataFrame:
    rows = []
    for offset, external in enumerate(BAOSTOCK_CORE_INDEX_SYMBOLS):
        low = 100.0 + offset
        rows.append(
            {
                "symbol": external[:6],
                "trade_date": trade_date,
                "open": low + 0.2,
                "high": low + 1.0,
                "low": low,
                "close": low + 0.5,
                "prev_close": low + 0.1,
                "volume_shares": 1_000_000.0,
                "amount_cny": (low + 0.5) * 1_000_000.0,
                "turnover_pct": float("nan"),
                "source": source,
                "retrieved_at": NOW.isoformat().replace("+00:00", "Z"),
            }
        )
    return pd.DataFrame(rows, columns=["symbol", *CANONICAL_DAILY_COLUMNS])


def _index_batch(**frame_options: object) -> DailyIncrementBatch:
    return DailyIncrementBatch(
        frame=_index_frame(**frame_options),
        target_date=TARGET,
        requested_symbols=BAOSTOCK_CORE_INDEX_SYMBOLS,
        received_symbols=BAOSTOCK_CORE_INDEX_SYMBOLS,
        fetched_at=NOW,
        trace_ids=("index-trace",),
        provider="baostock",
        cutoff_timestamp=_cutoff(),
    )


class FakeBaoStock:
    def __init__(
        self,
        *,
        index_batch: DailyIncrementBatch | None = None,
        symbols: tuple[str, ...] = STOCKS,
    ) -> None:
        self.index_batch = index_batch or _index_batch()
        self.symbols = symbols
        self.calendar_calls: list[tuple[date, date]] = []
        self.symbol_calls = 0
        self.index_calls: list[tuple[date, int | None]] = []

    def fetch_cn_trading_days(self, start: date, end: date) -> tuple[date, ...]:
        self.calendar_calls.append((start, end))
        return tuple(value for value in (TARGET,) if start <= value <= end)

    def fetch_cn_stock_symbols(self) -> tuple[str, ...]:
        self.symbol_calls += 1
        return self.symbols

    def fetch_core_index_daily(
        self,
        target_date: date,
        *,
        cutoff_timestamp: int | None = None,
    ) -> DailyIncrementBatch:
        self.index_calls.append((target_date, cutoff_timestamp))
        return self.index_batch


class FakeTushare:
    def __init__(self, fetch: TushareDailyFetch | None = None) -> None:
        frame = _stock_frame()
        self.fetch = fetch or TushareDailyFetch(
            frame=frame,
            requested_trade_date=TARGET,
            received_trade_dates=(TARGET,),
            received_symbols=tuple(frame["symbol"]),
            fetched_at=NOW,
            trace_ids=("stock-trace",),
        )
        self.calls: list[date] = []

    def fetch_daily(self, trade_date: date) -> TushareDailyFetch:
        self.calls.append(trade_date)
        return self.fetch


class FakeVerifier:
    def __init__(
        self,
        status: AKShareVerificationStatus = AKShareVerificationStatus.VERIFIED,
    ) -> None:
        self.status = status
        self.calls: list[tuple[pd.DataFrame, date, tuple[str, ...]]] = []

    def verify_stock_frame(
        self,
        frame: pd.DataFrame,
        target_date: date,
        requested_symbols: tuple[str, ...],
    ) -> object:
        self.calls.append((frame.copy(), target_date, requested_symbols))
        return SimpleNamespace(status=self.status)


def _adapter(
    *,
    baostock: object | None = None,
    tushare: object | None = None,
    verifier: object | None = None,
) -> ZeroBudgetEodMarketData:
    return ZeroBudgetEodMarketData(
        baostock=baostock or FakeBaoStock(),
        tushare=tushare or FakeTushare(),
        verifier=verifier or FakeVerifier(),
    )


def test_delegates_calendar_and_list_and_satisfies_daily_increment_port() -> None:
    bao = FakeBaoStock()
    adapter = _adapter(baostock=bao)

    assert isinstance(adapter, DailyIncrementPort)
    assert adapter.provider == ZERO_BUDGET_EOD_PROVIDER
    assert adapter.fetch_cn_trading_days(BASELINE, TARGET) == (TARGET,)
    assert adapter.fetch_cn_stock_symbols() == STOCKS
    assert bao.calendar_calls == [(BASELINE, TARGET)]
    assert bao.symbol_calls == 1


def test_stock_batch_filters_extras_and_uses_akshare_verification() -> None:
    tushare = FakeTushare()
    verifier = FakeVerifier()
    batch = _adapter(tushare=tushare, verifier=verifier).fetch_daily_increment(
        STOCKS,
        TARGET,
        asset_kind="stocks",
    )

    assert tushare.calls == [TARGET]
    assert batch.requested_symbols == STOCKS
    assert batch.received_symbols == STOCKS
    assert batch.provider == ZERO_BUDGET_EOD_PROVIDER
    assert batch.cutoff_timestamp == _cutoff()
    assert batch.trace_ids == ("tushare:stock-trace",)
    assert batch.frame["symbol"].tolist() == ["000001", "600000"]
    assert set(batch.frame["source"]) == {ZERO_BUDGET_STOCK_SOURCE}
    assert len(verifier.calls) == 1
    verified_frame, verified_date, verified_symbols = verifier.calls[0]
    assert verified_frame["symbol"].tolist() == ["000001", "600000"]
    assert set(verified_frame["source"]) == {"tushare:daily_unadjusted:stocks"}
    assert verified_date == TARGET
    assert verified_symbols == ("000001", "600000")


def test_index_batch_is_baostock_only_and_uses_identical_composite_unit_audit() -> None:
    bao = FakeBaoStock()
    adapter = _adapter(baostock=bao)
    stocks = adapter.fetch_daily_increment(STOCKS, TARGET, asset_kind="stocks")
    indices = adapter.fetch_daily_increment(
        BAOSTOCK_CORE_INDEX_SYMBOLS,
        TARGET,
        asset_kind="indices",
    )

    assert bao.index_calls == [(TARGET, _cutoff())]
    assert indices.requested_symbols == BAOSTOCK_CORE_INDEX_SYMBOLS
    assert indices.received_symbols == BAOSTOCK_CORE_INDEX_SYMBOLS
    assert indices.trace_ids == ("baostock:index-trace",)
    assert set(indices.frame["source"]) == {ZERO_BUDGET_INDEX_SOURCE}
    for batch in (stocks, indices):
        assert batch.unit_contract_version == ZERO_BUDGET_UNIT_CONTRACT_VERSION
        assert (
            batch.unit_resolution_method_version
            == ZERO_BUDGET_UNIT_RESOLUTION_METHOD_VERSION
        )
        assert batch.amount_multiplier_to_cny == ZERO_BUDGET_AMOUNT_MULTIPLIER_TO_CNY


def test_suspended_stock_gap_is_preserved_for_outer_coverage_gate() -> None:
    missing_frame = _stock_frame(("000001",))
    missing = TushareDailyFetch(
        frame=missing_frame,
        requested_trade_date=TARGET,
        received_trade_dates=(TARGET,),
        received_symbols=("000001",),
        fetched_at=NOW,
        trace_ids=(),
    )
    verifier = FakeVerifier()
    batch = _adapter(tushare=FakeTushare(missing), verifier=verifier).fetch_daily_increment(
        STOCKS,
        TARGET,
    )

    assert batch.requested_symbols == STOCKS
    assert batch.received_symbols == ("000001.SZ",)
    assert batch.frame["symbol"].tolist() == ["000001"]
    assert batch.coverage_ratio == 0.5
    assert verifier.calls[0][2] == ("000001",)


def test_duplicate_empty_or_wrong_date_tushare_rows_are_rejected_before_publish() -> None:

    duplicate_frame = pd.concat([_stock_frame(("000001",)), _stock_frame(("000001",))])
    duplicate = TushareDailyFetch(
        frame=duplicate_frame,
        requested_trade_date=TARGET,
        received_trade_dates=(TARGET,),
        received_symbols=("000001", "000001"),
        fetched_at=NOW,
        trace_ids=(),
    )
    with pytest.raises(DataQualityError, match="重复"):
        _adapter(tushare=FakeTushare(duplicate)).fetch_daily_increment(("000001.SZ",), TARGET)

    empty_frame = pd.DataFrame(columns=["symbol", *CANONICAL_DAILY_COLUMNS])
    empty = TushareDailyFetch(
        frame=empty_frame,
        requested_trade_date=TARGET,
        received_trade_dates=(TARGET,),
        received_symbols=(),
        fetched_at=NOW,
        trace_ids=(),
    )
    with pytest.raises(DataUnavailableError, match="没有任何证券身份"):
        _adapter(tushare=FakeTushare(empty)).fetch_daily_increment(STOCKS, TARGET)

    wrong_date_frame = _stock_frame(("000001", "600000"), trade_date=BASELINE)
    wrong_date = TushareDailyFetch(
        frame=wrong_date_frame,
        requested_trade_date=TARGET,
        received_trade_dates=(TARGET,),
        received_symbols=("000001", "600000"),
        fetched_at=NOW,
        trace_ids=(),
    )
    with pytest.raises(DataQualityError, match="质量校验失败|target trade_date"):
        _adapter(tushare=FakeTushare(wrong_date)).fetch_daily_increment(STOCKS, TARGET)


@pytest.mark.parametrize(
    "status, error",
    [
        (AKShareVerificationStatus.MISMATCH, DataQualityError),
        (AKShareVerificationStatus.UNAVAILABLE, DataUnavailableError),
    ],
)
def test_every_non_verified_akshare_result_fails_closed(status, error) -> None:
    with pytest.raises(error, match="不是VERIFIED|不可用"):
        _adapter(verifier=FakeVerifier(status)).fetch_daily_increment(STOCKS, TARGET)


def test_provider_exception_text_and_token_never_escape() -> None:
    secret = "private-token-in-upstream-error"

    class FailingTushare:
        def fetch_daily(self, _: date) -> object:
            raise RuntimeError(f"request failed token={secret}")

    with pytest.raises(DataUnavailableError) as captured:
        _adapter(tushare=FailingTushare()).fetch_daily_increment(STOCKS, TARGET)

    assert secret not in str(captured.value)
    assert captured.value.__suppress_context__ is True


def test_core_indices_reject_wrong_set_and_wrong_baostock_date_or_source() -> None:
    bao = FakeBaoStock()
    with pytest.raises(DataQualityError, match="未验证代码|六核心"):
        _adapter(baostock=bao).fetch_daily_increment(
            BAOSTOCK_CORE_INDEX_SYMBOLS[:-1],
            TARGET,
            asset_kind="indices",
        )
    assert bao.index_calls == []

    wrong = _index_batch(trade_date=BASELINE)
    with pytest.raises(DataQualityError, match="质量校验失败|target trade_date"):
        _adapter(baostock=FakeBaoStock(index_batch=wrong)).fetch_daily_increment(
            BAOSTOCK_CORE_INDEX_SYMBOLS,
            TARGET,
            asset_kind="indices",
        )


def test_invalid_cutoff_is_rejected_before_any_provider_call() -> None:
    tushare = FakeTushare()
    with pytest.raises(ValueError, match="cutoff_timestamp"):
        _adapter(tushare=tushare).fetch_daily_increment(
            STOCKS,
            TARGET,
            cutoff_timestamp=True,
        )
    assert tushare.calls == []


def test_existing_overlay_range_accepts_composite_stock_and_index_batches(tmp_path) -> None:
    report = sync_daily_overlay_range(
        _adapter(),
        MarketOverlayStore(tmp_path),
        baseline_cutoff=BASELINE,
        through_date=TARGET,
        core_index_symbols=BAOSTOCK_CORE_INDEX_SYMBOLS,
        required_stock_coverage_ratio=1.0,
    )

    assert report.ready_through_requested_date is True
    assert report.verified_cutoff == TARGET
    assert report.completed_sessions == (TARGET,)
    assert report.results[0].source_id == ZERO_BUDGET_EOD_PROVIDER
    assert report.results[0].stock_coverage_ratio == 1.0
    assert report.results[0].index_count == len(BAOSTOCK_CORE_INDEX_SYMBOLS)


def _large_stock_universe(count: int = 100) -> tuple[str, ...]:
    return tuple(f"{value:06d}.SZ" for value in range(1, count + 1))


def _fetch_for_received(symbols: tuple[str, ...]) -> TushareDailyFetch:
    codes = tuple(symbol[:6] for symbol in symbols)
    frame = _stock_frame(codes)
    return TushareDailyFetch(
        frame=frame,
        requested_trade_date=TARGET,
        received_trade_dates=(TARGET,),
        received_symbols=codes,
        fetched_at=NOW,
        trace_ids=("stock-trace",),
    )


def test_outer_sync_accepts_small_suspension_gap_at_98_percent(tmp_path) -> None:
    requested = _large_stock_universe()
    adapter = _adapter(
        baostock=FakeBaoStock(symbols=requested),
        tushare=FakeTushare(_fetch_for_received(requested[:-1])),
    )

    report = sync_daily_overlay_range(
        adapter,
        MarketOverlayStore(tmp_path),
        baseline_cutoff=BASELINE,
        through_date=TARGET,
        core_index_symbols=BAOSTOCK_CORE_INDEX_SYMBOLS,
        required_stock_coverage_ratio=0.98,
    )

    assert report.ready_through_requested_date is True
    assert report.verified_cutoff == TARGET
    assert report.completed_sessions == (TARGET,)
    assert report.results[0].stock_count == 99
    assert report.results[0].stock_coverage_ratio == 0.99


def test_outer_sync_quarantines_gap_below_98_percent(tmp_path) -> None:
    requested = _large_stock_universe()
    adapter = _adapter(
        baostock=FakeBaoStock(symbols=requested),
        tushare=FakeTushare(_fetch_for_received(requested[:97])),
    )

    report = sync_daily_overlay_range(
        adapter,
        MarketOverlayStore(tmp_path),
        baseline_cutoff=BASELINE,
        through_date=TARGET,
        core_index_symbols=BAOSTOCK_CORE_INDEX_SYMBOLS,
        required_stock_coverage_ratio=0.98,
    )

    assert report.ready_through_requested_date is False
    assert report.verified_cutoff == BASELINE
    assert report.completed_sessions == ()
    assert report.results[0].status.value == "failed"
    assert report.results[0].quarantine_path is not None
    assert "coverage 0.9700 is below 0.9800" in report.results[0].reason
