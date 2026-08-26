from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from ashare_lab.adapters.parquet_market_store import ParquetMarketStore
from ashare_lab.domain.data_sources import (
    AuthorizationBasis,
    DataAction,
    RightsPolicy,
    RightsViolationError,
    SourceDefinition,
    SourceId,
    SourceRegistry,
    SourceStatus,
)
from ashare_lab.ports.bulk_market_data import BulkMarketDataPort
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS
from ashare_lab.services.sync_full_market_daily import (
    TradeDateSyncStatus,
    sync_full_market_daily,
)

NOW = datetime(2026, 8, 24, 8, 0, tzinfo=UTC)
SYMBOLS = ("000001", "000002", "600000", "600150")
SESSIONS = (
    date(2026, 8, 18),
    date(2026, 8, 19),
    date(2026, 8, 20),
    date(2026, 8, 21),
    date(2026, 8, 24),
    date(2026, 8, 25),
)


def _rights_policy() -> RightsPolicy:
    registry = SourceRegistry(
        [
            SourceDefinition(
                source_id=SourceId.TUSHARE,
                display_name="Tushare Pro",
                status=SourceStatus.CONNECTED,
                purposes=("授权全A研究",),
                official_url="https://tushare.pro/",
                application_url="https://tushare.pro/register",
                authorization_basis=AuthorizationBasis.ACCOUNT_ENTITLEMENT,
                authorization_reference="local-account-entitlement",
                allowed_actions=frozenset(
                    {DataAction.MARKET_DATA_READ, DataAction.MARKET_DATA_CACHE}
                ),
            )
        ]
    )
    return RightsPolicy(registry)


def _master(*, symbols: tuple[str, ...] = SYMBOLS) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": list(symbols),
            "name": [f"测试{index}" for index in range(len(symbols))],
            "exchange": ["SH" if code.startswith("6") else "SZ" for code in symbols],
            "board": ["主板"] * len(symbols),
            "list_date": ["2010-01-01"] * len(symbols),
            "delist_date": [None] * len(symbols),
            "industry": [f"行业{index}" for index in range(len(symbols))],
            "is_st": [False] * len(symbols),
            "is_delisting": [False] * len(symbols),
            "is_suspended": [False] * len(symbols),
            "source": ["tushare"] * len(symbols),
            "retrieved_at": [NOW.isoformat()] * len(symbols),
        }
    )


def _cross_section(trade_date: date, symbols: tuple[str, ...] = SYMBOLS) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    day_index = SESSIONS.index(trade_date)
    for symbol_index, symbol in enumerate(symbols):
        close = 10.0 + symbol_index + day_index * 0.1
        rows.append(
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "open": close - 0.05,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "prev_close": close - 0.1 if day_index else None,
                "volume_shares": 1_000_000 + day_index,
                "amount_cny": close * 1_000_000,
                "turnover_pct": 1.0,
                "source": "tushare:licensed_daily",
                "retrieved_at": NOW.isoformat(),
            }
        )
    return pd.DataFrame(rows, columns=["symbol", *CANONICAL_DAILY_COLUMNS])


class FakeDateMajorProvider:
    source_id = SourceId.TUSHARE

    def __init__(self) -> None:
        self.master_calls: list[date] = []
        self.calendar_calls: list[tuple[date, date]] = []
        self.daily_calls: list[tuple[date, str]] = []

    def fetch_security_master(self, as_of: date) -> pd.DataFrame:
        self.master_calls.append(as_of)
        return _master()

    def fetch_trade_calendar(self, start: date, end: date) -> tuple[date, ...]:
        self.calendar_calls.append((start, end))
        return tuple(value for value in SESSIONS if start <= value <= end)

    def fetch_daily_for_trade_date(
        self,
        trade_date: date,
        *,
        adjust: str = "none",
    ) -> pd.DataFrame:
        self.daily_calls.append((trade_date, adjust))
        return _cross_section(trade_date)


def test_five_sessions_make_five_market_calls_not_stock_times_date_calls(
    tmp_path: Path,
) -> None:
    provider = FakeDateMajorProvider()
    assert isinstance(provider, BulkMarketDataPort)
    store = ParquetMarketStore(tmp_path / "market")

    report = sync_full_market_daily(
        provider,
        store,
        _rights_policy(),
        as_of=date(2026, 8, 24),
        bootstrap_start=date(2026, 8, 18),
        overlap_sessions=2,
        minimum_sessions=3,
        required_coverage_ratio=1.0,
        clock=lambda: NOW,
    )

    assert provider.master_calls == [date(2026, 8, 24)]
    assert provider.calendar_calls == [(date(2026, 8, 18), date(2026, 8, 24))]
    assert [value for value, _ in provider.daily_calls] == list(SESSIONS[:5])
    assert len(provider.daily_calls) == 5
    assert report.calendar_sessions == 5
    assert report.updated_sessions == 5
    assert report.failed_sessions == 0
    assert report.session_coverage_ratio == 1.0
    assert report.current_symbol_coverage_ratio == 1.0
    assert report.strategy_coverage_ratio == 1.0
    assert report.ready_for_full_market_screen is True
    assert report.market_cutoff == date(2026, 8, 24)
    assert Path(report.report_json_path or "").is_file()
    assert Path(report.report_symbol_detail_path or "").is_file()
    assert Path(report.report_date_detail_path or "").is_file()

    master = store.read_security_master(SourceId.TUSHARE, as_of=date(2026, 8, 24))
    assert master["symbol"].tolist() == sorted(SYMBOLS)
    latest = store.read_daily_cross_section(
        date(2026, 8, 24),
        source_id=SourceId.TUSHARE,
        adjustment="none",
    )
    assert latest["symbol"].tolist() == sorted(SYMBOLS)
    history = store.read_daily_market(
        source_id=SourceId.TUSHARE,
        adjustment="none",
        start=date(2026, 8, 18),
        end=date(2026, 8, 24),
        symbols=["600150"],
    )
    assert len(history) == 5
    assert set(history["symbol"]) == {"600150"}


def test_incremental_sync_fetches_only_overlap_and_missing_session(tmp_path: Path) -> None:
    provider = FakeDateMajorProvider()
    store = ParquetMarketStore(tmp_path / "market")
    common = {
        "bootstrap_start": date(2026, 8, 18),
        "overlap_sessions": 2,
        "minimum_sessions": 3,
        "required_coverage_ratio": 1.0,
        "clock": lambda: NOW,
    }
    sync_full_market_daily(
        provider,
        store,
        _rights_policy(),
        as_of=date(2026, 8, 24),
        **common,
    )
    provider.daily_calls.clear()

    report = sync_full_market_daily(
        provider,
        store,
        _rights_policy(),
        as_of=date(2026, 8, 25),
        **common,
    )

    assert [value for value, _ in provider.daily_calls] == [
        date(2026, 8, 24),
        date(2026, 8, 25),
    ]
    assert report.attempted_sessions == 2
    assert report.updated_sessions == 1
    assert report.unchanged_sessions == 1
    assert report.stored_sessions == 6
    assert report.market_cutoff == date(2026, 8, 25)


def test_missing_latest_session_fails_quality_gate_without_false_full_coverage(
    tmp_path: Path,
) -> None:
    class MissingLatestProvider(FakeDateMajorProvider):
        def fetch_daily_for_trade_date(
            self,
            trade_date: date,
            *,
            adjust: str = "none",
        ) -> pd.DataFrame:
            if trade_date == date(2026, 8, 24):
                raise RuntimeError("temporary unavailable")
            return super().fetch_daily_for_trade_date(trade_date, adjust=adjust)

    report = sync_full_market_daily(
        MissingLatestProvider(),
        ParquetMarketStore(tmp_path / "market"),
        _rights_policy(),
        as_of=date(2026, 8, 24),
        bootstrap_start=date(2026, 8, 18),
        minimum_sessions=3,
        required_coverage_ratio=1.0,
        clock=lambda: NOW,
    )

    failed = {item.trade_date: item for item in report.results}[date(2026, 8, 24)]
    assert failed.status is TradeDateSyncStatus.FAILED
    assert report.session_coverage_ratio == pytest.approx(0.8)
    assert report.market_cutoff is None
    assert report.ready_for_full_market_screen is False
    assert any("不得标称" in warning for warning in report.warnings)


def test_default_unconnected_rights_fail_before_provider_or_filesystem(
    tmp_path: Path,
) -> None:
    provider = FakeDateMajorProvider()
    root = tmp_path / "market"

    with pytest.raises(RightsViolationError):
        sync_full_market_daily(
            provider,
            ParquetMarketStore(root),
            RightsPolicy(SourceRegistry.load_default()),
            as_of=date(2026, 8, 24),
            bootstrap_start=date(2026, 8, 18),
            minimum_sessions=3,
            clock=lambda: NOW,
        )

    assert provider.master_calls == []
    assert provider.calendar_calls == []
    assert provider.daily_calls == []
    assert not root.exists()


def test_provider_cannot_inject_symbol_outside_security_master(tmp_path: Path) -> None:
    class UnexpectedProvider(FakeDateMajorProvider):
        def fetch_daily_for_trade_date(
            self,
            trade_date: date,
            *,
            adjust: str = "none",
        ) -> pd.DataFrame:
            return _cross_section(trade_date, (*SYMBOLS, "300001"))

    report = sync_full_market_daily(
        UnexpectedProvider(),
        ParquetMarketStore(tmp_path / "market"),
        _rights_policy(),
        as_of=date(2026, 8, 24),
        bootstrap_start=date(2026, 8, 18),
        minimum_sessions=3,
        clock=lambda: NOW,
    )

    assert report.failed_sessions == 5
    assert report.stored_sessions == 0
    assert all(item.status is TradeDateSyncStatus.FAILED for item in report.results)
    assert all("主表之外" in item.reason for item in report.results)
