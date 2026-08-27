from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from ashare_lab.adapters.market_overlay_store import MarketOverlayStore
from ashare_lab.ports.daily_increment import AssetKind, DailyIncrementBatch, DailyIncrementPort
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS, normalize_symbol
from ashare_lab.services.sync_daily_overlay import (
    DailyOverlaySyncStatus,
    sync_daily_overlay,
    sync_daily_overlay_range,
)

BASELINE = date(2026, 8, 24)
DAY_25 = date(2026, 8, 25)
DAY_26 = date(2026, 8, 26)
NOW = datetime(2026, 8, 27, 8, 0, tzinfo=UTC)
STOCKS = tuple(f"{value:06d}.SZ" for value in range(1, 101))
INDICES = ("000300.SH", "399001.SZ")


def _frame(symbols: tuple[str, ...], target_date: date) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, raw_symbol in enumerate(symbols):
        symbol = normalize_symbol(raw_symbol)
        close = 10.0 + index / 100 + (target_date - BASELINE).days / 10
        rows.append(
            {
                "symbol": symbol,
                "trade_date": target_date,
                "open": close - 0.05,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "prev_close": close - 0.1,
                "volume_shares": 1_000_000 + index,
                "amount_cny": close * (1_000_000 + index),
                "turnover_pct": None,
                "source": "infoway:eod_unadjusted",
                "retrieved_at": NOW.isoformat().replace("+00:00", "Z"),
            }
        )
    return pd.DataFrame(rows, columns=["symbol", *CANONICAL_DAILY_COLUMNS])


class FakeDailyIncrementProvider:
    provider = "infoway"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], date, str]] = []
        self.calendar_calls: list[tuple[date, date]] = []
        self.stock_missing: dict[date, int] = {}
        self.missing_index_dates: set[date] = set()
        self.raise_index_dates: set[date] = set()
        self.duplicate_stock_dates: set[date] = set()

    def fetch_cn_trading_days(self, start: date, end: date) -> tuple[date, ...]:
        self.calendar_calls.append((start, end))
        return tuple(day for day in (DAY_25, DAY_26) if start <= day <= end)

    def fetch_cn_stock_symbols(self) -> tuple[str, ...]:
        return STOCKS

    def fetch_daily_increment(
        self,
        symbols: tuple[str, ...],
        target_date: date,
        *,
        cutoff_timestamp: int | None = None,
        asset_kind: AssetKind = "stocks",
    ) -> DailyIncrementBatch:
        requested = tuple(symbols)
        self.calls.append((requested, target_date, asset_kind))
        is_index = asset_kind == "indices"
        if is_index and target_date in self.raise_index_dates:
            raise RuntimeError("index transport failed")
        received = requested
        if is_index and target_date in self.missing_index_dates:
            received = requested[:-1]
        if not is_index:
            missing = self.stock_missing.get(target_date, 0)
            received = requested[: len(requested) - missing] if missing else requested
        frame = _frame(received, target_date)
        if not is_index and target_date in self.duplicate_stock_dates:
            frame = pd.concat((frame, frame.iloc[[0]]), ignore_index=True)
        return DailyIncrementBatch(
            frame=frame,
            target_date=target_date,
            requested_symbols=requested,
            received_symbols=received,
            fetched_at=NOW,
            trace_ids=(f"trace-{target_date.isoformat()}",),
            provider=self.provider,
            cutoff_timestamp=cutoff_timestamp or 1_777_777_777,
        )


def test_protocol_and_range_add_25_and_26_after_24_baseline(tmp_path: Path) -> None:
    provider = FakeDailyIncrementProvider()
    assert isinstance(provider, DailyIncrementPort)
    store = MarketOverlayStore(tmp_path / "overlay")

    report = sync_daily_overlay_range(
        provider,
        store,
        baseline_cutoff=BASELINE,
        through_date=DAY_26,
        core_index_symbols=INDICES,
        required_stock_coverage_ratio=0.98,
        clock=lambda: NOW,
    )

    assert report.started_cutoff == BASELINE
    assert report.expected_sessions == (DAY_25, DAY_26)
    assert report.completed_sessions == (DAY_25, DAY_26)
    assert report.verified_cutoff == DAY_26
    assert report.ready_through_requested_date is True
    assert [result.status for result in report.results] == [
        DailyOverlaySyncStatus.VERIFIED,
        DailyOverlaySyncStatus.VERIFIED,
    ]
    assert [asset_kind for _, _, asset_kind in provider.calls] == [
        "stocks",
        "indices",
        "stocks",
        "indices",
    ]
    manifest = store.read_verified_manifest(source_id="infoway")
    receipt = json.loads(str(manifest.iloc[-1]["receipt_json"]))
    assert receipt["stocks"]["asset_kind"] == "stocks"
    assert receipt["indices"]["asset_kind"] == "indices"
    assert [pd.Timestamp(value).date() for value in manifest["trade_date"]] == [
        DAY_25,
        DAY_26,
    ]
    assert [pd.Timestamp(value).date() for value in manifest["previous_trade_date"]] == [
        BASELINE,
        DAY_25,
    ]
    assert store.verified_dates_from(source_id="infoway", baseline_cutoff=BASELINE) == (
        DAY_25,
        DAY_26,
    )


def test_partial_index_fetch_failure_is_quarantined_and_cutoff_does_not_advance(
    tmp_path: Path,
) -> None:
    provider = FakeDailyIncrementProvider()
    provider.raise_index_dates.add(DAY_25)
    store = MarketOverlayStore(tmp_path / "overlay")

    result = sync_daily_overlay(
        provider,
        store,
        target_date=DAY_25,
        previous_trade_date=BASELINE,
        core_index_symbols=INDICES,
        clock=lambda: NOW,
    )

    assert result.status is DailyOverlaySyncStatus.FAILED
    assert result.verified_cutoff == BASELINE
    assert result.quarantine_path is not None
    quarantine = Path(result.quarantine_path)
    assert (quarantine / "stocks.parquet").is_file()
    assert not (quarantine / "indices.parquet").exists()
    assert store.read_verified_manifest().empty


def test_missing_core_index_fails_closed(tmp_path: Path) -> None:
    provider = FakeDailyIncrementProvider()
    provider.missing_index_dates.add(DAY_25)
    store = MarketOverlayStore(tmp_path / "overlay")

    result = sync_daily_overlay(
        provider,
        store,
        target_date=DAY_25,
        previous_trade_date=BASELINE,
        core_index_symbols=INDICES,
        clock=lambda: NOW,
    )

    assert result.status is DailyOverlaySyncStatus.FAILED
    assert "incomplete" in result.reason
    assert store.latest_verified_date("infoway") is None


def test_98_percent_stock_coverage_passes_but_97_percent_does_not(tmp_path: Path) -> None:
    provider = FakeDailyIncrementProvider()
    provider.stock_missing[DAY_25] = 2
    store = MarketOverlayStore(tmp_path / "pass")

    passing = sync_daily_overlay(
        provider,
        store,
        target_date=DAY_25,
        previous_trade_date=BASELINE,
        core_index_symbols=INDICES,
        required_stock_coverage_ratio=0.98,
        clock=lambda: NOW,
    )

    assert passing.status is DailyOverlaySyncStatus.VERIFIED
    assert passing.stock_coverage_ratio == 0.98

    failing_provider = FakeDailyIncrementProvider()
    failing_provider.stock_missing[DAY_25] = 3
    failing_store = MarketOverlayStore(tmp_path / "fail")
    failing = sync_daily_overlay(
        failing_provider,
        failing_store,
        target_date=DAY_25,
        previous_trade_date=BASELINE,
        core_index_symbols=INDICES,
        required_stock_coverage_ratio=0.98,
        clock=lambda: NOW,
    )

    assert failing.status is DailyOverlaySyncStatus.FAILED
    assert "coverage" in failing.reason
    assert failing.verified_cutoff == BASELINE
    assert failing_store.read_verified_manifest().empty


def test_duplicate_stock_rows_are_quarantined(tmp_path: Path) -> None:
    provider = FakeDailyIncrementProvider()
    provider.duplicate_stock_dates.add(DAY_25)
    store = MarketOverlayStore(tmp_path / "overlay")

    result = sync_daily_overlay(
        provider,
        store,
        target_date=DAY_25,
        previous_trade_date=BASELINE,
        core_index_symbols=INDICES,
        clock=lambda: NOW,
    )

    assert result.status is DailyOverlaySyncStatus.FAILED
    assert "duplicate" in result.reason
    assert store.read_verified_manifest().empty


def test_same_session_rerun_is_unchanged_and_keeps_one_manifest_row(tmp_path: Path) -> None:
    provider = FakeDailyIncrementProvider()
    store = MarketOverlayStore(tmp_path / "overlay")
    common = {
        "target_date": DAY_25,
        "previous_trade_date": BASELINE,
        "core_index_symbols": INDICES,
        "clock": lambda: NOW,
    }

    first = sync_daily_overlay(provider, store, **common)
    second = sync_daily_overlay(provider, store, **common)

    assert first.status is DailyOverlaySyncStatus.VERIFIED
    assert second.status is DailyOverlaySyncStatus.UNCHANGED
    assert second.run_id == first.run_id
    assert len(store.read_verified_manifest(source_id="infoway")) == 1


def test_range_stops_at_first_failure_and_never_fetches_later_session(tmp_path: Path) -> None:
    provider = FakeDailyIncrementProvider()
    provider.stock_missing[DAY_25] = 3
    store = MarketOverlayStore(tmp_path / "overlay")

    report = sync_daily_overlay_range(
        provider,
        store,
        baseline_cutoff=BASELINE,
        through_date=DAY_26,
        core_index_symbols=INDICES,
        clock=lambda: NOW,
    )

    assert report.verified_cutoff == BASELINE
    assert report.completed_sessions == ()
    assert report.ready_through_requested_date is False
    assert all(target != DAY_26 for _, target, _ in provider.calls)
    assert store.read_verified_manifest().empty
