from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from ashare_lab.adapters.market_overlay_store import MarketOverlayStore
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS

TARGET = date(2026, 8, 25)
BASELINE = date(2026, 8, 24)
NOW = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)


def _daily(
    symbols: tuple[str, ...],
    *,
    source: str = "infoway:eod_unadjusted",
    retrieved_at: str = "2026-08-25T09:00:00Z",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, symbol in enumerate(symbols):
        close = 10.0 + index / 10
        rows.append(
            {
                "symbol": symbol,
                "trade_date": TARGET,
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "prev_close": close - 0.05,
                "volume_shares": 1_000_000 + index,
                "amount_cny": (1_000_000 + index) * close,
                "turnover_pct": None,
                "source": source,
                "retrieved_at": retrieved_at,
            }
        )
    return pd.DataFrame(rows, columns=["symbol", *CANONICAL_DAILY_COLUMNS])


def _commit(
    store: MarketOverlayStore,
    *,
    retrieved_at: str = "2026-08-25T09:00:00Z",
):
    stocks = _daily(("000001", "000002"), retrieved_at=retrieved_at)
    indices = _daily(("000300", "399001"), retrieved_at=retrieved_at)
    run = store.begin_staging(source_id="infoway", trade_date=TARGET)
    store.stage_asset(run, "stocks", stocks)
    store.stage_asset(run, "indices", indices)
    return store.commit_verified(
        run,
        stocks=stocks,
        indices=indices,
        previous_trade_date=BASELINE,
        expected_stock_count=2,
        stock_coverage_ratio=1.0,
        core_index_symbols=("000300.SH", "399001.SZ"),
        receipt={"trace_ids": ["stock-trace", "index-trace"]},
        verified_at=NOW,
    )


def test_verified_run_publishes_both_assets_and_auditable_manifest(tmp_path: Path) -> None:
    store = MarketOverlayStore(tmp_path / "overlay")

    summary = _commit(store)

    assert summary.trade_date == TARGET
    assert summary.previous_trade_date == BASELINE
    assert summary.core_index_symbols == ("000300", "399001")
    assert summary.stock_count == 2
    assert summary.index_count == 2
    manifest = store.read_verified_manifest(source_id="infoway")
    assert len(manifest) == 1
    row = manifest.iloc[0]
    assert row["adjustment"] == "none"
    assert pd.Timestamp(row["previous_trade_date"]).date() == BASELINE
    assert json.loads(row["receipt_json"])["trace_ids"] == [
        "stock-trace",
        "index-trace",
    ]
    assert Path(row["stock_file"]).is_file()
    assert Path(row["index_file"]).is_file()
    assert store.latest_verified_date("infoway") == TARGET
    assert store.verified_dates_from(source_id="infoway", baseline_cutoff=BASELINE) == (TARGET,)

    stocks = store.read_verified_daily(TARGET, source_id="infoway", asset_kind="stocks")
    indices = store.read_verified_daily(TARGET, source_id="infoway", asset_kind="indices")
    assert list(stocks.columns) == ["symbol", *CANONICAL_DAILY_COLUMNS]
    assert stocks["symbol"].tolist() == ["000001", "000002"]
    assert indices["symbol"].tolist() == ["000300", "399001"]
    assert pd.api.types.is_datetime64_ns_dtype(stocks["trade_date"])
    assert stocks["retrieved_at"].str.endswith("Z").all()


def test_same_market_payload_rerun_is_idempotent_even_if_retrieval_time_changes(
    tmp_path: Path,
) -> None:
    store = MarketOverlayStore(tmp_path / "overlay")
    original = _commit(store)

    rerun = _commit(store, retrieved_at="2026-08-25T10:00:00Z")

    assert rerun.unchanged is True
    assert rerun.run_id == original.run_id
    assert len(store.read_verified_manifest(source_id="infoway")) == 1
    staging = tmp_path / "overlay" / "source=infoway" / "adjust=none" / "staging"
    assert not list(staging.glob("run=*"))


def test_quarantine_keeps_partial_evidence_without_advancing_manifest(tmp_path: Path) -> None:
    store = MarketOverlayStore(tmp_path / "overlay")
    run = store.begin_staging(
        source_id="infoway",
        trade_date=TARGET,
        receipt={"state": "started"},
    )
    store.stage_asset(run, "stocks", _daily(("000001",)))

    quarantine = store.quarantine(
        run,
        reason="index provider failed",
        failed_at=NOW,
    )

    assert quarantine.is_dir()
    assert (quarantine / "stocks.parquet").is_file()
    assert json.loads((quarantine / "failure.json").read_text())["reason"] == (
        "index provider failed"
    )
    assert store.read_verified_manifest().empty
    assert store.latest_verified_date("infoway") is None


def test_provider_partitions_cannot_be_silently_mixed(tmp_path: Path) -> None:
    store = MarketOverlayStore(tmp_path / "overlay")
    run = store.begin_staging(source_id="infoway", trade_date=TARGET)
    stocks = _daily(("000001",), source="other:eod")
    indices = _daily(("000300",))

    with pytest.raises(DataQualityError, match="source must belong"):
        store.commit_verified(
            run,
            stocks=stocks,
            indices=indices,
            previous_trade_date=BASELINE,
            expected_stock_count=1,
            stock_coverage_ratio=1.0,
            core_index_symbols=("000300.SH",),
            receipt={},
            verified_at=NOW,
        )

    store.quarantine(run, reason="mixed provider", failed_at=NOW)
    with pytest.raises(DataUnavailableError):
        store.read_verified_daily(TARGET, source_id="infoway", asset_kind="stocks")


def test_duplicate_symbols_are_rejected_before_manifest_publish(tmp_path: Path) -> None:
    store = MarketOverlayStore(tmp_path / "overlay")
    run = store.begin_staging(source_id="infoway", trade_date=TARGET)
    duplicate = pd.concat((_daily(("000001",)), _daily(("000001",))), ignore_index=True)

    with pytest.raises(DataQualityError, match="duplicate"):
        store.commit_verified(
            run,
            stocks=duplicate,
            indices=_daily(("000300",)),
            previous_trade_date=BASELINE,
            expected_stock_count=1,
            stock_coverage_ratio=1.0,
            core_index_symbols=("000300.SH",),
            receipt={},
            verified_at=NOW,
        )
    assert store.read_verified_manifest().empty
