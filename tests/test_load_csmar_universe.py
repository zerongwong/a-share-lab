from __future__ import annotations

from datetime import date, timedelta
from importlib import import_module
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from ashare_lab.domain.errors import DataUnavailableError
from ashare_lab.services.load_csmar_universe import load_csmar_universe


def _build_catalog(root: Path, *, end: date = date(2026, 8, 24)) -> None:
    connection = duckdb.connect(str(root / "csmar.duckdb"))
    connection.execute(
        """
        CREATE TABLE security_master (
            symbol VARCHAR, name VARCHAR, exchange VARCHAR, board VARCHAR,
            industry VARCHAR, industry_csrc_2012 VARCHAR,
            industry_listing_association VARCHAR, is_st BOOLEAN,
            is_delisting BOOLEAN, list_date TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE daily_bars (
            symbol VARCHAR, trade_date TIMESTAMP, open DOUBLE, high DOUBLE,
            low DOUBLE, close DOUBLE, prev_close DOUBLE, volume_shares DOUBLE,
            amount_cny DOUBLE, turnover_pct DOUBLE, source VARCHAR,
            retrieved_at VARCHAR
        )
        """
    )
    for index in range(5):
        symbol = f"{index + 1:06d}"
        connection.execute(
            "INSERT INTO security_master VALUES (?, ?, 'SZSE', 'sz_main', ?, ?, ?, false, false, ?)",
            [
                symbol,
                f"股票{index}",
                f"行业{index}",
                f"行业{index}",
                f"行业{index}",
                date(2020, 1, 1),
            ],
        )
        rows = []
        previous = 10.0 + index
        for offset in range(130):
            day = end - timedelta(days=129 - offset)
            close = previous * 1.001
            rows.append(
                (
                    symbol,
                    day,
                    previous,
                    close,
                    previous,
                    close,
                    previous,
                    None,
                    50_000_000.0 + index * 1_000_000.0,
                    None,
                    "csmar:test",
                    "2026-08-25T00:00:00Z",
                )
            )
            previous = close
        connection.executemany(
            "INSERT INTO daily_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    connection.close()


def test_loads_active_current_universe_without_fabricated_optional_factors(tmp_path: Path) -> None:
    _build_catalog(tmp_path)
    snapshot = load_csmar_universe(
        tmp_path,
        as_of=date(2026, 8, 25),
        minimum_sessions=121,
        history_sessions=125,
    )

    assert snapshot.data_cutoff == date(2026, 8, 24)
    assert snapshot.master_symbols == 5
    assert snapshot.active_symbols == 5
    assert snapshot.eligible_symbols == 5
    assert all(len(frame) == 125 for frame in snapshot.histories.values())
    assert all(item["fundamental_score"] is None for item in snapshot.metadata.values())
    assert all(item["news_score"] is None for item in snapshot.metadata.values())
    assert all(item["market_regime_score"] is None for item in snapshot.metadata.values())
    assert all(item["sector_score"] is None for item in snapshot.metadata.values())
    assert all("quality_score" not in item for item in snapshot.metadata.values())
    assert all("catalyst_score" not in item for item in snapshot.metadata.values())
    assert all(item["is_limit_up_at_cutoff"] is False for item in snapshot.metadata.values())
    assert snapshot.full_day_volume_available is False
    assert snapshot.fundamental_scores_available is False
    assert snapshot.news_scores_available is False


def test_live_mode_rejects_a_stale_previous_session_instead_of_silent_fallback(
    tmp_path: Path,
) -> None:
    _build_catalog(tmp_path, end=date(2026, 8, 24))

    with pytest.raises(DataUnavailableError, match="实际只有2026-08-24"):
        load_csmar_universe(
            tmp_path,
            as_of=date(2026, 8, 26),
            decision_date=date(2026, 8, 26),
            minimum_sessions=121,
            history_sessions=125,
        )

    replay = load_csmar_universe(
        tmp_path,
        as_of=date(2026, 8, 26),
        decision_date=date(2026, 8, 26),
        mode="historical",
        minimum_sessions=121,
        history_sessions=125,
    )
    assert replay.data_cutoff == date(2026, 8, 24)


class _ReferenceFixture:
    def __init__(self, cutoff: date) -> None:
        self.cutoff = cutoff

    def read_index_daily(self, index_code: str, start: date, end: date) -> pd.DataFrame:
        dates = pd.bdate_range(end=self.cutoff, periods=130)
        return pd.DataFrame(
            {
                "index_code": index_code,
                "trade_date": dates,
                "close": range(100, 230),
                "historical_backtest_eligible": True,
                "common_cutoff_date": self.cutoff,
            }
        )

    def read_balance_sheet_snapshot(self) -> pd.DataFrame:
        rows = []
        for index in range(5):
            assets = 100.0
            liabilities = 70.0 - index * 5.0
            rows.append(
                {
                    "symbol": f"{index + 1:06d}",
                    "name": f"样本制造{index}",
                    "report_period": "2026-06-30",
                    "cash_cny": 10.0 + index,
                    "accounts_receivable_cny": 10.0 - index,
                    "inventory_cny": 8.0 - index,
                    "current_assets_cny": 50.0 + index,
                    "total_assets_cny": assets,
                    "current_liabilities_cny": 30.0 - index,
                    "total_liabilities_cny": liabilities,
                    "total_equity_cny": assets - liabilities,
                    "data_role": "current_snapshot",
                    "historical_backtest_eligible": False,
                    "common_cutoff_date": self.cutoff,
                    "retrieved_at": "2026-08-25",
                }
            )
        return pd.DataFrame(rows)


def test_current_balance_snapshot_is_rejected_before_safe_price_and_decision_cutoff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _build_catalog(tmp_path)
    module = import_module("ashare_lab.services.load_csmar_universe")
    monkeypatch.setattr(
        module, "CSMARReferenceData", lambda _: _ReferenceFixture(date(2026, 8, 24))
    )

    snapshot = load_csmar_universe(
        tmp_path,
        as_of=date(2026, 8, 25),
        decision_date=date(2026, 8, 25),
        reference_dataset_root=tmp_path / "reference",
        minimum_sessions=121,
        history_sessions=125,
        balance_sheet_minimum_group_size=3,
    )

    assert snapshot.balance_sheet_strength_available is False
    assert snapshot.balance_sheet_strength_reason == "decision_precedes_safe_current_snapshot_use"
    assert all(item["balance_sheet_strength_score"] is None for item in snapshot.metadata.values())
    assert len(snapshot.market_index_histories) == 6


def test_current_balance_snapshot_can_enter_only_after_safe_live_cutoffs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _build_catalog(tmp_path, end=date(2026, 8, 25))
    module = import_module("ashare_lab.services.load_csmar_universe")
    monkeypatch.setattr(
        module, "CSMARReferenceData", lambda _: _ReferenceFixture(date(2026, 8, 25))
    )

    snapshot = load_csmar_universe(
        tmp_path,
        as_of=date(2026, 8, 26),
        decision_date=date(2026, 8, 26),
        reference_dataset_root=tmp_path / "reference",
        minimum_sessions=121,
        history_sessions=125,
        balance_sheet_minimum_group_size=3,
    )

    assert snapshot.data_cutoff == date(2026, 8, 25)
    assert snapshot.balance_sheet_strength_available is True
    assert snapshot.balance_sheet_strength_symbols == 5
    assert snapshot.balance_sheet_strength_excluded_symbols == 0
    assert all(
        0.0 <= float(item["balance_sheet_strength_score"]) <= 1.0
        for item in snapshot.metadata.values()
    )


def test_historical_mode_never_uses_current_balance_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _build_catalog(tmp_path, end=date(2026, 8, 25))
    module = import_module("ashare_lab.services.load_csmar_universe")
    monkeypatch.setattr(
        module, "CSMARReferenceData", lambda _: _ReferenceFixture(date(2026, 8, 25))
    )

    snapshot = load_csmar_universe(
        tmp_path,
        as_of=date(2026, 8, 26),
        decision_date=date(2026, 8, 26),
        mode="historical",
        reference_dataset_root=tmp_path / "reference",
        minimum_sessions=121,
        history_sessions=125,
        balance_sheet_minimum_group_size=3,
    )

    assert snapshot.balance_sheet_strength_available is False
    assert snapshot.balance_sheet_strength_reason == "historical_mode_rejects_current_snapshot"
