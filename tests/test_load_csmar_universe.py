from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb

from ashare_lab.services.load_csmar_universe import load_csmar_universe


def _build_catalog(root: Path) -> None:
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
    end = date(2026, 8, 24)
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
