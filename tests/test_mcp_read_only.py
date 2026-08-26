from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from ashare_lab.mcp_server import MCPSettings, ReadOnlyResearchTools, build_server


def _settings(tmp_path: Path, *, allow: bool) -> MCPSettings:
    return MCPSettings(
        csmar_cache_dir=tmp_path / "csmar",
        research_db=tmp_path / "research.db",
        allow_licensed_derived_results=allow,
    )


def _seed_weekly_archive(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE runs (
                id TEXT PRIMARY KEY, run_type TEXT, as_of TEXT, data_cutoff TEXT,
                created_at TEXT, strategy_version TEXT, model_id TEXT,
                status TEXT, warning_json TEXT
            );
            CREATE TABLE portfolio_sets (
                id TEXT PRIMARY KEY, run_id TEXT, risk_profile TEXT,
                cash_weight REAL, borrowed_weight REAL, expected_return REAL,
                expected_vol REAL, expected_max_drawdown REAL, sharpe REAL,
                metric_window TEXT
            );
            CREATE TABLE portfolio_members (
                portfolio_id TEXT, symbol TEXT, weight REAL, rank INTEGER,
                reason_json TEXT
            );
            INSERT INTO runs VALUES (
                'older', 'weekly_portfolios', '2026-08-20', '2026-08-20',
                '2026-08-20T08:00:00Z', 'old', NULL, 'completed', '[]'
            );
            INSERT INTO runs VALUES (
                'latest', 'weekly_portfolios', '2026-08-25', '2026-08-25',
                '2026-08-25T08:00:00Z', 'medium-v1', NULL, 'completed',
                '["historical evidence only"]'
            );
            INSERT INTO portfolio_sets VALUES (
                'portfolio-1', 'latest', 'balanced', 0.2, 0.0,
                NULL, NULL, NULL, NULL, 'not_calibrated'
            );
            """
        )
        for rank, (symbol, weight) in enumerate(
            (("600001", 0.30), ("000001", 0.20), ("300001", 0.20), ("688001", 0.10)),
            start=1,
        ):
            connection.execute(
                "INSERT INTO portfolio_members VALUES (?, ?, ?, ?, ?)",
                (
                    "portfolio-1",
                    symbol,
                    weight,
                    rank,
                    json.dumps({"ranking_score_not_probability": 0.9 - rank / 10}),
                ),
            )


def test_data_status_does_not_expose_local_paths(tmp_path: Path) -> None:
    settings = _settings(tmp_path, allow=False)
    status = ReadOnlyResearchTools(settings).get_data_status()

    assert status["status"] == "unavailable"
    assert status["raw_data_exposed"] is False
    assert str(tmp_path) not in json.dumps(status, ensure_ascii=False)


def test_derived_results_are_fail_closed_by_default(tmp_path: Path) -> None:
    tools = ReadOnlyResearchTools(_settings(tmp_path, allow=False))

    with pytest.raises(PermissionError, match="disabled"):
        tools.get_latest_research()


def test_latest_weekly_research_is_read_without_raw_rows(tmp_path: Path) -> None:
    settings = _settings(tmp_path, allow=True)
    _seed_weekly_archive(settings.research_db)

    result = ReadOnlyResearchTools(settings).get_latest_research()

    assert result["status"] == "ready"
    assert result["run"]["id"] == "latest"
    assert result["raw_data_exposed"] is False
    assert [member["rank"] for member in result["portfolios"][0]["members"]] == [1, 2, 3, 4]
    assert "data_snapshots" not in result
    assert "evidence" not in result


def test_invalid_horizon_is_rejected_before_data_access(tmp_path: Path) -> None:
    tools = ReadOnlyResearchTools(_settings(tmp_path, allow=True))

    with pytest.raises(ValueError, match="holding_weeks"):
        tools.generate_portfolio(holding_weeks=2)


def test_registered_mcp_tools_are_all_read_only(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    server = build_server(_settings(tmp_path, allow=False))

    registered = asyncio.run(server.list_tools())

    assert {tool.name for tool in registered} == {
        "generate_portfolio",
        "get_data_status",
        "get_latest_research",
    }
    assert all(tool.annotations.readOnlyHint is True for tool in registered)
    assert all(tool.annotations.destructiveHint is False for tool in registered)
    assert all(tool.annotations.openWorldHint is False for tool in registered)
