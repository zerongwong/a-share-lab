"""Optional, read-only Model Context Protocol server for A Share Lab.

The tool handlers intentionally return only compact derived research results.
They never expose raw CSMAR rows, provider credentials, positions, or orders.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import numpy as np
import pandas as pd
from platformdirs import user_data_path

from ashare_lab.services.build_weekly_portfolios import (
    WeeklyPortfolioStatus,
    build_weekly_portfolios,
)
from ashare_lab.services.load_csmar_universe import load_csmar_universe

ALLOWED_HOLDING_WEEKS = frozenset({1, 4, 13, 26, 52})
ALLOWED_RUN_TYPES = frozenset({"stock_analysis", "weekly_portfolios", "limit_watchlist"})

DATA_RIGHTS_NOTICE = (
    "仅返回本机研究计算产生的精简衍生结果，不返回原始行情、财务报表或新闻正文。"
    "启用前请确认数据许可允许将衍生结果交给所连接的模型服务处理。"
)
SERVER_INSTRUCTIONS = (
    "只读A股研究工具。先调用get_data_status确认本地数据和许可开关。"
    "结果是历史研究证据，不是收益承诺或个性化投资建议；不得声称已下单。"
)


def _environment_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


@dataclass(frozen=True, slots=True)
class MCPSettings:
    """Local-only MCP settings loaded from environment variables."""

    csmar_cache_dir: Path
    research_db: Path
    allow_licensed_derived_results: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    transport: str = "streamable-http"

    @classmethod
    def from_env(cls) -> MCPSettings:
        default_data_dir = Path(user_data_path("A股研究助手", appauthor=False))
        cache_dir = Path(
            os.getenv("ASHARE_CSMAR_CACHE_DIR", str(default_data_dir / "cache" / "csmar"))
        ).expanduser()
        research_db = Path(
            os.getenv("ASHARE_RESEARCH_DB", str(default_data_dir / "research.db"))
        ).expanduser()
        host = os.getenv("ASHARE_MCP_HOST", "127.0.0.1").strip()
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                "ASHARE_MCP_HOST must remain loopback-only; use an authenticated HTTPS "
                "reverse proxy for remote access"
            )
        try:
            port = int(os.getenv("ASHARE_MCP_PORT", "8765"))
        except ValueError as exc:
            raise ValueError("ASHARE_MCP_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("ASHARE_MCP_PORT must be between 1 and 65535")
        transport = os.getenv("ASHARE_MCP_TRANSPORT", "streamable-http").strip().lower()
        if transport not in {"streamable-http", "stdio"}:
            raise ValueError("ASHARE_MCP_TRANSPORT must be streamable-http or stdio")
        return cls(
            csmar_cache_dir=cache_dir.resolve(),
            research_db=research_db.resolve(),
            allow_licensed_derived_results=_environment_flag(
                "ASHARE_MCP_ALLOW_LICENSED_DERIVED_RESULTS"
            ),
            host=host,
            port=port,
            transport=transport,
        )


class ReadOnlyResearchTools:
    """Pure read/compute facade used by MCP and unit tests."""

    def __init__(self, settings: MCPSettings) -> None:
        self.settings = settings

    def get_data_status(self) -> dict[str, Any]:
        """Report readiness without returning paths, credentials, or source rows."""

        catalog_available = (self.settings.csmar_cache_dir / "csmar.duckdb").is_file()
        return {
            "status": "ready" if catalog_available else "unavailable",
            "csmar_catalog_available": catalog_available,
            "research_archive_available": self.settings.research_db.is_file(),
            "derived_results_over_mcp_enabled": self.settings.allow_licensed_derived_results,
            "raw_data_exposed": False,
            "trading_actions_available": False,
            "notice": DATA_RIGHTS_NOTICE,
        }

    def generate_portfolio(
        self,
        *,
        as_of: str | None = None,
        holding_weeks: int = 13,
    ) -> dict[str, Any]:
        """Compute a portfolio from the local catalog without archiving it."""

        self._require_derived_result_permission()
        if holding_weeks not in ALLOWED_HOLDING_WEEKS:
            allowed = ", ".join(str(item) for item in sorted(ALLOWED_HOLDING_WEEKS))
            raise ValueError(f"holding_weeks must be one of: {allowed}")
        cutoff = date.today() if as_of is None else date.fromisoformat(as_of)
        if cutoff > date.today():
            raise ValueError("as_of cannot be in the future")

        snapshot = load_csmar_universe(self.settings.csmar_cache_dir, as_of=cutoff)
        kwargs: dict[str, Any] = {
            "as_of": snapshot.data_cutoff,
            "holding_weeks": holding_weeks,
        }
        batch = build_weekly_portfolios(snapshot.histories, snapshot.metadata, **kwargs)
        portfolio = batch.for_profile("balanced")

        response: dict[str, Any] = {
            "status": portfolio.status.value,
            "data_cutoff": (
                None
                if batch.data_cutoff is None
                else pd.Timestamp(batch.data_cutoff).date().isoformat()
            ),
            "holding_weeks": holding_weeks,
            "profile": "balanced_single_objective",
            "universe": {
                "master_symbols": snapshot.master_symbols,
                "active_symbols": snapshot.active_symbols,
                "eligible_symbols": snapshot.eligible_symbols,
                "excluded_symbols": snapshot.excluded_symbols,
            },
            "factor_coverage": _jsonable(batch.factor_coverage),
            "market_regime": _jsonable(batch.market_regime),
            "warnings": list(portfolio.risk_warnings),
            "disclaimer": portfolio.disclaimer,
            "raw_data_exposed": False,
        }
        if portfolio.status != WeeklyPortfolioStatus.READY or portfolio.allocation is None:
            response["reasons"] = list(portfolio.reasons)
            response["portfolio"] = None
            return response

        selected = {item.symbol: item for item in portfolio.selected}
        response["portfolio"] = {
            "cash_ratio": portfolio.allocation.cash_ratio,
            "margin_debt_ratio": portfolio.allocation.margin_debt_ratio,
            "positions": [
                {
                    "rank": rank,
                    "symbol": position.ticker,
                    "name": selected[position.ticker].name,
                    "industry": selected[position.ticker].industry,
                    "weight": position.weight,
                    "ranking_score_not_probability": selected[position.ticker].score,
                }
                for rank, position in enumerate(portfolio.allocation.positions, start=1)
            ],
            "historical_risk": _jsonable(portfolio.historical_risk),
            "historical_scenario": _jsonable(portfolio.historical_scenario),
        }
        return response

    def get_latest_research(self, *, run_type: str = "weekly_portfolios") -> dict[str, Any]:
        """Read the newest immutable research run without modifying SQLite."""

        self._require_derived_result_permission()
        normalized = run_type.strip().lower()
        if normalized not in ALLOWED_RUN_TYPES:
            raise ValueError(
                "run_type must be stock_analysis, weekly_portfolios, or limit_watchlist"
            )
        if not self.settings.research_db.is_file():
            return {
                "status": "unavailable",
                "reason": "research_archive_missing",
                "raw_data_exposed": False,
            }
        try:
            with _read_only_connection(self.settings.research_db) as connection:
                run = connection.execute(
                    """
                    SELECT id, run_type, as_of, data_cutoff, created_at,
                           strategy_version, model_id, status, warning_json
                    FROM runs
                    WHERE run_type = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (normalized,),
                ).fetchone()
                if run is None:
                    return {
                        "status": "unavailable",
                        "reason": "no_matching_research_run",
                        "run_type": normalized,
                        "raw_data_exposed": False,
                    }
                return _read_run_bundle(connection, run)
        except sqlite3.DatabaseError:
            return {
                "status": "unavailable",
                "reason": "research_archive_schema_unavailable",
                "run_type": normalized,
                "raw_data_exposed": False,
            }

    def _require_derived_result_permission(self) -> None:
        if not self.settings.allow_licensed_derived_results:
            raise PermissionError(
                "MCP derived-result access is disabled. Confirm your data rights, then set "
                "ASHARE_MCP_ALLOW_LICENSED_DERIVED_RESULTS=true in the local environment."
            )


def _read_only_connection(path: Path) -> sqlite3.Connection:
    encoded = quote(str(path.resolve()), safe="/")
    connection = sqlite3.connect(f"file:{encoded}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _read_run_bundle(connection: sqlite3.Connection, run: sqlite3.Row) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "status": "ready",
        "run": {
            "id": run["id"],
            "run_type": run["run_type"],
            "as_of": run["as_of"],
            "data_cutoff": run["data_cutoff"],
            "created_at": run["created_at"],
            "strategy_version": run["strategy_version"],
            "model_id": run["model_id"],
            "archive_status": run["status"],
            "warnings": _decode_json(run["warning_json"], fallback=[]),
        },
        "raw_data_exposed": False,
    }
    if run["run_type"] == "weekly_portfolios":
        sets = connection.execute(
            """
            SELECT id, risk_profile, cash_weight, borrowed_weight, expected_return,
                   expected_vol, expected_max_drawdown, sharpe, metric_window
            FROM portfolio_sets
            WHERE run_id = ?
            ORDER BY risk_profile
            """,
            (run["id"],),
        ).fetchall()
        portfolios: list[dict[str, Any]] = []
        for portfolio in sets:
            members = connection.execute(
                """
                SELECT symbol, weight, rank, reason_json
                FROM portfolio_members
                WHERE portfolio_id = ?
                ORDER BY rank
                """,
                (portfolio["id"],),
            ).fetchall()
            portfolios.append(
                {
                    "risk_profile": portfolio["risk_profile"],
                    "cash_weight": portfolio["cash_weight"],
                    "borrowed_weight": portfolio["borrowed_weight"],
                    "expected_return": portfolio["expected_return"],
                    "expected_vol": portfolio["expected_vol"],
                    "expected_max_drawdown": portfolio["expected_max_drawdown"],
                    "sharpe": portfolio["sharpe"],
                    "metric_window": portfolio["metric_window"],
                    "members": [
                        {
                            "symbol": member["symbol"],
                            "weight": member["weight"],
                            "rank": member["rank"],
                            "reason": _decode_json(member["reason_json"], fallback={}),
                        }
                        for member in members
                    ],
                }
            )
        bundle["portfolios"] = portfolios
    elif run["run_type"] == "stock_analysis":
        analyses = connection.execute(
            """
            SELECT symbol, horizon_sessions, trend_state, action_for_empty,
                   action_for_holder, entry_low, entry_high, add_above, reduce_low,
                   reduce_high, invalidation, confidence, rationale_json
            FROM stock_analyses
            WHERE run_id = ?
            ORDER BY symbol, horizon_sessions
            """,
            (run["id"],),
        ).fetchall()
        bundle["analyses"] = [
            {
                "symbol": row["symbol"],
                "horizon_sessions": row["horizon_sessions"],
                "trend_state": row["trend_state"],
                "action_for_empty": row["action_for_empty"],
                "action_for_holder": row["action_for_holder"],
                "entry_low": row["entry_low"],
                "entry_high": row["entry_high"],
                "add_above": row["add_above"],
                "reduce_low": row["reduce_low"],
                "reduce_high": row["reduce_high"],
                "invalidation": row["invalidation"],
                "confidence": row["confidence"],
                "rationale": _decode_json(row["rationale_json"], fallback={}),
            }
            for row in analyses
        ]
    return bundle


def _decode_json(value: object, *, fallback: Any) -> Any:
    if not isinstance(value, str):
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.Timestamp | datetime | date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return str(value)


def build_server(settings: MCPSettings | None = None) -> Any:
    """Create the official Python-SDK MCP server, importing it only on demand."""

    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        raise RuntimeError(
            'MCP support is optional. Install it with: pip install -e ".[mcp]"'
        ) from exc

    active_settings = settings or MCPSettings.from_env()
    tools = ReadOnlyResearchTools(active_settings)
    server = FastMCP(
        "a-share-lab",
        instructions=SERVER_INSTRUCTIONS,
        host=active_settings.host,
        port=active_settings.port,
    )
    read_only = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )

    @server.tool(
        name="get_data_status",
        title="检查A股研究数据状态",
        description=(
            "在生成组合前检查本机CSMAR目录、研究档案和数据许可开关；不返回路径、密钥或原始数据。"
        ),
        annotations=read_only,
    )
    def get_data_status() -> dict[str, Any]:
        return tools.get_data_status()

    @server.tool(
        name="generate_portfolio",
        title="生成中期四股研究组合",
        description=(
            "从本机已规范化的A股数据临时计算3:2:2:1四股研究组合。"
            "只返回精简衍生结果，不归档、不下单；持有期可选1、4、13、26或52周。"
        ),
        annotations=read_only,
    )
    def generate_portfolio(
        as_of: str | None = None,
        holding_weeks: Literal[1, 4, 13, 26, 52] = 13,
    ) -> dict[str, Any]:
        return tools.generate_portfolio(
            as_of=as_of,
            holding_weeks=holding_weeks,
        )

    @server.tool(
        name="get_latest_research",
        title="读取最近一次A股研究结果",
        description=(
            "从本机只读研究档案读取最近一次组合或个股研究。"
            "不返回原始行情、新闻正文、持仓或任何交易操作。"
        ),
        annotations=read_only,
    )
    def get_latest_research(
        run_type: Literal[
            "stock_analysis", "weekly_portfolios", "limit_watchlist"
        ] = "weekly_portfolios",
    ) -> dict[str, Any]:
        return tools.get_latest_research(run_type=run_type)

    return server


def main() -> None:
    settings = MCPSettings.from_env()
    server = build_server(settings)
    server.run(transport=settings.transport)


if __name__ == "__main__":
    main()
