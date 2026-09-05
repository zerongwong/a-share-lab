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
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from platformdirs import user_data_path

from ashare_lab.analytics.adaptive_portfolio import OPERATION_STOCK_SLEEVE_STEP
from ashare_lab.analytics.multi_timeframe import MULTI_TIMEFRAME_IMPLEMENTATION_STATUS
from ashare_lab.services.build_midterm_portfolio import (
    CENTRAL_MULTI_TIMEFRAME_IMPLEMENTATION_STATUS,
    HOLDING_PERIOD_SESSIONS,
    CandidateAction,
    ConditionalEntryPlanKind,
    build_midterm_portfolio,
    horizon_history_requirements,
)
from ashare_lab.services.load_hybrid_universe import load_hybrid_universe

ALLOWED_HOLDING_WEEKS = frozenset({1, 2, 4, 13, 26, 52})
ALLOWED_RUN_TYPES = frozenset({"stock_analysis", "weekly_portfolios", "limit_watchlist"})

DATA_RIGHTS_NOTICE = (
    "仅返回本机研究计算产生的精简衍生结果，不返回原始行情、财务报表或新闻正文。"
    "启用前请确认数据许可允许将衍生结果交给所连接的模型服务处理。"
)
SERVER_INSTRUCTIONS = (
    "只读A股研究工具。先调用get_data_status确认本地数据和许可开关。"
    "结果是历史研究证据，不是收益承诺或个性化投资建议；不得声称已下单。"
)

TOTAL_ACCOUNT_WEIGHT_BASIS = "total_account_capital"
ACTION_RESEARCH_ALLOCATION_FIELD = "action_research_allocation_not_brokerage_position"
ENTRY_PRICE_UNAVAILABLE_LABEL = "—（暂未形成可介入价格）"
PRICE_OBSERVATION_UNAVAILABLE_LABEL = "—（暂无价格观察线）"
OBSERVATION_WEIGHT_BASIS = "candidate_observation_pool_not_total_account"


def _weight_within_stock_sleeve(
    weight_of_total_capital: float | None,
    stock_exposure: float,
) -> float | None:
    """Convert a total-account weight to a stock-sleeve weight safely."""

    if weight_of_total_capital is None or stock_exposure <= 0.0:
        return None
    ratio = float(weight_of_total_capital) / float(stock_exposure)
    return ratio if np.isfinite(ratio) else None


def _conditional_entry_fields(
    *,
    action: object,
    plan: object,
    evidence_unknown: object,
    expected_cutoff: object,
    expected_holding_weeks: int,
) -> dict[str, Any]:
    """Serialize a horizon-aware daily entry plan without inventing a quote.

    This mirrors the concise UI semantics while failing closed for every
    non-actionable or incomplete row.  ``breakout_line`` deliberately is not an
    input: it remains trend evidence and can never become an MCP entry price.
    """

    action_value = getattr(action, "value", action)
    if (
        action_value != CandidateAction.CONDITIONAL_ENTRY.value
        or bool(evidence_unknown)
        or plan is None
    ):
        return {
            "conditional_entry_plan": None,
            "entry_price_condition_label": ENTRY_PRICE_UNAVAILABLE_LABEL,
        }

    cutoff = _normalized_timestamp(getattr(plan, "data_cutoff", None))
    expected = _normalized_timestamp(expected_cutoff)
    raw_sessions = getattr(plan, "sessions", None)
    try:
        sessions = int(raw_sessions)
    except (AttributeError, TypeError, ValueError):
        sessions = 0
    horizon = getattr(plan, "horizon", None)
    kind = getattr(getattr(plan, "kind", None), "value", getattr(plan, "kind", None))
    expected_sessions = HOLDING_PERIOD_SESSIONS.get(expected_holding_weeks)
    if (
        cutoff is None
        or expected is None
        or cutoff != expected
        or not isinstance(horizon, str)
        or not horizon.strip()
        or expected_sessions is None
        or raw_sessions != expected_sessions
    ):
        return {
            "conditional_entry_plan": None,
            "entry_price_condition_label": ENTRY_PRICE_UNAVAILABLE_LABEL,
        }

    price_low: float | None = None
    price_high: float | None = None
    trigger: float | None = None
    if kind == ConditionalEntryPlanKind.HEALTHY_PULLBACK.value:
        price_low = _positive_entry_price(getattr(plan, "price_low", None))
        price_high = _positive_entry_price(getattr(plan, "price_high", None))
        if price_low is None or price_high is None or price_low > price_high:
            return {
                "conditional_entry_plan": None,
                "entry_price_condition_label": ENTRY_PRICE_UNAVAILABLE_LABEL,
            }
        label = f"回踩至 **{price_low:.2f}–{price_high:.2f}元**"
    elif kind in {
        ConditionalEntryPlanKind.RECLAIM.value,
        ConditionalEntryPlanKind.VOLUME_BREAKOUT.value,
    }:
        trigger = _positive_entry_price(getattr(plan, "trigger_price", None))
        if trigger is None:
            return {
                "conditional_entry_plan": None,
                "entry_price_condition_label": ENTRY_PRICE_UNAVAILABLE_LABEL,
            }
        if kind == ConditionalEntryPlanKind.RECLAIM.value:
            label = f"收盘价 ≥ **{trigger:.2f}元**（重新站回确认）"
        else:
            label = f"收盘价 ≥ **{trigger:.2f}元**（放量突破确认）"
    else:
        return {
            "conditional_entry_plan": None,
            "entry_price_condition_label": ENTRY_PRICE_UNAVAILABLE_LABEL,
        }

    plan_payload: dict[str, Any] = {
        "kind": kind,
        "data_cutoff": cutoff.date().isoformat(),
        "horizon": horizon,
        "sessions": sessions,
        "price_low": price_low,
        "price_high": price_high,
        "trigger": trigger,
    }
    structure_timeframe = getattr(plan, "structure_timeframe", None)
    structure_cutoff = _date_string(getattr(plan, "structure_cutoff", None))
    execution_lookback = getattr(plan, "execution_lookback_sessions", None)
    method_version = getattr(plan, "method_version", None)
    if structure_timeframe is not None:
        plan_payload["structure_timeframe"] = structure_timeframe
    if structure_cutoff is not None:
        plan_payload["structure_cutoff"] = structure_cutoff
    if execution_lookback is not None:
        plan_payload["execution_lookback_sessions"] = execution_lookback
    if method_version is not None:
        plan_payload["method_version"] = method_version
    for field_name in (
        "price_source_timeframe",
        "primary_structure_timeframe",
        "invalidation_source_timeframe",
        "reduction_review_source_timeframe",
    ):
        field_value = getattr(plan, field_name, None)
        if field_value is not None:
            plan_payload[field_name] = field_value
    primary_cutoff = _date_string(getattr(plan, "primary_structure_cutoff", None))
    if primary_cutoff is not None:
        plan_payload["primary_structure_cutoff"] = primary_cutoff
    for field_name in (
        "entry_reference_price",
        "primary_structure_reference_price",
        "invalidation_price",
        "reduction_review_price",
    ):
        field_value = _positive_entry_price(getattr(plan, field_name, None))
        if field_value is not None:
            plan_payload[field_name] = field_value
    return {
        "conditional_entry_plan": plan_payload,
        "entry_price_condition_label": label,
    }


def _price_observation_fields(
    *,
    plan: object,
    expected_cutoff: object,
    expected_holding_weeks: int,
) -> dict[str, Any]:
    """Serialize neutral price levels in a namespace separate from entry permission."""

    serialized = _conditional_entry_fields(
        action=CandidateAction.CONDITIONAL_ENTRY,
        plan=plan,
        evidence_unknown=(),
        expected_cutoff=expected_cutoff,
        expected_holding_weeks=expected_holding_weeks,
    )["conditional_entry_plan"]
    if serialized is None:
        return {
            "price_observation_plan": None,
            "price_observation_condition_label": PRICE_OBSERVATION_UNAVAILABLE_LABEL,
            "price_observation_is_actionable": False,
        }
    kind = serialized["kind"]
    if kind == ConditionalEntryPlanKind.HEALTHY_PULLBACK.value:
        label = f"回踩观察区 **{serialized['price_low']:.2f}–{serialized['price_high']:.2f}元**"
    elif kind == ConditionalEntryPlanKind.RECLAIM.value:
        label = f"收盘站回观察线 ≥ **{serialized['trigger']:.2f}元**"
    else:
        label = f"收盘突破观察线 ≥ **{serialized['trigger']:.2f}元**，并满足期限量能确认"
    return {
        "price_observation_plan": {
            **serialized,
            "confirmation_rule": getattr(plan, "confirmation_rule", None),
        },
        "price_observation_condition_label": label,
        "price_observation_is_actionable": False,
    }


def _normalized_timestamp(value: object) -> pd.Timestamp | None:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tz is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _positive_entry_price(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if np.isfinite(price) and price > 0.0 else None


def _multi_timeframe_fields(assessment: object) -> dict[str, Any]:
    """Return compact derived timeframe evidence without raw bars."""

    if assessment is None:
        return {"multi_timeframe": None}
    contract = getattr(assessment, "contract", None)
    slow = getattr(assessment, "slow_direction", None)
    structure = getattr(assessment, "structure", None)
    execution = getattr(assessment, "execution", None)
    return {
        "multi_timeframe": {
            "method_version": getattr(assessment, "method_version", None),
            "implementation_status": getattr(assessment, "implementation_status", None),
            "holding_weeks": getattr(assessment, "holding_weeks", None),
            "completed_daily_cutoff": _date_string(getattr(assessment, "data_cutoff", None)),
            "completed_weekly_cutoff": _date_string(getattr(assessment, "weekly_cutoff", None)),
            "completed_monthly_cutoff": _date_string(getattr(assessment, "monthly_cutoff", None)),
            "candidate_qualified": bool(getattr(assessment, "candidate_qualified", False)),
            "daily_execution_ready": bool(getattr(assessment, "execution_ready", False)),
            "score_not_probability": getattr(assessment, "score", None),
            "slow_timeframe": getattr(
                getattr(slow, "timeframe", None),
                "value",
                None,
            ),
            "slow_direction": getattr(
                getattr(slow, "direction", None),
                "value",
                None,
            ),
            "slow_bar_cutoff": _date_string(getattr(assessment, "slow_bar_cutoff", None)),
            "primary_timeframe": getattr(
                getattr(structure, "timeframe", None),
                "value",
                None,
            ),
            "primary_structure": getattr(
                getattr(structure, "state", None),
                "value",
                None,
            ),
            "primary_breakout_line": getattr(structure, "breakout_line", None),
            "primary_bar_cutoff": _date_string(getattr(assessment, "structure_bar_cutoff", None)),
            "daily_execution_state": getattr(
                getattr(execution, "state", None),
                "value",
                None,
            ),
            "relative_strength_sessions": getattr(
                contract,
                "relative_strength_sessions",
                None,
            ),
            "incomplete_week_excluded": bool(
                getattr(assessment, "incomplete_week_excluded", False)
            ),
            "incomplete_month_excluded": bool(
                getattr(assessment, "incomplete_month_excluded", False)
            ),
        }
    }


def _derived_field(value: object, name: str, default: object = None) -> object:
    """Read one derived dataclass or mapping field without exposing source data."""

    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _horizon_risk_lcb_fields(
    portfolio: object,
    *,
    expected_holding_weeks: int,
) -> dict[str, Any]:
    """Serialize the selected horizon's derived risk and historical LCB audit.

    Action, risk-qualified research and rejected observation evaluations remain
    distinguishable.  A mismatched holding-period metric fails closed rather
    than being relabelled as the requested horizon.
    """

    expected_sessions = HOLDING_PERIOD_SESSIONS.get(expected_holding_weeks)
    if expected_sessions is None:
        raise ValueError("unsupported holding horizon")

    evaluation = getattr(portfolio, "evaluation", None)
    allocation_nature = "action_research"
    if evaluation is None:
        evaluation = getattr(portfolio, "research_evaluation", None)
        allocation_nature = "risk_qualified_research"
    if evaluation is None:
        evaluation = getattr(portfolio, "observation_evaluation", None)
        allocation_nature = "observation_only"

    base: dict[str, Any] = {
        "available": False,
        "holding_weeks": expected_holding_weeks,
        "holding_period_sessions": expected_sessions,
        "allocation_nature": "unavailable" if evaluation is None else allocation_nature,
        "risk_budget_passed": None,
        "risk_violations": [],
        "holding_period_return_lcb_gate_passed": None,
    }
    if evaluation is None:
        return base

    metrics = getattr(evaluation, "metrics", None)
    risk_budget = getattr(evaluation, "risk_budget", None)
    actual_sessions = _derived_field(metrics, "holding_period_sessions")
    if actual_sessions != expected_sessions:
        return {**base, "reason": "holding_period_metric_mismatch"}

    violations = tuple(_derived_field(risk_budget, "violations", ()) or ())
    risk_passed = bool(_derived_field(risk_budget, "passed", False))
    rejection_reasons = tuple(getattr(portfolio, "observation_rejection_reasons", ()) or ())
    lcb_passed = (
        "holding_period_return_lcb_below_minimum" not in rejection_reasons
        if allocation_nature == "observation_only"
        else True
    )
    return {
        **base,
        "available": True,
        "risk_budget_passed": risk_passed,
        "risk_violations": list(violations),
        "horizon_rolling_drawdown_window_sessions": _derived_field(
            metrics,
            "horizon_rolling_drawdown_window_sessions",
        ),
        "horizon_rolling_max_drawdown_p90": _derived_field(
            metrics,
            "horizon_rolling_max_drawdown_p90",
        ),
        "horizon_rolling_drawdown_limit": _derived_field(
            risk_budget,
            "horizon_rolling_drawdown_limit",
        ),
        "horizon_rolling_drawdown_passed": _derived_field(
            risk_budget,
            "horizon_rolling_drawdown_passed",
        ),
        "holding_period_sample_count": _derived_field(
            metrics,
            "holding_period_sample_count",
        ),
        "holding_period_return_mean": _derived_field(
            metrics,
            "holding_period_return_mean",
        ),
        "holding_period_return_lcb": _derived_field(
            metrics,
            "holding_period_return_lcb",
        ),
        "lcb_confidence": _derived_field(metrics, "lcb_confidence"),
        "holding_period_cost_rate": _derived_field(
            metrics,
            "holding_period_cost_rate",
        ),
        "holding_period_return_lcb_gate_passed": lcb_passed,
    }


def _date_string(value: object) -> str | None:
    timestamp = _normalized_timestamp(value)
    return None if timestamp is None else timestamp.date().isoformat()


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
    csmar_reference_dir: Path | None = None
    allow_licensed_derived_results: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    transport: str = "streamable-http"
    market_overlay_dir: Path | None = None

    @classmethod
    def from_env(cls) -> MCPSettings:
        default_data_dir = Path(user_data_path("A股研究助手", appauthor=False))
        cache_dir = Path(
            os.getenv("ASHARE_CSMAR_CACHE_DIR", str(default_data_dir / "cache" / "csmar"))
        ).expanduser()
        research_db = Path(
            os.getenv("ASHARE_RESEARCH_DB", str(default_data_dir / "research.db"))
        ).expanduser()
        reference_dir = Path(
            os.getenv(
                "ASHARE_CSMAR_REFERENCE_DIR",
                str(default_data_dir / "cache" / "csmar_reference"),
            )
        ).expanduser()
        overlay_dir = Path(
            os.getenv(
                "ASHARE_MARKET_OVERLAY_DIR",
                str(default_data_dir / "cache" / "market_overlay"),
            )
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
            csmar_reference_dir=reference_dir.resolve(),
            market_overlay_dir=overlay_dir.resolve(),
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
        reference_available = bool(
            self.settings.csmar_reference_dir is not None
            and (self.settings.csmar_reference_dir / "csmar_reference.duckdb").is_file()
        )
        overlay_root = (
            self.settings.market_overlay_dir
            if self.settings.market_overlay_dir is not None
            else self.settings.csmar_cache_dir.parent / "market_overlay"
        )
        overlay_available = (overlay_root / "verified_manifest.parquet").is_file()
        ready_for_current_portfolio = (
            catalog_available and reference_available and overlay_available
        )
        return {
            "status": (
                "ready"
                if ready_for_current_portfolio
                else ("partial" if catalog_available else "unavailable")
            ),
            "ready_for_current_portfolio": ready_for_current_portfolio,
            "csmar_catalog_available": catalog_available,
            "research_archive_available": self.settings.research_db.is_file(),
            "core_index_pit_catalog_available": reference_available,
            "balance_sheet_current_snapshot_available": reference_available,
            "balance_sheet_historical_pit_available": False,
            "verified_daily_overlay_available": overlay_available,
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
        mode: Literal["live", "historical"] = "live",
    ) -> dict[str, Any]:
        """Compute a portfolio from the local catalog without archiving it."""

        self._require_derived_result_permission()
        if holding_weeks not in ALLOWED_HOLDING_WEEKS:
            allowed = ", ".join(str(item) for item in sorted(ALLOWED_HOLDING_WEEKS))
            raise ValueError(f"holding_weeks must be one of: {allowed}")
        if mode not in {"live", "historical"}:
            raise ValueError("mode must be live or historical")
        shanghai_today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        cutoff = shanghai_today if as_of is None else date.fromisoformat(as_of)
        if cutoff > shanghai_today:
            raise ValueError("as_of cannot be in the future")
        if mode == "live" and cutoff != shanghai_today:
            raise ValueError("live mode requires today's Shanghai decision date")

        reference_root = (
            self.settings.csmar_reference_dir
            if self.settings.csmar_reference_dir is not None
            and (self.settings.csmar_reference_dir / "csmar_reference.duckdb").is_file()
            else None
        )
        history_requirements = horizon_history_requirements(holding_weeks)
        hybrid = load_hybrid_universe(
            self.settings.csmar_cache_dir,
            overlay_root=(
                self.settings.market_overlay_dir
                if self.settings.market_overlay_dir is not None
                else self.settings.csmar_cache_dir.parent / "market_overlay"
            ),
            as_of=cutoff,
            minimum_sessions=history_requirements.qualification_minimum_sessions,
            history_sessions=history_requirements.history_read_sessions,
            minimum_qualification_sessions=(history_requirements.qualification_minimum_sessions),
            reference_dataset_root=reference_root,
            decision_date=cutoff,
            mode=mode,
        )
        snapshot = hybrid.snapshot
        portfolio = build_midterm_portfolio(
            snapshot.histories,
            snapshot.metadata,
            as_of=snapshot.data_cutoff,
            holding_weeks=holding_weeks,
            market_index_histories=(
                snapshot.market_index_histories if reference_root is not None else None
            ),
        )
        research_action_by_symbol = {
            candidate.symbol: candidate.action for candidate in portfolio.research_candidates
        }

        action_research_allocation = {
            "stock_exposure": portfolio.stock_exposure,
            "cash_weight": portfolio.cash_weight,
            "borrowed_weight": portfolio.borrowed_weight,
            "stock_sleeve_weight_step": OPERATION_STOCK_SLEEVE_STEP,
            "weight_basis": TOTAL_ACCOUNT_WEIGHT_BASIS,
            "is_brokerage_account_position": False,
        }
        legacy_actual_allocation = {
            **action_research_allocation,
            "legacy_field_name": True,
            "canonical_field": ACTION_RESEARCH_ALLOCATION_FIELD,
        }
        response: dict[str, Any] = {
            "status": portfolio.status.value,
            "data_cutoff": (
                None
                if portfolio.data_cutoff is None
                else pd.Timestamp(portfolio.data_cutoff).date().isoformat()
            ),
            "holding_weeks": holding_weeks,
            "profile": "adaptive_3_to_5_maintrend",
            "method_version": portfolio.method_version,
            "central_implementation_status": getattr(
                portfolio,
                "central_implementation_status",
                CENTRAL_MULTI_TIMEFRAME_IMPLEMENTATION_STATUS,
            ),
            "multi_timeframe_component_status": getattr(
                portfolio,
                "multi_timeframe_component_status",
                MULTI_TIMEFRAME_IMPLEMENTATION_STATUS,
            ),
            "mode": mode,
            "data_lineage": {
                "historical_baseline_cutoff": (hybrid.historical_baseline_cutoff.isoformat()),
                "automatic_increment_cutoff": (
                    None
                    if hybrid.automatic_increment_cutoff is None
                    else hybrid.automatic_increment_cutoff.isoformat()
                ),
                "common_cutoff": hybrid.common_cutoff.isoformat(),
                "sources": list(hybrid.sources),
            },
            "universe": {
                "master_symbols": snapshot.master_symbols,
                "active_symbols": snapshot.active_symbols,
                "eligible_symbols": snapshot.eligible_symbols,
                "excluded_symbols": snapshot.excluded_symbols,
            },
            "entry_ready_count": portfolio.entry_ready_count,
            "horizon_candidate_count": getattr(
                portfolio,
                "horizon_candidate_count",
                portfolio.entry_ready_count,
            ),
            "actionable_candidate_count": portfolio.actionable_candidate_count,
            "search_pool_count": portfolio.search_pool_count,
            "evaluated_portfolio_count": portfolio.evaluated_portfolio_count,
            "action_evaluated_portfolio_count": (portfolio.action_evaluated_portfolio_count),
            "horizon_risk_and_lcb": _horizon_risk_lcb_fields(
                portfolio,
                expected_holding_weeks=holding_weeks,
            ),
            # Compatibility-only alias.  The canonical field below makes clear
            # that this is a research action layer, not a brokerage position.
            "actual_allocation": legacy_actual_allocation,
            ACTION_RESEARCH_ALLOCATION_FIELD: action_research_allocation,
            "research_allocation_not_current_holding": {
                "stock_exposure": portfolio.research_stock_exposure,
                "cash_weight": portfolio.research_cash_weight,
                "borrowed_weight": 0.0,
                "stock_sleeve_weight_step": OPERATION_STOCK_SLEEVE_STEP,
                "weight_basis": TOTAL_ACCOUNT_WEIGHT_BASIS,
                "is_brokerage_account_position": False,
            },
            "candidate_observation_allocation": {
                "available": getattr(portfolio, "observation_evaluation", None) is not None,
                "risk_gate_passed": False,
                "risk_violations": list(getattr(portfolio, "observation_rejection_reasons", ())),
                "stock_sleeve_weight_step": OPERATION_STOCK_SLEEVE_STEP,
                "weight_basis": OBSERVATION_WEIGHT_BASIS,
                "is_total_account_allocation": False,
                "is_actionable": False,
            },
            "market_regime": _jsonable(portfolio.market_regime),
            "index_regime": _jsonable(portfolio.index_regime),
            "price_cycle": _jsonable(portfolio.price_cycle),
            "research_candidates": [
                {
                    "rank": candidate.rank,
                    "symbol": candidate.symbol,
                    "name": candidate.name,
                    "industry": candidate.industry,
                    "action": candidate.action.value,
                    "action_reasons": list(candidate.action_reasons),
                    "entry_pattern": candidate.entry_pattern.value,
                    "breakout_line": candidate.breakout_line,
                    "days_since_breakout": candidate.days_since_breakout,
                    "legacy_sixty_session_absolute_return": candidate.absolute_return_60,
                    "sixty_session_absolute_return": candidate.absolute_return_60,
                    "horizon_absolute_return": getattr(
                        candidate,
                        "horizon_absolute_return",
                        None,
                    ),
                    "horizon_return_sessions": getattr(
                        candidate,
                        "relative_strength_sessions",
                        None,
                    ),
                    "full_universe_horizon_relative_strength_percentile": (
                        candidate.relative_strength_percentile
                    ),
                    "full_universe_relative_strength_60_percentile": (
                        candidate.relative_strength_percentile
                        if getattr(
                            getattr(getattr(candidate, "timeframe", None), "contract", None),
                            "relative_strength_sessions",
                            60,
                        )
                        == 60
                        else None
                    ),
                    "downside_capture_ratio": candidate.downside_capture_ratio,
                    "research_weight_not_current_holding": candidate.research_weight,
                    "exact_target_weight_of_total_capital": candidate.research_weight,
                    "weight_of_total_capital": candidate.research_weight,
                    "weight_within_stock_sleeve": _weight_within_stock_sleeve(
                        candidate.research_weight,
                        portfolio.research_stock_exposure,
                    ),
                    "operational_weight_of_total_capital": (candidate.operational_account_weight),
                    "operational_weight_within_stock_sleeve": (
                        candidate.operational_stock_sleeve_weight
                    ),
                    "observation_weight_within_candidate_pool": getattr(
                        candidate,
                        "observation_stock_sleeve_weight",
                        None,
                    ),
                    "observation_weight_basis": OBSERVATION_WEIGHT_BASIS,
                    "observation_allocation_is_actionable": False,
                    "stock_sleeve_weight_step": OPERATION_STOCK_SLEEVE_STEP,
                    "weight_basis": TOTAL_ACCOUNT_WEIGHT_BASIS,
                    "is_brokerage_account_position": False,
                    "research_downside_risk_contribution": (candidate.downside_risk_contribution),
                    "ranking_score_not_probability": candidate.signal_score,
                    "evidence_unknown": list(candidate.evidence_unknown),
                    **_conditional_entry_fields(
                        action=candidate.action,
                        plan=getattr(candidate, "conditional_entry_plan", None),
                        evidence_unknown=candidate.evidence_unknown,
                        expected_cutoff=portfolio.data_cutoff,
                        expected_holding_weeks=holding_weeks,
                    ),
                    **_price_observation_fields(
                        plan=getattr(candidate, "price_observation_plan", None),
                        expected_cutoff=portfolio.data_cutoff,
                        expected_holding_weeks=holding_weeks,
                    ),
                    **_multi_timeframe_fields(getattr(candidate, "timeframe", None)),
                }
                for candidate in portfolio.research_candidates
            ],
            "balance_sheet_strength": {
                "enabled": snapshot.balance_sheet_strength_available,
                "provided": snapshot.balance_sheet_strength_symbols,
                "excluded": snapshot.balance_sheet_strength_excluded_symbols,
                "retrieved_at": (
                    None
                    if snapshot.balance_sheet_snapshot_retrieved_at is None
                    else snapshot.balance_sheet_snapshot_retrieved_at.isoformat()
                ),
                "reason": snapshot.balance_sheet_strength_reason,
                "historical_point_in_time": False,
            },
            "warnings": list(portfolio.warnings),
            "evidence_review_required": portfolio.evidence_review_required,
            "disclaimer": portfolio.disclaimer,
            "raw_data_exposed": False,
        }
        if not portfolio.positions or portfolio.evaluation is None:
            response["reasons"] = list(portfolio.reasons)
            response["portfolio"] = None
            return response

        if portfolio.reasons:
            response["reasons"] = list(portfolio.reasons)
        response["portfolio"] = {
            "evidence_complete": not portfolio.evidence_review_required,
            "final_buy_list": False,
            "is_brokerage_account_position": False,
            "weight_basis": TOTAL_ACCOUNT_WEIGHT_BASIS,
            "stock_count": len(portfolio.positions),
            "stock_exposure": portfolio.stock_exposure,
            "cash_ratio": portfolio.cash_weight,
            "margin_debt_ratio": 0.0,
            "stock_sleeve_weight_step": OPERATION_STOCK_SLEEVE_STEP,
            "positions": [
                {
                    "rank": position.rank,
                    "symbol": position.symbol,
                    "name": position.name,
                    "industry": position.industry,
                    "weight": position.weight,
                    "exact_target_weight_of_total_capital": position.weight,
                    "weight_of_total_capital": position.weight,
                    "weight_within_stock_sleeve": _weight_within_stock_sleeve(
                        position.weight,
                        portfolio.stock_exposure,
                    ),
                    "operational_weight_of_total_capital": (position.operational_account_weight),
                    "operational_weight_within_stock_sleeve": (
                        position.operational_stock_sleeve_weight
                    ),
                    "stock_sleeve_weight_step": OPERATION_STOCK_SLEEVE_STEP,
                    "weight_basis": TOTAL_ACCOUNT_WEIGHT_BASIS,
                    "is_brokerage_account_position": False,
                    "entry_pattern": position.entry_pattern.value,
                    "breakout_line": position.breakout_line,
                    "days_since_breakout": position.days_since_breakout,
                    "horizon_absolute_return": getattr(
                        position,
                        "horizon_absolute_return",
                        None,
                    ),
                    "horizon_return_sessions": getattr(
                        position,
                        "relative_strength_sessions",
                        None,
                    ),
                    "ranking_score_not_probability": position.signal_score,
                    "downside_risk_contribution": position.downside_risk_contribution,
                    "evidence_unknown": list(position.evidence_unknown),
                    **_conditional_entry_fields(
                        action=research_action_by_symbol.get(
                            position.symbol,
                            CandidateAction.WAIT_CONFIRMATION,
                        ),
                        plan=getattr(position, "conditional_entry_plan", None),
                        evidence_unknown=position.evidence_unknown,
                        expected_cutoff=portfolio.data_cutoff,
                        expected_holding_weeks=holding_weeks,
                    ),
                    **_price_observation_fields(
                        plan=getattr(position, "price_observation_plan", None),
                        expected_cutoff=portfolio.data_cutoff,
                        expected_holding_weeks=holding_weeks,
                    ),
                    **_multi_timeframe_fields(getattr(position, "timeframe", None)),
                }
                for position in portfolio.positions
            ],
            "historical_downside_risk": _jsonable(portfolio.evaluation.metrics),
            "risk_budget": _jsonable(portfolio.evaluation.risk_budget),
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
        title="生成旧版期限研究对照组合",
        description=(
            "旧版固定期限研究对照，不是当前持续持仓的补位或微信行动计划。"
            "从本机已规范化的A股数据临时计算3至5股主升趋势研究组合和自动权重。"
            "只返回精简衍生结果，不归档、不下单；持有期可选1、2、4、13、26或52周。"
        ),
        annotations=read_only,
    )
    def generate_portfolio(
        as_of: str | None = None,
        holding_weeks: Literal[1, 2, 4, 13, 26, 52] = 13,
        mode: Literal["live", "historical"] = "live",
    ) -> dict[str, Any]:
        return tools.generate_portfolio(
            as_of=as_of,
            holding_weeks=holding_weeks,
            mode=mode,
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
