from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from ashare_lab.mcp_server import (
    ALLOWED_HOLDING_WEEKS,
    MCPSettings,
    ReadOnlyResearchTools,
    _conditional_entry_fields,
    _horizon_risk_lcb_fields,
    _multi_timeframe_fields,
    _price_observation_fields,
    build_server,
)


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


def test_data_status_distinguishes_partial_from_current_ready(tmp_path: Path) -> None:
    settings = _settings(tmp_path, allow=False)
    settings.csmar_cache_dir.mkdir(parents=True)
    (settings.csmar_cache_dir / "csmar.duckdb").touch()

    partial = ReadOnlyResearchTools(settings).get_data_status()

    assert partial["status"] == "partial"
    assert partial["ready_for_current_portfolio"] is False

    reference = tmp_path / "reference"
    overlay = tmp_path / "overlay"
    reference.mkdir()
    overlay.mkdir()
    (reference / "csmar_reference.duckdb").touch()
    (overlay / "verified_manifest.parquet").touch()
    ready_settings = MCPSettings(
        csmar_cache_dir=settings.csmar_cache_dir,
        research_db=settings.research_db,
        csmar_reference_dir=reference,
        market_overlay_dir=overlay,
    )

    ready = ReadOnlyResearchTools(ready_settings).get_data_status()

    assert ready["status"] == "ready"
    assert ready["ready_for_current_portfolio"] is True


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
        tools.generate_portfolio(holding_weeks=3)


def test_mcp_allows_the_two_week_horizon() -> None:
    assert frozenset({1, 2, 4, 13, 26, 52}) == ALLOWED_HOLDING_WEEKS


def test_mcp_exposes_horizon_sources_without_raw_bars() -> None:
    fields = _multi_timeframe_fields(
        SimpleNamespace(
            method_version="multi-timeframe-core-v0.2.0",
            implementation_status="analytics_core_only",
            holding_weeks=13,
            data_cutoff=date(2026, 8, 27),
            weekly_cutoff=date(2026, 8, 21),
            monthly_cutoff=date(2026, 7, 31),
            candidate_qualified=True,
            execution_ready=False,
            score=0.812,
            slow_direction=SimpleNamespace(
                timeframe=SimpleNamespace(value="monthly_completed"),
                direction=SimpleNamespace(value="up"),
            ),
            structure=SimpleNamespace(
                timeframe=SimpleNamespace(value="weekly_completed"),
                state=SimpleNamespace(value="near_breakout"),
                breakout_line=12.34,
            ),
            execution=SimpleNamespace(state=SimpleNamespace(value="wait_for_daily_confirmation")),
            contract=SimpleNamespace(relative_strength_sessions=60),
            slow_bar_cutoff=date(2026, 7, 31),
            structure_bar_cutoff=date(2026, 8, 21),
            incomplete_week_excluded=True,
            incomplete_month_excluded=True,
        )
    )["multi_timeframe"]

    assert fields["holding_weeks"] == 13
    assert fields["implementation_status"] == "analytics_core_only"
    assert fields["completed_daily_cutoff"] == "2026-08-27"
    assert fields["completed_weekly_cutoff"] == "2026-08-21"
    assert fields["completed_monthly_cutoff"] == "2026-07-31"
    assert fields["slow_timeframe"] == "monthly_completed"
    assert fields["primary_timeframe"] == "weekly_completed"
    assert fields["daily_execution_state"] == "wait_for_daily_confirmation"
    assert fields["slow_bar_cutoff"] == "2026-07-31"
    assert fields["primary_bar_cutoff"] == "2026-08-21"
    assert "daily" not in fields


def test_mcp_horizon_plan_keeps_price_and_primary_structure_sources_separate() -> None:
    fields = _conditional_entry_fields(
        action=SimpleNamespace(value="conditional_entry"),
        plan=SimpleNamespace(
            kind=SimpleNamespace(value="volume_breakout_close_confirmation"),
            data_cutoff=date(2026, 8, 27),
            horizon="三个月",
            sessions=60,
            trigger_price=15.2,
            price_source_timeframe="daily",
            primary_structure_timeframe="weekly_completed",
            primary_structure_cutoff=date(2026, 8, 21),
            entry_reference_price=15.0,
            primary_structure_reference_price=14.0,
            invalidation_price=13.8,
            reduction_review_price=14.3,
            invalidation_source_timeframe="weekly_completed",
            reduction_review_source_timeframe="weekly_completed",
        ),
        evidence_unknown=(),
        expected_cutoff=date(2026, 8, 27),
        expected_holding_weeks=13,
    )["conditional_entry_plan"]

    assert fields["price_source_timeframe"] == "daily"
    assert fields["primary_structure_timeframe"] == "weekly_completed"
    assert fields["primary_structure_cutoff"] == "2026-08-21"
    assert fields["entry_reference_price"] == pytest.approx(15.0)
    assert fields["primary_structure_reference_price"] == pytest.approx(14.0)
    assert fields["invalidation_price"] == pytest.approx(13.8)
    assert fields["reduction_review_price"] == pytest.approx(14.3)
    assert fields["invalidation_source_timeframe"] == "weekly_completed"
    assert fields["reduction_review_source_timeframe"] == "weekly_completed"


@pytest.mark.parametrize("action", ["wait_confirmation", "observe_only"])
def test_non_actionable_mcp_entry_plan_never_exposes_a_numeric_price(action: str) -> None:
    fields = _conditional_entry_fields(
        action=SimpleNamespace(value=action),
        plan=SimpleNamespace(
            kind=SimpleNamespace(value="volume_breakout_close_confirmation"),
            data_cutoff=date(2026, 8, 26),
            horizon="一周",
            sessions=5,
            trigger_price=99.99,
        ),
        evidence_unknown=(),
        expected_cutoff=date(2026, 8, 26),
        expected_holding_weeks=1,
    )

    assert fields["conditional_entry_plan"] is None
    assert fields["entry_price_condition_label"] == "—（暂未形成可介入价格）"
    assert "99.99" not in fields["entry_price_condition_label"]

    observation = _price_observation_fields(
        plan=SimpleNamespace(
            kind=SimpleNamespace(value="volume_breakout_close_confirmation"),
            data_cutoff=date(2026, 8, 26),
            horizon="一周",
            sessions=5,
            trigger_price=99.99,
            confirmation_rule="close and volume confirmation",
        ),
        expected_cutoff=date(2026, 8, 26),
        expected_holding_weeks=1,
    )
    assert observation["price_observation_plan"]["trigger"] == pytest.approx(99.99)
    assert "99.99" in observation["price_observation_condition_label"]
    assert observation["price_observation_is_actionable"] is False


@pytest.mark.parametrize("serializer", ["conditional", "observation"])
def test_mcp_price_plan_sessions_must_match_selected_horizon(serializer: str) -> None:
    plan = SimpleNamespace(
        kind=SimpleNamespace(value="volume_breakout_close_confirmation"),
        data_cutoff=date(2026, 8, 27),
        horizon="一周",
        sessions=5,
        trigger_price=12.3,
        confirmation_rule="close and volume confirmation",
    )
    if serializer == "conditional":
        fields = _conditional_entry_fields(
            action=SimpleNamespace(value="conditional_entry"),
            plan=plan,
            evidence_unknown=(),
            expected_cutoff=date(2026, 8, 27),
            expected_holding_weeks=13,
        )
        assert fields["conditional_entry_plan"] is None
        assert fields["entry_price_condition_label"] == "—（暂未形成可介入价格）"
    else:
        fields = _price_observation_fields(
            plan=plan,
            expected_cutoff=date(2026, 8, 27),
            expected_holding_weeks=13,
        )
        assert fields["price_observation_plan"] is None
        assert fields["price_observation_condition_label"] == "—（暂无价格观察线）"


def test_mcp_price_plan_rejects_non_integral_session_count() -> None:
    fields = _conditional_entry_fields(
        action=SimpleNamespace(value="conditional_entry"),
        plan=SimpleNamespace(
            kind=SimpleNamespace(value="volume_breakout_close_confirmation"),
            data_cutoff=date(2026, 8, 27),
            horizon="三个月",
            sessions=60.5,
            trigger_price=12.3,
        ),
        evidence_unknown=(),
        expected_cutoff=date(2026, 8, 27),
        expected_holding_weeks=13,
    )

    assert fields["conditional_entry_plan"] is None


def test_mcp_horizon_risk_and_lcb_exposes_only_selected_derived_metrics() -> None:
    fields = _horizon_risk_lcb_fields(
        SimpleNamespace(
            evaluation=SimpleNamespace(
                metrics=SimpleNamespace(
                    holding_period_sessions=60,
                    horizon_rolling_drawdown_window_sessions=60,
                    horizon_rolling_max_drawdown_p90=0.08,
                    holding_period_sample_count=48,
                    holding_period_return_mean=0.025,
                    holding_period_return_lcb=0.011,
                    lcb_confidence=0.90,
                    holding_period_cost_rate=0.002,
                ),
                risk_budget=SimpleNamespace(
                    passed=True,
                    violations=(),
                    horizon_rolling_drawdown_limit=0.14,
                    horizon_rolling_drawdown_passed=True,
                ),
            ),
            observation_rejection_reasons=(),
        ),
        expected_holding_weeks=13,
    )

    assert fields == {
        "available": True,
        "holding_weeks": 13,
        "holding_period_sessions": 60,
        "allocation_nature": "action_research",
        "risk_budget_passed": True,
        "risk_violations": [],
        "holding_period_return_lcb_gate_passed": True,
        "horizon_rolling_drawdown_window_sessions": 60,
        "horizon_rolling_max_drawdown_p90": pytest.approx(0.08),
        "horizon_rolling_drawdown_limit": pytest.approx(0.14),
        "horizon_rolling_drawdown_passed": True,
        "holding_period_sample_count": 48,
        "holding_period_return_mean": pytest.approx(0.025),
        "holding_period_return_lcb": pytest.approx(0.011),
        "lcb_confidence": pytest.approx(0.90),
        "holding_period_cost_rate": pytest.approx(0.002),
    }
    assert "raw_bars" not in fields


def test_mcp_horizon_risk_and_lcb_fails_closed_on_metric_horizon_mismatch() -> None:
    fields = _horizon_risk_lcb_fields(
        SimpleNamespace(
            evaluation=SimpleNamespace(
                metrics=SimpleNamespace(holding_period_sessions=60),
                risk_budget=SimpleNamespace(passed=True, violations=()),
            )
        ),
        expected_holding_weeks=1,
    )

    assert fields["available"] is False
    assert fields["holding_period_sessions"] == 5
    assert fields["reason"] == "holding_period_metric_mismatch"


def test_generated_portfolio_exposes_unambiguous_weight_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = SimpleNamespace(
        histories={},
        metadata={},
        data_cutoff=date(2026, 8, 26),
        market_index_histories={},
        master_symbols=5_000,
        active_symbols=4_900,
        eligible_symbols=4_500,
        excluded_symbols=500,
        balance_sheet_strength_available=True,
        balance_sheet_strength_symbols=4_000,
        balance_sheet_strength_excluded_symbols=500,
        balance_sheet_snapshot_retrieved_at=None,
        balance_sheet_strength_reason=None,
    )
    hybrid = SimpleNamespace(
        snapshot=snapshot,
        historical_baseline_cutoff=date(2026, 8, 24),
        automatic_increment_cutoff=date(2026, 8, 26),
        common_cutoff=date(2026, 8, 26),
        sources=("test",),
    )
    research_candidates = tuple(
        SimpleNamespace(
            rank=rank,
            symbol=symbol,
            name=symbol,
            industry=f"industry-{rank}",
            action=SimpleNamespace(value="wait_confirmation" if rank == 2 else "conditional_entry"),
            action_reasons=(),
            entry_pattern=SimpleNamespace(value="breakout"),
            breakout_line=10.0,
            days_since_breakout=1,
            absolute_return_60=0.1,
            relative_strength_percentile=0.9,
            downside_capture_ratio=0.5,
            research_weight=weight,
            operational_account_weight=operational_weight,
            operational_stock_sleeve_weight=operational_sleeve_weight,
            downside_risk_contribution=0.5,
            signal_score=0.8,
            evidence_unknown=("news_missing",) if rank == 3 else (),
            conditional_entry_plan=SimpleNamespace(
                kind=SimpleNamespace(value="volume_breakout_close_confirmation"),
                data_cutoff=date(2026, 8, 26),
                horizon="一周",
                sessions=5,
                price_low=None,
                price_high=None,
                trigger_price=10.5,
            ),
            price_observation_plan=SimpleNamespace(
                kind=SimpleNamespace(value="volume_breakout_close_confirmation"),
                data_cutoff=date(2026, 8, 26),
                horizon="一周",
                sessions=5,
                price_low=None,
                price_high=None,
                trigger_price=10.5,
                confirmation_rule="close and volume confirmation",
            ),
        )
        for rank, (symbol, weight, operational_weight, operational_sleeve_weight) in enumerate(
            (
                ("600001.SH", 0.075, 0.08, 0.40),
                ("000001.SZ", 0.065, 0.06, 0.30),
                ("300001.SZ", 0.060, 0.06, 0.30),
            ),
            start=1,
        )
    )
    positions = tuple(
        SimpleNamespace(
            rank=rank,
            symbol=symbol,
            name=symbol,
            industry=f"industry-{rank}",
            weight=weight,
            operational_account_weight=operational_weight,
            operational_stock_sleeve_weight=operational_sleeve_weight,
            entry_pattern=SimpleNamespace(value="breakout"),
            breakout_line=10.0,
            days_since_breakout=1,
            signal_score=0.8,
            downside_risk_contribution=0.5,
            evidence_unknown=(),
            conditional_entry_plan=(
                SimpleNamespace(
                    kind=SimpleNamespace(value="volume_breakout_close_confirmation"),
                    data_cutoff=date(2026, 8, 26),
                    horizon="一周",
                    sessions=5,
                    price_low=None,
                    price_high=None,
                    trigger_price=10.5,
                )
                if rank == 1
                else None
            ),
            price_observation_plan=SimpleNamespace(
                kind=SimpleNamespace(value="volume_breakout_close_confirmation"),
                data_cutoff=date(2026, 8, 26),
                horizon="一周",
                sessions=5,
                price_low=None,
                price_high=None,
                trigger_price=10.5,
                confirmation_rule="close and volume confirmation",
            ),
        )
        for rank, (symbol, weight, operational_weight, operational_sleeve_weight) in enumerate(
            (
                ("600001.SH", 0.15, 0.16, 0.40),
                ("000001.SZ", 0.13, 0.12, 0.30),
                ("300001.SZ", 0.12, 0.12, 0.30),
            ),
            start=1,
        )
    )
    portfolio = SimpleNamespace(
        status=SimpleNamespace(value="research_only"),
        data_cutoff=date(2026, 8, 26),
        method_version="test",
        central_implementation_status="partial_multiframe",
        multi_timeframe_component_status="analytics_core_only",
        entry_ready_count=2,
        actionable_candidate_count=2,
        search_pool_count=2,
        evaluated_portfolio_count=1,
        action_evaluated_portfolio_count=1,
        stock_exposure=0.40,
        cash_weight=0.60,
        borrowed_weight=0.0,
        research_stock_exposure=0.20,
        research_cash_weight=0.80,
        market_regime=None,
        index_regime=None,
        price_cycle=None,
        research_candidates=research_candidates,
        positions=positions,
        evaluation=SimpleNamespace(
            metrics=SimpleNamespace(
                holding_period_sessions=5,
                horizon_rolling_drawdown_window_sessions=5,
                horizon_rolling_max_drawdown_p90=0.04,
                holding_period_sample_count=20,
                holding_period_return_mean=0.02,
                holding_period_return_lcb=0.01,
                lcb_confidence=0.90,
                holding_period_cost_rate=0.002,
            ),
            risk_budget=SimpleNamespace(
                passed=True,
                violations=(),
                horizon_rolling_drawdown_limit=0.12,
                horizon_rolling_drawdown_passed=True,
            ),
        ),
        warnings=(),
        reasons=(),
        evidence_review_required=False,
        disclaimer="research only",
    )
    monkeypatch.setattr(
        "ashare_lab.mcp_server.load_hybrid_universe",
        lambda *_args, **_kwargs: hybrid,
    )
    monkeypatch.setattr(
        "ashare_lab.mcp_server.build_midterm_portfolio",
        lambda *_args, **_kwargs: portfolio,
    )

    result = ReadOnlyResearchTools(_settings(tmp_path, allow=True)).generate_portfolio(
        holding_weeks=1
    )

    assert result["central_implementation_status"] == "partial_multiframe"
    assert result["multi_timeframe_component_status"] == "analytics_core_only"
    canonical_allocation = result["action_research_allocation_not_brokerage_position"]
    assert canonical_allocation["weight_basis"] == "total_account_capital"
    assert canonical_allocation["is_brokerage_account_position"] is False
    assert canonical_allocation["stock_sleeve_weight_step"] == pytest.approx(0.10)
    assert result["actual_allocation"]["canonical_field"] == (
        "action_research_allocation_not_brokerage_position"
    )
    assert result["actual_allocation"]["stock_exposure"] == pytest.approx(0.40)
    assert result["horizon_risk_and_lcb"]["available"] is True
    assert result["horizon_risk_and_lcb"]["holding_period_sessions"] == 5
    assert result["horizon_risk_and_lcb"]["holding_period_return_lcb"] == pytest.approx(0.01)

    research = result["research_candidates"][0]
    assert research["research_weight_not_current_holding"] == pytest.approx(0.075)
    assert research["exact_target_weight_of_total_capital"] == pytest.approx(0.075)
    assert research["weight_of_total_capital"] == pytest.approx(0.075)
    assert research["weight_within_stock_sleeve"] == pytest.approx(0.375)
    assert research["operational_weight_of_total_capital"] == pytest.approx(0.08)
    assert research["operational_weight_within_stock_sleeve"] == pytest.approx(0.40)
    assert research["weight_basis"] == "total_account_capital"
    assert research["is_brokerage_account_position"] is False
    assert research["stock_sleeve_weight_step"] == pytest.approx(0.10)
    assert research["conditional_entry_plan"] == {
        "kind": "volume_breakout_close_confirmation",
        "data_cutoff": "2026-08-26",
        "horizon": "一周",
        "sessions": 5,
        "price_low": None,
        "price_high": None,
        "trigger": 10.5,
    }
    assert research["entry_price_condition_label"] == "收盘价 ≥ **10.50元**（放量突破确认）"
    assert result["research_candidates"][1]["conditional_entry_plan"] is None
    assert result["research_candidates"][1]["entry_price_condition_label"] == (
        "—（暂未形成可介入价格）"
    )
    assert result["research_candidates"][1]["price_observation_plan"]["trigger"] == (
        pytest.approx(10.5)
    )
    assert result["research_candidates"][1]["price_observation_is_actionable"] is False
    assert result["research_candidates"][2]["conditional_entry_plan"] is None
    assert "10.50" not in result["research_candidates"][2]["entry_price_condition_label"]

    action_position = result["portfolio"]["positions"][0]
    assert action_position["weight"] == pytest.approx(0.15)
    assert action_position["exact_target_weight_of_total_capital"] == pytest.approx(0.15)
    assert action_position["weight_of_total_capital"] == pytest.approx(0.15)
    assert action_position["weight_within_stock_sleeve"] == pytest.approx(0.375)
    assert action_position["operational_weight_of_total_capital"] == pytest.approx(0.16)
    assert action_position["operational_weight_within_stock_sleeve"] == pytest.approx(0.40)
    assert action_position["weight_basis"] == "total_account_capital"
    assert action_position["is_brokerage_account_position"] is False
    assert action_position["stock_sleeve_weight_step"] == pytest.approx(0.10)
    assert action_position["conditional_entry_plan"]["trigger"] == pytest.approx(10.5)
    assert action_position["entry_price_condition_label"] == (
        "收盘价 ≥ **10.50元**（放量突破确认）"
    )
    assert result["portfolio"]["positions"][1]["conditional_entry_plan"] is None
    assert result["raw_data_exposed"] is False


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
