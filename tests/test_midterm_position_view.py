from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pandas as pd
import pytest

from ashare_lab.services.build_midterm_portfolio import (
    CandidateAction,
    ConditionalEntryPlan,
    ConditionalEntryPlanKind,
)
from ashare_lab.ui.midterm_position_view import (
    MidtermPositionView,
    build_midterm_position_views,
)

CUTOFF = pd.Timestamp("2026-08-27")


def _candidate(
    symbol: str,
    rank: int,
    weight: float | None,
    *,
    operational_weight: float | None = None,
    operational_sleeve_weight: float | None = None,
    observation_sleeve_weight: float | None = None,
    action: CandidateAction = CandidateAction.WAIT_CONFIRMATION,
    plan: ConditionalEntryPlan | None = None,
    observation_plan: ConditionalEntryPlan | None = None,
    evidence_unknown: tuple[str, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        rank=rank,
        name=f"股票{symbol}",
        research_weight=weight,
        operational_account_weight=operational_weight,
        operational_stock_sleeve_weight=operational_sleeve_weight,
        observation_stock_sleeve_weight=observation_sleeve_weight,
        action=action,
        conditional_entry_plan=plan,
        price_observation_plan=observation_plan,
        evidence_unknown=evidence_unknown,
    )


def _result(
    *,
    candidates: tuple[SimpleNamespace, ...],
    positions: tuple[SimpleNamespace, ...] = (),
    research_exposure: float = 0.30,
    action_exposure: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        research_candidates=candidates,
        positions=positions,
        research_stock_exposure=research_exposure,
        stock_exposure=action_exposure,
        data_cutoff=CUTOFF,
    )


def _plan(
    kind: ConditionalEntryPlanKind,
    *,
    low: float | None = None,
    high: float | None = None,
    trigger: float | None = None,
) -> ConditionalEntryPlan:
    return ConditionalEntryPlan(
        kind=kind,
        data_cutoff=CUTOFF,
        horizon="一周",
        sessions=5,
        price_low=low,
        price_high=high,
        trigger_price=trigger,
    )


def test_position_view_has_exactly_three_concise_display_cells() -> None:
    result = _result(
        candidates=(
            _candidate("600001", 1, 0.09, operational_weight=0.09, operational_sleeve_weight=0.30),
            _candidate("600002", 2, 0.07, operational_weight=0.06, operational_sleeve_weight=0.20),
            _candidate("600003", 3, 0.06, operational_weight=0.06, operational_sleeve_weight=0.20),
            _candidate("600004", 4, 0.05, operational_weight=0.06, operational_sleeve_weight=0.20),
            _candidate("600005", 5, 0.03, operational_weight=0.03, operational_sleeve_weight=0.10),
        )
    )

    rows = build_midterm_position_views(result)

    assert [field.name for field in fields(MidtermPositionView)] == [
        "stock_label",
        "planned_weight_label",
        "entry_price_condition",
    ]
    assert "600001" in rows[0].stock_label
    assert "股票仓内 **30%**" in rows[0].planned_weight_label
    assert "对应总资金 **9.0%**" in rows[0].planned_weight_label


def test_waiting_candidates_show_no_entry_price_or_separate_action_state() -> None:
    result = _result(
        candidates=(
            _candidate("600001", 1, 0.30, operational_weight=0.30, operational_sleeve_weight=1.0),
        )
    )

    row = build_midterm_position_views(result)[0]

    assert row.entry_price_condition == "—（暂未形成可介入价格）"


def test_action_weight_is_still_validated_by_symbol_not_rank() -> None:
    result = _result(
        candidates=(
            _candidate(
                "600001",
                1,
                0.11,
                operational_weight=0.12,
                operational_sleeve_weight=0.40,
                action=CandidateAction.CONDITIONAL_ENTRY,
            ),
            _candidate(
                "600002",
                2,
                0.10,
                operational_weight=0.09,
                operational_sleeve_weight=0.30,
                action=CandidateAction.CONDITIONAL_ENTRY,
            ),
            _candidate(
                "600003",
                3,
                0.09,
                operational_weight=0.09,
                operational_sleeve_weight=0.30,
                action=CandidateAction.CONDITIONAL_ENTRY,
            ),
        ),
        positions=(
            SimpleNamespace(
                symbol="600002",
                operational_account_weight=0.09,
                operational_stock_sleeve_weight=0.30,
            ),
            SimpleNamespace(
                symbol="600001",
                operational_account_weight=0.12,
                operational_stock_sleeve_weight=0.40,
            ),
            SimpleNamespace(
                symbol="600003",
                operational_account_weight=0.09,
                operational_stock_sleeve_weight=0.30,
            ),
        ),
        action_exposure=0.30,
    )

    rows = build_midterm_position_views(result)

    assert "40%" in rows[0].planned_weight_label
    assert "30%" in rows[1].planned_weight_label


def test_duplicate_action_symbol_fails_closed() -> None:
    result = _result(
        candidates=(
            _candidate(
                "600001",
                1,
                0.30,
                operational_weight=0.30,
                operational_sleeve_weight=1.0,
                action=CandidateAction.CONDITIONAL_ENTRY,
            ),
        ),
        positions=(
            SimpleNamespace(
                symbol="600001",
                operational_account_weight=0.18,
                operational_stock_sleeve_weight=0.60,
            ),
            SimpleNamespace(
                symbol="600001",
                operational_account_weight=0.12,
                operational_stock_sleeve_weight=0.40,
            ),
        ),
        action_exposure=0.30,
    )

    with pytest.raises(ValueError, match="duplicate action position"):
        build_midterm_position_views(result)


def test_missing_research_weight_remains_unavailable_not_zero() -> None:
    result = _result(
        candidates=(_candidate("600001", 1, None),),
        research_exposure=0.0,
    )

    row = build_midterm_position_views(result)[0]

    assert row.planned_weight_label == "—（尚未形成风险合格权重）"
    assert row.entry_price_condition == "—（暂未形成可介入价格）"


def test_rejected_portfolio_shows_observation_ratio_without_total_account_weight() -> None:
    result = _result(
        candidates=(
            _candidate("600001", 1, None, observation_sleeve_weight=0.40),
            _candidate("600002", 2, None, observation_sleeve_weight=0.30),
            _candidate("600003", 3, None, observation_sleeve_weight=0.20),
            _candidate("600004", 4, None, observation_sleeve_weight=0.10),
        ),
        research_exposure=0.0,
    )

    rows = build_midterm_position_views(result)

    assert "候选观察配比 **40%**" in rows[0].planned_weight_label
    assert "非资金仓位｜风险门未通过" in rows[0].planned_weight_label
    assert "总资金" not in rows[0].planned_weight_label


@pytest.mark.parametrize(
    ("action", "prefix"),
    (
        (CandidateAction.WAIT_CONFIRMATION, "价格观察"),
        (CandidateAction.OBSERVE_ONLY, "仅观察"),
    ),
)
def test_wait_and_observe_show_numeric_observation_line_but_not_entry_permission(
    action: CandidateAction,
    prefix: str,
) -> None:
    observation = _plan(ConditionalEntryPlanKind.VOLUME_BREAKOUT, trigger=12.30)
    result = _result(
        candidates=(
            _candidate(
                "600001",
                1,
                None,
                observation_sleeve_weight=1.0,
                action=action,
                observation_plan=observation,
            ),
        ),
        research_exposure=0.0,
    )

    label = build_midterm_position_views(result)[0].entry_price_condition

    assert prefix in label
    assert "12.30元" in label
    assert "成交量不低于20日中位数1.2倍" in label
    assert "触及不等于可买" in label


def test_invalid_weight_fails_closed() -> None:
    result = _result(
        candidates=(
            _candidate(
                "600001", 1, 0.30, operational_weight=0.07, operational_sleeve_weight=0.2333333333
            ),
            _candidate(
                "600002", 2, None, operational_weight=0.14, operational_sleeve_weight=0.4666666667
            ),
            _candidate("600003", 3, None, operational_weight=0.09, operational_sleeve_weight=0.30),
        )
    )

    with pytest.raises(ValueError, match="10% grid"):
        build_midterm_position_views(result)


@pytest.mark.parametrize(
    ("plan", "expected"),
    (
        (
            _plan(ConditionalEntryPlanKind.HEALTHY_PULLBACK, low=10.10, high=10.60),
            "回踩至 **10.10–10.60元**",
        ),
        (
            _plan(ConditionalEntryPlanKind.RECLAIM, trigger=11.20),
            "收盘价 ≥ **11.20元**（重新站回确认）",
        ),
        (
            _plan(ConditionalEntryPlanKind.VOLUME_BREAKOUT, trigger=12.30),
            "收盘价 ≥ **12.30元**（放量突破确认）",
        ),
    ),
)
def test_conditional_entry_plan_is_formatted_as_a_price_condition(
    plan: ConditionalEntryPlan,
    expected: str,
) -> None:
    result = _result(
        candidates=(
            _candidate(
                "600001",
                1,
                0.30,
                operational_weight=0.30,
                operational_sleeve_weight=1.0,
                action=CandidateAction.CONDITIONAL_ENTRY,
                plan=plan,
            ),
        )
    )

    assert build_midterm_position_views(result)[0].entry_price_condition == expected


def test_unknown_evidence_never_displays_an_entry_price() -> None:
    result = _result(
        candidates=(
            _candidate(
                "600001",
                1,
                0.30,
                operational_weight=0.30,
                operational_sleeve_weight=1.0,
                action=CandidateAction.CONDITIONAL_ENTRY,
                plan=_plan(ConditionalEntryPlanKind.VOLUME_BREAKOUT, trigger=12.30),
                evidence_unknown=("fundamental",),
            ),
        )
    )

    assert (
        build_midterm_position_views(result)[0].entry_price_condition == "—（暂未形成可介入价格）"
    )
