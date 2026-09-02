from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from ashare_lab.services.settle_recommendation_performance import (
    ArchivedEntryPlan,
    ArchivedRecommendationBatch,
    ArchivedRecommendationMember,
    ArchivedRecommendationReport,
    ArchiveNature,
    BatchSettlementStatus,
    CohortNature,
    EntryPlanKind,
    EvaluationMode,
    MemberSettlementStatus,
    ObservationAnchor,
    VerifiedDailyEvidence,
    settle_recommendation_performance,
)

SESSIONS = (
    date(2026, 8, 28),
    date(2026, 8, 31),
    date(2026, 9, 1),
    date(2026, 9, 2),
    date(2026, 9, 3),
    date(2026, 9, 4),
)


def _report(*, reconstructed: bool = False) -> ArchivedRecommendationReport:
    return ArchivedRecommendationReport(
        report_id="report-2026-08-28",
        decision_date=date(2026, 8, 27),
        common_cutoff=date(2026, 8, 27),
        plan_for_date=date(2026, 8, 28),
        archive_nature=(ArchiveNature.RECONSTRUCTED if reconstructed else ArchiveNature.ORIGINAL),
        delivery_accepted=not reconstructed,
    )


def _action_batch() -> ArchivedRecommendationBatch:
    return ArchivedRecommendationBatch(
        batch_id="batch-1w-action",
        report_id="report-2026-08-28",
        holding_weeks=1,
        holding_sessions=5,
        evaluation_mode=EvaluationMode.ACTION_SIMULATION,
        cohort_nature=CohortNature.ACTION_QUALIFIED,
        stock_exposure=0.5,
    )


def _observation_batch() -> ArchivedRecommendationBatch:
    return ArchivedRecommendationBatch(
        batch_id="batch-1w-observation",
        report_id="report-2026-08-28",
        holding_weeks=1,
        holding_sessions=5,
        evaluation_mode=EvaluationMode.RECONSTRUCTED_OBSERVATION,
        cohort_nature=CohortNature.OBSERVATION_ONLY,
        stock_exposure=None,
    )


def _reclaim_member(
    *,
    member_id: str,
    symbol: str,
    trigger: float,
    sleeve: float,
) -> ArchivedRecommendationMember:
    return ArchivedRecommendationMember(
        member_id=member_id,
        batch_id="batch-1w-action",
        symbol=symbol,
        name=symbol,
        operational_stock_sleeve_weight=sleeve,
        operational_account_weight=sleeve * 0.5,
        reference_price=trigger,
        entry_plan=ArchivedEntryPlan(
            kind=EntryPlanKind.RECLAIM,
            trigger_price=trigger,
            invalidation_price=8.0,
        ),
    )


def _frame(
    values: dict[str, list[tuple[float, float, float, float, float]]],
    *,
    sessions: tuple[date, ...] = SESSIONS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, bars in values.items():
        for trade_date, (open_, high, low, close, volume) in zip(sessions, bars, strict=True):
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume_shares": volume,
                    "amount_cny": volume * close,
                }
            )
    return pd.DataFrame(rows)


def _evidence(
    frame: pd.DataFrame,
    *,
    sessions: tuple[date, ...] = SESSIONS,
    covered: tuple[str, ...] = ("600001", "600002"),
    actions: dict[str, frozenset[date]] | None = None,
    suspensions: dict[str, frozenset[date]] | None = None,
) -> VerifiedDailyEvidence:
    return VerifiedDailyEvidence(
        prices=frame,
        session_dates=sessions,
        corporate_action_coverage_symbols=frozenset(covered),
        corporate_action_dates_by_symbol=actions or {},
        suspended_dates_by_symbol=suspensions or {},
    )


def _bars(
    *,
    plan_close: float,
    entry_open: float,
    maturity_close: float,
) -> list[tuple[float, float, float, float, float]]:
    return [
        (plan_close, plan_close + 0.2, plan_close - 0.2, plan_close, 1_000.0),
        (entry_open, entry_open + 0.2, entry_open - 0.2, entry_open, 1_000.0),
        (entry_open, entry_open + 0.2, entry_open - 0.2, entry_open, 1_000.0),
        (entry_open, entry_open + 0.2, entry_open - 0.2, entry_open, 1_000.0),
        (entry_open, entry_open + 0.2, entry_open - 0.2, entry_open, 1_000.0),
        (
            maturity_close,
            maturity_close + 0.2,
            maturity_close - 0.2,
            maturity_close,
            1_000.0,
        ),
    ]


def test_action_simulation_uses_next_open_and_keeps_untriggered_weight_as_cash() -> None:
    members = (
        _reclaim_member(member_id="member-a", symbol="600001", trigger=10.0, sleeve=0.6),
        _reclaim_member(member_id="member-b", symbol="600002", trigger=20.0, sleeve=0.4),
    )
    prices = _frame(
        {
            "600001": _bars(plan_close=11.0, entry_open=12.0, maturity_close=15.0),
            "600002": _bars(plan_close=19.0, entry_open=20.0, maturity_close=15.0),
        }
    )

    result = settle_recommendation_performance(
        report=_report(),
        batch=_action_batch(),
        members=members,
        evidence=_evidence(prices),
    )

    assert result.status is BatchSettlementStatus.SETTLED_PARTIAL_ENTRY
    assert result.maturity_date == date(2026, 9, 4)
    assert result.members[0].entry_date == date(2026, 8, 31)
    assert result.members[0].entry_price == 12.0
    assert result.members[0].unadjusted_price_return == pytest.approx(0.5)
    assert result.members[0].simulated_action_return == pytest.approx(0.25)
    assert result.members[1].status is MemberSettlementStatus.EXPIRED_UNTRIGGERED
    assert result.members[1].unadjusted_price_return == pytest.approx(-0.25)
    assert result.members[1].simulated_action_return is None
    # Published-reference paper return remains distinct from executable simulation.
    assert result.stock_sleeve_return == pytest.approx(0.6 * 0.5 + 0.4 * -0.25)
    assert result.account_return == pytest.approx(0.3 * 0.5 + 0.2 * -0.25)
    assert result.simulated_action_stock_sleeve_return == pytest.approx(0.6 * 0.25)
    assert result.simulated_action_account_return == pytest.approx(0.3 * 0.25)
    assert result.entered_stock_sleeve_weight == pytest.approx(0.6)
    assert result.stock_sleeve_cash_weight == pytest.approx(0.4)
    assert result.account_cash_weight == pytest.approx(0.7)


def test_all_untriggered_action_weights_remain_cash_with_zero_simulated_return() -> None:
    member = _reclaim_member(
        member_id="member-a",
        symbol="600001",
        trigger=20.0,
        sleeve=1.0,
    )
    member = ArchivedRecommendationMember(
        member_id=member.member_id,
        batch_id=member.batch_id,
        symbol=member.symbol,
        name=member.name,
        operational_stock_sleeve_weight=1.0,
        operational_account_weight=0.5,
        entry_plan=member.entry_plan,
        reference_price=member.reference_price,
    )
    prices = _frame({"600001": _bars(plan_close=11.0, entry_open=12.0, maturity_close=15.0)})

    result = settle_recommendation_performance(
        report=_report(),
        batch=_action_batch(),
        members=(member,),
        evidence=_evidence(prices, covered=("600001",)),
    )

    assert result.status is BatchSettlementStatus.NO_ENTRY
    assert result.simulated_action_stock_sleeve_return == pytest.approx(0.0)
    assert result.simulated_action_account_return == pytest.approx(0.0)
    assert result.entered_stock_sleeve_weight == pytest.approx(0.0)
    assert result.stock_sleeve_cash_weight == pytest.approx(1.0)
    assert result.account_cash_weight == pytest.approx(1.0)


def test_five_session_horizon_remains_pending_until_fifth_future_session() -> None:
    template = _reclaim_member(member_id="member-a", symbol="600001", trigger=10.0, sleeve=1.0)
    member = ArchivedRecommendationMember(
        member_id=template.member_id,
        batch_id=template.batch_id,
        symbol=template.symbol,
        name=template.name,
        operational_stock_sleeve_weight=1.0,
        operational_account_weight=0.5,
        entry_plan=template.entry_plan,
        reference_price=template.reference_price,
    )
    prices = _frame({"600001": _bars(plan_close=11.0, entry_open=12.0, maturity_close=15.0)})

    result = settle_recommendation_performance(
        report=_report(),
        batch=_action_batch(),
        members=(member,),
        evidence=_evidence(
            prices.loc[prices["trade_date"] <= date(2026, 9, 3)],
            sessions=SESSIONS[:-1],
            covered=("600001",),
        ),
    )

    assert result.status is BatchSettlementStatus.PENDING
    assert result.maturity_date is None
    assert result.stock_sleeve_return is None
    assert result.members[0].entry_price == 12.0


def test_reconstructed_observation_keeps_reference_and_close_anchors_separate() -> None:
    members = (
        ArchivedRecommendationMember(
            member_id="obs-a",
            batch_id="batch-1w-observation",
            symbol="600001",
            name="A",
            operational_stock_sleeve_weight=0.6,
            operational_account_weight=None,
            observation_anchor=ObservationAnchor.PLAN_SESSION_CLOSE,
        ),
        ArchivedRecommendationMember(
            member_id="obs-b",
            batch_id="batch-1w-observation",
            symbol="600002",
            name="B",
            operational_stock_sleeve_weight=0.4,
            operational_account_weight=None,
            observation_anchor=ObservationAnchor.ARCHIVED_REFERENCE_PRICE,
            reference_price=20.0,
        ),
    )
    prices = _frame(
        {
            "600001": _bars(plan_close=10.0, entry_open=10.0, maturity_close=11.0),
            "600002": _bars(plan_close=19.0, entry_open=19.0, maturity_close=18.0),
        }
    )

    result = settle_recommendation_performance(
        report=_report(reconstructed=True),
        batch=_observation_batch(),
        members=members,
        evidence=_evidence(prices),
    )

    assert result.status is BatchSettlementStatus.SETTLED
    assert result.evaluation_mode is EvaluationMode.RECONSTRUCTED_OBSERVATION
    assert result.members[0].published_reference_price == 10.0
    assert result.members[1].published_reference_price == 20.0
    assert result.stock_sleeve_return == pytest.approx(0.6 * 0.1 + 0.4 * -0.1)
    assert result.account_return is None
    assert result.account_cash_weight is None


@pytest.mark.parametrize(
    ("suspensions", "expected"),
    [
        (
            {"600001": frozenset({date(2026, 9, 4)})},
            MemberSettlementStatus.MATURITY_SESSION_SUSPENDED,
        ),
        ({}, MemberSettlementStatus.MATURITY_PRICE_MISSING),
    ],
)
def test_missing_and_explicitly_suspended_maturity_are_not_conflated(
    suspensions: dict[str, frozenset[date]],
    expected: MemberSettlementStatus,
) -> None:
    member = _reclaim_member(member_id="member-a", symbol="600001", trigger=10.0, sleeve=1.0)
    member = ArchivedRecommendationMember(
        member_id=member.member_id,
        batch_id=member.batch_id,
        symbol=member.symbol,
        name=member.name,
        operational_stock_sleeve_weight=1.0,
        operational_account_weight=0.5,
        entry_plan=member.entry_plan,
        reference_price=member.reference_price,
    )
    prices = _frame({"600001": _bars(plan_close=11.0, entry_open=12.0, maturity_close=15.0)})
    prices = prices.loc[prices["trade_date"] != date(2026, 9, 4)]

    result = settle_recommendation_performance(
        report=_report(),
        batch=_action_batch(),
        members=(member,),
        evidence=_evidence(
            prices,
            covered=("600001",),
            suspensions=suspensions,
        ),
    )

    assert result.status is BatchSettlementStatus.DATA_QUALITY_FAILURE
    assert result.stock_sleeve_return is None
    assert result.members[0].status is expected


@pytest.mark.parametrize(
    ("covered", "actions", "expected"),
    [
        ((), {}, MemberSettlementStatus.CORPORATE_ACTION_EVIDENCE_UNKNOWN),
        (
            ("600001",),
            {"600001": frozenset({date(2026, 9, 2)})},
            MemberSettlementStatus.CORPORATE_ACTION_DETECTED,
        ),
    ],
)
def test_company_action_unknown_and_detected_both_fail_closed_but_remain_distinct(
    covered: tuple[str, ...],
    actions: dict[str, frozenset[date]],
    expected: MemberSettlementStatus,
) -> None:
    member = _reclaim_member(member_id="member-a", symbol="600001", trigger=10.0, sleeve=1.0)
    member = ArchivedRecommendationMember(
        member_id=member.member_id,
        batch_id=member.batch_id,
        symbol=member.symbol,
        name=member.name,
        operational_stock_sleeve_weight=1.0,
        operational_account_weight=0.5,
        entry_plan=member.entry_plan,
        reference_price=member.reference_price,
    )
    prices = _frame({"600001": _bars(plan_close=11.0, entry_open=12.0, maturity_close=15.0)})

    result = settle_recommendation_performance(
        report=_report(),
        batch=_action_batch(),
        members=(member,),
        evidence=_evidence(prices, covered=covered, actions=actions),
    )

    assert result.status is BatchSettlementStatus.DATA_QUALITY_FAILURE
    assert result.members[0].status is expected
    assert result.members[0].unadjusted_price_return is None
    assert result.members[0].raw_unadjusted_price_change == pytest.approx(0.5)
    assert result.raw_unadjusted_stock_sleeve_change == pytest.approx(0.5)
    assert result.raw_unadjusted_account_change == pytest.approx(0.25)


def test_raw_diagnostic_uses_frozen_multi_member_weights_without_reweighting() -> None:
    members = (
        _reclaim_member(member_id="member-a", symbol="600001", trigger=10.0, sleeve=0.6),
        _reclaim_member(member_id="member-b", symbol="600002", trigger=20.0, sleeve=0.4),
    )
    prices = _frame(
        {
            "600001": _bars(plan_close=11.0, entry_open=12.0, maturity_close=15.0),
            "600002": _bars(plan_close=21.0, entry_open=22.0, maturity_close=15.0),
        }
    )

    result = settle_recommendation_performance(
        report=_report(),
        batch=_action_batch(),
        members=members,
        evidence=_evidence(prices, covered=()),
    )

    assert result.status is BatchSettlementStatus.DATA_QUALITY_FAILURE
    assert result.stock_sleeve_return is None
    assert result.account_return is None
    assert [item.raw_unadjusted_price_change for item in result.members] == pytest.approx(
        [0.5, -0.25]
    )
    assert result.raw_unadjusted_stock_sleeve_change == pytest.approx(0.2)
    assert result.raw_unadjusted_account_change == pytest.approx(0.1)


def test_volume_breakout_without_archived_volume_threshold_fails_closed() -> None:
    member = ArchivedRecommendationMember(
        member_id="member-a",
        batch_id="batch-1w-action",
        symbol="600001",
        name="A",
        operational_stock_sleeve_weight=1.0,
        operational_account_weight=0.5,
        entry_plan=ArchivedEntryPlan(
            kind=EntryPlanKind.VOLUME_BREAKOUT,
            trigger_price=10.0,
            confirmation_volume_min=None,
        ),
        reference_price=10.0,
    )
    prices = _frame({"600001": _bars(plan_close=11.0, entry_open=12.0, maturity_close=15.0)})

    result = settle_recommendation_performance(
        report=_report(),
        batch=_action_batch(),
        members=(member,),
        evidence=_evidence(prices, covered=("600001",)),
    )

    assert result.status is BatchSettlementStatus.DATA_QUALITY_FAILURE
    assert result.members[0].status is MemberSettlementStatus.ENTRY_RULE_INCOMPLETE


def test_volume_breakout_replays_archived_amount_threshold() -> None:
    member = ArchivedRecommendationMember(
        member_id="member-a",
        batch_id="batch-1w-action",
        symbol="600001",
        name="A",
        operational_stock_sleeve_weight=1.0,
        operational_account_weight=0.5,
        entry_plan=ArchivedEntryPlan(
            kind=EntryPlanKind.VOLUME_BREAKOUT,
            trigger_price=10.0,
            confirmation_activity_metric="amount_cny",
            confirmation_activity_min=10_500.0,
        ),
        reference_price=10.0,
    )
    prices = _frame({"600001": _bars(plan_close=11.0, entry_open=12.0, maturity_close=15.0)})

    result = settle_recommendation_performance(
        report=_report(),
        batch=_action_batch(),
        members=(member,),
        evidence=_evidence(prices, covered=("600001",)),
    )

    assert result.status is BatchSettlementStatus.SETTLED
    assert result.members[0].condition_triggered is True
    assert result.members[0].simulated_action_return == pytest.approx(0.25)


def test_incomplete_rule_before_due_date_keeps_batch_pending() -> None:
    member = ArchivedRecommendationMember(
        member_id="member-a",
        batch_id="batch-1w-action",
        symbol="600001",
        name="A",
        operational_stock_sleeve_weight=1.0,
        operational_account_weight=0.5,
        entry_plan=ArchivedEntryPlan(
            kind=EntryPlanKind.VOLUME_BREAKOUT,
            trigger_price=10.0,
            confirmation_volume_min=None,
        ),
        reference_price=10.0,
    )
    prices = _frame(
        {
            "600001": _bars(
                plan_close=11.0,
                entry_open=12.0,
                maturity_close=15.0,
            )[:3]
        },
        sessions=SESSIONS[:3],
    )

    result = settle_recommendation_performance(
        report=_report(),
        batch=_action_batch(),
        members=(member,),
        evidence=_evidence(
            prices,
            sessions=SESSIONS[:3],
            covered=("600001",),
        ),
    )

    assert result.maturity_date is None
    assert result.status is BatchSettlementStatus.PENDING


def test_reconstructed_archive_cannot_be_promoted_to_action_performance() -> None:
    member = _reclaim_member(member_id="member-a", symbol="600001", trigger=10.0, sleeve=1.0)
    member = ArchivedRecommendationMember(
        member_id=member.member_id,
        batch_id=member.batch_id,
        symbol=member.symbol,
        name=member.name,
        operational_stock_sleeve_weight=1.0,
        operational_account_weight=0.5,
        entry_plan=member.entry_plan,
        reference_price=member.reference_price,
    )

    with pytest.raises(ValueError, match="accepted original archive"):
        settle_recommendation_performance(
            report=_report(reconstructed=True),
            batch=_action_batch(),
            members=(member,),
            evidence=_evidence(
                _frame(
                    {
                        "600001": _bars(
                            plan_close=11.0,
                            entry_open=12.0,
                            maturity_close=15.0,
                        )
                    }
                ),
                covered=("600001",),
            ),
        )


def test_original_observation_simulation_stays_in_its_own_evaluation_mode() -> None:
    batch = ArchivedRecommendationBatch(
        batch_id="batch-original-observation",
        report_id="report-2026-08-28",
        holding_weeks=1,
        holding_sessions=5,
        evaluation_mode=EvaluationMode.OBSERVATION_SIMULATION,
        cohort_nature=CohortNature.RISK_QUALIFIED,
        stock_exposure=0.5,
    )
    member = ArchivedRecommendationMember(
        member_id="original-obs-a",
        batch_id=batch.batch_id,
        symbol="600001",
        name="A",
        operational_stock_sleeve_weight=1.0,
        operational_account_weight=0.5,
        observation_anchor=ObservationAnchor.ARCHIVED_REFERENCE_PRICE,
        reference_price=10.0,
    )

    result = settle_recommendation_performance(
        report=_report(),
        batch=batch,
        members=(member,),
        evidence=_evidence(
            _frame(
                {
                    "600001": _bars(
                        plan_close=11.0,
                        entry_open=12.0,
                        maturity_close=15.0,
                    )
                }
            ),
            covered=("600001",),
        ),
    )

    assert result.status is BatchSettlementStatus.SETTLED
    assert result.evaluation_mode is EvaluationMode.OBSERVATION_SIMULATION
    assert result.stock_sleeve_return == pytest.approx(0.5)
    assert result.simulated_action_stock_sleeve_return is None
    assert result.account_cash_weight is None
