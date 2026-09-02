from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.analytics.cycle_policy import EntryStrictness
from ashare_lab.services.archive_recommendation_report import (
    ARCHIVE_METHOD_VERSION,
    archive_recommendation_report,
    build_recommendation_archive_bundle,
)
from ashare_lab.services.build_evening_digest import (
    EveningPeriodDigest,
    EveningResearchDigest,
    build_evening_research_digest,
)
from ashare_lab.services.build_midterm_portfolio import (
    CandidateAction,
    ConditionalEntryPlan,
    ConditionalEntryPlanKind,
    MidtermPortfolioResult,
    MidtermPortfolioStatus,
)

CUTOFF = date(2026, 8, 27)
PLAN_DATE = date(2026, 8, 28)


def _action_digest() -> EveningResearchDigest:
    plan = ConditionalEntryPlan(
        kind=ConditionalEntryPlanKind.VOLUME_BREAKOUT,
        data_cutoff=pd.Timestamp(CUTOFF),
        horizon="一周",
        sessions=5,
        trigger_price=10.60,
        confirmation_rule="完整日线收盘突破且量能确认",
        confirmation_activity_metric="amount_cny",
        confirmation_activity_min=10_000_000.0,
        method_version="structured-plan-v1",
        entry_reference_price=10.50,
        invalidation_price=9.80,
    )
    candidate = SimpleNamespace(
        rank=1,
        symbol="600001",
        name="结构样本",
        action=CandidateAction.CONDITIONAL_ENTRY,
        evidence_unknown=(),
        conditional_entry_plan=plan,
        price_observation_plan=None,
        operational_stock_sleeve_weight=1.0,
        operational_account_weight=0.30,
        observation_stock_sleeve_weight=None,
        timeframe=SimpleNamespace(
            holding_weeks=1,
            slow_direction=SimpleNamespace(
                timeframe=SimpleNamespace(value="weekly_completed"),
                direction=SimpleNamespace(value="up"),
            ),
            structure=SimpleNamespace(
                timeframe=SimpleNamespace(value="daily"),
                state=SimpleNamespace(value="volume_confirmed_breakout"),
                breakout_line=10.50,
            ),
            score=0.80,
            method_version="multi-timeframe-v1",
        ),
    )
    cycle = SimpleNamespace(
        label="上行修复",
        confidence=0.75,
        policy=SimpleNamespace(
            entry_strictness=EntryStrictness.STANDARD,
            max_stock_exposure=0.30,
        ),
    )
    result = MidtermPortfolioResult(
        status=MidtermPortfolioStatus.RESEARCH_ONLY,
        data_cutoff=pd.Timestamp(CUTOFF),
        holding_weeks=1,
        positions=(candidate,),
        research_candidates=(candidate,),
        stock_exposure=0.30,
        cash_weight=0.70,
        evaluation=SimpleNamespace(),
        research_evaluation=SimpleNamespace(),
        price_cycle=cycle,
        horizon_candidate_count=1,
    )
    snapshot = SimpleNamespace(
        histories={},
        metadata={},
        data_cutoff=CUTOFF,
        market_index_histories={},
    )
    digest = build_evening_research_digest(
        dataset_root="derived-only",
        overlay_root="derived-only",
        reference_dataset_root="derived-only",
        decision_date=CUTOFF,
        horizons=(1,),
        _hybrid_loader=lambda *_args, **_kwargs: SimpleNamespace(
            snapshot=snapshot,
            common_cutoff=CUTOFF,
        ),
        _portfolio_builder=lambda *_args, **_kwargs: result,
    )
    return replace(digest, plan_for_date=PLAN_DATE)


def _observation_digest(
    *,
    performance_nature: str = "observation_only",
    with_reference_price: bool = True,
) -> EveningResearchDigest:
    digest = _action_digest()
    candidate = digest.periods[0].candidates[0]
    allocation_nature = (
        "risk_qualified_research"
        if performance_nature == "risk_qualified_observation"
        else "observation_only"
    )
    candidate = replace(
        candidate,
        action=CandidateAction.WAIT_CONFIRMATION.value,
        allocation_nature=allocation_nature,
        stock_sleeve_weight=1.0,
        account_weight=None,
        price_nature="observation_only",
        operational_stock_sleeve_weight=None,
        operational_account_weight=None,
        price_plan_evaluation_price=(10.60 if with_reference_price else None),
        price_plan_kind=("volume_breakout_close_confirmation" if with_reference_price else None),
        price_plan_trigger=(10.60 if with_reference_price else None),
    )
    period = replace(
        digest.periods[0],
        action_nature="行动层保持现金；研究观察组合独立跟踪",
        action_stock_exposure=0.0,
        action_cash_weight=1.0,
        candidates=(candidate,),
        performance_nature=performance_nature,
    )
    return replace(digest, periods=(period,))


def test_digest_preserves_structured_plan_without_calling_reference_a_fill() -> None:
    candidate = _action_digest().periods[0].candidates[0]

    assert candidate.operational_stock_sleeve_weight == 1.0
    assert candidate.operational_account_weight == 0.30
    assert candidate.price_plan_kind == "volume_breakout_close_confirmation"
    assert candidate.price_plan_trigger == 10.60
    assert candidate.price_plan_evaluation_price == 10.60
    assert candidate.price_plan_confirmation_rule == "完整日线收盘突破且量能确认"
    assert candidate.price_plan_invalidation_price == 9.80
    assert candidate.price_plan_cutoff == CUTOFF
    assert candidate.price_plan_method_version == "structured-plan-v1"
    assert _action_digest().periods[0].performance_nature == "official_action"


def test_archive_bundle_is_deterministic_derived_and_primary_action_is_explicit() -> None:
    digest = _action_digest()
    first = build_recommendation_archive_bundle(
        digest,
        created_at=datetime(2026, 8, 27, 13, 0, tzinfo=UTC),
    )
    second = build_recommendation_archive_bundle(
        digest,
        created_at=datetime(2026, 8, 27, 13, 5, tzinfo=UTC),
    )

    assert first.content_hash == second.content_hash
    assert first.report_id == second.report_id
    assert first.batches[0]["evaluation_mode"] == "action_simulation"
    assert first.batches[0]["cohort_nature"] == "action_qualified"
    assert first.batches[0]["actionability"] == "primary_action"
    assert first.report["metadata_json"]["archive_method_version"] == ARCHIVE_METHOD_VERSION
    member = first.members[0]
    assert member["operational_stock_sleeve_weight"] == 1.0
    assert member["operational_account_weight"] == 0.30
    assert member["entry_plan_json"]["trigger_price"] == 10.60
    assert member["entry_plan_json"]["confirmation_activity_metric"] == "amount_cny"
    assert member["entry_plan_json"]["confirmation_activity_min"] == 10_000_000.0
    assert member["entry_plan_json"]["evaluation_price"] == 10.60
    assert (
        member["entry_plan_json"]["evaluation_rule"]
        == "complete_daily_close_at_or_above_archived_trigger"
    )
    assert member["entry_plan_json"]["evaluation_price_is_fill_price"] is False
    assert member["metadata_json"]["performance_eligible"] is True

    text = repr((first.report, first.batches, first.members)).lower()
    assert "ohlcv" not in text
    assert "sendkey" not in text
    assert "api.day.app" not in text


def test_reconstruction_and_full_cash_never_enter_primary_action_results() -> None:
    reconstructed = build_recommendation_archive_bundle(
        _action_digest(),
        archive_nature="reconstructed",
        created_at=datetime(2026, 8, 31, 1, 0, tzinfo=UTC),
    )
    assert reconstructed.batches[0]["evaluation_mode"] == "reconstructed_observation"
    assert reconstructed.batches[0]["actionability"] == "research_only"
    assert reconstructed.members[0]["metadata_json"]["performance_eligible"] is False

    full_cash_period = EveningPeriodDigest(
        holding_weeks=1,
        holding_sessions=5,
        label="1周",
        data_cutoff=CUTOFF,
        source_status="no_eligible_portfolio",
        action_nature="行动层保持现金",
        risk_nature="没有合格组合",
        action_stock_exposure=0.0,
        action_cash_weight=1.0,
        performance_nature="full_cash",
    )
    full_cash_digest = replace(_action_digest(), periods=(full_cash_period,))
    full_cash = build_recommendation_archive_bundle(full_cash_digest)
    assert full_cash.batches[0]["evaluation_mode"] == "unavailable"
    assert full_cash.batches[0]["cohort_nature"] == "unavailable"
    assert full_cash.batches[0]["member_count"] == 0


def test_original_observation_cohorts_are_pending_but_separate_from_actions() -> None:
    for nature, expected_cohort in (
        ("risk_qualified_observation", "risk_qualified"),
        ("observation_only", "observation_only"),
    ):
        bundle = build_recommendation_archive_bundle(_observation_digest(performance_nature=nature))
        batch = bundle.batches[0]
        member = bundle.members[0]

        assert batch["evaluation_mode"] == "observation_simulation"
        assert batch["cohort_nature"] == expected_cohort
        assert batch["actionability"] == "research_only"
        assert batch["status"] == "pending"
        assert member["operational_stock_sleeve_weight"] == 1.0
        assert member["operational_account_weight"] is None
        assert member["observation_anchor"] == "archived_reference_price"
        assert member["reference_price"] == 10.60
        assert member["metadata_json"]["performance_eligible"] is False
        assert member["metadata_json"]["observation_performance_eligible"] is True


def test_original_observation_without_reference_uses_plan_session_close() -> None:
    bundle = build_recommendation_archive_bundle(_observation_digest(with_reference_price=False))

    member = bundle.members[0]
    assert member["observation_anchor"] == "plan_session_close"
    assert member["reference_price"] is None
    assert member["operational_stock_sleeve_weight"] == 1.0
    assert member["operational_account_weight"] is None


def test_observation_archive_excludes_display_candidate_without_frozen_weight() -> None:
    digest = _observation_digest()
    weighted = digest.periods[0].candidates[0]
    display_only = replace(
        weighted,
        rank=2,
        symbol="600002",
        name="未入观察组合",
        stock_sleeve_weight=None,
        account_weight=None,
        operational_stock_sleeve_weight=None,
        operational_account_weight=None,
    )
    digest = replace(
        digest,
        periods=(replace(digest.periods[0], candidates=(weighted, display_only)),),
    )

    bundle = build_recommendation_archive_bundle(digest, archive_nature="reconstructed")

    assert bundle.batches[0]["member_count"] == 1
    assert bundle.batches[0]["metadata_json"]["observation_eligible_member_count"] == 1
    assert [member["symbol"] for member in bundle.members] == [weighted.symbol]


def test_archive_service_calls_atomic_repository_boundary() -> None:
    calls: list[tuple[object, tuple[object, ...], tuple[object, ...]]] = []

    class Repository:
        def archive_recommendation_report(
            self,
            report,
            *,
            batches=(),
            members=(),
            delivery_events=(),
        ) -> None:
            assert tuple(delivery_events) == ()
            calls.append((report, tuple(batches), tuple(members)))

    created_at = datetime.now(UTC) - timedelta(minutes=1)
    receipt = archive_recommendation_report(
        _action_digest(),
        Repository(),
        created_at=created_at,
    )

    assert len(calls) == 1
    assert calls[0][0]["id"] == receipt.report_id
    assert len(calls[0][1]) == 1
    assert len(calls[0][2]) == 1


def test_archive_bundle_round_trips_through_sqlite_repository(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "research.db")

    receipt = archive_recommendation_report(
        _action_digest(),
        repository,
        created_at=datetime(2026, 8, 27, 13, 0, tzinfo=UTC),
    )

    report = repository.get_recommendation_report(receipt.report_id)
    assert report is not None
    assert report["content_hash"] == receipt.content_hash
    batch = repository.list_recommendation_batches(receipt.report_id)[0]
    assert batch["evaluation_mode"] == "action_simulation"
    assert batch["cohort_nature"] == "action_qualified"
    assert batch["status"] == "pending"
    member = repository.list_recommendation_members(batch["id"])[0]
    assert member["plan_kind"] == "volume_breakout_close_confirmation"
    assert member["trigger_price"] == 10.60
    assert member["reference_price"] == 10.60
    assert member["confirmation_rule"] == "完整日线收盘突破且量能确认"
    assert member["invalidation_price"] == 9.80
    assert member["plan_cutoff"] == CUTOFF.isoformat()


def test_observation_archive_round_trips_as_separate_pending_cohort(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "research.db")

    receipt = archive_recommendation_report(
        _observation_digest(),
        repository,
        created_at=datetime(2026, 8, 27, 13, 0, tzinfo=UTC),
    )

    batch = repository.list_recommendation_batches(receipt.report_id)[0]
    assert batch["evaluation_mode"] == "observation_simulation"
    assert batch["cohort_nature"] == "observation_only"
    assert batch["status"] == "pending"
    member = repository.list_recommendation_members(batch["id"])[0]
    assert member["stock_sleeve_weight"] == 1.0
    assert member["account_weight"] is None
    assert member["observation_anchor"] == "archived_reference_price"
    assert member["reference_price"] == 10.60
