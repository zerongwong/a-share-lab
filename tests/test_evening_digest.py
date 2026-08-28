from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pandas as pd

from ashare_lab.analytics.cycle_policy import EntryStrictness
from ashare_lab.domain.errors import DataUnavailableError
from ashare_lab.services.build_evening_digest import (
    EVENING_DIGEST_HORIZONS,
    EveningResearchDigest,
    build_evening_research_digest,
    render_evening_digest_bark_compact,
    render_evening_digest_markdown,
)
from ashare_lab.services.build_midterm_portfolio import (
    HOLDING_PERIOD_SESSIONS,
    CandidateAction,
    CandidateExclusion,
    ConditionalEntryPlan,
    ConditionalEntryPlanKind,
    MidtermPortfolioResult,
    MidtermPortfolioStatus,
)

CUTOFF = date(2026, 8, 27)
PLAN_DATE = date(2026, 8, 28)


def _replace_namespace(value: SimpleNamespace, **changes) -> SimpleNamespace:
    return SimpleNamespace(**(vars(value) | changes))


def _cycle():
    return SimpleNamespace(
        label="中期下行｜短线修复反弹",
        confidence=0.875,
        policy=SimpleNamespace(
            entry_strictness=EntryStrictness.DEFENSIVE,
            max_stock_exposure=0.30,
        ),
    )


def _candidate(
    rank: int,
    *,
    sleeve: float,
    weeks: int,
    action: CandidateAction = CandidateAction.WAIT_CONFIRMATION,
):
    symbols = ("SYN001", "SYN002", "SYN003", "SYN004")
    names = ("合成甲", "合成乙", "合成丙", "合成丁")
    return SimpleNamespace(
        rank=rank,
        symbol=symbols[rank - 1],
        name=names[rank - 1],
        action=action,
        evidence_unknown=("fundamental_evidence_unknown",),
        conditional_entry_plan=None,
        price_observation_plan=ConditionalEntryPlan(
            kind=ConditionalEntryPlanKind.RECLAIM,
            data_cutoff=pd.Timestamp(CUTOFF),
            horizon={1: "一周", 2: "两周", 4: "一个月", 13: "三个月", 26: "六个月", 52: "一年"}[
                weeks
            ],
            sessions=HOLDING_PERIOD_SESSIONS[weeks],
            trigger_price=10.0 + rank,
        ),
        observation_stock_sleeve_weight=sleeve,
        operational_stock_sleeve_weight=None,
        operational_account_weight=None,
        timeframe=SimpleNamespace(
            holding_weeks=weeks,
            slow_direction=SimpleNamespace(
                timeframe=SimpleNamespace(
                    value="weekly_completed" if weeks <= 4 else "monthly_completed"
                ),
                direction=SimpleNamespace(value=f"up_{weeks}w"),
            ),
            structure=SimpleNamespace(
                timeframe=SimpleNamespace(value="daily" if weeks <= 4 else "weekly_completed"),
                state=SimpleNamespace(value="volume_confirmed_breakout"),
                breakout_line=10.0 + rank,
            ),
            score=0.80,
            method_version="multi-timeframe-core-v0.2.0",
        ),
    )


def _result(weeks: int, *, portfolio_nature: str = "observation") -> MidtermPortfolioResult:
    action = (
        CandidateAction.CONDITIONAL_ENTRY
        if portfolio_nature == "action"
        else CandidateAction.WAIT_CONFIRMATION
    )
    candidates = tuple(
        _candidate(rank, sleeve=sleeve, weeks=weeks, action=action)
        for rank, sleeve in enumerate((0.30, 0.30, 0.20, 0.20), start=1)
    )
    if portfolio_nature in {"risk", "action"}:
        candidates = tuple(
            _replace_namespace(
                candidate,
                operational_stock_sleeve_weight=candidate.observation_stock_sleeve_weight,
                operational_account_weight=candidate.observation_stock_sleeve_weight * 0.30,
                evidence_unknown=(),
            )
            for candidate in candidates
        )
    return MidtermPortfolioResult(
        status=(
            MidtermPortfolioStatus.RESEARCH_ONLY
            if portfolio_nature == "action"
            else MidtermPortfolioStatus.VALIDATION_NOT_READY
        ),
        data_cutoff=pd.Timestamp(CUTOFF),
        holding_weeks=weeks,
        positions=candidates if portfolio_nature == "action" else (),
        research_candidates=candidates,
        stock_exposure=0.30 if portfolio_nature == "action" else 0.0,
        cash_weight=0.70 if portfolio_nature == "action" else 1.0,
        price_cycle=_cycle(),
        evaluation=SimpleNamespace() if portfolio_nature == "action" else None,
        research_evaluation=(SimpleNamespace() if portfolio_nature in {"risk", "action"} else None),
        observation_evaluation=(SimpleNamespace() if portfolio_nature == "observation" else None),
        observation_rejection_reasons=(
            ("holding_period_return_lcb_below_minimum",)
            if portfolio_nature == "observation"
            else ()
        ),
        horizon_candidate_count=len(candidates),
    )


def _hybrid(token: object):
    snapshot = SimpleNamespace(
        histories={"token": token},
        metadata={},
        data_cutoff=CUTOFF,
        market_index_histories={},
    )
    return SimpleNamespace(snapshot=snapshot, common_cutoff=CUTOFF)


def test_six_horizons_share_only_equal_history_requirements() -> None:
    loads: list[tuple[int, int, object]] = []
    builds: list[tuple[int, object]] = []

    def loader(_root, *, minimum_sessions, history_sessions, **_kwargs):
        token = object()
        loads.append((minimum_sessions, history_sessions, token))
        return _hybrid(token)

    def builder(histories, _metadata, *, holding_weeks, **_kwargs):
        builds.append((holding_weeks, histories["token"]))
        return _result(holding_weeks)

    digest = build_evening_research_digest(
        dataset_root="csmar",
        overlay_root="overlay",
        reference_dataset_root="reference",
        decision_date=CUTOFF,
        _hybrid_loader=loader,
        _portfolio_builder=builder,
    )

    assert [item[:2] for item in loads] == [
        (252, 322),
        (252, 551),
        (280, 1031),
        (540, 2087),
    ]
    token_by_weeks = dict(builds)
    assert token_by_weeks[1] is token_by_weeks[2] is token_by_weeks[4]
    assert token_by_weeks[13] is not token_by_weeks[4]
    assert tuple(period.holding_weeks for period in digest.periods) == EVENING_DIGEST_HORIZONS
    assert digest.common_cutoff == CUTOFF
    assert digest.cycle_label == "中期下行｜短线修复反弹"
    assert digest.central_implementation_status == "partial_multiframe"
    assert digest.multi_timeframe_component_status == "analytics_core_only"
    assert {period.central_implementation_status for period in digest.periods} == {
        "partial_multiframe"
    }


def test_one_history_group_failure_is_isolated_to_that_period() -> None:
    def loader(_root, *, minimum_sessions, **_kwargs):
        if minimum_sessions == 280:
            raise DataUnavailableError("private path must not be copied")
        return _hybrid(object())

    digest = build_evening_research_digest(
        dataset_root="csmar",
        overlay_root="overlay",
        reference_dataset_root="reference",
        decision_date=CUTOFF,
        _hybrid_loader=loader,
        _portfolio_builder=lambda _histories, _metadata, *, holding_weeks, **_kwargs: _result(
            holding_weeks
        ),
    )

    failed = next(period for period in digest.periods if period.holding_weeks == 26)
    assert failed.failure_code == "data_or_quality_gate_failed"
    assert failed.candidates == ()
    assert sum(period.failure_code is None for period in digest.periods) == 5


def test_markdown_is_compact_derived_and_explicitly_non_actionable() -> None:
    digest = build_evening_research_digest(
        dataset_root="csmar",
        overlay_root="overlay",
        reference_dataset_root="reference",
        decision_date=CUTOFF,
        _hybrid_loader=lambda *_args, **_kwargs: _hybrid(object()),
        _portfolio_builder=lambda _histories, _metadata, *, holding_weeks, **_kwargs: _result(
            holding_weeks
        ),
    )
    digest = replace(digest, plan_for_date=PLAN_DATE)

    markdown = render_evening_digest_markdown(digest)

    assert "共同截止日：2026-08-27" in markdown
    assert "实现状态：`partial_multiframe`" in markdown
    assert "组件：`analytics_core_only`" in markdown
    assert "计划适用日：2026-08-28（周五）" in markdown
    assert "下一交易日" not in markdown
    assert "规则一致度，不是未来方向概率" in markdown
    assert "股票敞口上限：30%；最低现金：70%" in markdown
    assert "观察配比30%股票仓（非资金仓位）" in markdown
    assert "收盘站回观察线" in markdown
    assert "触及不等于可买" in markdown
    assert "不自动下单" in markdown
    assert "private path" not in markdown
    assert len(markdown) < 7_800


def test_bark_compact_covers_all_six_horizons_candidates_weights_and_prices() -> None:
    digest = build_evening_research_digest(
        dataset_root="csmar",
        overlay_root="overlay",
        reference_dataset_root="reference",
        decision_date=CUTOFF,
        _hybrid_loader=lambda *_args, **_kwargs: _hybrid(object()),
        _portfolio_builder=lambda _histories, _metadata, *, holding_weeks, **_kwargs: _result(
            holding_weeks
        ),
    )
    compact = render_evening_digest_bark_compact(replace(digest, plan_for_date=PLAN_DATE))

    for label in ("1周", "2周", "1个月", "3个月", "6个月", "1年"):
        assert f"{label}｜" in compact
    assert compact.count("SYN001合成甲") == 6
    assert "仓30%" in compact
    assert "观察站回 ≥ 11.00" in compact
    assert "仓%=股票仓内10%档" in compact
    assert "不自动下单" in compact
    assert len(compact.encode("utf-8")) <= 2_400
    assert "/Users/" not in compact
    assert "SCT" not in compact


def test_no_successful_hybrid_load_fails_without_fabricating_digest() -> None:
    def fail(*_args, **_kwargs):
        raise DataUnavailableError("unavailable")

    try:
        build_evening_research_digest(
            dataset_root="csmar",
            overlay_root="overlay",
            reference_dataset_root="reference",
            decision_date=CUTOFF,
            _hybrid_loader=fail,
        )
    except DataUnavailableError as exc:
        assert "共同截止日" in str(exc)
    else:
        raise AssertionError("all-load failure must not fabricate a report")


def test_digest_contract_contains_no_raw_data_or_order_capability() -> None:
    digest = EveningResearchDigest(
        common_cutoff=CUTOFF,
        decision_date=CUTOFF,
        cycle_label="价格周期数据不可用",
        entry_strictness=EntryStrictness.UNAVAILABLE.value,
        max_stock_exposure=0.0,
        minimum_cash_weight=1.0,
        cycle_rule_agreement=None,
        periods=(),
    )
    assert digest.raw_data_exposed is False
    assert digest.brokerage_connected is False
    assert digest.orders_enabled is False
    assert digest.plan_for_date is None
    assert digest.central_implementation_status == "partial_multiframe"
    assert digest.multi_timeframe_component_status == "analytics_core_only"


def test_markdown_without_verified_plan_date_is_explicitly_non_executable() -> None:
    digest = EveningResearchDigest(
        common_cutoff=CUTOFF,
        decision_date=CUTOFF,
        cycle_label="价格周期数据不可用",
        entry_strictness=EntryStrictness.UNAVAILABLE.value,
        max_stock_exposure=0.0,
        minimum_cash_weight=1.0,
        cycle_rule_agreement=None,
        periods=(),
    )

    markdown = render_evening_digest_markdown(digest)

    assert "尚未通过官方交易日历确认" in markdown
    assert "不得据此执行" in markdown


def test_all_fifteen_candidate_pairs_and_qualified_portfolio_pairs_are_audited() -> None:
    digest = build_evening_research_digest(
        dataset_root="csmar",
        overlay_root="overlay",
        reference_dataset_root="reference",
        decision_date=CUTOFF,
        _hybrid_loader=lambda *_args, **_kwargs: _hybrid(object()),
        _portfolio_builder=lambda _histories, _metadata, *, holding_weeks, **_kwargs: _result(
            holding_weeks,
            portfolio_nature="action",
        ),
    )

    assert len(digest.horizon_overlaps) == 5
    assert len(digest.candidate_pairwise_overlaps) == 15
    assert len(digest.risk_qualified_pairwise_overlaps) == 15
    assert len(digest.action_pairwise_overlaps) == 15
    assert {item.set_nature for item in digest.candidate_pairwise_overlaps} == {"candidate"}
    assert {item.jaccard for item in digest.candidate_pairwise_overlaps} == {1.0}
    assert {item.jaccard for item in digest.risk_qualified_pairwise_overlaps} == {1.0}
    assert {item.jaccard for item in digest.action_pairwise_overlaps} == {1.0}


def test_repeated_symbol_attribution_documents_each_horizon_without_auto_confluence() -> None:
    digest = build_evening_research_digest(
        dataset_root="csmar",
        overlay_root="overlay",
        reference_dataset_root="reference",
        decision_date=CUTOFF,
        _hybrid_loader=lambda *_args, **_kwargs: _hybrid(object()),
        _portfolio_builder=lambda _histories, _metadata, *, holding_weeks, **_kwargs: _result(
            holding_weeks
        ),
    )

    assert len(digest.repeated_symbol_attributions) == 4
    first = digest.repeated_symbol_attributions[0]
    assert first.symbol == "SYN001"
    assert len(first.appearances) == 6
    assert first.independent_gate_count == 6
    assert first.conclusion == "independent_horizon_gates_documented_not_automatic_confluence"
    assert {item.slow_context for item in first.appearances} == {
        "周线/up_1w",
        "周线/up_2w",
        "周线/up_4w",
        "月线/up_13w",
        "月线/up_26w",
        "月线/up_52w",
    }
    assert {item.price_nature for item in first.appearances} == {"observation_only"}

    markdown = render_evening_digest_markdown(digest)
    assert "候选15对" in markdown
    assert "风险合格组合：15对均不可比较" in markdown
    assert "各期独立门已有记录，但重合本身仍不是自动共振" in markdown


def test_repeated_attribution_uses_the_same_audit_candidates_as_candidate_overlap() -> None:
    def builder(_histories, _metadata, *, holding_weeks, **_kwargs):
        result = _result(holding_weeks, portfolio_nature="action")
        if holding_weeks != 2:
            return result
        return replace(result, positions=result.positions[:3])

    digest = build_evening_research_digest(
        dataset_root="csmar",
        overlay_root="overlay",
        reference_dataset_root="reference",
        decision_date=CUTOFF,
        _hybrid_loader=lambda *_args, **_kwargs: _hybrid(object()),
        _portfolio_builder=builder,
    )

    pair = next(
        item
        for item in digest.candidate_pairwise_overlaps
        if (item.left_holding_weeks, item.right_holding_weeks) == (1, 2)
    )
    assert "SYN004" in pair.shared_symbols
    attribution = next(
        item for item in digest.repeated_symbol_attributions if item.symbol == "SYN004"
    )
    assert len(attribution.appearances) == 6
    assert {item.holding_weeks for item in attribution.appearances} == set(EVENING_DIGEST_HORIZONS)


def test_misaligned_price_plan_sessions_fail_closed_for_its_period() -> None:
    def builder(_histories, _metadata, *, holding_weeks, **_kwargs):
        result = _result(holding_weeks)
        if holding_weeks != 13:
            return result
        bad_candidates = tuple(
            _replace_namespace(
                candidate,
                price_observation_plan=replace(candidate.price_observation_plan, sessions=5),
            )
            for candidate in result.research_candidates
        )
        return replace(result, research_candidates=bad_candidates)

    digest = build_evening_research_digest(
        dataset_root="csmar",
        overlay_root="overlay",
        reference_dataset_root="reference",
        decision_date=CUTOFF,
        _hybrid_loader=lambda *_args, **_kwargs: _hybrid(object()),
        _portfolio_builder=builder,
    )

    three_month = next(period for period in digest.periods if period.holding_weeks == 13)
    assert {candidate.price_nature for candidate in three_month.candidates} == {"unavailable"}
    assert {candidate.price_condition for candidate in three_month.candidates} == {
        "—（价格计划与本期不一致）"
    }
    attribution = next(
        item for item in digest.repeated_symbol_attributions if item.symbol == "SYN001"
    )
    three_month_attribution = next(
        item for item in attribution.appearances if item.holding_weeks == 13
    )
    assert three_month_attribution.independent_gate_documented is False


def test_bark_remains_under_budget_with_full_pairwise_overlap_audit() -> None:
    digest = build_evening_research_digest(
        dataset_root="csmar",
        overlay_root="overlay",
        reference_dataset_root="reference",
        decision_date=CUTOFF,
        _hybrid_loader=lambda *_args, **_kwargs: _hybrid(object()),
        _portfolio_builder=lambda _histories, _metadata, *, holding_weeks, **_kwargs: _result(
            holding_weeks,
            portfolio_nature="action",
        ),
    )

    compact = render_evening_digest_bark_compact(replace(digest, plan_for_date=PLAN_DATE))

    assert "候选重合：15对=100%" in compact
    assert "风险组合重合：15对=100%" in compact
    assert "行动组合重合：15对=100%" in compact
    assert "重复≠自动共振" in compact
    assert "/Users/" not in compact
    assert len(compact.encode("utf-8")) <= 2_400


def test_all_three_overlap_layers_keep_fifteen_unavailable_pairs_without_false_zero() -> None:
    def loader(_root, *, minimum_sessions, **_kwargs):
        if minimum_sessions == 280:
            raise DataUnavailableError("private details")
        return _hybrid(object())

    digest = build_evening_research_digest(
        dataset_root="csmar",
        overlay_root="overlay",
        reference_dataset_root="reference",
        decision_date=CUTOFF,
        _hybrid_loader=loader,
        _portfolio_builder=lambda _histories, _metadata, *, holding_weeks, **_kwargs: _result(
            holding_weeks
        ),
    )

    for overlaps in (
        digest.candidate_pairwise_overlaps,
        digest.risk_qualified_pairwise_overlaps,
        digest.action_pairwise_overlaps,
    ):
        assert len(overlaps) == 15
    failed_candidate_pairs = tuple(
        item
        for item in digest.candidate_pairwise_overlaps
        if 26 in {item.left_holding_weeks, item.right_holding_weeks}
    )
    assert len(failed_candidate_pairs) == 5
    assert {item.comparison_status for item in failed_candidate_pairs} == {"unavailable"}
    assert {item.jaccard for item in failed_candidate_pairs} == {None}
    assert all(
        "data_unavailable" in (item.unavailable_reason or "") for item in failed_candidate_pairs
    )
    assert {item.comparison_status for item in digest.risk_qualified_pairwise_overlaps} == {
        "unavailable"
    }


def test_pairwise_left_right_reasons_use_sanitized_upstream_evidence_or_rank() -> None:
    def builder(_histories, _metadata, *, holding_weeks, **_kwargs):
        result = _result(holding_weeks)
        if holding_weeks != 2:
            return result
        return replace(
            result,
            research_candidates=(),
            search_pool_count=0,
            horizon_candidate_count=10,
            exclusions=(
                CandidateExclusion("SYN001", ("slow_weekly_completed:downtrend",)),
                CandidateExclusion("SYN002", ("primary_daily:failed",)),
                CandidateExclusion(
                    "SYN003",
                    ("insufficient_holding_risk_history:available_10;required_20",),
                ),
            ),
        )

    digest = build_evening_research_digest(
        dataset_root="csmar",
        overlay_root="overlay",
        reference_dataset_root="reference",
        decision_date=CUTOFF,
        _hybrid_loader=lambda *_args, **_kwargs: _hybrid(object()),
        _portfolio_builder=builder,
    )

    pair = next(
        item
        for item in digest.candidate_pairwise_overlaps
        if (item.left_holding_weeks, item.right_holding_weeks) == (1, 2)
    )
    assert pair.comparison_status == "comparable"
    assert pair.jaccard == 0.0
    assert {item.symbol: item.reason for item in pair.left_only} == {
        "SYN001": "slow_context_failure",
        "SYN002": "primary_structure_failure",
        "SYN003": "risk_history_unavailable",
        "SYN004": "ranking_not_selected",
    }
    assert pair.right_only == ()
    assert "private" not in render_evening_digest_markdown(digest)


def test_missing_symbol_without_exclusion_or_rank_audit_is_evidence_unavailable() -> None:
    def builder(_histories, _metadata, *, holding_weeks, **_kwargs):
        result = _result(holding_weeks)
        if holding_weeks != 2:
            return result
        return replace(
            result,
            research_candidates=result.research_candidates[:3],
            exclusions=(),
            search_pool_count=3,
            horizon_candidate_count=3,
        )

    digest = build_evening_research_digest(
        dataset_root="csmar",
        overlay_root="overlay",
        reference_dataset_root="reference",
        decision_date=CUTOFF,
        _hybrid_loader=lambda *_args, **_kwargs: _hybrid(object()),
        _portfolio_builder=builder,
    )
    pair = next(
        item
        for item in digest.candidate_pairwise_overlaps
        if (item.left_holding_weeks, item.right_holding_weeks) == (1, 2)
    )
    difference = next(item for item in pair.left_only if item.symbol == "SYN004")
    assert difference.reason == "evidence_unavailable"


def test_risk_and_action_differences_do_not_guess_beyond_available_evidence() -> None:
    def builder(_histories, _metadata, *, holding_weeks, **_kwargs):
        if holding_weeks == 1:
            return _result(holding_weeks, portfolio_nature="action")
        if holding_weeks == 2:
            return _result(holding_weeks, portfolio_nature="risk")
        return _result(holding_weeks)

    digest = build_evening_research_digest(
        dataset_root="csmar",
        overlay_root="overlay",
        reference_dataset_root="reference",
        decision_date=CUTOFF,
        _hybrid_loader=lambda *_args, **_kwargs: _hybrid(object()),
        _portfolio_builder=builder,
    )

    action_pair = next(
        item
        for item in digest.action_pairwise_overlaps
        if (item.left_holding_weeks, item.right_holding_weeks) == (1, 2)
    )
    assert action_pair.comparison_status == "unavailable"
    assert {item.reason for item in action_pair.left_only} == {"price_or_action_not_triggered"}

    risk_pair = next(
        item
        for item in digest.risk_qualified_pairwise_overlaps
        if (item.left_holding_weeks, item.right_holding_weeks) == (1, 4)
    )
    assert risk_pair.comparison_status == "unavailable"
    assert {item.reason for item in risk_pair.left_only} == {"risk_budget_or_lcb_failure"}
    compact = render_evening_digest_bark_compact(replace(digest, plan_for_date=PLAN_DATE))
    assert len(compact.encode("utf-8")) <= 2_400


def test_independent_gate_requires_horizon_sessions_and_expected_frames() -> None:
    def builder(_histories, _metadata, *, holding_weeks, **_kwargs):
        result = _result(holding_weeks)
        if holding_weeks != 13:
            return result
        mismatched = tuple(
            _replace_namespace(
                candidate,
                timeframe=_replace_namespace(
                    candidate.timeframe,
                    holding_weeks=4,
                    slow_direction=_replace_namespace(
                        candidate.timeframe.slow_direction,
                        timeframe=SimpleNamespace(value="weekly_completed"),
                    ),
                ),
            )
            for candidate in result.research_candidates
        )
        return replace(result, research_candidates=mismatched)

    digest = build_evening_research_digest(
        dataset_root="csmar",
        overlay_root="overlay",
        reference_dataset_root="reference",
        decision_date=CUTOFF,
        _hybrid_loader=lambda *_args, **_kwargs: _hybrid(object()),
        _portfolio_builder=builder,
    )

    attribution = next(
        item for item in digest.repeated_symbol_attributions if item.symbol == "SYN001"
    )
    three_month = next(item for item in attribution.appearances if item.holding_weeks == 13)
    assert three_month.independent_gate_documented is False
    assert attribution.independent_gate_count == 5
    assert attribution.conclusion == "repeated_candidate_evidence_incomplete_not_confluence"
