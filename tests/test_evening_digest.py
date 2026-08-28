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
    CandidateAction,
    ConditionalEntryPlan,
    ConditionalEntryPlanKind,
    MidtermPortfolioResult,
    MidtermPortfolioStatus,
)

CUTOFF = date(2026, 8, 27)
PLAN_DATE = date(2026, 8, 28)


def _cycle():
    return SimpleNamespace(
        label="中期下行｜短线修复反弹",
        confidence=0.875,
        policy=SimpleNamespace(
            entry_strictness=EntryStrictness.DEFENSIVE,
            max_stock_exposure=0.30,
        ),
    )


def _candidate(rank: int, *, sleeve: float):
    symbols = ("SYN001", "SYN002", "SYN003", "SYN004")
    names = ("合成甲", "合成乙", "合成丙", "合成丁")
    return SimpleNamespace(
        rank=rank,
        symbol=symbols[rank - 1],
        name=names[rank - 1],
        action=CandidateAction.WAIT_CONFIRMATION,
        evidence_unknown=("fundamental_evidence_unknown",),
        conditional_entry_plan=None,
        price_observation_plan=ConditionalEntryPlan(
            kind=ConditionalEntryPlanKind.RECLAIM,
            data_cutoff=pd.Timestamp(CUTOFF),
            horizon="一周",
            sessions=5,
            trigger_price=10.0 + rank,
        ),
        observation_stock_sleeve_weight=sleeve,
        operational_stock_sleeve_weight=None,
        operational_account_weight=None,
    )


def _result(weeks: int) -> MidtermPortfolioResult:
    candidates = tuple(
        _candidate(rank, sleeve=sleeve)
        for rank, sleeve in enumerate((0.30, 0.30, 0.20, 0.20), start=1)
    )
    return MidtermPortfolioResult(
        status=MidtermPortfolioStatus.VALIDATION_NOT_READY,
        data_cutoff=pd.Timestamp(CUTOFF),
        holding_weeks=weeks,
        research_candidates=candidates,
        stock_exposure=0.0,
        cash_weight=1.0,
        price_cycle=_cycle(),
        observation_evaluation=SimpleNamespace(),
        observation_rejection_reasons=("holding_period_return_lcb_below_minimum",),
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
        (481, 551),
        (961, 1031),
        (2017, 2087),
    ]
    token_by_weeks = dict(builds)
    assert token_by_weeks[1] is token_by_weeks[2] is token_by_weeks[4]
    assert token_by_weeks[13] is not token_by_weeks[4]
    assert tuple(period.holding_weeks for period in digest.periods) == EVENING_DIGEST_HORIZONS
    assert digest.common_cutoff == CUTOFF
    assert digest.cycle_label == "中期下行｜短线修复反弹"


def test_one_history_group_failure_is_isolated_to_that_period() -> None:
    def loader(_root, *, minimum_sessions, **_kwargs):
        if minimum_sessions == 961:
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
