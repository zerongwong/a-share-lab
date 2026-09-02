"""Build one compact, provider-neutral evening research digest.

The service deliberately has no notification, credential, brokerage, order or
state-writing capability.  It loads only verified hybrid snapshots through an
injected/read-only loader, runs the existing deterministic mid-term model and
returns a small derived report.  Raw CSMAR/overlay frames never enter the
returned object.

History requirements differ materially by holding period.  The builder groups
periods that share the same qualification/read-depth contract, holds only one
such hybrid snapshot at a time, and then releases it before loading the next
group.  Deep holding-period risk history is read for finalist validation but
never reused as a full-market coverage gate.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ashare_lab.analytics.cycle_policy import EntryStrictness
from ashare_lab.analytics.multi_timeframe import MULTI_TIMEFRAME_IMPLEMENTATION_STATUS
from ashare_lab.domain.errors import AShareLabError, DataUnavailableError
from ashare_lab.ports.notifications import MAX_COMPACT_NOTIFICATION_BODY_BYTES
from ashare_lab.services.build_midterm_portfolio import (
    CENTRAL_MULTI_TIMEFRAME_IMPLEMENTATION_STATUS,
    HOLDING_PERIOD_SESSIONS,
    CandidateAction,
    ConditionalEntryPlan,
    ConditionalEntryPlanKind,
    MidtermPortfolioResult,
    MidtermPortfolioStatus,
    build_midterm_portfolio,
    horizon_history_requirements,
)
from ashare_lab.services.load_hybrid_universe import (
    HybridUniverseLoad,
    load_hybrid_universe,
)
from ashare_lab.services.review_active_holdings import (
    HoldingAction,
    HoldingReviewRowStatus,
    HoldingReviewSummaryStatus,
    HoldingTreeReviewRow,
    HoldingTreeReviewSummary,
)

EVENING_DIGEST_HORIZONS = (1, 2, 4, 13, 26, 52)
EVENING_DIGEST_METHOD_VERSION = "evening-six-horizon-digest-v0.4.0"

_HORIZON_LABELS = {
    1: "1周",
    2: "2周",
    4: "1个月",
    13: "3个月",
    26: "6个月",
    52: "1年",
}
_ENTRY_STRICTNESS_LABELS = {
    EntryStrictness.STANDARD.value: "标准确认",
    EntryStrictness.TIGHT.value: "加强确认",
    EntryStrictness.DEFENSIVE.value: "防守确认",
    EntryStrictness.EXCEPTION_ONLY.value: "例外级确认",
    EntryStrictness.UNAVAILABLE.value: "不可用",
}
_ACTION_LABELS = {
    CandidateAction.CONDITIONAL_ENTRY.value: "条件介入研究",
    CandidateAction.WAIT_CONFIRMATION.value: "等待确认",
    CandidateAction.OBSERVE_ONLY.value: "仅观察",
}
_REJECTION_LABELS = {
    "annual_downside_volatility": "年化下行波动超限",
    "rolling_drawdown_60_p90": "60日滚动回撤超限",
    "horizon_rolling_drawdown_p90": "本持有期滚动回撤超限",
    "es95_5d": "5日预期损失超限",
    "down_period_correlation": "下跌期相关性超限",
    "position_downside_risk_contribution": "单股下行风险贡献超限",
    "industry_concentration": "行业集中度超限",
    "holding_period_return_lcb_below_minimum": "历史持有期收益下界未达门槛",
}
_TIMEFRAME_LABELS = {
    "daily": "日线",
    "weekly_completed": "周线",
    "monthly_completed": "月线",
}
_STRUCTURE_LABELS = {
    "near_breakout": "临近突破",
    "healthy_post_breakout_pullback": "健康回踩",
    "volume_confirmed_breakout": "突破确认",
    "base_not_yet_near_breakout": "底座形成",
    "trend_continuation_without_entry_structure": "趋势延续",
    "failed": "结构失效",
    "insufficient": "证据不足",
}


@dataclass(frozen=True, slots=True)
class EveningDigestCandidate:
    """One derived candidate row; never a brokerage position or order."""

    rank: int
    symbol: str
    name: str
    action: str
    allocation_nature: Literal[
        "action_research",
        "risk_qualified_research",
        "observation_only",
        "unavailable",
    ]
    stock_sleeve_weight: float | None
    account_weight: float | None
    price_condition: str
    price_nature: Literal["conditional_entry", "observation_only", "unavailable"]
    evidence_pending: bool
    slow_timeframe: str | None = None
    slow_direction: str | None = None
    primary_timeframe: str | None = None
    primary_structure: str | None = None
    primary_breakout_line: float | None = None
    multi_timeframe_score: float | None = None
    multi_timeframe_method_version: str | None = None
    timeframe_holding_weeks: int | None = None
    price_plan_sessions: int | None = None
    # Structured, derived plan fields are archived for later point-in-time
    # outcome evaluation.  They deliberately remain separate from the display
    # string above: ``price_plan_evaluation_price`` is a reference used to
    # evaluate the condition, never evidence that an order was filled there.
    operational_stock_sleeve_weight: float | None = None
    operational_account_weight: float | None = None
    price_plan_kind: str | None = None
    price_plan_low: float | None = None
    price_plan_high: float | None = None
    price_plan_trigger: float | None = None
    price_plan_evaluation_price: float | None = None
    price_plan_confirmation_rule: str | None = None
    price_plan_confirmation_activity_metric: str | None = None
    price_plan_confirmation_activity_min: float | None = None
    price_plan_invalidation_price: float | None = None
    price_plan_cutoff: date | None = None
    price_plan_method_version: str | None = None


DifferenceReason = Literal[
    "data_unavailable",
    "slow_context_failure",
    "primary_structure_failure",
    "risk_history_unavailable",
    "risk_budget_or_lcb_failure",
    "price_or_action_not_triggered",
    "ranking_not_selected",
    "evidence_unavailable",
]


@dataclass(frozen=True, slots=True)
class PairwiseSymbolDifference:
    """Why one derived set contains a symbol while the peer set does not."""

    symbol: str
    name: str
    side: Literal["left_only", "right_only"]
    reason: DifferenceReason


@dataclass(frozen=True, slots=True)
class SymbolExclusionDigest:
    """Sanitized exclusion categories; never contains raw source rows or exceptions."""

    symbol: str
    categories: tuple[DifferenceReason, ...]


@dataclass(frozen=True, slots=True)
class HorizonOverlapDigest:
    """Adjacent-horizon overlap; high values require independent evidence review."""

    left_holding_weeks: int
    right_holding_weeks: int
    left_label: str
    right_label: str
    shared_symbols: tuple[str, ...]
    union_count: int
    jaccard: float | None
    set_nature: Literal["candidate", "risk_qualified", "action"] = "candidate"
    comparison_status: Literal["comparable", "unavailable"] = "comparable"
    unavailable_reason: str | None = None
    left_only: tuple[PairwiseSymbolDifference, ...] = ()
    right_only: tuple[PairwiseSymbolDifference, ...] = ()


@dataclass(frozen=True, slots=True)
class HorizonSymbolDifference:
    """One repeated symbol's derived evidence in one independent horizon."""

    holding_weeks: int
    label: str
    independent_gate_documented: bool
    slow_context: str
    primary_structure: str
    action: str
    allocation_nature: str
    stock_sleeve_weight: float | None
    price_nature: str
    price_condition: str


@dataclass(frozen=True, slots=True)
class RepeatedSymbolAttribution:
    """Compact, non-probabilistic attribution for a cross-horizon repeat."""

    symbol: str
    name: str
    appearances: tuple[HorizonSymbolDifference, ...]
    independent_gate_count: int
    conclusion: Literal[
        "independent_horizon_gates_documented_not_automatic_confluence",
        "repeated_candidate_evidence_incomplete_not_confluence",
    ]


@dataclass(frozen=True, slots=True)
class EveningPeriodDigest:
    """One holding-period result in the six-horizon report."""

    holding_weeks: int
    holding_sessions: int
    label: str
    data_cutoff: date | None
    source_status: str
    action_nature: str
    risk_nature: str
    action_stock_exposure: float
    action_cash_weight: float
    candidates: tuple[EveningDigestCandidate, ...] = ()
    failure_code: str | None = None
    audit_candidates: tuple[EveningDigestCandidate, ...] = ()
    exclusion_categories: tuple[SymbolExclusionDigest, ...] = ()
    exclusion_audit_available: bool = False
    portfolio_failure_categories: tuple[DifferenceReason, ...] = ()
    ranked_pool_count: int | None = None
    central_implementation_status: str = CENTRAL_MULTI_TIMEFRAME_IMPLEMENTATION_STATUS
    multi_timeframe_component_status: str = MULTI_TIMEFRAME_IMPLEMENTATION_STATUS
    performance_nature: Literal[
        "official_action",
        "risk_qualified_observation",
        "observation_only",
        "full_cash",
        "data_unavailable",
    ] = "full_cash"


@dataclass(frozen=True, slots=True)
class EveningResearchDigest:
    """Compact derived report safe for a personal notification channel."""

    common_cutoff: date
    decision_date: date
    cycle_label: str
    entry_strictness: str
    max_stock_exposure: float
    minimum_cash_weight: float
    cycle_rule_agreement: float | None
    periods: tuple[EveningPeriodDigest, ...]
    method_version: str = EVENING_DIGEST_METHOD_VERSION
    raw_data_exposed: bool = False
    brokerage_connected: bool = False
    orders_enabled: bool = False
    plan_for_date: date | None = None
    horizon_overlaps: tuple[HorizonOverlapDigest, ...] = ()
    candidate_pairwise_overlaps: tuple[HorizonOverlapDigest, ...] = ()
    risk_qualified_pairwise_overlaps: tuple[HorizonOverlapDigest, ...] = ()
    action_pairwise_overlaps: tuple[HorizonOverlapDigest, ...] = ()
    repeated_symbol_attributions: tuple[RepeatedSymbolAttribution, ...] = ()
    central_implementation_status: str = CENTRAL_MULTI_TIMEFRAME_IMPLEMENTATION_STATUS
    multi_timeframe_component_status: str = MULTI_TIMEFRAME_IMPLEMENTATION_STATUS


HybridLoader = Callable[..., HybridUniverseLoad]
PortfolioBuilder = Callable[..., MidtermPortfolioResult]


def build_evening_research_digest(
    *,
    dataset_root: str | Path,
    overlay_root: str | Path,
    reference_dataset_root: str | Path,
    decision_date: date,
    horizons: Iterable[int] = EVENING_DIGEST_HORIZONS,
    _hybrid_loader: HybridLoader = load_hybrid_universe,
    _portfolio_builder: PortfolioBuilder = build_midterm_portfolio,
) -> EveningResearchDigest:
    """Build six horizon summaries while keeping only one large snapshot live.

    Expected data/history failures are isolated to the affected requirement
    group.  Unexpected programming errors still escape so a caller cannot send
    a plausible-looking but fabricated report.
    """

    requested = tuple(horizons)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("horizons must be unique and non-empty")
    unsupported = [weeks for weeks in requested if weeks not in HOLDING_PERIOD_SESSIONS]
    if unsupported:
        raise ValueError(f"unsupported holding horizons: {unsupported}")
    if not isinstance(decision_date, date):
        raise TypeError("decision_date must be a date")

    requirement_groups: dict[tuple[int, int], list[int]] = {}
    for weeks in requested:
        requirements = horizon_history_requirements(weeks)
        key = (
            requirements.qualification_minimum_sessions,
            requirements.history_read_sessions,
        )
        requirement_groups.setdefault(key, []).append(weeks)

    period_by_weeks: dict[int, EveningPeriodDigest] = {}
    loaded_cutoff_by_weeks: dict[int, date] = {}
    cycle_by_weeks: dict[int, Any] = {}

    # Each group is an intentionally short-lived cache.  Keeping all four
    # large snapshots would multiply memory use without improving evidence.
    for (minimum_sessions, history_sessions), group_weeks in requirement_groups.items():
        try:
            hybrid = _hybrid_loader(
                dataset_root,
                overlay_root=overlay_root,
                as_of=decision_date,
                minimum_sessions=minimum_sessions,
                history_sessions=history_sessions,
                minimum_qualification_sessions=minimum_sessions,
                reference_dataset_root=reference_dataset_root,
                decision_date=decision_date,
                mode="live",
            )
        except (AShareLabError, FileNotFoundError, OSError, ValueError) as exc:
            failure_code = _failure_code(exc)
            for weeks in group_weeks:
                period_by_weeks[weeks] = _data_unavailable_period(weeks, failure_code)
            continue

        snapshot = hybrid.snapshot
        group_cutoff = _as_date(hybrid.common_cutoff)
        for weeks in group_weeks:
            loaded_cutoff_by_weeks[weeks] = group_cutoff
            try:
                result = _portfolio_builder(
                    snapshot.histories,
                    snapshot.metadata,
                    as_of=snapshot.data_cutoff,
                    holding_weeks=weeks,
                    market_index_histories=snapshot.market_index_histories,
                )
            except (AShareLabError, ValueError) as exc:
                period_by_weeks[weeks] = _data_unavailable_period(
                    weeks,
                    _failure_code(exc),
                    data_cutoff=group_cutoff,
                )
                continue
            period_by_weeks[weeks] = _summarize_period(result, weeks=weeks)
            cycle_by_weeks[weeks] = result.price_cycle

        # Do not retain this group's CSMAR frames while the next, much longer
        # evidence window is loaded.  The compact period summaries remain.
        del snapshot
        del hybrid

    if not loaded_cutoff_by_weeks:
        raise DataUnavailableError("没有任何持有周期取得已验证共同截止日")

    common_cutoff = _select_common_cutoff(loaded_cutoff_by_weeks)
    for weeks, cutoff in loaded_cutoff_by_weeks.items():
        if cutoff != common_cutoff:
            period_by_weeks[weeks] = _data_unavailable_period(
                weeks,
                "common_cutoff_mismatch",
                data_cutoff=cutoff,
            )
            cycle_by_weeks.pop(weeks, None)

    cycle = _select_cycle(cycle_by_weeks, common_cutoff, period_by_weeks)
    if cycle is None:
        cycle_label = "价格周期数据不可用"
        strictness = EntryStrictness.UNAVAILABLE.value
        max_exposure = 0.0
        minimum_cash = 1.0
        agreement = None
    else:
        cycle_label = _clean_text(getattr(cycle, "label", "价格周期数据不可用"))
        policy = getattr(cycle, "policy", None)
        strictness_value = getattr(
            getattr(policy, "entry_strictness", EntryStrictness.UNAVAILABLE),
            "value",
            getattr(policy, "entry_strictness", EntryStrictness.UNAVAILABLE.value),
        )
        strictness = str(strictness_value)
        max_exposure = _finite_fraction(getattr(policy, "max_stock_exposure", 0.0))
        minimum_cash = 1.0 - max_exposure
        agreement = _optional_fraction(getattr(cycle, "confidence", None))

    periods = tuple(period_by_weeks[weeks] for weeks in requested)
    central_statuses = {period.central_implementation_status for period in periods}
    component_statuses = {period.multi_timeframe_component_status for period in periods}
    if len(central_statuses) != 1 or len(component_statuses) != 1:
        raise ValueError("six-horizon implementation status mismatch")
    candidate_overlaps = _build_pairwise_overlaps(periods, set_nature="candidate")
    adjacent_pairs = {
        (left.holding_weeks, right.holding_weeks)
        for left, right in zip(periods, periods[1:], strict=False)
    }
    return EveningResearchDigest(
        common_cutoff=common_cutoff,
        decision_date=decision_date,
        cycle_label=cycle_label,
        entry_strictness=strictness,
        max_stock_exposure=max_exposure,
        minimum_cash_weight=minimum_cash,
        cycle_rule_agreement=agreement,
        periods=periods,
        central_implementation_status=next(iter(central_statuses)),
        multi_timeframe_component_status=next(iter(component_statuses)),
        horizon_overlaps=tuple(
            item
            for item in candidate_overlaps
            if (item.left_holding_weeks, item.right_holding_weeks) in adjacent_pairs
        ),
        candidate_pairwise_overlaps=candidate_overlaps,
        risk_qualified_pairwise_overlaps=_build_pairwise_overlaps(
            periods,
            set_nature="risk_qualified",
        ),
        action_pairwise_overlaps=_build_pairwise_overlaps(
            periods,
            set_nature="action",
        ),
        repeated_symbol_attributions=_build_repeated_symbol_attributions(periods),
    )


def render_evening_digest_markdown(
    digest: EveningResearchDigest,
    holding_review: HoldingTreeReviewSummary | None = None,
    *,
    include_holding_summary: bool = False,
) -> str:
    """Render a calm ServerChan summary; holding disclosure defaults off."""

    if not isinstance(digest, EveningResearchDigest):
        raise TypeError("digest must be an EveningResearchDigest")
    for name_bytes, price_bytes, cycle_bytes in _NOTIFICATION_LAYOUT_BUDGETS:
        body = _render_notification_summary(
            digest,
            holding_review=holding_review,
            include_holding_summary=include_holding_summary,
            markdown=True,
            name_bytes=name_bytes,
            price_bytes=price_bytes,
            cycle_bytes=cycle_bytes,
        )
        if len(body.encode("utf-8")) <= MAX_COMPACT_NOTIFICATION_BODY_BYTES:
            return body
    raise ValueError("ServerChan digest exceeds the safe UTF-8 byte budget")


def render_evening_digest_bark_compact(
    digest: EveningResearchDigest,
    holding_review: HoldingTreeReviewSummary | None = None,
    *,
    include_holding_summary: bool = False,
) -> str:
    """Render a six-horizon plain-text Bark body within the APNs-safe budget.

    The compact body is derived from the same audited digest as the full
    ServerChan Markdown.  It never truncates the whole report or drops a
    holding period/candidate.  Instead, bounded display fields are shortened
    deterministically; if an upstream invariant is broken, rendering fails
    closed instead of submitting an incomplete notification.
    """

    if not isinstance(digest, EveningResearchDigest):
        raise TypeError("digest must be an EveningResearchDigest")
    if any(len(period.candidates) > 5 for period in digest.periods):
        raise ValueError("Bark compact digest supports at most five candidates per period")

    for name_bytes, price_bytes, cycle_bytes in _NOTIFICATION_LAYOUT_BUDGETS:
        body = _render_notification_summary(
            digest,
            holding_review=holding_review,
            include_holding_summary=include_holding_summary,
            markdown=False,
            name_bytes=name_bytes,
            price_bytes=price_bytes,
            cycle_bytes=cycle_bytes,
        )
        if len(body.encode("utf-8")) <= MAX_COMPACT_NOTIFICATION_BODY_BYTES:
            return body
    raise ValueError("Bark compact digest exceeds the safe UTF-8 byte budget")


_NOTIFICATION_LAYOUT_BUDGETS = (
    (18, 45, 54),
    (15, 36, 45),
    (12, 30, 36),
    (9, 24, 30),
    (6, 18, 24),
)


def _render_notification_summary(
    digest: EveningResearchDigest,
    *,
    holding_review: HoldingTreeReviewSummary | None,
    include_holding_summary: bool,
    markdown: bool,
    name_bytes: int,
    price_bytes: int,
    cycle_bytes: int,
) -> str:
    plan = (
        "未确认"
        if digest.plan_for_date is None
        else format_cn_plan_date(digest.plan_for_date).replace("（", "").replace("）", "")
    )
    cycle = _truncate_utf8(_clean_text(digest.cycle_label), cycle_bytes)
    posture = _cycle_posture(digest.entry_strictness)
    title = f"# {plan} A股研究计划" if markdown else f"{plan} A股研究计划"
    lines = [
        title,
        f"数据{digest.common_cutoff.isoformat()}｜{cycle}｜{posture}｜"
        f"股≤{digest.max_stock_exposure:.0%}/现≥{digest.minimum_cash_weight:.0%}",
    ]
    if digest.plan_for_date is None:
        lines.append("计划日未通过交易日历确认｜不据此执行")
    if include_holding_summary:
        lines.extend(("", "## 当前持仓修枝" if markdown else "当前持仓修枝"))
        lines.extend(
            _holding_review_lines(
                holding_review,
                name_bytes=name_bytes,
                reason_bytes=price_bytes,
            )
        )
    lines.extend(("", "## 六期限计划" if markdown else "六期限计划"))
    for period in digest.periods:
        if period.failure_code is not None:
            lines.append(f"- {period.label}｜数据不足")
            continue
        if not period.candidates:
            lines.append(f"- {period.label}｜无合格")
            continue
        candidates: list[str] = []
        for candidate in period.candidates:
            symbol = _truncate_utf8(_symbol(candidate.symbol), 12)
            name = _truncate_utf8(_clean_text(candidate.name), name_bytes)
            sleeve = (
                "—"
                if candidate.stock_sleeve_weight is None
                else f"{candidate.stock_sleeve_weight:.0%}"
            )
            price = _truncate_utf8(
                _compact_price_condition(candidate.price_condition),
                price_bytes,
            )
            candidates.append(f"{name}({symbol}) {sleeve}@{price}")
        lines.append(f"- {period.label}｜" + "；".join(candidates))
    lines.extend(
        (
            "",
            "仅研究｜仓%=股票仓内10%档｜价格为条件/观察线｜不自动下单。",
        )
    )
    return "\n".join(lines).strip()


def _cycle_posture(entry_strictness: str) -> str:
    return {
        EntryStrictness.STANDARD.value: "偏进攻",
        EntryStrictness.TIGHT.value: "均衡偏防守",
        EntryStrictness.DEFENSIVE.value: "防守",
        EntryStrictness.EXCEPTION_ONLY.value: "强防守",
        EntryStrictness.UNAVAILABLE.value: "观望",
    }.get(entry_strictness, "观望")


def _holding_review_lines(
    review: HoldingTreeReviewSummary | None,
    *,
    name_bytes: int,
    reason_bytes: int,
) -> list[str]:
    if review is None:
        return ["- 持仓复核不可用｜本次不生成持仓动作"]
    if review.status is HoldingReviewSummaryStatus.NO_HOLDINGS:
        if review.portfolio_id is not None:
            return ["- 已明确空仓｜如有变化请重新登记"]
        return ["- 未登记持仓｜告诉我股票和周期后开始每日修枝"]
    if not review.rows:
        return ["- 持仓数据待复核｜本次不生成减仓或退出结论"]
    priority = {
        HoldingAction.EXIT: 0,
        HoldingAction.REDUCE: 1,
        HoldingAction.REVIEW: 2,
        HoldingAction.TIGHTEN: 3,
        HoldingAction.HOLD: 4,
    }
    rows = sorted(review.rows, key=lambda row: (priority[row.action], row.symbol))
    return [
        _holding_review_row_line(
            row,
            name_bytes=name_bytes,
            reason_bytes=reason_bytes,
        )
        for row in rows
    ]


def _holding_review_row_line(
    row: HoldingTreeReviewRow,
    *,
    name_bytes: int,
    reason_bytes: int,
) -> str:
    name = _truncate_utf8(_clean_text(row.name), name_bytes)
    action = {
        HoldingAction.HOLD: "HOLD",
        HoldingAction.TIGHTEN: "收紧",
        HoldingAction.REDUCE: "减仓",
        HoldingAction.EXIT: "退出",
        HoldingAction.REVIEW: "复核",
    }[row.action]
    horizon = _HORIZON_LABELS.get(row.holding_weeks, f"{row.holding_weeks}周")
    protection = "保护线待核验" if row.effective_stop is None else f"保护线{row.effective_stop:.2f}"
    reason = _truncate_utf8(_holding_reason(row), reason_bytes)
    return f"- {name}({_symbol(row.symbol)})｜{horizon}｜{action}｜{protection}｜{reason}"


def _holding_reason(row: HoldingTreeReviewRow) -> str:
    if row.status is HoldingReviewRowStatus.DATA_NOT_READY:
        if any("company_action_evidence_blocks" in reason for reason in row.reasons):
            return "除权/分红证据待核验，不作动作"
        return "数据不足，不作持仓动作"
    labels = (
        ("complete_close_confirmed_below_effective_stop", "收盘已跌破保护线"),
        ("multiple_timeframe_weakness_confirmed", "多周期转弱确认"),
        (
            "single_dimension_weakness_warning_not_multi_timeframe_confirmation",
            "单维度转弱预警",
        ),
        ("completed_multitimeframe_weakness", "多周期结构转弱"),
        ("confirmed_pivot_raised_protection_line", "新基准点上移"),
        ("no_completed_close_exit_or_reduce_signal", "结构仍完整"),
    )
    for prefix, label in labels:
        if any(reason.startswith(prefix) for reason in row.reasons):
            return label
    return "等待下一完整收盘复核"


def _compact_price_condition(value: str) -> str:
    text = _clean_text(value)
    replacements = (
        ("—（未形成价格条件）", "价格待定"),
        ("仅观察：", "观察"),
        ("条件介入：", ""),
        ("回踩观察区", "回踩"),
        ("回踩至", "回踩"),
        ("收盘站回观察线", "站回"),
        ("收盘突破观察线", "突破"),
        ("，并满足量能确认", "+量"),
        ("（重新站回确认）", ""),
        ("（放量突破确认）", "+量"),
        ("（触及不等于可买）", ""),
        ("元", ""),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text.replace("+量+量", "+量").strip() or "价格待定"


def _truncate_utf8(value: str, maximum_bytes: int) -> str:
    if maximum_bytes < 3:
        raise ValueError("maximum_bytes must leave room for an ellipsis")
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return value
    budget = maximum_bytes - len("…".encode())
    output: list[str] = []
    used = 0
    for character in value:
        size = len(character.encode("utf-8"))
        if used + size > budget:
            break
        output.append(character)
        used += size
    return "".join(output).rstrip() + "…"


def format_cn_plan_date(value: date) -> str:
    """Return an audit-stable Chinese label for one verified trading day."""

    if not isinstance(value, date):
        raise TypeError("plan date must be a date")
    weekday = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")[value.weekday()]
    return f"{value.isoformat()}（{weekday}）"


def _summarize_period(result: MidtermPortfolioResult, *, weeks: int) -> EveningPeriodDigest:
    cutoff = None if result.data_cutoff is None else _as_date(result.data_cutoff)
    if result.status is MidtermPortfolioStatus.DATA_NOT_READY:
        return _data_unavailable_period(weeks, "model_data_not_ready", data_cutoff=cutoff)

    action_by_symbol = {
        candidate.symbol: candidate.action for candidate in result.research_candidates
    }
    if result.positions and result.evaluation is not None:
        rows = tuple(
            _candidate_from_position(
                position,
                action=action_by_symbol.get(position.symbol, CandidateAction.WAIT_CONFIRMATION),
                expected_cutoff=cutoff,
                expected_sessions=HOLDING_PERIOD_SESSIONS[weeks],
            )
            for position in result.positions
        )
        audit_rows = tuple(
            _candidate_from_research(
                candidate,
                allocation_nature=(
                    "risk_qualified_research"
                    if result.research_evaluation is not None
                    else "unavailable"
                ),
                expected_cutoff=cutoff,
                expected_sessions=HOLDING_PERIOD_SESSIONS[weeks],
            )
            for candidate in result.research_candidates
        )
        risk_nature = "下行风险与历史收益下界门已通过；仍是研究行动方案，不是实盘持仓"
        action_nature = "研究行动组合（非交易指令）"
    else:
        allocation_nature: Literal["risk_qualified_research", "observation_only", "unavailable"]
        if result.research_evaluation is not None:
            allocation_nature = "risk_qualified_research"
            risk_nature = "研究风险门通过；当前证据或介入门未形成行动组合"
        elif result.observation_evaluation is not None:
            allocation_nature = "observation_only"
            rejected = "、".join(
                _REJECTION_LABELS.get(reason, _clean_text(reason))
                for reason in result.observation_rejection_reasons
            )
            risk_nature = f"最接近组合未通过：{rejected or '风险或收益门'}；配比只供观察"
        else:
            allocation_nature = "unavailable"
            risk_nature = "尚未形成风险合格或可审计观察权重"
        rows = tuple(
            _candidate_from_research(
                candidate,
                allocation_nature=allocation_nature,
                expected_cutoff=cutoff,
                expected_sessions=HOLDING_PERIOD_SESSIONS[weeks],
            )
            for candidate in result.research_candidates
        )
        audit_rows = rows
        if result.status is MidtermPortfolioStatus.VALIDATION_NOT_READY:
            action_nature = "证据待核验；行动层保持现金"
        elif result.status is MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO:
            action_nature = "暂无可介入组合；行动层保持现金"
        else:
            action_nature = "研究候选；当前不是交易指令"

    return EveningPeriodDigest(
        holding_weeks=weeks,
        holding_sessions=HOLDING_PERIOD_SESSIONS[weeks],
        label=_HORIZON_LABELS.get(weeks, f"{weeks}周"),
        data_cutoff=cutoff,
        source_status=result.status.value,
        action_nature=action_nature,
        risk_nature=risk_nature,
        action_stock_exposure=_finite_fraction(result.stock_exposure),
        action_cash_weight=_finite_fraction(result.cash_weight),
        candidates=rows,
        audit_candidates=audit_rows,
        exclusion_categories=tuple(
            SymbolExclusionDigest(
                symbol=_symbol(exclusion.symbol),
                categories=_categorize_exclusion_reasons(exclusion.reasons),
            )
            for exclusion in result.exclusions
        ),
        exclusion_audit_available=True,
        portfolio_failure_categories=_categorize_portfolio_failures(
            result.observation_rejection_reasons,
            result.reasons,
        ),
        ranked_pool_count=_nonnegative_int_optional(result.horizon_candidate_count),
        central_implementation_status=result.central_implementation_status,
        multi_timeframe_component_status=result.multi_timeframe_component_status,
        performance_nature=_period_performance_nature(result, rows),
    )


def _candidate_from_position(
    position: Any,
    *,
    action: CandidateAction,
    expected_cutoff: date | None,
    expected_sessions: int,
) -> EveningDigestCandidate:
    evidence_pending = bool(position.evidence_unknown)
    conditional = (
        position.conditional_entry_plan
        if action is CandidateAction.CONDITIONAL_ENTRY and not evidence_pending
        else None
    )
    price, price_nature = _price_label(
        conditional_plan=conditional,
        observation_plan=position.price_observation_plan,
        expected_cutoff=expected_cutoff,
        expected_sessions=expected_sessions,
    )
    sleeve = _operational_weight(position.operational_stock_sleeve_weight)
    account = _finite_optional_fraction(position.operational_account_weight)
    selected_plan = _selected_structured_plan(
        price_nature=price_nature,
        conditional_plan=conditional,
        observation_plan=position.price_observation_plan,
    )
    return EveningDigestCandidate(
        rank=int(position.rank),
        symbol=_symbol(position.symbol),
        name=_clean_text(position.name),
        action=action.value,
        allocation_nature="action_research",
        stock_sleeve_weight=sleeve,
        account_weight=account,
        price_condition=price,
        price_nature=price_nature,
        evidence_pending=evidence_pending,
        operational_stock_sleeve_weight=sleeve,
        operational_account_weight=account,
        price_plan_sessions=_selected_plan_sessions(
            conditional_plan=conditional,
            observation_plan=position.price_observation_plan,
        ),
        **_structured_plan_fields(selected_plan),
        **_candidate_timeframe_fields(getattr(position, "timeframe", None)),
    )


def _candidate_from_research(
    candidate: Any,
    *,
    allocation_nature: Literal["risk_qualified_research", "observation_only", "unavailable"],
    expected_cutoff: date | None,
    expected_sessions: int,
) -> EveningDigestCandidate:
    evidence_pending = bool(candidate.evidence_unknown)
    conditional = (
        candidate.conditional_entry_plan
        if candidate.action is CandidateAction.CONDITIONAL_ENTRY and not evidence_pending
        else None
    )
    price, price_nature = _price_label(
        conditional_plan=conditional,
        observation_plan=candidate.price_observation_plan,
        expected_cutoff=expected_cutoff,
        expected_sessions=expected_sessions,
    )
    if allocation_nature == "risk_qualified_research":
        sleeve = _operational_weight(candidate.operational_stock_sleeve_weight)
        account = _finite_optional_fraction(candidate.operational_account_weight)
    elif allocation_nature == "observation_only":
        sleeve = _operational_weight(candidate.observation_stock_sleeve_weight)
        account = None
    else:
        sleeve = None
        account = None
    selected_plan = _selected_structured_plan(
        price_nature=price_nature,
        conditional_plan=conditional,
        observation_plan=candidate.price_observation_plan,
    )
    return EveningDigestCandidate(
        rank=int(candidate.rank),
        symbol=_symbol(candidate.symbol),
        name=_clean_text(candidate.name),
        action=candidate.action.value,
        allocation_nature=allocation_nature,
        stock_sleeve_weight=sleeve,
        account_weight=account,
        price_condition=price,
        price_nature=price_nature,
        evidence_pending=evidence_pending,
        operational_stock_sleeve_weight=(
            sleeve if allocation_nature == "risk_qualified_research" else None
        ),
        operational_account_weight=(
            account if allocation_nature == "risk_qualified_research" else None
        ),
        price_plan_sessions=_selected_plan_sessions(
            conditional_plan=conditional,
            observation_plan=candidate.price_observation_plan,
        ),
        **_structured_plan_fields(selected_plan),
        **_candidate_timeframe_fields(getattr(candidate, "timeframe", None)),
    )


def _period_performance_nature(
    result: MidtermPortfolioResult,
    rows: tuple[EveningDigestCandidate, ...],
) -> Literal[
    "official_action",
    "risk_qualified_observation",
    "observation_only",
    "full_cash",
    "data_unavailable",
]:
    """Classify what may later enter each explicitly separated scorecard."""

    has_archivable_action = any(
        candidate.allocation_nature == "action_research"
        and candidate.action == CandidateAction.CONDITIONAL_ENTRY.value
        and candidate.price_nature == "conditional_entry"
        and candidate.operational_stock_sleeve_weight is not None
        and candidate.operational_account_weight is not None
        for candidate in rows
    )
    if result.positions and result.evaluation is not None and has_archivable_action:
        return "official_action"
    if result.research_evaluation is not None:
        return "risk_qualified_observation"
    if result.observation_evaluation is not None:
        return "observation_only"
    return "full_cash"


def _price_label(
    *,
    conditional_plan: ConditionalEntryPlan | None,
    observation_plan: ConditionalEntryPlan | None,
    expected_cutoff: date | None,
    expected_sessions: int,
) -> tuple[str, Literal["conditional_entry", "observation_only", "unavailable"]]:
    if conditional_plan is not None:
        label = _format_plan(
            conditional_plan,
            expected_cutoff=expected_cutoff,
            expected_sessions=expected_sessions,
            observation=False,
        )
        if label is not None:
            return label, "conditional_entry"
        return "—（价格计划与本期不一致）", "unavailable"
    if observation_plan is not None:
        label = _format_plan(
            observation_plan,
            expected_cutoff=expected_cutoff,
            expected_sessions=expected_sessions,
            observation=True,
        )
        if label is not None:
            return f"仅观察：{label}（触及不等于可买）", "observation_only"
        return "—（价格计划与本期不一致）", "unavailable"
    return "—（未形成价格条件）", "unavailable"


def _candidate_timeframe_fields(assessment: object) -> dict[str, Any]:
    if assessment is None:
        return {}
    slow = getattr(assessment, "slow_direction", None)
    structure = getattr(assessment, "structure", None)
    return {
        "slow_timeframe": getattr(getattr(slow, "timeframe", None), "value", None),
        "slow_direction": getattr(getattr(slow, "direction", None), "value", None),
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
        "primary_breakout_line": _positive_optional(getattr(structure, "breakout_line", None)),
        "multi_timeframe_score": _optional_fraction(getattr(assessment, "score", None)),
        "multi_timeframe_method_version": getattr(assessment, "method_version", None),
        "timeframe_holding_weeks": _positive_int_optional(
            getattr(assessment, "holding_weeks", None)
        ),
    }


def _selected_plan_sessions(
    *,
    conditional_plan: ConditionalEntryPlan | None,
    observation_plan: ConditionalEntryPlan | None,
) -> int | None:
    plan = conditional_plan if conditional_plan is not None else observation_plan
    return None if plan is None else _positive_int_optional(getattr(plan, "sessions", None))


def _selected_structured_plan(
    *,
    price_nature: Literal["conditional_entry", "observation_only", "unavailable"],
    conditional_plan: ConditionalEntryPlan | None,
    observation_plan: ConditionalEntryPlan | None,
) -> ConditionalEntryPlan | None:
    """Return only the plan that actually produced the audited display row."""

    if price_nature == "conditional_entry":
        return conditional_plan
    if price_nature == "observation_only":
        return observation_plan
    return None


def _structured_plan_fields(plan: ConditionalEntryPlan | None) -> dict[str, Any]:
    """Extract safe scalar plan evidence; no OHLCV frame enters the digest."""

    if plan is None:
        return {}
    kind = getattr(plan.kind, "value", plan.kind)
    cutoff = getattr(plan, "data_cutoff", None)
    evaluation_price = (
        getattr(plan, "price_high", None)
        if kind == ConditionalEntryPlanKind.HEALTHY_PULLBACK.value
        else getattr(plan, "trigger_price", None)
    )
    return {
        "price_plan_kind": _clean_text(kind),
        "price_plan_low": _positive_optional(getattr(plan, "price_low", None)),
        "price_plan_high": _positive_optional(getattr(plan, "price_high", None)),
        "price_plan_trigger": _positive_optional(getattr(plan, "trigger_price", None)),
        "price_plan_evaluation_price": _positive_optional(evaluation_price),
        "price_plan_confirmation_rule": _clean_optional_text(
            getattr(plan, "confirmation_rule", None)
        ),
        "price_plan_confirmation_activity_metric": _clean_optional_text(
            getattr(plan, "confirmation_activity_metric", None)
        ),
        "price_plan_confirmation_activity_min": _positive_optional(
            getattr(plan, "confirmation_activity_min", None)
        ),
        "price_plan_invalidation_price": _positive_optional(
            getattr(plan, "invalidation_price", None)
        ),
        "price_plan_cutoff": None if cutoff is None else _as_date(cutoff),
        "price_plan_method_version": _clean_optional_text(getattr(plan, "method_version", None)),
    }


def _candidate_timeframe_label(candidate: EveningDigestCandidate) -> str:
    slow = _TIMEFRAME_LABELS.get(
        candidate.slow_timeframe or "", candidate.slow_timeframe or "慢周期—"
    )
    primary = _TIMEFRAME_LABELS.get(
        candidate.primary_timeframe or "",
        candidate.primary_timeframe or "主周期—",
    )
    structure = _STRUCTURE_LABELS.get(
        candidate.primary_structure or "",
        candidate.primary_structure or "结构—",
    )
    direction = candidate.slow_direction or "—"
    return f"{slow}{direction} / {primary}{structure}"


def _compact_timeframe_label(candidate: EveningDigestCandidate) -> str:
    frame_codes = {"daily": "D", "weekly_completed": "W", "monthly_completed": "M"}
    slow = frame_codes.get(candidate.slow_timeframe or "", "?")
    direction_value = (candidate.slow_direction or "").lower()
    if direction_value.startswith("up"):
        direction = "↑"
    elif direction_value.startswith("down"):
        direction = "↓"
    else:
        direction = "?"
    primary = frame_codes.get(candidate.primary_timeframe or "", "?")
    structure = {
        "near_breakout": "近",
        "healthy_post_breakout_pullback": "回",
        "volume_confirmed_breakout": "突",
        "base_not_yet_near_breakout": "基",
        "trend_continuation_without_entry_structure": "延",
        "failed": "失",
        "insufficient": "缺",
    }.get(candidate.primary_structure or "", "?")
    return f"{slow}{direction}>{primary}{structure}"


def _compact_action(value: str) -> str:
    return {
        CandidateAction.CONDITIONAL_ENTRY.value: "条",
        CandidateAction.WAIT_CONFIRMATION.value: "等",
        CandidateAction.OBSERVE_ONLY.value: "观",
    }.get(value, "—")


def _compact_price_nature(value: str) -> str:
    return {
        "conditional_entry": "条价",
        "observation_only": "察价",
        "unavailable": "价—",
    }.get(value, "价—")


def _build_pairwise_overlaps(
    periods: tuple[EveningPeriodDigest, ...],
    *,
    set_nature: Literal["candidate", "risk_qualified", "action"],
) -> tuple[HorizonOverlapDigest, ...]:
    rows: list[HorizonOverlapDigest] = []
    for left, right in combinations(periods, 2):
        left_available, left_symbols, left_reason = _period_symbol_set(
            left,
            set_nature=set_nature,
        )
        right_available, right_symbols, right_reason = _period_symbol_set(
            right,
            set_nature=set_nature,
        )
        union = left_symbols | right_symbols
        shared = tuple(sorted(left_symbols & right_symbols))
        comparable = left_available and right_available and bool(union)
        unavailable_reason = None
        if not comparable:
            unavailable_reason = _comparison_unavailable_reason(
                left_available=left_available,
                left_reason=left_reason,
                right_available=right_available,
                right_reason=right_reason,
                union=union,
            )
        rows.append(
            HorizonOverlapDigest(
                left_holding_weeks=left.holding_weeks,
                right_holding_weeks=right.holding_weeks,
                left_label=left.label,
                right_label=right.label,
                shared_symbols=shared,
                union_count=len(union),
                jaccard=(len(shared) / len(union) if comparable else None),
                set_nature=set_nature,
                comparison_status="comparable" if comparable else "unavailable",
                unavailable_reason=unavailable_reason,
                left_only=tuple(
                    _pairwise_difference(
                        symbol,
                        source=left,
                        target=right,
                        side="left_only",
                        set_nature=set_nature,
                    )
                    for symbol in sorted(left_symbols - right_symbols)
                ),
                right_only=tuple(
                    _pairwise_difference(
                        symbol,
                        source=right,
                        target=left,
                        side="right_only",
                        set_nature=set_nature,
                    )
                    for symbol in sorted(right_symbols - left_symbols)
                ),
            )
        )
    return tuple(rows)


def _period_symbol_set(
    period: EveningPeriodDigest,
    *,
    set_nature: Literal["candidate", "risk_qualified", "action"],
) -> tuple[bool, set[str], str | None]:
    if period.failure_code is not None:
        return False, set(), "data_unavailable"
    audit_candidates = period.audit_candidates or period.candidates
    if set_nature == "candidate":
        return True, {candidate.symbol for candidate in audit_candidates}, None
    if set_nature == "risk_qualified":
        accepted = {"risk_qualified_research", "action_research"}
        symbols = {
            candidate.symbol
            for candidate in audit_candidates
            if candidate.allocation_nature in accepted and candidate.stock_sleeve_weight is not None
        }
    else:
        symbols = {
            candidate.symbol
            for candidate in period.candidates
            if candidate.allocation_nature == "action_research"
        }
    if not symbols:
        return False, set(), f"{set_nature}_set_not_formed"
    return True, symbols, None


def _comparison_unavailable_reason(
    *,
    left_available: bool,
    left_reason: str | None,
    right_available: bool,
    right_reason: str | None,
    union: set[str],
) -> str:
    if left_available and right_available and not union:
        return "both_sets_empty"
    reasons = []
    if not left_available:
        reasons.append(f"left:{left_reason or 'evidence_unavailable'}")
    if not right_available:
        reasons.append(f"right:{right_reason or 'evidence_unavailable'}")
    return ";".join(reasons) or "evidence_unavailable"


def _pairwise_difference(
    symbol: str,
    *,
    source: EveningPeriodDigest,
    target: EveningPeriodDigest,
    side: Literal["left_only", "right_only"],
    set_nature: Literal["candidate", "risk_qualified", "action"],
) -> PairwiseSymbolDifference:
    source_candidate = _find_period_candidate(source, symbol)
    return PairwiseSymbolDifference(
        symbol=symbol,
        name="—" if source_candidate is None else source_candidate.name,
        side=side,
        reason=_difference_reason(symbol, target=target, set_nature=set_nature),
    )


def _difference_reason(
    symbol: str,
    *,
    target: EveningPeriodDigest,
    set_nature: Literal["candidate", "risk_qualified", "action"],
) -> DifferenceReason:
    if target.failure_code is not None:
        return "data_unavailable"
    candidate = _find_period_candidate(target, symbol)
    if candidate is None:
        categories = _exclusion_categories_for_symbol(target, symbol)
        if categories:
            return categories[0]
        audit_candidates = target.audit_candidates or target.candidates
        if target.ranked_pool_count is not None and target.ranked_pool_count > len(
            audit_candidates
        ):
            return "ranking_not_selected"
        return "evidence_unavailable"
    if set_nature == "candidate":
        return "evidence_unavailable"

    _, risk_symbols, _ = _period_symbol_set(target, set_nature="risk_qualified")
    if symbol not in risk_symbols:
        if target.portfolio_failure_categories:
            return target.portfolio_failure_categories[0]
        if set_nature == "action" and (
            candidate.action != CandidateAction.CONDITIONAL_ENTRY.value
            or candidate.price_nature != "conditional_entry"
        ):
            return "price_or_action_not_triggered"
        return "evidence_unavailable"
    if set_nature == "risk_qualified":
        return "ranking_not_selected"

    if (
        candidate.action != CandidateAction.CONDITIONAL_ENTRY.value
        or candidate.price_nature != "conditional_entry"
    ):
        return "price_or_action_not_triggered"
    _, action_symbols, _ = _period_symbol_set(target, set_nature="action")
    if action_symbols:
        return "ranking_not_selected"
    return "evidence_unavailable"


def _find_period_candidate(
    period: EveningPeriodDigest,
    symbol: str,
) -> EveningDigestCandidate | None:
    audit_candidates = period.audit_candidates or period.candidates
    return next(
        (candidate for candidate in audit_candidates if candidate.symbol == symbol),
        None,
    )


def _exclusion_categories_for_symbol(
    period: EveningPeriodDigest,
    symbol: str,
) -> tuple[DifferenceReason, ...]:
    row = next(
        (item for item in period.exclusion_categories if item.symbol == symbol),
        None,
    )
    return () if row is None else row.categories


def _build_repeated_symbol_attributions(
    periods: tuple[EveningPeriodDigest, ...],
) -> tuple[RepeatedSymbolAttribution, ...]:
    appearances: dict[str, list[tuple[EveningPeriodDigest, EveningDigestCandidate]]] = {}
    for period in periods:
        audit_candidates = period.audit_candidates or period.candidates
        for candidate in audit_candidates:
            appearances.setdefault(candidate.symbol, []).append((period, candidate))

    rows: list[RepeatedSymbolAttribution] = []
    for symbol in sorted(appearances):
        source_rows = appearances[symbol]
        if len(source_rows) < 2:
            continue
        differences = tuple(
            _symbol_difference(period, candidate) for period, candidate in source_rows
        )
        documented = sum(item.independent_gate_documented for item in differences)
        rows.append(
            RepeatedSymbolAttribution(
                symbol=symbol,
                name=_clean_text(source_rows[0][1].name),
                appearances=differences,
                independent_gate_count=documented,
                conclusion=(
                    "independent_horizon_gates_documented_not_automatic_confluence"
                    if documented == len(differences)
                    else "repeated_candidate_evidence_incomplete_not_confluence"
                ),
            )
        )
    return tuple(rows)


def _symbol_difference(
    period: EveningPeriodDigest,
    candidate: EveningDigestCandidate,
) -> HorizonSymbolDifference:
    documented = _independent_gate_documented(period, candidate)
    return HorizonSymbolDifference(
        holding_weeks=period.holding_weeks,
        label=period.label,
        independent_gate_documented=bool(documented),
        slow_context=(
            f"{_TIMEFRAME_LABELS.get(candidate.slow_timeframe or '', candidate.slow_timeframe or '—')}"
            f"/{_clean_text(candidate.slow_direction or '—', limit=24)}"
        ),
        primary_structure=(
            f"{_TIMEFRAME_LABELS.get(candidate.primary_timeframe or '', candidate.primary_timeframe or '—')}"
            f"/{_STRUCTURE_LABELS.get(candidate.primary_structure or '', candidate.primary_structure or '—')}"
        ),
        action=_ACTION_LABELS.get(candidate.action, _clean_text(candidate.action)),
        allocation_nature=candidate.allocation_nature,
        stock_sleeve_weight=candidate.stock_sleeve_weight,
        price_nature=candidate.price_nature,
        price_condition=candidate.price_condition,
    )


def _independent_gate_documented(
    period: EveningPeriodDigest,
    candidate: EveningDigestCandidate,
) -> bool:
    expected_frames = {
        1: ("weekly_completed", "daily"),
        2: ("weekly_completed", "daily"),
        4: ("weekly_completed", "daily"),
        13: ("monthly_completed", "weekly_completed"),
        26: ("monthly_completed", "weekly_completed"),
        52: ("monthly_completed", "weekly_completed"),
    }
    expected = expected_frames.get(period.holding_weeks)
    if expected is None:
        return False
    expected_slow, expected_primary = expected
    return bool(
        candidate.timeframe_holding_weeks == period.holding_weeks
        and candidate.price_plan_sessions == period.holding_sessions
        and candidate.slow_timeframe == expected_slow
        and candidate.primary_timeframe == expected_primary
        and candidate.slow_direction
        and candidate.primary_structure
        and candidate.multi_timeframe_method_version
        and candidate.multi_timeframe_score is not None
    )


def _overlap_line(overlaps: tuple[HorizonOverlapDigest, ...]) -> str:
    if not overlaps:
        return "无双方已形成的组合"
    if all(item.comparison_status == "unavailable" for item in overlaps):
        return f"{len(overlaps)}对均不可比较（数据不足、集合未形成或双方均为空）"
    return "；".join(
        f"{item.left_label}↔{item.right_label} "
        + ("不可比较" if item.comparison_status == "unavailable" else f"{item.jaccard:.0%}")
        for item in overlaps
    )


def _compact_overlap_line(overlaps: tuple[HorizonOverlapDigest, ...]) -> str:
    if not overlaps:
        return "无"
    values = {item.jaccard for item in overlaps}
    if len(values) == 1:
        value = next(iter(values))
        display = "U" if value is None else f"{value:.0%}"
        return f"{len(overlaps)}对={display}"
    counts = Counter("U" if item.jaccard is None else f"{item.jaccard:.0%}" for item in overlaps)
    distribution = "/".join(f"{value}×{count}" for value, count in sorted(counts.items()))
    return f"{len(overlaps)}对({distribution})"


def _pair_difference_line(item: HorizonOverlapDigest) -> str:
    nature = {
        "candidate": "候选",
        "risk_qualified": "风险",
        "action": "行动",
    }[item.set_nature]
    left = _side_difference_label(item.left_only)
    right = _side_difference_label(item.right_only)
    return f"{nature} {item.left_label}↔{item.right_label}｜左[{left}]｜右[{right}]"


def _side_difference_label(rows: tuple[PairwiseSymbolDifference, ...]) -> str:
    if not rows:
        return "无"
    labels = {
        "data_unavailable": "数据不可用",
        "slow_context_failure": "慢周期",
        "primary_structure_failure": "主结构",
        "risk_history_unavailable": "风险历史",
        "risk_budget_or_lcb_failure": "风险/LCB",
        "price_or_action_not_triggered": "价格/动作",
        "ranking_not_selected": "排名",
        "evidence_unavailable": "证据不可用",
    }
    return ",".join(f"{row.symbol}:{labels[row.reason]}" for row in rows)


def _attribution_markdown(item: RepeatedSymbolAttribution) -> str:
    appearances = []
    for difference in item.appearances:
        sleeve = (
            "配比—"
            if difference.stock_sleeve_weight is None
            else f"股票仓{difference.stock_sleeve_weight:.0%}"
        )
        allocation = {
            "action_research": "行动组合",
            "risk_qualified_research": "风险合格",
            "observation_only": "观察层",
            "unavailable": "配比不可用",
        }.get(difference.allocation_nature, "性质—")
        evidence = "独立门✓" if difference.independent_gate_documented else "独立门证据缺"
        appearances.append(
            f"{difference.label}[{difference.slow_context}；{difference.primary_structure}；"
            f"{difference.action}；{allocation}/{sleeve}；"
            f"{_compact_price_nature(difference.price_nature)}；"
            f"{evidence}]"
        )
    conclusion = (
        "各期独立门已有记录，但重合本身仍不是自动共振"
        if item.conclusion == "independent_horizon_gates_documented_not_automatic_confluence"
        else "独立门证据不完整，不称多周期共振"
    )
    return "；".join(appearances) + f"｜{conclusion}"


def _format_plan(
    plan: ConditionalEntryPlan,
    *,
    expected_cutoff: date | None,
    expected_sessions: int,
    observation: bool,
) -> str | None:
    if expected_cutoff is None or _as_date(plan.data_cutoff) != expected_cutoff:
        return None
    if (
        not isinstance(plan.horizon, str)
        or not plan.horizon.strip()
        or plan.sessions != expected_sessions
    ):
        return None
    if plan.kind is ConditionalEntryPlanKind.HEALTHY_PULLBACK:
        low = _positive_optional(plan.price_low)
        high = _positive_optional(plan.price_high)
        if low is None or high is None or low > high:
            return None
        return (
            f"回踩观察区 {low:.2f}–{high:.2f}元"
            if observation
            else f"条件介入：回踩至 {low:.2f}–{high:.2f}元"
        )
    trigger = _positive_optional(plan.trigger_price)
    if trigger is None:
        return None
    if plan.kind is ConditionalEntryPlanKind.RECLAIM:
        return (
            f"收盘站回观察线 ≥ {trigger:.2f}元"
            if observation
            else f"条件介入：收盘价 ≥ {trigger:.2f}元（重新站回确认）"
        )
    if plan.kind is ConditionalEntryPlanKind.VOLUME_BREAKOUT:
        return (
            f"收盘突破观察线 ≥ {trigger:.2f}元，并满足量能确认"
            if observation
            else f"条件介入：收盘价 ≥ {trigger:.2f}元（放量突破确认）"
        )
    return None


def _allocation_label(candidate: EveningDigestCandidate) -> str:
    if candidate.stock_sleeve_weight is None:
        return "配比—"
    if candidate.allocation_nature == "observation_only":
        return f"观察配比{candidate.stock_sleeve_weight:.0%}股票仓（非资金仓位）"
    if candidate.account_weight is None:
        return f"股票仓内{candidate.stock_sleeve_weight:.0%}（总资金待定）"
    noun = "计划配比" if candidate.allocation_nature == "action_research" else "研究配比"
    return f"{noun}{candidate.stock_sleeve_weight:.0%}股票仓/{candidate.account_weight:.1%}总资金"


def _data_unavailable_period(
    weeks: int,
    failure_code: str,
    *,
    data_cutoff: date | None = None,
) -> EveningPeriodDigest:
    return EveningPeriodDigest(
        holding_weeks=weeks,
        holding_sessions=HOLDING_PERIOD_SESSIONS[weeks],
        label=_HORIZON_LABELS.get(weeks, f"{weeks}周"),
        data_cutoff=data_cutoff,
        source_status=MidtermPortfolioStatus.DATA_NOT_READY.value,
        action_nature="数据不足；不形成行动结论",
        risk_nature="未计算，不以缺失值替代",
        action_stock_exposure=0.0,
        action_cash_weight=1.0,
        failure_code=failure_code,
        performance_nature="data_unavailable",
    )


def _select_common_cutoff(cutoffs: dict[int, date]) -> date:
    if 13 in cutoffs:
        return cutoffs[13]
    counts = Counter(cutoffs.values())
    return min(counts, key=lambda value: (-counts[value], value))


def _select_cycle(
    cycles: dict[int, Any],
    common_cutoff: date,
    periods: dict[int, EveningPeriodDigest],
) -> Any | None:
    ordered = (13, 1, 2, 4, 26, 52)
    for weeks in ordered:
        cycle = cycles.get(weeks)
        period = periods.get(weeks)
        if cycle is None or period is None or period.data_cutoff != common_cutoff:
            continue
        if getattr(cycle, "label", None):
            return cycle
    return None


def _failure_code(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return "required_local_dataset_missing"
    if isinstance(exc, OSError):
        return "local_data_unavailable"
    if isinstance(exc, AShareLabError):
        return "data_or_quality_gate_failed"
    return "model_input_validation_failed"


def _failure_label(code: str) -> str:
    return {
        "required_local_dataset_missing": "本地必需数据缺失",
        "local_data_unavailable": "本地数据暂不可用",
        "data_or_quality_gate_failed": "数据或质量门未通过",
        "model_input_validation_failed": "模型输入校验未通过",
        "model_data_not_ready": "模型所需证据不足",
        "common_cutoff_mismatch": "共同截止日不一致",
    }.get(code, "数据未通过完整性校验")


def _categorize_exclusion_reasons(
    reasons: Iterable[object],
) -> tuple[DifferenceReason, ...]:
    categories: list[DifferenceReason] = []
    for value in reasons:
        reason = str(value).strip().lower()
        if reason.startswith("slow_"):
            category: DifferenceReason = "slow_context_failure"
        elif (
            reason.startswith("primary_")
            or "structure_not_qualified" in reason
            or reason.startswith("below_daily_ma")
        ):
            category = "primary_structure_failure"
        elif "insufficient_holding_risk_history" in reason:
            category = "risk_history_unavailable"
        else:
            category = "evidence_unavailable"
        if category not in categories:
            categories.append(category)
    return tuple(categories) or ("evidence_unavailable",)


def _categorize_portfolio_failures(
    rejection_reasons: Iterable[object],
    result_reasons: Iterable[object],
) -> tuple[DifferenceReason, ...]:
    values = tuple(str(value).strip().lower() for value in rejection_reasons) + tuple(
        str(value).strip().lower() for value in result_reasons
    )
    if any("insufficient_holding_risk_history" in value for value in values):
        return ("risk_history_unavailable",)
    risk_markers = (
        "holding_period_return_lcb",
        "risk_budget",
        "drawdown",
        "downside",
        "correlation",
        "expected_shortfall",
        "es95",
        "industry_concentration",
        "no_3_to_5_stock_set_passed",
    )
    if any(any(marker in value for marker in risk_markers) for value in values):
        return ("risk_budget_or_lcb_failure",)
    return ()


def _as_date(value: object) -> date:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("invalid data cutoff")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.date()


def _operational_weight(value: object) -> float | None:
    weight = _finite_optional_fraction(value)
    if weight is None:
        return None
    units = weight / 0.10
    if not math.isclose(units, round(units), abs_tol=1e-9):
        raise ValueError("operational stock-sleeve weight is not on the 10% grid")
    return weight


def _finite_fraction(value: object) -> float:
    result = _finite_optional_fraction(value)
    if result is None:
        raise ValueError("required allocation fraction is unavailable")
    return result


def _finite_optional_fraction(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        return None
    return result


def _optional_fraction(value: object) -> float | None:
    return _finite_optional_fraction(value)


def _positive_optional(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _positive_int_optional(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 and result == value else None


def _nonnegative_int_optional(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 and result == value else None


def _symbol(value: object) -> str:
    symbol = str(value).strip().upper()
    if (
        not symbol
        or len(symbol) > 16
        or not all(char.isalnum() or char in ".-_" for char in symbol)
    ):
        raise ValueError("invalid candidate symbol")
    return symbol


def _clean_text(value: object, *, limit: int = 120) -> str:
    text = " ".join(str(value).replace("|", "／").split())
    return text[:limit] or "—"


def _clean_optional_text(value: object, *, limit: int = 240) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).replace("|", "／").split())
    return text[:limit] or None
