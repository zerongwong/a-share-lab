"""Build one compact, provider-neutral evening research digest.

The service deliberately has no notification, credential, brokerage, order or
state-writing capability.  It loads only verified hybrid snapshots through an
injected/read-only loader, runs the existing deterministic mid-term model and
returns a small derived report.  Raw CSMAR/overlay frames never enter the
returned object.

History requirements differ materially by holding period.  The builder groups
periods that share the same requirement, holds only one such hybrid snapshot at
a time, and then releases it before loading the next group.  In particular,
1/2/4 weeks share the 252/322-session load, while 13/26/52 weeks retain their
own stricter evidence gates instead of being silently shortened.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ashare_lab.analytics.cycle_policy import EntryStrictness
from ashare_lab.domain.errors import AShareLabError, DataUnavailableError
from ashare_lab.ports.notifications import MAX_COMPACT_NOTIFICATION_BODY_BYTES
from ashare_lab.services.build_midterm_portfolio import (
    HOLDING_PERIOD_SESSIONS,
    CandidateAction,
    ConditionalEntryPlan,
    ConditionalEntryPlanKind,
    MidtermPortfolioResult,
    MidtermPortfolioStatus,
    build_midterm_portfolio,
)
from ashare_lab.services.load_hybrid_universe import (
    HybridUniverseLoad,
    load_hybrid_universe,
)

EVENING_DIGEST_HORIZONS = (1, 2, 4, 13, 26, 52)
EVENING_DIGEST_METHOD_VERSION = "evening-six-horizon-digest-v0.1.0"

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
    "es95_5d": "5日预期损失超限",
    "down_period_correlation": "下跌期相关性超限",
    "position_downside_risk_contribution": "单股下行风险贡献超限",
    "industry_concentration": "行业集中度超限",
    "holding_period_return_lcb_below_minimum": "历史持有期收益下界未达门槛",
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
        sessions = HOLDING_PERIOD_SESSIONS[weeks]
        minimum_sessions = max(252, sessions * 8 + 1)
        requirement_groups.setdefault((minimum_sessions, minimum_sessions + 70), []).append(weeks)

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

    return EveningResearchDigest(
        common_cutoff=common_cutoff,
        decision_date=decision_date,
        cycle_label=cycle_label,
        entry_strictness=strictness,
        max_stock_exposure=max_exposure,
        minimum_cash_weight=minimum_cash,
        cycle_rule_agreement=agreement,
        periods=tuple(period_by_weeks[weeks] for weeks in requested),
    )


def render_evening_digest_markdown(digest: EveningResearchDigest) -> str:
    """Render concise ServerChan-compatible Markdown from derived fields."""

    if not isinstance(digest, EveningResearchDigest):
        raise TypeError("digest must be an EveningResearchDigest")
    cutoff = digest.common_cutoff.isoformat()
    strictness = _ENTRY_STRICTNESS_LABELS.get(
        digest.entry_strictness,
        _clean_text(digest.entry_strictness),
    )
    agreement = (
        "不可用"
        if digest.cycle_rule_agreement is None
        else f"{digest.cycle_rule_agreement:.1%}（规则一致度，不是未来方向概率）"
    )
    if digest.plan_for_date is None:
        plan_line = "- **计划适用日：尚未通过官方交易日历确认**；不得据此执行"
    else:
        plan_line = (
            f"- **计划适用日：{format_cn_plan_date(digest.plan_for_date)}**；"
            "开盘前仍须复核停牌、涨停和跳空等可买性"
        )
    lines = [
        "# A股六周期研究日报",
        f"- **共同截止日：{cutoff}**",
        plan_line,
        "- 仅作研究，不连接券商、不自动下单",
        "",
        "## 市场价格周期",
        f"- 状态：**{_clean_text(digest.cycle_label)}**",
        f"- 介入严格度：{strictness}",
        f"- 股票敞口上限：{digest.max_stock_exposure:.0%}；最低现金：{digest.minimum_cash_weight:.0%}",
        f"- 置信说明：{agreement}",
    ]

    for period in digest.periods:
        lines.extend(("", f"## {period.label}（{period.holding_sessions}个交易日）"))
        if period.failure_code is not None:
            lines.append(
                f"- 数据不足：{_failure_label(period.failure_code)}；本周期不生成股票、配比或价格。"
            )
            continue
        lines.append(
            f"- 行动性质：{_clean_text(period.action_nature)}；"
            f"行动层股票/现金 {period.action_stock_exposure:.0%}/{period.action_cash_weight:.0%}"
        )
        lines.append(f"- 风险性质：{_clean_text(period.risk_nature)}")
        if not period.candidates:
            lines.append("- 没有通过个股资格门的候选；不以次优股票凑数。")
            continue
        for candidate in period.candidates:
            allocation = _allocation_label(candidate)
            action = _ACTION_LABELS.get(candidate.action, _clean_text(candidate.action))
            if candidate.evidence_pending:
                action += "（财务、公告或可买性待核验）"
            lines.append(
                f"{candidate.rank}. **{_clean_text(candidate.name)} {candidate.symbol}**｜"
                f"{allocation}｜{candidate.price_condition}｜{action}"
            )

    lines.extend(
        (
            "",
            "> 10%档均指股票仓内部比例；观察配比不是资金仓位。价格观察线触及不等于可买。",
            "> 本日报不是收益预测、投资建议或最大回撤保证。",
        )
    )
    body = "\n".join(lines).strip()
    if len(body) > 7_800:
        raise ValueError("evening digest exceeds notification-safe length")
    return body


def render_evening_digest_bark_compact(digest: EveningResearchDigest) -> str:
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

    # Four increasingly compact layouts retain every period and every
    # candidate. The ordinary 3-5-stock/six-period report fits the first or
    # second layout; later layouts protect against unexpectedly long names.
    budgets = (
        (18, 54, 60),
        (15, 42, 48),
        (12, 30, 36),
        (9, 21, 24),
    )
    for name_bytes, price_bytes, cycle_bytes in budgets:
        body = _render_bark_compact_with_budgets(
            digest,
            name_bytes=name_bytes,
            price_bytes=price_bytes,
            cycle_bytes=cycle_bytes,
        )
        if len(body.encode("utf-8")) <= MAX_COMPACT_NOTIFICATION_BODY_BYTES:
            return body
    raise ValueError("Bark compact digest exceeds the safe UTF-8 byte budget")


def _render_bark_compact_with_budgets(
    digest: EveningResearchDigest,
    *,
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
    lines = [
        f"A股六周期｜数据{digest.common_cutoff.isoformat()}｜计划{plan}",
        f"周期：{cycle}｜股≤{digest.max_stock_exposure:.0%}/现≥{digest.minimum_cash_weight:.0%}",
    ]
    for period in digest.periods:
        if period.failure_code is not None:
            lines.append(f"{period.label}｜数据不足")
            continue
        lines.append(
            f"{period.label}｜股{period.action_stock_exposure:.0%}/现{period.action_cash_weight:.0%}"
        )
        if not period.candidates:
            lines.append("  无候选")
            continue
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
            lines.append(f"  {candidate.rank}.{symbol}{name} 仓{sleeve} {price}")
    lines.append("仅研究；仓%=股票仓内10%档；价格为条件/观察线；不自动下单。")
    return "\n".join(lines).strip()


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
            )
            for position in result.positions
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
            )
            for candidate in result.research_candidates
        )
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
    )


def _candidate_from_position(
    position: Any,
    *,
    action: CandidateAction,
    expected_cutoff: date | None,
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
    )
    sleeve = _operational_weight(position.operational_stock_sleeve_weight)
    account = _finite_optional_fraction(position.operational_account_weight)
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
    )


def _candidate_from_research(
    candidate: Any,
    *,
    allocation_nature: Literal["risk_qualified_research", "observation_only", "unavailable"],
    expected_cutoff: date | None,
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
    )


def _price_label(
    *,
    conditional_plan: ConditionalEntryPlan | None,
    observation_plan: ConditionalEntryPlan | None,
    expected_cutoff: date | None,
) -> tuple[str, Literal["conditional_entry", "observation_only", "unavailable"]]:
    if conditional_plan is not None:
        label = _format_plan(conditional_plan, expected_cutoff=expected_cutoff, observation=False)
        if label is not None:
            return label, "conditional_entry"
    if observation_plan is not None:
        label = _format_plan(observation_plan, expected_cutoff=expected_cutoff, observation=True)
        if label is not None:
            return f"仅观察：{label}（触及不等于可买）", "observation_only"
    return "—（未形成价格条件）", "unavailable"


def _format_plan(
    plan: ConditionalEntryPlan,
    *,
    expected_cutoff: date | None,
    observation: bool,
) -> str | None:
    if expected_cutoff is None or _as_date(plan.data_cutoff) != expected_cutoff:
        return None
    if plan.horizon != "一周" or plan.sessions != 5:
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
