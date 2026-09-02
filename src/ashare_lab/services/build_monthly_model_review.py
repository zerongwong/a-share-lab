"""Build a read-only monthly review of matured recommendation cohorts.

The review never mutates an archived prediction and never promotes an
observation cohort into the formal action population.  It is deliberately a
descriptive audit: weak outcomes create versioned *experiment proposals*, not
automatic parameter changes.  Any proposed model upgrade still requires
point-in-time validation and explicit user approval.
"""

from __future__ import annotations

import calendar
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from ashare_lab.ports.notifications import NotificationMessage

MONTHLY_REVIEW_METHOD_VERSION = "monthly-model-review-v0.1.0"
MIN_MATURE_BATCHES_FOR_DIRECTIONAL_CONCLUSION = 12
MIN_BENCHMARK_COVERAGE_FOR_DIRECTIONAL_CONCLUSION = 0.8
MONTHLY_REVIEW_STATE_VERSION = 1

_HORIZONS: tuple[tuple[str, int, str], ...] = (
    ("1w", 5, "1周"),
    ("2w", 10, "2周"),
    ("1m", 20, "1个月"),
    ("3m", 60, "3个月"),
    ("6m", 120, "6个月"),
    ("1y", 252, "1年"),
)


class ReviewPopulation(StrEnum):
    FORMAL_ACTION = "formal_action"
    ORIGINAL_OBSERVATION = "original_observation"
    RECONSTRUCTED_OBSERVATION = "reconstructed_observation"


class MonthlyReviewRepository(Protocol):
    def list_recommendation_reports(self, *, limit: int = 100) -> list[dict[str, Any]]: ...

    def list_recommendation_batches(self, report_id: str) -> list[dict[str, Any]]: ...

    def get_recommendation_batch_result(self, batch_id: str) -> dict[str, Any] | None: ...

    def list_recommendation_members(self, batch_id: str) -> list[dict[str, Any]]: ...

    def list_recommendation_member_results(self, batch_id: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class BenchmarkEvidence:
    """Same-interval benchmark evidence supplied by an independent caller."""

    batch_id: str
    benchmark_id: str
    plan_for_date: date
    maturity_date: date
    data_cutoff: date
    evaluated_at: date
    benchmark_return: float
    adjustment: str
    source: str
    method_version: str


@dataclass(frozen=True, slots=True)
class ReviewedBatch:
    batch_id: str
    report_id: str
    plan_for_date: date
    maturity_date: date
    primary_return: float | None
    stock_sleeve_return: float | None
    account_return: float | None
    benchmark_return: float | None
    relative_return: float | None
    result_status: str
    diagnostic_evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HorizonMonthlyReview:
    population: ReviewPopulation
    horizon_key: str
    holding_sessions: int
    label: str
    mature_batch_count: int
    valid_return_count: int
    mean_weighted_portfolio_return: float | None
    mean_stock_sleeve_return: float | None
    mean_account_return: float | None
    benchmark_available_count: int
    mean_benchmark_return: float | None
    mean_relative_return: float | None
    negative_batches: tuple[ReviewedBatch, ...]
    below_benchmark_batches: tuple[ReviewedBatch, ...]
    member_count: int
    data_return_member_count: int
    company_action_evidence_member_count: int
    company_action_clear_member_count: int
    untriggered_member_count: int
    evidence_assessment: str


@dataclass(frozen=True, slots=True)
class ExperimentProposal:
    proposal_id: str
    population: ReviewPopulation
    horizon_key: str
    hypothesis: str
    validation_plan: str
    status: str = "proposal_only_user_confirmation_required"


@dataclass(frozen=True, slots=True)
class MonthlyModelReview:
    review_month: str
    period_start: date
    period_end: date
    generated_as_of: date
    method_version: str
    horizon_reviews: tuple[HorizonMonthlyReview, ...]
    experiment_proposals: tuple[ExperimentProposal, ...]
    excluded_batches: tuple[str, ...]
    evidence_gaps: tuple[str, ...]
    archive_scan_truncated: bool
    conclusion: str


@dataclass(frozen=True, slots=True)
class MonthlyReviewCompletionState:
    """Small, independent dedup state for the scheduled-sync orchestrator."""

    completed_months: tuple[str, ...] = ()
    state_version: int = MONTHLY_REVIEW_STATE_VERSION


@dataclass(slots=True)
class _Accumulator:
    batches: list[ReviewedBatch]
    member_count: int = 0
    data_return_member_count: int = 0
    company_action_evidence_member_count: int = 0
    company_action_clear_member_count: int = 0
    untriggered_member_count: int = 0


def build_monthly_model_review(
    repository: MonthlyReviewRepository,
    *,
    review_month: date,
    as_of: date,
    benchmark_evidence_by_batch: Mapping[str, BenchmarkEvidence] | None = None,
    report_limit: int = 1000,
) -> MonthlyModelReview:
    """Review cohorts that matured in one fully completed calendar month.

    ``review_month`` may be any date in the target month.  ``as_of`` must be
    on or after that month's final calendar day, so a partial month can never
    be mistaken for a complete model review.  Horizons and archive populations
    are never pooled because their outcomes are overlapping, correlated
    evidence rather than independent trials.
    """

    if report_limit <= 0 or report_limit > 1000:
        raise ValueError("report_limit must be between 1 and 1000")
    period_start = review_month.replace(day=1)
    period_end = review_month.replace(
        day=calendar.monthrange(review_month.year, review_month.month)[1]
    )

    if as_of < period_end:
        raise ValueError("monthly review requires a completed calendar month")

    benchmark_map = benchmark_evidence_by_batch or {}
    reports = repository.list_recommendation_reports(limit=report_limit)
    accumulators = {
        (population, horizon_key): _Accumulator(batches=[])
        for population in ReviewPopulation
        for horizon_key, _, _ in _HORIZONS
    }
    excluded: list[str] = []
    evidence_gaps: set[str] = {
        "point_in_time_industry_membership_not_archived_for_performance_attribution"
    }

    for report in reports:
        report_id = str(report.get("id") or "")
        if not report_id:
            excluded.append("report_missing_id")
            continue
        for batch in repository.list_recommendation_batches(report_id):
            batch_id = str(batch.get("id") or batch.get("batch_id") or "")
            if not batch_id:
                excluded.append(f"{report_id}:batch_missing_id")
                continue
            result = repository.get_recommendation_batch_result(batch_id)
            if result is None or result.get("maturity_date") in (None, ""):
                continue
            if not _record_known_by_as_of(result, as_of=as_of):
                excluded.append(f"{batch_id}:result_not_known_as_of")
                continue
            try:
                maturity_date = _as_date(result["maturity_date"])
            except (TypeError, ValueError):
                excluded.append(f"{batch_id}:invalid_maturity_date")
                continue
            if not period_start <= maturity_date <= period_end:
                continue

            population = _population(report, batch)
            if population is None:
                excluded.append(f"{batch_id}:not_in_three_review_populations")
                continue
            horizon_key = str(batch.get("horizon_key") or "")
            if horizon_key not in {item[0] for item in _HORIZONS}:
                excluded.append(f"{batch_id}:unsupported_horizon")
                continue
            members = repository.list_recommendation_members(batch_id)
            raw_member_results = repository.list_recommendation_member_results(batch_id)
            member_results = [
                item for item in raw_member_results if _record_known_by_as_of(item, as_of=as_of)
            ]
            if len(member_results) != len(raw_member_results):
                evidence_gaps.add("member_result_not_known_as_of_excluded")
            benchmark = _validated_benchmark(
                benchmark_map.get(batch_id),
                batch_id=batch_id,
                plan_for_date=_as_date(batch.get("plan_for_date") or report["plan_for_date"]),
                maturity_date=maturity_date,
                as_of=as_of,
            )
            if batch_id in benchmark_map and benchmark is None:
                evidence_gaps.add(f"benchmark_evidence_invalid:{batch_id}")
            reviewed = _reviewed_batch(
                report=report,
                batch=batch,
                result=result,
                members=members,
                member_results=member_results,
                maturity_date=maturity_date,
                benchmark=benchmark,
                population=population,
            )
            accumulator = accumulators[(population, horizon_key)]
            accumulator.batches.append(reviewed)
            _add_member_coverage(accumulator, members, member_results)

    horizon_reviews = tuple(
        _summarize_horizon(
            population=population,
            horizon_key=horizon_key,
            holding_sessions=holding_sessions,
            label=label,
            accumulator=accumulators[(population, horizon_key)],
        )
        for population in ReviewPopulation
        for horizon_key, holding_sessions, label in _HORIZONS
    )
    archive_scan_truncated = len(reports) == report_limit
    if archive_scan_truncated:
        evidence_gaps.add("archive_scan_truncated_no_directional_conclusion")
    proposals = (
        ()
        if archive_scan_truncated
        else tuple(
            proposal
            for summary in horizon_reviews
            for proposal in _experiment_proposals(summary, review_month=period_start)
        )
    )
    mature_total = sum(item.mature_batch_count for item in horizon_reviews)
    benchmark_total = sum(item.benchmark_available_count for item in horizon_reviews)
    if mature_total and benchmark_total == 0:
        evidence_gaps.add(
            "benchmark_unavailable_no_production_loader_or_verified_same_interval_evidence"
        )
    elif benchmark_total < mature_total:
        evidence_gaps.add("benchmark_coverage_partial")
    conclusion = (
        "本月没有已成熟组合；继续积累不可变样本，不调整模型。"
        if mature_total == 0
        else "本月结果只用于描述性归因；任何模型升级均需用户确认和新的样本外验证。"
    )
    if archive_scan_truncated:
        conclusion = "档案扫描已达上限，样本总体不完整；只列事实，不下结论、不提出实验。"
    if (
        all(
            item.valid_return_count < MIN_MATURE_BATCHES_FOR_DIRECTIONAL_CONCLUSION
            for item in horizon_reviews
        )
        and not archive_scan_truncated
    ):
        conclusion = "样本不足，只列事实和证据缺口，不对模型优劣下结论，也不自动调整参数。"

    return MonthlyModelReview(
        review_month=period_start.strftime("%Y-%m"),
        period_start=period_start,
        period_end=period_end,
        generated_as_of=as_of,
        method_version=MONTHLY_REVIEW_METHOD_VERSION,
        horizon_reviews=horizon_reviews,
        experiment_proposals=proposals,
        excluded_batches=tuple(excluded),
        evidence_gaps=tuple(sorted(evidence_gaps)),
        archive_scan_truncated=archive_scan_truncated,
        conclusion=conclusion,
    )


def monthly_review_due(
    *,
    as_of: date,
    latest_verified_session: date | None,
    completed_months: Sequence[str] = (),
) -> date | None:
    """Return the previous complete month on its first safe verified session.

    The function intentionally has no fixed ``Day=1`` assumption.  A review
    becomes due only after a verified market session exists in ``as_of``'s
    current calendar month.  Therefore weekends and exchange holidays simply
    defer the review until the first successful daily sync.  A completed month
    key suppresses all later 15:30/18:30 retries for that month.
    """

    if latest_verified_session is None:
        return None
    if latest_verified_session > as_of:
        raise ValueError("latest_verified_session cannot be later than as_of")
    if (latest_verified_session.year, latest_verified_session.month) != (
        as_of.year,
        as_of.month,
    ):
        return None
    target = _previous_month(as_of)
    key = target.strftime("%Y-%m")
    normalized_completed = {_normalized_month_key(value) for value in completed_months}
    return None if key in normalized_completed else target


def load_monthly_review_completion_state(path: str | Path) -> MonthlyReviewCompletionState:
    """Read the secret-free local dedup file; a missing file means no completions."""

    source = Path(path).expanduser()
    if not source.exists():
        return MonthlyReviewCompletionState()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("monthly review state must be a JSON object")
    if int(payload.get("state_version", -1)) != MONTHLY_REVIEW_STATE_VERSION:
        raise ValueError("unsupported monthly review state version")
    raw_months = payload.get("completed_months", [])
    if not isinstance(raw_months, list):
        raise ValueError("monthly review completed_months must be a list")
    completed = tuple(sorted({_normalized_month_key(value) for value in raw_months}))
    return MonthlyReviewCompletionState(completed_months=completed)


def mark_monthly_review_completed(
    path: str | Path,
    *,
    review_month: str,
) -> MonthlyReviewCompletionState:
    """Atomically add one completed month after the caller's success boundary.

    The scheduled orchestrator decides that boundary (for example, after a
    notification provider accepts the concise review).  Calling this function
    repeatedly with the same month is idempotent.
    """

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    current = load_monthly_review_completion_state(destination)
    month = _normalized_month_key(review_month)
    updated = MonthlyReviewCompletionState(
        completed_months=tuple(sorted({*current.completed_months, month}))
    )
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    payload = {
        "state_version": updated.state_version,
        "completed_months": list(updated.completed_months),
    }
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return updated


def render_monthly_model_review_message(review: MonthlyModelReview) -> NotificationMessage:
    """Render a concise review message without submitting it to any provider."""

    title = f"A股模型月度复盘｜{review.review_month}"
    lines = [
        "到期组合按正式行动、原始观察、历史重建分开统计；各期限不合并。",
        (
            "正式行动按条件触发后的次日开盘模拟，未触发权重留现金；"
            "观察榜按发布参考价统计，均不代表真实账户总回报。"
        ),
    ]
    benchmark_gap = any(
        value.startswith("benchmark_unavailable_no_production_loader")
        for value in review.evidence_gaps
    )
    if benchmark_gap:
        lines.append("市场基准：当前自动流程未接入同区间点时点基准，因此不判断是否跑赢市场。")
    if review.archive_scan_truncated:
        lines.append("档案扫描已达上限：总体不完整，本月不生成方向性结论或实验建议。")
    compact = [f"{review.review_month} 模型复盘"]
    population_labels = {
        ReviewPopulation.FORMAL_ACTION: "正式行动",
        ReviewPopulation.ORIGINAL_OBSERVATION: "原始观察",
        ReviewPopulation.RECONSTRUCTED_OBSERVATION: "重建观察",
    }
    for population in ReviewPopulation:
        available = [
            item
            for item in review.horizon_reviews
            if item.population is population and item.mature_batch_count > 0
        ]
        lines.extend(("", f"### {population_labels[population]}"))
        if not available:
            lines.append("本月暂无成熟组合。")
            continue
        for item in available:
            benchmark_text = (
                "基准不可用"
                if item.mean_relative_return is None
                else f"相对基准{_pct(item.mean_relative_return)}"
            )
            return_text = (
                "收益不可核验"
                if item.mean_weighted_portfolio_return is None
                else (
                    ("总资金" if population is ReviewPopulation.FORMAL_ACTION else "股票仓")
                    + _pct(item.mean_weighted_portfolio_return)
                )
            )
            if item.mean_relative_return is not None:
                benchmark_text = f"股票仓相对基准{_pct(item.mean_relative_return)}"
            lines.append(
                f"- {item.label}：成熟{item.mature_batch_count}，组合{return_text}，"
                f"{benchmark_text}；负收益{len(item.negative_batches)}。"
            )
            compact.append(
                f"{population_labels[population]}·{item.label} "
                f"{item.mature_batch_count}组/{return_text}/{benchmark_text}"
            )
    negative_count = sum(len(item.negative_batches) for item in review.horizon_reviews)
    below_benchmark_count = sum(
        len(item.below_benchmark_batches) for item in review.horizon_reviews
    )
    unverifiable_count = sum(
        item.mature_batch_count - item.valid_return_count for item in review.horizon_reviews
    )
    lines.extend(
        (
            "",
            "### 本月问题",
            f"负收益组合{negative_count}个；弱于可比基准{below_benchmark_count}个；"
            f"收益不可核验{unverifiable_count}个。",
        )
    )
    if review.experiment_proposals:
        lines.extend(("", "### 待讨论实验"))
        for proposal in review.experiment_proposals[:3]:
            lines.append(f"- {proposal.hypothesis}（仅建议，需确认）")
    else:
        lines.append("实验建议：样本不足，本月不提出模型改动。")
    lines.extend(
        (
            "",
            f"结论：{review.conclusion}",
            "只提出版本化实验建议；不会用事后数据改写旧推荐，也不会自动改参数。",
        )
    )
    compact.append(review.conclusion)
    return NotificationMessage(
        title=title,
        body="\n".join(lines),
        compact_body="\n".join(compact),
        group="A股研究室·模型复盘",
    )


def _population(
    report: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> ReviewPopulation | None:
    archive_nature = str(batch.get("archive_nature") or report.get("archive_nature") or "")
    mode = str(batch.get("evaluation_mode") or "")
    delivered = _as_bool(batch.get("delivery_accepted", False))
    if archive_nature == "reconstructed" and mode == "reconstructed_observation":
        return ReviewPopulation.RECONSTRUCTED_OBSERVATION
    if archive_nature != "original":
        return None
    if mode == "action_simulation" and delivered:
        return ReviewPopulation.FORMAL_ACTION
    if mode == "observation_simulation":
        return ReviewPopulation.ORIGINAL_OBSERVATION
    return None


def _reviewed_batch(
    *,
    report: Mapping[str, Any],
    batch: Mapping[str, Any],
    result: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    member_results: Sequence[Mapping[str, Any]],
    maturity_date: date,
    benchmark: BenchmarkEvidence | None,
    population: ReviewPopulation,
) -> ReviewedBatch:
    result_details = result.get("details_json")
    result_details = result_details if isinstance(result_details, Mapping) else {}
    if population is ReviewPopulation.FORMAL_ACTION:
        stock_return = _optional_finite_float(
            result_details.get("simulated_action_stock_sleeve_return")
        )
        account_return = _optional_finite_float(
            result_details.get("simulated_action_account_return")
        )
        # Older terminal all-cash records may predate explicit zero-valued
        # simulation fields.  Their archived action result is nevertheless
        # exactly zero because no weight entered and none was redistributed.
        if str(result.get("status") or "") in {"no_entry", "no_entries"}:
            stock_return = 0.0 if stock_return is None else stock_return
            account_return = 0.0 if account_return is None else account_return
        return_basis = "conditional_next_open_simulation_untriggered_cash"
    else:
        stock_return = _optional_finite_float(result.get("stock_sleeve_return"))
        account_return = _optional_finite_float(result.get("account_return"))
        return_basis = "archived_reference_price_observation"
    primary_return = (
        account_return if population is ReviewPopulation.FORMAL_ACTION else stock_return
    )
    benchmark_return = None if benchmark is None else benchmark.benchmark_return
    relative_return = (
        None
        if stock_return is None or benchmark_return is None
        else stock_return - benchmark_return
    )
    member_result_by_id = {str(item.get("member_id") or ""): item for item in member_results}
    primary_structures = Counter(
        str(member.get("primary_structure"))
        for member in members
        if member.get("primary_structure") not in (None, "")
    )
    untriggered = sum(
        _member_untriggered(member_result_by_id.get(str(member.get("id") or "")))
        for member in members
    )
    valid_returns = sum(
        _optional_finite_float(item.get("realized_return")) is not None for item in member_results
    )
    action_known = sum(
        item.get("company_action_clear") in (0, 1, False, True) for item in member_results
    )
    batch_metadata = batch.get("metadata_json")
    batch_metadata = batch_metadata if isinstance(batch_metadata, Mapping) else {}
    evidence = (
        f"candidate_count={len(members)}",
        "industry_concentration=unavailable:not_archived_point_in_time",
        f"cycle_label={report.get('cycle_label') or 'unavailable'}",
        "primary_structure="
        + (
            ",".join(f"{key}:{value}" for key, value in sorted(primary_structures.items()))
            or "unavailable"
        ),
        "risk_lcb="
        + str(
            batch_metadata.get("risk_nature") or batch_metadata.get("failure_code") or "unavailable"
        ),
        f"untriggered={untriggered}/{len(members)}",
        f"member_return_coverage={valid_returns}/{len(members)}",
        f"company_action_evidence={action_known}/{len(members)}",
        f"return_basis={return_basis}",
        "causal_status=diagnostic_only_not_proven",
    )
    return ReviewedBatch(
        batch_id=str(batch.get("id") or batch.get("batch_id")),
        report_id=str(report.get("id") or batch.get("report_id")),
        plan_for_date=_as_date(batch.get("plan_for_date") or report["plan_for_date"]),
        maturity_date=maturity_date,
        primary_return=primary_return,
        stock_sleeve_return=stock_return,
        account_return=account_return,
        benchmark_return=benchmark_return,
        relative_return=relative_return,
        result_status=str(result.get("status") or "unknown"),
        diagnostic_evidence=evidence,
    )


def _add_member_coverage(
    accumulator: _Accumulator,
    members: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> None:
    accumulator.member_count += len(members)
    by_id = {str(item.get("member_id") or ""): item for item in results}
    for member in members:
        result = by_id.get(str(member.get("id") or ""))
        if result is None:
            continue
        if _optional_finite_float(result.get("realized_return")) is not None:
            accumulator.data_return_member_count += 1
        clear = result.get("company_action_clear")
        if clear in (0, 1, False, True):
            accumulator.company_action_evidence_member_count += 1
        if clear in (1, True):
            accumulator.company_action_clear_member_count += 1
        accumulator.untriggered_member_count += _member_untriggered(result)


def _member_untriggered(result: Mapping[str, Any] | None) -> int:
    if result is None:
        return 0
    if str(result.get("status") or "") == "not_entered":
        return 1
    details = result.get("details_json")
    return int(isinstance(details, Mapping) and details.get("condition_triggered") is False)


def _summarize_horizon(
    *,
    population: ReviewPopulation,
    horizon_key: str,
    holding_sessions: int,
    label: str,
    accumulator: _Accumulator,
) -> HorizonMonthlyReview:
    batches = tuple(accumulator.batches)
    primary_returns = tuple(
        item.primary_return for item in batches if item.primary_return is not None
    )
    stock_returns = tuple(
        item.stock_sleeve_return for item in batches if item.stock_sleeve_return is not None
    )
    account_returns = tuple(
        item.account_return for item in batches if item.account_return is not None
    )
    benchmark_items = tuple(item for item in batches if item.benchmark_return is not None)
    benchmark_pairs = tuple(
        item
        for item in batches
        if item.relative_return is not None and item.benchmark_return is not None
    )
    assessment = (
        "evidence_insufficient_no_directional_conclusion"
        if len(primary_returns) < MIN_MATURE_BATCHES_FOR_DIRECTIONAL_CONCLUSION
        else "descriptive_threshold_met_causality_not_established"
    )
    return HorizonMonthlyReview(
        population=population,
        horizon_key=horizon_key,
        holding_sessions=holding_sessions,
        label=label,
        mature_batch_count=len(batches),
        valid_return_count=len(primary_returns),
        mean_weighted_portfolio_return=_mean(primary_returns),
        mean_stock_sleeve_return=_mean(stock_returns),
        mean_account_return=_mean(account_returns),
        benchmark_available_count=len(benchmark_items),
        mean_benchmark_return=_mean(
            tuple(
                item.benchmark_return
                for item in benchmark_items
                if item.benchmark_return is not None
            )
        ),
        mean_relative_return=_mean(
            tuple(
                item.relative_return for item in benchmark_pairs if item.relative_return is not None
            )
        ),
        negative_batches=tuple(
            item for item in batches if item.primary_return is not None and item.primary_return < 0
        ),
        below_benchmark_batches=tuple(
            item
            for item in batches
            if item.relative_return is not None and item.relative_return < 0
        ),
        member_count=accumulator.member_count,
        data_return_member_count=accumulator.data_return_member_count,
        company_action_evidence_member_count=(accumulator.company_action_evidence_member_count),
        company_action_clear_member_count=accumulator.company_action_clear_member_count,
        untriggered_member_count=accumulator.untriggered_member_count,
        evidence_assessment=assessment,
    )


def _experiment_proposals(
    summary: HorizonMonthlyReview,
    *,
    review_month: date,
) -> tuple[ExperimentProposal, ...]:
    if summary.mature_batch_count < MIN_MATURE_BATCHES_FOR_DIRECTIONAL_CONCLUSION:
        return ()
    if summary.valid_return_count < MIN_MATURE_BATCHES_FOR_DIRECTIONAL_CONCLUSION:
        return ()
    proposals: list[ExperimentProposal] = []
    prefix = f"{review_month:%Y-%m}-{summary.population.value}-{summary.horizon_key}"
    benchmark_coverage = summary.benchmark_available_count / summary.mature_batch_count
    if (
        benchmark_coverage >= MIN_BENCHMARK_COVERAGE_FOR_DIRECTIONAL_CONCLUSION
        and summary.mean_relative_return is not None
        and summary.mean_relative_return < 0
    ):
        proposals.append(
            ExperimentProposal(
                proposal_id=f"{prefix}-relative-strength-ablation-v1",
                population=summary.population,
                horizon_key=summary.horizon_key,
                hypothesis="行业内相对强度或基准过滤可能改善该期限的弱于基准表现。",
                validation_plan=(
                    "冻结当前版本作为对照，只在未来点时点数据上进行purged walk-forward；"
                    "同时报告换手、成本、最差折和多重比较调整后的区间。"
                ),
            )
        )
    if summary.member_count > 0 and summary.untriggered_member_count / summary.member_count >= 0.5:
        proposals.append(
            ExperimentProposal(
                proposal_id=f"{prefix}-entry-confirmation-ablation-v1",
                population=summary.population,
                horizon_key=summary.horizon_key,
                hypothesis="介入条件未触发比例偏高，确认规则可能与该期限不匹配。",
                validation_plan=(
                    "保持旧档案不变，预注册一组确认规则消融实验；不得用本月到期价格"
                    "反向调出最佳阈值，需在后续未见样本验证。"
                ),
            )
        )
    return tuple(proposals)


def _validated_benchmark(
    evidence: BenchmarkEvidence | None,
    *,
    batch_id: str,
    plan_for_date: date,
    maturity_date: date,
    as_of: date,
) -> BenchmarkEvidence | None:
    if evidence is None:
        return None
    if evidence.batch_id != batch_id:
        return None
    if evidence.plan_for_date != plan_for_date or evidence.maturity_date != maturity_date:
        return None
    if evidence.data_cutoff != maturity_date:
        return None
    if evidence.evaluated_at < evidence.data_cutoff or evidence.evaluated_at > as_of:
        return None
    if evidence.adjustment.strip().lower() != "none":
        return None
    if (
        not evidence.benchmark_id.strip()
        or not evidence.source.strip()
        or not evidence.method_version.strip()
    ):
        return None
    if not math.isfinite(evidence.benchmark_return) or evidence.benchmark_return <= -1:
        return None
    return evidence


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _record_known_by_as_of(record: Mapping[str, Any], *, as_of: date) -> bool:
    """Require an auditable observation timestamp and reject future knowledge.

    Result rows are mutable settlement observations.  A later retry can update
    a member row before its parent batch row, so each row must independently
    prove that it was known by the requested review date.
    """

    evaluated_at = record.get("evaluated_at")
    if evaluated_at in (None, ""):
        return False
    try:
        if _as_date(evaluated_at) > as_of:
            return False
        for field in ("updated_at", "data_cutoff"):
            value = record.get(field)
            if value not in (None, "") and _as_date(value) > as_of:
                return False
    except (TypeError, ValueError):
        return False
    return True


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return parsed


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _as_bool(value: Any) -> bool:
    return value in (True, 1, "1", "true", "True")


def _previous_month(value: date) -> date:
    if value.month == 1:
        return date(value.year - 1, 12, 1)
    return date(value.year, value.month - 1, 1)


def _normalized_month_key(value: Any) -> str:
    text = str(value)
    try:
        parsed = date.fromisoformat(f"{text}-01")
    except ValueError as exc:
        raise ValueError("completed month keys must use YYYY-MM") from exc
    if parsed.strftime("%Y-%m") != text:
        raise ValueError("completed month keys must use YYYY-MM")
    return text


def _pct(value: float) -> str:
    return f"{value:+.2%}"
