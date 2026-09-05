"""Settle archived recommendation cohorts and report newly mature results.

The orchestration in this module is intentionally small and dependency
injected.  It reads only provider-verified, unadjusted overlays, delegates all
return arithmetic to :mod:`settle_recommendation_performance`, persists the
idempotent result records, and sends a maturity review only after a provider
accepts the notification.

The published-reference return and the next-session-open action simulation are
kept separate.  Observation and reconstructed cohorts are visibly labelled and
never promoted into the official action population.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol

import pandas as pd

from ashare_lab.domain.data_sources import DEFAULT_MARKET_OVERLAY_SOURCE_ID
from ashare_lab.domain.errors import AShareLabError
from ashare_lab.ports.market_data import normalize_symbol
from ashare_lab.ports.notifications import NotificationMessage
from ashare_lab.services.settle_recommendation_performance import (
    EvaluationMode,
    VerifiedDailyEvidence,
    archived_batch_from_mapping,
    archived_member_from_mapping,
    archived_report_from_mapping,
    batch_performance_record,
    member_performance_record,
    settle_recommendation_performance,
)


class RecommendationPerformanceRepository(Protocol):
    def list_recommendation_batches_pending_settlement(
        self, *, as_of: date | str | None = None
    ) -> list[dict[str, Any]]: ...

    def list_recommendation_members(self, batch_id: str) -> list[dict[str, Any]]: ...

    def record_recommendation_member_result(self, result: Mapping[str, Any]) -> None: ...

    def record_recommendation_batch_result(self, result: Mapping[str, Any]) -> None: ...

    def record_recommendation_settlement(
        self, *, batch_result: Mapping[str, Any], member_results: Sequence[Mapping[str, Any]]
    ) -> None: ...

    def list_maturity_results_pending_notification(
        self, *, limit: int = 100
    ) -> list[dict[str, Any]]: ...

    def get_recommendation_batch_performance(self, batch_id: str) -> dict[str, Any] | None: ...

    def record_recommendation_delivery_event(self, event: Mapping[str, Any]) -> None: ...


class VerifiedOverlayReader(Protocol):
    def read_verified_manifest(self, *, source_id: str) -> pd.DataFrame: ...

    def read_verified_daily(
        self, trade_date: date, *, source_id: str, asset_kind: str
    ) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class CorporateActionEvidence:
    """Authoritative point-in-time corporate-action coverage for one interval.

    Daily OHLCV ``prev_close`` is not sufficient to certify that a price move
    excludes a corporate action.  Only a separately sourced action or
    adjustment-factor feed may populate this object.
    """

    coverage_symbols: frozenset[str]
    action_dates_by_symbol: Mapping[str, frozenset[date]]


class CorporateActionEvidenceLoader(Protocol):
    def __call__(
        self,
        *,
        symbols: tuple[str, ...],
        start: date,
        end: date,
    ) -> CorporateActionEvidence: ...


def load_available_local_corporate_action_evidence(
    *, symbols: tuple[str, ...], start: date, end: date
) -> CorporateActionEvidence:
    """Optional shared evidence hook; absent infrastructure never implies clear."""

    try:
        from ashare_lab.services.corporate_action_evidence import (
            load_local_corporate_action_evidence,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "ashare_lab.services.corporate_action_evidence":
            raise
        return CorporateActionEvidence(frozenset(), {})
    evidence = load_local_corporate_action_evidence(symbols=symbols, start=start, end=end)
    return CorporateActionEvidence(evidence.coverage_symbols, evidence.action_dates_by_symbol)


@dataclass(frozen=True, slots=True)
class RecommendationPerformanceRunSummary:
    latest_verified_date: date | None
    pending_batches: int
    evaluated_batches: int
    persisted_batches: int
    mature_batches: int
    notification_attempts: int
    notification_accepted_batches: int
    notification_failed_batches: int
    accepted_channels: tuple[str, ...]
    failed_batch_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _NotificationAcceptance:
    successful_channels: tuple[str, ...]
    failed_channels: tuple[str, ...]


def run_recommendation_performance(
    *,
    repository: RecommendationPerformanceRepository,
    overlay_store: VerifiedOverlayReader,
    notifier: Callable[[NotificationMessage], object] | None = None,
    source_id: str = DEFAULT_MARKET_OVERLAY_SOURCE_ID,
    as_of: date | None = None,
    clock: Callable[[], datetime] | None = None,
    notification_limit: int = 100,
    corporate_action_loader: CorporateActionEvidenceLoader | None = None,
) -> RecommendationPerformanceRunSummary:
    """Settle all eligible batches, then notify each newly mature batch once.

    A batch error is isolated: its immutable archive remains untouched and it
    is eligible for a later retry.  A notification is considered delivered
    only when at least one injected provider reports acceptance.
    """

    now = _aware_utc((clock or (lambda: datetime.now(UTC)))())
    manifest = _normalized_manifest(
        overlay_store.read_verified_manifest(source_id=source_id),
        as_of=as_of,
    )
    latest = None if manifest.empty else pd.Timestamp(manifest.iloc[-1]["trade_date"]).date()
    pending = repository.list_recommendation_batches_pending_settlement(as_of=latest)
    evaluated = 0
    persisted = 0
    mature = 0
    failures: list[str] = []
    daily_cache: dict[date, pd.DataFrame] = {}

    for batch_row in pending:
        batch_id = str(batch_row.get("id") or batch_row.get("batch_id") or "")
        try:
            report = archived_report_from_mapping(batch_row)
            batch = archived_batch_from_mapping(batch_row)
            member_rows = repository.list_recommendation_members(batch.batch_id)
            members = tuple(archived_member_from_mapping(row) for row in member_rows)
            evidence = _build_verified_evidence(
                store=overlay_store,
                manifest=manifest,
                source_id=source_id,
                plan_date=report.plan_for_date,
                symbols=tuple(member.symbol for member in members),
                daily_cache=daily_cache,
                corporate_action_loader=corporate_action_loader,
            )
            result = settle_recommendation_performance(
                report=report,
                batch=batch,
                members=members,
                evidence=evidence,
            )
            evaluated += 1
            member_records = []
            for member_result in result.members:
                record = member_performance_record(member_result)
                record["evaluated_at"] = now.isoformat()
                member_records.append(record)
            batch_record = batch_performance_record(result)
            batch_record["evaluated_at"] = now.isoformat()
            repository.record_recommendation_settlement(
                batch_result=batch_record, member_results=member_records
            )
            persisted += 1
            if result.maturity_date is not None and result.status.value != "pending":
                mature += 1
        except (AShareLabError, KeyError, TypeError, ValueError, OSError, sqlite3.Error) as exc:
            # Fail closed and keep the cohort pending for a later, auditable retry.
            failures.append(batch_id or f"unknown:{type(exc).__name__}")

    attempts = 0
    accepted_batches = 0
    notification_failures = 0
    accepted_channels: list[str] = []
    if notifier is not None:
        for row in repository.list_maturity_results_pending_notification(limit=notification_limit):
            batch_id = str(row.get("batch_id") or "")
            report_id = str(row.get("report_id") or "")
            if not batch_id or not report_id:
                failures.append(batch_id or "notification:unknown")
                continue
            performance = repository.get_recommendation_batch_performance(batch_id)
            if performance is None:
                failures.append(batch_id)
                continue
            attempts += 1
            try:
                acceptance = _notification_acceptance(
                    notifier(render_maturity_notification(performance))
                )
            except Exception:  # provider adapters may surface network-specific errors
                notification_failures += 1
                continue
            if not acceptance.successful_channels:
                notification_failures += 1
                continue
            result_status = str(row.get("status") or "")
            result_method_version = str(row.get("method_version") or "")
            if not result_status or not result_method_version:
                failures.append(batch_id)
                continue
            accepted_batches += 1
            accepted_channels.extend(acceptance.successful_channels)
            for channel in acceptance.successful_channels:
                repository.record_recommendation_delivery_event(
                    {
                        "id": _maturity_event_id(
                            batch_id,
                            channel,
                            result_status=result_status,
                            result_method_version=result_method_version,
                        ),
                        "report_id": report_id,
                        "batch_id": batch_id,
                        "delivery_kind": "maturity_provider_accepted",
                        "channel": channel,
                        "attempted_at": now.isoformat(),
                        "provider_status": "provider_accepted",
                        "detail_json": {
                            "semantic": "provider_acceptance_not_device_confirmation",
                            "failed_channels": list(acceptance.failed_channels),
                            "result_status": result_status,
                            "result_method_version": result_method_version,
                        },
                    }
                )

    return RecommendationPerformanceRunSummary(
        latest_verified_date=latest,
        pending_batches=len(pending),
        evaluated_batches=evaluated,
        persisted_batches=persisted,
        mature_batches=mature,
        notification_attempts=attempts,
        notification_accepted_batches=accepted_batches,
        notification_failed_batches=notification_failures,
        accepted_channels=tuple(dict.fromkeys(accepted_channels)),
        failed_batch_ids=tuple(failures),
    )


def render_maturity_notification(performance: Mapping[str, Any]) -> NotificationMessage:
    """Render one transparent Chinese maturity review from stored result rows."""

    batch = _mapping(performance.get("batch"), "batch")
    result = _mapping(performance.get("result"), "result")
    members = performance.get("members")
    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
        raise ValueError("performance members are missing")

    label = str(batch.get("label") or batch.get("horizon_key") or "到期")
    plan_date = str(batch.get("plan_for_date") or "未知")
    maturity_date = str(result.get("maturity_date") or "未知")
    mode = str(batch.get("evaluation_mode") or "")
    archive_nature = str(batch.get("archive_nature") or "original")
    nature = _nature_label(mode, archive_nature)
    title = f"A股推荐到期复盘｜{label}｜{maturity_date}"

    lines = [
        f"**{nature}**",
        f"计划日：{plan_date}　到期日：{maturity_date}",
        "口径：发布参考价→到期收盘；原权重、不重标。固定持有价格表现未计分红、交易费用和期间止损换股，不代表实际账户收益。",
        _holding_clock_label(batch),
        "",
        "### 逐股结果",
    ]
    compact = [f"{nature} {label}（{plan_date}→{maturity_date}）"]
    for index, item in enumerate(members, start=1):
        member = _mapping(item, f"member[{index}]")
        recommendation = _mapping(member.get("recommendation"), "recommendation")
        member_result = member.get("result")
        name = str(recommendation.get("name") or recommendation.get("symbol") or "未知")
        symbol = str(recommendation.get("symbol") or "")
        weight = _as_optional_float(
            recommendation.get("stock_sleeve_weight", recommendation.get("sleeve_weight"))
        )
        weight_text = "—" if weight is None else _pct(weight)
        if not isinstance(member_result, Mapping):
            detail = "未形成可核验结果"
            lines.append(f"{index}. **{name} {symbol}**｜仓内{weight_text}｜{detail}")
            compact.append(f"{name}：{detail}")
            continue
        entry = _as_optional_float(member_result.get("entry_price"))
        close = _as_optional_float(member_result.get("maturity_close"))
        realized = _as_optional_float(member_result.get("realized_return"))
        member_details = member_result.get("details_json")
        member_details = member_details if isinstance(member_details, Mapping) else {}
        raw_change = _as_optional_float(member_details.get("raw_unadjusted_price_change"))
        status = str(member_result.get("status") or "unknown")
        price_limit_blocked = member_result.get("reason_code") == "entry_price_limit_exceeded"
        structure_invalidated = member_result.get("reason_code") == "entry_invalidated_before_fill"
        if entry is not None and close is not None and realized is not None:
            detail = f"{entry:.2f}→{close:.2f}，{_pct(realized)}"
        elif entry is not None and close is not None and raw_change is not None:
            action_warning = (
                "区间已检测到公司行动"
                if status == "corporate_action_detected"
                else "公司行动证据不足"
            )
            detail = (
                f"{entry:.2f}→{close:.2f}，原始变化{_pct(raw_change)}"
                f"（{action_warning}，不计正式收益）"
            )
        else:
            detail = _status_label(status, str(member_result.get("reason_code") or ""))
        if price_limit_blocked:
            detail = "开盘超过最高买价，保持现金；参考价观察 " + detail
        elif structure_invalidated:
            detail = "开盘已触及或跌破保护线，保持现金；参考价观察 " + detail
        lines.append(f"{index}. **{name} {symbol}**｜仓内{weight_text}｜{detail}")
        compact.append(f"{name}：{detail}")
        if mode == EvaluationMode.ACTION_SIMULATION.value:
            triggered = member_details.get("condition_triggered")
            simulated = _as_optional_float(member_details.get("simulated_action_return"))
            simulated_entry = _as_optional_float(member_details.get("simulated_entry_price"))
            if price_limit_blocked:
                lines.append(
                    "   - 条件已确认，但次日开盘超出归档最高买价；未模拟成交，该权重保持现金。"
                )
            elif structure_invalidated:
                lines.append(
                    "   - 条件曾确认，但次日开盘已触及或跌破冻结保护线；未模拟成交，该权重保持现金。"
                )
            elif triggered is False:
                lines.append("   - 行动条件未触发；该权重保持现金。")
            elif simulated is not None and simulated_entry is not None:
                lines.append(
                    f"   - 条件已触发；次交易日开盘模拟价{simulated_entry:.2f}，"
                    f"行动模拟收益{_pct(simulated)}。"
                )
            else:
                lines.append("   - 行动模拟证据不足，未计收益。")

    lines.extend(("", "### 组合结果"))
    stock_return = _as_optional_float(result.get("stock_sleeve_return"))
    account_return = _as_optional_float(result.get("account_return"))
    result_details = result.get("details_json")
    result_details = result_details if isinstance(result_details, Mapping) else {}
    raw_sleeve = _as_optional_float(result_details.get("raw_unadjusted_stock_sleeve_change"))
    raw_account = _as_optional_float(result_details.get("raw_unadjusted_account_change"))
    lines.append(f"- 股票仓收益：{_pct_or_dash(stock_return)}")
    lines.append(f"- 总资金收益：{_pct_or_dash(account_return)}")
    compact.append(f"股票仓{_pct_or_dash(stock_return)}；总资金{_pct_or_dash(account_return)}")
    if stock_return is None and raw_sleeve is not None:
        action_warning = (
            "区间已检测到公司行动"
            if str(result.get("reason_code") or "") == "corporate_action_detected"
            else "公司行动证据不足"
        )
        lines.append(
            "- 未核验原始价格变化："
            f"股票仓{_pct(raw_sleeve)}｜总资金{_pct_or_dash(raw_account)}"
            f"（{action_warning}，不计正式收益或胜率）"
        )
        compact.append(f"原始变化·股票仓{_pct(raw_sleeve)}（非正式收益）")
    if mode == EvaluationMode.ACTION_SIMULATION.value:
        simulated_sleeve = _as_optional_float(
            result_details.get("simulated_action_stock_sleeve_return")
        )
        simulated_account = _as_optional_float(
            result_details.get("simulated_action_account_return")
        )
        lines.append(f"- 次日开盘行动模拟·股票仓：{_pct_or_dash(simulated_sleeve)}")
        lines.append(f"- 次日开盘行动模拟·总资金：{_pct_or_dash(simulated_account)}")
        lines.append(f"- 未入场现金：{_pct_or_dash(_as_optional_float(result.get('cash_weight')))}")
    else:
        lines.append("- 本榜仅作观察，不并入正式行动业绩。")
    lines.append("系统未连接券商，不会自动下单。")

    compact_body = "\n".join(compact)
    if len(compact_body.encode("utf-8")) > 2_400:
        compact_body = "\n".join((compact[0], compact[-1]))
    return NotificationMessage(
        title=title,
        body="\n".join(lines),
        compact_body=compact_body,
        group="A股研究室·到期复盘",
    )


def _build_verified_evidence(
    *,
    store: VerifiedOverlayReader,
    manifest: pd.DataFrame,
    source_id: str,
    plan_date: date,
    symbols: tuple[str, ...],
    daily_cache: dict[date, pd.DataFrame],
    corporate_action_loader: CorporateActionEvidenceLoader | None,
) -> VerifiedDailyEvidence:
    normalized_symbols = tuple(dict.fromkeys(normalize_symbol(value) for value in symbols))
    rows = manifest.loc[pd.to_datetime(manifest["trade_date"]).dt.date >= plan_date]
    if rows.empty:
        return VerifiedDailyEvidence(
            prices=_empty_prices(),
            session_dates=(),
            source_adjustment="none",
        )

    rows = rows.sort_values("trade_date").reset_index(drop=True)
    session_dates: list[date] = []
    previous: date | None = None
    for row in rows.itertuples(index=False):
        trade_date = pd.Timestamp(row.trade_date).date()
        predecessor = pd.Timestamp(row.previous_trade_date).date()
        if previous is not None and predecessor != previous:
            break
        session_dates.append(trade_date)
        previous = trade_date

    load_dates = [*session_dates]
    selected_frames: list[pd.DataFrame] = []
    for trade_date in load_dates:
        frame = daily_cache.get(trade_date)
        if frame is None:
            frame = store.read_verified_daily(
                trade_date,
                source_id=source_id,
                asset_kind="stocks",
            )
            daily_cache[trade_date] = frame
        normalized = frame.copy()
        normalized["symbol"] = normalized["symbol"].map(normalize_symbol)
        chosen = normalized.loc[normalized["symbol"].isin(normalized_symbols)].copy()
        selected_frames.append(chosen)

    prices = pd.concat(selected_frames, ignore_index=True) if selected_frames else _empty_prices()
    suspended_dates: dict[str, frozenset[date]] = {}
    for symbol in normalized_symbols:
        symbol_rows = prices.loc[prices["symbol"] == symbol].copy()
        symbol_rows["_date"] = pd.to_datetime(symbol_rows["trade_date"]).dt.date
        symbol_rows = symbol_rows.sort_values("_date")
        found_dates = set(symbol_rows["_date"])
        interior_missing = {
            value
            for value in session_dates
            if value not in found_dates
            and any(found < value for found in found_dates)
            and any(found > value for found in found_dates)
        }
        if interior_missing:
            suspended_dates[symbol] = frozenset(interior_missing)

    corporate_action = (
        CorporateActionEvidence(frozenset(), {})
        if corporate_action_loader is None or not session_dates
        else corporate_action_loader(
            symbols=normalized_symbols,
            start=plan_date,
            end=session_dates[-1],
        )
    )

    return VerifiedDailyEvidence(
        prices=prices,
        session_dates=tuple(session_dates),
        source_adjustment="none",
        corporate_action_coverage_symbols=corporate_action.coverage_symbols,
        corporate_action_dates_by_symbol=corporate_action.action_dates_by_symbol,
        suspended_dates_by_symbol=suspended_dates,
    )


def _normalized_manifest(frame: pd.DataFrame, *, as_of: date | None) -> pd.DataFrame:
    required = {"trade_date", "previous_trade_date", "adjustment"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("verified manifest is missing: " + ", ".join(sorted(missing)))
    output = frame.copy()
    output["trade_date"] = pd.to_datetime(output["trade_date"], errors="raise")
    output["previous_trade_date"] = pd.to_datetime(output["previous_trade_date"], errors="raise")
    if as_of is not None:
        output = output.loc[output["trade_date"].dt.date <= as_of]
    if not output.empty and not bool(output["adjustment"].astype(str).eq("none").all()):
        raise ValueError("maturity settlement requires unadjusted verified overlays")
    if bool(output["trade_date"].duplicated().any()):
        raise ValueError("verified manifest has duplicate trade dates")
    return output.sort_values("trade_date").reset_index(drop=True)


def _notification_acceptance(value: object) -> _NotificationAcceptance:
    if value is None:
        return _NotificationAcceptance((), ())
    successful = getattr(value, "successful_channels", ())
    failed = getattr(value, "failed_channels", ())
    return _NotificationAcceptance(
        tuple(str(channel) for channel in successful),
        tuple(str(channel) for channel in failed),
    )


def _nature_label(mode: str, archive_nature: str) -> str:
    if archive_nature == "reconstructed" or mode == "reconstructed_observation":
        return "历史重建观察组合（不计入正式行动业绩）"
    if mode == EvaluationMode.ACTION_SIMULATION.value:
        return "正式行动组合"
    return "原始观察组合（不计入正式行动业绩）"


def _holding_clock_label(batch: Mapping[str, Any]) -> str:
    metadata = batch.get("metadata_json") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    if metadata.get("holding_clock") == "calendar":
        return "到期时钟：自然周/月/年；非交易日顺延至下一交易日。"
    return "到期时钟：原归档交易日口径（5/10/20/60/120/252日）。"


def _status_label(status: str, reason: str) -> str:
    labels = {
        "not_entered": "条件未触发，保持现金",
        "needs_review": "证据需复核，未计收益",
        "unavailable": "数据不可用，未计收益",
        "pending": "尚未到期",
    }
    return labels.get(status, f"未计收益（{reason or status}）")


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _pct_or_dash(value: float | None) -> str:
    return "—" if value is None else _pct(value)


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _maturity_event_id(
    batch_id: str,
    channel: str,
    *,
    result_status: str,
    result_method_version: str,
) -> str:
    digest = hashlib.sha256(
        (f"{batch_id}|{channel}|{result_status}|{result_method_version}|maturity-v2").encode()
    ).hexdigest()[:24]
    return f"maturity-delivery:{digest}"


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing")
    return value


def _empty_prices() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "prev_close",
            "volume_shares",
            "amount_cny",
        ]
    )


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC)
