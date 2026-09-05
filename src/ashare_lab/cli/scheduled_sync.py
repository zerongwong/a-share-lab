"""LaunchAgent-safe completed-session synchronization.

This entrypoint is deliberately independent from Streamlit.  It serializes
scheduled runs with an advisory lock, writes only derived and sanitized status
events to a private rotating log, and sends best-effort failure/recovery
notifications through credentials already stored in macOS Keychain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import plistlib
import re
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from ashare_lab.adapters.macos_keychain import load_serverchan_sendkey
from ashare_lab.adapters.market_overlay_store import MarketOverlayStore
from ashare_lab.adapters.notification_channels import ServerChanNotificationChannel
from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.bootstrap import application_data_dir, project_root
from ashare_lab.domain.errors import AShareLabError
from ashare_lab.ports.notifications import NotificationMessage, NotificationUrgency
from ashare_lab.services.build_monthly_model_review import (
    MonthlyModelReview,
    build_monthly_model_review,
    build_verified_benchmark_evidence,
    load_monthly_review_completion_state,
    mark_monthly_review_completed,
    monthly_review_due,
)
from ashare_lab.services.daily_update_lock import daily_update_lock
from ashare_lab.services.dispatch_notifications import dispatch_notification
from ashare_lab.services.review_active_holdings import HoldingTreeReviewSummary
from ashare_lab.services.run_active_holding_review import run_active_holding_review
from ashare_lab.services.run_daily_update import DailyUpdateReport
from ashare_lab.services.run_holding_stop_shadows import (
    HoldingStopShadowRunSummary,
    run_holding_stop_shadows,
)
from ashare_lab.services.run_recommendation_performance import (
    RecommendationPerformanceRunSummary,
    load_available_local_corporate_action_evidence,
    run_recommendation_performance,
)
from ashare_lab.services.run_zero_budget_daily_update import run_zero_budget_daily_update

EXIT_CURRENT = 0
EXIT_INCOMPLETE = 1
EXIT_ERROR = 2

_MAX_LOG_BYTES = 1_048_576
_LOG_BACKUPS = 5
_URL = re.compile(r"https?://\S+", flags=re.IGNORECASE)
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_ -]?key|sendkey|device[_ -]?key|token|secret)\b\s*[:=]\s*\S+"
)
_QUOTED_CREDENTIAL = re.compile(
    r"""(?i)(["']?(?:api[_ -]?key|apikey|sendkey|device[_ -]?key|token|secret)["']?\s*[:=]\s*)["'][^"']+["']"""
)
_AUTHORIZATION = re.compile(r"(?i)\b(authorization\s*:\s*)(?:bearer\s+)?\S+")
_SERVERCHAN_SECRET = re.compile(r"\bSCT[A-Za-z0-9_-]{8,192}\b")
_INFOWAY_SECRET = re.compile(r"(?i)\b[a-f0-9]{24,64}-infoway\b")
_LAUNCHAGENT_LABEL = "com.zerong.asharelab.daily-sync"
_LAUNCHAGENT_MODULE = "ashare_lab.cli.scheduled_sync"


@dataclass(frozen=True, slots=True)
class NotificationSummary:
    configured_channels: tuple[str, ...] = ()
    successful_channels: tuple[str, ...] = ()
    failed_channels: tuple[str, ...] = ()

    @property
    def any_succeeded(self) -> bool:
        return bool(self.successful_channels)


@dataclass(frozen=True, slots=True)
class ScheduledSyncOutcome:
    exit_code: int
    event: dict[str, Any]


PerformanceRunner = Callable[..., RecommendationPerformanceRunSummary]
HoldingReviewRunner = Callable[..., HoldingTreeReviewSummary]
HoldingShadowRunner = Callable[..., HoldingStopShadowRunSummary]
MonthlyReviewBuilder = Callable[..., MonthlyModelReview]

# Preserve the scheduler's injection seam while switching production to the
# free, independently verified EOD chain.
run_daily_update = run_zero_budget_daily_update

_HOLDING_ACTIONS = ("hold", "tighten", "reduce", "exit", "review")
_HOLDING_STATUSES = {"ready", "partial", "data_not_ready", "no_holdings"}


def render_launchagent_plist(
    template_path: str | Path,
    output_path: str | Path,
    python_bin: str | Path,
    project_root: str | Path,
) -> Path:
    """Render one exact LaunchAgent plist without array-index mutation.

    macOS ``plutil -replace ProgramArguments.0`` can insert before an existing
    array element on some releases.  Replacing the complete array through
    ``plistlib`` guarantees there are exactly three arguments in the required
    order, including when filesystem paths contain spaces.
    """

    source = Path(template_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    # Keep the virtual-environment launcher path intact.  Resolving this
    # symlink would point launchd at the base interpreter and silently drop
    # the venv's installed package context.
    interpreter = os.path.abspath(os.fspath(Path(python_bin).expanduser()))
    working_directory = str(Path(project_root).expanduser().resolve())
    document = plistlib.loads(source.read_bytes())
    if not isinstance(document, dict) or document.get("Label") != _LAUNCHAGENT_LABEL:
        raise ValueError("daily sync LaunchAgent template has an unexpected label")
    if "KeepAlive" in document:
        raise ValueError("daily sync LaunchAgent must not contain KeepAlive")
    if document.get("StartCalendarInterval") != [
        {"Hour": 15, "Minute": 30},
        {"Hour": 18, "Minute": 30},
        {"Hour": 20, "Minute": 0},
    ]:
        raise ValueError("daily sync LaunchAgent schedule is not the approved three-run contract")
    document["ProgramArguments"] = [interpreter, "-m", _LAUNCHAGENT_MODULE]
    document["WorkingDirectory"] = working_directory
    destination.write_bytes(plistlib.dumps(document, fmt=plistlib.FMT_XML, sort_keys=False))
    os.chmod(destination, 0o600)
    return destination


def send_scheduled_notification(message: NotificationMessage) -> NotificationSummary:
    """Submit only to the user-approved ServerChan channel, without exposing keys."""

    channels: list[Any] = []
    configured: list[str] = []
    construction_failures: list[str] = []
    channel_specs = (("serverchan", load_serverchan_sendkey, ServerChanNotificationChannel),)
    try:
        for name, loader, constructor in channel_specs:
            try:
                secret = loader()
            except (AShareLabError, ValueError):
                construction_failures.append(name)
                continue
            if not secret:
                continue
            configured.append(name)
            try:
                channels.append(constructor(secret))
            except (AShareLabError, ValueError):
                construction_failures.append(name)

        report = dispatch_notification(channels, message)
        failed = tuple(dict.fromkeys((*construction_failures, *report.failed_channels)))
        return NotificationSummary(
            configured_channels=tuple(configured),
            successful_channels=report.successful_channels,
            failed_channels=failed,
        )
    finally:
        for channel in channels:
            close = getattr(channel, "close", None)
            if callable(close):
                close()


def run_scheduled_sync(
    *,
    csmar_root: str | Path | None = None,
    overlay_root: str | Path | None = None,
    scheduler_root: str | Path | None = None,
    log_root: str | Path | None = None,
    clock: Callable[[], datetime] | None = None,
    _run_update: Callable[..., DailyUpdateReport] = run_daily_update,
    _notifier: Callable[[NotificationMessage], NotificationSummary] = send_scheduled_notification,
    _performance_runner: PerformanceRunner | None = None,
    _holding_review_runner: HoldingReviewRunner | None = None,
    _holding_shadow_runner: HoldingShadowRunner | None = None,
    _monthly_review_builder: MonthlyReviewBuilder | None = None,
) -> ScheduledSyncOutcome:
    """Run one serialized update and return a process exit contract.

    Exit 0 means the verified data are current, already current, a non-trading
    day produced no work, or another invocation already owns the lock.  Exit 1
    means the provider call completed but the verified cutoff did not advance
    through the latest completed session.  Exit 2 is a setup/runtime error.
    Notification delivery is best effort and never changes the data exit code.
    """

    fallback_now = datetime.now(UTC)
    try:
        fallback_log_path = (
            Path(log_root or Path.home() / "Library" / "Logs" / "A股研究助手").expanduser()
            / "daily-sync.jsonl"
        )
    except (TypeError, ValueError):
        fallback_log_path = Path.home() / "Library" / "Logs" / "A股研究助手" / "daily-sync.jsonl"
    try:
        now = _aware_utc((clock or (lambda: datetime.now(UTC)))())
        default_scheduler_root = (
            application_data_dir() / "scheduler" if scheduler_root is None else scheduler_root
        )
        resolved_scheduler_root = Path(default_scheduler_root).expanduser().resolve()
        resolved_log_root = (
            Path(log_root or Path.home() / "Library" / "Logs" / "A股研究助手")
            .expanduser()
            .resolve()
        )
    except Exception as exc:  # noqa: BLE001 - initialization must return the exit contract
        event = _base_event(fallback_now, status="scheduler_setup_error", exit_code=EXIT_ERROR)
        event["error_type"] = type(exc).__name__
        event["reason"] = "scheduler_initialization_error"
        with suppress(OSError):
            _write_log_event(fallback_log_path, event)
        return ScheduledSyncOutcome(EXIT_ERROR, event)
    lock_path = resolved_scheduler_root / "daily-sync.lock"
    state_path = resolved_scheduler_root / "daily-sync-state.json"
    log_path = resolved_log_root / "daily-sync.jsonl"

    try:
        with daily_update_lock(lock_path) as acquired:
            if not acquired:
                event = _base_event(now, status="already_running", exit_code=EXIT_CURRENT)
                return ScheduledSyncOutcome(EXIT_CURRENT, event)

            prior_state = _read_state(state_path)
            try:
                report = _run_update(csmar_root=csmar_root, overlay_root=overlay_root)
            except (AShareLabError, ValueError) as exc:
                event = _base_event(now, status="error", exit_code=EXIT_ERROR)
                event["error_type"] = type(exc).__name__
                event["reason"] = "update_entrypoint_error"
                _handle_failure(
                    event,
                    state_path=state_path,
                    prior_state=prior_state,
                    notifier=_notifier,
                )
                _write_log_event(log_path, event)
                return ScheduledSyncOutcome(EXIT_ERROR, event)
            except Exception as exc:  # noqa: BLE001 - final scheduler boundary
                event = _base_event(now, status="error", exit_code=EXIT_ERROR)
                event["error_type"] = type(exc).__name__
                event["reason"] = "unexpected_scheduler_error"
                _handle_failure(
                    event,
                    state_path=state_path,
                    prior_state=prior_state,
                    notifier=_notifier,
                )
                _write_log_event(log_path, event)
                return ScheduledSyncOutcome(EXIT_ERROR, event)

            event = _report_event(report, now)
            if report.current_through_latest_complete_session:
                if bool(prior_state.get("failure_active")):
                    recovery = _safe_notify(_notifier, _recovery_message(report))
                    event["recovery_notification_successful_channels"] = list(
                        recovery.successful_channels
                    )
                    event["recovery_notification_failed_channels"] = list(recovery.failed_channels)
                if _performance_runner is not None:
                    _run_performance_reviews(
                        event,
                        runner=_performance_runner,
                        overlay_root=Path(
                            overlay_root or application_data_dir() / "cache" / "market_overlay"
                        )
                        .expanduser()
                        .resolve(),
                        scheduler_root=resolved_scheduler_root,
                        common_cutoff=report.common_cutoff,
                        notifier=_notifier,
                        now=now,
                    )
                if _holding_review_runner is not None:
                    _run_local_holding_review(
                        event,
                        runner=_holding_review_runner,
                        csmar_root=csmar_root,
                        overlay_root=overlay_root,
                        scheduler_root=resolved_scheduler_root,
                        common_cutoff=report.common_cutoff,
                        now=now,
                    )
                if _holding_shadow_runner is not None:
                    _run_local_holding_shadow(
                        event,
                        runner=_holding_shadow_runner,
                        csmar_root=csmar_root,
                        overlay_root=overlay_root,
                        scheduler_root=resolved_scheduler_root,
                        common_cutoff=report.common_cutoff,
                        now=now,
                    )
                if _monthly_review_builder is not None:
                    _run_local_monthly_review(
                        event,
                        builder=_monthly_review_builder,
                        scheduler_root=resolved_scheduler_root,
                        common_cutoff=report.common_cutoff,
                        performance_ready=event.get("performance_status") in (None, "completed"),
                        overlay_root=Path(
                            overlay_root or application_data_dir() / "cache" / "market_overlay"
                        )
                        .expanduser()
                        .resolve(),
                    )
                _write_state(
                    state_path,
                    {
                        "failure_active": False,
                        "last_success_common_cutoff": report.common_cutoff.isoformat(),
                        "updated_at": now.isoformat().replace("+00:00", "Z"),
                    },
                )
                _write_log_event(log_path, event)
                return ScheduledSyncOutcome(EXIT_CURRENT, event)

            _handle_failure(
                event,
                state_path=state_path,
                prior_state=prior_state,
                notifier=_notifier,
            )
            _write_log_event(log_path, event)
            return ScheduledSyncOutcome(EXIT_INCOMPLETE, event)
    except Exception as exc:  # noqa: BLE001 - final local scheduler boundary
        event = _base_event(now, status="scheduler_setup_error", exit_code=EXIT_ERROR)
        event["error_type"] = type(exc).__name__
        event["reason"] = "scheduler_local_state_unavailable"
        with suppress(OSError):
            _write_log_event(log_path, event)
        return ScheduledSyncOutcome(EXIT_ERROR, event)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "独立于网页的收盘数据定时同步；密钥只从macOS钥匙串读取，不会连接券商或自动下单。"
        )
    )
    parser.add_argument(
        "--csmar-root",
        type=Path,
        default=application_data_dir() / "cache" / "csmar",
        help="只读CSMAR DuckDB目录",
    )
    parser.add_argument(
        "--overlay-root",
        type=Path,
        default=application_data_dir() / "cache" / "market_overlay",
        help="零预算三源已验证收盘增量目录",
    )
    parser.add_argument(
        "--scheduler-root",
        type=Path,
        default=application_data_dir() / "scheduler",
        help="仅保存锁与脱敏调度状态的本机目录",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path.home() / "Library" / "Logs" / "A股研究助手",
        help="脱敏轮转日志目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outcome = run_scheduled_sync(
        csmar_root=args.csmar_root,
        overlay_root=args.overlay_root,
        scheduler_root=args.scheduler_root,
        log_root=args.log_root,
        _performance_runner=run_recommendation_performance,
        _holding_review_runner=run_active_holding_review,
        _holding_shadow_runner=run_holding_stop_shadows,
        _monthly_review_builder=build_monthly_model_review,
    )
    print(json.dumps(outcome.event, ensure_ascii=False, sort_keys=True, default=str))
    return outcome.exit_code


def _run_performance_reviews(
    event: dict[str, Any],
    *,
    runner: PerformanceRunner,
    overlay_root: Path,
    scheduler_root: Path,
    common_cutoff: date,
    notifier: Callable[[NotificationMessage], NotificationSummary],
    now: datetime,
) -> None:
    """Settle mature recommendation cohorts without changing sync success.

    The daily data result remains authoritative.  A local archive or provider
    failure is recorded as a stable category and retried by the next scheduled
    run; exception text and recommendation payloads never enter this log.
    """

    try:
        repository = SQLiteRepository(
            scheduler_root.parent / "research.db",
            project_root() / "migrations",
        )
        summary = runner(
            repository=repository,
            overlay_store=MarketOverlayStore(overlay_root),
            notifier=notifier,
            as_of=common_cutoff,
            clock=lambda: now,
            corporate_action_loader=load_available_local_corporate_action_evidence,
        )
    except Exception as exc:  # noqa: BLE001 - non-fatal post-sync boundary
        event["performance_status"] = "error_retry_later"
        event["performance_error_type"] = type(exc).__name__
        return

    failed_batch_ids = tuple(getattr(summary, "failed_batch_ids", ()))
    event.update(
        {
            "performance_status": ("partial_retry_later" if failed_batch_ids else "completed"),
            "performance_pending_batches": summary.pending_batches,
            "performance_evaluated_batches": summary.evaluated_batches,
            "performance_mature_batches": summary.mature_batches,
            "performance_notification_attempts": summary.notification_attempts,
            "performance_notification_accepted_batches": (summary.notification_accepted_batches),
            "performance_notification_failed_batches": summary.notification_failed_batches,
            "performance_failed_batch_count": len(failed_batch_ids),
        }
    )


def _run_local_holding_review(
    event: dict[str, Any],
    *,
    runner: HoldingReviewRunner,
    csmar_root: str | Path | None,
    overlay_root: str | Path | None,
    scheduler_root: Path,
    common_cutoff: date,
    now: datetime,
) -> None:
    """Review only the explicit local holding ledger after data are current.

    The full rows stay in the private SQLite ledger.  This scheduler event is
    intentionally aggregate-only: it cannot reveal a symbol, name, cost,
    price, protection line, or reason string.  Failure is non-fatal and the
    next scheduled run retries the same verified cutoff.
    """

    try:
        resolved_csmar_root = (
            Path(csmar_root or application_data_dir() / "cache" / "csmar").expanduser().resolve()
        )
        resolved_overlay_root = (
            Path(overlay_root or application_data_dir() / "cache" / "market_overlay")
            .expanduser()
            .resolve()
        )
        repository = SQLiteRepository(
            scheduler_root.parent / "research.db",
            project_root() / "migrations",
        )
        summary = runner(
            repository,
            dataset_root=resolved_csmar_root,
            overlay_root=resolved_overlay_root,
            as_of=common_cutoff,
            reviewed_at=now,
            persist=True,
        )
        raw_status = getattr(getattr(summary, "status", None), "value", None)
        status = raw_status if raw_status in _HOLDING_STATUSES else "unknown"
        rows = tuple(getattr(summary, "rows", ()))
        raw_counts = Counter(
            getattr(getattr(row, "action", None), "value", "unknown") for row in rows
        )
        action_counts = {action: int(raw_counts.get(action, 0)) for action in _HOLDING_ACTIONS}
        unknown_count = sum(
            count for action, count in raw_counts.items() if action not in _HOLDING_ACTIONS
        )
        if unknown_count:
            action_counts["unknown"] = int(unknown_count)
        event.update(
            {
                "holding_review_status": status,
                "holding_review_row_count": len(rows),
                "holding_review_ready_count": sum(
                    getattr(getattr(row, "status", None), "value", None) == "ready" for row in rows
                ),
                "holding_review_urgent_count": sum(
                    bool(getattr(row, "urgent", False)) for row in rows
                ),
                "holding_review_action_counts": action_counts,
            }
        )
    except Exception:  # noqa: BLE001 - non-fatal private post-sync boundary
        event["holding_review_status"] = "error_retry_later"


def _run_local_holding_shadow(
    event: dict[str, Any],
    *,
    runner: HoldingShadowRunner,
    csmar_root: str | Path | None,
    overlay_root: str | Path | None,
    scheduler_root: Path,
    common_cutoff: date,
    now: datetime,
) -> None:
    """Archive local-only stop experiments after the production review.

    The runner receives no notifier and its result contributes aggregate
    counters only.  A shadow failure is retryable and never changes the daily
    data result or any production holding action.
    """

    try:
        resolved_csmar_root = (
            Path(csmar_root or application_data_dir() / "cache" / "csmar").expanduser().resolve()
        )
        resolved_overlay_root = (
            Path(overlay_root or application_data_dir() / "cache" / "market_overlay")
            .expanduser()
            .resolve()
        )
        repository = SQLiteRepository(
            scheduler_root.parent / "research.db",
            project_root() / "migrations",
        )
        summary = runner(
            repository,
            dataset_root=resolved_csmar_root,
            overlay_root=resolved_overlay_root,
            as_of=common_cutoff,
            evaluated_at=now,
            persist=True,
        )
        raw_status = getattr(getattr(summary, "status", None), "value", None)
        status = (
            raw_status
            if raw_status in {"completed", "partial", "data_not_ready", "no_holdings"}
            else "unknown"
        )
        event.update(
            {
                "holding_shadow_status": status,
                "holding_shadow_row_count": int(getattr(summary, "observation_count", 0)),
                "holding_shadow_variant_count": int(getattr(summary, "variant_count", 0)),
            }
        )
    except Exception:  # noqa: BLE001 - non-fatal private experiment boundary
        event["holding_shadow_status"] = "error_retry_later"


def _run_local_monthly_review(
    event: dict[str, Any],
    *,
    builder: MonthlyReviewBuilder,
    scheduler_root: Path,
    common_cutoff: date,
    performance_ready: bool,
    overlay_root: Path,
) -> None:
    """Generate one private previous-month audit after its first verified session.

    No notification channel is accepted here.  The complete payload is written
    atomically with owner-only permissions; the public scheduler event records
    only a month key, stable status, and aggregate counts.  Completion is
    marked strictly after the archive write succeeds, so every failed step is
    retryable.
    """

    if not performance_ready:
        event["monthly_review_status"] = "deferred_performance_incomplete"
        return
    state_path = scheduler_root / "monthly-model-review-state.json"
    try:
        _ensure_private_directory(scheduler_root)
        completion = load_monthly_review_completion_state(state_path)
        due_month = monthly_review_due(
            as_of=common_cutoff,
            latest_verified_session=common_cutoff,
            completed_months=completion.completed_months,
        )
        if due_month is None:
            event["monthly_review_status"] = "not_due"
            return
        repository = SQLiteRepository(
            scheduler_root.parent / "research.db",
            project_root() / "migrations",
        )
        review = builder(
            repository,
            review_month=due_month,
            as_of=common_cutoff,
            benchmark_evidence_by_batch=build_verified_benchmark_evidence(
                repository,
                overlay_store=MarketOverlayStore(overlay_root),
                as_of=common_cutoff,
            ),
        )
        expected_month = due_month.strftime("%Y-%m")
        if review.review_month != expected_month or review.generated_as_of != common_cutoff:
            raise ValueError("monthly review identity does not match its due cutoff")
        archive_path = scheduler_root / "monthly-model-reviews" / f"{expected_month}.json"
        _write_private_json(archive_path, asdict(review))
        mark_monthly_review_completed(state_path, review_month=expected_month)
        os.chmod(state_path, 0o600)
        event.update(
            {
                "monthly_review_status": "completed_local_only",
                "monthly_review_month": expected_month,
                "monthly_review_horizon_count": len(review.horizon_reviews),
                "monthly_review_mature_batch_count": sum(
                    item.mature_batch_count for item in review.horizon_reviews
                ),
                "monthly_review_proposal_count": len(review.experiment_proposals),
                "monthly_review_excluded_batch_count": len(review.excluded_batches),
                "monthly_review_evidence_gap_count": len(review.evidence_gaps),
            }
        )
    except Exception:  # noqa: BLE001 - non-fatal private post-sync boundary
        event["monthly_review_status"] = "error_retry_later"


def _report_event(report: DailyUpdateReport, now: datetime) -> dict[str, Any]:
    current = bool(report.current_through_latest_complete_session)
    if current:
        status = "updated" if report.updated_sessions else "noop_current"
        exit_code = EXIT_CURRENT
    else:
        status = "incomplete"
        exit_code = EXIT_INCOMPLETE
    reasons = [_sanitize_text(item.reason) for item in report.quarantined_failures[:3]]
    event = _base_event(now, status=status, exit_code=exit_code)
    event.update(
        {
            "source_id": _sanitize_text(report.source_id, limit=64),
            "requested_complete_date": report.requested_complete_date.isoformat(),
            "latest_complete_session": report.latest_complete_session.isoformat(),
            "common_cutoff": report.common_cutoff.isoformat(),
            "updated_sessions": [value.isoformat() for value in report.updated_sessions],
            "unchanged_session_count": len(report.unchanged_sessions),
            "quarantined_failure_count": len(report.quarantined_failures),
            "failure_reasons": reasons,
            "provider_contract_changed": bool(report.provider_contract_changed),
            "csmar_mutated": bool(report.csmar_mutated),
        }
    )
    return event


def _base_event(now: datetime, *, status: str, exit_code: int) -> dict[str, Any]:
    return {
        "timestamp": now.isoformat().replace("+00:00", "Z"),
        "job": "com.zerong.asharelab.daily-sync",
        "status": status,
        "exit_code": exit_code,
    }


def _handle_failure(
    event: dict[str, Any],
    *,
    state_path: Path,
    prior_state: dict[str, Any],
    notifier: Callable[[NotificationMessage], NotificationSummary],
) -> None:
    fingerprint = _failure_fingerprint(event)
    already_delivered = (
        bool(prior_state.get("failure_active"))
        and prior_state.get("failure_fingerprint") == fingerprint
        and bool(prior_state.get("notification_succeeded"))
    )
    notification = NotificationSummary()
    if not already_delivered:
        notification = _safe_notify(notifier, _failure_message(event))
    event["notification_deduplicated"] = already_delivered
    event["notification_successful_channels"] = list(notification.successful_channels)
    event["notification_failed_channels"] = list(notification.failed_channels)
    previous_success = (
        bool(prior_state.get("notification_succeeded")) if already_delivered else False
    )
    _write_state(
        state_path,
        {
            "failure_active": True,
            "failure_fingerprint": fingerprint,
            "notification_succeeded": previous_success or notification.any_succeeded,
            "updated_at": str(event["timestamp"]),
        },
    )


def _failure_message(event: dict[str, Any]) -> NotificationMessage:
    target = event.get("latest_complete_session") or event.get("requested_complete_date") or "未知"
    cutoff = event.get("common_cutoff") or "未取得"
    reasons = event.get("failure_reasons") or [event.get("reason", "同步未完成")]
    reason = _sanitize_text(str(reasons[0]))
    body = (
        f"收盘数据未追平。\n目标完整交易日：{target}\n当前共同截止：{cutoff}\n"
        f"原因：{reason}\n原有已验证数据保持不变；系统没有连接券商或自动下单。"
    )
    return NotificationMessage(
        "A股收盘数据同步未完成",
        body,
        urgency=NotificationUrgency.TIME_SENSITIVE,
        group="A股研究室·数据",
    )


def _recovery_message(report: DailyUpdateReport) -> NotificationMessage:
    body = (
        f"收盘数据已经恢复并追平至{report.common_cutoff.isoformat()}。"
        "原有失败状态已清除；系统没有连接券商或自动下单。"
    )
    return NotificationMessage("A股收盘数据已恢复", body, group="A股研究室·数据")


def _safe_notify(
    notifier: Callable[[NotificationMessage], NotificationSummary],
    message: NotificationMessage,
) -> NotificationSummary:
    try:
        return notifier(message)
    except Exception:  # noqa: BLE001 - notifications must never mask the data result
        return NotificationSummary(failed_channels=("notification_boundary",))


def _failure_fingerprint(event: dict[str, Any]) -> str:
    payload = {
        "status": event.get("status"),
        "target": event.get("latest_complete_session") or event.get("requested_complete_date"),
        "cutoff": event.get("common_cutoff"),
        "error_type": event.get("error_type"),
        "reason": event.get("reason"),
        "failure_reasons": event.get("failure_reasons", []),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sanitize_text(value: str, *, limit: int = 500) -> str:
    text = " ".join(str(value).split())
    text = _URL.sub("[redacted-url]", text)
    text = _QUOTED_CREDENTIAL.sub(r"\1[redacted]", text)
    text = _CREDENTIAL_ASSIGNMENT.sub(r"\1=[redacted]", text)
    text = _AUTHORIZATION.sub(r"\1[redacted]", text)
    text = _SERVERCHAN_SECRET.sub("[redacted-serverchan-key]", text)
    text = _INFOWAY_SECRET.sub("[redacted-infoway-key]", text)
    return text[:limit] or "unspecified_failure"


def _write_log_event(path: Path, event: dict[str, Any]) -> None:
    _ensure_private_directory(path.parent)
    logger = logging.getLogger(f"ashare_lab.scheduled_sync.{hash(path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = RotatingFileHandler(
        path,
        maxBytes=_MAX_LOG_BYTES,
        backupCount=_LOG_BACKUPS,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    try:
        logger.info(json.dumps(event, ensure_ascii=False, sort_keys=True, default=str))
    finally:
        handler.close()
        logger.removeHandler(handler)
    os.chmod(path, 0o600)


def _read_state(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return document if isinstance(document, dict) else {}


def _write_state(path: Path, document: dict[str, Any]) -> None:
    _ensure_private_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_private_json(path: Path, document: Any) -> None:
    """Atomically write a complete local artifact with owner-only access."""

    _ensure_private_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _json_default(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("scheduler clock must be timezone-aware")
    return value.astimezone(UTC)


if __name__ == "__main__":
    raise SystemExit(main())
