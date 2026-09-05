"""Submit one deduplicated continuous digest, retaining legacy test/archive APIs.

Credentials are read only from macOS Keychain.  The command has no account,
brokerage or order capability.  Provider acceptance is not described as
end-device delivery confirmation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from ashare_lab.adapters.baostock_eod import BaoStockEodMarketData
from ashare_lab.adapters.infoway_eod import InfowayEodMarketData
from ashare_lab.adapters.macos_keychain import (
    load_bark_device_key,
    load_infoway_api_key,
    load_serverchan_sendkey,
)
from ashare_lab.adapters.market_overlay_store import MarketOverlayStore
from ashare_lab.adapters.notification_channels import (
    BarkNotificationChannel,
    ServerChanNotificationChannel,
)
from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.bootstrap import application_data_dir, project_root
from ashare_lab.domain.data_sources import DEFAULT_MARKET_OVERLAY_SOURCE_ID
from ashare_lab.domain.errors import AShareLabError, DataUnavailableError
from ashare_lab.ports.notifications import (
    MAX_COMPACT_NOTIFICATION_BODY_BYTES,
    NotificationMessage,
    PrivateImagePublisher,
    PrivateImagePublishReceipt,
    prepare_notification_for_channel,
)
from ashare_lab.services.archive_recommendation_report import (
    RecommendationArchiveBundle,
    archive_recommendation_report,
)
from ashare_lab.services.build_evening_digest import (
    EveningResearchDigest,
    format_cn_plan_date,
    render_evening_digest_bark_compact,
    render_evening_digest_markdown,
)
from ashare_lab.services.build_holding_chart_report import (
    HoldingChartBuildResult,
    HoldingChartBuildStatus,
    build_holding_chart_report,
)
from ashare_lab.services.chart_publisher_settings import (
    CLOUDFLARE_R2_PUBLISHER_ID,
    build_configured_chart_publisher,
)
from ashare_lab.services.daily_update_lock import daily_update_lock
from ashare_lab.services.holding_ledger import (
    get_active_holding_portfolio,
    holding_chart_delivery_channels,
    holding_chart_publisher_id,
    holding_knowledge_context,
    holding_summary_delivery_channels,
)
from ashare_lab.services.render_holding_chart_report import (
    COMPOSITE_WIDTH,
    MAX_HOLDING_COUNT,
    MIN_HOLDING_COUNT,
    holding_chart_composite_height,
)
from ashare_lab.services.review_active_holdings import HoldingTreeReviewSummary
from ashare_lab.services.run_active_holding_review import build_evening_holding_review

EXIT_OK = 0
EXIT_RETRY = 1
EXIT_ERROR = 2
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_MAX_LOG_BYTES = 1_048_576
_LOG_BACKUPS = 5
_TRADING_CALENDAR_LOOKAHEAD_DAYS = 14
_HOLDING_CHART_CHANNELS = frozenset({"serverchan"})
_HOLDING_CHART_MAX_URL_TTL_SECONDS = 3_600
_MAX_CHART_ATTEMPTS = 4
_NOTIFICATION_CHANNELS_ENV = "ASHARE_EVENING_NOTIFICATION_CHANNELS"
_SUPPORTED_NOTIFICATION_CHANNELS = ("serverchan", "bark")
_LOG_EVENT_KEYS = (
    "job",
    "status",
    "reason",
    "common_cutoff",
    "plan_for_date",
    "period_count",
    "configured_channels",
    "accepted_channels",
    "failed_channels",
    "provider_receipt_ids",
    "image_accepted_channels",
    "delivery_confirmed",
    "raw_data_exposed",
    "orders_enabled",
    "chart_status",
    "chart_reason",
    "chart_attempts",
    "text_already_accepted",
)


@dataclass(frozen=True, slots=True)
class EveningDigestOutcome:
    exit_code: int
    event: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EveningNotificationSummary:
    configured_channels: tuple[str, ...] = ()
    accepted_channels: tuple[str, ...] = ()
    failed_channels: tuple[str, ...] = ()
    provider_receipt_ids: tuple[str, ...] = ()
    image_accepted_channels: tuple[str, ...] = ()

    @property
    def any_accepted(self) -> bool:
        return bool(self.accepted_channels)


DigestBuilder = Callable[..., EveningResearchDigest]
Notifier = Callable[[NotificationMessage], EveningNotificationSummary]
LatestCutoffReader = Callable[[Path], date | None]
NextTradingDayResolver = Callable[[date], date]
RecommendationArchiver = Callable[
    [EveningResearchDigest, SQLiteRepository], RecommendationArchiveBundle
]
HoldingReviewBuilder = Callable[..., HoldingTreeReviewSummary]
HoldingChartBuilder = Callable[..., HoldingChartBuildResult]
HoldingChartPublisher = PrivateImagePublisher


@dataclass(slots=True, repr=False)
class _PublishedHoldingChart:
    publisher: PrivateImagePublisher = field(repr=False)
    receipt: PrivateImagePublishReceipt = field(repr=False)
    revoked: bool = False

    def revoke_once(self) -> None:
        if self.revoked:
            return
        self.revoked = True
        try:
            self.publisher.revoke(self.receipt.revoke_key)
        except Exception:  # noqa: BLE001 - never expose a signed URL or object key
            return


def run_evening_digest(
    *,
    csmar_root: str | Path | None = None,
    overlay_root: str | Path | None = None,
    reference_root: str | Path | None = None,
    state_root: str | Path | None = None,
    log_root: str | Path | None = None,
    decision_date: date | None = None,
    _build_digest: DigestBuilder | None = None,
    _notifier: Notifier | None = None,
    _latest_cutoff: LatestCutoffReader | None = None,
    _next_trading_day: NextTradingDayResolver | None = None,
    _repository: SQLiteRepository | None = None,
    _archive_digest: RecommendationArchiver | None = None,
    _build_holding_review: HoldingReviewBuilder = build_evening_holding_review,
    _build_holding_chart_report: HoldingChartBuilder = build_holding_chart_report,
    _holding_chart_publisher: HoldingChartPublisher | None = None,
    _clock: Callable[[], datetime] | None = None,
) -> EveningDigestOutcome:
    """Build/send once per verified common cutoff; never place an order."""

    try:
        fallback_log_path = (
            Path(log_root or Path.home() / "Library" / "Logs" / "A股研究助手").expanduser()
            / "evening-report.jsonl"
        )
    except (TypeError, ValueError):
        fallback_log_path = (
            Path.home() / "Library" / "Logs" / "A股研究助手" / "evening-report.jsonl"
        )
    try:
        data_root = application_data_dir()
        resolved_csmar = Path(csmar_root or data_root / "cache" / "csmar").expanduser().resolve()
        resolved_overlay = (
            Path(overlay_root or data_root / "cache" / "market_overlay").expanduser().resolve()
        )
        resolved_reference = (
            Path(reference_root or data_root / "cache" / "csmar_reference").expanduser().resolve()
        )
        resolved_state_root = Path(state_root or data_root / "scheduler").expanduser().resolve()
        resolved_log_root = (
            Path(log_root or Path.home() / "Library" / "Logs" / "A股研究助手")
            .expanduser()
            .resolve()
        )
        now = (_clock or (lambda: datetime.now(_SHANGHAI)))()
        if now.tzinfo is None:
            raise ValueError("evening clock must be timezone aware")
        now = now.astimezone(_SHANGHAI)
        target_date = decision_date or now.date()
        if not isinstance(target_date, date):
            raise TypeError("decision_date must be a date")
    except (OSError, TypeError, ValueError):
        outcome = _error("digest_initialization_error")
        _safe_write_log_outcome(fallback_log_path, outcome)
        return outcome

    state_path = resolved_state_root / "evening-digest-state.json"
    lock_path = resolved_state_root / "daily-sync.lock"
    log_path = resolved_log_root / "evening-report.jsonl"
    notifier = _notifier or send_evening_digest
    latest_reader = _latest_cutoff or latest_verified_overlay_cutoff
    next_trading_day = _next_trading_day or resolve_next_zero_budget_trading_day
    repository = _repository or SQLiteRepository(
        resolved_state_root.parent / "research.db",
        project_root() / "migrations",
    )
    archiver = _archive_digest or _archive_original_digest

    def finish(outcome: EveningDigestOutcome) -> EveningDigestOutcome:
        if outcome.exit_code != EXIT_OK and now.hour == 21 and now.minute >= 45:
            _send_final_failure_notice(
                repository=repository,
                notifier=notifier,
                state_root=resolved_state_root,
                dataset_root=resolved_csmar,
                overlay_root=resolved_overlay,
                now=now,
                holding_reviewer=_build_holding_review,
            )
        _safe_write_log_outcome(log_path, outcome)
        return outcome

    # ``RunAtLoad`` is independent of launchd's StartCalendarInterval and can
    # invoke this command after a login on any calendar day.  Keep the send
    # boundary in the real execution path as well: Python uses Monday=0, so
    # 4 and 5 are Friday and Saturday.  Return before locks, state or market
    # data are read and before the digest builder/notifier can be reached.
    if target_date.weekday() in {4, 5}:
        return finish(
            EveningDigestOutcome(
                EXIT_OK,
                _event("noop_weekend_send_window_closed"),
            )
        )
    if now.hour != 21:
        return finish(EveningDigestOutcome(EXIT_OK, _event("noop_outside_evening_window")))

    try:
        with daily_update_lock(lock_path) as acquired:
            if not acquired:
                return finish(
                    EveningDigestOutcome(
                        EXIT_RETRY,
                        _event("already_running", reason="daily_data_lock_busy"),
                    )
                )

            prior_state = _read_state(state_path)
            last_sent = _state_cutoff(prior_state)
            current_method_matches = (
                _build_digest is not None
                or prior_state.get("method_version") == "continuous-signal-v1"
            )
            latest = latest_reader(resolved_overlay)
            retry_chart = (
                prior_state.get("chart_status") == "pending"
                and int(prior_state.get("chart_attempts", 0)) < _MAX_CHART_ATTEMPTS
                and _state_plan_for_date(prior_state)
                == (target_date + timedelta(days=1)).isoformat()
            )
            if (
                latest is not None
                and last_sent is not None
                and latest <= last_sent
                and not retry_chart
                and current_method_matches
                and _state_plan_for_date(prior_state)
                == (target_date + timedelta(days=1)).isoformat()
            ):
                return finish(
                    EveningDigestOutcome(
                        EXIT_OK,
                        _event(
                            "noop_no_new_trading_day",
                            common_cutoff=last_sent.isoformat(),
                            plan_for_date=_state_plan_for_date(prior_state),
                        ),
                    )
                )

            if _build_digest is None:
                from ashare_lab.services.build_continuous_digest import (
                    build_continuous_research_digest,
                )

                def digest_builder(**kwargs):
                    return build_continuous_research_digest(
                        **kwargs, repository=repository, known_at=now
                    )
            else:
                digest_builder = _build_digest
            digest = digest_builder(
                dataset_root=resolved_csmar,
                overlay_root=resolved_overlay,
                reference_dataset_root=resolved_reference,
                decision_date=target_date,
            )
            text_already_accepted = last_sent == digest.common_cutoff and current_method_matches
            if (
                last_sent is not None
                and digest.common_cutoff <= last_sent
                and not retry_chart
                and current_method_matches
                and _state_plan_for_date(prior_state)
                == (target_date + timedelta(days=1)).isoformat()
            ):
                return finish(
                    EveningDigestOutcome(
                        EXIT_OK,
                        _event(
                            "noop_no_new_trading_day",
                            common_cutoff=digest.common_cutoff.isoformat(),
                            plan_for_date=_state_plan_for_date(prior_state),
                        ),
                    )
                )

            cutoff = digest.common_cutoff.isoformat()
            try:
                plan_for_date = _validate_next_trading_day(
                    digest.common_cutoff,
                    next_trading_day(digest.common_cutoff),
                )
            except Exception:  # noqa: BLE001 - provider details and secrets stay private
                return finish(
                    _error(
                        "next_trading_day_not_verified",
                        common_cutoff=cutoff,
                        plan_for_date=None,
                    )
                )
            digest = replace(digest, plan_for_date=plan_for_date)
            if plan_for_date != target_date + timedelta(days=1):
                if plan_for_date <= target_date:
                    return finish(
                        _error("verified_market_data_stale_for_tomorrow", common_cutoff=cutoff)
                    )
                return finish(
                    EveningDigestOutcome(
                        EXIT_OK,
                        _event(
                            "noop_not_next_session_eve",
                            common_cutoff=cutoff,
                            plan_for_date=plan_for_date.isoformat(),
                        ),
                    )
                )
            # Freeze the exact derived report before submitting it. Provider
            # outcomes and later maturity observations are append-only and can
            # never rewrite this recommendation snapshot.
            holding_identity: tuple[str, int] | None = None
            current_ledger_identity: tuple[str, int] | None = None
            holding_symbols: tuple[str, ...] = ()
            holding_context = None
            holding_known_at = now
            chart_channels: frozenset[str] = frozenset()
            chart_publisher_id: str | None = None
            try:
                portfolio = get_active_holding_portfolio(repository)
                current_ledger_identity = (
                    None if portfolio is None else (portfolio.id, portfolio.version)
                )
                holding_channels = holding_summary_delivery_channels(portfolio)
                chart_channels = holding_chart_delivery_channels(portfolio)
                chart_publisher_id = holding_chart_publisher_id(portfolio)
                if portfolio is not None and holding_channels:
                    holding_identity = (portfolio.id, portfolio.version)
                    holding_symbols = tuple(position.symbol for position in portfolio.positions)
                    holding_context = holding_knowledge_context(
                        portfolio,
                        known_at=holding_known_at,
                    )
            except Exception:  # noqa: BLE001 - a local ledger fault cannot grant disclosure
                holding_channels = frozenset()
                chart_channels = frozenset()
                chart_publisher_id = None
                holding_context = None
            holding_review = None
            if holding_channels and holding_identity is not None and holding_context is not None:
                try:
                    holding_review = _build_holding_review(
                        repository,
                        dataset_root=resolved_csmar,
                        overlay_root=resolved_overlay,
                        decision_date=digest.common_cutoff,
                        reviewed_at=holding_known_at,
                        persist=True,
                        holding_context=holding_context,
                    )
                    review_identity = (
                        holding_review.portfolio_id,
                        holding_review.holding_version,
                    )
                    if review_identity != holding_identity or not _holding_authorization_matches(
                        repository,
                        expected_identity=holding_identity,
                        expected_channels=holding_channels,
                    ):
                        holding_review = None
                except Exception:  # noqa: BLE001 - private failure cannot suppress the plan
                    holding_review = None

            if (
                digest.continuous_plan is not None
                and digest.continuous_plan.get("holding_based")
                and digest.continuous_plan.get("holding_identity")
                != (None if current_ledger_identity is None else list(current_ledger_identity))
            ):
                return finish(
                    _error("holding_version_changed_before_submission", common_cutoff=cutoff)
                )
            archive = archiver(digest, repository)

            public_body = render_evening_digest_markdown(
                digest,
                None,
                include_holding_summary=False,
            )
            try:
                public_compact_body = render_evening_digest_bark_compact(
                    digest,
                    None,
                    include_holding_summary=False,
                )
            except ValueError:
                # An APNs layout failure must not suppress the independent
                # ServerChan Markdown. Bark still enforces its own byte cap.
                public_compact_body = None
            body = render_evening_digest_markdown(
                digest,
                holding_review,
                include_holding_summary="serverchan" in holding_channels,
            )
            try:
                compact_body = render_evening_digest_bark_compact(
                    digest,
                    holding_review,
                    include_holding_summary="bark" in holding_channels,
                )
            except ValueError:
                compact_body = None
            publication: _PublishedHoldingChart | None = None
            chart_requested = bool(
                holding_identity is not None
                and holding_symbols
                and holding_channels.issuperset(_HOLDING_CHART_CHANNELS)
                and chart_channels == _HOLDING_CHART_CHANNELS
                and chart_publisher_id == CLOUDFLARE_R2_PUBLISHER_ID
            )
            chart_status = "pending" if chart_requested else "not_requested"
            chart_reason = "publisher_or_holding_review_unavailable" if chart_requested else None
            chart_attempts = (
                int(prior_state.get("chart_attempts", 0)) if text_already_accepted else 0
            ) + int(chart_requested)
            chart_delivery_authorized = bool(
                holding_review is not None
                and holding_symbols
                and holding_identity is not None
                and holding_channels.issuperset(_HOLDING_CHART_CHANNELS)
                and chart_channels == _HOLDING_CHART_CHANNELS
                and chart_publisher_id == CLOUDFLARE_R2_PUBLISHER_ID
                and getattr(_holding_chart_publisher, "provider_id", None)
                == CLOUDFLARE_R2_PUBLISHER_ID
            )
            if chart_delivery_authorized:
                try:
                    assert holding_identity is not None
                    chart_result = _build_holding_chart_report(
                        repository,
                        dataset_root=resolved_csmar,
                        overlay_root=resolved_overlay,
                        as_of=digest.common_cutoff,
                        reviewed_at=holding_known_at,
                        archive_directory=resolved_state_root / "holding-charts",
                        holding_context=holding_context,
                    )
                    chart_payload = _validated_holding_chart_payload(
                        chart_result,
                        holding_review=holding_review,
                        expected_identity=holding_identity,
                        expected_cutoff=digest.common_cutoff,
                        expected_symbols=holding_symbols,
                    )
                    if not _holding_chart_authorization_matches(
                        repository,
                        expected_identity=holding_identity,
                        expected_summary_channels=holding_channels,
                    ):
                        raise ValueError("holding chart authorization changed")
                    assert _holding_chart_publisher is not None
                    publication = _publish_holding_chart(
                        _holding_chart_publisher,
                        chart_payload,
                    )
                    chart_reason = None
                    if not _holding_chart_authorization_matches(
                        repository,
                        expected_identity=holding_identity,
                        expected_summary_channels=holding_channels,
                    ):
                        publication.revoke_once()
                        publication = None
                        chart_status = "cancelled"
                        chart_reason = "holding_authorization_changed"
                except Exception:  # noqa: BLE001 - chart failure keeps the text report usable
                    if publication is not None:
                        publication.revoke_once()
                    publication = None
                    chart_reason = "chart_generation_or_upload_failed"

            # A successful text submission is never repeated to retry its image.
            # Every image retry builds a new publication and therefore a fresh
            # signed URL; signed URLs are never stored in delivery state.
            if text_already_accepted and publication is None:
                chart_status = (
                    "exhausted" if chart_attempts >= _MAX_CHART_ATTEMPTS else chart_status
                )
                _write_state(
                    state_path,
                    {
                        **prior_state,
                        "chart_status": chart_status,
                        "chart_attempts": chart_attempts,
                        "chart_reason": chart_reason,
                    },
                )
                return finish(
                    EveningDigestOutcome(
                        EXIT_RETRY if chart_status == "pending" else EXIT_OK,
                        _event(
                            "chart_pending" if chart_status == "pending" else "chart_not_sent",
                            common_cutoff=cutoff,
                            plan_for_date=plan_for_date.isoformat(),
                            chart_status=chart_status,
                            chart_reason=chart_reason,
                            chart_attempts=chart_attempts,
                            text_already_accepted=True,
                        ),
                    )
                )

            if chart_requested and publication is None:
                body += "\n\n🩷 持仓日/周图暂缺，今晚稍后自动补图。"
            if text_already_accepted:
                body = (
                    f"# 🩵 {format_cn_plan_date(plan_for_date)} 持仓日/周图补充\n"
                    f"对应数据 {cutoff} 的同一期研究计划；文字建议已提交。\n"
                    "图中蓝线为介入参考，粉线为保护参考；请留意图内核验状态。"
                )
                compact_body = body
                public_body = "持仓图授权已变化，本次补图已取消。"
                public_compact_body = public_body

            plan_label = format_cn_plan_date(plan_for_date)
            holding_guard = None
            if holding_identity is not None and (
                holding_review is not None
                or bool(digest.continuous_plan and digest.continuous_plan.get("holding_based"))
            ):

                def holding_guard_impl(channel_name: str) -> bool:
                    return channel_name in holding_channels and _holding_authorization_matches(
                        repository,
                        expected_identity=holding_identity,
                        expected_channels=holding_channels,
                    )

                holding_guard = holding_guard_impl
            image_guard = None
            if publication is not None and holding_identity is not None:

                def image_guard_impl(channel_name: str) -> bool:
                    return channel_name == "serverchan" and _holding_chart_authorization_matches(
                        repository,
                        expected_identity=holding_identity,
                        expected_summary_channels=holding_channels,
                    )

                image_guard = image_guard_impl
            message = NotificationMessage(
                title=f"A股{'补图' if text_already_accepted else '日报'}｜{plan_label}计划",
                body=body,
                group="A股研究室·晚间日报",
                compact_body=compact_body,
                image_url=(None if publication is None else publication.receipt.image_url),
                image_authorized_channels=(
                    frozenset() if publication is None else _HOLDING_CHART_CHANNELS
                ),
                holding_authorization_guard=holding_guard,
                image_authorization_guard=image_guard,
                unauthorized_body=(None if holding_guard is None else public_body),
                unauthorized_compact_body=(None if holding_guard is None else public_compact_body),
                image_revoke_callback=(None if publication is None else publication.revoke_once),
            )
            send_now = (_clock or (lambda: datetime.now(_SHANGHAI)))().astimezone(_SHANGHAI)
            if send_now.hour != 21 or send_now.date() != now.date():
                if publication is not None:
                    publication.revoke_once()
                return finish(
                    EveningDigestOutcome(
                        EXIT_OK,
                        _event("noop_evening_window_ended_before_submission"),
                    )
                )
            notification = notifier(message)
            if not isinstance(notification, EveningNotificationSummary):
                raise TypeError("notifier must return EveningNotificationSummary")
            if publication is not None and "serverchan" not in notification.image_accepted_channels:
                publication.revoke_once()
                chart_reason = "image_provider_not_accepted"
            if "serverchan" in notification.image_accepted_channels:
                chart_status = "accepted"
                chart_reason = None
            elif chart_status == "pending" and chart_attempts >= _MAX_CHART_ATTEMPTS:
                chart_status = "exhausted"
            notification_event = {
                "configured_channels": list(notification.configured_channels),
                "accepted_channels": list(notification.accepted_channels),
                "failed_channels": list(notification.failed_channels),
                "provider_receipt_ids": list(notification.provider_receipt_ids),
                "image_accepted_channels": list(notification.image_accepted_channels),
                "delivery_confirmed": False,
                "plan_for_date": plan_for_date.isoformat(),
                "chart_status": chart_status,
                "chart_reason": chart_reason,
                "chart_attempts": chart_attempts,
                "text_already_accepted": text_already_accepted,
            }
            _record_evening_delivery_events(
                repository,
                archive=archive,
                notification=notification,
            )
            if not notification.any_accepted:
                if text_already_accepted:
                    _write_state(
                        state_path,
                        {
                            **prior_state,
                            "chart_status": chart_status,
                            "chart_attempts": chart_attempts,
                            "chart_reason": chart_reason,
                        },
                    )
                reason = (
                    "notification_channels_not_configured"
                    if not notification.configured_channels and not notification.failed_channels
                    else "notification_providers_not_accepted"
                )
                return finish(_error(reason, common_cutoff=cutoff, **notification_event))

            _write_state(
                state_path,
                {
                    "last_provider_accepted_common_cutoff": cutoff,
                    "method_version": digest.method_version,
                    "plan_for_date": plan_for_date.isoformat(),
                    "provider_accepted_at": now.isoformat(),
                    "accepted_channels": list(notification.accepted_channels),
                    "provider_receipt_ids": list(notification.provider_receipt_ids),
                    "delivery_confirmed": False,
                    "chart_status": chart_status,
                    "chart_attempts": chart_attempts,
                    "chart_reason": chart_reason,
                },
            )
            return finish(
                EveningDigestOutcome(
                    EXIT_OK,
                    _event(
                        "provider_accepted",
                        common_cutoff=cutoff,
                        period_count=len(digest.periods),
                        **notification_event,
                        raw_data_exposed=False,
                        orders_enabled=False,
                    ),
                )
            )
    except (AShareLabError, FileNotFoundError, OSError, TypeError, ValueError):
        return finish(_error("digest_or_provider_submission_failed"))
    except Exception:  # noqa: BLE001 - never copy an unsafe exception into output
        return finish(_error("unexpected_evening_digest_error"))


def _send_final_failure_notice(
    *, repository, notifier, state_root, dataset_root, overlay_root, now, holding_reviewer
):
    """One fail-closed notice on the final retry; never claim a missing report.

    A small holding-only review is independent of the full-market builder.
    It uses the same per-provider authorization guard and exposes no credentials,
    amounts or raw data. Provider errors cannot recurse into this function.
    """
    path = state_root / "evening-failure-notice-state.json"
    try:
        state = _read_state(path)
        if state.get("accepted_date") == now.date().isoformat():
            return
        public = "今晚完整研究计划未能生成或提交。请勿把旧报告当作明日买入依据；没有自动下单。"
        body, guard = public, None
        try:
            portfolio = get_active_holding_portfolio(repository)
            channels = holding_summary_delivery_channels(portfolio)
        except Exception:
            portfolio, channels = None, frozenset()
        if portfolio is not None and "serverchan" in channels:
            identity = (portfolio.id, portfolio.version)
            try:
                cutoff = latest_verified_overlay_cutoff(overlay_root)
            except Exception:
                cutoff = None
            if cutoff is not None:
                try:
                    review = holding_reviewer(
                        repository,
                        dataset_root=dataset_root,
                        overlay_root=overlay_root,
                        decision_date=cutoff,
                        reviewed_at=now,
                        persist=False,
                        holding_context=holding_knowledge_context(portfolio, known_at=now),
                    )
                    if (review.portfolio_id, review.holding_version) == identity:
                        from ashare_lab.services.build_evening_digest import _holding_review_lines

                        lines = _holding_review_lines(review, name_bytes=36, reason_bytes=90)
                        body += f"\n\n持仓单独核验（数据{cutoff}，非明日新买计划）：\n" + "\n".join(
                            lines
                        )

                        def guard(channel):
                            return channel == "serverchan" and _holding_authorization_matches(
                                repository, expected_identity=identity, expected_channels=channels
                            )
                except Exception:
                    body += "\n持仓核验也未完成：当前不确认持有或卖出。"
        message = NotificationMessage(
            title="A股计划未完成｜请勿沿用旧买入建议",
            body=body,
            group="A股研究室·晚间日报",
            holding_authorization_guard=guard,
            unauthorized_body=public if guard else None,
        )
        receipt = notifier(message)
        if (
            isinstance(receipt, EveningNotificationSummary)
            and "serverchan" in receipt.accepted_channels
        ):
            _write_state(
                path, {"accepted_date": now.date().isoformat(), "delivery_confirmed": False}
            )
    except Exception:
        return


def _archive_original_digest(
    digest: EveningResearchDigest,
    repository: SQLiteRepository,
) -> RecommendationArchiveBundle:
    bundle = archive_recommendation_report(
        replace(digest, periods=()) if digest.continuous_plan is not None else digest,
        repository,
        archive_nature="original",
        created_at=datetime.now(_SHANGHAI),
    )
    if digest.continuous_plan is not None:
        from ashare_lab.services.continuous_strategy_journal import archive_continuous_decision

        with repository.connection() as connection:
            archive_continuous_decision(
                connection,
                bundle.report_id,
                digest.common_cutoff.isoformat(),
                {"plan_for_date": digest.plan_for_date.isoformat(), **digest.continuous_plan},
            )
    return bundle


def _holding_authorization_matches(
    repository: SQLiteRepository,
    *,
    expected_identity: tuple[str, int] | None,
    expected_channels: frozenset[str],
) -> bool:
    """Re-read one exact holding identity and its provider-scoped consent."""

    if expected_identity is None or not expected_channels:
        return False
    try:
        portfolio = get_active_holding_portfolio(repository)
        if portfolio is None:
            return False
        return (
            portfolio.id,
            portfolio.version,
        ) == expected_identity and holding_summary_delivery_channels(portfolio) == expected_channels
    except Exception:  # noqa: BLE001 - a local read fault cannot grant disclosure
        return False


def _holding_chart_authorization_matches(
    repository: SQLiteRepository,
    *,
    expected_identity: tuple[str, int],
    expected_summary_channels: frozenset[str],
) -> bool:
    """Recheck the exact revision and the separately granted chart scope."""

    try:
        portfolio = get_active_holding_portfolio(repository)
        if portfolio is None or (portfolio.id, portfolio.version) != expected_identity:
            return False
        return (
            holding_summary_delivery_channels(portfolio) == expected_summary_channels
            and "serverchan" in expected_summary_channels
            and holding_chart_delivery_channels(portfolio) == _HOLDING_CHART_CHANNELS
            and holding_chart_publisher_id(portfolio) == CLOUDFLARE_R2_PUBLISHER_ID
        )
    except Exception:  # noqa: BLE001 - a local read fault cannot grant disclosure
        return False


def _validated_holding_chart_payload(
    result: HoldingChartBuildResult,
    *,
    holding_review: HoldingTreeReviewSummary,
    expected_identity: tuple[str, int],
    expected_cutoff: date,
    expected_symbols: tuple[str, ...],
) -> bytes:
    """Allow only the exact registered holdings and their bounded 2N panels."""

    if result.status is not HoldingChartBuildStatus.READY or result.rendered is None:
        raise ValueError("holding chart is not ready")
    if (
        (result.portfolio_id, result.holding_version) != expected_identity
        or result.holding_weeks != holding_review.holding_weeks
        or result.data_cutoff != expected_cutoff
    ):
        raise ValueError("holding chart identity mismatch")
    metadata = result.rendered.metadata
    review_identity = metadata.review_identity
    if (
        (review_identity.portfolio_id, review_identity.holding_version) != expected_identity
        or review_identity.holding_weeks != holding_review.holding_weeks
        or review_identity.data_cutoff != expected_cutoff
    ):
        raise ValueError("holding chart review identity mismatch")
    if (
        not MIN_HOLDING_COUNT <= len(expected_symbols) <= MAX_HOLDING_COUNT
        or tuple(row.symbol for row in holding_review.rows) != expected_symbols
        or metadata.panel_count != 2 * len(expected_symbols)
        or tuple(metadata.symbols) != expected_symbols
        or metadata.width != COMPOSITE_WIDTH
        or metadata.height != holding_chart_composite_height(len(expected_symbols))
        or metadata.raw_rows_embedded is not False
        or metadata.sensitive_fields_embedded is not False
    ):
        raise ValueError("holding chart disclosure contract mismatch")
    payload = result.rendered.composite_png
    if not isinstance(payload, bytes) or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("holding chart PNG is invalid")
    return payload


def _publish_holding_chart(
    publisher: PrivateImagePublisher,
    payload: bytes,
) -> _PublishedHoldingChart:
    """Publish one opaque PNG and verify the one-hour, revocable R2 receipt."""

    if getattr(publisher, "provider_id", None) != CLOUDFLARE_R2_PUBLISHER_ID:
        raise ValueError("holding chart publisher is not authorized")
    receipt = publisher.publish_png(payload)
    try:
        if receipt.provider_id != CLOUDFLARE_R2_PUBLISHER_ID:
            raise ValueError("holding chart receipt provider mismatch")
        expires_at = receipt.expires_at
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            raise ValueError("holding chart receipt expiry is naive")
        seconds = (expires_at.astimezone(UTC) - datetime.now(UTC)).total_seconds()
        if seconds <= 0 or seconds > _HOLDING_CHART_MAX_URL_TTL_SECONDS + 5:
            raise ValueError("holding chart receipt expiry is outside one hour")
        if not isinstance(receipt.image_url, str) or not isinstance(receipt.revoke_key, str):
            raise ValueError("holding chart receipt is invalid")
        # Let the shared message contract perform the complete HTTPS/public-host
        # validation without retaining the signed address anywhere.
        NotificationMessage(
            title="图片地址安全校验",
            body="仅校验，不发送。",
            image_url=receipt.image_url,
            image_authorized_channels=_HOLDING_CHART_CHANNELS,
        )
    except Exception:
        with suppress(Exception):
            publisher.revoke(receipt.revoke_key)
        raise
    return _PublishedHoldingChart(publisher=publisher, receipt=receipt)


def _record_evening_delivery_events(
    repository: SQLiteRepository,
    *,
    archive: RecommendationArchiveBundle,
    notification: EveningNotificationSummary,
) -> None:
    """Append provider outcomes without storing a message body or secret."""

    attempted_at = datetime.now(_SHANGHAI)
    receipt_by_channel: dict[str, str] = {}
    for value in notification.provider_receipt_ids:
        channel, separator, receipt = value.partition(":")
        if separator and channel and receipt:
            receipt_by_channel[channel] = receipt
    statuses: tuple[tuple[str, str, str], ...] = tuple(
        (channel, "evening_provider_accepted", "provider_accepted")
        for channel in notification.accepted_channels
    ) + tuple(
        (channel, "evening_provider_failed", "provider_failed")
        for channel in notification.failed_channels
    )
    for channel, delivery_kind, provider_status in statuses:
        repository.record_recommendation_delivery_event(
            {
                "id": uuid4().hex,
                "report_id": archive.report_id,
                "batch_id": None,
                "delivery_kind": delivery_kind,
                "channel": channel,
                "attempted_at": attempted_at,
                "provider_status": provider_status,
                "provider_receipt_id": receipt_by_channel.get(channel),
                "detail_json": {
                    "delivery_confirmed": False,
                    "raw_data_exposed": False,
                    "orders_enabled": False,
                },
            }
        )


def send_evening_digest(message: NotificationMessage) -> EveningNotificationSummary:
    """Try ServerChan and Bark independently without exposing provider credentials."""

    channels: list[Any] = []
    configured: list[str] = []
    accepted: list[str] = []
    failed: list[str] = []
    receipt_ids: list[str] = []
    image_accepted: list[str] = []
    channel_specs = (
        ("serverchan", load_serverchan_sendkey, ServerChanNotificationChannel),
        ("bark", load_bark_device_key, BarkNotificationChannel),
    )
    enabled_channels = _enabled_notification_channels()
    channel_specs = tuple(spec for spec in channel_specs if spec[0] in enabled_channels)
    try:
        for name, loader, constructor in channel_specs:
            try:
                credential = loader()
            except Exception:  # noqa: BLE001 - one provider must not suppress the other
                failed.append(name)
                continue
            if not credential:
                continue
            configured.append(name)
            try:
                channels.append(constructor(credential))
            except Exception:  # noqa: BLE001 - keep the second provider independent
                failed.append(name)

        for channel in channels:
            name = channel.channel_name
            try:
                channel_message = _message_for_channel(message, channel_name=name)
                if channel_message is None:
                    failed.append(name)
                    continue
                receipt = channel.send(channel_message)
            except Exception:  # noqa: BLE001 - never copy provider exceptions into events
                failed.append(name)
                continue
            if receipt.accepted and receipt.provider_status == "provider_accepted":
                accepted.append(name)
                if receipt.image_accepted:
                    image_accepted.append(name)
                if receipt.provider_receipt_id:
                    receipt_ids.append(f"{name}:{receipt.provider_receipt_id}")
            else:
                failed.append(name)
    finally:
        for channel in channels:
            close = getattr(channel, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - cleanup cannot suppress other results
                    continue

    return EveningNotificationSummary(
        configured_channels=tuple(dict.fromkeys(configured)),
        accepted_channels=tuple(dict.fromkeys(accepted)),
        failed_channels=tuple(dict.fromkeys(failed)),
        provider_receipt_ids=tuple(dict.fromkeys(receipt_ids)),
        image_accepted_channels=tuple(dict.fromkeys(image_accepted)),
    )


def _enabled_notification_channels() -> tuple[str, ...]:
    """Default personal evening reports to the user's ServerChan channel."""

    raw = os.environ.get(_NOTIFICATION_CHANNELS_ENV)
    if raw is None:
        return ("serverchan",)
    values = tuple(
        dict.fromkeys(value.strip().lower() for value in raw.split(",") if value.strip())
    )
    if not values or any(value not in _SUPPORTED_NOTIFICATION_CHANNELS for value in values):
        raise ValueError("evening notification channel allow-list is invalid")
    return values


def _message_for_channel(
    message: NotificationMessage,
    *,
    channel_name: str,
) -> NotificationMessage | None:
    """Keep full ServerChan Markdown and use only the bounded body for Bark."""

    message = prepare_notification_for_channel(message, channel_name=channel_name)
    if message is None:
        return None
    if channel_name != "bark":
        return message
    if message.compact_body is not None:
        return replace(message, body=message.compact_body, compact_body=None)
    if len(message.body.encode("utf-8")) > MAX_COMPACT_NOTIFICATION_BODY_BYTES:
        return None
    return message


def send_serverchan_digest(message: NotificationMessage) -> bool:
    """Backward-compatible single-provider helper; new evening jobs use both providers."""

    sendkey = load_serverchan_sendkey()
    if not sendkey:
        raise DataUnavailableError("Server酱尚未配置")
    channel_message = _message_for_channel(message, channel_name="serverchan")
    if channel_message is None:
        raise DataUnavailableError("Server酱通知正文不可发送")
    with ServerChanNotificationChannel(sendkey) as channel:
        receipt = channel.send(channel_message)
    return receipt.accepted and receipt.provider_status == "provider_accepted"


def resolve_next_infoway_trading_day(
    common_cutoff: date,
    *,
    _api_key_loader: Callable[[], str | None] = load_infoway_api_key,
    _provider_factory: Callable[..., Any] = InfowayEodMarketData,
) -> date:
    """Resolve the first official CN session after the verified data cutoff.

    Production credentials can enter only through the macOS Keychain loader.
    The bounded 14-day request is sufficient to cross ordinary weekends and
    mainland statutory-holiday closures without guessing from weekdays.
    """

    if not isinstance(common_cutoff, date) or isinstance(common_cutoff, datetime):
        raise TypeError("common_cutoff must be a date")
    credential = _api_key_loader()
    if not credential:
        raise DataUnavailableError("未配置Infoway凭据，无法确认下一交易日")

    start = common_cutoff + timedelta(days=1)
    end = common_cutoff + timedelta(days=_TRADING_CALENDAR_LOOKAHEAD_DAYS)
    provider: Any | None = None
    try:
        provider = _provider_factory(credential)
        sessions = tuple(provider.fetch_cn_trading_days(start, end))
    finally:
        if provider is not None:
            close = getattr(provider, "close", None)
            if not callable(close):
                raise TypeError("Infoway calendar adapter must be safely closeable")
            close()

    if not sessions:
        raise DataUnavailableError("未能在14日窗口内确认下一个A股交易日")
    if any(not isinstance(value, date) or isinstance(value, datetime) for value in sessions):
        raise DataUnavailableError("Infoway CN交易日历包含无效日期")
    if sessions != tuple(sorted(set(sessions))):
        raise DataUnavailableError("Infoway CN交易日历顺序或唯一性异常")
    return _validate_next_trading_day(common_cutoff, sessions[0])


def resolve_next_zero_budget_trading_day(
    common_cutoff: date,
    *,
    _provider_factory: Callable[..., Any] = BaoStockEodMarketData,
) -> date:
    """Resolve the next official A-share session without a paid credential."""

    if not isinstance(common_cutoff, date) or isinstance(common_cutoff, datetime):
        raise TypeError("common_cutoff must be a date")
    start = common_cutoff + timedelta(days=1)
    end = common_cutoff + timedelta(days=_TRADING_CALENDAR_LOOKAHEAD_DAYS)
    provider = _provider_factory()
    try:
        sessions = tuple(provider.fetch_cn_trading_days(start, end))
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    if not sessions:
        raise DataUnavailableError("未能在14日窗口内确认下一个A股交易日")
    if any(not isinstance(value, date) or isinstance(value, datetime) for value in sessions):
        raise DataUnavailableError("BaoStock CN交易日历包含无效日期")
    if sessions != tuple(sorted(set(sessions))):
        raise DataUnavailableError("BaoStock CN交易日历顺序或唯一性异常")
    return _validate_next_trading_day(common_cutoff, sessions[0])


def latest_verified_overlay_cutoff(overlay_root: Path) -> date | None:
    """Read only the latest verified manifest date for an inexpensive NOOP."""

    manifest = MarketOverlayStore(overlay_root).read_verified_manifest(
        source_id=DEFAULT_MARKET_OVERLAY_SOURCE_ID
    )
    if manifest.empty or "trade_date" not in manifest:
        return None
    dates = pd.to_datetime(manifest["trade_date"], errors="coerce")
    if bool(dates.isna().any()):
        raise DataUnavailableError("已验证增量清单日期无效")
    return dates.max().date()


def build_parser() -> argparse.ArgumentParser:
    data_root = application_data_dir()
    parser = argparse.ArgumentParser(
        description=(
            "生成六周期A股研究日报并提交给本机钥匙串中已配置的通知通道；不连接券商、不自动下单。"
        )
    )
    parser.add_argument("--csmar-root", type=Path, default=data_root / "cache" / "csmar")
    parser.add_argument("--overlay-root", type=Path, default=data_root / "cache" / "market_overlay")
    parser.add_argument(
        "--reference-root", type=Path, default=data_root / "cache" / "csmar_reference"
    )
    parser.add_argument("--state-root", type=Path, default=data_root / "scheduler")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path.home() / "Library" / "Logs" / "A股研究助手",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    publisher: PrivateImagePublisher | None = None
    try:
        configured = build_configured_chart_publisher()
        if configured is not None and hasattr(configured, "publish_png"):
            publisher = configured  # type: ignore[assignment]
    except Exception:  # noqa: BLE001 - invalid/missing R2 setup means text-only fallback
        publisher = None
    try:
        outcome = run_evening_digest(
            csmar_root=args.csmar_root,
            overlay_root=args.overlay_root,
            reference_root=args.reference_root,
            state_root=args.state_root,
            log_root=args.log_root,
            _holding_chart_publisher=publisher,
        )
    finally:
        if publisher is not None:
            with suppress(Exception):
                publisher.close()
    print(json.dumps(outcome.event, ensure_ascii=False, sort_keys=True))
    return outcome.exit_code


def _event(status: str, **values: Any) -> dict[str, Any]:
    return {
        "job": "ashare-evening-digest",
        "status": status,
        **values,
    }


def _error(reason: str, **values: Any) -> EveningDigestOutcome:
    return EveningDigestOutcome(
        EXIT_ERROR,
        _event("error", reason=reason, **values),
    )


def _state_cutoff(state: dict[str, Any]) -> date | None:
    value = state.get("last_provider_accepted_common_cutoff")
    if not isinstance(value, str):
        # Read the pre-reliability-fix state once so an upgrade does not resend
        # the same cutoff unexpectedly.  New state never writes this legacy key.
        value = state.get("last_sent_common_cutoff")
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _state_plan_for_date(state: dict[str, Any]) -> str | None:
    value = state.get("plan_for_date")
    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed.isoformat()


def _validate_next_trading_day(common_cutoff: date, value: date) -> date:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError("next trading day resolver must return a date")
    latest = common_cutoff + timedelta(days=_TRADING_CALENDAR_LOOKAHEAD_DAYS)
    if value <= common_cutoff or value > latest:
        raise ValueError("next trading day is outside the verified lookahead window")
    return value


def _read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_state(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
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


def _safe_write_log_outcome(path: Path, outcome: EveningDigestOutcome) -> None:
    """Best-effort audit logging that cannot change the provider result."""

    try:
        _write_log_event(path, outcome)
    except Exception:  # noqa: BLE001 - private logging is deliberately non-fatal
        return


def _write_log_event(path: Path, outcome: EveningDigestOutcome) -> None:
    """Append a whitelisted event; never persist messages, exceptions or market rows."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    document = {
        "logged_at": datetime.now(_SHANGHAI).isoformat(),
        "exit_code": outcome.exit_code,
        **{key: outcome.event[key] for key in _LOG_EVENT_KEYS if key in outcome.event},
    }
    logger = logging.getLogger(f"ashare_lab.evening_digest.{hash(path)}")
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
        logger.info(json.dumps(document, ensure_ascii=False, sort_keys=True))
    finally:
        handler.close()
        logger.removeHandler(handler)
    os.chmod(path, 0o600)


if __name__ == "__main__":
    raise SystemExit(main())
