from __future__ import annotations

import json
import plistlib
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from ashare_lab.cli import scheduled_sync
from ashare_lab.domain.errors import DataUnavailableError
from ashare_lab.services.build_monthly_model_review import MonthlyModelReview
from ashare_lab.services.daily_update_lock import daily_update_lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLIST_TEMPLATE = PROJECT_ROOT / "config" / "com.zerong.asharelab.daily-sync.plist.template"
INSTALL_SCRIPT = PROJECT_ROOT / "scripts" / "install_daily_sync_launchagent.sh"
UNINSTALL_SCRIPT = PROJECT_ROOT / "scripts" / "uninstall_daily_sync_launchagent.sh"
NOW = datetime(2026, 8, 27, 13, 40, tzinfo=UTC)


@dataclass(frozen=True)
class _NotificationResult:
    configured_channels: tuple[str, ...] = ("serverchan", "bark")
    successful_channels: tuple[str, ...] = ("serverchan", "bark")
    failed_channels: tuple[str, ...] = ()

    @property
    def any_succeeded(self) -> bool:
        return bool(self.successful_channels)


def _report(
    *,
    current: bool,
    updated: tuple[date, ...] = (),
    reason: str = "",
    cutoff: date = date(2026, 8, 27),
):
    failures = ()
    if reason:
        failures = (SimpleNamespace(reason=reason),)
    return SimpleNamespace(
        source_id="infoway",
        requested_complete_date=cutoff,
        latest_complete_session=cutoff,
        common_cutoff=cutoff if current else cutoff - timedelta(days=1),
        updated_sessions=updated,
        unchanged_sessions=(),
        quarantined_failures=failures,
        provider_contract_changed=False,
        current_through_latest_complete_session=current,
        csmar_mutated=False,
    )


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "csmar_root": tmp_path / "csmar",
        "overlay_root": tmp_path / "overlay",
        "scheduler_root": tmp_path / "scheduler",
        "log_root": tmp_path / "logs",
    }


def test_current_update_exits_zero_logs_private_event_and_sends_no_failure(
    tmp_path: Path,
) -> None:
    notified = []

    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: _report(
            current=True,
            updated=(date(2026, 8, 27),),
        ),
        _notifier=lambda message: notified.append(message) or _NotificationResult(),
    )

    assert outcome.exit_code == scheduled_sync.EXIT_CURRENT
    assert outcome.event["status"] == "updated"
    assert outcome.event["common_cutoff"] == "2026-08-27"
    assert notified == []
    log_path = tmp_path / "logs" / "daily-sync.jsonl"
    line = json.loads(log_path.read_text(encoding="utf-8"))
    assert line["status"] == "updated"
    assert stat.S_IMODE(log_path.stat().st_mode) == 0o600


def test_current_update_settles_mature_recommendations_and_logs_counts(
    tmp_path: Path,
) -> None:
    calls = []

    def performance_runner(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            pending_batches=6,
            evaluated_batches=2,
            mature_batches=1,
            notification_attempts=1,
            notification_accepted_batches=1,
            notification_failed_batches=0,
            failed_batch_ids=(),
        )

    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: _report(current=True),
        _notifier=lambda _message: _NotificationResult(),
        _performance_runner=performance_runner,
    )

    assert outcome.exit_code == scheduled_sync.EXIT_CURRENT
    assert len(calls) == 1
    assert calls[0]["as_of"] == date(2026, 8, 27)
    assert calls[0]["overlay_store"].root == (tmp_path / "overlay").resolve()
    assert (
        calls[0]["corporate_action_loader"]
        is scheduled_sync.load_available_local_corporate_action_evidence
    )
    assert outcome.event["performance_status"] == "completed"
    assert outcome.event["performance_pending_batches"] == 6
    assert outcome.event["performance_evaluated_batches"] == 2
    assert outcome.event["performance_mature_batches"] == 1
    assert outcome.event["performance_notification_accepted_batches"] == 1
    assert outcome.event["performance_failed_batch_count"] == 0


def test_performance_partial_failure_is_visible_and_retryable(tmp_path: Path) -> None:
    summary = SimpleNamespace(
        pending_batches=6,
        evaluated_batches=5,
        mature_batches=1,
        notification_attempts=0,
        notification_accepted_batches=0,
        notification_failed_batches=0,
        failed_batch_ids=("batch-a",),
    )

    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: _report(current=True),
        _notifier=lambda _message: _NotificationResult(),
        _performance_runner=lambda **_kwargs: summary,
    )

    assert outcome.exit_code == scheduled_sync.EXIT_CURRENT
    assert outcome.event["performance_status"] == "partial_retry_later"
    assert outcome.event["performance_failed_batch_count"] == 1
    assert "batch-a" not in json.dumps(outcome.event)


def test_performance_review_failure_never_masks_data_sync_success(tmp_path: Path) -> None:
    secret = "must-never-reach-log"

    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: _report(current=True),
        _notifier=lambda _message: _NotificationResult(),
        _performance_runner=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )

    assert outcome.exit_code == scheduled_sync.EXIT_CURRENT
    assert outcome.event["performance_status"] == "error_retry_later"
    assert outcome.event["performance_error_type"] == "RuntimeError"
    assert secret not in json.dumps(outcome.event)


def test_current_update_runs_private_holding_review_and_logs_only_aggregate_counts(
    tmp_path: Path,
) -> None:
    calls = []
    secret_values = (
        "600919",
        "000001",
        "江苏银行",
        "12.34",
        "10.87",
        "private-reason",
    )
    summary = SimpleNamespace(
        status=SimpleNamespace(value="ready"),
        rows=(
            SimpleNamespace(
                symbol=secret_values[0],
                name=secret_values[2],
                cost_price=12.34,
                effective_stop=10.87,
                reasons=(secret_values[5],),
                status=SimpleNamespace(value="ready"),
                action=SimpleNamespace(value="hold"),
                urgent=False,
            ),
            SimpleNamespace(
                symbol=secret_values[1],
                status=SimpleNamespace(value="data_not_ready"),
                action=SimpleNamespace(value="review"),
                urgent=True,
            ),
        ),
    )

    def holding_runner(repository, **kwargs):
        calls.append((repository, kwargs))
        return summary

    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: _report(current=True),
        _notifier=lambda _message: (_ for _ in ()).throw(AssertionError("must not notify")),
        _holding_review_runner=holding_runner,
    )

    assert outcome.exit_code == scheduled_sync.EXIT_CURRENT
    assert len(calls) == 1
    repository, kwargs = calls[0]
    assert repository.db_path == (tmp_path / "research.db").resolve()
    assert kwargs == {
        "dataset_root": (tmp_path / "csmar").resolve(),
        "overlay_root": (tmp_path / "overlay").resolve(),
        "as_of": date(2026, 8, 27),
        "reviewed_at": NOW,
        "persist": True,
    }
    assert outcome.event["holding_review_status"] == "ready"
    assert outcome.event["holding_review_row_count"] == 2
    assert outcome.event["holding_review_ready_count"] == 1
    assert outcome.event["holding_review_urgent_count"] == 1
    assert outcome.event["holding_review_action_counts"] == {
        "hold": 1,
        "tighten": 0,
        "reduce": 0,
        "exit": 0,
        "review": 1,
    }
    event_text = json.dumps(outcome.event, ensure_ascii=False)
    assert all(value not in event_text for value in secret_values)


def test_no_holdings_is_a_local_noop_with_zero_counts(tmp_path: Path) -> None:
    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: _report(current=True),
        _notifier=lambda _message: (_ for _ in ()).throw(AssertionError("must not notify")),
        _holding_review_runner=scheduled_sync.run_active_holding_review,
    )

    assert outcome.exit_code == scheduled_sync.EXIT_CURRENT
    assert outcome.event["holding_review_status"] == "no_holdings"
    assert outcome.event["holding_review_row_count"] == 0
    assert outcome.event["holding_review_urgent_count"] == 0
    assert set(outcome.event["holding_review_action_counts"].values()) == {0}


def test_holding_review_failure_is_private_retryable_and_never_masks_sync(
    tmp_path: Path,
) -> None:
    secret = "600919 江苏银行 cost=12.34 stop=10.87"

    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: _report(current=True),
        _notifier=lambda _message: (_ for _ in ()).throw(AssertionError("must not notify")),
        _holding_review_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError(secret)
        ),
    )

    assert outcome.exit_code == scheduled_sync.EXIT_CURRENT
    assert outcome.event["status"] == "noop_current"
    assert outcome.event["holding_review_status"] == "error_retry_later"
    assert secret not in json.dumps(outcome.event, ensure_ascii=False)


def test_shadow_runs_after_holding_review_and_logs_only_aggregate_counts(
    tmp_path: Path,
) -> None:
    order = []
    secret = "600919 江苏银行 stop=10.87 private-shadow-reason"

    def holding_runner(*_args, **_kwargs):
        order.append("holding")
        return SimpleNamespace(status=SimpleNamespace(value="no_holdings"), rows=())

    def shadow_runner(repository, **kwargs):
        order.append("shadow")
        assert repository.db_path == (tmp_path / "research.db").resolve()
        assert kwargs == {
            "dataset_root": (tmp_path / "csmar").resolve(),
            "overlay_root": (tmp_path / "overlay").resolve(),
            "as_of": date(2026, 8, 27),
            "evaluated_at": NOW,
            "persist": True,
        }
        return SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            observation_count=8,
            variant_count=2,
            private_payload=secret,
        )

    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: _report(current=True),
        _notifier=lambda _message: (_ for _ in ()).throw(AssertionError("must not notify")),
        _holding_review_runner=holding_runner,
        _holding_shadow_runner=shadow_runner,
    )

    assert outcome.exit_code == scheduled_sync.EXIT_CURRENT
    assert order == ["holding", "shadow"]
    assert outcome.event["holding_shadow_status"] == "completed"
    assert outcome.event["holding_shadow_row_count"] == 8
    assert outcome.event["holding_shadow_variant_count"] == 2
    assert secret not in json.dumps(outcome.event, ensure_ascii=False)


def test_shadow_failure_is_nonfatal_private_and_retryable(tmp_path: Path) -> None:
    secret = "601919 shadow-price=17.00"
    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: _report(current=True),
        _notifier=lambda _message: (_ for _ in ()).throw(AssertionError("must not notify")),
        _holding_shadow_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError(secret)
        ),
    )

    assert outcome.exit_code == scheduled_sync.EXIT_CURRENT
    assert outcome.event["holding_shadow_status"] == "error_retry_later"
    assert secret not in json.dumps(outcome.event, ensure_ascii=False)


def test_first_verified_session_archives_monthly_review_locally_once(
    tmp_path: Path,
) -> None:
    cutoff = date(2026, 9, 1)
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
    build_calls = []
    notified = []
    review = MonthlyModelReview(
        review_month="2026-08",
        period_start=date(2026, 8, 1),
        period_end=date(2026, 8, 31),
        generated_as_of=cutoff,
        method_version="monthly-model-review-v0.1.0",
        horizon_reviews=(),
        experiment_proposals=(),
        excluded_batches=(),
        evidence_gaps=("benchmark_unavailable",),
        archive_scan_truncated=False,
        conclusion="样本不足，不自动调整参数。",
    )

    def monthly_builder(repository, **kwargs):
        build_calls.append((repository, kwargs))
        return review

    performance = SimpleNamespace(
        pending_batches=0,
        evaluated_batches=0,
        mature_batches=0,
        notification_attempts=0,
        notification_accepted_batches=0,
        notification_failed_batches=0,
        failed_batch_ids=(),
    )
    common = {
        **_paths(tmp_path),
        "clock": lambda: now,
        "_run_update": lambda **_kwargs: _report(current=True, cutoff=cutoff),
        "_notifier": lambda message: notified.append(message) or _NotificationResult(),
        "_performance_runner": lambda **_kwargs: performance,
        "_monthly_review_builder": monthly_builder,
    }

    first = scheduled_sync.run_scheduled_sync(**common)
    second = scheduled_sync.run_scheduled_sync(**common)

    assert first.exit_code == scheduled_sync.EXIT_CURRENT
    assert first.event["monthly_review_status"] == "completed_local_only"
    assert first.event["monthly_review_month"] == "2026-08"
    assert first.event["monthly_review_horizon_count"] == 0
    assert first.event["monthly_review_mature_batch_count"] == 0
    assert first.event["monthly_review_proposal_count"] == 0
    assert second.event["monthly_review_status"] == "not_due"
    assert len(build_calls) == 1
    repository, kwargs = build_calls[0]
    assert repository.db_path == (tmp_path / "research.db").resolve()
    assert kwargs == {
        "review_month": date(2026, 8, 1),
        "as_of": cutoff,
        "benchmark_evidence_by_batch": {},
    }
    archive_dir = tmp_path / "scheduler" / "monthly-model-reviews"
    archive_path = archive_dir / "2026-08.json"
    state_path = tmp_path / "scheduler" / "monthly-model-review-state.json"
    assert json.loads(archive_path.read_text(encoding="utf-8"))["review_month"] == "2026-08"
    assert json.loads(state_path.read_text(encoding="utf-8"))["completed_months"] == ["2026-08"]
    assert stat.S_IMODE(archive_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(archive_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    assert notified == []


def test_monthly_archive_failure_does_not_mark_complete_and_retries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cutoff = date(2026, 9, 1)
    review = MonthlyModelReview(
        review_month="2026-08",
        period_start=date(2026, 8, 1),
        period_end=date(2026, 8, 31),
        generated_as_of=cutoff,
        method_version="monthly-model-review-v0.1.0",
        horizon_reviews=(),
        experiment_proposals=(),
        excluded_batches=(),
        evidence_gaps=(),
        archive_scan_truncated=False,
        conclusion="local only",
    )
    secret = "private monthly payload"
    monkeypatch.setattr(
        scheduled_sync,
        "_write_private_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(secret)),
    )

    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: datetime(2026, 9, 1, 10, 0, tzinfo=UTC),
        _run_update=lambda **_kwargs: _report(current=True, cutoff=cutoff),
        _notifier=lambda _message: (_ for _ in ()).throw(AssertionError("must not notify")),
        _monthly_review_builder=lambda *_args, **_kwargs: review,
    )

    assert outcome.exit_code == scheduled_sync.EXIT_CURRENT
    assert outcome.event["monthly_review_status"] == "error_retry_later"
    assert secret not in json.dumps(outcome.event)
    assert not (tmp_path / "scheduler" / "monthly-model-review-state.json").exists()


def test_monthly_review_waits_until_same_run_performance_settlement_is_complete(
    tmp_path: Path,
) -> None:
    cutoff = date(2026, 9, 1)
    partial = SimpleNamespace(
        pending_batches=1,
        evaluated_batches=1,
        mature_batches=0,
        notification_attempts=0,
        notification_accepted_batches=0,
        notification_failed_batches=0,
        failed_batch_ids=("private-batch-id",),
    )

    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: datetime(2026, 9, 1, 10, 0, tzinfo=UTC),
        _run_update=lambda **_kwargs: _report(current=True, cutoff=cutoff),
        _notifier=lambda _message: _NotificationResult(),
        _performance_runner=lambda **_kwargs: partial,
        _monthly_review_builder=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("monthly review must wait for complete settlement")
        ),
    )

    assert outcome.exit_code == scheduled_sync.EXIT_CURRENT
    assert outcome.event["performance_status"] == "partial_retry_later"
    assert outcome.event["monthly_review_status"] == "deferred_performance_incomplete"
    assert "private-batch-id" not in json.dumps(outcome.event)
    assert not (tmp_path / "scheduler" / "monthly-model-review-state.json").exists()


def test_incomplete_sync_does_not_run_holding_or_monthly_postprocessing(
    tmp_path: Path,
) -> None:
    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: _report(current=False, reason="coverage gate"),
        _notifier=lambda _message: _NotificationResult(),
        _holding_review_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("holding review must wait for current data")
        ),
        _holding_shadow_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("holding shadow must wait for current data")
        ),
        _monthly_review_builder=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("monthly review must wait for current data")
        ),
    )

    assert outcome.exit_code == scheduled_sync.EXIT_INCOMPLETE
    assert "holding_review_status" not in outcome.event
    assert "holding_shadow_status" not in outcome.event
    assert "monthly_review_status" not in outcome.event


def test_non_trading_or_already_current_run_is_noop_success(tmp_path: Path) -> None:
    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: _report(current=True),
        _notifier=lambda _message: (_ for _ in ()).throw(AssertionError("must not notify")),
    )

    assert outcome.exit_code == 0
    assert outcome.event["status"] == "noop_current"


def test_invalid_scheduler_clock_returns_stable_error_exit(tmp_path: Path) -> None:
    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: datetime(2026, 8, 27, 21, 0),
        _run_update=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
        _notifier=lambda _message: (_ for _ in ()).throw(AssertionError("must not notify")),
    )

    assert outcome.exit_code == scheduled_sync.EXIT_ERROR
    assert outcome.event["status"] == "scheduler_setup_error"
    assert outcome.event["reason"] == "scheduler_initialization_error"


def test_incomplete_report_exits_one_notifies_once_and_deduplicates(tmp_path: Path) -> None:
    messages = []

    def notifier(message):
        messages.append(message)
        return _NotificationResult()

    common = {
        **_paths(tmp_path),
        "clock": lambda: NOW,
        "_run_update": lambda **_kwargs: _report(
            current=False,
            reason="DataQualityError: coverage below gate",
        ),
        "_notifier": notifier,
    }
    first = scheduled_sync.run_scheduled_sync(**common)
    second = scheduled_sync.run_scheduled_sync(**common)

    assert first.exit_code == scheduled_sync.EXIT_INCOMPLETE
    assert first.event["status"] == "incomplete"
    assert first.event["notification_successful_channels"] == ["serverchan", "bark"]
    assert second.exit_code == scheduled_sync.EXIT_INCOMPLETE
    assert second.event["notification_deduplicated"] is True
    assert len(messages) == 1
    assert "coverage below gate" in messages[0].body


def test_recovery_after_failure_sends_one_recovery_notice(tmp_path: Path) -> None:
    messages = []

    def notifier(message):
        messages.append(message)
        return _NotificationResult()

    common = {**_paths(tmp_path), "clock": lambda: NOW, "_notifier": notifier}
    scheduled_sync.run_scheduled_sync(
        **common,
        _run_update=lambda **_kwargs: _report(current=False, reason="temporary provider failure"),
    )
    recovery = scheduled_sync.run_scheduled_sync(
        **common,
        _run_update=lambda **_kwargs: _report(
            current=True,
            updated=(date(2026, 8, 27),),
        ),
    )

    assert recovery.exit_code == 0
    assert [message.title for message in messages] == [
        "A股收盘数据同步未完成",
        "A股收盘数据已恢复",
    ]


def test_expected_error_exits_two_and_redacts_url_and_assigned_secret(tmp_path: Path) -> None:
    secret = "never-print-this"
    messages = []

    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: (_ for _ in ()).throw(
            DataUnavailableError(
                f"provider failed https://example.test/path?token={secret} api_key={secret}"
            )
        ),
        _notifier=lambda message: messages.append(message) or _NotificationResult(),
    )

    assert outcome.exit_code == scheduled_sync.EXIT_ERROR
    serialized = json.dumps(outcome.event, ensure_ascii=False)
    log = (tmp_path / "logs" / "daily-sync.jsonl").read_text(encoding="utf-8")
    assert secret not in serialized
    assert secret not in log
    assert secret not in messages[0].body
    assert outcome.event["reason"] == "update_entrypoint_error"


def test_unexpected_error_never_copies_exception_message(tmp_path: Path) -> None:
    secret = "unexpected-secret"

    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
        _notifier=lambda _message: _NotificationResult(),
    )

    assert outcome.exit_code == scheduled_sync.EXIT_ERROR
    assert outcome.event["reason"] == "unexpected_scheduler_error"
    assert secret not in json.dumps(outcome.event)


def test_notification_boundary_failure_never_masks_sync_exit_code(tmp_path: Path) -> None:
    outcome = scheduled_sync.run_scheduled_sync(
        **_paths(tmp_path),
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: _report(
            current=False,
            reason="provider quality gate failed",
        ),
        _notifier=lambda _message: (_ for _ in ()).throw(RuntimeError("channel crashed")),
    )

    assert outcome.exit_code == scheduled_sync.EXIT_INCOMPLETE
    assert outcome.event["notification_failed_channels"] == ["notification_boundary"]


def test_recovery_notification_failure_never_masks_success(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    scheduled_sync.run_scheduled_sync(
        **paths,
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: _report(current=False, reason="temporary failure"),
        _notifier=lambda _message: _NotificationResult(),
    )

    outcome = scheduled_sync.run_scheduled_sync(
        **paths,
        clock=lambda: NOW,
        _run_update=lambda **_kwargs: _report(current=True),
        _notifier=lambda _message: (_ for _ in ()).throw(RuntimeError("channel crashed")),
    )

    assert outcome.exit_code == scheduled_sync.EXIT_CURRENT
    assert outcome.event["recovery_notification_failed_channels"] == ["notification_boundary"]


def test_sanitizer_covers_quoted_headers_and_known_bare_key_shapes() -> None:
    synthetic_serverchan_shape = "SCT" + "a" * 16
    synthetic_infoway_shape = "a" * 32 + "-infoway"
    secrets = (
        "quotedSecret",
        "bearerSecret",
        synthetic_serverchan_shape,
        synthetic_infoway_shape,
    )
    raw = (
        '{"apiKey":"quotedSecret"} Authorization: Bearer bearerSecret '
        f"{synthetic_serverchan_shape} {synthetic_infoway_shape}"
    )

    sanitized = scheduled_sync._sanitize_text(raw)

    assert all(secret not in sanitized for secret in secrets)
    assert "[redacted]" in sanitized


def test_second_process_lock_owner_is_harmless_noop(tmp_path: Path) -> None:
    scheduler_root = tmp_path / "scheduler"
    lock_path = scheduler_root / "daily-sync.lock"
    called = False

    def update(**_kwargs):
        nonlocal called
        called = True
        return _report(current=True)

    with daily_update_lock(lock_path) as acquired:
        assert acquired is True
        outcome = scheduled_sync.run_scheduled_sync(
            **_paths(tmp_path),
            clock=lambda: NOW,
            _run_update=update,
            _notifier=lambda _message: _NotificationResult(),
        )

    assert outcome.exit_code == 0
    assert outcome.event["status"] == "already_running"
    assert called is False


def test_shared_lock_is_really_exclusive_across_processes(tmp_path: Path) -> None:
    lock_path = tmp_path / "scheduler" / "daily-sync.lock"
    program = (
        "import sys\n"
        "from ashare_lab.services.daily_update_lock import daily_update_lock\n"
        "with daily_update_lock(sys.argv[1]) as acquired:\n"
        "    print('locked' if acquired else 'busy', flush=True)\n"
        "    sys.stdin.readline()\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", program, str(lock_path)],
        cwd=PROJECT_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "locked"
        with daily_update_lock(lock_path) as acquired:
            assert acquired is False
    finally:
        if child.stdin is not None:
            child.stdin.write("\n")
            child.stdin.flush()
        child.wait(timeout=5)

    assert child.returncode == 0
    with daily_update_lock(lock_path) as acquired_after_exit:
        assert acquired_after_exit is True


def test_cli_help_has_no_secret_arguments() -> None:
    help_text = scheduled_sync.build_parser().format_help().lower()

    assert "--csmar-root" in help_text
    assert "--overlay-root" in help_text
    assert "--scheduler-root" in help_text
    assert "--log-root" in help_text
    assert "--api-key" not in help_text
    assert "--sendkey" not in help_text
    assert "--token" not in help_text


def test_launchagent_template_is_independent_bounded_and_secret_free() -> None:
    document = plistlib.loads(PLIST_TEMPLATE.read_bytes())

    assert document["Label"] == "com.zerong.asharelab.daily-sync"
    assert document["Label"] != "com.zerong.asharelab"
    assert document["ProgramArguments"] == [
        "__PYTHON_BIN__",
        "-m",
        "ashare_lab.cli.scheduled_sync",
    ]
    assert document["RunAtLoad"] is True
    assert document["StartCalendarInterval"] == [
        {"Hour": 15, "Minute": 30},
        {"Hour": 18, "Minute": 30},
        {"Hour": 20, "Minute": 0},
    ]
    assert "KeepAlive" not in document
    assert document["ProcessType"] == "Background"
    assert document["LowPriorityIO"] is True
    assert document["Nice"] == 10
    assert document["Umask"] == 63
    assert document["StandardOutPath"] == "/dev/null"
    assert document["StandardErrorPath"] == "/dev/null"
    serialized = PLIST_TEMPLATE.read_text(encoding="utf-8").lower()
    assert "api_key" not in serialized
    assert "sendkey" not in serialized
    assert "device_key" not in serialized
    assert "sct" not in serialized


def test_launchagent_renderer_replaces_placeholders_without_inserting_arguments(
    tmp_path: Path,
) -> None:
    output = tmp_path / "rendered.plist"
    python_bin = tmp_path / "Project With Spaces" / ".venv" / "bin" / "python"
    project_root = tmp_path / "Project With Spaces"
    python_bin.parent.mkdir(parents=True)
    python_bin.symlink_to(sys.executable)

    scheduled_sync.render_launchagent_plist(
        PLIST_TEMPLATE,
        output,
        python_bin,
        project_root,
    )

    document = plistlib.loads(output.read_bytes())
    assert document["ProgramArguments"] == [
        str(python_bin.absolute()),
        "-m",
        "ashare_lab.cli.scheduled_sync",
    ]
    assert document["WorkingDirectory"] == str(project_root.resolve())
    assert document["StartCalendarInterval"] == [
        {"Hour": 15, "Minute": 30},
        {"Hour": 18, "Minute": 30},
        {"Hour": 20, "Minute": 0},
    ]
    assert "__PYTHON_BIN__" not in document["ProgramArguments"]
    assert output.stat().st_mode & 0o777 == 0o600


def test_installers_only_manage_daily_label_and_preserve_data_and_keys() -> None:
    install = INSTALL_SCRIPT.read_text(encoding="utf-8")
    uninstall = UNINSTALL_SCRIPT.read_text(encoding="utf-8")

    assert 'LABEL="com.zerong.asharelab.daily-sync"' in install
    assert 'LABEL="com.zerong.asharelab.daily-sync"' in uninstall
    assert "plutil -lint" in install
    assert "launchctl bootstrap" in install
    assert "launchctl bootout" in uninstall
    assert '"$EUID" -eq 0' in install
    assert "import ashare_lab.cli.scheduled_sync" in install
    assert "render_launchagent_plist" in install
    assert "plutil -replace ProgramArguments.0" not in install
    assert "15:30首次同步，18:30质量复核，20:00晚报前预检" in install
    assert "BACKUP_PLIST" in install
    assert "正在恢复安装前状态" in install
    assert "research.db" not in uninstall
    assert "market_overlay" not in uninstall
    assert "security delete" not in uninstall
    assert "com.zerong.asharelab.plist" not in install
    assert "com.zerong.asharelab.plist" not in uninstall


def test_pyproject_registers_scheduled_command() -> None:
    source = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'ashare-scheduled-sync = "ashare_lab.cli.scheduled_sync:main"' in source
