from __future__ import annotations

import json
import plistlib
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

from ashare_lab.cli import scheduled_sync
from ashare_lab.domain.errors import DataUnavailableError
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


def _report(*, current: bool, updated: tuple[date, ...] = (), reason: str = ""):
    failures = ()
    if reason:
        failures = (SimpleNamespace(reason=reason),)
    return SimpleNamespace(
        source_id="infoway",
        requested_complete_date=date(2026, 8, 27),
        latest_complete_session=date(2026, 8, 27),
        common_cutoff=date(2026, 8, 27) if current else date(2026, 8, 26),
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
    assert "15:30首次同步，18:30质量复核" in install
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
