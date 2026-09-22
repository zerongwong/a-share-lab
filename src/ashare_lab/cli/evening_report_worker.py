"""Bound one automatic pre-open attempt without exposing child output or secrets.

The report runs in its own process group. Only that group may be terminated;
kernel-owned data locks are then released. Timeout notices have a separate
per-evening receipt from the actual next-session plan.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import datetime, time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

from ashare_lab.bootstrap import application_data_dir

WORKER_TIMEOUT_SECONDS = 8 * 60
WORKER_TERMINATE_GRACE_SECONDS = 5
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_REPORT_START = time(8, 45)
_REPORT_END = time(9, 30)
_LAST_RETRY = time(9, 20)


def _read_state(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_notice_state(path: Path, now: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps({"accepted_date": now.date().isoformat(), "delivery_confirmed": False}),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_log(log_root: Path, event: dict) -> None:
    """Only fixed outcome codes reach disk; never forward worker output."""
    try:
        log_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = log_root / "evening-report.jsonl"
        handler = RotatingFileHandler(path, maxBytes=1_048_576, backupCount=5, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        try:
            handler.emit(
                logging.LogRecord(
                    "ashare.evening.worker",
                    logging.INFO,
                    "",
                    0,
                    json.dumps(
                        {"logged_at": datetime.now(_SHANGHAI).isoformat(), **event},
                        ensure_ascii=False,
                    ),
                    (),
                    None,
                )
            )
        finally:
            handler.close()
        os.chmod(path, 0o600)
    except Exception:
        # Even a broken log path must not leak an exception or hide exit status.
        return


def _send_serverchan_notice(message) -> bool:
    from ashare_lab.cli.evening_digest import send_serverchan_digest

    return send_serverchan_digest(message)


def notify_incomplete_once(
    *, state_root: Path, now: datetime, notifier: Callable | None = None
) -> str:
    """A plain public notice, independent from recommendation deduplication."""
    try:
        plan_state = _read_state(state_root / "evening-digest-state.json")
        if plan_state.get(
            "plan_for_date"
        ) == now.date().isoformat() and "serverchan" in plan_state.get("accepted_channels", []):
            return "plan_already_provider_accepted"
        path = state_root / "evening-failure-notice-state.json"
        if _read_state(path).get("accepted_date") == now.date().isoformat():
            return "failure_notice_already_provider_accepted"
        from ashare_lab.ports.notifications import NotificationMessage

        accepted = (notifier or _send_serverchan_notice)(
            NotificationMessage(
                title="A股盘前计划暂未完成｜暂停新买",
                body=(
                    "今日盘前研究计划计算或提交尚未完成确认。"
                    "暂不依据旧报告新买；请以随后完整计划为准。"
                    "本条未确认任何持仓的买卖信号。"
                ),
                group="A股研究室·盘前计划",
            )
        )
        if accepted is True:
            _write_notice_state(path, now)
            return "failure_notice_provider_accepted"
        return "failure_notice_provider_not_accepted"
    except Exception:
        return "failure_notice_provider_not_accepted"


def terminate_worker_group(process, *, kill_group: Callable = os.killpg) -> None:
    """Terminate only the process group created by this invocation."""
    with suppress(ProcessLookupError):
        kill_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=WORKER_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # A leader can exit while one of its descendants ignores SIGTERM.
        # Always close out the dedicated group before declaring its locks free.
        with suppress(ProcessLookupError):
            kill_group(process.pid, signal.SIGKILL)
    process.wait(timeout=WORKER_TERMINATE_GRACE_SECONDS)


def supervise_evening_report(
    argv: Sequence[str],
    *,
    timeout_seconds: float = WORKER_TIMEOUT_SECONDS,
    _popen: Callable = subprocess.Popen,
    _kill_group: Callable = os.killpg,
    _clock: Callable[[], datetime] | None = None,
    _notifier: Callable | None = None,
) -> tuple[int, dict]:
    """Run one regular 08:45–09:29 attempt with a finite wall-clock lifetime."""
    if not 0 < timeout_seconds <= WORKER_TIMEOUT_SECONDS:
        raise ValueError("worker timeout must be within the automatic attempt budget")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parsed, _ = parser.parse_known_args(list(argv))
    state_root = (parsed.state_root or application_data_dir() / "scheduler").expanduser()
    log_root = (parsed.log_root or Path.home() / "Library" / "Logs" / "A股研究助手").expanduser()
    now = (_clock or (lambda: datetime.now(_SHANGHAI)))().astimezone(_SHANGHAI)

    def finish(code: int, status: str, *, reason: str | None = None):
        event = {"job": "ashare-evening-worker", "status": status, "exit_code": code}
        if reason is not None:
            event["reason"] = reason
        _safe_log(log_root, event)
        return code, event

    local_time = now.timetz().replace(tzinfo=None)
    if now.weekday() in {5, 6} or not _REPORT_START <= local_time < _REPORT_END:
        return finish(0, "noop_outside_preopen_window")
    # A late login must not start an eight-minute build past the opening bell.
    deadline = now.replace(hour=9, minute=30, second=0, microsecond=0)
    attempt_timeout = min(timeout_seconds, max(0.1, (deadline - now).total_seconds() - 5))
    process = None
    try:
        _safe_log(log_root, {"job": "ashare-evening-worker", "status": "worker_started"})
        process = _popen(
            [sys.executable, "-m", "ashare_lab.cli.evening_digest", *argv],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        code = process.wait(timeout=attempt_timeout)
    except subprocess.TimeoutExpired:
        try:
            if process is not None:
                terminate_worker_group(process, kill_group=_kill_group)
        except Exception:
            return finish(2, "error", reason="evening_worker_termination_unconfirmed")
        outcome = finish(2, "error", reason="evening_worker_deadline_exceeded")
        if local_time >= _LAST_RETRY:
            notice = notify_incomplete_once(state_root=state_root, now=now, notifier=_notifier)
            _safe_log(
                log_root,
                {"job": "ashare-evening-worker", "status": notice, "delivery_confirmed": False},
            )
        return outcome
    except Exception:
        if process is not None:
            with suppress(Exception):
                terminate_worker_group(process, kill_group=_kill_group)
        return finish(2, "error", reason="evening_worker_start_or_wait_failed")
    if code == 0:
        return finish(0, "worker_completed")
    if code != 0 and local_time >= _LAST_RETRY:
        notice = notify_incomplete_once(state_root=state_root, now=now, notifier=_notifier)
        _safe_log(
            log_root,
            {"job": "ashare-evening-worker", "status": notice, "delivery_confirmed": False},
        )
    return finish(1 if code == 1 else 2, "error", reason="evening_worker_not_completed")
