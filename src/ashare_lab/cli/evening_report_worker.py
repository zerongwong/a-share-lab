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
from ashare_lab.cli.preopen_deadline import notify_incomplete_once

WORKER_TIMEOUT_SECONDS = 8 * 60
WORKER_TERMINATE_GRACE_SECONDS = 5
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_REPORT_START = time(8, 20)
_REPORT_END = time(8, 58)
_LAST_RETRY = time(8, 50)
_TERMINATION_RESERVE_SECONDS = 2 * WORKER_TERMINATE_GRACE_SECONDS


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
    """Run one 08:20–08:57 attempt, reserving shutdown time before 08:58."""
    if not 0 < timeout_seconds <= WORKER_TIMEOUT_SECONDS:
        raise ValueError("worker timeout must be within the automatic attempt budget")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parsed, _ = parser.parse_known_args(list(argv))
    state_root = (parsed.state_root or application_data_dir() / "scheduler").expanduser()
    log_root = (parsed.log_root or Path.home() / "Library" / "Logs" / "A股研究助手").expanduser()
    clock = _clock or (lambda: datetime.now(_SHANGHAI))
    now = clock().astimezone(_SHANGHAI)

    def finish(code: int, status: str, *, reason: str | None = None):
        event = {"job": "ashare-evening-worker", "status": status, "exit_code": code}
        if reason is not None:
            event["reason"] = reason
        _safe_log(log_root, event)
        return code, event

    def final_notice() -> None:
        # A long attempt must not keep its start time as its delivery clock.
        fresh_now = clock().astimezone(_SHANGHAI)
        if fresh_now.date() != now.date() or fresh_now.timetz().replace(tzinfo=None) < _LAST_RETRY:
            return
        notice = notify_incomplete_once(
            state_root=state_root, now=fresh_now, notifier=_notifier, clock=clock
        )
        _safe_log(
            log_root,
            {"job": "ashare-evening-worker", "status": notice, "delivery_confirmed": False},
        )

    local_time = now.timetz().replace(tzinfo=None)
    if now.weekday() in {5, 6} or not _REPORT_START <= local_time < _REPORT_END:
        return finish(0, "noop_outside_preopen_window")
    deadline = now.replace(hour=8, minute=58, second=0, microsecond=0)

    def remaining_timeout() -> float:
        return min(
            timeout_seconds,
            (deadline - clock().astimezone(_SHANGHAI)).total_seconds()
            - _TERMINATION_RESERVE_SECONDS,
        )

    if remaining_timeout() <= 0:
        final_notice()
        return finish(0, "noop_insufficient_worker_budget")
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
        attempt_timeout = remaining_timeout()
        if attempt_timeout <= 0:
            raise subprocess.TimeoutExpired("automatic-preopen-worker", 0)
        code = process.wait(timeout=attempt_timeout)
    except subprocess.TimeoutExpired:
        try:
            if process is not None:
                terminate_worker_group(process, kill_group=_kill_group)
        except Exception:
            final_notice()
            return finish(2, "error", reason="evening_worker_termination_unconfirmed")
        outcome = finish(2, "error", reason="evening_worker_deadline_exceeded")
        final_notice()
        return outcome
    except Exception:
        if process is not None:
            with suppress(Exception):
                terminate_worker_group(process, kill_group=_kill_group)
        final_notice()
        return finish(2, "error", reason="evening_worker_start_or_wait_failed")
    if code == 0:
        return finish(0, "worker_completed")
    # A crashed leader can leave network/helper descendants alive. The owned
    # session is still ours to close even though wait() already reaped its leader.
    try:
        terminate_worker_group(process, kill_group=_kill_group)
    except Exception:
        final_notice()
        return finish(2, "error", reason="evening_worker_termination_unconfirmed")
    final_notice()
    return finish(1 if code == 1 else 2, "error", reason="evening_worker_not_completed")
