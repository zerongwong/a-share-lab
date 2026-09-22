"""Small, independent pre-open status watchdog; never builds a trading plan.

The public helper shares the notice lock and accepted receipt with automatic
report failures. Its isolated sender is wall-clock bounded. Unknown network
outcomes remain unknown and may be retried in the two deadline slots: accepted
receipt deduplication is not an exactly-once delivery guarantee.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from ashare_lab.bootstrap import application_data_dir

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_NOTICE_START = time(8, 50)
_WATCHDOG_START = time(8, 58)
_SEND_END = time(9, 0)
NOTICE_TIMEOUT_SECONDS = 12.0
_NOTICE_LOCK = "preopen-notice.lock"
_NOTICE_STATE = "evening-failure-notice-state.json"


def _now() -> datetime:
    return datetime.now(_SHANGHAI)


def _read_state(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_notice_state(path: Path, now: datetime) -> None:
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


def _within_notice_window(now: datetime, expected_date: str) -> bool:
    local = now.astimezone(_SHANGHAI)
    return (
        local.date().isoformat() == expected_date
        and local.weekday() < 5
        and _NOTICE_START <= local.timetz().replace(tzinfo=None) < _SEND_END
    )


def _known_session(state_root: Path, now: datetime) -> bool | None:
    state = _read_state(state_root / "preopen-session-state.json")
    if state.get("date") != now.date().isoformat() or type(state.get("is_trading_day")) is not bool:
        return None
    try:
        checked = datetime.fromisoformat(state["checked_at"])
        if checked.tzinfo is None or checked > now:
            return None
        if checked.astimezone(_SHANGHAI).date() != now.date():
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return state["is_trading_day"]


def _existing_receipt(state_root: Path, now: datetime) -> str | None:
    state = _read_state(state_root / "evening-digest-state.json")
    channels = state.get("accepted_channels")
    if (
        state.get("plan_for_date") == now.date().isoformat()
        and isinstance(channels, list)
        and "serverchan" in channels
    ):
        return "plan_already_provider_accepted"
    if _read_state(state_root / _NOTICE_STATE).get("accepted_date") == now.date().isoformat():
        return "failure_notice_already_provider_accepted"
    return None


def _send_serverchan_notice(message, *, before_send: Callable[[], bool]) -> bool:
    # Deliberately avoid evening_digest: importing analytics is not necessary
    # for a public system-status notice, and its normal transport retries DNS.
    import httpx

    from ashare_lab.adapters.macos_keychain import load_serverchan_sendkey
    from ashare_lab.adapters.notification_channels import ServerChanNotificationChannel

    key = load_serverchan_sendkey()
    if not key:
        return False
    with (
        httpx.Client(
            timeout=httpx.Timeout(3.0, connect=2.0),
            transport=httpx.HTTPTransport(retries=0, trust_env=False),
            follow_redirects=False,
            trust_env=False,
        ) as client,
        ServerChanNotificationChannel(key, client=client) as channel,
    ):
        # This check is after credential and client setup, immediately
        # before the sole provider attempt. The supervisor bounds it too.
        if not before_send():
            return False
        receipt = channel.send(message)
    return receipt.accepted is True and receipt.provider_status == "provider_accepted"


def _notify_locked(
    *,
    state_root: Path,
    expected_date: str,
    clock: Callable[[], datetime],
    notifier: Callable | None,
) -> str:
    descriptor = None
    acquired = False
    try:
        now = clock().astimezone(_SHANGHAI)
        if not _within_notice_window(now, expected_date):
            return "noop_outside_preopen_notice_window"
        state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(state_root / _NOTICE_LOCK, os.O_CREAT | os.O_RDWR, 0o600)
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            return "notice_already_running"
        now = clock().astimezone(_SHANGHAI)
        if not _within_notice_window(now, expected_date):
            return "noop_outside_preopen_notice_window"
        existing = _existing_receipt(state_root, now)
        if existing:
            return existing
        session = _known_session(state_root, now)
        if session is False:
            return "noop_verified_nontrading_day"
        from ashare_lab.ports.notifications import NotificationMessage

        if session is True:
            title = "A股盘前计划暂未完成｜暂停新买"
            body = "盘前计划尚未完成或未获推送受理确认。请勿沿用旧买入建议。本条不是荐股结论，也未确认持仓买卖信号。"
        else:
            title = "A股盘前计划／交易日核验尚未完成"
            body = "盘前计划或交易日核验尚未完成，请勿沿用旧买入建议。本条仅为系统状态提醒，不代表今日一定开市或没有合适股票。"
        message = NotificationMessage(title=title, body=body, group="A股研究室·盘前状态")
        last_guard_status = None

        def before_send() -> bool:
            nonlocal last_guard_status
            checked_now = clock().astimezone(_SHANGHAI)
            if not _within_notice_window(checked_now, expected_date):
                last_guard_status = "noop_outside_preopen_notice_window"
            elif _known_session(state_root, checked_now) is False:
                last_guard_status = "noop_verified_nontrading_day"
            else:
                last_guard_status = _existing_receipt(state_root, checked_now)
            return last_guard_status is None

        if not before_send():
            return last_guard_status
        accepted = (
            notifier(message)
            if notifier is not None
            else _send_serverchan_notice(message, before_send=before_send)
        )
        if last_guard_status:
            return last_guard_status
        if accepted is True:
            # Date is tied to the guarded submission, not the response time.
            _write_notice_state(state_root / _NOTICE_STATE, now)
            return "failure_notice_provider_accepted"
        return "failure_notice_provider_not_accepted"
    except Exception:
        return "failure_notice_provider_not_accepted"
    finally:
        if descriptor is not None:
            if acquired:
                with suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def notify_incomplete_once(
    *,
    state_root: Path,
    now: datetime | None = None,
    notifier: Callable | None = None,
    clock: Callable[[], datetime] | None = None,
) -> str:
    """Bounded ServerChan-only fallback, deduplicated after confirmed acceptance.

    ``notifier`` and ``clock`` are injectable for offline tests. Production
    calls must leave notifier unset to retain the process-wide timeout.
    If the provider response is lost, no accepted receipt is invented; a later
    bounded retry may duplicate an earlier submission whose outcome is unknown.
    """
    actual_clock = clock or _now
    initial = (now or actual_clock()).astimezone(_SHANGHAI)
    expected_date = initial.date().isoformat()
    current = actual_clock().astimezone(_SHANGHAI)
    if not _within_notice_window(current, expected_date):
        return "noop_outside_preopen_notice_window"
    if notifier is not None:
        return _notify_locked(
            state_root=state_root,
            expected_date=expected_date,
            clock=actual_clock,
            notifier=notifier,
        )
    deadline = current.replace(hour=9, minute=0, second=0, microsecond=0)
    budget = min(NOTICE_TIMEOUT_SECONDS, (deadline - current).total_seconds() - 1)
    if budget <= 0:
        return "noop_insufficient_notice_budget"
    process = None
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ashare_lab.cli.preopen_deadline",
                "--notice-child",
                "--state-root",
                str(state_root),
                "--expected-date",
                expected_date,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            text=True,
        )
        current = actual_clock().astimezone(_SHANGHAI)
        budget = min(budget, (deadline - current).total_seconds() - 1)
        if budget <= 0:
            raise subprocess.TimeoutExpired("preopen-status", 0)
        stdout, _ = process.communicate(timeout=budget)
        value = json.loads(stdout)
        status = value.get("status") if isinstance(value, dict) else None
        # Never propagate arbitrary child output or provider exceptions.
        allowed = {
            "plan_already_provider_accepted",
            "failure_notice_already_provider_accepted",
            "failure_notice_provider_accepted",
            "failure_notice_provider_not_accepted",
            "noop_outside_preopen_notice_window",
            "noop_verified_nontrading_day",
            "notice_already_running",
        }
        return (
            status
            if process.returncode == 0 and status in allowed
            else "failure_notice_worker_failed"
        )
    except subprocess.TimeoutExpired:
        return "failure_notice_deadline_exceeded"
    except Exception:
        return "failure_notice_worker_failed"
    finally:
        if process is not None and process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=1)


def run_preopen_deadline(
    *,
    state_root: Path,
    clock: Callable[[], datetime] | None = None,
    notifier: Callable | None = None,
) -> tuple[int, dict]:
    actual_clock = clock or _now
    now = actual_clock().astimezone(_SHANGHAI)
    if now.weekday() >= 5 or not _WATCHDOG_START <= now.timetz().replace(tzinfo=None) < _SEND_END:
        status = "noop_outside_preopen_watchdog_window"
    else:
        status = notify_incomplete_once(
            state_root=state_root, now=now, clock=actual_clock, notifier=notifier
        )
    failed = status in {
        "failure_notice_provider_not_accepted",
        "failure_notice_worker_failed",
        "failure_notice_deadline_exceeded",
    }
    return (2 if failed else 0), {
        "job": "ashare-preopen-deadline",
        "status": status,
        "delivery_confirmed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check only the pre-open notification receipt.")
    parser.add_argument("--state-root", type=Path)
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path.home() / "Library" / "Logs" / "A股研究助手",
    )
    parser.add_argument("--notice-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-date", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    state_root = (args.state_root or application_data_dir() / "scheduler").expanduser()
    if args.notice_child:
        status = _notify_locked(
            state_root=state_root, expected_date=args.expected_date or "", clock=_now, notifier=None
        )
        print(json.dumps({"status": status}))
        return 0
    code, event = run_preopen_deadline(state_root=state_root)
    from ashare_lab.cli.evening_report_worker import _safe_log

    # Installed launchd tasks discard stdout/stderr. Always persist the safe
    # result, including timeouts and unknown provider outcomes, without keys.
    _safe_log(args.log_root.expanduser(), event)
    print(json.dumps(event, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
