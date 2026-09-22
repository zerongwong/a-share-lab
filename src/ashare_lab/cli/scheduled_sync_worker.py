"""Bound silent background sync and relinquish the pre-open report lock window."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from ashare_lab.bootstrap import application_data_dir
from ashare_lab.cli.evening_report_worker import (
    WORKER_TERMINATE_GRACE_SECONDS,
    terminate_worker_group,
)

SYNC_TIMEOUT_SECONDS = 12 * 60
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_RESERVATION_START = time(8, 18)
_RESERVATION_END = time(9, 0)
# terminate_worker_group has two bounded waits; also leave five seconds of
# scheduling slack so even delayed starts relinquish the data lock before 08:18.
SYNC_TERMINATION_RESERVE_SECONDS = 2 * WORKER_TERMINATE_GRACE_SECONDS + 5


def sync_timeout_budget(now: datetime, timeout: float) -> float | None:
    if now.utcoffset() is None:
        raise ValueError("sync supervisor clock must be timezone-aware")
    local = now.astimezone(_SHANGHAI)
    if local.weekday() >= 5:
        return float(timeout)
    local_time = local.timetz().replace(tzinfo=None)
    if _RESERVATION_START <= local_time < _RESERVATION_END:
        return None
    if local_time < _RESERVATION_START:
        boundary = datetime.combine(local.date(), _RESERVATION_START, tzinfo=_SHANGHAI)
        remaining = (boundary - local).total_seconds() - SYNC_TERMINATION_RESERVE_SECONDS
        return min(float(timeout), remaining) if remaining > 0 else None
    return float(timeout)


def supervise(argv, *, timeout=SYNC_TIMEOUT_SECONDS, popen=subprocess.Popen, clock=None):
    if not 0 < timeout <= SYNC_TIMEOUT_SECONDS:
        raise ValueError("invalid sync deadline")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scheduler-root", type=Path, default=application_data_dir() / "scheduler")
    parser.add_argument("--log-root", type=Path, default=Path.home() / "Library/Logs/A股研究助手")
    args, _ = parser.parse_known_args(argv)
    clock = clock or (lambda: datetime.now(UTC))
    now = clock()
    budget = sync_timeout_budget(now, timeout)
    if budget is None:
        from ashare_lab.cli.scheduled_sync import _write_log_event

        _write_log_event(
            args.log_root / "daily-sync.jsonl",
            {
                "job": "com.zerong.asharelab.daily-sync",
                "status": "deferred_for_morning_report",
                "exit_code": 0,
                "reason": "morning_report_lock_reservation",
                "timestamp": now.astimezone(UTC).isoformat(),
            },
        )
        return 0
    reservation_bounded = budget < timeout
    child = None
    try:
        child = popen(
            [sys.executable, "-m", "ashare_lab.cli.scheduled_sync", *argv],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Process setup can take time; recompute the absolute morning boundary
        # after spawn rather than giving a delayed child the old full budget.
        budget_after_spawn = sync_timeout_budget(clock(), timeout)
        if budget_after_spawn is None:
            reservation_bounded = True
            raise subprocess.TimeoutExpired("morning reservation", 0)
        budget = min(budget, budget_after_spawn)
        reservation_bounded = reservation_bounded or budget < timeout
        code = child.wait(timeout=budget)
        if code == 0:
            return code
        # A nonzero leader exit does not prove its helper descendants exited.
        # Only close this invocation's dedicated process group, then preserve
        # the sync CLI's own failure code and detailed local diagnostic.
        try:
            terminate_worker_group(child)
        except Exception:
            reason = "sync_worker_termination_unconfirmed"
        else:
            return code
    except subprocess.TimeoutExpired:
        reason = (
            "sync_worker_morning_reservation_deadline"
            if reservation_bounded
            else "sync_worker_deadline_exceeded"
        )
        try:
            terminate_worker_group(child)
        except Exception:
            reason = "sync_worker_termination_unconfirmed"
    except OSError:
        reason = "sync_worker_start_or_wait_failed"
        if child is not None:
            try:
                terminate_worker_group(child)
            except Exception:
                reason = "sync_worker_termination_unconfirmed"
    from ashare_lab.cli.scheduled_sync import (
        _handle_failure,
        _read_state,
        _write_log_event,
    )

    event = {
        "job": "com.zerong.asharelab.daily-sync",
        "status": "error",
        "exit_code": 2,
        "reason": reason,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    state_path = args.scheduler_root / "daily-sync-state.json"
    try:
        _handle_failure(
            event,
            state_path=state_path,
            prior_state=_read_state(state_path),
            notifier=None,
        )
    except Exception:
        event["local_state_status"] = "failure_state_unavailable"
    finally:
        _write_log_event(args.log_root / "daily-sync.jsonl", event)
    return 2


def main(argv=None):
    return supervise(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
