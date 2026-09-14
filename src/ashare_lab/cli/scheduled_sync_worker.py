"""Bound the entire scheduled sync, including SDK, accounting and notifications."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from ashare_lab.bootstrap import application_data_dir
from ashare_lab.cli.evening_report_worker import terminate_worker_group

SYNC_TIMEOUT_SECONDS = 12 * 60


def supervise(argv, *, timeout=SYNC_TIMEOUT_SECONDS, popen=subprocess.Popen):
    if not 0 < timeout <= SYNC_TIMEOUT_SECONDS:
        raise ValueError("invalid sync deadline")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scheduler-root", type=Path, default=application_data_dir() / "scheduler")
    parser.add_argument("--log-root", type=Path, default=Path.home() / "Library/Logs/A股研究助手")
    args, _ = parser.parse_known_args(argv)
    child = None
    try:
        child = popen(
            [sys.executable, "-m", "ashare_lab.cli.scheduled_sync", *argv],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        reason = "sync_worker_deadline_exceeded"
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
        send_scheduled_notification,
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
            notifier=send_scheduled_notification,
        )
    except Exception:
        event["notification_status"] = "failure_notification_unavailable"
    finally:
        _write_log_event(args.log_root / "daily-sync.jsonl", event)
    return 2


def main(argv=None):
    return supervise(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
