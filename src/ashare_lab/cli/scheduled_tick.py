"""Cheap, host-timezone-independent dispatch for the four local scheduled jobs.

launchd calendar times use the host timezone, not the child process's TZ.  A
minute tick therefore checks explicit Shanghai slots before loading any market,
holding or notification code.  Each job has its own lock and durable slot claim:
restarts catch up only the currently open slot, never every missed invocation.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TICK_MODULE = "ashare_lab.cli.scheduled_tick"
TICK_SECONDS = 60
SHANGHAI = ZoneInfo("Asia/Shanghai")
JOB_MODULES = {
    "daily-sync": "ashare_lab.cli.scheduled_sync_worker",
    "evening-report": "ashare_lab.cli.evening_report",
    "preopen-deadline": "ashare_lab.cli.preopen_deadline",
    "holding-pnl": "ashare_lab.cli.holding_pnl",
}
# Minute-of-day windows are left-inclusive/right-exclusive.  A delayed wake
# may use the current slot once, but cannot replay stale slots.  Daily sync
# retains weekend updates; the three user reports remain weekday-only and
# their own CLIs still independently verify the official trading calendar.
SLOT_WINDOWS = {
    "daily-sync": (
        (390, 405),
        (450, 465),
        (480, 498),
        (930, 945),
        (990, 1005),
        (1110, 1125),
        (1220, 1235),
    ),
    "evening-report": ((500, 510), (510, 520), (520, 530), (530, 538)),
    "preopen-deadline": ((538, 539), (539, 540)),
    "holding-pnl": ((935, 945), (945, 955), (955, 960)),
}


def due_slot(job: str, now: datetime) -> str | None:
    """Return a Shanghai date/slot key, independent of host timezone and DST."""
    if job not in JOB_MODULES:
        raise ValueError("unknown scheduled job")
    if now.utcoffset() is None:
        raise ValueError("scheduled tick requires a timezone-aware clock")
    local = now.astimezone(SHANGHAI)
    if job != "daily-sync" and local.weekday() >= 5:
        return None
    minute = local.hour * 60 + local.minute
    for start, end in SLOT_WINDOWS[job]:
        if start <= minute < end:
            return f"{local.date().isoformat()}_{start // 60:02d}{start % 60:02d}"
    return None


def dispatch(
    job: str,
    *,
    state_root: Path | None = None,
    clock: Callable[[], datetime] | None = None,
    execute: Callable = os.execv,
) -> int:
    """Atomically claim one slot and replace this process with its bounded CLI.

    The per-job flock survives exec, so another tick cannot start the same job
    while its supervisor is running.  It never blocks the independent deadline
    watchdog.  An attempted slot stays claimed after a crash or exec failure;
    the next approved slot can retry, without minute-by-minute expensive work.
    """
    clock = clock or (lambda: datetime.now(UTC))
    now = clock()
    slot = due_slot(job, now)
    if slot is None:
        return 0  # no app-directory creation, credential lookup or routine log
    if state_root is None:
        from ashare_lab.bootstrap import application_data_dir

        state_root = application_data_dir() / "scheduler" / "scheduled-ticks"
    root = Path(state_root).expanduser() / job
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_fd = os.open(root / "dispatch.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        # Directory/lock operations may have been delayed.  Never launch an
        # old slot after its deadline or across a Shanghai date boundary.
        if due_slot(job, clock()) != slot:
            return 0
        try:
            claim_fd = os.open(root / f"{slot}.claim", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return 0
        with os.fdopen(claim_fd, "w", encoding="utf-8") as claim:
            json.dump({"job": job, "slot": slot, "claimed_at": now.isoformat()}, claim)
            claim.flush()
            os.fsync(claim.fileno())
        if due_slot(job, clock()) != slot:
            return 0
        os.set_inheritable(lock_fd, True)
        try:
            execute(sys.executable, [sys.executable, "-m", JOB_MODULES[job]])
        except OSError:
            # Preserve the claim: no unbounded repeat after a broken runtime.
            # This metadata contains no symbols, account details or secrets.
            failure = root / f"{slot}.dispatch-failed"
            fd = os.open(failure, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
            return 2
        return 0  # only injected test executors return; real exec replaces us
    finally:
        os.close(lock_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dispatch approved Shanghai-time job slots.")
    parser.add_argument("--job", choices=tuple(JOB_MODULES), required=True)
    parser.add_argument("--state-root", type=Path)
    args = parser.parse_args(argv)
    try:
        return dispatch(args.job, state_root=args.state_root)
    except (OSError, ValueError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
