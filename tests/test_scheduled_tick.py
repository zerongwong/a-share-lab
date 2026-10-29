from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from ashare_lab.cli import scheduled_tick as tick

CN = ZoneInfo("Asia/Shanghai")
LA = ZoneInfo("America/Los_Angeles")


def stamp(hour=8, minute=20, *, day=29, month=10, second=0):
    return datetime(2026, month, day, hour, minute, second, tzinfo=CN)


@pytest.mark.parametrize("day,month", ((29, 10), (2, 11), (9, 3)))
def test_shanghai_schedule_ignores_host_timezone_and_us_dst(day, month):
    now = stamp(day=day, month=month)
    assert tick.due_slot("evening-report", now) == tick.due_slot(
        "evening-report", now.astimezone(LA)
    )
    assert tick.due_slot("evening-report", now) == f"2026-{month:02d}-{day:02d}_0820"


def test_shanghai_monday_runs_while_los_angeles_is_still_sunday():
    now = stamp(day=2, month=11).astimezone(LA)
    assert now.weekday() == 6
    assert tick.due_slot("evening-report", now) == "2026-11-02_0820"


@pytest.mark.parametrize(
    "job,hour,minute,expected",
    (
        ("evening-report", 8, 19, None),
        ("evening-report", 8, 20, "0820"),
        ("evening-report", 8, 29, "0820"),
        ("evening-report", 8, 30, "0830"),
        ("evening-report", 8, 57, "0850"),
        ("evening-report", 8, 58, None),
        ("preopen-deadline", 8, 57, None),
        ("preopen-deadline", 8, 58, "0858"),
        ("preopen-deadline", 8, 59, "0859"),
        ("preopen-deadline", 9, 0, None),
        ("holding-pnl", 15, 34, None),
        ("holding-pnl", 15, 35, "1535"),
        ("holding-pnl", 15, 54, "1545"),
        ("holding-pnl", 15, 59, "1555"),
        ("holding-pnl", 16, 0, None),
        ("daily-sync", 6, 30, "0630"),
        ("daily-sync", 6, 45, None),
        ("daily-sync", 8, 17, "0800"),
        ("daily-sync", 8, 18, None),
        ("daily-sync", 8, 58, None),
        ("daily-sync", 20, 34, "2020"),
        ("daily-sync", 20, 35, None),
    ),
)
def test_narrow_slots_and_boundaries(job, hour, minute, expected):
    value = tick.due_slot(job, stamp(hour, minute))
    assert value == (f"2026-10-29_{expected}" if expected else None)


def test_weekend_user_reports_quiet_but_sync_keeps_approved_updates():
    saturday = stamp(day=31)
    for job, hour, minute in (
        ("evening-report", 8, 20),
        ("holding-pnl", 15, 35),
        ("preopen-deadline", 8, 58),
    ):
        assert tick.due_slot(job, saturday.replace(hour=hour, minute=minute)) is None
    assert tick.due_slot("daily-sync", saturday.replace(hour=6, minute=30))


def test_naive_clock_and_unknown_job_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        tick.due_slot("evening-report", datetime(2026, 10, 29, 8, 20))
    with pytest.raises(ValueError, match="unknown"):
        tick.due_slot("arbitrary.module", stamp())


def test_outside_window_has_no_files_no_business_work_or_output(tmp_path, capsys):
    def forbidden(*args):
        pytest.fail("must not launch outside Shanghai window")

    root = tmp_path / "absent"
    for _ in range(20):
        assert (
            tick.dispatch(
                "evening-report", state_root=root, clock=lambda: stamp(20), execute=forbidden
            )
            == 0
        )
    assert not root.exists()
    assert capsys.readouterr() == ("", "")


def test_slot_is_claimed_once_across_restarts_but_next_slot_retries(tmp_path):
    called = []
    for minute in (20, 21, 22, 29):
        assert (
            tick.dispatch(
                "evening-report",
                state_root=tmp_path,
                clock=lambda minute=minute: stamp(8, minute),
                execute=lambda *args: called.append(args),
            )
            == 0
        )
    assert len(called) == 1
    assert called[0][1] == [sys.executable, "-m", "ashare_lab.cli.evening_report"]
    assert (
        tick.dispatch(
            "evening-report",
            state_root=tmp_path,
            clock=lambda: stamp(8, 30),
            execute=lambda *args: called.append(args),
        )
        == 0
    )
    assert len(called) == 2
    assert {p.name for p in (tmp_path / "evening-report").glob("*.claim")} == {
        "2026-10-29_0820.claim",
        "2026-10-29_0830.claim",
    }
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in (tmp_path / "evening-report").iterdir())


def test_restart_catches_up_current_slot_only_not_all_missed_slots(tmp_path):
    called = []
    tick.dispatch(
        "evening-report",
        state_root=tmp_path,
        clock=lambda: stamp(8, 47),
        execute=lambda *args: called.append(args),
    )
    assert len(called) == 1
    assert [p.name for p in (tmp_path / "evening-report").glob("*.claim")] == [
        "2026-10-29_0840.claim"
    ]


def test_concurrent_ticks_atomically_launch_once(tmp_path):
    called = []

    def run(_):
        return tick.dispatch(
            "evening-report",
            state_root=tmp_path,
            clock=lambda: stamp(),
            execute=lambda *args: called.append(args),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(run, range(20))) == [0] * 20
    assert len(called) == 1


def test_each_job_lock_is_independent_and_busy_job_does_not_claim_next_slot(tmp_path):
    root = tmp_path / "evening-report"
    root.mkdir()
    fd = os.open(root / "dispatch.lock", os.O_RDWR | os.O_CREAT, 0o600)
    called = []
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tick.dispatch(
            "evening-report",
            state_root=tmp_path,
            clock=lambda: stamp(),
            execute=lambda *args: called.append(args),
        )
        assert list(root.glob("*.claim")) == []
        tick.dispatch(
            "preopen-deadline",
            state_root=tmp_path,
            clock=lambda: stamp(8, 58),
            execute=lambda *args: called.append(args),
        )
        assert len(called) == 1
    finally:
        os.close(fd)


def test_failed_0858_does_not_prevent_independent_0859_retry(tmp_path):
    def failed(*_):
        raise OSError("private runtime detail not to log")

    assert (
        tick.dispatch(
            "preopen-deadline", state_root=tmp_path, clock=lambda: stamp(8, 58), execute=failed
        )
        == 2
    )
    called = []
    tick.dispatch(
        "preopen-deadline",
        state_root=tmp_path,
        clock=lambda: stamp(8, 58),
        execute=lambda *args: called.append(args),
    )
    assert called == []
    tick.dispatch(
        "preopen-deadline",
        state_root=tmp_path,
        clock=lambda: stamp(8, 59),
        execute=lambda *args: called.append(args),
    )
    assert len(called) == 1
    assert (tmp_path / "preopen-deadline" / "2026-10-29_0858.dispatch-failed").read_text() == ""


def test_delayed_filesystem_does_not_launch_after_slot_or_date_changes(tmp_path):
    times = iter((stamp(8, 59), stamp(9, 0)))
    called = []
    tick.dispatch(
        "preopen-deadline",
        state_root=tmp_path,
        clock=lambda: next(times),
        execute=lambda *args: called.append(args),
    )
    assert called == []
    assert list(tmp_path.rglob("*.claim")) == []


def test_state_storage_failure_fails_closed_and_leaves_other_jobs_available(tmp_path):
    unavailable = tmp_path / "not-a-directory"
    unavailable.write_text("fixture")
    called = []
    with pytest.raises(OSError):
        tick.dispatch(
            "evening-report",
            state_root=unavailable,
            clock=lambda: stamp(),
            execute=lambda *args: called.append(args),
        )
    assert called == []
    tick.dispatch(
        "preopen-deadline",
        state_root=tmp_path,
        clock=lambda: stamp(8, 59),
        execute=lambda *args: called.append(args),
    )
    assert len(called) == 1


def test_per_job_lock_survives_exec_without_orphan_wrapper_process(tmp_path):
    lock = str(tmp_path / "evening-report" / "dispatch.lock")
    child_code = (
        "import os,fcntl,sys\n"
        f"fd=os.open({lock!r},os.O_RDWR)\n"
        "try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        "except BlockingIOError: sys.exit(0)\n"
        "sys.exit(10)\n"
    )
    launcher = (
        "import os,sys\nfrom pathlib import Path\n"
        "from datetime import datetime\nfrom zoneinfo import ZoneInfo\n"
        "from ashare_lab.cli.scheduled_tick import dispatch\n"
        f"dispatch('evening-report',state_root=Path({str(tmp_path)!r}),"
        "clock=lambda:datetime(2026,10,29,8,20,tzinfo=ZoneInfo('Asia/Shanghai')),"
        f"execute=lambda *_:os.execv(sys.executable,[sys.executable,'-c',{child_code!r}]))\n"
    )
    result = subprocess.run([sys.executable, "-c", launcher], capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
