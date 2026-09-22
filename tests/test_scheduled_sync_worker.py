from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime

import pytest

from ashare_lab.cli import scheduled_sync
from ashare_lab.cli import scheduled_sync_worker as worker

NOW = datetime(2026, 9, 22, 6, tzinfo=UTC)


class Child:
    def __init__(self, code=0):
        self.code = code

    def wait(self, timeout):
        assert timeout == 720
        if self.code == "timeout":
            raise subprocess.TimeoutExpired("synthetic", timeout)
        return self.code


def test_sync_supervisor_has_wall_deadline_and_owned_process_group(tmp_path):
    calls = []

    def popen(args, **kwargs):
        calls.append((args, kwargs))
        return Child()

    assert (
        worker.supervise(
            ["--scheduler-root", str(tmp_path / "state")], popen=popen, clock=lambda: NOW
        )
        == 0
    )
    assert calls[0][0][:3] == [sys.executable, "-m", "ashare_lab.cli.scheduled_sync"]
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["stderr"] == subprocess.DEVNULL


def test_timeout_ends_owned_job_and_writes_failure_not_success(tmp_path, monkeypatch):
    child = Child("timeout")
    killed = []
    monkeypatch.setattr(worker, "terminate_worker_group", lambda p: killed.append(p))
    notices = []
    monkeypatch.setattr(
        scheduled_sync, "_handle_failure", lambda event, **_: notices.append(event.copy())
    )
    code = worker.supervise(
        ["--scheduler-root", str(tmp_path / "state"), "--log-root", str(tmp_path / "logs")],
        popen=lambda *_args, **_kwargs: child,
        clock=lambda: NOW,
    )
    assert code == 2 and killed == [child]
    assert notices[0]["reason"] == "sync_worker_deadline_exceeded"
    log = json.loads((tmp_path / "logs/daily-sync.jsonl").read_text())
    assert log["status"] == "error"


@pytest.mark.parametrize("code", [1, 2, -11])
def test_nonzero_sync_exit_closes_only_its_owned_group_and_preserves_failure(
    tmp_path, monkeypatch, code
):
    child, killed = Child(code), []
    unrelated_child = Child()
    monkeypatch.setattr(worker, "terminate_worker_group", lambda p: killed.append(p))
    result = worker.supervise(
        ["--scheduler-root", str(tmp_path / "state"), "--log-root", str(tmp_path / "logs")],
        popen=lambda *_a, **_k: child,
        clock=lambda: NOW,
    )
    assert result == code
    assert killed == [child]
    assert unrelated_child not in killed


def test_nonzero_sync_cleanup_failure_is_logged_as_unconfirmed(tmp_path, monkeypatch):
    def denied(_process):
        raise PermissionError("synthetic owned group cannot be signalled")

    monkeypatch.setattr(worker, "terminate_worker_group", denied)
    result = worker.supervise(
        ["--scheduler-root", str(tmp_path / "state"), "--log-root", str(tmp_path / "logs")],
        popen=lambda *_a, **_k: Child(1),
        clock=lambda: NOW,
    )
    assert result == 2
    event = json.loads((tmp_path / "logs/daily-sync.jsonl").read_text())
    assert event["reason"] == "sync_worker_termination_unconfirmed"


def test_spawn_failure_is_logged_even_when_notification_fails(tmp_path, monkeypatch):
    def failed(*_args, **_kwargs):
        raise OSError("synthetic-secret")

    monkeypatch.setattr(scheduled_sync, "_handle_failure", failed)
    code = worker.supervise(
        ["--scheduler-root", str(tmp_path / "state"), "--log-root", str(tmp_path / "logs")],
        popen=failed,
        clock=lambda: NOW,
    )
    assert code == 2
    raw = (tmp_path / "logs/daily-sync.jsonl").read_text()
    assert "synthetic-secret" not in raw
    assert json.loads(raw)["reason"] == "sync_worker_start_or_wait_failed"
    assert json.loads(raw)["local_state_status"] == "failure_state_unavailable"


@pytest.mark.parametrize(
    ("hour", "minute", "second", "expected"),
    [
        (0, 0, 0, 720),
        (0, 10, 0, 465),
        (0, 17, 0, 45),
        (0, 17, 45, None),
        (0, 18, 0, None),
        (0, 59, 59, None),
        (1, 0, 0, 720),
    ],
)
def test_morning_budget_reserves_process_group_termination_time(hour, minute, second, expected):
    now = datetime(2026, 9, 22, hour, minute, second, tzinfo=UTC)
    assert worker.sync_timeout_budget(now, 720) == expected


def test_weekend_has_no_weekday_preopen_reservation():
    assert worker.sync_timeout_budget(datetime(2026, 9, 26, 0, 30, tzinfo=UTC), 720) == 720


def test_reserved_window_never_starts_child_or_notifies(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(
        scheduled_sync, "send_scheduled_notification", lambda _: called.append("notify")
    )

    def popen(*args, **kwargs):
        called.append("child")
        raise AssertionError("must not launch")

    assert (
        worker.supervise(
            ["--scheduler-root", str(tmp_path / "state"), "--log-root", str(tmp_path / "logs")],
            popen=popen,
            clock=lambda: datetime(2026, 9, 22, 0, 18, tzinfo=UTC),
        )
        == 0
    )
    assert called == []
    assert (
        json.loads((tmp_path / "logs/daily-sync.jsonl").read_text())["status"]
        == "deferred_for_morning_report"
    )


def test_delayed_start_budget_is_clamped_then_group_terminated_before_reservation(
    tmp_path, monkeypatch
):
    budgets, killed, notices = [], [], []

    class DelayedChild:
        def wait(self, timeout):
            budgets.append(timeout)
            raise subprocess.TimeoutExpired("synthetic", timeout)

    child = DelayedChild()
    monkeypatch.setattr(worker, "terminate_worker_group", lambda p: killed.append(p))
    monkeypatch.setattr(
        scheduled_sync, "send_scheduled_notification", lambda message: notices.append(message)
    )
    now = datetime(2026, 9, 22, 0, 10, tzinfo=UTC)
    code = worker.supervise(
        ["--scheduler-root", str(tmp_path / "state"), "--log-root", str(tmp_path / "logs")],
        popen=lambda *a, **k: child,
        clock=lambda: now,
    )
    assert budgets == [465]
    assert killed == [child] and code == 2
    assert notices == []
    event = json.loads((tmp_path / "logs/daily-sync.jsonl").read_text())
    assert event["reason"] == "sync_worker_morning_reservation_deadline"
    assert event["notification_policy"] == "automatic_sync_silent"


def test_spawn_delay_cannot_cross_reservation_and_keep_the_old_budget(tmp_path, monkeypatch):
    times = iter(
        [datetime(2026, 9, 22, 0, 17, tzinfo=UTC), datetime(2026, 9, 22, 0, 18, tzinfo=UTC)]
    )
    child, killed = Child(), []
    monkeypatch.setattr(worker, "terminate_worker_group", lambda p: killed.append(p))
    code = worker.supervise(
        ["--scheduler-root", str(tmp_path / "state"), "--log-root", str(tmp_path / "logs")],
        popen=lambda *a, **k: child,
        clock=lambda: next(times),
    )
    assert code == 2 and killed == [child]
