from __future__ import annotations

import json
import signal
import subprocess
from datetime import datetime

import pytest

from ashare_lab.cli import holding_pnl as cli

NOW = datetime(2026, 9, 22, 15, 35, tzinfo=cli.CN)


class Process:
    def __init__(self, outcomes, pid=44111):
        self.outcomes = list(outcomes)
        self.pid = pid
        self.returncode = None
        self.timeouts = []

    def wait(self, timeout):
        self.timeouts.append(timeout)
        outcome = self.outcomes.pop(0) if self.outcomes else self.returncode
        if outcome == "timeout":
            raise subprocess.TimeoutExpired("SCT-private-exception", timeout)
        self.returncode = outcome
        return outcome

    def poll(self):
        return self.returncode


@pytest.mark.parametrize(
    "now",
    [
        NOW.replace(hour=14),
        NOW.replace(minute=29),
        NOW.replace(minute=59),
        NOW.replace(hour=16),
        NOW.replace(day=26),
    ],
)
def test_outside_window_does_not_start_child(tmp_path, now):
    code, event = cli.supervise(
        root=tmp_path,
        clock=lambda: now,
        popen=lambda *_args, **_kwargs: pytest.fail("outside report window"),
    )
    assert code == 0 and event["status"] == "outside_report_window"


def test_normal_worker_is_isolated_and_budgeted(tmp_path):
    process, calls = Process([0]), []

    def popen(args, **kwargs):
        calls.append((args, kwargs))
        return process

    code, event = cli.supervise(
        root=tmp_path,
        clock=lambda: NOW,
        popen=popen,
        kill_group=lambda *_: pytest.fail("normal child must not be signaled"),
    )
    assert code == 0 and event["status"] == "holding_pnl_worker_completed"
    assert 87 <= process.timeouts[0] <= 88
    assert "once" in calls[0][0]
    assert calls[0][1] == {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "start_new_session": True,
    }


@pytest.mark.parametrize("outcomes", [[2], ["timeout", -9]])
def test_final_failure_kills_descendants_even_after_leader_exit_and_uses_same_report_fallback(
    tmp_path, outcomes
):
    child, fallback = Process(outcomes), Process([0], pid=44112)
    processes, calls, signals = iter([child, fallback]), [], []
    now = NOW.replace(minute=55)

    def popen(args, **kwargs):
        calls.append(args)
        return next(processes)

    code, event = cli.supervise(
        root=tmp_path,
        clock=lambda: now,
        popen=popen,
        kill_group=lambda pid, sig: signals.append((pid, sig)),
    )
    assert code == 2
    assert signals == [(child.pid, signal.SIGKILL)]
    assert child.timeouts[0] <= 74
    assert fallback.timeouts[0] <= 12
    assert "failure-notice" in calls[1]
    assert "SCT-private" not in json.dumps(event)
    assert "SCT-private" not in (tmp_path / "last-status.json").read_text()


def test_final_spawn_failure_also_attempts_bounded_fallback(tmp_path):
    calls, fallback = [], Process([0])

    def popen(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise OSError("private token")
        return fallback

    code, event = cli.supervise(root=tmp_path, clock=lambda: NOW.replace(minute=55), popen=popen)
    assert code == 2 and event["status"] == "holding_pnl_worker_start_or_wait_failed"
    assert "failure-notice" in calls[1]
    assert fallback.timeouts[0] <= 12


def test_early_failure_waits_for_next_scheduled_retry(tmp_path):
    calls = []

    def popen(args, **kwargs):
        calls.append(args)
        return Process([2])

    cli.supervise(root=tmp_path, clock=lambda: NOW, popen=popen, kill_group=lambda *_: None)
    assert len(calls) == 1


def test_late_start_preserves_fallback_and_termination_budget(tmp_path):
    now = NOW.replace(minute=58)
    child, fallback = Process([2]), Process([0])
    children = iter([child, fallback])
    cli.supervise(
        root=tmp_path,
        clock=lambda: now,
        popen=lambda *_args, **_kwargs: next(children),
        kill_group=lambda *_: None,
    )
    assert child.timeouts[0] <= 44
    assert fallback.timeouts[0] <= 12


def test_status_logs_allow_only_nonprivate_metadata(tmp_path):
    cli._record(
        tmp_path,
        {
            "status": "synthetic",
            "checked_at": NOW.isoformat(),
            "symbol": "600919",
            "pnl_amount": 12345,
            "body": "private",
        },
    )
    raw = (tmp_path / "last-status.json").read_text()
    log = (tmp_path / "logs" / "holding-pnl.jsonl").read_text()
    for value in (raw, log):
        assert "600919" not in value and "12345" not in value and "private" not in value


def test_final_sender_rechecks_guard_after_slow_keychain(monkeypatch):
    from ashare_lab.adapters import macos_keychain, notification_channels
    from ashare_lab.ports.notifications import NotificationMessage

    authorized = True

    def key():
        nonlocal authorized
        authorized = False
        return "SCTsyntheticTestOnlyKey"

    monkeypatch.setattr(macos_keychain, "load_serverchan_sendkey", key)
    monkeypatch.setattr(
        notification_channels.ServerChanNotificationChannel,
        "send",
        lambda *_: pytest.fail("revoked private disclosure must not submit"),
    )
    message = NotificationMessage(
        title="synthetic",
        body="private",
        holding_authorization_guard=lambda _: authorized,
        unauthorized_body="public",
    )
    assert cli.send_serverchan(message) is False
