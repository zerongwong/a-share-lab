from __future__ import annotations

import json
import subprocess
import sys

from ashare_lab.cli import scheduled_sync
from ashare_lab.cli import scheduled_sync_worker as worker


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

    assert worker.supervise(["--scheduler-root", str(tmp_path / "state")], popen=popen) == 0
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
    )
    assert code == 2 and killed == [child]
    assert notices[0]["reason"] == "sync_worker_deadline_exceeded"
    log = json.loads((tmp_path / "logs/daily-sync.jsonl").read_text())
    assert log["status"] == "error"


def test_spawn_failure_is_logged_even_when_notification_fails(tmp_path, monkeypatch):
    def failed(*_args, **_kwargs):
        raise OSError("synthetic-secret")

    monkeypatch.setattr(scheduled_sync, "_handle_failure", failed)
    code = worker.supervise(
        ["--scheduler-root", str(tmp_path / "state"), "--log-root", str(tmp_path / "logs")],
        popen=failed,
    )
    assert code == 2
    raw = (tmp_path / "logs/daily-sync.jsonl").read_text()
    assert "synthetic-secret" not in raw
    assert json.loads(raw)["reason"] == "sync_worker_start_or_wait_failed"
    assert json.loads(raw)["notification_status"] == "failure_notification_unavailable"
