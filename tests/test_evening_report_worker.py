from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime

import pytest

from ashare_lab.cli import evening_report_worker as worker
from ashare_lab.services.daily_update_lock import daily_update_lock


def _options(tmp_path):
    return {
        "argv": ["--state-root", str(tmp_path / "state"), "--log-root", str(tmp_path / "logs")],
        "_clock": lambda: datetime(2026, 9, 8, 21, tzinfo=worker._SHANGHAI),
    }


class _Process:
    pid = 45678

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.timeouts = []

    def wait(self, timeout):
        self.timeouts.append(timeout)
        result = next(self.outcomes)
        if result == "timeout":
            raise subprocess.TimeoutExpired("private-worker-argument", timeout)
        return result


def test_automatic_worker_is_isolated_and_bounded_without_forwarding_output(tmp_path):
    process = _Process([0])
    commands = []

    def popen(command, **kwargs):
        commands.append((command, kwargs))
        return process

    code, event = worker.supervise_evening_report(
        **_options(tmp_path),
        _popen=popen,
        _notifier=lambda _message: pytest.fail("successful worker must not send fallback"),
    )
    assert code == 0
    assert event["status"] == "worker_completed"
    assert process.timeouts == [720]
    command, options = commands[0]
    assert command[:3] == [sys.executable, "-m", "ashare_lab.cli.evening_digest"]
    assert options == {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "start_new_session": True,
    }


def test_timeout_terminates_only_owned_group_and_deduplicates_failure_notice(tmp_path):
    messages, signals = [], []

    def notifier(message):
        messages.append(message)
        return True

    for _ in range(2):
        process = _Process(["timeout", "timeout", -signal.SIGKILL])
        code, event = worker.supervise_evening_report(
            **_options(tmp_path),
            _popen=lambda *_args, _process=process, **_kwargs: _process,
            _kill_group=lambda pid, sig: signals.append((pid, sig)),
            _notifier=notifier,
        )
        assert code == 2
        assert event["reason"] == "evening_worker_deadline_exceeded"
        assert process.timeouts == [720, 5, 5]
    assert signals == [(45678, signal.SIGTERM), (45678, signal.SIGKILL)] * 2
    assert len(messages) == 1
    assert "暂停新买" in messages[0].title
    assert messages[0].image_url is None
    assert not (tmp_path / "state" / "evening-digest-state.json").exists()
    assert json.loads((tmp_path / "state" / "evening-failure-notice-state.json").read_text()) == {
        "accepted_date": "2026-09-08",
        "delivery_confirmed": False,
    }
    logged = (tmp_path / "logs" / "evening-report.jsonl").read_text()
    assert "evening_worker_deadline_exceeded" in logged
    assert "private-worker-argument" not in logged


def test_leader_exit_still_terminates_remaining_descendants():
    process, signals = _Process([-signal.SIGTERM, -signal.SIGTERM]), []
    worker.terminate_worker_group(process, kill_group=lambda pid, sig: signals.append((pid, sig)))
    assert signals == [(45678, signal.SIGTERM), (45678, signal.SIGKILL)]


def test_provider_rejection_does_not_suppress_later_timeout_notice(tmp_path):
    calls = []
    now = _options(tmp_path)["_clock"]()
    for accepted in (False, True):
        status = worker.notify_incomplete_once(
            state_root=tmp_path / "state",
            now=now,
            notifier=lambda message, _accepted=accepted: calls.append(message) or _accepted,
        )
    assert len(calls) == 2
    assert status == "failure_notice_provider_accepted"


def test_known_plan_receipt_prevents_misleading_failure_notice(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    original = {"plan_for_date": "2026-09-09", "accepted_channels": ["serverchan"]}
    (state / "evening-digest-state.json").write_text(json.dumps(original))
    status = worker.notify_incomplete_once(
        state_root=state,
        now=_options(tmp_path)["_clock"](),
        notifier=lambda _message: pytest.fail("already accepted plan must not be contradicted"),
    )
    assert status == "plan_already_provider_accepted"
    assert json.loads((state / "evening-digest-state.json").read_text()) == original


@pytest.mark.parametrize("code", [1, 2, -signal.SIGSEGV])
def test_failed_worker_is_observable_even_if_it_cannot_import_analytics(tmp_path, code):
    messages = []
    result, event = worker.supervise_evening_report(
        **_options(tmp_path),
        _popen=lambda *_args, **_kwargs: _Process([code]),
        _notifier=lambda message: messages.append(message) or True,
    )
    assert result == (1 if code == 1 else 2)
    assert event["reason"] == "evening_worker_not_completed"
    assert len(messages) == 1


def test_spawn_failure_is_logged_without_unsafe_exception_details(tmp_path):
    def popen(*_args, **_kwargs):
        raise OSError("SCT-secret/provider-private-url")

    result, event = worker.supervise_evening_report(**_options(tmp_path), _popen=popen)
    assert result == 2
    assert event["reason"] == "evening_worker_start_or_wait_failed"
    assert "SCT-secret" not in (tmp_path / "logs" / "evening-report.jsonl").read_text()


@pytest.mark.parametrize("date_hour", [(5, 21), (8, 20), (8, 22)])
def test_unscheduled_invocation_never_starts_worker_or_notifies(tmp_path, date_hour):
    options = _options(tmp_path)
    day, hour = date_hour
    options["_clock"] = lambda: datetime(2026, 9, day, hour, tzinfo=worker._SHANGHAI)
    result, event = worker.supervise_evening_report(
        **options,
        _popen=lambda *_args, **_kwargs: pytest.fail("outside window must not spawn"),
        _notifier=lambda _message: pytest.fail("outside window must not notify"),
    )
    assert result == 0
    assert event["status"] == "noop_outside_evening_window"


def test_timeout_releases_real_child_advisory_lock(tmp_path):
    """Kill a synthetic owned worker; no market data or provider is involved."""
    lock_path = tmp_path / "synthetic.lock"
    ready_path = tmp_path / "ready"
    code = (
        "import fcntl, os, sys, time; "
        "fd=os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o600); "
        "fcntl.flock(fd, fcntl.LOCK_EX); "
        "open(sys.argv[2], 'w').close(); time.sleep(60)"
    )
    processes = []

    def popen(_command, **kwargs):
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(lock_path), str(ready_path)], **kwargs
        )
        processes.append(process)
        return process

    try:
        result, event = worker.supervise_evening_report(
            **_options(tmp_path),
            timeout_seconds=0.5,
            _popen=popen,
            _notifier=lambda _message: True,
        )
        assert ready_path.exists(), "synthetic worker must have actually acquired its lock"
        assert result == 2
        assert event["reason"] == "evening_worker_deadline_exceeded"
        assert processes[0].poll() is not None
        with daily_update_lock(lock_path) as acquired:
            assert acquired is True
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def test_regular_stable_entry_uses_supervisor(monkeypatch, capsys):
    from ashare_lab.cli import evening_report

    calls = []
    monkeypatch.setattr(
        worker,
        "supervise_evening_report",
        lambda args: calls.append(args) or (2, {"reason": "synthetic-timeout"}),
    )
    assert evening_report.main(["--log-root", "/synthetic/logs"]) == 2
    assert calls == [["--log-root", "/synthetic/logs"]]
    assert json.loads(capsys.readouterr().out)["reason"] == "synthetic-timeout"
