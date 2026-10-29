from __future__ import annotations

import json
import signal
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ashare_lab.cli import preopen_deadline as deadline
from ashare_lab.services.daily_update_lock import daily_update_lock

NOW = datetime(2026, 9, 22, 8, 58, tzinfo=deadline._SHANGHAI)


def _state(tmp_path, *, trading=True, **overrides):
    value = {
        "date": NOW.date().isoformat(),
        "is_trading_day": trading,
        "checked_at": NOW.replace(hour=8, minute=0).isoformat(),
        **overrides,
    }
    (tmp_path / "preopen-session-state.json").write_text(json.dumps(value))


def _run(tmp_path, *, now=NOW, messages=None, accepted=True):
    messages = [] if messages is None else messages
    return deadline.run_preopen_deadline(
        state_root=tmp_path,
        clock=lambda: now,
        notifier=lambda message: messages.append(message) or accepted,
    )


@pytest.mark.parametrize(
    "now",
    [
        NOW.replace(minute=57),
        NOW.replace(hour=9, minute=0),
        NOW.replace(hour=9, minute=1),
        NOW.replace(day=26),
    ],
)
def test_only_weekday_deadline_slots_may_start(tmp_path, now):
    messages = []
    _, event = _run(tmp_path, now=now, messages=messages)
    assert event["status"] == "noop_outside_preopen_watchdog_window"
    assert messages == []


@pytest.mark.parametrize("minute", [58, 59])
def test_known_trading_day_reports_status_not_fake_recommendations(tmp_path, minute):
    _state(tmp_path)
    messages = []
    code, event = _run(tmp_path, now=NOW.replace(minute=minute), messages=messages)
    assert code == 0
    assert event["status"] == "failure_notice_provider_accepted"
    assert event["delivery_confirmed"] is False
    assert "暂停新买" in messages[0].title
    assert "不是荐股结论" in messages[0].body
    assert messages[0].image_url is None
    assert json.loads((tmp_path / "evening-failure-notice-state.json").read_text()) == {
        "accepted_date": NOW.date().isoformat(),
        "delivery_confirmed": False,
    }


def test_verified_nontrading_day_skips(tmp_path):
    _state(tmp_path, trading=False)
    messages = []
    _, event = _run(tmp_path, messages=messages)
    assert event["status"] == "noop_verified_nontrading_day"
    assert messages == []


@pytest.mark.parametrize(
    "bad_state",
    [
        None,
        {},
        {"date": "2026-09-21"},
        {"is_trading_day": "false"},
        {"checked_at": "2026-09-22T08:00:00"},
        {"checked_at": "2026-09-22T09:00:00+08:00"},
        {"checked_at": "2026-09-21T08:00:00+08:00"},
    ],
)
def test_unknown_or_stale_calendar_uses_conservative_status(tmp_path, bad_state):
    if bad_state is not None:
        _state(tmp_path, **bad_state)
        if not bad_state:
            (tmp_path / "preopen-session-state.json").write_text("{}")
    messages = []
    _run(tmp_path, messages=messages)
    assert "交易日待核验" in messages[0].title
    assert "交易日核验尚未完成" in messages[0].body
    assert "不代表今日一定开市" in messages[0].body


def test_shared_nonblocking_notice_lock_prevents_duplicates_and_ignores_sync_lock(tmp_path):
    _state(tmp_path)
    nested = []

    def notifier(message):
        nested.append(_run(tmp_path)[1]["status"])
        assert "暂停新买" in message.title
        return True

    with daily_update_lock(tmp_path / "daily-sync.lock") as acquired:
        assert acquired
        status = deadline.notify_incomplete_once(
            state_root=tmp_path, now=NOW, clock=lambda: NOW, notifier=notifier
        )
    assert status == "failure_notice_provider_accepted"
    assert nested == ["notice_already_running"]
    assert _run(tmp_path)[1]["status"] == "failure_notice_already_provider_accepted"


@pytest.mark.parametrize(
    "channels,suppressed",
    [
        (["serverchan"], True),
        (["bark"], False),
        ("serverchan", False),
        ([], False),
    ],
)
def test_only_current_serverchan_plan_receipt_suppresses_notice(tmp_path, channels, suppressed):
    (tmp_path / "evening-digest-state.json").write_text(
        json.dumps(
            {
                "plan_for_date": NOW.date().isoformat(),
                "accepted_channels": channels,
            }
        )
    )
    messages = []
    _, event = _run(tmp_path, messages=messages)
    assert (event["status"] == "plan_already_provider_accepted") is suppressed
    assert bool(messages) is not suppressed


def test_stale_plan_receipt_does_not_suppress(tmp_path):
    (tmp_path / "evening-digest-state.json").write_text(
        json.dumps(
            {
                "plan_for_date": "2026-09-21",
                "accepted_channels": ["serverchan"],
            }
        )
    )
    assert _run(tmp_path)[1]["status"] == "failure_notice_provider_accepted"


def test_failure_does_not_mark_accepted_and_can_retry(tmp_path):
    assert _run(tmp_path, accepted=False)[0] == 2
    assert not (tmp_path / "evening-failure-notice-state.json").exists()
    assert _run(tmp_path)[1]["status"] == "failure_notice_provider_accepted"


def test_unknown_provider_outcome_may_retry_but_never_claims_acceptance_or_phone_delivery(tmp_path):
    messages = []
    for minute in (58, 59):
        code, event = _run(
            tmp_path, now=NOW.replace(minute=minute), messages=messages, accepted=False
        )
        assert code == 2
        assert event["status"] == "failure_notice_provider_not_accepted"
        assert event["delivery_confirmed"] is False
        assert not (tmp_path / "evening-failure-notice-state.json").exists()
    assert len(messages) == 2  # Bounded at-least-once, not promised exactly-once.


def test_default_entrypoint_always_logs_watchdog_result_without_launchd_log_argument(
    tmp_path, monkeypatch, capsys
):
    from ashare_lab.cli import evening_report_worker

    event = {
        "job": "ashare-preopen-deadline",
        "status": "failure_notice_deadline_exceeded",
        "delivery_confirmed": False,
    }
    logged = []
    monkeypatch.setattr(deadline, "run_preopen_deadline", lambda **_: (2, event))
    monkeypatch.setattr(
        evening_report_worker, "_safe_log", lambda root, row: logged.append((root, row))
    )
    assert deadline.main(["--state-root", str(tmp_path)]) == 2
    assert logged == [(Path.home() / "Library" / "Logs" / "A股研究助手", event)]
    assert json.loads(capsys.readouterr().out) == event


def test_entrypoint_honors_explicit_log_root(tmp_path, monkeypatch, capsys):
    from ashare_lab.cli import evening_report_worker

    event = {
        "job": "ashare-preopen-deadline",
        "status": "plan_already_provider_accepted",
        "delivery_confirmed": False,
    }
    logged = []
    monkeypatch.setattr(deadline, "run_preopen_deadline", lambda **_: (0, event))
    monkeypatch.setattr(
        evening_report_worker, "_safe_log", lambda root, row: logged.append((root, row))
    )
    assert deadline.main(["--state-root", str(tmp_path), "--log-root", str(tmp_path / "logs")]) == 0
    assert logged == [(tmp_path / "logs", event)]
    capsys.readouterr()


def test_guard_rereads_time_and_receipt_after_setup(tmp_path, monkeypatch):
    _state(tmp_path)
    current = NOW

    def sender(_message, *, before_send):
        nonlocal current
        current = NOW.replace(hour=9, minute=0)
        assert before_send() is False
        return False

    monkeypatch.setattr(deadline, "_send_serverchan_notice", sender)
    status = deadline._notify_locked(
        state_root=tmp_path,
        expected_date=NOW.date().isoformat(),
        clock=lambda: current,
        notifier=None,
    )
    assert status == "noop_outside_preopen_notice_window"
    assert not (tmp_path / "evening-failure-notice-state.json").exists()


def test_guard_rechecks_receipt_immediately_before_sending(tmp_path, monkeypatch):
    def sender(_message, *, before_send):
        (tmp_path / "evening-digest-state.json").write_text(
            json.dumps(
                {
                    "plan_for_date": NOW.date().isoformat(),
                    "accepted_channels": ["serverchan"],
                }
            )
        )
        assert before_send() is False
        return False

    monkeypatch.setattr(deadline, "_send_serverchan_notice", sender)
    status = deadline._notify_locked(
        state_root=tmp_path, expected_date=NOW.date().isoformat(), clock=lambda: NOW, notifier=None
    )
    assert status == "plan_already_provider_accepted"


def test_slow_credential_setup_cannot_send_after_nine(monkeypatch):
    from ashare_lab.adapters import macos_keychain, notification_channels

    allowed = True

    def load_key():
        nonlocal allowed
        allowed = False
        return "SCTsyntheticKeyForOfflineTests"

    monkeypatch.setattr(macos_keychain, "load_serverchan_sendkey", load_key)
    monkeypatch.setattr(
        notification_channels.ServerChanNotificationChannel,
        "send",
        lambda *_: pytest.fail("deadline guard must run after loading credentials"),
    )
    assert deadline._send_serverchan_notice(object(), before_send=lambda: allowed) is False


def test_notice_supervisor_bounds_stalled_child_and_never_forwards_private_output(
    tmp_path, monkeypatch
):
    signals, waits, calls = [], [], []

    class Process:
        pid = 43123
        returncode = None

        def communicate(self, *, timeout):
            calls.append(timeout)
            raise subprocess.TimeoutExpired("SCT-secret-url", timeout)

        def poll(self):
            return self.returncode

        def wait(self, *, timeout):
            waits.append(timeout)
            self.returncode = -signal.SIGKILL

    monkeypatch.setattr(deadline.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(deadline.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    status = deadline.notify_incomplete_once(state_root=tmp_path, now=NOW, clock=lambda: NOW)
    assert status == "failure_notice_deadline_exceeded"
    assert calls == [12]
    assert waits == [1]
    assert signals == [(43123, signal.SIGKILL)]


def test_notice_supervisor_reserves_time_before_nine(tmp_path, monkeypatch):
    budgets = []

    class Process:
        returncode = 0

        def communicate(self, *, timeout):
            budgets.append(timeout)
            return json.dumps({"status": "failure_notice_provider_accepted"}), None

        def poll(self):
            return 0

    monkeypatch.setattr(deadline.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    now = NOW.replace(minute=59, second=56)
    assert (
        deadline.notify_incomplete_once(state_root=tmp_path, now=now, clock=lambda: now)
        == "failure_notice_provider_accepted"
    )
    assert budgets == [3]


def test_stale_now_argument_does_not_override_fresh_clock(tmp_path):
    status = deadline.notify_incomplete_once(
        state_root=tmp_path,
        now=NOW,
        clock=lambda: NOW + timedelta(minutes=2),
        notifier=lambda _: pytest.fail("stale initial timestamp cannot permit late sending"),
    )
    assert status == "noop_outside_preopen_notice_window"


def _offline_sender(monkeypatch, handler):
    """Exercise the real channel/redaction path without any external send."""
    import httpx

    from ashare_lab.adapters import macos_keychain

    monkeypatch.setattr(
        macos_keychain, "load_serverchan_sendkey", lambda: "SCTsyntheticKeyForOfflineTests"
    )
    monkeypatch.setattr(httpx, "HTTPTransport", lambda **_: httpx.MockTransport(handler))


def _locked_send(tmp_path, *, clock=lambda: NOW):
    return deadline._notify_locked(
        state_root=tmp_path,
        expected_date=NOW.date().isoformat(),
        clock=clock,
        notifier=None,
    )


@pytest.mark.parametrize(
    "key_case,status",
    [
        ("missing", "failure_notice_credentials_missing"),
        ("error", "failure_notice_credentials_unavailable"),
        ("invalid", "failure_notice_credentials_invalid"),
    ],
)
def test_credential_failures_have_safe_distinct_categories(tmp_path, monkeypatch, key_case, status):
    from ashare_lab.adapters import macos_keychain

    _offline_sender(monkeypatch, lambda _: pytest.fail("credentials failed; must not send"))

    def load_key():
        if key_case == "error":
            raise RuntimeError("SCT-secret-private-credential-context")
        return None if key_case == "missing" else "invalid-private-key"

    monkeypatch.setattr(macos_keychain, "load_serverchan_sendkey", load_key)
    assert _locked_send(tmp_path) == status
    assert not (tmp_path / "evening-failure-notice-state.json").exists()


@pytest.mark.parametrize(
    "error_kind,status,attempt_count",
    [
        ("ConnectError", "failure_notice_connection_failed", 2),
        ("ConnectTimeout", "failure_notice_connect_timeout", 2),
        ("ReadTimeout", "failure_notice_send_timeout_unknown", 1),
        ("WriteTimeout", "failure_notice_send_timeout_unknown", 1),
        ("ReadError", "failure_notice_request_failed_unknown", 1),
        ("WriteError", "failure_notice_request_failed_unknown", 1),
        ("http_error", "failure_notice_provider_http_error", 1),
        ("rejected", "failure_notice_provider_rejected_or_invalid", 1),
        ("malformed_receipt", "failure_notice_provider_rejected_or_invalid", 1),
        ("invalid_json", "failure_notice_provider_response_invalid", 1),
    ],
)
def test_transport_failure_categories_and_retries_are_bounded_and_redacted(
    tmp_path, monkeypatch, capsys, error_kind, status, attempt_count
):
    import httpx

    calls = []

    def handler(request):
        calls.append(request)
        if error_kind == "http_error":
            return httpx.Response(503, text="private provider response SCT-secret")
        if error_kind == "rejected":
            return httpx.Response(200, json={"code": 40001, "message": "SCT-private quota"})
        if error_kind == "malformed_receipt":
            return httpx.Response(200, json={"code": "0", "message": "private malformed"})
        if error_kind == "invalid_json":
            return httpx.Response(200, text="SCT-private invalid JSON")
        raise getattr(httpx, error_kind)(f"private response and URL {request.url}", request=request)

    _offline_sender(monkeypatch, handler)
    assert _locked_send(tmp_path) == status
    assert len(calls) == attempt_count
    assert not (tmp_path / "evening-failure-notice-state.json").exists()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("first_failure", [False, True])
def test_connection_retry_stops_at_first_acceptance_and_receipt_deduplicates(
    tmp_path, monkeypatch, first_failure
):
    import httpx

    calls = []

    def handler(request):
        calls.append(request)
        if first_failure and len(calls) == 1:
            raise httpx.ConnectError("private connection failure", request=request)
        return httpx.Response(200, json={"code": 0, "message": "SUCCESS"})

    _offline_sender(monkeypatch, handler)
    assert _locked_send(tmp_path) == "failure_notice_provider_accepted"
    assert _locked_send(tmp_path) == "failure_notice_already_provider_accepted"
    assert len(calls) == (2 if first_failure else 1)


@pytest.mark.parametrize("change", ["deadline", "accepted", "nontrading"])
def test_connection_retry_rechecks_deadline_receipt_and_calendar(tmp_path, monkeypatch, change):
    import httpx

    current, calls = NOW, []

    def handler(request):
        nonlocal current
        calls.append(request)
        if change == "deadline":
            current = NOW.replace(hour=9, minute=0)
        elif change == "accepted":
            (tmp_path / "evening-digest-state.json").write_text(
                json.dumps(
                    {"plan_for_date": NOW.date().isoformat(), "accepted_channels": ["serverchan"]}
                )
            )
        else:
            _state(tmp_path, trading=False)
        raise httpx.ConnectError("connection could not be established", request=request)

    _offline_sender(monkeypatch, handler)
    status = _locked_send(tmp_path, clock=lambda: current)
    assert status == {
        "deadline": "noop_outside_preopen_notice_window",
        "accepted": "plan_already_provider_accepted",
        "nontrading": "noop_verified_nontrading_day",
    }[change]
    assert len(calls) == 1


def test_acceptance_persistence_failure_never_immediately_resends(tmp_path, monkeypatch):
    import httpx

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"code": 0, "message": "SUCCESS"})

    def broken_state(*_):
        raise OSError("private path should not enter output")

    _offline_sender(monkeypatch, handler)
    monkeypatch.setattr(deadline, "_write_notice_state", broken_state)
    assert _locked_send(tmp_path) == "failure_notice_acceptance_state_write_failed"
    assert len(calls) == 1


@pytest.mark.parametrize("status", sorted(deadline._SEND_FAILURE_STATUSES))
def test_supervisor_preserves_only_safe_diagnostics_and_marks_failure(tmp_path, monkeypatch, status):
    class Process:
        returncode = 0

        def communicate(self, *, timeout):
            assert timeout <= 12
            return json.dumps({"status": status, "private": "SCT-not-forwarded"}), None

        def poll(self):
            return 0

    monkeypatch.setattr(deadline.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    code, event = deadline.run_preopen_deadline(state_root=tmp_path, clock=lambda: NOW)
    assert code == 2
    assert event == {
        "job": "ashare-preopen-deadline",
        "status": status,
        "delivery_confirmed": False,
    }


def test_unexpected_sender_error_is_not_mislabelled_provider_rejection(tmp_path, monkeypatch):
    def broken_sender(*_, **__):
        raise RuntimeError("private application failure SCT-secret")

    monkeypatch.setattr(deadline, "_send_serverchan_notice", broken_sender)
    assert _locked_send(tmp_path) == "failure_notice_worker_failed"


def test_early_failure_does_not_consume_fixed_deadline_receipt(tmp_path):
    early = NOW.replace(minute=50)
    status = deadline.notify_incomplete_once(
        state_root=tmp_path, now=early, clock=lambda: early,
        notifier=lambda _: pytest.fail("08:50 cannot preempt the fixed deadline"),
    )
    assert status == "noop_outside_preopen_notice_window"
    assert not (tmp_path / "evening-failure-notice-state.json").exists()


def test_deadline_sends_today_partial_observations_under_existing_guards(tmp_path):
    from ashare_lab.services.preopen_partial_report import write_preopen_partial_snapshot

    _state(tmp_path)
    assert write_preopen_partial_snapshot(
        tmp_path, report_date=NOW.date(), generated_at=NOW - timedelta(minutes=2),
        data_cutoff=NOW.date() - timedelta(days=1),
        cycle_label="中期上行｜短线增强", entry_strictness="standard",
        screening_candidates=[{"name": "示例股份", "formation": "回踩企稳", "reason": "财务待核验"}],
        missing_codes=("fundamental_incomplete",),
    )
    messages = []
    code, event = _run(tmp_path, messages=messages)
    assert code == 0 and event["status"] == "failure_notice_provider_accepted"
    assert "示例股份" in messages[0].body and "仅观察，不代表可以买" in messages[0].body
    assert "2026-09-21" in messages[0].body
    _run(tmp_path, now=NOW + timedelta(minutes=1), messages=messages)
    assert len(messages) == 1


def test_snapshot_read_cannot_bypass_before_send_deadline_guard(tmp_path, monkeypatch):
    from ashare_lab.services import preopen_partial_report

    current = NOW
    original = preopen_partial_report.render_preopen_partial_message

    def delayed(*args, **kwargs):
        nonlocal current
        message = original(*args, **kwargs)
        current = NOW.replace(hour=9, minute=0)
        return message

    monkeypatch.setattr(preopen_partial_report, "render_preopen_partial_message", delayed)
    status = deadline._notify_locked(
        state_root=tmp_path, expected_date=NOW.date().isoformat(), clock=lambda: current,
        notifier=lambda _: pytest.fail("no late send after rendering"),
    )
    assert status == "noop_outside_preopen_notice_window"
