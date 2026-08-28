from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ashare_lab.cli import evening_report
from ashare_lab.domain.errors import NotificationDeliveryError
from ashare_lab.ports.notifications import NotificationReceipt
from ashare_lab.services.build_evening_digest import EveningResearchDigest
from ashare_lab.services.daily_update_lock import daily_update_lock

CUTOFF = date(2026, 8, 27)
FRIDAY = date(2026, 8, 28)
SATURDAY = date(2026, 8, 29)
SUNDAY = date(2026, 8, 30)


def _digest() -> EveningResearchDigest:
    return EveningResearchDigest(
        common_cutoff=CUTOFF,
        decision_date=CUTOFF,
        cycle_label="中期下行｜短线修复反弹",
        entry_strictness="defensive",
        max_stock_exposure=0.30,
        minimum_cash_weight=0.70,
        cycle_rule_agreement=0.875,
        periods=(),
    )


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "csmar_root": tmp_path / "csmar",
        "overlay_root": tmp_path / "overlay",
        "reference_root": tmp_path / "reference",
        "state_root": tmp_path / "state",
        "log_root": tmp_path / "logs",
    }


def _accepted_summary(
    *channels: str,
    failed: tuple[str, ...] = (),
    receipt_ids: tuple[str, ...] = (),
) -> evening_report.EveningNotificationSummary:
    accepted = channels or ("serverchan",)
    return evening_report.EveningNotificationSummary(
        configured_channels=tuple(dict.fromkeys((*accepted, *failed))),
        accepted_channels=accepted,
        failed_channels=failed,
        provider_receipt_ids=receipt_ids,
    )


def _rejected_summary(*channels: str) -> evening_report.EveningNotificationSummary:
    rejected = channels or ("serverchan", "bark")
    return evening_report.EveningNotificationSummary(
        configured_channels=rejected,
        failed_channels=rejected,
    )


def test_first_provider_acceptance_writes_state_and_second_run_is_noop(tmp_path: Path) -> None:
    messages = []
    builds = []

    first = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **kwargs: builds.append(kwargs) or _digest(),
        _notifier=lambda message: (
            messages.append(message)
            or _accepted_summary(
                "serverchan",
                "bark",
                receipt_ids=("serverchan:0123456789abcdef",),
            )
        ),
    )
    second = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _build_digest=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("NOOP must not rebuild")
        ),
        _notifier=lambda _message: (_ for _ in ()).throw(AssertionError("NOOP must not resend")),
    )

    assert first.exit_code == evening_report.EXIT_OK
    assert first.event["status"] == "provider_accepted"
    assert first.event["delivery_confirmed"] is False
    assert first.event["accepted_channels"] == ["serverchan", "bark"]
    assert first.event["plan_for_date"] == "2026-08-28"
    assert second.exit_code == evening_report.EXIT_OK
    assert second.event["status"] == "noop_no_new_trading_day"
    assert len(builds) == len(messages) == 1
    assert messages[0].title == "A股日报｜2026-08-28（周五）计划"
    assert "共同截止日：2026-08-27" in messages[0].body
    assert "计划适用日：2026-08-28（周五）" in messages[0].body
    assert messages[0].compact_body is not None
    assert "数据2026-08-27" in messages[0].compact_body
    assert len(messages[0].compact_body.encode("utf-8")) <= 2_400
    state = json.loads(
        (tmp_path / "state" / "evening-digest-state.json").read_text(encoding="utf-8")
    )
    assert set(state) == {
        "accepted_channels",
        "delivery_confirmed",
        "last_provider_accepted_common_cutoff",
        "plan_for_date",
        "provider_accepted_at",
        "provider_receipt_ids",
    }
    assert state["last_provider_accepted_common_cutoff"] == "2026-08-27"
    assert state["plan_for_date"] == "2026-08-28"
    assert state["delivery_confirmed"] is False
    assert state["provider_receipt_ids"] == ["serverchan:0123456789abcdef"]
    log_path = tmp_path / "logs" / "evening-report.jsonl"
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [entry["status"] for entry in entries] == [
        "provider_accepted",
        "noop_no_new_trading_day",
    ]
    assert entries[0]["exit_code"] == evening_report.EXIT_OK
    assert entries[0]["raw_data_exposed"] is False
    assert entries[0]["orders_enabled"] is False
    assert entries[0]["plan_for_date"] == "2026-08-28"
    assert entries[1]["plan_for_date"] == "2026-08-28"
    assert "A股六周期研究日报" not in log_path.read_text(encoding="utf-8")
    assert log_path.stat().st_mode & 0o777 == 0o600
    assert log_path.parent.stat().st_mode & 0o777 == 0o700


def test_provider_failure_does_not_mark_cutoff_as_accepted(tmp_path: Path) -> None:
    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _notifier=lambda _message: _rejected_summary(),
    )

    assert outcome.exit_code == evening_report.EXIT_ERROR
    assert outcome.event["reason"] == "notification_providers_not_accepted"
    assert outcome.event["delivery_confirmed"] is False
    assert not (tmp_path / "state" / "evening-digest-state.json").exists()


def test_no_configured_provider_does_not_write_deduplication_state(tmp_path: Path) -> None:
    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _notifier=lambda _message: evening_report.EveningNotificationSummary(),
    )

    assert outcome.exit_code == evening_report.EXIT_ERROR
    assert outcome.event["reason"] == "notification_channels_not_configured"
    assert outcome.event["delivery_confirmed"] is False
    assert not (tmp_path / "state" / "evening-digest-state.json").exists()


def test_official_infoway_calendar_resolves_first_session_and_closes_adapter() -> None:
    observed: dict[str, object] = {}

    class CalendarProvider:
        def __init__(self, credential: str) -> None:
            observed["credential_was_passed"] = credential == "keychain-only-secret"

        def fetch_cn_trading_days(self, start: date, end: date) -> tuple[date, ...]:
            observed["range"] = (start, end)
            return (FRIDAY, date(2026, 8, 31))

        def close(self) -> None:
            observed["closed"] = True

    result = evening_report.resolve_next_infoway_trading_day(
        CUTOFF,
        _api_key_loader=lambda: "keychain-only-secret",
        _provider_factory=CalendarProvider,
    )

    assert result == FRIDAY
    assert observed == {
        "credential_was_passed": True,
        "range": (date(2026, 8, 28), date(2026, 9, 10)),
        "closed": True,
    }


def test_unverified_next_trading_day_fails_closed_before_notification(tmp_path: Path) -> None:
    secret = "calendar-provider-must-not-leak-this"
    notified: list[object] = []

    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: (_ for _ in ()).throw(RuntimeError(secret)),
        _build_digest=lambda **_kwargs: _digest(),
        _notifier=lambda message: notified.append(message) or _accepted_summary(),
    )

    assert outcome.exit_code == evening_report.EXIT_ERROR
    assert outcome.event == {
        "job": "ashare-evening-digest",
        "status": "error",
        "reason": "next_trading_day_not_verified",
        "common_cutoff": "2026-08-27",
        "plan_for_date": None,
    }
    assert notified == []
    assert not (tmp_path / "state" / "evening-digest-state.json").exists()
    log_text = (tmp_path / "logs" / "evening-report.jsonl").read_text(encoding="utf-8")
    assert secret not in log_text
    assert json.loads(log_text)["plan_for_date"] is None


def test_default_notifier_attempts_both_providers_and_keeps_independent_results(
    monkeypatch,
) -> None:
    from ashare_lab.cli import evening_digest

    closed: list[str] = []

    class Channel:
        def __init__(self, name: str, *, fails: bool = False) -> None:
            self.channel_name = name
            self.fails = fails

        def send(self, _message):
            if self.fails:
                raise NotificationDeliveryError("sanitized")
            return NotificationReceipt(
                channel=self.channel_name,
                accepted=True,
                provider_status="provider_accepted",
                provider_receipt_id="0123456789abcdef",
            )

        def close(self) -> None:
            closed.append(self.channel_name)

    monkeypatch.setattr(evening_digest, "load_serverchan_sendkey", lambda: "server-key")
    monkeypatch.setattr(evening_digest, "load_bark_device_key", lambda: "bark-key")
    monkeypatch.setattr(
        evening_digest,
        "ServerChanNotificationChannel",
        lambda _key: Channel("serverchan", fails=True),
    )
    monkeypatch.setattr(
        evening_digest,
        "BarkNotificationChannel",
        lambda _key: Channel("bark"),
    )

    summary = evening_report.send_evening_digest(
        evening_digest.NotificationMessage("行动单", "正文")
    )

    assert summary.configured_channels == ("serverchan", "bark")
    assert summary.accepted_channels == ("bark",)
    assert summary.failed_channels == ("serverchan",)
    assert summary.provider_receipt_ids == ("bark:0123456789abcdef",)
    assert summary.any_accepted is True
    assert closed == ["serverchan", "bark"]


def test_default_notifier_with_no_configured_channel_is_not_accepted(monkeypatch) -> None:
    from ashare_lab.cli import evening_digest

    monkeypatch.setattr(evening_digest, "load_serverchan_sendkey", lambda: None)
    monkeypatch.setattr(evening_digest, "load_bark_device_key", lambda: None)

    summary = evening_report.send_evening_digest(
        evening_digest.NotificationMessage("行动单", "正文")
    )

    assert summary == evening_report.EveningNotificationSummary()
    assert summary.any_accepted is False


def test_default_notifier_keeps_full_serverchan_body_and_uses_compact_bark_body(
    monkeypatch,
) -> None:
    from ashare_lab.cli import evening_digest

    seen: dict[str, str] = {}

    class Channel:
        def __init__(self, name: str) -> None:
            self.channel_name = name

        def send(self, message):
            seen[self.channel_name] = message.body
            return NotificationReceipt(
                channel=self.channel_name,
                accepted=True,
                provider_status="provider_accepted",
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(evening_digest, "load_serverchan_sendkey", lambda: "server-key")
    monkeypatch.setattr(evening_digest, "load_bark_device_key", lambda: "bark-key")
    monkeypatch.setattr(
        evening_digest,
        "ServerChanNotificationChannel",
        lambda _key: Channel("serverchan"),
    )
    monkeypatch.setattr(
        evening_digest,
        "BarkNotificationChannel",
        lambda _key: Channel("bark"),
    )
    full = "完整六周期正文" * 300
    compact = "六周期紧凑行动摘要"

    summary = evening_report.send_evening_digest(
        evening_digest.NotificationMessage(full[:8], full, compact_body=compact)
    )

    assert summary.accepted_channels == ("serverchan", "bark")
    assert seen == {"serverchan": full, "bark": compact}


def test_oversize_body_without_compact_fails_closed_for_bark_only(monkeypatch) -> None:
    from ashare_lab.cli import evening_digest

    sent: list[str] = []

    class Channel:
        def __init__(self, name: str) -> None:
            self.channel_name = name

        def send(self, _message):
            sent.append(self.channel_name)
            return NotificationReceipt(
                channel=self.channel_name,
                accepted=True,
                provider_status="provider_accepted",
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(evening_digest, "load_serverchan_sendkey", lambda: "server-key")
    monkeypatch.setattr(evening_digest, "load_bark_device_key", lambda: "bark-key")
    monkeypatch.setattr(
        evening_digest,
        "ServerChanNotificationChannel",
        lambda _key: Channel("serverchan"),
    )
    monkeypatch.setattr(
        evening_digest,
        "BarkNotificationChannel",
        lambda _key: Channel("bark"),
    )

    summary = evening_report.send_evening_digest(
        evening_digest.NotificationMessage("行动单", "中" * 801)
    )

    assert sent == ["serverchan"]
    assert summary.accepted_channels == ("serverchan",)
    assert summary.failed_channels == ("bark",)


def test_friday_and_saturday_are_hard_noop_before_state_or_data_reads(
    tmp_path: Path, monkeypatch
) -> None:
    from ashare_lab.cli import evening_digest

    paths = _paths(tmp_path)
    state_path = paths["state_root"] / "evening-digest-state.json"
    state_path.parent.mkdir(parents=True)
    original_state = {
        "last_sent_common_cutoff": "2026-08-27",
        "updated_at": "2026-08-27T21:00:00+08:00",
    }
    state_path.write_text(json.dumps(original_state), encoding="utf-8")
    monkeypatch.setattr(
        evening_digest,
        "_read_state",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("weekend NOOP must not read delivery state")
        ),
    )

    for blocked_date in (FRIDAY, SATURDAY):
        outcome = evening_report.run_evening_digest(
            **paths,
            decision_date=blocked_date,
            _latest_cutoff=lambda _root: (_ for _ in ()).throw(
                AssertionError("weekend NOOP must not inspect market data")
            ),
            _build_digest=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("weekend NOOP must not build")
            ),
            _notifier=lambda _message: (_ for _ in ()).throw(
                AssertionError("weekend NOOP must not send")
            ),
        )

        assert outcome.exit_code == evening_report.EXIT_OK
        assert outcome.event == {
            "job": "ashare-evening-digest",
            "status": "noop_weekend_send_window_closed",
        }
        assert json.loads(state_path.read_text(encoding="utf-8")) == original_state

    entries = [
        json.loads(line)
        for line in (paths["log_root"] / "evening-report.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [entry["status"] for entry in entries] == [
        "noop_weekend_send_window_closed",
        "noop_weekend_send_window_closed",
    ]
    assert all(entry["exit_code"] == evening_report.EXIT_OK for entry in entries)


def test_sunday_with_no_new_cutoff_uses_normal_deduplication(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    state_path = paths["state_root"] / "evening-digest-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "last_sent_common_cutoff": CUTOFF.isoformat(),
                "updated_at": "2026-08-27T21:00:00+08:00",
            }
        ),
        encoding="utf-8",
    )

    outcome = evening_report.run_evening_digest(
        **paths,
        decision_date=SUNDAY,
        _latest_cutoff=lambda _root: CUTOFF,
        _build_digest=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("same cutoff must not rebuild on Sunday")
        ),
        _notifier=lambda _message: (_ for _ in ()).throw(
            AssertionError("same cutoff must not resend on Sunday")
        ),
    )

    assert outcome.exit_code == evening_report.EXIT_OK
    assert outcome.event["status"] == "noop_no_new_trading_day"


def test_busy_lock_is_logged_without_building_or_sending(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    lock_path = paths["state_root"] / "daily-sync.lock"

    with daily_update_lock(lock_path) as acquired:
        assert acquired is True
        outcome = evening_report.run_evening_digest(
            **paths,
            decision_date=CUTOFF,
            _latest_cutoff=lambda _root: (_ for _ in ()).throw(
                AssertionError("busy run must not inspect data")
            ),
            _build_digest=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("busy run must not build")
            ),
            _notifier=lambda _message: (_ for _ in ()).throw(
                AssertionError("busy run must not send")
            ),
        )

    assert outcome.exit_code == evening_report.EXIT_RETRY
    assert outcome.event["status"] == "already_running"
    logged = json.loads((tmp_path / "logs" / "evening-report.jsonl").read_text(encoding="utf-8"))
    assert logged["status"] == "already_running"
    assert logged["reason"] == "daily_data_lock_busy"


def test_unexpected_failure_never_copies_exception_or_secret(tmp_path: Path) -> None:
    secret = "SCT-do-not-print-this-secret"
    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _build_digest=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
        _notifier=lambda _message: _accepted_summary(),
    )

    assert outcome.exit_code == evening_report.EXIT_ERROR
    assert secret not in json.dumps(outcome.event, ensure_ascii=False)
    log_text = (tmp_path / "logs" / "evening-report.jsonl").read_text(encoding="utf-8")
    assert secret not in log_text
    log_event = json.loads(log_text)
    assert log_event["status"] == "error"
    assert log_event["reason"] == "unexpected_evening_digest_error"
    assert set(log_event) <= {
        "accepted_channels",
        "common_cutoff",
        "configured_channels",
        "delivery_confirmed",
        "exit_code",
        "failed_channels",
        "job",
        "logged_at",
        "orders_enabled",
        "period_count",
        "provider_receipt_ids",
        "raw_data_exposed",
        "reason",
        "status",
    }


def test_cli_has_stable_module_and_no_credential_arguments() -> None:
    help_text = evening_report.build_parser().format_help().lower()
    assert "sendkey" not in help_text
    assert "token" not in help_text
    assert "--log-root" in help_text
    assert callable(evening_report.main)


def test_private_log_rotates_and_keeps_at_most_five_backups(tmp_path: Path, monkeypatch) -> None:
    from ashare_lab.cli import evening_digest

    monkeypatch.setattr(evening_digest, "_MAX_LOG_BYTES", 300)
    log_path = tmp_path / "logs" / "evening-report.jsonl"
    outcome = evening_report.EveningDigestOutcome(
        evening_report.EXIT_ERROR,
        {
            "job": "ashare-evening-digest",
            "status": "error",
            "reason": "stable_failure_code",
            "unsafe_exception": "SCT-do-not-log-this",
        },
    )

    for _ in range(30):
        evening_digest._write_log_event(log_path, outcome)

    backups = sorted(log_path.parent.glob("evening-report.jsonl.*"))
    assert 1 <= len(backups) <= 5
    for candidate in (log_path, *backups):
        assert candidate.stat().st_mode & 0o777 == 0o600
        assert "SCT-do-not-log-this" not in candidate.read_text(encoding="utf-8")


def test_log_failure_does_not_turn_rejected_delivery_into_success(
    tmp_path: Path, monkeypatch
) -> None:
    from ashare_lab.cli import evening_digest

    monkeypatch.setattr(
        evening_digest,
        "_write_log_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("private log unavailable")),
    )
    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _notifier=lambda _message: _rejected_summary(),
    )

    assert outcome.exit_code == evening_report.EXIT_ERROR
    assert outcome.event["reason"] == "notification_providers_not_accepted"
    assert not (tmp_path / "state" / "evening-digest-state.json").exists()
