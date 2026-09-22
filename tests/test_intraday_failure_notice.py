from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.cli import intraday_risk
from ashare_lab.services.holding_ledger import (
    HoldingPositionInput,
    clear_active_holdings,
    replace_active_holdings,
)
from ashare_lab.services.intraday_stop_monitor import write_private_json

CN = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 9, 22, 10, 0, tzinfo=CN)


def register(repository, *, effective_at=NOW - timedelta(days=1)):
    return replace_active_holdings(
        repository,
        [
            HoldingPositionInput(
                symbol="600919",
                name="测试股票",
                entry_date=date(2026, 9, 21),
                cost_price=10,
                stock_sleeve_weight=1.0,
            )
        ],
        holding_weeks=4,
        effective_at=effective_at,
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repository = SQLiteRepository(
        tmp_path / "research.db", Path(__file__).resolve().parents[1] / "migrations"
    )
    repository.initialize()
    register(repository)
    root = tmp_path / "monitor"
    write_private_json(
        root / "config.json", {"enabled": True, "authorized_channels": ["serverchan"]}
    )
    monkeypatch.setattr(intraday_risk, "_root", lambda: root)
    sent = []

    def send(message, *, before_send):
        if not before_send():
            return False
        sent.append(message)
        return True

    monkeypatch.setattr(intraday_risk, "send_serverchan", send)
    return repository, root, sent


def run(setup, *, now=NOW, clock=None):
    return intraday_risk.failure_notice(
        _repository=setup[0], _clock=clock or (lambda: now)
    )


@pytest.mark.parametrize(
    "now",
    [
        NOW.replace(hour=8),
        NOW.replace(hour=12),
        NOW.replace(hour=16),
        NOW.replace(hour=23),
        NOW.replace(day=26),
    ],
)
def test_failure_notice_silent_outside_market_monitor_window(setup, now):
    assert run(setup, now=now)["status"] == "outside_session"
    assert not setup[2]
    assert not (setup[1] / "worker-failure-notice.json").exists()


def test_known_same_day_holiday_suppresses_notice(setup):
    write_private_json(
        setup[1] / "calendar.json", {"date": NOW.date().isoformat(), "open": False}
    )
    assert run(setup)["status"] == "market_closed"
    assert not setup[2]


@pytest.mark.parametrize("calendar_state", [None, {"open": None}, {"open": False, "date": "2026-09-21"}])
def test_unknown_calendar_allows_only_nontrading_health_notice_once(setup, calendar_state):
    if calendar_state is not None:
        write_private_json(setup[1] / "calendar.json", calendar_state)
    assert run(setup)["status"] == "provider_accepted"
    assert run(setup)["status"] == "failure_notice_already_attempted"
    assert len(setup[2]) == 1
    assert "不是买卖信号" in setup[2][0].body
    assert "不代表今日一定开市" in setup[2][0].body
    assert "600919" not in setup[2][0].body


def test_future_holding_does_not_authorize_notice(setup):
    register(setup[0], effective_at=NOW + timedelta(minutes=1))
    assert run(setup)["status"] == "holding_not_yet_effective"
    assert not setup[2]


@pytest.mark.parametrize("change", ["disabled", "cleared", "version", "closed", "outside"])
def test_last_transport_check_cancels_obsolete_notice(setup, monkeypatch, change):
    clock_value = [NOW]

    def changed_transport(message, *, before_send):
        if change == "disabled":
            write_private_json(setup[1] / "config.json", {"enabled": False})
        elif change == "cleared":
            clear_active_holdings(setup[0], effective_at=NOW)
        elif change == "version":
            register(setup[0], effective_at=NOW)
        elif change == "closed":
            write_private_json(
                setup[1] / "calendar.json", {"date": NOW.date().isoformat(), "open": False}
            )
        else:
            clock_value[0] = NOW.replace(hour=16)
        if before_send():
            setup[2].append(message)
            return True
        return False

    monkeypatch.setattr(intraday_risk, "send_serverchan", changed_transport)
    assert run(setup, clock=lambda: clock_value[0])["status"] in {
        "no_holdings_or_disabled", "holding_or_session_changed", "market_closed", "outside_session"
    }
    assert not setup[2]


def test_real_sender_rechecks_after_keychain_and_client_setup(monkeypatch):
    from ashare_lab.adapters import macos_keychain, notification_channels
    from ashare_lab.ports.notifications import NotificationMessage

    events = []
    monkeypatch.setattr(macos_keychain, "load_serverchan_sendkey", lambda: events.append("key") or "test")

    class Channel:
        def __init__(self, *_a, **_kw):
            events.append("client")

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            pass

        def send(self, _message):
            pytest.fail("canceled health notice must not reach provider")

    monkeypatch.setattr(notification_channels, "ServerChanNotificationChannel", Channel)
    assert intraday_risk.send_serverchan(
        NotificationMessage(title="test", body="test"),
        before_send=lambda: events.append("guard") or False,
    ) is False
    assert events == ["key", "client", "guard"]
