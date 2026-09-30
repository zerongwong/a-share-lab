from __future__ import annotations

import json
import signal
import subprocess
from datetime import date, datetime
from types import SimpleNamespace

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


def test_calendar_range_backup_is_bounded_and_returns_only_verified_dates(monkeypatch):
    from ashare_lab.adapters.bounded_baostock import BoundedBaoStockEod
    from ashare_lab.domain.errors import DataUnavailableError

    def unavailable(_self, _start, _end):
        raise DataUnavailableError("synthetic")

    calls = []

    def child(args, **kwargs):
        calls.append((args, kwargs))
        if "calendar-range-akshare" in args:
            return SimpleNamespace(returncode=2, stdout=json.dumps({"status": "calendar_unavailable"}))
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"sessions": ["2026-09-21", "2026-09-22"]})
        )

    monkeypatch.setattr(BoundedBaoStockEod, "fetch_cn_trading_days", unavailable)
    monkeypatch.setattr(cli.subprocess, "run", child)
    assert cli._load_trading_sessions(date(2026, 9, 21), date(2026, 9, 22)) == (
        date(2026, 9, 21),
        date(2026, 9, 22),
    )
    assert "calendar-range-akshare" in calls[0][0]
    assert calls[0][1]["timeout"] <= 8.0
    assert "calendar-range" in calls[1][0]
    assert calls[1][1]["timeout"] == 8.0


def test_akshare_calendar_is_first_bounded_source(monkeypatch):
    from ashare_lab.adapters.bounded_baostock import BoundedBaoStockEod

    calls = []

    def child(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"sessions": ["2026-09-21", "2026-09-22"]}),
        )

    monkeypatch.setattr(cli.subprocess, "run", child)
    monkeypatch.setattr(
        BoundedBaoStockEod,
        "fetch_cn_trading_days",
        lambda *_: pytest.fail("successful AKShare calendar must not call backups"),
    )
    assert cli._load_trading_sessions(date(2026, 9, 21), date(2026, 9, 22)) == (
        date(2026, 9, 21),
        date(2026, 9, 22),
    )
    assert len(calls) == 1
    assert "calendar-range-akshare" in calls[0][0]
    assert calls[0][1]["timeout"] <= 8.0


def test_akshare_child_filters_and_validates_full_public_calendar(monkeypatch, capsys):
    import akshare as ak
    import pandas as pd

    values = pd.date_range("1990-01-01", "2026-09-30", freq="B").date
    monkeypatch.setattr(
        ak,
        "tool_trade_date_hist_sina",
        lambda: pd.DataFrame({"trade_date": values}),
    )
    assert (
        cli.main(
            ["calendar-range-akshare", "--start", "2026-09-28", "--end", "2026-09-30"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"sessions": ["2026-09-28", "2026-09-29", "2026-09-30"]}


def test_akshare_stale_horizon_fails_safely_for_fallback(monkeypatch, capsys):
    import akshare as ak
    import pandas as pd

    values = pd.date_range("1990-01-01", "2026-09-29", freq="B").date
    monkeypatch.setattr(
        ak,
        "tool_trade_date_hist_sina",
        lambda: pd.DataFrame({"trade_date": values}),
    )
    assert (
        cli.main(
            ["calendar-range-akshare", "--start", "2026-09-28", "--end", "2026-09-30"]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out) == {"status": "calendar_unavailable"}


def test_malformed_successful_calendar_response_does_not_switch_sources(monkeypatch):
    from ashare_lab.adapters.bounded_baostock import BoundedBaoStockEod
    from ashare_lab.domain.errors import DataQualityError

    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps({"sessions": ["not-a-date"]})
        ),
    )
    monkeypatch.setattr(
        BoundedBaoStockEod,
        "fetch_cn_trading_days",
        lambda *_: pytest.fail("quality error must not be hidden by a provider switch"),
    )
    with pytest.raises(DataQualityError):
        cli._load_trading_sessions(date(2026, 9, 21), date(2026, 9, 22))


def test_calendar_quality_error_does_not_silently_switch_source(monkeypatch):
    from ashare_lab.adapters.bounded_baostock import BoundedBaoStockEod
    from ashare_lab.domain.errors import DataQualityError, DataUnavailableError

    def bad_quality(_self, _start, _end):
        raise DataQualityError("synthetic")

    monkeypatch.setattr(
        cli,
        "_bounded_calendar_child",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DataUnavailableError("synthetic")),
    )
    monkeypatch.setattr(BoundedBaoStockEod, "fetch_cn_trading_days", bad_quality)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("quality failure must not trigger a source switch"),
    )
    with pytest.raises(DataQualityError):
        cli._load_trading_sessions(date(2026, 9, 21), date(2026, 9, 22))


def test_calendar_range_child_keeps_provider_errors_out_of_output(monkeypatch, capsys):
    from ashare_lab.cli import evening_digest

    def bad_provider(*_args):
        raise RuntimeError("synthetic-private-provider-error")

    monkeypatch.setattr(evening_digest, "_tushare_calendar_backup", bad_provider)
    assert cli.main(["calendar-range", "--start", "2026-09-21", "--end", "2026-09-22"]) == 2
    assert "synthetic-private-provider-error" not in capsys.readouterr().out


def test_public_recent_calendar_request_and_cache_extend_without_holding_dates(
    monkeypatch, tmp_path
):
    from ashare_lab.adapters.bounded_baostock import BoundedBaoStockEod
    from ashare_lab.domain.errors import DataUnavailableError

    monkeypatch.setattr(
        cli,
        "_bounded_calendar_child",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DataUnavailableError("synthetic")),
    )

    calls = []
    observations = [
        (date(2026, 9, 21), date(2026, 9, 22)),
        (date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23)),
        (date(2026, 9, 23), date(2026, 10, 6)),
    ]

    def calendar(_self, start, end):
        calls.append((start, end))
        return observations.pop(0)

    monkeypatch.setattr(BoundedBaoStockEod, "fetch_cn_trading_days", calendar)
    old_holding_date = date(2024, 1, 1)
    cli._load_trading_sessions(old_holding_date, date(2026, 9, 22), state_root=tmp_path)
    cli._load_trading_sessions(old_holding_date, date(2026, 9, 23), state_root=tmp_path)
    sessions = cli._load_trading_sessions(
        old_holding_date, date(2026, 10, 6), state_root=tmp_path
    )
    assert calls == [
        (date(2026, 9, 9), date(2026, 9, 22)),
        (date(2026, 9, 10), date(2026, 9, 23)),
        (date(2026, 9, 23), date(2026, 10, 6)),
    ]
    assert sessions == (
        date(2026, 9, 21),
        date(2026, 9, 22),
        date(2026, 9, 23),
        date(2026, 10, 6),
    )
    path = tmp_path / "verified-trading-sessions.json"
    cache = json.loads(path.read_text())
    assert cache["verified_start"] == "2026-09-09"
    assert cache["verified_end"] == "2026-10-06"
    assert "2024-01-01" not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600


def test_calendar_cache_gap_resets_instead_of_counting_unverified_days(monkeypatch, tmp_path):
    from ashare_lab.adapters.bounded_baostock import BoundedBaoStockEod
    from ashare_lab.domain.errors import DataUnavailableError

    monkeypatch.setattr(
        cli,
        "_bounded_calendar_child",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DataUnavailableError("synthetic")),
    )

    observations = [(date(2026, 9, 22),), (date(2026, 10, 22),)]
    monkeypatch.setattr(
        BoundedBaoStockEod,
        "fetch_cn_trading_days",
        lambda *_: observations.pop(0),
    )
    cli._load_trading_sessions(date(1990, 1, 1), date(2026, 9, 22), state_root=tmp_path)
    sessions = cli._load_trading_sessions(
        date(1990, 1, 1), date(2026, 10, 22), state_root=tmp_path
    )
    assert sessions == (date(2026, 10, 22),)
    cache = json.loads((tmp_path / "verified-trading-sessions.json").read_text())
    assert cache["verified_start"] == "2026-10-09"


def test_calendar_cache_reuses_same_day_verified_response(monkeypatch, tmp_path):
    from ashare_lab.adapters.bounded_baostock import BoundedBaoStockEod
    from ashare_lab.domain.errors import DataUnavailableError

    monkeypatch.setattr(
        cli,
        "_bounded_calendar_child",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DataUnavailableError("synthetic")),
    )

    calls = []

    def calendar(_self, *_dates):
        calls.append(1)
        return (date(2026, 9, 21), date(2026, 9, 22))

    monkeypatch.setattr(BoundedBaoStockEod, "fetch_cn_trading_days", calendar)
    first = cli._load_trading_sessions(
        date(1990, 1, 1), date(2026, 9, 22), state_root=tmp_path
    )
    second = cli._load_trading_sessions(
        date(1990, 1, 1), date(2026, 9, 22), state_root=tmp_path
    )
    assert first == second == (date(2026, 9, 21), date(2026, 9, 22))
    assert calls == [1]


def test_calendar_cache_conflicting_overlap_fails_closed(monkeypatch, tmp_path):
    from ashare_lab.adapters.bounded_baostock import BoundedBaoStockEod
    from ashare_lab.domain.errors import DataQualityError, DataUnavailableError

    monkeypatch.setattr(
        cli,
        "_bounded_calendar_child",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DataUnavailableError("synthetic")),
    )

    observations = [
        (date(2026, 9, 21), date(2026, 9, 22)),
        (date(2026, 9, 23),),
    ]
    monkeypatch.setattr(
        BoundedBaoStockEod,
        "fetch_cn_trading_days",
        lambda *_: observations.pop(0),
    )
    cli._load_trading_sessions(date(1990, 1, 1), date(2026, 9, 22), state_root=tmp_path)
    path = tmp_path / "verified-trading-sessions.json"
    before = path.read_text()
    with pytest.raises(DataQualityError):
        cli._load_trading_sessions(date(1990, 1, 1), date(2026, 9, 23), state_root=tmp_path)
    assert path.read_text() == before


def test_calendar_cache_rejects_invalid_provider_window(monkeypatch, tmp_path):
    from ashare_lab.adapters.bounded_baostock import BoundedBaoStockEod
    from ashare_lab.domain.errors import DataQualityError, DataUnavailableError

    monkeypatch.setattr(
        cli,
        "_bounded_calendar_child",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DataUnavailableError("synthetic")),
    )

    monkeypatch.setattr(
        BoundedBaoStockEod,
        "fetch_cn_trading_days",
        lambda *_: (date(2026, 9, 21),),
    )
    with pytest.raises(DataQualityError):
        cli._load_trading_sessions(date(1990, 1, 1), date(2026, 9, 22), state_root=tmp_path)
    assert not (tmp_path / "verified-trading-sessions.json").exists()


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
