"""Install/run a bounded one-minute personal risk watcher; no trading API."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import sys
import time
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

CN = ZoneInfo("Asia/Shanghai")
LABEL = "com.zerong.asharelab.intraday-risk"


def _root():
    from ashare_lab.bootstrap import application_data_dir

    return application_data_dir() / "scheduler" / "intraday-risk"


def calendar_query(day, provider):
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if provider == "tushare":
            from ashare_lab.cli.evening_digest import _tushare_calendar_backup

            sessions = _tushare_calendar_backup(day - timedelta(days=20), day)
        else:
            from ashare_lab.adapters.baostock_eod import BaoStockEodMarketData

            sessions = BaoStockEodMarketData().fetch_cn_trading_days(day - timedelta(days=20), day)
    return {
        "date": day.isoformat(),
        "open": day in sessions,
        "source": provider,
        "previous_session": max(s for s in sessions if s < day).isoformat(),
    }


def calendar_for_day(root, day):
    from ashare_lab.services.intraday_stop_monitor import write_private_json

    target = root / "calendar.json"
    try:
        previous = json.loads(target.read_text())
        if previous.get("date") == day.isoformat():
            if type(previous.get("open")) is bool:
                return previous["open"]
            if (
                datetime.now(CN) - datetime.fromisoformat(previous["attempted_at"])
            ).total_seconds() < 300:
                return None
    except (OSError, ValueError, KeyError):
        pass
    for provider in ("tushare", "baostock"):
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ashare_lab.cli.intraday_risk",
                    "calendar",
                    "--date",
                    day.isoformat(),
                    "--provider",
                    provider,
                ],
                capture_output=True,
                text=True,
                timeout=9,
                check=False,
            )
            value = json.loads(result.stdout)
            if (
                result.returncode == 0
                and value.get("date") == day.isoformat()
                and type(value.get("open")) is bool
            ):
                write_private_json(target, value)
                return value["open"]
        except (subprocess.TimeoutExpired, OSError, ValueError):
            continue
    write_private_json(
        target,
        {"date": day.isoformat(), "open": None, "attempted_at": datetime.now(CN).isoformat()},
    )
    return None


def send_serverchan(message):
    import httpx

    from ashare_lab.adapters.macos_keychain import load_serverchan_sendkey
    from ashare_lab.adapters.notification_channels import (
        ServerChanNotificationChannel,
        _ServerChanAddressFallbackTransport,
    )

    token = load_serverchan_sendkey()
    if not token:
        return False
    # Only connect failures (no request sent) may try an alternate address.
    # Read/write uncertainty is handled by the bounded outbox retry cadence.
    with (
        httpx.Client(
            timeout=httpx.Timeout(3.0, connect=1.0),
            trust_env=False,
            follow_redirects=False,
            transport=_ServerChanAddressFallbackTransport(),
        ) as client,
        ServerChanNotificationChannel(token, client=client) as channel,
    ):
        return channel.send(message).accepted


def run_once():
    from ashare_lab.adapters.free_intraday_quotes import fetch_intraday_quotes
    from ashare_lab.bootstrap import build_repository
    from ashare_lab.services.daily_update_lock import daily_update_lock
    from ashare_lab.services.intraday_stop_monitor import run_monitor, write_private_json

    root = _root()
    # This is explicitly NOT daily-sync.lock.
    with daily_update_lock(root / "monitor.lock") as acquired:
        if not acquired:
            return {"status": "already_running"}
        event = run_monitor(
            build_repository(),
            root=root,
            now=datetime.now(CN),
            quote_fetcher=fetch_intraday_quotes,
            calendar=lambda day: calendar_for_day(root, day),
            notifier=send_serverchan,
            clock=lambda: datetime.now(CN),
        )
        write_private_json(root / "last-status.json", event)
        return event


def supervise():
    # A stuck SDK/Keychain/HTTP call cannot monopolize later checks or the EOD job.
    from ashare_lab.cli.evening_report_worker import terminate_worker_group
    from ashare_lab.services.intraday_stop_monitor import write_private_json

    child = subprocess.Popen(
        [sys.executable, "-m", "ashare_lab.cli.intraday_risk", "once"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        code = child.wait(timeout=45)
    except subprocess.TimeoutExpired:
        terminate_worker_group(child)
        write_private_json(
            _root() / "last-status.json",
            {
                "status": "monitor_deadline_exceeded",
                "checked_at": datetime.now(CN).isoformat(),
                "delivery_confirmed": False,
            },
        )
        code = 2
    else:
        if code:
            write_private_json(
                _root() / "last-status.json",
                {
                    "status": "monitor_worker_error",
                    "checked_at": datetime.now(CN).isoformat(),
                    "delivery_confirmed": False,
                },
            )
    if code:
        with suppress(subprocess.TimeoutExpired, OSError):
            subprocess.run(
                [sys.executable, "-m", "ashare_lab.cli.intraday_risk", "failure-notice"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
                check=False,
            )
    return code


def worker():
    from ashare_lab.bootstrap import build_repository
    from ashare_lab.services.holding_ledger import get_active_holding_portfolio
    from ashare_lab.services.intraday_stop_monitor import in_monitor_hours, read_config

    while True:
        started = time.monotonic()
        code = supervise()
        if read_config(_root()) is None or not in_monitor_hours(datetime.now(CN)):
            return code
        current = get_active_holding_portfolio(build_repository())
        if current is None or not current.positions:
            return code
        try:
            if (
                json.loads((_root() / "last-status.json").read_text()).get("status")
                == "market_closed"
            ):
                return code
        except (OSError, ValueError):
            pass
        # caffeinate stays alive through the active session, without changing
        # system power settings. Closed lids/shutdown remain a limitation.
        time.sleep(max(1, 60 - (time.monotonic() - started)))


def failure_notice():
    from ashare_lab.bootstrap import build_repository
    from ashare_lab.ports.notifications import NotificationMessage, NotificationUrgency
    from ashare_lab.services.holding_ledger import get_active_holding_portfolio
    from ashare_lab.services.intraday_stop_monitor import read_config, write_private_json

    root, now = _root(), datetime.now(CN)
    current = get_active_holding_portfolio(build_repository())
    if read_config(root) is None or current is None or not current.positions:
        return {"status": "no_holdings_or_disabled"}
    path = root / "worker-failure-notice.json"
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        state = {}
    if state.get("date") != now.date().isoformat():
        state = {"date": now.date().isoformat(), "attempts": 0}
    if state.get("accepted") or state["attempts"] >= 3:
        return {"status": "failure_notice_already_attempted"}
    state["attempts"] += 1
    write_private_json(path, state)
    accepted = send_serverchan(
        NotificationMessage(
            title="A股盘中监控中断提醒",
            urgency=NotificationUrgency.TIME_SENSITIVE,
            body=f"{now:%m-%d %H:%M}｜本机监控任务超时或异常，当前不能保证止损提醒。请直接查看券商行情；程序将继续重试，不会自动下单。",
        )
    )
    state["accepted"] = accepted
    write_private_json(path, state)
    return {"status": "provider_accepted" if accepted else "delivery_unconfirmed"}


def launchagent_document(python_bin, project):
    return {
        "Label": LABEL,
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-i",
            str(python_bin),
            "-m",
            "ashare_lab.cli.intraday_risk",
            "worker",
        ],
        "WorkingDirectory": str(project),
        "StartInterval": 60,
        "RunAtLoad": True,
        "ProcessType": "Background",
        "Umask": 63,
        "StandardOutPath": "/dev/null",
        "StandardErrorPath": "/dev/null",
    }


def test_notification():
    """One explicitly requested activation test per installation, not a fake stop."""
    import hashlib

    from ashare_lab.ports.notifications import NotificationMessage
    from ashare_lab.services.daily_update_lock import daily_update_lock
    from ashare_lab.services.intraday_stop_monitor import read_config, write_private_json

    root = _root()
    with daily_update_lock(root / "activation-test.lock") as acquired:
        config = read_config(root)
        if not acquired or config is None:
            return {"status": "disabled_or_already_running"}
        identity = hashlib.sha256(str(config.get("authorized_at")).encode()).hexdigest()[:16]
        path = root / f"activation-test-{identity}.json"
        if path.exists():
            # A timeout may mean the server accepted the request. Do not blindly resend.
            return json.loads(path.read_text())
        state = {"status": "attempted", "attempted_at": datetime.now(CN).isoformat()}
        write_private_json(path, state)
        accepted = send_serverchan(
            NotificationMessage(
                title="A股盘中预警｜启用测试",
                body="这是一条启用测试，不是买卖建议。\n\n交易时段每60秒检查已确认持仓；成本亏损达到8%即触发风险核验与提醒，结构触线先预警、收盘确认。\n\n需要电脑开机联网；不会自动下单。行情与微信可能延迟，8%不是实际亏损保证上限。",
            )
        )
        state["status"] = "provider_accepted" if accepted else "delivery_unconfirmed"
        write_private_json(path, state)
        return state


def install():
    from ashare_lab.bootstrap import project_root
    from ashare_lab.services.intraday_stop_monitor import METHOD_VERSION, write_private_json

    if sys.platform != "darwin":
        raise ValueError("macOS_launchd_required")
    project = project_root()
    root = _root()
    target = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
    service = f"gui/{os.getuid()}/{LABEL}"
    document = launchagent_document(project / ".venv/bin/python", project)
    if target.exists():
        previous = plistlib.loads(target.read_bytes())
        if previous.get("Label") != LABEL:
            raise ValueError("unexpected_existing_launchagent")
        # Preserve the previous exact target before replacing our own worker.
        backup = root / f"launchagent-backup-{datetime.now(CN):%Y%m%d%H%M%S}.plist"
        backup.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with backup.open("xb") as stream:
            stream.write(target.read_bytes())
        os.chmod(backup, 0o600)
    write_private_json(
        root / "config.json",
        {
            "enabled": True,
            "authorized_channels": ["serverchan"],
            "interval_seconds": 60,
            "method": METHOD_VERSION,
            "authorization": "user_confirmed_intraday_stop_alerts",
            "authorized_at": datetime.now(CN).isoformat(),
            "orders_enabled": False,
            "scope": "user_confirmed_active_holdings_only",
        },
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(plistlib.dumps(document))
    temporary.replace(target)
    if subprocess.run(["launchctl", "print", service], capture_output=True).returncode == 0:
        subprocess.run(["launchctl", "bootout", service], check=True, capture_output=True)
    subprocess.run(["launchctl", "enable", service], check=True, capture_output=True)
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(target)],
        check=True,
        capture_output=True,
    )
    subprocess.run(["launchctl", "print", service], check=True, capture_output=True)
    return {
        "installed": True,
        "interval_seconds": 60,
        "channel": "serverchan",
        "scope": "confirmed_holdings",
        "instant_delivery_guaranteed": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "once",
            "worker",
            "install",
            "calendar",
            "status",
            "failure-notice",
            "test-notification",
        ),
    )
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--date")
    parser.add_argument("--provider", choices=("tushare", "baostock"))
    args = parser.parse_args(argv)
    try:
        if args.command == "worker":
            return worker()
        if args.command == "install":
            if not args.yes:
                raise ValueError("explicit_authorization_required")
            result = install()
        elif args.command == "failure-notice":
            result = failure_notice()
        elif args.command == "test-notification":
            if not args.yes:
                raise ValueError("explicit_test_send_authorization_required")
            result = test_notification()
        elif args.command == "calendar":
            from datetime import date

            result = calendar_query(date.fromisoformat(args.date), args.provider)
        elif args.command == "status":
            result = json.loads((_root() / "last-status.json").read_text())
        else:
            result = run_once()
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception:
        # No raw SDK/HTTP/credential-bearing traceback may reach launchd logs.
        print(json.dumps({"status": "intraday_monitor_error"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
