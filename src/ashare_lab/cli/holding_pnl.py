"""Bounded, separately authorized post-close holding floating-P&L report."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import monotonic
from zoneinfo import ZoneInfo

CN = ZoneInfo("Asia/Shanghai")
TOTAL_BUDGET_SECONDS = 90
_EXIT_RESERVE_SECONDS = 2
_FINAL_COMPUTE_SECONDS = 74
_FALLBACK_SECONDS = 12


def _root() -> Path:
    from ashare_lab.bootstrap import application_data_dir

    return application_data_dir() / "scheduler" / "holding-pnl"


def _record(root: Path, event: dict, log_root: Path | None = None) -> None:
    from ashare_lab.services.intraday_stop_monitor import write_private_json

    # Only allow fixed status metadata. No symbols, prices, quantities, costs,
    # private notification text or provider exception is ever logged.
    safe = {
        key: value
        for key, value in event.items()
        if key in {"job", "method", "checked_at", "status", "delivery_confirmed", "orders_enabled"}
    }
    try:
        write_private_json(root / "last-status.json", safe)
        log_root = log_root or root / "logs"
        log_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = log_root / "holding-pnl.jsonl"
        handler = RotatingFileHandler(path, maxBytes=262_144, backupCount=3, encoding="utf-8")
        try:
            handler.emit(
                logging.LogRecord(
                    "ashare.holding-pnl",
                    logging.INFO,
                    "",
                    0,
                    json.dumps(safe, ensure_ascii=False),
                    (),
                    None,
                )
            )
        finally:
            handler.close()
        os.chmod(path, 0o600)
    except Exception:
        return


def send_serverchan(message) -> bool:
    import httpx

    from ashare_lab.adapters.macos_keychain import load_serverchan_sendkey
    from ashare_lab.adapters.notification_channels import ServerChanNotificationChannel

    token = load_serverchan_sendkey()
    if not token:
        return False
    with (
        httpx.Client(
            timeout=httpx.Timeout(3.0, connect=2.0),
            trust_env=False,
            follow_redirects=False,
            transport=httpx.HTTPTransport(retries=0, trust_env=False),
        ) as client,
        ServerChanNotificationChannel(token, client=client) as channel,
    ):
        # Recheck after potentially slow credential/client setup; the adapter
        # will also recheck the guard immediately before disclosing P&L.
        if message.holding_authorization_guard is None or not message.holding_authorization_guard(
            "serverchan"
        ):
            return False
        receipt = channel.send(message)
    return receipt.accepted is True and receipt.provider_status == "provider_accepted"


def run_once(root: Path, *, log_root: Path | None = None, fallback: bool = False) -> dict:
    try:
        from ashare_lab.bootstrap import build_repository
        from ashare_lab.services.holding_pnl_report import (
            run_holding_pnl,
            send_unverified_holding_pnl,
        )

        if fallback:
            event = send_unverified_holding_pnl(
                build_repository(),
                root=root,
                notifier=send_serverchan,
                clock=lambda: datetime.now(CN),
            )
        else:
            from ashare_lab.adapters.free_intraday_quotes import fetch_intraday_quotes
            from ashare_lab.cli.intraday_risk import calendar_for_day
            from ashare_lab.services.company_action_evidence import (
                refresh_and_load_company_action_clearances,
            )

            event = run_holding_pnl(
                build_repository(),
                root=root,
                now=datetime.now(CN),
                quote_fetcher=fetch_intraday_quotes,
                calendar=lambda day: calendar_for_day(root, day),
                notifier=send_serverchan,
                company_action_clearance_loader=refresh_and_load_company_action_clearances,
                clock=lambda: datetime.now(CN),
            )
    except Exception:
        event = {
            "status": "holding_pnl_worker_error",
            "checked_at": datetime.now(CN).isoformat(),
            "delivery_confirmed": False,
            "orders_enabled": False,
        }
    _record(root, event, log_root)
    return event


def supervise(
    *,
    root: Path,
    log_root: Path | None = None,
    clock: Callable[[], datetime] | None = None,
    popen: Callable = subprocess.Popen,
    kill_group: Callable = os.killpg,
) -> tuple[int, dict]:
    clock = clock or (lambda: datetime.now(CN))
    now = clock().astimezone(CN)
    event = {
        "job": "ashare-holding-pnl",
        "checked_at": now.isoformat(),
        "delivery_confirmed": False,
        "orders_enabled": False,
    }
    if now.weekday() >= 5 or not time(15, 30) <= now.time() < time(15, 59):
        return 0, {**event, "status": "outside_report_window"}
    end = now.replace(hour=15, minute=59, second=0, microsecond=0)
    started = monotonic()
    final_slot = now.time() >= time(15, 55)
    process = None
    completed_normally = False
    status, code = "holding_pnl_worker_error", 2
    try:
        args = [
            sys.executable,
            "-m",
            "ashare_lab.cli.holding_pnl",
            "once",
            "--state-root",
            str(root),
        ]
        if log_root is not None:
            args.extend(["--log-root", str(log_root)])
        process = popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        budget = min(
            TOTAL_BUDGET_SECONDS - _EXIT_RESERVE_SECONDS - (monotonic() - started),
            (end - clock().astimezone(CN)).total_seconds()
            - _EXIT_RESERVE_SECONDS
            - (_FALLBACK_SECONDS + 2 if final_slot else 0),
            _FINAL_COMPUTE_SECONDS if final_slot else TOTAL_BUDGET_SECONDS,
        )
        if budget <= 0:
            raise subprocess.TimeoutExpired("holding-pnl", 0)
        returned = process.wait(timeout=budget)
        if returned == 0:
            completed_normally = True
            # Do not overwrite the child's useful accepted/pending/no-holdings status.
            return 0, {**event, "status": "holding_pnl_worker_completed"}
        status = "holding_pnl_worker_error"
    except subprocess.TimeoutExpired:
        status = "holding_pnl_deadline_exceeded"
    except Exception:
        status = "holding_pnl_worker_start_or_wait_failed"
    finally:
        if process is not None and not completed_normally:
            # The leader may already have failed while an SDK descendant still
            # holds report.lock. Close the entire dedicated group either way.
            with suppress(ProcessLookupError):
                kill_group(process.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=_EXIT_RESERVE_SECONDS)
    result = {**event, "status": status}
    _record(root, result, log_root)
    fresh = clock().astimezone(CN)
    if fresh.date() == now.date() and time(15, 55) <= fresh.time() < time(15, 59):
        fallback_budget = min(
            _FALLBACK_SECONDS,
            TOTAL_BUDGET_SECONDS - (monotonic() - started) - _EXIT_RESERVE_SECONDS,
            (end - fresh).total_seconds() - _EXIT_RESERVE_SECONDS,
        )
        if fallback_budget > 0:
            fallback_process = None
            try:
                args = [
                    sys.executable,
                    "-m",
                    "ashare_lab.cli.holding_pnl",
                    "failure-notice",
                    "--state-root",
                    str(root),
                ]
                if log_root is not None:
                    args.extend(["--log-root", str(log_root)])
                fallback_process = popen(
                    args,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                fallback_process.wait(
                    timeout=min(
                        fallback_budget,
                        max(
                            0.01,
                            (end - clock().astimezone(CN)).total_seconds() - _EXIT_RESERVE_SECONDS,
                        ),
                        max(
                            0.01,
                            TOTAL_BUDGET_SECONDS - (monotonic() - started) - _EXIT_RESERVE_SECONDS,
                        ),
                    )
                )
            except Exception:
                pass
            finally:
                if fallback_process is not None and fallback_process.poll() != 0:
                    with suppress(ProcessLookupError):
                        kill_group(fallback_process.pid, signal.SIGKILL)
                    with suppress(subprocess.TimeoutExpired):
                        fallback_process.wait(timeout=_EXIT_RESERVE_SECONDS)
    return code, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Private post-close floating holding P&L report")
    parser.add_argument(
        "action", nargs="?", choices=("run", "once", "failure-notice"), default="run"
    )
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    args = parser.parse_args(argv)
    root = (args.state_root or _root()).expanduser()
    if args.action in {"once", "failure-notice"}:
        event = run_once(root, log_root=args.log_root, fallback=args.action == "failure-notice")
        code = 2 if event["status"] in {"holding_pnl_worker_error", "provider_not_accepted"} else 0
    else:
        code, event = supervise(root=root, log_root=args.log_root)
    print(json.dumps(event, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
