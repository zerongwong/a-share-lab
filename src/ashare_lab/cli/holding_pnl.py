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
from datetime import date, datetime, time, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import monotonic
from zoneinfo import ZoneInfo

CN = ZoneInfo("Asia/Shanghai")
TOTAL_BUDGET_SECONDS = 90
_EXIT_RESERVE_SECONDS = 2
_FINAL_COMPUTE_SECONDS = 74
_FALLBACK_SECONDS = 12
_CALENDAR_LOOKBACK_DAYS = 14
_SESSION_CACHE_VERSION = 1
_AKSHARE_CALENDAR_TIMEOUT = 6.0


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


def _verified_calendar_window(values, start: date, end: date) -> tuple[date, ...]:
    """Accept only a complete, ordered provider response ending on the report day."""
    from ashare_lab.domain.errors import DataQualityError

    try:
        sessions = tuple(values)
        if (
            not sessions
            or sessions != tuple(sorted(set(sessions)))
            or any(type(day) is not date or not start <= day <= end for day in sessions)
            or end not in sessions
        ):
            raise DataQualityError("交易日历区间未通过核验。")
        return sessions
    except (TypeError, ValueError):
        raise DataQualityError("交易日历区间未通过核验。") from None


def _read_session_cache(root: Path) -> tuple[date, date, tuple[date, ...]] | None:
    """The cache contains public calendar dates only, never any account details."""
    from ashare_lab.domain.errors import DataQualityError

    path = root / "verified-trading-sessions.json"
    try:
        if path.stat().st_size > 100_000:
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("version") != _SESSION_CACHE_VERSION:
            return None
        start = date.fromisoformat(document["verified_start"])
        end = date.fromisoformat(document["verified_end"])
        if start > end or (end - start).days > 3660:
            return None
        raw = document["sessions"]
        if not isinstance(raw, list) or len(raw) > 3000:
            return None
        sessions = tuple(date.fromisoformat(item) for item in raw)
        return start, end, _verified_calendar_window(sessions, start, end)
    except (OSError, KeyError, TypeError, ValueError, DataQualityError):
        # Corrupt or incompatible public metadata must never create a holding age.
        return None


def _cache_trading_sessions(
    root: Path, start: date, end: date, fresh: tuple[date, ...]
) -> tuple[date, ...]:
    from ashare_lab.domain.errors import DataQualityError
    from ashare_lab.services.intraday_stop_monitor import write_private_json

    old = _read_session_cache(root)
    verified_start, verified_end, sessions = start, end, fresh
    if old is not None:
        old_start, old_end, old_sessions = old
        if old_end <= end and start <= old_end + timedelta(days=1):
            overlap_start = max(start, old_start)
            overlap_end = min(end, old_end)
            if overlap_start <= overlap_end:
                old_overlap = tuple(day for day in old_sessions if overlap_start <= day <= overlap_end)
                new_overlap = tuple(day for day in fresh if overlap_start <= day <= overlap_end)
                if old_overlap != new_overlap:
                    raise DataQualityError("交易日历缓存与最新核验区间冲突。")
            verified_start = min(old_start, start)
            sessions = tuple(sorted(set(old_sessions) | set(fresh)))
        # If a run was missed for longer than the verified lookback, the
        # intervening dates are unknown. Reset instead of inventing coverage.
    write_private_json(
        root / "verified-trading-sessions.json",
        {
            "version": _SESSION_CACHE_VERSION,
            "verified_start": verified_start.isoformat(),
            "verified_end": verified_end.isoformat(),
            "sessions": [day.isoformat() for day in sessions],
        },
    )
    return sessions


def _bounded_calendar_child(action: str, start: date, end: date, *, timeout: float):
    from ashare_lab.domain.errors import DataQualityError, DataUnavailableError

    try:
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "ashare_lab.cli.holding_pnl",
                action,
                "--start",
                start.isoformat(),
                "--end",
                end.isoformat(),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise DataUnavailableError("免费交易日历未取得。") from None
    if child.returncode:
        raise DataUnavailableError("免费交易日历未取得。")
    if len(child.stdout) > 8192:
        raise DataQualityError("免费交易日历响应过大。")
    try:
        payload = json.loads(child.stdout)
        if not isinstance(payload, dict) or not isinstance(payload.get("sessions"), list):
            raise DataQualityError("免费交易日历响应结构异常。")
        sessions = tuple(date.fromisoformat(value) for value in payload["sessions"])
    except (ValueError, TypeError):
        raise DataQualityError("免费交易日历响应结构异常。") from None
    return _verified_calendar_window(sessions, start, end)


def _akshare_trading_sessions(start: date, end: date) -> tuple[date, ...]:
    """Validate the *whole* public Sina calendar before returning a small slice."""
    from ashare_lab.domain.errors import DataUnavailableError

    if start > end or (end - start).days > 30:
        raise DataUnavailableError("交易日历请求区间无效。")
    import akshare as ak
    import pandas as pd

    frame = ak.tool_trade_date_hist_sina()
    if (
        not isinstance(frame, pd.DataFrame)
        or list(frame.columns) != ["trade_date"]
        or len(frame) < 1000
        or len(frame) > 20000
    ):
        raise DataUnavailableError("新浪交易日历结构不可用。")
    values = tuple(frame["trade_date"].to_list())
    if (
        any(type(day) is not date for day in values)
        or values != tuple(sorted(set(values)))
        or values[0] > start
        or values[-1] < end
        or end not in values
    ):
        # A stale future horizon or a non-session report day can use backups.
        raise DataUnavailableError("新浪交易日历未覆盖报告日。")
    return _verified_calendar_window(
        (day for day in values if start <= day <= end), start, end
    )


def _load_trading_sessions(
    start: date, end: date, *, state_root: Path | None = None
) -> tuple[date, ...]:
    """Bounded AKShare→BaoStock→Tushare read plus contiguous local cache.

    The provider sees a fixed recent public window, never a holding entry date.
    A missing run or conflicting overlap cannot silently inflate held-day counts.
    """
    from ashare_lab.adapters.bounded_baostock import BoundedBaoStockEod
    from ashare_lab.domain.errors import DataUnavailableError

    if state_root is not None:
        cached = _read_session_cache(state_root)
        if cached is not None and cached[1] == end:
            return cached[2]
    request_start = (
        end - timedelta(days=_CALENDAR_LOOKBACK_DAYS - 1) if state_root is not None else start
    )
    try:
        verified = _bounded_calendar_child(
            "calendar-range-akshare",
            request_start,
            end,
            timeout=_AKSHARE_CALENDAR_TIMEOUT,
        )
    except DataUnavailableError:
        try:
            verified = _verified_calendar_window(
                BoundedBaoStockEod(timeout=8.0).fetch_cn_trading_days(request_start, end),
                request_start,
                end,
            )
        except DataUnavailableError:
            verified = _bounded_calendar_child(
                "calendar-range", request_start, end, timeout=8.0
            )
    return (
        _cache_trading_sessions(state_root, request_start, end, verified)
        if state_root is not None
        else verified
    )


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
                trading_sessions_loader=lambda start, end: _load_trading_sessions(
                    start, end, state_root=root
                ),
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
        "action",
        nargs="?",
        choices=("run", "once", "failure-notice", "calendar-range", "calendar-range-akshare"),
        default="run",
    )
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args(argv)
    if args.action in {"calendar-range", "calendar-range-akshare"}:
        try:
            start = date.fromisoformat(args.start)
            end = date.fromisoformat(args.end)
            if args.action == "calendar-range-akshare":
                sessions = _akshare_trading_sessions(start, end)
            else:
                from ashare_lab.cli.evening_digest import _tushare_calendar_backup

                sessions = _tushare_calendar_backup(start, end)
            print(json.dumps({"sessions": [day.isoformat() for day in sessions]}))
            return 0
        except Exception:
            # No raw provider exception, request details or credential-bearing text.
            print(json.dumps({"status": "calendar_unavailable"}))
            return 2
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
