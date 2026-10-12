"""Bounded online-first calendar verification with a private, expiring receipt.

Only the BaoStock/Tushare adapters that validate every natural date may supply
this receipt. It is not a weekday calendar and is never inferred from prices.
Unknown provider availability may use a previously verified range; a provider
quality failure cannot. Cache timestamps describe acquisition, not publication.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from ashare_lab.domain.errors import DataQualityError, DataUnavailableError

CALENDAR_LOOKAHEAD_DAYS = 14
CALENDAR_FETCH_DAYS = 35
CALENDAR_TIMEOUT_SECONDS = 8.0
CALENDAR_CACHE_MAX_AGE = timedelta(hours=72)
CALENDAR_CACHE_FILENAME = "verified-preopen-calendar.json"
_CONTRACT = "complete-cn-calendar-v1"
_SOURCES = frozenset({"baostock:query_trade_dates", "tushare:trade_cal:SSE"})
_MAX_CACHE_BYTES = 32_768


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DataQualityError("交易日历知识时间必须包含时区。")
    return value.astimezone(UTC)


def _sessions(values: Iterable[date], start: date, end: date) -> tuple[date, ...]:
    try:
        result = tuple(values)
    except (TypeError, ValueError):
        raise DataQualityError("交易日历日期结构无效。") from None
    if any(type(value) is not date for value in result):
        raise DataQualityError("交易日历包含无效日期。")
    if result != tuple(sorted(set(result))):
        raise DataQualityError("交易日历顺序或唯一性异常。")
    if any(value < start or value > end for value in result):
        raise DataQualityError("交易日历返回请求区间外日期。")
    return result


def _digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_receipt(
    root: Path,
    *,
    source: str,
    start: date,
    end: date,
    sessions: tuple[date, ...],
    fetched_at: datetime,
) -> None:
    if source not in _SOURCES:
        raise DataQualityError("交易日历来源未获核验。")
    opened = set(_sessions(sessions, start, end))
    # These closed dates derive from an adapter's complete-range contract,
    # not from weekday arithmetic or a sparse quote history.
    rows = [
        {"date": day.isoformat(), "is_open": day in opened}
        for offset in range((end - start).days + 1)
        for day in (start + timedelta(days=offset),)
    ]
    payload = {
        "version": 1,
        "contract": _CONTRACT,
        "source": source,
        "market": "CN_A_SSE_SZSE",
        "verified_start": start.isoformat(),
        "verified_end": end.isoformat(),
        "fetched_at": _aware(fetched_at).isoformat(),
        "calendar": rows,
    }
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / CALENDAR_CACHE_FILENAME
    temporary = root / f".{CALENDAR_CACHE_FILENAME}.{uuid4().hex}.tmp"
    try:
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({**payload, "sha256": _digest(payload)}, stream, sort_keys=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_verified_calendar_cache(
    root: Path, *, start: date, end: date, known_at: datetime
) -> tuple[date, ...] | None:
    """Pure read: reject stale/future, partial, corrupt, or unknown-source data."""
    known = _aware(known_at)
    try:
        path = root / CALENDAR_CACHE_FILENAME
        if path.is_symlink() or path.stat().st_size > _MAX_CACHE_BYTES:
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            return None
        payload = {key: value for key, value in document.items() if key != "sha256"}
        if document.get("sha256") != _digest(payload):
            return None
        if (
            type(payload.get("version")) is not int
            or payload["version"] != 1
            or payload.get("contract") != _CONTRACT
            or payload.get("source") not in _SOURCES
            or payload.get("market") != "CN_A_SSE_SZSE"
        ):
            return None
        fetched = _aware(datetime.fromisoformat(payload["fetched_at"]))
        if fetched > known or known - fetched > CALENDAR_CACHE_MAX_AGE:
            return None
        verified_start = date.fromisoformat(payload["verified_start"])
        verified_end = date.fromisoformat(payload["verified_end"])
        if not verified_start <= start <= end <= verified_end:
            return None
        span = (verified_end - verified_start).days + 1
        if not 1 <= span <= CALENDAR_FETCH_DAYS:
            return None
        rows = payload["calendar"]
        if not isinstance(rows, list) or len(rows) != span:
            return None
        opened = []
        for offset, row in enumerate(rows):
            expected = verified_start + timedelta(days=offset)
            if (
                not isinstance(row, dict)
                or set(row) != {"date", "is_open"}
                or row["date"] != expected.isoformat()
                or type(row["is_open"]) is not bool
            ):
                return None
            if row["is_open"] and start <= expected <= end:
                opened.append(expected)
        return tuple(opened)
    except (OSError, ValueError, TypeError, KeyError, DataQualityError):
        return None


def resolve_verified_next_trading_day(
    common_cutoff: date,
    *,
    primary: Callable[[date, date], Iterable[date]],
    backup: Callable[[date, date], Iterable[date]],
    cache_root: Path | None,
    clock: Callable[[], datetime] | None = None,
) -> date:
    """Use bounded complete-range providers before a <=72-hour cached receipt.

    Fetch 35 natural dates, but resolve only within the first 14. This gives a
    previous successful run enough coverage for subsequent daily lookups. The
    caller supplies only the complete-range adapters, never arbitrary history.
    A bad live answer always fails closed even when an older cache is present.
    """
    if type(common_cutoff) is not date:
        raise TypeError("common_cutoff must be a date")
    now = clock or (lambda: datetime.now(UTC))
    initial = _aware(now())
    start = common_cutoff + timedelta(days=1)
    required_end = common_cutoff + timedelta(days=CALENDAR_LOOKAHEAD_DAYS)
    fetched_end = common_cutoff + timedelta(days=CALENDAR_FETCH_DAYS)
    for source, fetch in (
        ("baostock:query_trade_dates", primary),
        ("tushare:trade_cal:SSE", backup),
    ):
        try:
            raw = fetch(start, fetched_end)
        except DataUnavailableError:
            continue
        verified = _sessions(raw, start, fetched_end)
        relevant = tuple(value for value in verified if value <= required_end)
        if not relevant:
            # An empty verified answer does not let an old calendar overrule it.
            raise DataUnavailableError("未能在14日窗口内确认下一个A股交易日。")
        captured = _aware(now())
        if captured < initial:
            raise DataQualityError("交易日历取得时间早于请求知识时间。")
        if cache_root is not None:
            # A failed optional cache write must not invalidate fresh evidence.
            with suppress(OSError):
                _write_receipt(
                    cache_root,
                    source=source,
                    start=start,
                    end=fetched_end,
                    sessions=verified,
                    fetched_at=captured,
                )
        return relevant[0]
    if cache_root is not None:
        cached = read_verified_calendar_cache(
            cache_root, start=start, end=required_end, known_at=_aware(now())
        )
        if cached:
            return cached[0]
    raise DataUnavailableError("联网交易日历不可用，且无覆盖区间的未过期核验缓存。")


def bounded_tushare_calendar(
    start: date,
    end: date,
    *,
    timeout: float = CALENDAR_TIMEOUT_SECONDS,
    runner: Callable = subprocess.run,
) -> tuple[date, ...]:
    """Keep credentials inside a bounded child; expose only sanitized enums."""
    if not 0 < timeout <= CALENDAR_TIMEOUT_SECONDS:
        raise ValueError("calendar timeout must be in (0, 8] seconds")
    try:
        child = runner(
            [sys.executable, "-m", "ashare_lab.cli.preopen_calendar_read"],
            input=json.dumps({"start": start.isoformat(), "end": end.isoformat()}),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise DataUnavailableError("免费备用日历隔离读取未完成。") from None
    if child.returncode and not child.stdout.strip():
        # An interpreter/import crash produced no claimed calendar evidence.
        raise DataUnavailableError("免费备用日历隔离进程未返回结果。")
    if len(child.stdout) > _MAX_CACHE_BYTES:
        raise DataQualityError("免费备用日历隔离响应过大。")
    try:
        payload = json.loads(child.stdout)
    except (ValueError, TypeError):
        raise DataQualityError("免费备用日历隔离响应格式无效。") from None
    if not isinstance(payload, dict):
        raise DataQualityError("免费备用日历隔离响应结构无效。")
    if payload.get("error") == "quality":
        raise DataQualityError("免费备用日历质量未通过，禁止使用缓存掩盖。")
    if child.returncode or payload.get("error") == "unavailable":
        raise DataUnavailableError("免费备用日历暂不可用。")
    try:
        if set(payload) != {"sessions"} or not isinstance(payload["sessions"], list):
            raise ValueError
        values = tuple(date.fromisoformat(value) for value in payload["sessions"])
    except (ValueError, TypeError):
        raise DataQualityError("免费备用日历隔离响应日期无效。") from None
    return _sessions(values, start, end)
