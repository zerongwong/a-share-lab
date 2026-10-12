from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.services.preopen_trading_calendar import (
    CALENDAR_CACHE_FILENAME,
    CALENDAR_CACHE_MAX_AGE,
    CALENDAR_FETCH_DAYS,
    bounded_tushare_calendar,
    read_verified_calendar_cache,
    resolve_verified_next_trading_day,
)

CUTOFF = date(2026, 10, 9)
NEXT = date(2026, 10, 12)
NOW = datetime(2026, 10, 12, 0, 20, tzinfo=UTC)
SESSIONS = (NEXT, date(2026, 10, 13), date(2026, 10, 14))


def unavailable(*_):
    raise DataUnavailableError("synthetic unavailable")


def bad_quality(*_):
    raise DataQualityError("synthetic invalid coverage")


def resolve(root, *, primary=lambda *_: SESSIONS, backup=unavailable, now=NOW, cutoff=CUTOFF):
    return resolve_verified_next_trading_day(
        cutoff, primary=primary, backup=backup, cache_root=root, clock=lambda: now
    )


def read(root, *, now=NOW, start=CUTOFF + timedelta(days=1), end=CUTOFF + timedelta(days=14)):
    return read_verified_calendar_cache(root, start=start, end=end, known_at=now)


def rewrite(root, mutate, *, rehash=True):
    path = root / CALENDAR_CACHE_FILENAME
    payload = json.loads(path.read_text())
    mutate(payload)
    if rehash:
        body = {key: value for key, value in payload.items() if key != "sha256"}
        payload["sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    path.write_text(json.dumps(payload))


def test_online_first_writes_complete_private_range_then_outage_uses_it(tmp_path):
    calls = []

    def fresh(start, end):
        calls.append((start, end))
        return SESSIONS

    assert resolve(tmp_path, primary=fresh) == NEXT
    assert calls == [(CUTOFF + timedelta(days=1), CUTOFF + timedelta(days=CALENDAR_FETCH_DAYS))]
    receipt = json.loads((tmp_path / CALENDAR_CACHE_FILENAME).read_text())
    assert receipt["source"] == "baostock:query_trade_dates"
    assert len(receipt["calendar"]) == CALENDAR_FETCH_DAYS
    assert receipt["calendar"][0] == {"date": "2026-10-10", "is_open": False}
    assert stat.S_IMODE((tmp_path / CALENDAR_CACHE_FILENAME).stat().st_mode) == 0o600
    assert read(tmp_path) == SESSIONS
    assert resolve(tmp_path, primary=unavailable, now=NOW + timedelta(minutes=10)) == NEXT


def test_prior_cutoff_cache_covers_later_required_range(tmp_path):
    assert resolve(tmp_path) == NEXT
    # Cached full range also covers the next decision's entire 14-day lookup.
    assert resolve(tmp_path, primary=unavailable, cutoff=NEXT, now=NOW + timedelta(days=1)) == date(
        2026, 10, 13
    )


def test_friday_premarket_cache_survives_monday_0820_but_not_after_72_hours(tmp_path):
    friday = datetime(2026, 10, 9, 0, 20, tzinfo=UTC)
    friday_session = date(2026, 10, 9)
    assert resolve(
        tmp_path,
        primary=lambda *_: (friday_session, *SESSIONS),
        cutoff=date(2026, 10, 8),
        now=friday,
    ) == friday_session
    # Friday and Monday 08:20 Asia/Shanghai are precisely 72 hours apart.
    assert resolve(tmp_path, primary=unavailable, now=NOW) == NEXT
    with pytest.raises(DataUnavailableError, match="无覆盖区间"):
        resolve(tmp_path, primary=unavailable, now=NOW + timedelta(microseconds=1))


def test_malformed_baostock_wire_cannot_be_hidden_by_valid_cache(tmp_path):
    from ashare_lab.adapters.bounded_baostock import BoundedBaoStockEod

    resolve(tmp_path)
    provider = BoundedBaoStockEod(
        runner=lambda *_, **__: SimpleNamespace(returncode=0, stdout="not-json-private")
    )
    with pytest.raises(DataQualityError):
        resolve(
            tmp_path,
            primary=provider.fetch_cn_trading_days,
            backup=lambda *_: pytest.fail("malformed evidence cannot trigger backup"),
        )


def test_backup_receipt_retains_true_source(tmp_path):
    assert resolve(tmp_path, primary=unavailable, backup=lambda *_: SESSIONS) == NEXT
    receipt = json.loads((tmp_path / CALENDAR_CACHE_FILENAME).read_text())
    assert receipt["source"] == "tushare:trade_cal:SSE"


def test_live_data_overrules_older_calendar_and_is_attempted_before_cache(tmp_path):
    assert resolve(tmp_path) == NEXT
    assert (
        resolve(tmp_path, primary=lambda *_: SESSIONS[1:], now=NOW + timedelta(minutes=1))
        == SESSIONS[1]
    )
    assert read(tmp_path, now=NOW + timedelta(minutes=1)) == SESSIONS[1:]


@pytest.mark.parametrize("primary,backup", [(bad_quality, unavailable), (unavailable, bad_quality)])
def test_provider_quality_failure_never_falls_back_to_old_cache(tmp_path, primary, backup):
    resolve(tmp_path)
    with pytest.raises(DataQualityError):
        resolve(tmp_path, primary=primary, backup=backup)


@pytest.mark.parametrize(
    "values",
    [
        ("2026-10-12",),
        (datetime(2026, 10, 12),),
        (NEXT, NEXT),
        (date(2026, 10, 13), NEXT),
        (CUTOFF,),
        (CUTOFF + timedelta(days=36),),
    ],
)
def test_malformed_fresh_sessions_never_become_cache_or_use_backup(tmp_path, values):
    with pytest.raises(DataQualityError):
        resolve(
            tmp_path,
            primary=lambda *_: values,
            backup=lambda *_: pytest.fail("quality cannot switch source"),
        )
    assert not (tmp_path / CALENDAR_CACHE_FILENAME).exists()


def test_empty_verified_answer_does_not_get_overruled_by_old_cache(tmp_path):
    resolve(tmp_path)
    with pytest.raises(DataUnavailableError, match="14日"):
        resolve(
            tmp_path, primary=lambda *_: (), backup=lambda *_: pytest.fail("empty live calendar")
        )


def test_no_weekday_guess_when_sources_and_cache_are_missing(tmp_path):
    with pytest.raises(DataUnavailableError, match="无覆盖区间"):
        resolve(tmp_path, primary=unavailable)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(source="untrusted:calendar"),
        lambda p: p.update(market="US"),
        lambda p: p.update(version=True),
        lambda p: p.update(contract="weekdays-only"),
        lambda p: p.update(fetched_at=(NOW + timedelta(seconds=1)).isoformat()),
        lambda p: p.update(fetched_at=NOW.replace(tzinfo=None).isoformat()),
        lambda p: p.update(
            fetched_at=(NOW - CALENDAR_CACHE_MAX_AGE - timedelta(seconds=1)).isoformat()
        ),
        lambda p: p["calendar"].pop(0),
        lambda p: p["calendar"][2].update(is_open=1),
        lambda p: p["calendar"][2].update(date="2026-10-13"),
        lambda p: p.update(verified_start="2026-10-11"),
        lambda p: p.update(verified_end="2026-10-20"),
    ],
)
def test_invalid_receipt_is_never_used_even_with_recomputed_checksum(tmp_path, mutate):
    resolve(tmp_path)
    rewrite(tmp_path, mutate)
    assert read(tmp_path) is None
    with pytest.raises(DataUnavailableError):
        resolve(tmp_path, primary=unavailable)


def test_corrupt_checksum_future_knowledge_expiry_and_uncovered_end(tmp_path):
    resolve(tmp_path)
    assert read(tmp_path, now=NOW - timedelta(microseconds=1)) is None
    assert read(tmp_path, now=NOW + CALENDAR_CACHE_MAX_AGE) == SESSIONS
    assert read(tmp_path, now=NOW + CALENDAR_CACHE_MAX_AGE + timedelta(microseconds=1)) is None
    assert read(tmp_path, end=CUTOFF + timedelta(days=36)) is None
    rewrite(tmp_path, lambda p: p["calendar"][2].update(is_open=False), rehash=False)
    assert read(tmp_path) is None


def test_symlink_and_oversized_receipt_rejected(tmp_path):
    resolve(tmp_path)
    original = tmp_path / CALENDAR_CACHE_FILENAME
    saved = tmp_path / "saved.json"
    original.rename(saved)
    original.symlink_to(saved)
    assert read(tmp_path) is None
    original.unlink()
    original.write_text(" " * 32_769)
    assert read(tmp_path) is None


def test_read_cache_does_not_create_missing_directory(tmp_path):
    root = tmp_path / "not-created"
    assert read(root) is None
    assert not root.exists()


def test_cache_write_error_does_not_discard_valid_live_evidence(tmp_path):
    root = tmp_path / "file-not-dir"
    root.write_text("original")
    assert resolve(root) == NEXT
    assert root.read_text() == "original"


def test_clock_moving_backwards_cannot_create_receipt(tmp_path):
    instants = iter((NOW, NOW - timedelta(seconds=1)))
    with pytest.raises(DataQualityError, match="知识时间"):
        resolve_verified_next_trading_day(
            CUTOFF,
            primary=lambda *_: SESSIONS,
            backup=unavailable,
            cache_root=tmp_path,
            clock=lambda: next(instants),
        )
    assert not (tmp_path / CALENDAR_CACHE_FILENAME).exists()


def test_bounded_backup_has_8_second_limit_and_safe_protocol():
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"sessions": [NEXT.isoformat()]}))

    assert bounded_tushare_calendar(CUTOFF, NEXT, runner=runner) == (NEXT,)
    assert calls[0][1]["timeout"] == 8.0
    assert calls[0][1]["capture_output"] is True
    assert "ashare_lab.cli.preopen_calendar_read" in calls[0][0]
    assert set(json.loads(calls[0][1]["input"])) == {"start", "end"}


def test_bounded_backup_timeout_is_unavailable_without_exception_echo():
    def runner(*_, **__):
        raise subprocess.TimeoutExpired("sensitive-private-command", 8)

    with pytest.raises(DataUnavailableError) as caught:
        bounded_tushare_calendar(CUTOFF, NEXT, runner=runner)
    assert "sensitive" not in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"error": "quality"},
        {},
        [],
        {"sessions": ["bad"]},
        {"sessions": ["2026-10-08"]},
    ],
)
def test_backup_quality_protocol_is_not_unavailability(payload):
    def runner(*_, **__):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload))

    with pytest.raises(DataQualityError):
        bounded_tushare_calendar(CUTOFF, NEXT, runner=runner)


def test_child_preserves_quality_class_and_never_prints_raw_error(monkeypatch, capsys):
    from io import StringIO

    from ashare_lab.cli import evening_digest, preopen_calendar_read

    monkeypatch.setattr(
        preopen_calendar_read.sys, "stdin", StringIO('{"start":"2026-10-10","end":"2026-11-13"}')
    )

    def fail(*_):
        raise DataQualityError("private provider request body")

    monkeypatch.setattr(evening_digest, "_tushare_calendar_backup", fail)
    assert preopen_calendar_read.main() == 2
    assert json.loads(capsys.readouterr().out) == {"error": "quality"}


def test_backup_child_start_crash_without_evidence_is_unavailable():
    def runner(*_, **__):
        return SimpleNamespace(returncode=1, stdout="")

    with pytest.raises(DataUnavailableError):
        bounded_tushare_calendar(CUTOFF, NEXT, runner=runner)
