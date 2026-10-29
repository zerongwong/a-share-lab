from __future__ import annotations

import fcntl
import json
import os
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from ashare_lab.services import preopen_partial_report as partial

NOW = datetime(2026, 10, 29, 8, 58, tzinfo=ZoneInfo("Asia/Shanghai"))
CUTOFF = date(2026, 10, 28)


def _write(root, **changes):
    values = {
        "report_date": NOW.date(), "generated_at": NOW - timedelta(minutes=5),
        "data_cutoff": CUTOFF, "cycle_label": "中期下行｜短线修复反弹",
        "entry_strictness": "defensive", "method_version": "method-v5",
        "screening_candidates": [{"name": "示例股份", "symbol": "600999",
                                  "formation": "回踩企稳", "reason": "财务与重大风险证据不足",
                                  "cost": 123, "shares": 100, "account_weight": 0.2}],
        "missing_codes": ("fundamental_incomplete",),
    }
    values.update(changes)
    return partial.write_preopen_partial_snapshot(root, **values)


def test_today_snapshot_is_bounded_observation_not_buy_plan(tmp_path):
    assert _write(tmp_path)
    message = partial.render_preopen_partial_message(tmp_path, now=NOW, session=True)
    assert "2026-10-29" in message.body and "2026-10-28" in message.body
    assert "示例股份｜回踩企稳｜基本面证据仍需核验" in message.body
    assert "防守" in message.body and "仅观察，不代表可以买" in message.body
    assert "未核验不等于通过" in message.body
    raw = (tmp_path / partial.SNAPSHOT_NAME).read_text()
    for private in ("cost", "shares", "account_weight", "symbol", "600999"):
        assert private not in raw and private not in message.body
    assert (tmp_path / partial.SNAPSHOT_NAME).stat().st_mode & 0o777 == 0o600


def test_no_snapshot_has_dated_unknowns_and_does_not_claim_no_opportunities(tmp_path):
    message = partial.render_preopen_partial_message(tmp_path, now=NOW, session=None)
    assert "2026-10-29" in message.body and "数据截至：未知" in message.body
    assert "不等于没有合适股票" in message.body and "交易日核验尚未完成" in message.body


@pytest.mark.parametrize("mutation", [
    {"plan_for_date": "2026-10-28"}, {"generated_at": "2026-10-28T08:50:00+08:00"},
    {"generated_at": "2026-10-29T09:30:00+08:00"}, {"generated_at": "2026-10-29T08:50:00"},
    {"data_cutoff": "2026-10-29"}, {"data_cutoff": "2026-10-30"},
    {"data_cutoff": "not-a-date"}, {"scope": "holding_journal"}, {"schema": "unknown"},
])
def test_stale_future_and_invalid_snapshot_are_not_current_recommendations(tmp_path, mutation):
    assert _write(tmp_path)
    path = tmp_path / partial.SNAPSHOT_NAME
    data = json.loads(path.read_text())
    data.update(mutation)
    path.write_text(json.dumps(data))
    assert partial.read_preopen_partial_snapshot(tmp_path, now=NOW) is None
    assert "示例股份" not in partial.render_preopen_partial_message(tmp_path, now=NOW, session=True).body


@pytest.mark.parametrize("content", [b"broken", b"[]", b"null", b"\xff", b"a" * 20000])
def test_malformed_snapshot_degrades_without_exposing_payload(tmp_path, content):
    (tmp_path / partial.SNAPSHOT_NAME).write_bytes(content)
    message = partial.render_preopen_partial_message(tmp_path, now=NOW, session=True)
    assert "数据截至：未知" in message.body
    assert "broken" not in message.body


def test_fifo_and_symlink_cannot_block_or_redirect_the_deadline_reader(tmp_path):
    path = tmp_path / partial.SNAPSHOT_NAME
    os.mkfifo(path)
    assert partial.read_preopen_partial_snapshot(tmp_path, now=NOW) is None
    path.unlink()
    other = tmp_path / "private.json"
    other.write_text("private")
    path.symlink_to(other)
    assert partial.read_preopen_partial_snapshot(tmp_path, now=NOW) is None


def test_untrusted_text_is_never_copied_into_external_message(tmp_path):
    assert _write(tmp_path, screening_candidates=[
        {"name": "[外链](https://private)", "formation": "回踩企稳"},
        {"name": "SCTSuperSecret", "formation": "确认突破"},
        {"name": "股份SECRET", "formation": "确认突破"},
        {"name": "安全股份", "formation": ["boom"], "reason": "https://private/?key=SCTsecret"},
    ], cycle_label=["private"], entry_strictness={"private": 1}, missing_codes=[{"token": 1}, "secret"])
    message = partial.render_preopen_partial_message(tmp_path, now=NOW, session=True)
    assert "安全股份｜形态待核验｜完整买入条件未确认" in message.body
    assert "核验流程未完成" in message.body
    for forbidden in ("private", "secret", "SECRET", "SCT", "外链"):
        assert forbidden not in message.body


def test_candidates_without_a_cutoff_never_claim_a_current_formation(tmp_path):
    assert _write(tmp_path, data_cutoff=None)
    snapshot = partial.read_preopen_partial_snapshot(tmp_path, now=NOW)
    assert snapshot["candidates"] == []
    assert snapshot["cycle_label"] is None


def test_merge_keeps_known_fields_but_replaces_missing_codes(tmp_path):
    assert _write(tmp_path)
    assert partial.write_preopen_partial_snapshot(
        tmp_path, report_date=NOW.date(), generated_at=NOW,
        missing_codes=("holding_review_incomplete",), stage="private traceback",
    )
    snap = partial.read_preopen_partial_snapshot(tmp_path, now=NOW)
    assert snap["data_cutoff"] == CUTOFF.isoformat()
    assert snap["candidates"][0]["name"] == "示例股份"
    assert snap["missing_codes"] == ["holding_review_incomplete"]
    assert "traceback" not in (tmp_path / partial.SNAPSHOT_NAME).read_text()


@pytest.mark.parametrize("changed", [{"data_cutoff": date(2026, 10, 27)}, {"method_version": "method-v6"}])
def test_cutoff_or_method_change_invalidates_previous_evidence(tmp_path, changed):
    assert _write(tmp_path)
    assert partial.write_preopen_partial_snapshot(
        tmp_path, report_date=NOW.date(), generated_at=NOW, **changed,
    )
    snapshot = partial.read_preopen_partial_snapshot(tmp_path, now=NOW)
    assert snapshot["candidates"] == []
    assert snapshot["cycle_label"] is None and snapshot["entry_strictness"] is None


def test_out_of_order_write_rejected_and_new_day_never_merges_old_candidates(tmp_path):
    assert _write(tmp_path)
    before = (tmp_path / partial.SNAPSHOT_NAME).read_bytes()
    assert not _write(tmp_path, generated_at=NOW - timedelta(minutes=10))
    assert (tmp_path / partial.SNAPSHOT_NAME).read_bytes() == before
    tomorrow = NOW + timedelta(days=1)
    assert partial.write_preopen_partial_snapshot(tmp_path, report_date=tomorrow.date(), generated_at=tomorrow)
    snapshot = partial.read_preopen_partial_snapshot(tmp_path, now=tomorrow)
    assert snapshot["candidates"] == [] and snapshot["data_cutoff"] is None


def test_locked_writer_is_nonblocking_and_preserves_previous_snapshot(tmp_path):
    assert _write(tmp_path)
    descriptor = os.open(tmp_path / "preopen-partial-snapshot.lock", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not _write(tmp_path, generated_at=NOW)
    finally:
        os.close(descriptor)
    assert partial.read_preopen_partial_snapshot(tmp_path, now=NOW)["candidates"]


@pytest.mark.parametrize("independent", [False, True])
def test_private_account_plan_is_not_a_public_candidate_source(tmp_path, independent):
    plan = {"holding_based": True, "screening_candidates": [{"name": "示例股份", "formation": "回踩企稳"}]}
    if independent:
        plan["market_model_portfolio"] = {"entries": []}
    digest = SimpleNamespace(continuous_plan=plan, plan_for_date=NOW.date(), common_cutoff=CUTOFF,
                             cycle_label="中期上行｜短线增强", entry_strictness="standard", method_version="v5")
    assert partial.write_preopen_partial_digest(tmp_path, digest, generated_at=NOW)
    rows = partial.read_preopen_partial_snapshot(tmp_path, now=NOW)["candidates"]
    assert bool(rows) is independent


def test_only_five_distinct_names_and_no_advice_prices_are_rendered(tmp_path):
    candidates = [{"name": f"样本{i}股份", "formation": "回踩企稳", "entry_qualified": True,
                   "trigger_price": 9.12, "account_weight": 0.2, "reason": "请买入价格9.12"} for i in range(20)]
    assert _write(tmp_path, screening_candidates=candidates)
    message = partial.render_preopen_partial_message(tmp_path, now=NOW, session=True)
    assert message.body.count("股份｜") == 5
    assert "9.12" not in message.body and "20%" not in message.body


@pytest.mark.parametrize("reason,expected", [
    ("财务与重大风险核验通过", "基本面核验通过，完整买入条件未确认"),
    ("财务与重大风险核验通过；可交易性未确认", "可交易性仍需核验"),
    ("基本面风险不合格，不新买", "风险条件不合格，不新买"),
    ("结构风险9.0%，超过8%", "风险条件不合格，不新买"),
])
def test_verified_or_vetoed_checks_are_not_misrepresented_as_unknown(tmp_path, reason, expected):
    assert _write(tmp_path, screening_candidates=[{"name": "示例股份", "formation": "回踩企稳", "reason": reason}])
    message = partial.render_preopen_partial_message(tmp_path, now=NOW, session=True)
    assert expected in message.body
    assert "未确认今日买入条件" in message.body
