from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.ports.company_actions import CompanyActionEvidenceStatus
from ashare_lab.services import run_recommendation_tracking as service

CUTOFF = date(2026, 9, 30)
PLAN = date(2026, 10, 8)
NEXT = date(2026, 10, 9)


@pytest.fixture
def repository(tmp_path):
    repo = SQLiteRepository(tmp_path / "research.db")
    with repo.connection() as connection:
        connection.executescript("""
            CREATE TABLE recommendation_reports (
                id TEXT PRIMARY KEY, plan_for_date TEXT, common_cutoff TEXT,
                method_version TEXT, archive_nature TEXT);
            CREATE TABLE recommendation_delivery_events (
                report_id TEXT, provider_status TEXT, delivery_kind TEXT, attempted_at TEXT);
            CREATE TABLE continuous_strategy_decisions (
                decision_id TEXT PRIMARY KEY, as_of TEXT, method_version TEXT,
                payload_json TEXT, payload_hash TEXT);
        """)
    return repo


def _entry():
    return {
        "symbol": "600001", "name": "合成样本", "entry_qualified": True,
        "account_weight": 0.1, "structural_invalidation_price": 9.5,
        "entry_rule": {
            "schema_version": "recommendation-entry-rule-v1",
            "kind": "reclaim_close_confirmation", "data_cutoff": CUTOFF.isoformat(),
            "trigger_price": 10.0, "maximum_entry_price": 10.2,
            "structural_invalidation_price": 9.5,
        },
    }


def _archive(repo, entries, *, identifier="report", accepted=True, nature="original",
             tamper=False, model_entries=None, plan=PLAN, cutoff=CUTOFF):
    payload = {"mode": "continuous", "plan_for_date": plan.isoformat(), "entries": entries,
               "screening_candidates": [{"symbol": "600999", "entry_qualified": False}]}
    if model_entries is not None:
        payload["market_model_portfolio"] = {"entries": model_entries}
    document = {"decision_id": identifier, "as_of": cutoff.isoformat(),
                "method_version": "continuous-signal-v4", "payload": payload}
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"))
    with repo.connection() as connection:
        connection.execute("INSERT INTO recommendation_reports VALUES (?,?,?,?,?)", (
            identifier, plan.isoformat(), cutoff.isoformat(), "continuous-signal-v4", nature,
        ))
        connection.execute("INSERT INTO continuous_strategy_decisions VALUES (?,?,?,?,?)", (
            identifier, cutoff.isoformat(), "continuous-signal-v4", encoded,
            "bad" if tamper else hashlib.sha256(encoded.encode()).hexdigest(),
        ))
        connection.execute("INSERT INTO recommendation_delivery_events VALUES (?,?,?,?)", (
            identifier, "provider_accepted" if accepted else "provider_failed",
            "evening_provider_accepted" if accepted else "evening_provider_failed",
            f"{plan.isoformat()}T00:30:00+00:00",
        ))


class Overlay:
    def __init__(self, *, gap=False, adjusted=False, anchor=True):
        self.gap, self.adjusted, self.anchor = gap, adjusted, anchor
        self.read_days = []

    def read_verified_manifest(self, *, source_id):
        assert source_id == "zero_budget_eod"
        pairs = [(CUTOFF, date(2026, 9, 29)), (PLAN, CUTOFF),
                 (NEXT, date(2026, 10, 7) if self.gap else PLAN)]
        if not self.anchor:
            pairs = pairs[1:]
        return pd.DataFrame([
            {"trade_date": day, "previous_trade_date": prior,
             "adjustment": "qfq" if self.adjusted else "none"}
            for day, prior in pairs
        ])

    def read_verified_daily(self, day, *, source_id, asset_kind):
        self.read_days.append(day)
        return pd.DataFrame([{
            "symbol": "600001.SH", "trade_date": pd.Timestamp(day),
            "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1,
            "prev_close": 10.0, "volume_shares": 1_000_000.0, "amount_cny": 10_000_000.0,
        }])


def _clear_reader(calls):
    def read(symbols, as_of, *, timeout_seconds):
        calls.append((symbols, as_of))
        assert 0 < timeout_seconds <= 10
        return tuple(SimpleNamespace(
            symbol=symbol, status=CompanyActionEvidenceStatus.CLEAR,
            coverage_from=date(1990, 1, 1), coverage_through=as_of,
            event_dates=(), response_hash="a" * 64, reason_code="CLEAR",
            knowledge_time=datetime(2026, 10, 11, tzinfo=UTC),
            stream_receipts=tuple(SimpleNamespace(stream=stream, complete=True)
                                  for stream in service._STREAMS),
        ) for symbol in symbols)
    return read


def test_empty_accepted_reports_are_recorded_once_without_market_or_provider_reads(repository):
    _archive(repository, [])
    _archive(repository, [_entry()], identifier="failed", accepted=False)
    _archive(repository, [_entry()], identifier="reconstructed", nature="reconstructed")
    result = service.run_recommendation_tracking(
        repository, overlay_store=object(), through=NEXT,
        company_action_reader=lambda *a, **kw: pytest.fail("no symbols means no network"),
    )
    assert result["status"] == "no_qualified_recommendations"
    assert result["reports_ingested"] == result["reports_seen"] == 1
    assert result["entries_seen"] == 0
    assert result["summary"]["zero_entry_reports"] == 1
    assert result["summary"]["episodes"] == []
    assert service.ingest_published_recommendations(repository)["reports_ingested"] == 0


def test_bad_hash_is_rejected_and_observations_never_count(repository):
    _archive(repository, [_entry()], tamper=True)
    result = service.ingest_published_recommendations(repository)
    assert result["reports_ingested"] == result["entries_seen"] == 0
    assert result["archive_rejected"] == 1


def test_actual_and_market_plan_mentions_deduplicate_without_ledger_access(repository):
    _archive(repository, [_entry()], model_entries=[_entry()])
    result = service.ingest_published_recommendations(repository)
    assert result["reports_ingested"] == result["entries_seen"] == 1
    with repository.connection() as connection:
        payload = json.loads(connection.execute("SELECT payload_json FROM tracking_reports").fetchone()[0])
    assert payload["entries"][0]["entry_rule"] == _entry()["entry_rule"]


def test_conflicting_conditions_do_not_silently_select_one(repository):
    altered = _entry()
    altered["entry_rule"]["trigger_price"] = 10.15
    _archive(repository, [_entry()], model_entries=[altered])
    result = service.ingest_published_recommendations(repository)
    assert result["archive_rejected"] == 1
    assert result["reports_ingested"] == 0


def test_verified_chain_does_not_jump_over_a_missing_session():
    store = Overlay(gap=True)
    sessions, bars, reason = service._verified_chain(
        store, source_id="zero_budget_eod", start=CUTOFF, through=NEXT, symbols=["600001"]
    )
    assert sessions == [CUTOFF, PLAN]
    assert NEXT not in store.read_days and NEXT not in bars["600001"]
    assert reason == "verified_session_chain_gap"
    assert bars["600001"][PLAN]["prev_close"] == 10.0


def test_missing_anchor_and_adjusted_data_never_qualify():
    result = service._verified_chain(
        Overlay(anchor=False), source_id="zero_budget_eod", start=CUTOFF,
        through=NEXT, symbols=["600001"],
    )
    assert result == ([], {}, "anchor_session_missing")
    with pytest.raises(ValueError, match="adjusted"):
        service._verified_chain(Overlay(adjusted=True), source_id="zero_budget_eod",
                                start=CUTOFF, through=NEXT, symbols=["600001"])


def test_replay_uses_verified_bars_and_reuses_complete_same_day_public_evidence(repository):
    _archive(repository, [_entry()])
    calls = []
    for _ in range(2):
        result = service.run_recommendation_tracking(
            repository, overlay_store=Overlay(), through=NEXT,
            company_action_reader=_clear_reader(calls),
        )
    assert result["status"] == "evaluated"
    assert result["latest_verified_date"] == NEXT.isoformat()
    assert calls == [(("600001",), NEXT)]
    assert result["summary"]["episodes"][0]["entry_price"] == 10.0
    assert result["summary"]["episodes"][0]["entry_date"] == NEXT.isoformat()


def test_unknown_company_actions_are_retried_not_cached_as_clear(repository):
    calls = []
    def failed(symbols, as_of, **kwargs):
        calls.append(symbols)
        raise OSError("unavailable")
    for _ in range(2):
        evidence = service._cached_company_clearances(repository, ["600001"], NEXT, reader=failed)
        assert evidence["600001"]["status"] == "unknown"
        assert evidence["600001"]["complete"] is False
    assert len(calls) == 2


def test_missing_structured_rule_remains_a_rule_missing_episode(repository):
    entry = _entry()
    entry.pop("entry_rule")
    _archive(repository, [entry])
    result = service.run_recommendation_tracking(
        repository, overlay_store=Overlay(), through=NEXT, company_action_reader=_clear_reader([])
    )
    episode = result["summary"]["episodes"][0]
    assert episode["status"] == "rule_missing"
    assert episode["entry_price"] is None


def test_sync_replays_recommendations_even_when_actual_account_snapshot_is_unavailable(
    monkeypatch, tmp_path,
):
    from ashare_lab.cli import scheduled_sync
    from ashare_lab.services import run_continuous_performance

    def actual_unavailable(*args, **kwargs):
        raise ValueError("actual account inputs unavailable")

    calls = []
    def track(repo, *, overlay_store, through, source_id):
        calls.append((repo.db_path, overlay_store.root, through, source_id))
        return {"status": "evaluated", "reports_ingested": 1, "archive_rejected": 0}

    monkeypatch.setattr(run_continuous_performance, "run_continuous_performance", actual_unavailable)
    monkeypatch.setattr(service, "run_recommendation_tracking", track)
    event = {}
    scheduled_sync._run_continuous_accounting(
        event, scheduler_root=tmp_path / "scheduler", common_cutoff=NEXT,
        overlay_root=tmp_path / "overlay", source_id="test_source",
    )
    assert event["continuous_performance_status"] == "error_retry_later"
    assert event["recommendation_tracking_status"] == "evaluated"
    assert calls == [(tmp_path / "research.db", tmp_path / "overlay", NEXT, "test_source")]


class LaterOverlay(Overlay):
    def read_verified_manifest(self, *, source_id):
        pairs = [(CUTOFF, date(2026, 9, 29)), (PLAN, CUTOFF), (NEXT, PLAN),
                 (date(2026, 10, 12), NEXT), (date(2026, 10, 13), date(2026, 10, 12))]
        return pd.DataFrame([{"trade_date": day, "previous_trade_date": prior, "adjustment": "none"}
                             for day, prior in pairs])

    def read_verified_daily(self, day, **kwargs):
        frame = super().read_verified_daily(day, **kwargs)
        if day == date(2026, 10, 12):
            frame.loc[:, ["open", "high", "low", "close"]] = [9.3, 9.5, 9.0, 9.4]
        return frame


def test_closed_episode_survives_refresh_failure_with_new_recommendation(repository):
    close_day, later_day = date(2026, 10, 12), date(2026, 10, 13)
    _archive(repository, [_entry()])
    first = service.run_recommendation_tracking(
        repository, overlay_store=LaterOverlay(), through=close_day,
        company_action_reader=_clear_reader([]),
    )
    closed = first["summary"]["episodes"][0]
    assert closed["status"] == "closed"
    new_entry = _entry()
    new_entry["entry_rule"]["data_cutoff"] = close_day.isoformat()
    _archive(repository, [new_entry], identifier="new_report", plan=later_day, cutoff=close_day)
    calls = []
    def failing(symbols, as_of, **kwargs):
        calls.append((symbols, as_of))
        raise OSError("temporary network failure")
    replay = service.run_recommendation_tracking(
        repository, overlay_store=LaterOverlay(), through=later_day, company_action_reader=failing,
    )
    episodes = replay["summary"]["episodes"]
    assert calls == [(("600001",), later_day)]
    assert episodes[0]["status"] == "closed"
    assert episodes[0]["exit_price"] == closed["exit_price"]
    assert episodes[0]["return_fraction"] == closed["return_fraction"]
    assert episodes[1]["status"] == "corporate_action_review_required"


def test_all_closed_and_no_new_reports_reuses_sufficient_evidence_without_network(repository):
    _archive(repository, [_entry()])
    service.run_recommendation_tracking(
        repository, overlay_store=LaterOverlay(), through=date(2026, 10, 12),
        company_action_reader=_clear_reader([]),
    )
    result = service.run_recommendation_tracking(
        repository, overlay_store=LaterOverlay(), through=date(2026, 10, 13),
        company_action_reader=lambda *a, **kw: pytest.fail("closed episode needs no refresh"),
    )
    assert result["summary"]["episodes"][0]["status"] == "closed"


def test_past_clearance_never_claims_to_cover_later_session(repository):
    service._cached_company_clearances(repository, ["600001"], NEXT, reader=_clear_reader([]))
    def failing(*args, **kwargs):
        raise OSError("unavailable")
    result = service._cached_company_clearances(
        repository, ["600001"], date(2026, 10, 12), reader=failing,
    )
    assert result["600001"]["coverage_through"] == NEXT.isoformat()
    assert result["600001"]["complete"] is True
    with repository.connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM tracking_company_action_cache WHERE through_date='2026-10-12'"
        ).fetchone()[0] == 0


def test_budget_limited_provider_queries_rotate_persistently(repository, monkeypatch):
    symbols = [f"6000{i:02d}" for i in range(12)]
    calls = []
    def failing(batch, as_of, **kwargs):
        calls.append(batch)
        raise OSError("provider unavailable")
    for _ in range(3):
        ticks = iter([0.0, 0.0, 50.0])
        monkeypatch.setattr(service.time, "monotonic", lambda ticks=ticks: next(ticks))
        service._cached_company_clearances(repository, symbols, NEXT, reader=failing)
    assert calls[0] == tuple(symbols[:5])
    assert calls[1] == tuple(symbols[5:10])
    assert calls[2][:2] == tuple(symbols[10:])


def test_all_unfilled_and_no_new_reports_does_not_refresh_completed_observation(repository):
    entry = _entry()
    entry["entry_rule"]["trigger_price"] = 10.15
    _archive(repository, [entry])
    first = service.run_recommendation_tracking(
        repository, overlay_store=LaterOverlay(), through=NEXT,
        company_action_reader=_clear_reader([]),
    )
    assert first["summary"]["episodes"][0]["status"] == "unfilled"
    result = service.run_recommendation_tracking(
        repository, overlay_store=LaterOverlay(), through=date(2026, 10, 13),
        company_action_reader=lambda *a, **kw: pytest.fail("unfilled plan needs no refresh"),
    )
    assert result["summary"]["episodes"][0]["status"] == "unfilled"
