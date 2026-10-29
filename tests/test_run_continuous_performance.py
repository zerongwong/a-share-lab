from __future__ import annotations

import copy
import json
import sqlite3
from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest

from ashare_lab.services import run_continuous_performance as service

CUTOFF = date(2026, 9, 4)


class _Repository:
    def __init__(self):
        self.db = sqlite3.connect(":memory:")

    @contextmanager
    def connection(self):
        yield self.db


@pytest.fixture
def repository():
    repo = _Repository()
    yield repo
    repo.db.close()


def _snapshot(**changes):
    return {
        "as_of": CUTOFF.isoformat(),
        "data_cutoff": CUTOFF.isoformat(),
        "holding_revision_id": "synthetic-holding-revision",
        "holding_version": 2,
        "mode": "actual",
        "method_version": service.CONTINUOUS_METHOD_VERSION,
        "portfolio_id": "synthetic-stable-continuous-account",
        "position_units": {"600001.SH": 10},
        "close_prices": {"600001": 20},
        "cash_before_fees_and_dividends": 100,
        "fees": 1,
        "dividend_cash": 2,
        "external_flow": 0,
        "evidence": {
            "positions_complete": True,
            "cash_complete": True,
            "trades_complete": True,
            "fees_complete": True,
            "dividends_complete": True,
            "corporate_actions_complete": True,
            "prices_verified": True,
            "external_flows_complete": True,
            "user_confirmed": True,
            "evidence_ids": ["synthetic-account-evidence"],
            "covered_from": "2026-09-01",
            "covered_through": "2026-09-04",
        },
        **changes,
    }


def _portfolio(snapshot):
    return SimpleNamespace(
        id="synthetic-holding-revision",
        version=2,
        positions=(SimpleNamespace(symbol="600001", name="私有合成名字"),),
        metadata={"continuous_valuation_snapshots": {CUTOFF.isoformat(): snapshot}},
    )


def _use_portfolio(monkeypatch, portfolio):
    monkeypatch.setattr(service, "get_active_holding_portfolio", lambda _repo: portfolio)


def _assert_no_journal(repository):
    assert not repository.db.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'continuous_strategy_valuations'"
    ).fetchone()


@pytest.mark.parametrize("kind", ["no_portfolio", "no_metadata", "no_today"])
def test_absent_explicit_snapshot_is_missing_without_poisoning_future_chain(
    repository, monkeypatch, kind
):
    portfolio = _portfolio(_snapshot())
    if kind == "no_portfolio":
        portfolio = None
    elif kind == "no_metadata":
        portfolio.metadata = {}
    else:
        portfolio.metadata = {"continuous_valuation_snapshots": {}}
    _use_portfolio(monkeypatch, portfolio)
    result = service.run_continuous_performance(repository, as_of=CUTOFF)
    assert result["status"] == "inputs_missing"
    assert result["valuations_recorded"] == 0
    _assert_no_journal(repository)
    _use_portfolio(monkeypatch, _portfolio(_snapshot()))
    valid = service.run_continuous_performance(repository, as_of=CUTOFF)
    assert valid["status"] == "recorded"
    assert valid["nav_available_count"] == 1


def test_explicit_snapshot_records_once_without_mutating_holdings_or_logging_amounts(
    repository, monkeypatch
):
    portfolio = _portfolio(_snapshot())
    original = copy.deepcopy(portfolio.metadata)
    _use_portfolio(monkeypatch, portfolio)
    result = service.run_continuous_performance(repository, as_of=CUTOFF)
    assert result["status"] == "recorded"
    assert result["valuations_recorded"] == 1
    assert result["external_delivery_allowed"] is False
    assert result["orders_enabled"] is False
    repeat = service.run_continuous_performance(repository, as_of=CUTOFF)
    assert repeat["status"] == "already_recorded"
    assert repeat["valuations_recorded"] == 0
    assert portfolio.metadata == original
    record = repository.db.execute(
        "SELECT input_json, result_json FROM continuous_strategy_valuations"
    ).fetchone()
    inputs, valuation = map(json.loads, record)
    assert inputs["position_units"] == {"600001": 10.0}
    assert inputs["evidence"]["holding_revision_id"] == portfolio.id
    assert valuation["net_assets"] == 301.0
    encoded = json.dumps(result)
    for private in (
        "600001",
        "私有合成名字",
        "301",
        "synthetic-holding-revision",
        "net_assets",
        "cumulative_return",
    ):
        assert private not in encoded


@pytest.mark.parametrize("method", [f"continuous-signal-v{version}" for version in range(1, 6)])
def test_supported_historical_stream_keeps_original_version_and_identity(
    repository, monkeypatch, method
):
    snapshot = _snapshot(method_version=method)
    portfolio = _portfolio(snapshot)
    original = copy.deepcopy(portfolio.metadata)
    _use_portfolio(monkeypatch, portfolio)
    first = service.run_continuous_performance(repository, as_of=CUTOFF)
    repeat = service.run_continuous_performance(repository, as_of=CUTOFF)
    assert first["status"] == "recorded"
    assert repeat["status"] == "already_recorded"
    record = repository.db.execute(
        "SELECT valuation_id, portfolio_id, method_version FROM continuous_strategy_valuations"
    ).fetchone()
    expected_id = str(
        uuid5(
            NAMESPACE_URL,
            f"a-share-lab:continuous-actual-valuation:{method}:"
            f"{snapshot['portfolio_id']}:{CUTOFF.isoformat()}",
        )
    )
    assert record == (expected_id, snapshot["portfolio_id"], method)
    assert portfolio.metadata == original


def test_v4_account_cannot_be_silently_migrated_to_v5_after_upgrade(repository, monkeypatch):
    portfolio = _portfolio(_snapshot(method_version="continuous-signal-v4"))
    _use_portfolio(monkeypatch, portfolio)
    assert service.run_continuous_performance(repository, as_of=CUTOFF)["status"] == "recorded"
    before = repository.db.execute("SELECT * FROM continuous_strategy_valuations").fetchall()
    next_date = date(2026, 9, 7)
    snapshot = _snapshot(as_of=next_date.isoformat(), data_cutoff=next_date.isoformat())
    snapshot["evidence"]["covered_through"] = next_date.isoformat()
    portfolio.metadata["continuous_valuation_snapshots"][next_date.isoformat()] = snapshot
    result = service.run_continuous_performance(repository, as_of=next_date)
    assert result["status"] == "inputs_invalid"
    assert (
        repository.db.execute("SELECT * FROM continuous_strategy_valuations").fetchall() == before
    )


def test_v4_account_continues_original_nav_chain_under_v5_runner(repository, monkeypatch):
    portfolio = _portfolio(_snapshot(method_version="continuous-signal-v4"))
    _use_portfolio(monkeypatch, portfolio)
    assert service.run_continuous_performance(repository, as_of=CUTOFF)["status"] == "recorded"
    next_date = date(2026, 9, 7)
    snapshot = _snapshot(
        as_of=next_date.isoformat(),
        data_cutoff=next_date.isoformat(),
        method_version="continuous-signal-v4",
    )
    snapshot["evidence"]["covered_through"] = next_date.isoformat()
    portfolio.metadata["continuous_valuation_snapshots"][next_date.isoformat()] = snapshot
    assert service.run_continuous_performance(repository, as_of=next_date)["status"] == "recorded"
    rows = repository.db.execute(
        "SELECT method_version, result_json, record_hash FROM continuous_strategy_valuations "
        "ORDER BY as_of"
    ).fetchall()
    assert len(rows) == 2
    assert all(row[0] == "continuous-signal-v4" for row in rows)
    assert json.loads(rows[1][1])["previous_record_hash"] == rows[0][2]


@pytest.mark.parametrize(
    "changes",
    [
        {"as_of": "2026-09-03"},
        {"data_cutoff": "2026-09-03"},
        {"holding_revision_id": "another-revision"},
        {"holding_version": 1},
        {"holding_version": True},
        {"mode": "shadow"},
        {"method_version": "legacy-one-month"},
        {"method_version": "continuous-signal-v6"},
        {"method_version": []},
        {"portfolio_id": None},
        {"position_units": {"600002": 10}},
        {"position_units": {}},
        {"position_units": {"600001": 10, "600001.SH": 10}},
        {"evidence": {"user_confirmed": False}},
        {"evidence": {"user_confirmed": "yes"}},
    ],
)
def test_mismatched_or_unconfirmed_snapshot_has_no_write(repository, monkeypatch, changes):
    _use_portfolio(monkeypatch, _portfolio(_snapshot(**changes)))
    result = service.run_continuous_performance(repository, as_of=CUTOFF)
    assert result["status"] == "inputs_invalid"
    assert result["valuations_recorded"] == 0
    _assert_no_journal(repository)


def test_missing_required_financial_field_is_not_assumed_zero(repository, monkeypatch):
    snapshot = _snapshot()
    del snapshot["fees"]
    _use_portfolio(monkeypatch, _portfolio(snapshot))
    result = service.run_continuous_performance(repository, as_of=CUTOFF)
    assert result["status"] == "inputs_invalid"
    _assert_no_journal(repository)


def test_explicit_unknown_corporate_actions_journals_no_invented_nav(repository, monkeypatch):
    snapshot = _snapshot()
    snapshot["evidence"]["corporate_actions_complete"] = False
    _use_portfolio(monkeypatch, _portfolio(snapshot))
    result = service.run_continuous_performance(repository, as_of=CUTOFF)
    assert result["status"] == "unavailable"
    assert result["nav_available_count"] == 0
    stored = json.loads(
        repository.db.execute("SELECT result_json FROM continuous_strategy_valuations").fetchone()[
            0
        ]
    )
    assert stored["normalized_nav"] is None
    assert stored["cumulative_return"] is None


def test_external_deposit_is_not_earned_return(repository, monkeypatch):
    _use_portfolio(monkeypatch, _portfolio(_snapshot(external_flow=100)))
    result = service.run_continuous_performance(repository, as_of=CUTOFF)
    assert result["status"] == "unavailable"
    assert result["nav_available_count"] == 0


def test_later_user_edit_cannot_rewrite_same_day_valuation(repository, monkeypatch):
    portfolio = _portfolio(_snapshot())
    _use_portfolio(monkeypatch, portfolio)
    service.run_continuous_performance(repository, as_of=CUTOFF)
    before = repository.db.execute(
        "SELECT input_json, result_json FROM continuous_strategy_valuations"
    ).fetchall()
    portfolio.metadata["continuous_valuation_snapshots"][CUTOFF.isoformat()]["fees"] = 3
    result = service.run_continuous_performance(repository, as_of=CUTOFF)
    assert result["status"] == "inputs_invalid"
    assert (
        repository.db.execute(
            "SELECT input_json, result_json FROM continuous_strategy_valuations"
        ).fetchall()
        == before
    )


def test_storage_error_reports_fixed_reason_without_private_exception_text(repository, monkeypatch):
    def broken(_repo):
        raise sqlite3.OperationalError("private file path and account details")

    monkeypatch.setattr(service, "get_active_holding_portfolio", broken)
    result = service.run_continuous_performance(repository, as_of=CUTOFF)
    assert result["status"] == "error_retry_later"
    assert "private" not in json.dumps(result)
