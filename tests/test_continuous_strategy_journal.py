from __future__ import annotations

import json
import sqlite3
from datetime import date

import pytest

from ashare_lab.services.continuous_strategy_journal import (
    archive_continuous_decision,
    ensure_continuous_strategy_schema,
    record_continuous_valuation,
)


@pytest.fixture
def connection():
    connection = sqlite3.connect(":memory:")
    yield connection
    connection.close()


def _evidence(**changes):
    return {
        "positions_complete": True,
        "cash_complete": True,
        "trades_complete": True,
        "fees_complete": True,
        "dividends_complete": True,
        "corporate_actions_complete": True,
        "prices_verified": True,
        "external_flows_complete": True,
        "user_confirmed": True,
        "covered_from": "2026-09-01",
        "covered_through": "2026-09-30",
        "evidence_ids": ["local-user-confirmation-1", "verified-close-and-actions-1"],
        **changes,
    }


def _valuation(connection, **changes):
    args = {
        "valuation_id": "valuation-1",
        "portfolio_id": "new-continuous-account-v1",
        "as_of": date(2026, 9, 1),
        "mode": "actual",
        "position_units": {"600000": 10.0},
        "close_prices": {"600000": 10.0},
        "cash_before_fees_and_dividends": 100.0,
        "fees": 0.0,
        "dividend_cash": 0.0,
        "external_flow": 0.0,
        "evidence": _evidence(),
        **changes,
    }
    return record_continuous_valuation(connection, **args)


def test_decision_is_idempotent_not_a_fill_and_copies_input(connection):
    payload = {"candidates": [{"symbol": "600000", "action": "conditional_entry"}]}
    first = archive_continuous_decision(connection, "d1", date(2026, 9, 1), payload)
    assert archive_continuous_decision(connection, "d1", "2026-09-01", payload) == first
    assert first["record_nature"] == "decision_not_execution"
    assert first["external_delivery_allowed"] is False
    assert first["orders_enabled"] is False
    assert (
        connection.execute("SELECT COUNT(*) FROM continuous_strategy_valuations").fetchone()[0] == 0
    )
    payload["candidates"][0]["symbol"] = "600001"
    assert first["payload"]["candidates"][0]["symbol"] == "600000"


@pytest.mark.parametrize(
    "changes",
    [
        {"payload": {"different": True}},
        {"as_of": date(2026, 9, 2)},
        {"method_version": "different-method"},
    ],
)
def test_decision_conflict_never_rewrites(connection, changes):
    args = {"decision_id": "d1", "as_of": date(2026, 9, 1), "payload": {"x": 1}}
    original = archive_continuous_decision(connection, **args)
    with pytest.raises(ValueError, match="different immutable content"):
        archive_continuous_decision(connection, **(args | changes))
    assert archive_continuous_decision(connection, **args) == original


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_and_valuation_numbers_are_rejected(connection, value):
    with pytest.raises(ValueError, match="finite"):
        archive_continuous_decision(connection, "d1", date(2026, 9, 1), {"nested": [value]})
    with pytest.raises(ValueError, match="finite"):
        _valuation(connection, close_prices={"600000": value})


@pytest.mark.parametrize("payload", [{1: "non-string-key"}, {"object": object()}, {"set": {1}}])
def test_non_json_payload_cannot_be_archived(connection, payload):
    with pytest.raises(TypeError):
        archive_continuous_decision(connection, "d1", date(2026, 9, 1), payload)


def test_unrealized_gain_cash_fees_and_dividends_all_enter_nav(connection):
    initial = _valuation(connection)
    assert initial["net_assets"] == 200.0
    assert initial["normalized_nav"] == 1.0
    assert initial["cumulative_return"] == 0.0  # Explicit positive inception baseline.
    assert initial["previous_snapshot_return"] is None
    second = _valuation(
        connection,
        valuation_id="valuation-2",
        as_of=date(2026, 9, 2),
        close_prices={"600000": 12.0},
        fees=2.0,
        dividend_cash=3.0,
    )
    assert second["open_position_market_value"] == 120.0
    assert second["cash_after_fees_and_dividends"] == 101.0
    assert second["net_assets"] == 221.0
    assert second["normalized_nav"] == pytest.approx(1.105)
    assert second["cumulative_return"] == pytest.approx(0.105)
    assert second["previous_snapshot_return"] == pytest.approx(0.105)
    assert second["previous_valuation_id"] == "valuation-1"
    assert second["previous_record_hash"]
    assert second["performance_nature"] == "actual_user_confirmed_accounting"
    assert second["external_delivery_allowed"] is False
    assert second["orders_enabled"] is False
    # Earlier fee/dividend changes are now already in the input cash. They are
    # not implicitly re-applied by the accounting engine on future snapshots.
    third = _valuation(
        connection,
        valuation_id="valuation-3",
        as_of=date(2026, 9, 3),
        close_prices={"600000": 11.0},
        cash_before_fees_and_dividends=101.0,
    )
    assert third["net_assets"] == 211.0
    assert third["cumulative_return"] == pytest.approx(0.055)
    assert third["previous_snapshot_return"] == pytest.approx(211.0 / 221.0 - 1.0)


def test_open_loss_is_not_hidden_by_only_counting_closed_trades(connection):
    _valuation(connection)
    result = _valuation(
        connection,
        valuation_id="valuation-2",
        as_of=date(2026, 9, 2),
        close_prices={"600000": 6.0},
    )
    assert result["cumulative_return"] == pytest.approx(-0.2)


def test_confirmed_sale_and_empty_stock_sleeve_keep_proceeds_as_cash(connection):
    _valuation(connection)
    result = _valuation(
        connection,
        valuation_id="valuation-2",
        as_of=date(2026, 9, 2),
        position_units={},
        close_prices={},
        cash_before_fees_and_dividends=210.0,
        fees=1.0,
    )
    assert result["open_position_market_value"] == 0.0
    assert result["net_assets"] == 209.0
    assert result["cumulative_return"] == pytest.approx(0.045)


@pytest.mark.parametrize(
    "flag",
    [
        "positions_complete",
        "cash_complete",
        "trades_complete",
        "fees_complete",
        "dividends_complete",
        "corporate_actions_complete",
        "prices_verified",
        "external_flows_complete",
        "user_confirmed",
    ],
)
def test_missing_required_evidence_is_unavailable_not_zero(connection, flag):
    result = _valuation(connection, evidence=_evidence(**{flag: False}))
    assert result["status"] == result["snapshot_status"] == "unavailable"
    assert result["net_assets"] is None
    assert result["normalized_nav"] is None
    assert result["cumulative_return"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"close_prices": {}},
        {"close_prices": {"600000": None}},
        {"close_prices": {"600000": 0.0}},
        {"cash_before_fees_and_dividends": None},
        {"fees": None},
        {"dividend_cash": None},
        {"evidence": _evidence(evidence_ids=[])},
        {"evidence": _evidence(covered_from="2026-09-02")},
        {"evidence": _evidence(covered_through="2026-08-31")},
        {"evidence": _evidence(covered_from=None)},
    ],
)
def test_incomplete_snapshot_fails_closed(connection, changes):
    result = _valuation(connection, **changes)
    assert result["status"] == "unavailable"
    assert result["net_assets"] is None
    assert result["cumulative_return"] is None
    assert result["reasons"]


def test_interval_evidence_must_cover_since_previous_snapshot_not_just_today(connection):
    _valuation(connection)
    result = _valuation(
        connection,
        valuation_id="valuation-2",
        as_of=date(2026, 9, 4),
        evidence=_evidence(covered_from="2026-09-04"),
    )
    assert "evidence_does_not_cover_valuation_interval" in result["reasons"]


@pytest.mark.parametrize("flow", [100.0, -50.0, None])
def test_external_cash_flow_cannot_be_reported_as_profit_or_silently_rebased(connection, flow):
    _valuation(connection)
    second = _valuation(
        connection,
        valuation_id="valuation-2",
        as_of=date(2026, 9, 2),
        cash_before_fees_and_dividends=200.0,
        external_flow=flow,
    )
    assert second["net_assets"] == 300.0  # Snapshot equity can be independently known.
    assert second["snapshot_status"] == "ready"
    assert second["status"] == "unavailable"
    assert second["cumulative_return"] is None
    assert second["normalized_nav"] is None
    third = _valuation(connection, valuation_id="valuation-3", as_of=date(2026, 9, 3))
    assert third["cumulative_return"] is None
    assert "previous_return_chain_unavailable_no_silent_rebase" in third["reasons"]


def test_actual_and_shadow_are_separate_and_shadow_never_claims_user_fills(connection):
    actual = _valuation(connection, evidence=_evidence(user_confirmed=False))
    shadow = _valuation(
        connection,
        valuation_id="shadow-1",
        mode="shadow",
        evidence=_evidence(user_confirmed=False),
    )
    assert actual["status"] == "unavailable"
    assert shadow["status"] == "ready"
    assert shadow["performance_nature"] == "shadow_simulation_not_actual_account"
    assert shadow["previous_valuation_id"] is None


def test_snapshot_retries_remain_idempotent_after_later_dates(connection):
    original = _valuation(connection)
    _valuation(connection, valuation_id="valuation-2", as_of=date(2026, 9, 2))
    assert _valuation(connection) == original
    assert (
        connection.execute("SELECT COUNT(*) FROM continuous_strategy_valuations").fetchone()[0] == 2
    )
    with pytest.raises(ValueError, match="different immutable content"):
        _valuation(connection, cash_before_fees_and_dividends=101.0)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"valuation_id": "same-day-another-id"}, "chronological"),
        ({"valuation_id": "earlier", "as_of": date(2026, 8, 31)}, "chronological"),
        (
            {"valuation_id": "new-method", "as_of": date(2026, 9, 2), "method_version": "v2"},
            "method_version",
        ),
    ],
)
def test_no_same_day_replacement_backdating_or_method_mixing(connection, changes, match):
    _valuation(connection)
    with pytest.raises(ValueError, match=match):
        _valuation(connection, **changes)


@pytest.mark.parametrize(
    "table,id_column",
    [
        ("continuous_strategy_decisions", "decision_id"),
        ("continuous_strategy_valuations", "valuation_id"),
    ],
)
def test_sql_update_delete_and_replace_are_blocked(connection, table, id_column):
    archive_continuous_decision(connection, "d1", date(2026, 9, 1), {})
    _valuation(connection)
    for statement in (
        f"UPDATE {table} SET method_version = 'altered'",
        f"DELETE FROM {table}",
        f"INSERT OR REPLACE INTO {table} SELECT * FROM {table}",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(statement)
    assert connection.execute(f"SELECT COUNT({id_column}) FROM {table}").fetchone()[0] == 1


def test_replacing_valuation_via_unique_stream_date_is_also_blocked(connection):
    _valuation(connection)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        connection.execute(
            """INSERT OR REPLACE INTO continuous_strategy_valuations
            SELECT 'replacement-id', portfolio_id, mode, as_of, method_version,
            input_json, input_hash, result_json, record_hash FROM continuous_strategy_valuations"""
        )


def test_initial_zero_equity_is_unknown_return_but_complete_loss_is_negative_one(connection):
    empty = _valuation(
        connection,
        portfolio_id="empty",
        position_units={},
        close_prices={},
        cash_before_fees_and_dividends=0.0,
    )
    assert empty["status"] == "unavailable"
    assert empty["cumulative_return"] is None
    _valuation(connection, valuation_id="funded-1")
    loss = _valuation(
        connection,
        valuation_id="loss",
        as_of=date(2026, 9, 2),
        position_units={},
        close_prices={},
        cash_before_fees_and_dividends=0.0,
    )
    assert loss["status"] == "ready"
    assert loss["cumulative_return"] == -1.0


def test_zero_units_need_no_quote_and_financing_fails_closed(connection):
    zero = _valuation(connection, position_units={"600000": 0}, close_prices={})
    assert zero["net_assets"] == 100.0
    invalid = _valuation(
        connection,
        portfolio_id="financed",
        valuation_id="financed",
        cash_before_fees_and_dividends=1.0,
        fees=2.0,
    )
    assert invalid["net_assets"] is None
    assert "invalid_net_assets_or_financed_cash" in invalid["reasons"]


def test_flags_are_not_truthy_strings_and_units_cannot_be_negative(connection):
    with pytest.raises(TypeError, match="boolean"):
        _valuation(connection, evidence=_evidence(user_confirmed="yes"))
    with pytest.raises(ValueError, match="nonnegative"):
        _valuation(connection, position_units={"600000": -1})


def test_unavailable_history_is_not_silently_recovered_into_valid_returns(connection):
    _valuation(connection, close_prices={})
    recovered = _valuation(connection, valuation_id="valuation-2", as_of=date(2026, 9, 2))
    assert recovered["net_assets"] == 200.0
    assert recovered["cumulative_return"] is None


def test_legacy_data_untouched_and_caller_transaction_not_committed(connection):
    connection.execute("CREATE TABLE recommendation_reports (id TEXT, content TEXT)")
    connection.execute("INSERT INTO recommendation_reports VALUES ('legacy', 'six-horizon')")
    connection.commit()
    before = connection.execute("SELECT * FROM recommendation_reports").fetchall()
    ensure_continuous_strategy_schema(connection)
    connection.execute("BEGIN")
    connection.execute("INSERT INTO recommendation_reports VALUES ('pending', 'caller-owned')")
    archive_continuous_decision(connection, "d1", date(2026, 9, 1), {})
    _valuation(connection)
    assert connection.in_transaction
    connection.rollback()
    assert connection.execute("SELECT * FROM recommendation_reports").fetchall() == before
    assert (
        connection.execute("SELECT COUNT(*) FROM continuous_strategy_decisions").fetchone()[0] == 0
    )
    assert (
        connection.execute("SELECT COUNT(*) FROM continuous_strategy_valuations").fetchone()[0] == 0
    )


def test_tampered_payload_hash_fails_closed_on_retry(connection):
    _valuation(connection)
    # A database owner can remove triggers; hashes detect casual changes even
    # then. This is not presented as security against a privileged DB owner.
    connection.execute("DROP TRIGGER continuous_strategy_valuations_no_update")
    connection.execute(
        "UPDATE continuous_strategy_valuations SET result_json = ?",
        (json.dumps({"cumulative_return": 9}),),
    )
    with pytest.raises(ValueError, match="integrity"):
        _valuation(connection)


def test_verified_split_uses_explicit_post_action_units_not_false_price_loss(connection):
    _valuation(connection)
    result = _valuation(
        connection,
        valuation_id="post-split",
        as_of=date(2026, 9, 2),
        position_units={"600000": 20.0},
        close_prices={"600000": 5.0},
    )
    assert result["net_assets"] == 200.0
    assert result["cumulative_return"] == 0.0


def test_funded_cash_baseline_captures_first_trade_cost(connection):
    _valuation(connection, position_units={}, close_prices={}, cash_before_fees_and_dividends=200.0)
    result = _valuation(
        connection,
        valuation_id="first-confirmed-trade",
        as_of=date(2026, 9, 2),
        fees=1.0,
    )
    assert result["net_assets"] == 199.0
    assert result["cumulative_return"] == pytest.approx(-0.005)


def test_conflict_rolls_back_only_owned_savepoint_not_caller_changes(connection):
    _valuation(connection)
    connection.execute("CREATE TABLE caller_data (value TEXT)")
    connection.execute("INSERT INTO caller_data VALUES ('still-pending')")
    with pytest.raises(ValueError, match="different immutable content"):
        _valuation(connection, fees=1.0)
    assert connection.in_transaction
    assert connection.execute("SELECT value FROM caller_data").fetchone()[0] == "still-pending"


def test_overflowing_market_value_is_unavailable_and_remains_strict_json(connection):
    result = _valuation(
        connection, position_units={"600000": 1e308}, close_prices={"600000": 1e308}
    )
    assert result["status"] == "unavailable"
    assert result["net_assets"] is None
    assert result["cumulative_return"] is None
    json.dumps(result, allow_nan=False)
