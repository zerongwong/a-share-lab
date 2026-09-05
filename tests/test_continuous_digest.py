from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ashare_lab.analytics.continuous_signals import CONTINUOUS_METHOD_VERSION
from ashare_lab.services import build_continuous_digest as service
from ashare_lab.services.build_evening_digest import build_evening_research_digest
from ashare_lab.services.build_midterm_portfolio import (
    ConditionalEntryPlan,
    ConditionalEntryPlanKind,
    MidtermPortfolioStatus,
)
from ashare_lab.services.review_active_holdings import HoldingAction

AS_OF = date(2026, 9, 4)


def _fixture(count=3):
    histories = {}
    symbols = [f"60000{i}" for i in range(count)]
    for index, symbol in enumerate(symbols):
        dates = pd.bdate_range(end=AS_OF, periods=201)
        returns = np.random.default_rng(index).normal(0.0005, 0.01, len(dates))
        histories[symbol] = pd.DataFrame(
            {"trade_date": dates, "close": 10 * np.cumprod(1 + returns)}
        )
    positions = tuple(
        SimpleNamespace(
            symbol=symbol, entry_date=AS_OF - timedelta(days=100), position_key=f"position-{symbol}"
        )
        for symbol in symbols
    )
    snapshot = {
        "as_of": AS_OF.isoformat(),
        "user_confirmed": True,
        "no_external_flows_since_snapshot": True,
        "no_trades_since_snapshot": True,
        "account_weights": dict.fromkeys(symbols, 0.15),
        "reference_prices": {symbol: float(histories[symbol].close.iloc[-1]) for symbol in symbols},
        "cash_weight": 1 - 0.15 * count,
    }
    portfolio = SimpleNamespace(
        id="portfolio-1", version=3, positions=positions, metadata={"account_snapshot": snapshot}
    )
    review = SimpleNamespace(
        portfolio_id=portfolio.id,
        holding_version=portfolio.version,
        data_cutoff=AS_OF,
        rows=tuple(
            SimpleNamespace(
                symbol=symbol,
                holding_version=3,
                position_key=f"position-{symbol}",
                action=HoldingAction.HOLD,
                company_action_clear=True,
                company_action_clear_from=AS_OF - timedelta(days=100),
                company_action_clear_through=AS_OF,
            )
            for symbol in symbols
        ),
    )
    result = SimpleNamespace(
        data_cutoff=pd.Timestamp(AS_OF),
        price_cycle=None,
        qualified_entry_universe=(),
        status=MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO,
        positions=(),
        cash_weight=1.0,
    )
    return {
        "result": result,
        "histories": histories,
        "metadata": {symbol: {"industry": f"industry-{symbol}"} for symbol in symbols},
        "portfolio": portfolio,
        "review": review,
        "as_of": AS_OF,
    }


def _price_plan(**changes):
    return ConditionalEntryPlan(
        kind=ConditionalEntryPlanKind.VOLUME_BREAKOUT,
        data_cutoff=pd.Timestamp(changes.pop("data_cutoff", AS_OF)),
        horizon="continuous_daily_weekly_v1",
        sessions=20,
        trigger_price=10.0,
        invalidation_price=9.4,
        maximum_entry_price=10.21,
        initial_risk_qualified=True,
        **changes,
    )


def _new_candidate(**changes):
    dates = pd.bdate_range(end=AS_OF, periods=200)
    return SimpleNamespace(
        **(
            {
                "symbol": "601999",
                "name": "合成替补",
                "industry": "new-industry",
                "signal_score": 0.9,
                "returns": pd.Series(
                    np.random.default_rng(9).normal(0.004, 0.01, 200), index=dates
                ),
                "price_observation_plan": _price_plan(),
            }
            | changes
        )
    )


def test_snapshot_marks_fixed_shares_and_cash_without_rebalancing_or_using_cost():
    args = _fixture(2)
    portfolio = args["portfolio"]
    snap = portfolio.metadata["account_snapshot"]
    anchor = AS_OF - timedelta(days=1)
    snap.update(
        as_of=anchor.isoformat(),
        account_weights={"600000": 0.2, "600001": 0.3},
        reference_prices={"600000": 10.0, "600001": 10.0},
        cash_weight=0.5,
    )
    histories = {
        "600000": pd.DataFrame({"trade_date": [anchor, AS_OF], "close": [10.0, 12.0]}),
        "600001": pd.DataFrame({"trade_date": [anchor, AS_OF], "close": [10.0, 8.0]}),
    }
    original = deepcopy(portfolio.metadata)
    weights, cash = service.mark_locked_account_weights(
        portfolio, histories, as_of=AS_OF, review=args["review"]
    )
    assert weights == pytest.approx({"600000": 0.24 / 0.98, "600001": 0.24 / 0.98})
    assert cash == pytest.approx(0.5 / 0.98)
    assert sum(weights.values()) + cash == pytest.approx(1.0)
    assert portfolio.metadata == original


@pytest.mark.parametrize(
    "change",
    [
        {"user_confirmed": False},
        {"no_external_flows_since_snapshot": False},
        {"as_of": "2026-09-05"},
        {"account_weights": {"600000": 0.1}},
        {"reference_prices": {}},
        {"cash_weight": 0.9},
    ],
)
def test_snapshot_missing_membership_or_cash_flow_evidence_is_rejected(change):
    args = _fixture()
    args["portfolio"].metadata["account_snapshot"].update(change)
    with pytest.raises((ValueError, KeyError, TypeError)):
        service.mark_locked_account_weights(
            args["portfolio"], args["histories"], as_of=AS_OF, review=args["review"]
        )


@pytest.mark.parametrize(
    "change",
    [
        {"company_action_clear": False},
        {"company_action_clear": None},
        {"company_action_clear_through": AS_OF - timedelta(days=1)},
        {"company_action_clear_through": None},
    ],
)
def test_drift_requires_company_action_clearance_through_current_cutoff(change):
    args = _fixture()
    for key, value in change.items():
        setattr(args["review"].rows[0], key, value)
    with pytest.raises(ValueError, match="corporate_action"):
        service.mark_locked_account_weights(
            args["portfolio"], args["histories"], as_of=AS_OF, review=args["review"]
        )


@pytest.mark.parametrize(
    "failure",
    [
        "anchor_before_entry",
        "missing_anchor",
        "bad_reference",
        "missing_current",
        "duplicate_current",
    ],
)
def test_snapshot_uncovered_or_unverifiable_price_intervals_are_rejected(failure):
    args = _fixture()
    snap = args["portfolio"].metadata["account_snapshot"]
    frame = args["histories"]["600000"]
    if failure == "anchor_before_entry":
        snap["as_of"] = (AS_OF - timedelta(days=101)).isoformat()
    elif failure == "missing_anchor":
        snap["as_of"] = (AS_OF - timedelta(days=5)).isoformat()  # Sunday.
    elif failure == "bad_reference":
        snap["reference_prices"]["600000"] *= 2
    elif failure == "missing_current":
        args["histories"]["600000"] = frame.iloc[:-1]
    else:
        args["histories"]["600000"] = pd.concat([frame, frame.iloc[[-1]]])
    with pytest.raises(ValueError):
        service.mark_locked_account_weights(
            args["portfolio"], args["histories"], as_of=AS_OF, review=args["review"]
        )


def test_joint_plan_locks_old_account_weights_and_does_not_reapply_new_entry_gate():
    args = _fixture()
    args["result"].qualified_entry_universe = (_new_candidate(),)
    original = deepcopy(args["portfolio"].metadata)
    plan = service.build_locked_replacement_plan(**args)
    assert plan["entries"][0]["symbol"] == "601999"
    assert plan["entries"][0]["entry_label"] == "确认≥10.00，买≤10.21+量"
    assert dict(plan["joint_evaluation"]["account_weights"]) | {"601999": 0} == {
        "600000": 0.15,
        "600001": 0.15,
        "600002": 0.15,
        "601999": 0,
    }
    assert args["portfolio"].metadata == original


@pytest.mark.parametrize("action", [HoldingAction.REDUCE, HoldingAction.REVIEW])
def test_reduce_or_unknown_review_blocks_new_purchase_before_selection(monkeypatch, action):
    args = _fixture()
    args["review"].rows[0].action = action
    monkeypatch.setattr(
        service,
        "select_continuous_replacement",
        lambda *_a, **_k: pytest.fail("must not run selector"),
    )
    plan = service.build_locked_replacement_plan(**args)
    assert plan["entries"] == []
    assert plan["cash_weight"] is None


@pytest.mark.parametrize(
    "field,value",
    [("portfolio_id", "other"), ("holding_version", 4), ("data_cutoff", AS_OF - timedelta(days=1))],
)
def test_review_identity_and_cutoff_must_match_exactly(monkeypatch, field, value):
    args = _fixture()
    setattr(args["review"], field, value)
    monkeypatch.setattr(
        service,
        "select_continuous_replacement",
        lambda *_a, **_k: pytest.fail("must not run selector"),
    )
    assert service.build_locked_replacement_plan(**args)["entries"] == []


def test_exit_is_only_a_contingent_replacement_and_never_a_recorded_sale():
    args = _fixture(4)
    args["review"].rows[0].action = HoldingAction.EXIT
    args["result"].qualified_entry_universe = (_new_candidate(),)
    positions_before = tuple(row.symbol for row in args["portfolio"].positions)
    plan = service.build_locked_replacement_plan(**args)
    assert plan["pending_exit_symbols"] == ["600000"]
    assert "先确认卖出" in plan["status_note"]
    assert plan["entries"][0]["symbol"] == "601999"
    assert tuple(row.symbol for row in args["portfolio"].positions) == positions_before
    assert not plan["joint_evaluation"]["holding_membership_changed"]


@pytest.mark.parametrize("count", [3, 5])
def test_no_candidate_or_five_intact_holdings_does_not_force_replacement(count):
    args = _fixture(count)
    plan = service.build_locked_replacement_plan(**args)
    assert plan["entries"] == []
    assert plan["cash_weight"] == pytest.approx(1 - count * 0.15)
    assert dict(plan["joint_evaluation"]["account_weights"]) == {
        f"60000{i}": 0.15 for i in range(count)
    }


def test_stale_research_result_cannot_supply_current_replacement():
    args = _fixture()
    args["result"].data_cutoff = pd.Timestamp(AS_OF - timedelta(days=1))
    args["result"].qualified_entry_universe = (_new_candidate(),)
    plan = service.build_locked_replacement_plan(**args)
    assert plan["entries"] == []
    assert plan["cash_weight"] is None


def test_stale_structured_entry_plan_cannot_be_self_certified_by_its_own_cutoff():
    args = _fixture()
    args["result"].qualified_entry_universe = (
        _new_candidate(
            price_observation_plan=_price_plan(data_cutoff=AS_OF - timedelta(days=1)),
        ),
    )
    with pytest.raises(ValueError, match="cutoff mismatch"):
        service.build_locked_replacement_plan(**args)


@pytest.mark.parametrize(
    "change",
    [
        {"company_action_clear_from": None},
        {"company_action_clear_from": AS_OF + timedelta(days=1)},
        {"position_key": "different-position"},
    ],
)
def test_clearance_must_cover_snapshot_interval_and_exact_holding_position(change):
    args = _fixture()
    for key, value in change.items():
        setattr(args["review"].rows[0], key, value)
    with pytest.raises(ValueError, match="interval_or_position"):
        service.mark_locked_account_weights(
            args["portfolio"], args["histories"], as_of=AS_OF, review=args["review"]
        )


def test_entry_formatter_cannot_return_empty_condition_or_reverse_price_interval():
    plan = _price_plan()
    for invalid in (replace(plan, sessions=5), replace(plan, maximum_entry_price=9.99)):
        with pytest.raises(ValueError, match="price condition unavailable"):
            service._entry("601999", "合成", 0.1, invalid, expected_cutoff=AS_OF)


def test_continuous_outer_builder_requests_one_frozen_profile_and_versions_only_new_report(
    monkeypatch,
):
    from test_evening_digest import CUTOFF, _hybrid, _result

    calls = []
    monkeypatch.setattr(service, "get_active_holding_portfolio", lambda _repo: None)

    def builder(_histories, _metadata, *, holding_weeks, **kwargs):
        calls.append((holding_weeks, kwargs.get("continuous_entry_policy")))
        return _result(holding_weeks)

    digest = service.build_continuous_research_digest(
        dataset_root="synthetic",
        overlay_root="synthetic",
        reference_dataset_root="synthetic",
        decision_date=CUTOFF,
        repository=object(),
        known_at=datetime.now(UTC),
        _hybrid_loader=lambda *_a, **_k: _hybrid(object()),
        _portfolio_builder=builder,
    )
    assert calls == [(4, True)]
    assert len(digest.periods) == 1
    assert digest.method_version == CONTINUOUS_METHOD_VERSION
    assert digest.continuous_plan["planned_exit_date"] is None
    assert digest.continuous_plan["holding_based"] is False
    calls.clear()
    legacy = build_evening_research_digest(
        dataset_root="synthetic",
        overlay_root="synthetic",
        reference_dataset_root="synthetic",
        decision_date=CUTOFF,
        _hybrid_loader=lambda *_a, **_k: _hybrid(object()),
        _portfolio_builder=builder,
    )
    assert {weeks for weeks, _ in calls} == {1, 2, 4, 13, 26, 52}
    assert all(policy is None for _, policy in calls)
    assert legacy.method_version != CONTINUOUS_METHOD_VERSION
    assert legacy.continuous_plan is None


def test_unreadable_ledger_is_not_interpreted_as_an_empty_initial_account(monkeypatch):
    from test_evening_digest import CUTOFF, _hybrid, _result

    def fail(_repository):
        raise OSError("synthetic ledger unavailable")

    monkeypatch.setattr(service, "get_active_holding_portfolio", fail)
    digest = service.build_continuous_research_digest(
        dataset_root="synthetic",
        overlay_root="synthetic",
        reference_dataset_root="synthetic",
        decision_date=CUTOFF,
        repository=object(),
        _hybrid_loader=lambda *_a, **_k: _hybrid(object()),
        _portfolio_builder=lambda _h, _m, **kwargs: _result(kwargs["holding_weeks"]),
    )
    assert digest.continuous_plan["entries"] == []
    assert digest.continuous_plan["cash_weight"] is None
    assert "读取失败" in digest.continuous_plan["status_note"]
