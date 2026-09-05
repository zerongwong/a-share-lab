"""One continuous research plan; no deadline, orders, or implicit ledger edits.

Daily/weekly formation and risk-observation windows are fixed, not user holding
deadlines. Existing holdings stay locked. An unconfirmed exit is a contingency,
never a fill. A complete, user-confirmed account snapshot is required to project
drifted old weights; stock-sleeve percentages alone cannot identify idle cash.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from ashare_lab.analytics.adaptive_portfolio import AdaptiveCandidate
from ashare_lab.analytics.continuous_portfolio import select_continuous_replacement
from ashare_lab.analytics.continuous_signals import CONTINUOUS_METHOD_VERSION
from ashare_lab.services.build_evening_digest import _format_plan, build_evening_research_digest
from ashare_lab.services.build_midterm_portfolio import (
    MidtermPortfolioStatus,
    _resolve_risk_budget,
    _returns_at_cutoff,
    build_midterm_portfolio,
)
from ashare_lab.services.holding_ledger import (
    get_active_holding_portfolio,
    holding_knowledge_context,
)
from ashare_lab.services.load_hybrid_universe import load_hybrid_universe
from ashare_lab.services.review_active_holdings import HoldingAction
from ashare_lab.services.run_active_holding_review import build_evening_holding_review


def build_continuous_research_digest(
    *,
    dataset_root,
    overlay_root,
    reference_dataset_root,
    decision_date: date,
    repository,
    known_at: datetime | None = None,
    _hybrid_loader=load_hybrid_universe,
    _portfolio_builder=build_midterm_portfolio,
    _holding_reviewer=build_evening_holding_review,
):
    """Keep legacy digest transport, but return one separately versioned plan."""
    captured: dict[str, Any] = {}

    def loader(*args, **kwargs):
        hybrid = _hybrid_loader(*args, **kwargs)
        captured["snapshot"] = hybrid.snapshot
        return hybrid

    def builder(*args, **kwargs):
        result = _portfolio_builder(*args, **kwargs, continuous_entry_policy=True)
        captured["result"] = result
        return result

    digest = build_evening_research_digest(
        dataset_root=dataset_root,
        overlay_root=overlay_root,
        reference_dataset_root=reference_dataset_root,
        decision_date=decision_date,
        horizons=(4,),
        _hybrid_loader=loader,
        _portfolio_builder=builder,
    )
    plan: dict[str, Any] = {
        "mode": "continuous",
        "method_version": CONTINUOUS_METHOD_VERSION,
        "planned_exit_date": None,
        "signal_profile": "daily_weekly_v1",
        "risk_observation_sessions": 20,
        "validation": "research_only_not_walk_forward_validated",
        "entries": [],
        "cash_weight": None,
        "holding_based": False,
        "status_note": "数据或市场证据不足，暂不生成新买计划。",
        "search_scope": "initial_top36_beam128;single_replacement_all_admitted_plus_cash",
    }
    result = captured.get("result")
    snapshot = captured.get("snapshot")
    # A ledger read error is not proof of an empty portfolio.
    try:
        portfolio = get_active_holding_portfolio(repository)
    except Exception:
        plan["status_note"] = "持仓登记读取失败，暂停新买，等待核验。"
        return replace(digest, method_version=CONTINUOUS_METHOD_VERSION, continuous_plan=plan)
    if portfolio is not None and portfolio.positions:
        plan["holding_based"] = True
        plan["holding_identity"] = [portfolio.id, portfolio.version]
    if result is None or snapshot is None or result.price_cycle is None:
        return replace(digest, method_version=CONTINUOUS_METHOD_VERSION, continuous_plan=plan)
    if portfolio is None or not portfolio.positions:
        # Absence of a registered portfolio means an illustrative initial plan,
        # NOT a claim that the user actually holds 100% cash.
        plan.update(_initial_plan(result))
        plan["status_note"] += " 未登记持仓时仅为初建研究方案。" if portfolio is None else ""
    else:
        try:
            known = known_at or datetime.now(UTC)
            context = holding_knowledge_context(portfolio, known_at=known)
            review = _holding_reviewer(
                repository,
                dataset_root=dataset_root,
                overlay_root=overlay_root,
                decision_date=digest.common_cutoff,
                reviewed_at=known,
                persist=False,
                holding_context=context,
            )
            plan.update(
                build_locked_replacement_plan(
                    result=result,
                    histories=snapshot.histories,
                    metadata=snapshot.metadata,
                    portfolio=portfolio,
                    review=review,
                    as_of=digest.common_cutoff,
                )
            )
        except (ValueError, TypeError, KeyError, OSError):
            plan["status_note"] = "旧仓或账户风险证据不完整；不猜仓位，暂停补位。"
    return replace(digest, method_version=CONTINUOUS_METHOD_VERSION, continuous_plan=plan)


def _entry(
    symbol: str, name: str, weight: float, price_plan, *, expected_cutoff: date
) -> dict[str, Any]:
    if price_plan is None or price_plan.initial_risk_qualified is not True:
        raise ValueError("qualified structured entry/stop required")
    if price_plan.invalidation_price is None or price_plan.maximum_entry_price is None:
        raise ValueError("entry ceiling and structural protection are mandatory")
    if pd.Timestamp(price_plan.data_cutoff).date() != expected_cutoff:
        raise ValueError("entry plan cutoff mismatch")
    label = _format_plan(
        price_plan, expected_cutoff=expected_cutoff, expected_sessions=20, observation=False
    )
    if not label or label.startswith("不入场"):
        raise ValueError("entry price condition unavailable")
    return {
        "symbol": symbol,
        "name": name,
        "account_weight": weight,
        "entry_qualified": True,
        "entry_label": label,
        "protection_line": price_plan.invalidation_price,
        "maximum_entry_price": price_plan.maximum_entry_price,
    }


def _initial_plan(result) -> dict[str, Any]:
    if result.status is not MidtermPortfolioStatus.RESEARCH_ONLY or not result.positions:
        return {
            "entries": [],
            "cash_weight": 1.0,
            "status_note": "暂无同时通过早期形态、证据和组合风险门的初建组合；暂不新买。",
        }
    entries = [
        _entry(
            row.symbol,
            row.name,
            row.operational_account_weight,
            row.conditional_entry_plan,
            expected_cutoff=pd.Timestamp(result.data_cutoff).date(),
        )
        for row in result.positions
    ]
    return {
        "entries": entries,
        "cash_weight": result.cash_weight,
        "status_note": "初建研究方案；次日仍需核对可成交性，超过买价上限不追。",
    }


def mark_locked_account_weights(portfolio, histories, *, as_of: date, review):
    """Mark fixed shares from one explicit whole-account snapshot, not cost.

    account_snapshot: as_of, account_weights, reference_prices, cash_weight,
    user_confirmed=True. Reconfirm after ANY actual trade/deposit/withdrawal.
    No external cash flows or corporate actions may be silently inferred away.
    """
    snap = portfolio.metadata.get("account_snapshot")
    if not isinstance(snap, Mapping) or snap.get("user_confirmed") is not True:
        raise ValueError("whole_account_snapshot_required")
    anchor = date.fromisoformat(snap["as_of"])
    if anchor > as_of or snap.get("no_external_flows_since_snapshot") is not True:
        raise ValueError("account_snapshot_or_flows_unknown")
    weights, refs = snap["account_weights"], snap["reference_prices"]
    symbols = {row.symbol for row in portfolio.positions}
    if set(weights) != symbols or set(refs) != symbols:
        raise ValueError("snapshot_membership_mismatch")
    cash = _fraction(snap["cash_weight"])
    initial = {symbol: _fraction(weights[symbol]) for symbol in symbols}
    if not math.isclose(sum(initial.values()) + cash, 1.0, abs_tol=1e-9):
        raise ValueError("account_weights_and_cash_must_sum_to_one")
    rows = {row.symbol: row for row in review.rows}
    values = {}
    for symbol in sorted(symbols):
        row = rows[symbol]
        if (
            row.company_action_clear is not True
            or row.company_action_clear_through is None
            or row.company_action_clear_through < as_of
        ):
            raise ValueError("corporate_action_clearance_required_for_drift")
        covered_from = getattr(row, "company_action_clear_from", None)
        if (
            covered_from is None
            or covered_from > anchor
            or row.position_key
            != next(item.position_key for item in portfolio.positions if item.symbol == symbol)
        ):
            raise ValueError("corporate_action_interval_or_position_identity_unknown")
        # Clearance supporting holding stops starts at entry. The snapshot may
        # not predate entry, otherwise that interval is not covered.
        holding = next(item for item in portfolio.positions if item.symbol == symbol)
        if anchor < holding.entry_date:
            raise ValueError("snapshot_predates_holding_entry")
        frame = histories[symbol]
        dates = pd.to_datetime(frame["trade_date"], errors="raise").dt.date
        anchor_close = frame.loc[dates == anchor, "close"]
        current = frame.loc[dates == as_of, "close"]
        ref = float(refs[symbol])
        if len(current) != 1 or len(anchor_close) != 1 or not math.isfinite(ref) or ref <= 0:
            raise ValueError("verified_snapshot_and_current_closes_required")
        if not math.isclose(ref, float(anchor_close.iloc[0]), rel_tol=1e-6):
            raise ValueError("snapshot_reference_price_mismatch")
        price = float(current.iloc[0])
        if not math.isfinite(price) or price <= 0:
            raise ValueError("invalid_current_close")
        values[symbol] = initial[symbol] * price / ref
    equity = sum(values.values()) + cash
    if equity <= 0:
        raise ValueError("invalid_account_equity")
    return {symbol: value / equity for symbol, value in values.items()}, cash / equity


def build_locked_replacement_plan(*, result, histories, metadata, portfolio, review, as_of):
    """Evaluate the entire retained set plus ONE eligible replacement or cash."""
    blocked = {
        "entries": [],
        "cash_weight": None,
        "status_note": "持仓或风险证据待核验，暂不补位；未确认的卖出不视为成交。",
    }
    if result.data_cutoff is None or pd.Timestamp(result.data_cutoff).date() != as_of:
        return blocked
    if (review.portfolio_id, review.holding_version) != (portfolio.id, portfolio.version):
        return blocked
    if {row.symbol for row in review.rows} != {row.symbol for row in portfolio.positions}:
        return blocked
    if len(review.rows) != len(portfolio.positions) or any(
        row.holding_version != portfolio.version for row in review.rows
    ):
        return blocked
    if review.data_cutoff != as_of:
        return blocked
    if any(row.action in {HoldingAction.REDUCE, HoldingAction.REVIEW} for row in review.rows):
        return blocked
    try:
        weights, cash = mark_locked_account_weights(
            portfolio, histories, as_of=as_of, review=review
        )
    except (KeyError, TypeError, ValueError):
        return {
            **blocked,
            "status_note": "缺少完整账户基准快照或除权核验：暂不能可靠计算补位比例；旧仓不自动改动。",
        }
    exits = {row.symbol for row in review.rows if row.action is HoldingAction.EXIT}
    for symbol in exits:
        cash += weights.pop(symbol)
    if not weights:
        plan = _initial_plan(result)
        plan["status_note"] = (
            "仅在原仓卖出已确认后评估初建；当前不视为已空仓。 " + plan["status_note"]
        )
        return {**plan, "pending_exit_symbols": sorted(exits)}
    if any(
        not isinstance(metadata[symbol].get("industry"), str)
        or not metadata[symbol]["industry"].strip()
        for symbol in weights
    ):
        return {**blocked, "status_note": "旧仓行业信息缺失，不能确认组合分散风险，暂不补位。"}
    retained = tuple(
        AdaptiveCandidate(
            symbol=symbol,
            industry=metadata[symbol]["industry"],
            signal_score=0.5,
            returns=_returns_at_cutoff(histories[symbol], pd.Timestamp(as_of)),
        )
        for symbol in sorted(weights)
    )
    admitted = {
        row.symbol: row
        for row in result.qualified_entry_universe
        if row.symbol not in {pos.symbol for pos in portfolio.positions}
    }
    replacements = tuple(
        AdaptiveCandidate(
            symbol=row.symbol,
            industry=row.industry,
            signal_score=row.signal_score,
            returns=row.returns,
        )
        for row in admitted.values()
    )
    budget = _resolve_risk_budget(result.price_cycle, None, holding_sessions=20)
    decision = select_continuous_replacement(
        retained, weights, replacements, cash_weight=cash, budget=budget
    )
    entries = []
    if decision.selected_symbol is not None:
        row = admitted[decision.selected_symbol]
        entries.append(
            _entry(
                row.symbol,
                row.name,
                decision.new_account_weight,
                row.price_observation_plan,
                expected_cutoff=as_of,
            )
        )
    note = "保留旧仓不再平衡；本轮没有优于留现金且风险合格的替补。"
    if entries:
        note = "补位按旧仓＋新股联合比较；超过买价上限不追。"
    if str(decision.status) == "review_required":
        note = "旧仓组合风险超限或证据不足，先复核风险，不新增仓位。"
    if exits:
        note = "先确认卖出并复核实际可用现金，再考虑以下补位。 " + note
    joint = asdict(decision)
    joint["data_cutoff"] = (
        None if decision.data_cutoff is None else decision.data_cutoff.isoformat()
    )
    return {
        "entries": entries,
        "cash_weight": decision.cash_weight,
        "status_note": note,
        "pending_exit_symbols": sorted(exits),
        "joint_evaluation": joint,
    }


def _fraction(value):
    if isinstance(value, bool):
        raise ValueError("invalid account fraction")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("invalid account fraction")
    return number
