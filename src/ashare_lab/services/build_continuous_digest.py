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
from ashare_lab.analytics.continuous_signals import (
    CONTINUOUS_METHOD_VERSION,
    CONTINUOUS_SIGNAL_CONTRACT,
)
from ashare_lab.analytics.portfolio_count_policy import (
    CONTINUOUS_COUNT_POLICY_VERSION,
    MAX_CONTINUOUS_HOLDINGS,
    MAX_NEW_ACCOUNT_WEIGHT,
    NORMAL_NEW_ACCOUNT_WEIGHT,
    continuous_count_state,
)
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
    _evidence_resolver=None,
):
    """Keep legacy digest transport, but return one separately versioned plan."""
    captured: dict[str, Any] = {}

    def loader(*args, **kwargs):
        hybrid = _hybrid_loader(*args, **kwargs)
        captured["snapshot"] = hybrid.snapshot
        return hybrid

    def builder(*args, **kwargs):
        def resolve_evidence(metadata, *, cutoff):
            from ashare_lab.adapters.csmar_local import infer_a_share_exchange
            from ashare_lab.bootstrap import application_data_dir
            from ashare_lab.ports.market_data import normalize_symbol
            from ashare_lab.services.candidate_evidence import enrich_candidate_evidence

            identities = {}
            canonical_metadata = {}
            for symbol, item in metadata.items():
                code = normalize_symbol(symbol)
                exchange = str(item.get("exchange") or infer_a_share_exchange(code) or "")
                canonical = f"{code}.{exchange}"
                if exchange not in {"SH", "SZ", "BJ"} or canonical in identities:
                    raise ValueError("candidate evidence identity is ambiguous")
                identities[canonical] = symbol
                canonical_metadata[canonical] = item
            batch = (_evidence_resolver or enrich_candidate_evidence)(
                canonical_metadata, symbols=tuple(canonical_metadata), cutoff=cutoff,
                knowledge_time=known_at or datetime.now(UTC),
                cache_dir=application_data_dir() / "cache" / "candidate_evidence",
                review_dir=application_data_dir() / "candidate_reviews",
                total_timeout_seconds=75,
            )
            captured["evidence_diagnostics"] = batch.diagnostics
            captured["candidate_evidence_records"] = [
                {
                    "symbol": row.symbol,
                    "fundamental_gate": row.fundamental_gate,
                    "announcement_gate": row.announcement_gate,
                    "execution_gate": row.execution_gate,
                    "fundamental_reasons": list(row.fundamental_reasons),
                    "announcement_reasons": list(row.announcement_reasons),
                    "execution_reasons": list(row.execution_reasons),
                    "financial_hash": row.financial_hash,
                    "announcement_manifest_hash": row.announcement_manifest_hash,
                    "execution_hash": row.execution_hash,
                    "retrieved_at": row.retrieved_at,
                    "financial_period": row.financial_period,
                    "publication_dates": list(row.publication_dates),
                }
                for row in batch.results
            ]
            return {identities[key]: item for key, item in batch.metadata.items() if key in identities}

        result = _portfolio_builder(
            *args, **kwargs, continuous_entry_policy=True,
            candidate_evidence_resolver=resolve_evidence,
        )
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
        "signal_profile": CONTINUOUS_SIGNAL_CONTRACT.label,
        "risk_observation_sessions": 20,
        "validation": "research_only_not_walk_forward_validated",
        "entries": [],
        "cash_weight": None,
        "holding_count": None,
        "count_state": "unknown",
        "count_policy_version": CONTINUOUS_COUNT_POLICY_VERSION,
        "normal_new_account_weight": NORMAL_NEW_ACCOUNT_WEIGHT,
        "maximum_new_account_weight": MAX_NEW_ACCOUNT_WEIGHT,
        "replacement_account_weight_options": [0.10, 0.20],
        "holding_based": False,
        "status_note": "数据或市场证据不足，暂不生成新买计划。",
        "search_scope": (
            "initial_compare_all_0_to_5_under_joint_risk;"
            "single_replacement_all_admitted_plus_cash;one_stock_per_industry"
        ),
    }
    result = captured.get("result")
    snapshot = captured.get("snapshot")
    plan["candidate_evidence"] = captured.get("evidence_diagnostics", {})
    plan["candidate_evidence_records"] = captured.get("candidate_evidence_records", [])
    plan["screening_candidates"] = _screening_candidates(result)
    _describe_candidate_evidence(plan["screening_candidates"], plan["candidate_evidence_records"])
    plan["screening_candidate_count"] = getattr(result, "horizon_candidate_count", 0)
    # Independent daily audit only: its cap/state NEVER feeds selection, the
    # holding ledger, protective stops, or the externally rendered plan.
    try:
        from ashare_lab.services.marks_cycle_shadow import run_marks_cycle_shadow

        plan["marks_cycle_shadow"] = run_marks_cycle_shadow(
            repository,
            price_cutoff=digest.common_cutoff,
            known_at=known_at or datetime.now(UTC),
            incumbent_cap=digest.max_stock_exposure,
        )
    except Exception:
        plan["marks_cycle_shadow"] = {
            "state": "shadow_unavailable",
            "mode": "shadow_only",
            "production_decision_input": False,
        }
    # A ledger read error is not proof of an empty portfolio.
    try:
        portfolio = get_active_holding_portfolio(repository)
    except Exception:
        plan["status_note"] = "持仓登记读取失败，暂停新买，等待核验。"
        return replace(digest, method_version=CONTINUOUS_METHOD_VERSION, continuous_plan=plan)
    if portfolio is not None:
        if portfolio.positions:
            plan["holding_based"] = True
            plan["holding_identity"] = [portfolio.id, portfolio.version]
            if len(portfolio.positions) <= MAX_CONTINUOUS_HOLDINGS:
                plan.update(_count_metadata(len(portfolio.positions)))
            else:
                plan["status_note"] = "登记持仓超过五只，须先复核组合结构；暂停新增。"
        else:
            plan.update(_count_metadata(0))
    if result is None or snapshot is None or result.price_cycle is None:
        return replace(digest, method_version=CONTINUOUS_METHOD_VERSION, continuous_plan=plan)
    if portfolio is None or not portfolio.positions:
        # Absence of a registered portfolio means an illustrative initial plan,
        # NOT a claim that the user actually holds 100% cash.
        plan.update(_initial_plan(result))
        evidence = plan["candidate_evidence"]
        if evidence and not plan["entries"]:
            plan["status_note"] = (
                f"本轮核验{evidence.get('requested_count', 0)}只周线候选："
                f"财务通过{evidence.get('financial_pass_count', 0)}只，"
                f"双重确认{evidence.get('double_confirmation_count', 0)}只，"
                f"待核验{evidence.get('unknown_count', 0)}只，"
                f"排除{evidence.get('veto_count', 0)}只。"
                + (
                    "证据尚未齐备，暂不新买。"
                    if evidence.get("unknown_count", 0)
                    else (
                        "财务或公告核验未通过，暂不新买。"
                        if not evidence.get("double_confirmation_count", 0)
                        else "可买性、买入位置或组合风险未通过，暂不新买。"
                    )
                )
            )
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
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or not 0 < float(weight) <= MAX_NEW_ACCOUNT_WEIGHT
    ):
        raise ValueError("entry account weight must be positive and at most 20%")
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
        "initial_risk_policy": price_plan.initial_risk_policy,
        "entry_label": label,
        "protection_line": (
            price_plan.effective_initial_protection_price or price_plan.invalidation_price
        ),
        "structural_invalidation_price": price_plan.invalidation_price,
        "planned_cost_stop_price": price_plan.planned_cost_stop_price,
        "cost_stop_basis": "recalculate_from_user_confirmed_actual_cost_after_fill",
        "maximum_entry_price": price_plan.maximum_entry_price,
    }


def _initial_plan(result) -> dict[str, Any]:
    if result.status is not MidtermPortfolioStatus.RESEARCH_ONLY or not result.positions:
        screened = _screening_candidates(result)
        count = getattr(result, "horizon_candidate_count", len(screened))
        if result.status is MidtermPortfolioStatus.DATA_NOT_READY:
            note = "数据核验未完成，尚不能判断是否存在合格股票；暂停新买。"
        elif count:
            note = f"已找到{count}只确认形态候选；买点风险、证据或组合门尚未全部通过，暂不新买。"
        else:
            note = "本轮没有通过已完成周线突破及后续核验的股票，暂不新买。"
        return {
            "entries": [],
            "cash_weight": None if result.status is MidtermPortfolioStatus.DATA_NOT_READY else 1.0,
            **_count_metadata(0),
            "status_note": note,
        }
    industries = tuple(
        row.industry.strip() if isinstance(row.industry, str) else ""
        for row in result.positions
    )
    if (
        len(result.positions) > MAX_CONTINUOUS_HOLDINGS
        or any(not industry for industry in industries)
        or len(set(industries)) != len(industries)
    ):
        return {
            "entries": [],
            "cash_weight": 1.0,
            **_count_metadata(0),
            "status_note": "组合数量或行业分散核验未通过；不降低门槛、不凑数，暂不新买。",
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
    count = len(entries)
    state = continuous_count_state(count)
    note = "初建成型组合；次日仍需核对可成交性，超过买价上限不追。"
    if state == "concentrated":
        note = "初建低仓集中组合；已与其他数量方案比较，剩余资金留现金。"
    return {
        "entries": entries,
        "cash_weight": result.cash_weight,
        **_count_metadata(count),
        "status_note": note,
    }


def _count_metadata(count: int) -> dict[str, Any]:
    return {"holding_count": count, "count_state": continuous_count_state(count)}


def _describe_candidate_evidence(candidates, records) -> None:
    """Name the missing evidence instead of calling a pipeline gap no signal."""
    indexed = {row["symbol"].split(".")[0]: row for row in records}
    for candidate in candidates:
        record = indexed.get(candidate["symbol"].split(".")[0])
        if record is None or candidate["reason"] != "财务、公告或可成交性待核验":
            continue
        parts = []
        if record["fundamental_gate"] == "pass":
            parts.append("财务初筛通过")
        elif "FINANCIAL_SECTOR_SPECIFIC_REVIEW_REQUIRED" in record["fundamental_reasons"]:
            parts.append("金融行业财务待专项核验")
        else:
            parts.append("财务证据待补齐")
        if record["announcement_gate"] != "pass":
            parts.append(
                "公告正文待审"
                if "OFFICIAL_DOCUMENT_CONTENT_REVIEW_REQUIRED" in record["announcement_reasons"]
                else "公告证据待补齐"
            )
        if record["execution_gate"] != "pass":
            parts.append("可交易性待核验")
        candidate["reason"] = "；".join(parts)


def _screening_candidates(result) -> list[dict[str, Any]]:
    """Expose up to five ranked formations without assigning an entry or weight."""
    candidates = getattr(result, "screening_candidates", ()) or getattr(
        result, "research_candidates", ()
    )
    rows = []
    for candidate in candidates[:5]:
        price = candidate.price_observation_plan
        risk = None if price is None else price.initial_risk_fraction
        if price is None:
            reason = "保护线证据不足"
        elif (
            price.initial_risk_policy == "legacy_structure_distance_8pct"
            and risk is not None and math.isfinite(risk) and risk > 0.08 + 1e-12
        ):
            reason = f"结构风险{risk:.1%}，超过8%"
        elif price.initial_risk_qualified is not True:
            reason = "买价与保护线尚不匹配"
        elif candidate.evidence_unknown:
            reason = "财务、公告或可成交性待核验"
        elif not candidate.risk_history_available:
            reason = "风险计算历史不足"
        elif candidate.action.value != "conditional_entry":
            reason = "当前周期的量价确认尚不足"
        else:
            reason = "个股门已通过，须结合组合与当日买价"
        pattern = candidate.timeframe.structure.state.value if candidate.timeframe else ""
        rows.append({
            "symbol": candidate.symbol,
            "name": candidate.name,
            "formation": {
                "volume_confirmed_breakout": "确认突破",
                "healthy_post_breakout_pullback": "健康回踩",
            }.get(pattern, "形态待核验"),
            "reason": reason,
            "entry_qualified": False,
        })
    return rows


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
        "holding_count": None,
        "count_state": "unknown",
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
        **_count_metadata(len(decision.account_weights)),
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
