"""Pure presentation helpers for medium-term portfolio allocation.

The strategy intentionally distinguishes a risk-evaluated research allocation
from the allocation that currently passes every entry and evidence gate.  This
module keeps that distinction explicit before the values reach Streamlit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ashare_lab.services.build_midterm_portfolio import (
    CandidateAction,
    ConditionalEntryPlan,
    ConditionalEntryPlanKind,
    MidtermPortfolioResult,
)


@dataclass(frozen=True, slots=True)
class MidtermPositionView:
    """The three cells in the concise portfolio table."""

    stock_label: str
    planned_weight_label: str
    entry_price_condition: str


def build_midterm_position_views(
    result: MidtermPortfolioResult,
) -> tuple[MidtermPositionView, ...]:
    """Return rows keyed by symbol, never by rank or table position.

    The returned view is deliberately only three display cells: stock, planned
    allocation, and conditional entry price.  Exact-target and action-layer
    values are still validated internally, but are not leaked into this simple
    table.  No field is a brokerage account position or order.
    """

    research_exposure = _finite_weight(
        result.research_stock_exposure,
        field="research_stock_exposure",
    )
    action_exposure = _finite_weight(
        result.stock_exposure,
        field="stock_exposure",
    )
    action_by_symbol: dict[str, float] = {}
    for position in result.positions:
        if position.symbol in action_by_symbol:
            raise ValueError(f"duplicate action position: {position.symbol}")
        operational_weight = _optional_finite_weight(
            position.operational_account_weight,
            field=f"action_operational_weight:{position.symbol}",
        )
        if operational_weight is None:
            raise ValueError(f"missing action operational weight: {position.symbol}")
        action_sleeve_weight = _optional_finite_weight(
            position.operational_stock_sleeve_weight,
            field=f"action_operational_sleeve_weight:{position.symbol}",
        )
        _validate_operational_pair(
            operational_weight,
            action_sleeve_weight,
            exposure=action_exposure,
            field=f"action:{position.symbol}",
        )
        action_by_symbol[position.symbol] = _finite_weight(
            operational_weight,
            field=f"action_operational_weight:{position.symbol}",
        )
    rows: list[MidtermPositionView] = []
    exact_research_total = 0.0
    operational_research_total = 0.0
    observation_sleeve_total = 0.0
    for candidate in result.research_candidates:
        research_weight = _optional_finite_weight(
            candidate.research_weight,
            field=f"research_weight:{candidate.symbol}",
        )
        if research_weight is None or research_exposure <= 0.0:
            sleeve_weight = None
        else:
            sleeve_weight = research_weight / research_exposure
            if not math.isfinite(sleeve_weight) or not 0.0 <= sleeve_weight <= 1.0 + 1e-9:
                raise ValueError("research stock-sleeve weight is outside [0, 1]")
        operational_weight = _optional_finite_weight(
            candidate.operational_account_weight,
            field=f"research_operational_weight:{candidate.symbol}",
        )
        operational_sleeve_weight = _optional_finite_weight(
            candidate.operational_stock_sleeve_weight,
            field=f"research_operational_sleeve_weight:{candidate.symbol}",
        )
        if operational_weight is not None:
            _validate_operational_pair(
                operational_weight,
                operational_sleeve_weight,
                exposure=research_exposure,
                field=f"research:{candidate.symbol}",
            )
            operational_research_total += operational_weight
        elif operational_sleeve_weight is not None:
            raise ValueError(
                f"research:{candidate.symbol} has sleeve weight without account weight"
            )
        observation_sleeve_weight = _optional_finite_weight(
            getattr(candidate, "observation_stock_sleeve_weight", None),
            field=f"observation_sleeve_weight:{candidate.symbol}",
        )
        if observation_sleeve_weight is not None:
            if operational_weight is not None or operational_sleeve_weight is not None:
                raise ValueError(
                    f"research:{candidate.symbol} cannot mix risk-qualified and observation weights"
                )
            _validate_observation_sleeve_weight(
                observation_sleeve_weight,
                field=f"observation:{candidate.symbol}",
            )
            observation_sleeve_total += observation_sleeve_weight
        if research_weight is not None:
            exact_research_total += research_weight
        action_weight = action_by_symbol.get(candidate.symbol, 0.0)
        if action_weight > 0.0 and candidate.action is not CandidateAction.CONDITIONAL_ENTRY:
            raise ValueError(f"non-actionable candidate has action weight: {candidate.symbol}")
        rows.append(
            MidtermPositionView(
                stock_label=f"**{candidate.rank}. {candidate.name}**  \n`{candidate.symbol}`",
                planned_weight_label=_planned_weight_label(
                    operational_weight,
                    operational_sleeve_weight,
                    observation_sleeve_weight,
                ),
                entry_price_condition=_entry_price_condition_label(
                    candidate.action,
                    getattr(candidate, "conditional_entry_plan", None),
                    getattr(candidate, "price_observation_plan", None),
                    evidence_passed=not bool(getattr(candidate, "evidence_unknown", ())),
                    expected_cutoff=result.data_cutoff,
                    risk_qualified=operational_weight is not None,
                ),
            )
        )
    candidate_symbols = {candidate.symbol for candidate in result.research_candidates}
    if not set(action_by_symbol).issubset(candidate_symbols):
        raise ValueError("action positions must be present in research candidates")
    if operational_research_total > 0.0 and not math.isclose(
        operational_research_total,
        research_exposure,
        abs_tol=1e-9,
    ):
        raise ValueError("research operational weights do not sum to stock exposure")
    if exact_research_total > 0.0 and not math.isclose(
        exact_research_total,
        research_exposure,
        abs_tol=1e-9,
    ):
        raise ValueError("research exact targets do not sum to stock exposure")
    if observation_sleeve_total > 0.0 and not math.isclose(
        observation_sleeve_total,
        1.0,
        abs_tol=1e-9,
    ):
        raise ValueError("observation stock-sleeve weights do not sum to one")
    if action_by_symbol and not math.isclose(
        sum(action_by_symbol.values()),
        action_exposure,
        abs_tol=1e-9,
    ):
        raise ValueError("action operational weights do not sum to stock exposure")
    return tuple(rows)


def _planned_weight_label(
    account_weight: float | None,
    stock_sleeve_weight: float | None,
    observation_sleeve_weight: float | None,
) -> str:
    if account_weight is None or stock_sleeve_weight is None:
        if observation_sleeve_weight is not None:
            return f"候选观察配比 **{observation_sleeve_weight:.0%}**  \n非资金仓位｜风险门未通过"
        return "—（尚未形成风险合格权重）"
    return f"股票仓内 **{stock_sleeve_weight:.0%}**  \n对应总资金 **{account_weight:.1%}**"


def _entry_price_condition_label(
    action: CandidateAction,
    conditional_plan: ConditionalEntryPlan | None,
    observation_plan: ConditionalEntryPlan | None,
    *,
    evidence_passed: bool,
    expected_cutoff: object,
    risk_qualified: bool,
) -> str:
    unavailable = "—（暂未形成可介入价格）"
    if (
        action is CandidateAction.CONDITIONAL_ENTRY
        and evidence_passed
        and conditional_plan is not None
    ):
        label = _validated_price_plan_label(
            conditional_plan,
            expected_cutoff=expected_cutoff,
            observation=False,
        )
        if risk_qualified:
            return label
        return f"个股条件线：{label}  \n组合风险门未通过，不可据此买入"
    if observation_plan is None:
        return unavailable
    label = _validated_price_plan_label(
        observation_plan,
        expected_cutoff=expected_cutoff,
        observation=True,
    )
    prefix = "仅观察" if action is CandidateAction.OBSERVE_ONLY else "价格观察"
    return f"{prefix}：{label}  \n触及不等于可买，仍需完整复核"


def _validated_price_plan_label(
    plan: ConditionalEntryPlan,
    *,
    expected_cutoff: object,
    observation: bool,
) -> str:
    cutoff = getattr(plan, "data_cutoff", None)
    if expected_cutoff is not None and _normalize_timestamp(cutoff) != _normalize_timestamp(
        expected_cutoff
    ):
        raise ValueError("conditional entry plan cutoff does not match portfolio cutoff")
    if plan.horizon != "一周" or plan.sessions != 5:
        raise ValueError("conditional entry plan must use the one-week horizon")
    if plan.kind is ConditionalEntryPlanKind.HEALTHY_PULLBACK:
        low = _positive_price(plan.price_low, field="pullback_entry_low")
        high = _positive_price(plan.price_high, field="pullback_entry_high")
        if low > high:
            raise ValueError("pullback entry range is inverted")
        noun = "回踩观察区" if observation else "回踩至"
        return f"{noun} **{low:.2f}–{high:.2f}元**"
    trigger = _positive_price(plan.trigger_price, field="entry_trigger")
    if plan.kind is ConditionalEntryPlanKind.RECLAIM:
        if observation:
            return f"收盘站回观察线 ≥ **{trigger:.2f}元**"
        return f"收盘价 ≥ **{trigger:.2f}元**（重新站回确认）"
    if plan.kind is ConditionalEntryPlanKind.VOLUME_BREAKOUT:
        if observation:
            return f"收盘突破观察线 ≥ **{trigger:.2f}元**，且成交量不低于20日中位数1.2倍"
        return f"收盘价 ≥ **{trigger:.2f}元**（放量突破确认）"
    raise ValueError("unknown conditional entry plan kind")


def _normalize_timestamp(value: object) -> pd.Timestamp | None:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if timestamp.tz is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _positive_price(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive finite price")
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive finite price") from exc
    if not math.isfinite(price) or price <= 0.0:
        raise ValueError(f"{field} must be a positive finite price")
    return price


def _validate_operational_pair(
    account_weight: float,
    sleeve_weight: float | None,
    *,
    exposure: float,
    field: str,
) -> None:
    if sleeve_weight is None or exposure <= 0.0:
        raise ValueError(f"{field} operational weight requires positive exposure and sleeve weight")
    expected = account_weight / exposure
    if not math.isclose(sleeve_weight, expected, abs_tol=1e-9):
        raise ValueError(f"{field} account and sleeve weights disagree")
    grid_units = sleeve_weight / 0.10
    if not math.isclose(grid_units, round(grid_units), abs_tol=1e-9):
        raise ValueError(f"{field} stock-sleeve weight is not on the 10% grid")


def _validate_observation_sleeve_weight(weight: float, *, field: str) -> None:
    if weight <= 0.0:
        raise ValueError(f"{field} observation sleeve weight must be positive")
    grid_units = weight / 0.10
    if not math.isclose(grid_units, round(grid_units), abs_tol=1e-9):
        raise ValueError(f"{field} observation sleeve weight is not on the 10% grid")


def _optional_finite_weight(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _finite_weight(value, field=field)


def _finite_weight(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite weight")
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite weight") from exc
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return weight
