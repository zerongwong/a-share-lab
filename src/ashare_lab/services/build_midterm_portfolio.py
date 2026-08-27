"""Build one adaptive 3--5 stock medium-term A-share research portfolio.

This is the main strategy service for the post-prototype model.  It keeps the
legacy fixed-four builder intact for archive compatibility, but it does not use
its composite factor score or fixed 3:2:2:1 allocation.

The service is deliberately deterministic:

* market evidence decides whether new entries are allowed;
* price/turnover structure is a hard candidate gate;
* fundamentals and official announcements can veto, never boost ranking;
* every 3-, 4-, and 5-stock set is evaluated with the same downside-risk
  engine; a bounded beam is a documented computational approximation;
* only risk-budget-passing sets with a positive historical holding-period lower
  bound may be returned, and the result remains ``RESEARCH_ONLY`` until genuine
  point-in-time walk-forward validation exists.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from ashare_lab.analytics.adaptive_portfolio import (
    AdaptiveCandidate,
    AdaptivePortfolioDataError,
    AdaptivePortfolioEvaluation,
    AdaptiveRiskBudget,
    optimize_adaptive_portfolio,
)
from ashare_lab.analytics.entry_readiness import (
    EntryPattern,
    EntryReadinessAssessment,
    assess_entry_readiness,
)
from ashare_lab.analytics.index_regime import (
    IndexRegimeAssessment,
    IndexRegimeState,
    assess_index_regime,
)
from ashare_lab.analytics.market_regime import (
    MarketRegimeAssessment,
    MarketRegimeState,
    assess_market_regime,
)

RESEARCH_DISCLAIMER = (
    "本结果是共同截止日历史数据上的确定性研究筛选。持有期收益下界尚未经过严格的"
    "point-in-time walk-forward、费用、滑点和不可成交验证，不是未来收益预测、上涨概率、"
    "投资建议或最大回撤保证。"
)


class MidtermPortfolioStatus(StrEnum):
    DATA_NOT_READY = "data_not_ready"
    VALIDATION_NOT_READY = "validation_not_ready"
    NO_ELIGIBLE_PORTFOLIO = "no_eligible_portfolio"
    RESEARCH_ONLY = "research_only"


@dataclass(frozen=True, slots=True)
class CandidateExclusion:
    symbol: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MidtermCandidate:
    symbol: str
    name: str
    industry: str
    signal_score: float
    entry: EntryReadinessAssessment
    returns: pd.Series
    evidence_unknown: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MidtermSelectedPosition:
    rank: int
    symbol: str
    name: str
    industry: str
    weight: float
    signal_score: float
    entry_pattern: EntryPattern
    breakout_line: float
    days_since_breakout: int
    annual_downside_volatility: float
    downside_risk_contribution: float
    evidence_unknown: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MidtermPortfolioResult:
    status: MidtermPortfolioStatus
    data_cutoff: pd.Timestamp | None
    holding_weeks: int
    positions: tuple[MidtermSelectedPosition, ...] = ()
    stock_exposure: float = 0.0
    cash_weight: float = 1.0
    borrowed_weight: float = 0.0
    evaluation: AdaptivePortfolioEvaluation | None = None
    entry_ready_count: int = 0
    search_pool_count: int = 0
    evaluated_portfolio_count: int = 0
    exclusions: tuple[CandidateExclusion, ...] = ()
    market_regime: MarketRegimeAssessment | None = None
    index_regime: IndexRegimeAssessment | None = None
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence_review_required: bool = False
    method_version: str = "midterm-maintrend-v0.1.0"
    disclaimer: str = RESEARCH_DISCLAIMER


_MOMENTUM_WEIGHTS: dict[int, tuple[tuple[int, float], ...]] = {
    1: ((5, 0.50), (20, 0.30), (60, 0.20)),
    4: ((20, 0.50), (60, 0.35), (120, 0.15)),
    13: ((20, 0.20), (60, 0.50), (120, 0.30)),
    26: ((20, 0.10), (60, 0.35), (120, 0.55)),
    52: ((20, 0.10), (60, 0.25), (120, 0.65)),
}


def build_midterm_portfolio(
    histories: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    as_of: object,
    holding_weeks: int = 13,
    market_index_histories: Mapping[str, pd.DataFrame] | None = None,
    risk_budget: AdaptiveRiskBudget | None = None,
    candidate_pool_size: int = 36,
    beam_width: int = 128,
    minimum_universe_size: int = 1_000,
    require_index_regime: bool = True,
    minimum_balance_sheet_strength: float = 0.10,
    minimum_historical_return_lcb: float = 0.0,
) -> MidtermPortfolioResult:
    """Return one risk-budget-passing 3--5 stock research portfolio.

    Optional metadata keys ``fundamental_gate`` and ``announcement_gate`` use
    ``pass``, ``veto`` or ``unknown``.  The boolean aliases
    ``fundamental_veto`` and ``announcement_veto`` are also accepted.  Vetoes
    exclude before ranking.  Missing evidence is recorded on the finalist and
    never translated into a neutral score.
    """

    if holding_weeks not in _MOMENTUM_WEIGHTS:
        raise ValueError("holding_weeks must be one of 1, 4, 13, 26 or 52")
    if candidate_pool_size < 5:
        raise ValueError("candidate_pool_size must be at least five")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if minimum_universe_size < 3:
        raise ValueError("minimum_universe_size must be at least three")
    if not 0.0 <= minimum_balance_sheet_strength <= 1.0:
        raise ValueError("minimum_balance_sheet_strength must be in [0, 1]")
    if not math.isfinite(minimum_historical_return_lcb):
        raise ValueError("minimum_historical_return_lcb must be finite")

    cutoff = _normalize_cutoff(as_of)
    if cutoff is None:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.DATA_NOT_READY,
            data_cutoff=None,
            holding_weeks=holding_weeks,
            reasons=("invalid_data_cutoff",),
        )

    normalized_histories = _normalize_mapping(histories, "history")
    normalized_metadata = _normalize_mapping(metadata, "metadata")
    if not normalized_histories:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.DATA_NOT_READY,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            reasons=("empty_history_universe",),
        )
    if len(normalized_histories) < minimum_universe_size:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.DATA_NOT_READY,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            reasons=(
                f"only_{len(normalized_histories)}_symbols_in_universe;"
                f"minimum_{minimum_universe_size}_required",
            ),
        )

    market_regime = assess_market_regime(
        normalized_histories,
        cutoff,
        minimum_symbols=minimum_universe_size,
    )
    if market_regime.state == MarketRegimeState.UNAVAILABLE:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.DATA_NOT_READY,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            market_regime=market_regime,
            reasons=("market_regime_unavailable",),
        )
    if require_index_regime and not market_index_histories:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.DATA_NOT_READY,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            market_regime=market_regime,
            reasons=("core_index_regime_required_but_missing",),
        )
    index_regime = (
        None
        if market_index_histories is None
        else assess_index_regime(market_index_histories, cutoff)
    )
    if market_regime.state == MarketRegimeState.RISK_OFF:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            market_regime=market_regime,
            index_regime=index_regime,
            reasons=("market_regime_risk_off_new_entries_paused",),
        )
    if index_regime is not None and index_regime.state in {
        IndexRegimeState.RISK_OFF,
        IndexRegimeState.UNAVAILABLE,
    }:
        reason = (
            "core_index_regime_risk_off_new_entries_paused"
            if index_regime.state == IndexRegimeState.RISK_OFF
            else "core_index_regime_unavailable_new_entries_paused"
        )
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.DATA_NOT_READY,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            market_regime=market_regime,
            index_regime=index_regime,
            reasons=(reason,),
        )

    raw_candidates: list[tuple[str, str, str, EntryReadinessAssessment, pd.Series, tuple[str, ...]]]
    raw_candidates = []
    exclusions: list[CandidateExclusion] = []
    for symbol, frame in sorted(normalized_histories.items()):
        item = normalized_metadata.get(symbol)
        hard_reasons = _metadata_hard_gate(
            symbol,
            item,
            minimum_balance_sheet_strength=minimum_balance_sheet_strength,
        )
        if hard_reasons:
            exclusions.append(CandidateExclusion(symbol, hard_reasons))
            continue
        assert item is not None
        entry = assess_entry_readiness(frame, as_of=cutoff)
        if not entry.ready:
            exclusions.append(CandidateExclusion(symbol, entry.reasons))
            continue
        try:
            returns = _returns_at_cutoff(frame, cutoff)
        except ValueError as exc:
            exclusions.append(CandidateExclusion(symbol, (f"invalid_returns:{exc}",)))
            continue
        unknown = _unknown_evidence(item)
        raw_candidates.append(
            (
                symbol,
                str(item.get("name", symbol)).strip() or symbol,
                str(item.get("industry", "")).strip(),
                entry,
                returns,
                unknown,
            )
        )

    if len(raw_candidates) < 3:
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            entry_ready_count=len(raw_candidates),
            exclusions=tuple(exclusions),
            market_regime=market_regime,
            index_regime=index_regime,
            reasons=(f"only_{len(raw_candidates)}_entry_ready_candidates;minimum_three",),
        )

    candidates = _rank_candidates(raw_candidates, holding_weeks)
    search_pool = candidates[:candidate_pool_size]
    holding_sessions = holding_weeks * 5
    budget = risk_budget or AdaptiveRiskBudget(
        holding_period_sessions=holding_sessions,
        minimum_observations=max(160, holding_sessions * 8),
        minimum_holding_period_samples=8,
        holding_period_cost_rate=0.002,
    )
    if budget.holding_period_sessions != holding_sessions:
        raise ValueError("risk_budget.holding_period_sessions must equal holding_weeks * 5")

    evaluated = 0
    viable_by_count: dict[
        int,
        list[tuple[float, AdaptivePortfolioEvaluation, tuple[MidtermCandidate, ...]]],
    ] = {3: [], 4: [], 5: []}
    for stock_count in (3, 4, 5):
        if len(search_pool) < stock_count:
            continue
        for indices in _beam_candidate_sets(search_pool, stock_count, beam_width):
            selected = tuple(search_pool[index] for index in indices)
            try:
                adaptive = _evaluate_candidate_set(selected, budget)
            except AdaptivePortfolioDataError:
                continue
            evaluated += 1
            if not adaptive.risk_budget.passed:
                continue
            if adaptive.metrics.holding_period_return_lcb < minimum_historical_return_lcb:
                continue
            objective = adaptive.metrics.holding_period_return_lcb
            viable_by_count[stock_count].append((objective, adaptive, selected))

    if not any(viable_by_count.values()):
        return MidtermPortfolioResult(
            status=MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO,
            data_cutoff=cutoff,
            holding_weeks=holding_weeks,
            entry_ready_count=len(candidates),
            search_pool_count=len(search_pool),
            evaluated_portfolio_count=evaluated,
            exclusions=tuple(exclusions),
            market_regime=market_regime,
            index_regime=index_regime,
            reasons=("no_3_to_5_stock_set_passed_all_downside_risk_budgets",),
        )

    for rows in viable_by_count.values():
        rows.sort(key=_viable_sort_key)
    _, best, selected = _select_stock_count(viable_by_count)
    selected_by_symbol = {item.symbol: item for item in selected}
    ordered_positions = sorted(
        best.positions,
        key=lambda item: (-item.weight, -item.signal_score, item.symbol),
    )
    positions = tuple(
        MidtermSelectedPosition(
            rank=rank,
            symbol=position.symbol,
            name=selected_by_symbol[position.symbol].name,
            industry=position.industry,
            weight=position.weight,
            signal_score=position.signal_score,
            entry_pattern=selected_by_symbol[position.symbol].entry.pattern,
            breakout_line=float(selected_by_symbol[position.symbol].entry.breakout_line),
            days_since_breakout=int(selected_by_symbol[position.symbol].entry.days_since_breakout),
            annual_downside_volatility=position.annual_downside_volatility,
            downside_risk_contribution=position.downside_risk_contribution,
            evidence_unknown=selected_by_symbol[position.symbol].evidence_unknown,
        )
        for rank, position in enumerate(ordered_positions, start=1)
    )
    warnings: list[str] = [
        "组合收益下界是历史非重叠窗口代理，is_out_of_sample=false；正式验证前只作研究。"
    ]
    unknown_symbols = [position.symbol for position in positions if position.evidence_unknown]
    if unknown_symbols:
        warnings.append(
            "以下入选标的仍有财务或官方公告证据待补充，不应据此直接交易："
            + "、".join(unknown_symbols)
        )
    review_required = bool(unknown_symbols)
    return MidtermPortfolioResult(
        status=(
            MidtermPortfolioStatus.VALIDATION_NOT_READY
            if review_required
            else MidtermPortfolioStatus.RESEARCH_ONLY
        ),
        data_cutoff=cutoff,
        holding_weeks=holding_weeks,
        positions=positions,
        stock_exposure=best.stock_exposure,
        cash_weight=best.cash_weight,
        borrowed_weight=0.0,
        evaluation=best,
        entry_ready_count=len(candidates),
        search_pool_count=len(search_pool),
        evaluated_portfolio_count=evaluated,
        exclusions=tuple(exclusions),
        market_regime=market_regime,
        index_regime=index_regime,
        warnings=tuple(warnings),
        reasons=(
            ("provisional_positions_require_evidence_and_execution_review",)
            if review_required
            else ()
        ),
        evidence_review_required=review_required,
    )


def _normalize_cutoff(value: object) -> pd.Timestamp | None:
    try:
        cutoff = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(cutoff):
        return None
    if cutoff.tz is not None:
        cutoff = cutoff.tz_localize(None)
    return cutoff.normalize()


def _normalize_mapping(values: Mapping[str, Any], label: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_symbol, value in values.items():
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in normalized:
            raise ValueError(f"{label} symbols must be non-blank and unique")
        normalized[symbol] = value
    return normalized


def _metadata_hard_gate(
    symbol: str,
    item: Mapping[str, Any] | None,
    *,
    minimum_balance_sheet_strength: float,
) -> tuple[str, ...]:
    if item is None:
        return ("metadata_missing",)
    reasons: list[str] = []
    for flag in ("is_st", "is_delisting"):
        if flag not in item or not isinstance(item[flag], (bool, np.bool_)):
            reasons.append(f"{flag}_must_be_explicit_boolean")
    if reasons:
        return tuple(reasons)
    name = str(item.get("name", symbol)).replace(" ", "").upper()
    if bool(item["is_st"]) or name.startswith(("ST", "*ST")):
        reasons.append("st_stock_excluded")
    if bool(item["is_delisting"]) or "退" in name:
        reasons.append("delisting_stock_excluded")
    if item.get("is_suspended") is True:
        reasons.append("suspended_stock_excluded")
    if item.get("is_buyable_at_cutoff") is False:
        reasons.append("formation_unbuyable_at_cutoff")
    if item.get("is_limit_up_at_cutoff") is True:
        reasons.append("formation_limit_up_excluded_pending_execution_check")
    industry = str(item.get("industry", "")).strip()
    if not industry:
        reasons.append("industry_missing")
    if item.get("fundamental_veto") is True or _gate_value(item.get("fundamental_gate")) == "veto":
        reasons.append("fundamental_veto")
    if (
        item.get("announcement_veto") is True
        or _gate_value(item.get("announcement_gate")) == "veto"
    ):
        reasons.append("official_announcement_veto")
    balance = item.get("balance_sheet_strength_score")
    if balance is not None:
        try:
            score = float(balance)
        except (TypeError, ValueError):
            reasons.append("invalid_balance_sheet_strength")
        else:
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                reasons.append("invalid_balance_sheet_strength")
            elif score < minimum_balance_sheet_strength:
                reasons.append("balance_sheet_strength_veto")
    return tuple(reasons)


def _gate_value(value: object) -> str:
    return str(value).strip().lower() if value is not None else "unknown"


def _unknown_evidence(item: Mapping[str, Any]) -> tuple[str, ...]:
    unknown: list[str] = []
    if item.get("is_buyable_at_cutoff") is not True:
        unknown.append("formation_execution_evidence_unknown")
    if (
        item.get("fundamental_veto") is not True
        and _gate_value(item.get("fundamental_gate")) != "pass"
    ):
        unknown.append("fundamental_evidence_unknown")
    if (
        item.get("announcement_veto") is not True
        and _gate_value(item.get("announcement_gate")) != "pass"
    ):
        unknown.append("official_announcement_evidence_unknown")
    return tuple(unknown)


def _returns_at_cutoff(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    if "trade_date" not in frame or "close" not in frame:
        raise ValueError("trade_date_and_close_required")
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    if bool(dates.isna().any()) or bool(close.isna().any()):
        raise ValueError("invalid_date_or_close")
    clean = pd.Series(close.to_numpy(dtype=float), index=pd.DatetimeIndex(dates).normalize())
    clean = clean.loc[clean.index <= cutoff].sort_index()
    if clean.index.has_duplicates or clean.empty or clean.index[-1] != cutoff:
        raise ValueError("duplicate_or_stale_cutoff")
    returns = clean.pct_change(fill_method=None).dropna()
    if not bool(np.isfinite(returns.to_numpy()).all()) or bool((returns <= -1.0).any()):
        raise ValueError("invalid_simple_returns")
    return returns


def _rank_candidates(
    rows: list[tuple[str, str, str, EntryReadinessAssessment, pd.Series, tuple[str, ...]]],
    holding_weeks: int,
) -> list[MidtermCandidate]:
    raw: list[dict[str, float]] = []
    for _, _, _, entry, returns, _ in rows:
        close_proxy = (1.0 + returns).cumprod()
        features: dict[str, float] = {}
        for sessions, _weight in _MOMENTUM_WEIGHTS[holding_weeks]:
            features[f"momentum_{sessions}"] = (
                float(close_proxy.iloc[-1] / close_proxy.iloc[-sessions - 1] - 1.0)
                if len(close_proxy) > sessions
                else float("nan")
            )
        downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
        features["downside_volatility"] = float(
            np.sqrt(np.mean(np.square(downside))) * math.sqrt(252)
        )
        equity = (1.0 + returns).cumprod()
        features["max_drawdown"] = abs(float((equity / equity.cummax() - 1.0).min()))
        features["entry"] = entry.score
        raw.append(features)
    if any(not all(math.isfinite(value) for value in item.values()) for item in raw):
        raise ValueError("candidate ranking contains non-finite evidence")

    percentiles: dict[str, list[float]] = {}
    for key in raw[0]:
        values = [item[key] for item in raw]
        ranks = pd.Series(values).rank(method="average", pct=True).to_list()
        percentiles[key] = [float(value) for value in ranks]

    ranked: list[MidtermCandidate] = []
    for index, (symbol, name, industry, entry, returns, unknown) in enumerate(rows):
        momentum = sum(
            weight * percentiles[f"momentum_{sessions}"][index]
            for sessions, weight in _MOMENTUM_WEIGHTS[holding_weeks]
        )
        downside_quality = 1.0 - percentiles["downside_volatility"][index]
        drawdown_quality = 1.0 - percentiles["max_drawdown"][index]
        risk_quality = 0.60 * downside_quality + 0.40 * drawdown_quality
        signal = 0.55 * entry.score + 0.25 * momentum + 0.20 * risk_quality
        ranked.append(
            MidtermCandidate(
                symbol=symbol,
                name=name,
                industry=industry,
                signal_score=round(float(signal), 6),
                entry=entry,
                returns=returns,
                evidence_unknown=unknown,
            )
        )
    return sorted(ranked, key=lambda item: (-item.signal_score, item.symbol))


def _beam_candidate_sets(
    candidates: list[MidtermCandidate],
    stock_count: int,
    beam_width: int,
) -> tuple[tuple[int, ...], ...]:
    correlations: dict[tuple[int, int], float] = {}
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            aligned = pd.concat(
                (candidates[left].returns, candidates[right].returns),
                axis=1,
                join="inner",
            ).dropna()
            correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            correlations[(left, right)] = correlation if math.isfinite(correlation) else 1.0

    states: list[tuple[int, ...]] = [()]
    for _depth in range(stock_count):
        expanded: list[tuple[int, ...]] = []
        for state in states:
            start = state[-1] + 1 if state else 0
            remaining = stock_count - len(state) - 1
            stop = len(candidates) - remaining
            for index in range(start, stop):
                expanded.append((*state, index))
        expanded.sort(
            key=lambda state: (
                -_partial_set_score(state, candidates, correlations),
                tuple(candidates[index].symbol for index in state),
            )
        )
        states = expanded[:beam_width]
        if not states:
            break
    return tuple(states)


def _partial_set_score(
    indices: tuple[int, ...],
    candidates: list[MidtermCandidate],
    correlations: Mapping[tuple[int, int], float],
) -> float:
    base = float(np.mean([candidates[index].signal_score for index in indices]))
    industries = {candidates[index].industry for index in indices}
    diversity = len(industries) / len(indices)
    pairs = [
        correlations[(left, right)]
        for offset, left in enumerate(indices)
        for right in indices[offset + 1 :]
    ]
    correlation_penalty = max(0.0, float(np.mean(pairs))) if pairs else 0.0
    return base + 0.04 * diversity - 0.10 * correlation_penalty


def _evaluate_candidate_set(
    selected: tuple[MidtermCandidate, ...],
    budget: AdaptiveRiskBudget,
) -> AdaptivePortfolioEvaluation:
    aligned = pd.concat(
        {candidate.symbol: candidate.returns for candidate in selected},
        axis=1,
        join="inner",
    ).dropna()
    adaptive = tuple(
        AdaptiveCandidate(
            symbol=candidate.symbol,
            industry=candidate.industry,
            signal_score=candidate.signal_score,
            returns=aligned[candidate.symbol],
        )
        for candidate in selected
    )
    return optimize_adaptive_portfolio(adaptive, budget=budget)


def _viable_sort_key(
    row: tuple[float, AdaptivePortfolioEvaluation, tuple[MidtermCandidate, ...]],
) -> tuple[float, float, float, float, float, float, tuple[str, ...]]:
    objective, evaluation, selected = row
    metrics = evaluation.metrics
    return (
        -objective,
        metrics.annual_downside_volatility,
        metrics.rolling_max_drawdown_60_p90,
        metrics.es95_5d,
        metrics.max_down_period_correlation,
        metrics.max_position_downside_risk_contribution,
        tuple(sorted(item.symbol for item in selected)),
    )


def _select_stock_count(
    viable_by_count: Mapping[
        int,
        list[tuple[float, AdaptivePortfolioEvaluation, tuple[MidtermCandidate, ...]]],
    ],
) -> tuple[float, AdaptivePortfolioEvaluation, tuple[MidtermCandidate, ...]]:
    """Choose four normally; use three for scarcity and five for real diversification.

    A viable four-stock set is the baseline.  A five-stock set may replace it
    only when it has a higher conservative return lower bound, does not worsen
    the three path/tail risk measures by more than 50 basis points, and improves
    either down-period correlation by five points or maximum single-name risk
    contribution by three points.  Three stocks are selected only when no
    four-stock set passes every hard risk budget.  If a five-stock set is the
    only diversified feasible construction, it remains preferable to a
    concentrated three-stock fallback.
    """

    best_three = viable_by_count.get(3, [])[:1]
    best_four = viable_by_count.get(4, [])[:1]
    best_five = viable_by_count.get(5, [])[:1]
    if best_four:
        baseline = best_four[0]
        for five_stock_row in viable_by_count.get(5, []):
            if _five_stock_diversification_is_material(baseline[1], five_stock_row[1]):
                return five_stock_row
        return baseline
    if best_five:
        return best_five[0]
    if best_three:
        return best_three[0]
    raise RuntimeError("at least one viable 3-to-5 stock set is required")


def _five_stock_diversification_is_material(
    four: AdaptivePortfolioEvaluation,
    five: AdaptivePortfolioEvaluation,
) -> bool:
    four_metrics = four.metrics
    five_metrics = five.metrics
    if five_metrics.holding_period_return_lcb <= four_metrics.holding_period_return_lcb:
        return False
    path_and_tail_not_worse = (
        five_metrics.annual_downside_volatility <= four_metrics.annual_downside_volatility + 0.005
        and five_metrics.rolling_max_drawdown_60_p90
        <= four_metrics.rolling_max_drawdown_60_p90 + 0.005
        and five_metrics.es95_5d <= four_metrics.es95_5d + 0.005
    )
    diversification_improved = (
        five_metrics.max_down_period_correlation <= four_metrics.max_down_period_correlation - 0.05
        or five_metrics.max_position_downside_risk_contribution
        <= four_metrics.max_position_downside_risk_contribution - 0.03
    )
    return path_and_tail_not_worse and diversification_improved
