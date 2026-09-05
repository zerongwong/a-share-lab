"""Pure, constrained single-slot research comparison; never a holding-ledger writer.

Inputs must already have passed point-in-time market/entry/price-risk gates. This
module cannot verify those gates from returns alone. It locks retained *account*
weights, compares every supplied replacement at 10/20/30% account weight with
leaving that cash idle, and never rebalances a retained position. The historical
20-session LCB is only a proxy, not a dynamic-strategy backtest or future optimum.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from statistics import NormalDist

import numpy as np
import pandas as pd

from ashare_lab.analytics.adaptive_portfolio import (
    ANNUAL_SESSIONS,
    FIVE_DAY_SESSIONS,
    MINIMUM_FIVE_DAY_SAMPLES,
    MINIMUM_ROLLING_DRAWDOWN_WINDOWS,
    ROLLING_DRAWDOWN_SESSIONS,
    AdaptiveCandidate,
    AdaptivePortfolioDataError,
    AdaptiveRiskBudget,
    _downside_risk_contributions,
    _fixed_share_daily_returns,
    _fixed_share_rolling_drawdown_magnitudes,
    _non_overlapping_portfolio_returns,
)

CONTINUOUS_PORTFOLIO_METHOD_VERSION = "locked-holdings-single-replacement-lcb20-v0.1.0"
_PROXY_SESSIONS = 20
_NEW_WEIGHTS = (0.10, 0.20, 0.30)
_WARNINGS = (
    "research_only_not_future_return_or_global_optimum",
    "historical_fixed_shares_cash_20_session_proxy_not_dynamic_strategy_validation",
    "input_entry_gates_calendar_company_actions_and_cutoff_require_upstream_verification",
    "flat_historical_account_cost_not_incremental_order_cost_model",
    "no_trade_no_ledger_update_unconfirmed_sales_are_not_cash",
)


class ContinuousPortfolioStatus(StrEnum):
    SELECTED = "selected"
    HOLD_CASH = "hold_cash"
    REVIEW_REQUIRED = "review_required"


@dataclass(frozen=True, slots=True)
class ContinuousPortfolioMetrics:
    observation_count: int
    holding_period_sample_count: int
    annual_downside_volatility: float
    rolling_max_drawdown_60_p90: float
    horizon_rolling_max_drawdown_p90: float
    es95_5d: float
    max_down_period_correlation: float | None
    down_period_count: int
    max_position_downside_risk_contribution: float | None
    holding_period_return_mean: float
    holding_period_return_lcb: float
    holding_period_cost_rate: float
    lcb_confidence: float
    correlation_applicable: bool
    cash_only: bool = False
    holding_period_sessions: int = _PROXY_SESSIONS
    is_out_of_sample: bool = False
    path_method_version: str = "fixed-shares-plus-cash-per-window-v1.0.0"


@dataclass(frozen=True, slots=True)
class ContinuousCandidateRejection:
    symbol: str
    new_account_weight: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContinuousPortfolioDecision:
    status: ContinuousPortfolioStatus
    selected_symbol: str | None
    new_account_weight: float
    account_weights: tuple[tuple[str, float], ...]
    cash_weight: float
    metrics: ContinuousPortfolioMetrics | None
    baseline_metrics: ContinuousPortfolioMetrics | None
    reasons: tuple[str, ...]
    candidate_rejections: tuple[ContinuousCandidateRejection, ...]
    evaluated_count: int
    candidate_count: int
    data_cutoff: pd.Timestamp | None
    warnings: tuple[str, ...] = _WARNINGS
    method_version: str = CONTINUOUS_PORTFOLIO_METHOD_VERSION
    auto_order_allowed: bool = False
    holding_membership_changed: bool = False


def _number(value: object, *, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a finite numeric fraction")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite numeric fraction") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite numeric fraction")
    return number


def _history(candidate: AdaptiveCandidate, length: int) -> pd.Series:
    series = candidate.returns
    if not isinstance(series.index, pd.DatetimeIndex):
        raise AdaptivePortfolioDataError("verified_session_datetime_index_required")
    if (
        series.index.hasnans
        or series.index.has_duplicates
        or not series.index.is_monotonic_increasing
    ):
        raise AdaptivePortfolioDataError("invalid_session_index")
    if len(series) < length:
        raise AdaptivePortfolioDataError("insufficient_common_history")
    values = pd.to_numeric(series, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise AdaptivePortfolioDataError("nonfinite_return_history")
    if (values <= -1.0).any():
        raise AdaptivePortfolioDataError("invalid_simple_return")
    return values.iloc[-length:]


def _metrics(
    matrix: np.ndarray, weights: np.ndarray, budget: AdaptiveRiskBudget
) -> ContinuousPortfolioMetrics:
    if not len(weights):
        # Idle cash has deterministic zero price return, not an estimated equity
        # return, zero estimated beta, or a fabricated stock correlation.
        return ContinuousPortfolioMetrics(
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            None,
            0,
            None,
            0.0,
            0.0,
            0.0,
            budget.lcb_confidence,
            False,
            cash_only=True,
        )
    if np.any(np.mean(np.minimum(matrix, 0.0) ** 2, axis=0) <= 0.0):
        raise AdaptivePortfolioDataError("individual_downside_risk_unavailable")
    daily = _fixed_share_daily_returns(matrix, weights, _PROXY_SESSIONS)
    downside = np.minimum(daily, 0.0)
    annual = float(np.sqrt(np.mean(downside**2)) * math.sqrt(ANNUAL_SESSIONS))
    dd60 = _fixed_share_rolling_drawdown_magnitudes(matrix, weights, ROLLING_DRAWDOWN_SESSIONS)
    dd20 = _fixed_share_rolling_drawdown_magnitudes(matrix, weights, _PROXY_SESSIONS)
    five = _non_overlapping_portfolio_returns(matrix, weights, FIVE_DAY_SESSIONS)
    tail = five[five <= np.quantile(five, 0.05) + 1e-15]
    es = max(0.0, -float(tail.mean()))
    down = matrix[matrix.mean(axis=1) < 0.0]
    if len(down) < budget.minimum_down_periods:
        raise AdaptivePortfolioDataError("insufficient_down_periods")
    correlation = None
    if len(weights) > 1:
        if np.any(np.ptp(down, axis=0) <= 0.0):
            raise AdaptivePortfolioDataError("undefined_down_period_correlation")
        pairs = np.corrcoef(down, rowvar=False)[np.triu_indices(len(weights), k=1)]
        if not np.isfinite(pairs).all():
            raise AdaptivePortfolioDataError("undefined_down_period_correlation")
        correlation = float(pairs.max())
    contribution = float(_downside_risk_contributions(matrix, weights).max())
    holding = _non_overlapping_portfolio_returns(matrix, weights, _PROXY_SESSIONS)
    net = holding - budget.holding_period_cost_rate
    mean = float(net.mean())
    lcb = mean - NormalDist().inv_cdf(budget.lcb_confidence) * float(net.std(ddof=1)) / math.sqrt(
        len(net)
    )
    finite = (annual, es, contribution, mean, lcb, *dd60, *dd20)
    if not all(math.isfinite(value) for value in finite):
        raise AdaptivePortfolioDataError("nonfinite_portfolio_metrics")
    return ContinuousPortfolioMetrics(
        len(matrix),
        len(net),
        annual,
        float(np.quantile(dd60, 0.90)),
        float(np.quantile(dd20, 0.90)),
        es,
        correlation,
        len(down),
        contribution,
        mean,
        lcb,
        budget.holding_period_cost_rate,
        budget.lcb_confidence,
        len(weights) > 1,
    )


def _risk_reasons(
    metrics: ContinuousPortfolioMetrics, budget: AdaptiveRiskBudget
) -> tuple[str, ...]:
    horizon_limit = (
        budget.max_rolling_drawdown_60_p90
        if budget.max_horizon_rolling_drawdown_p90 is None
        else budget.max_horizon_rolling_drawdown_p90
    )
    checks = (
        (
            metrics.annual_downside_volatility,
            budget.max_annual_downside_volatility,
            "annual_downside_volatility",
        ),
        (metrics.horizon_rolling_max_drawdown_p90, horizon_limit, "horizon_rolling_drawdown_p90"),
        (metrics.es95_5d, budget.max_es95_5d, "es95_5d"),
        (
            metrics.max_down_period_correlation,
            budget.max_down_period_correlation,
            "down_period_correlation",
        ),
        (
            metrics.max_position_downside_risk_contribution,
            budget.max_position_downside_risk_contribution,
            "position_downside_risk_contribution",
        ),
    )
    return tuple(
        name for value, limit, name in checks if value is not None and value > limit + 1e-12
    )


def _structure_reasons(
    candidates: Sequence[AdaptiveCandidate],
    weights: Mapping[str, float],
    budget: AdaptiveRiskBudget,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if len(candidates) > 5:
        reasons.append("maximum_five_holdings")
    if any(weight > 0.30 + 1e-12 for weight in weights.values()):
        reasons.append("maximum_30pct_account_weight")
    industries: dict[str, float] = {}
    for candidate in candidates:
        if candidate.industry in industries:
            reasons.append("one_stock_per_industry")
        industries[candidate.industry] = (
            industries.get(candidate.industry, 0.0) + weights[candidate.symbol]
        )
    if any(
        weight > min(0.40, budget.industry_weight_limit) + 1e-12 for weight in industries.values()
    ):
        reasons.append("industry_concentration")
    exposure_limit = (
        0.85 if budget.maximum_stock_exposure is None else budget.maximum_stock_exposure
    )
    if math.fsum(weights.values()) > exposure_limit + 1e-12:
        reasons.append("maximum_stock_exposure")
    return tuple(dict.fromkeys(reasons))


def select_continuous_replacement(
    retained: Sequence[AdaptiveCandidate],
    retained_account_weights: Mapping[str, float],
    replacements: Sequence[AdaptiveCandidate],
    *,
    cash_weight: float,
    budget: AdaptiveRiskBudget,
) -> ContinuousPortfolioDecision:
    """Compare one replacement plus frozen holdings against frozen holdings/cash.

    Return histories must be upstream-verified simple returns on the same
    exchange-session chain. All compared paths use the same required trailing
    window, never a candidate-dependent shorter sample. This routine does not
    establish official calendar completeness or point-in-time corporate actions.
    ``cash_weight`` is confirmed, available cash; proposed sales are not cash.
    An invalid input allocation raises ValueError. Unverifiable retained risk
    returns REVIEW_REQUIRED; an invalid replacement is recorded and skipped.
    """
    if not isinstance(budget, AdaptiveRiskBudget):
        raise TypeError("budget must be AdaptiveRiskBudget")
    if budget.holding_period_sessions != _PROXY_SESSIONS:
        raise ValueError("continuous proxy requires an explicit 20-session risk budget")
    retained = tuple(retained)
    replacements = tuple(replacements)
    if any(not isinstance(item, AdaptiveCandidate) for item in retained):
        raise TypeError("retained holdings must be AdaptiveCandidate objects")
    if not isinstance(retained_account_weights, Mapping):
        raise TypeError("retained_account_weights must be a mapping")
    symbols = [candidate.symbol for candidate in retained]
    if len(set(symbols)) != len(symbols) or set(retained_account_weights) != set(symbols):
        raise ValueError("retained symbols and exact account weights must match uniquely")
    weights = {
        symbol: _number(retained_account_weights[symbol], name="account_weight")
        for symbol in sorted(symbols)
    }
    cash = _number(cash_weight, name="cash_weight")
    if any(value <= 0.0 or value > 1.0 for value in weights.values()) or not 0.0 <= cash <= 1.0:
        raise ValueError("cash and positive holding weights must be fractions of account equity")
    if not math.isclose(math.fsum(weights.values()) + cash, 1.0, abs_tol=1e-10, rel_tol=0.0):
        raise ValueError("retained account weights plus confirmed cash must equal one")
    retained = tuple(sorted(retained, key=lambda row: row.symbol))
    baseline: ContinuousPortfolioMetrics | None = None
    cutoff: pd.Timestamp | None = None
    rejections: list[ContinuousCandidateRejection] = []
    evaluated = 0

    def result(status, reasons, *, selected=None, added=0.0, metrics=None):
        final = weights | ({selected: added} if selected is not None else {})
        return ContinuousPortfolioDecision(
            status,
            selected,
            added,
            tuple(sorted(final.items())),
            cash - added,
            metrics if metrics is not None else baseline,
            baseline,
            tuple(reasons),
            tuple(rejections),
            evaluated,
            len(replacements),
            cutoff,
        )

    structural = _structure_reasons(retained, weights, budget)
    if structural:
        return result(
            ContinuousPortfolioStatus.REVIEW_REQUIRED,
            ("retained_portfolio_requires_review", *structural),
        )
    length = max(
        budget.minimum_observations,
        _PROXY_SESSIONS * budget.minimum_holding_period_samples,
        ROLLING_DRAWDOWN_SESSIONS + MINIMUM_ROLLING_DRAWDOWN_WINDOWS - 1,
        FIVE_DAY_SESSIONS * MINIMUM_FIVE_DAY_SAMPLES,
    )
    histories: dict[str, pd.Series] = {}
    reference: pd.DatetimeIndex | None = None
    try:
        for candidate in retained:
            series = _history(candidate, length)
            if reference is not None and not series.index.equals(reference):
                raise AdaptivePortfolioDataError("retained_history_not_aligned")
            reference = series.index
            histories[candidate.symbol] = series
        if reference is not None:
            cutoff = reference[-1]
        matrix = (
            np.column_stack([histories[item.symbol] for item in retained])
            if retained
            else np.empty((0, 0))
        )
        baseline = _metrics(matrix, np.asarray(list(weights.values())), budget)
    except (AdaptivePortfolioDataError, TypeError, ValueError):
        return result(
            ContinuousPortfolioStatus.REVIEW_REQUIRED, ("retained_risk_evidence_unavailable",)
        )
    evaluated += 1  # Cash/no-new-purchase is a real, eligible baseline.
    risk = _risk_reasons(baseline, budget)
    if risk:
        return result(
            ContinuousPortfolioStatus.REVIEW_REQUIRED, ("retained_portfolio_requires_review", *risk)
        )
    if len(retained) == 5:
        return result(ContinuousPortfolioStatus.HOLD_CASH, ("maximum_five_holdings_no_free_slot",))
    valid: list[AdaptiveCandidate] = []
    replacement_symbols = [row.symbol for row in replacements if isinstance(row, AdaptiveCandidate)]
    duplicates = {symbol for symbol, count in Counter(replacement_symbols).items() if count > 1}
    occupied_industries = {row.industry for row in retained}
    for candidate in replacements:
        symbol = (
            candidate.symbol if isinstance(candidate, AdaptiveCandidate) else "invalid_candidate"
        )
        reason = None
        if not isinstance(candidate, AdaptiveCandidate):
            reason = "invalid_candidate_type"
        elif symbol in duplicates:
            reason = "duplicate_replacement_symbol"
        elif symbol in weights:
            reason = "retained_position_not_an_addition_candidate"
        elif candidate.industry in occupied_industries:
            reason = "one_stock_per_industry"
        if reason is not None:
            rejections.append(ContinuousCandidateRejection(symbol, None, (reason,)))
            continue
        try:
            histories[symbol] = _history(candidate, length)
        except (AdaptivePortfolioDataError, TypeError, ValueError):
            rejections.append(
                ContinuousCandidateRejection(symbol, None, ("candidate_history_unavailable",))
            )
            continue
        valid.append(candidate)
    if reference is None and valid:
        # With no retained risk, use the most recent supplied complete window;
        # upstream still owns the verified market cutoff and official calendar.
        anchor = min(valid, key=lambda item: (-histories[item.symbol].index[-1].value, item.symbol))
        reference = histories[anchor.symbol].index
        cutoff = reference[-1]
    best_metrics = baseline
    best_symbol = None
    best_weight = 0.0
    for candidate in sorted(valid, key=lambda item: item.symbol):
        symbol = candidate.symbol
        if not histories[symbol].index.equals(reference):
            rejections.append(
                ContinuousCandidateRejection(symbol, None, ("candidate_history_not_aligned",))
            )
            continue
        combined = (*retained, candidate)
        matrix = np.column_stack([histories[item.symbol] for item in combined])
        for added in _NEW_WEIGHTS:
            if added > cash:
                rejections.append(
                    ContinuousCandidateRejection(symbol, added, ("insufficient_confirmed_cash",))
                )
                continue
            proposal = weights | {symbol: added}
            structural = _structure_reasons(combined, proposal, budget)
            if structural:
                rejections.append(ContinuousCandidateRejection(symbol, added, structural))
                continue
            try:
                metrics = _metrics(
                    matrix, np.asarray([proposal[item.symbol] for item in combined]), budget
                )
            except (AdaptivePortfolioDataError, TypeError, ValueError):
                rejections.append(
                    ContinuousCandidateRejection(
                        symbol, added, ("candidate_joint_risk_unavailable",)
                    )
                )
                continue
            evaluated += 1
            risk = _risk_reasons(metrics, budget)
            if risk:
                rejections.append(ContinuousCandidateRejection(symbol, added, risk))
                continue
            # Strict improvement keeps cash on an exact/numeric tie. Never
            # force a replacement merely because one passes the risk budget.
            if metrics.holding_period_return_lcb > best_metrics.holding_period_return_lcb + 1e-12:
                best_symbol, best_weight, best_metrics = symbol, added, metrics
    if best_symbol is None:
        return result(
            ContinuousPortfolioStatus.HOLD_CASH,
            ("cash_option_not_improved_by_any_feasible_replacement",),
        )
    return result(
        ContinuousPortfolioStatus.SELECTED,
        ("historical_lcb_proxy_improves_locked_cash_baseline",),
        selected=best_symbol,
        added=best_weight,
        metrics=best_metrics,
    )
