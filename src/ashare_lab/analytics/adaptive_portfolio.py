"""Deterministic 3--5 stock allocation and downside-risk evaluation.

This module is deliberately independent from the existing weekly portfolio
builder.  It consumes an already screened candidate set and never fetches data,
changes positions, or places orders.

Allocation formula
------------------

For stock ``i`` the unprojected priority is::

    priority_i = (1 / downside_volatility_i) * (0.75 + 0.50 * signal_score_i)

where ``signal_score`` must be in ``[0, 1]`` and daily downside volatility is
``sqrt(mean(min(return, 0)^2)) * sqrt(252)``.  The signal multiplier is therefore
bounded to ``[0.75, 1.25]`` and cannot overwhelm the inverse-risk term.  Priorities
are projected first to industry totals and then to individual weights with a
deterministic box-simplex projection.  These continuous total-account weights
remain the audit target.  Their limits are:

* 3 stocks: 70% stock exposure, 15%--35% per stock;
* 4 stocks: 80% stock exposure, 10%--30% per stock;
* 5 stocks: 85% stock exposure, 8%--25% per stock.

The operational stock sleeve is then selected by exhaustive search on a 10%
integer grid.  Three-stock sleeves permit 20%--50% per name, four-stock sleeves
10%--40%, and five-stock sleeves 10%--30%.  The grid must sum to 100% of the
stock sleeve and must still respect the total-account industry cap.  All risk
and return metrics use these operational weights, not the continuous target.
The nearest structurally feasible grid is chosen before risk evaluation; if
that grid breaches a risk budget, the candidate group fails.  The optimizer
does not search a farther grid allocation merely to rescue the group.
Unallocated capital is cash and borrowing is always zero.

Risk formulas
-------------

* annual downside volatility uses zero as the minimum acceptable daily return,
  with fixed shares and idle cash within each non-overlapping holding block;
* the legacy 60-session rolling drawdown severity is retained as a reporting
  metric;
* the binding rolling drawdown severity uses an overlapping window equal to the
  configured holding period, so one-, two-, four-, thirteen-, twenty-six-, and
  fifty-two-week research paths do not share one fixed risk horizon; both
  calculations start each window at equity one, hold shares and cash fixed,
  and allow position weights to drift until that window ends;
* 5-session ES95 is the positive loss magnitude of the mean of the worst 5% of
  non-overlapping, buy-and-hold-within-block portfolio returns;
* down-period correlation is the largest pairwise Pearson correlation on dates
  when the equal-weight candidate basket return is negative;
* downside risk contribution uses the uncentred downside second-moment matrix
  ``S = min(R, 0).T @ min(R, 0) / n`` and Euler contribution
  ``w_i * (S @ w)_i / (w.T @ S @ w)``;
* the holding-period lower confidence bound uses non-overlapping,
  buy-and-hold-within-block returns, subtracts the configured total-account cost
  per holding period, and reports the one-sided normal bound
  ``mean - z(confidence) * sample_std / sqrt(n)``.

These statistics describe the supplied history; they are not a forecast or a
substitute for purged walk-forward validation.  Missing, misaligned, short,
non-finite, or degenerate data raises :class:`AdaptivePortfolioDataError` so a
caller cannot silently manufacture a portfolio from incomplete evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from ashare_lab.analytics.weight_quantization import (
    WEIGHT_QUANTIZATION_METHOD_VERSION,
    quantize_stock_sleeve_weights,
)

ANNUAL_SESSIONS = 252
ROLLING_DRAWDOWN_SESSIONS = 60
FIVE_DAY_SESSIONS = 5
MINIMUM_FIVE_DAY_SAMPLES = 20
MINIMUM_ROLLING_DRAWDOWN_WINDOWS = 20
SIGNAL_TILT_FLOOR = 0.75
SIGNAL_TILT_CEILING = 1.25
INDUSTRY_WEIGHT_HARD_CAP = 0.40

# Limits are total-account weights, not fractions of the stock sleeve.  They
# describe the normal-regime defaults.  A cycle policy may lower total exposure
# through ``AdaptiveRiskBudget.maximum_stock_exposure``; the position box is
# then scaled proportionally instead of changing the stock-sleeve geometry.
POSITION_LIMITS: dict[int, tuple[float, float, float]] = {
    3: (0.70, 0.15, 0.35),
    4: (0.80, 0.10, 0.30),
    5: (0.85, 0.08, 0.25),
}

# Bounds are fractions of the stock sleeve, not total-account weights.
OPERATION_STOCK_SLEEVE_LIMITS: dict[int, tuple[float, float]] = {
    3: (0.20, 0.50),
    4: (0.10, 0.40),
    5: (0.10, 0.30),
}
OPERATION_STOCK_SLEEVE_STEP = 0.10
CONTINUOUS_TARGET_OPERATION_METHOD = (
    "inverse downside volatility x bounded signal tilt; continuous bounded target; "
    "nearest structurally feasible exhaustive 10% stock-sleeve grid selected before risk; "
    "operational-weight historical risk evaluation; no farther-grid risk rescue"
)
DIRECT_OPERATION_METHOD = (
    "caller-supplied operational total-account weights; validated 10% stock-sleeve grid; "
    "direct historical risk evaluation without quantization"
)
OPERATIONAL_WEIGHT_VALIDATION_METHOD_VERSION = "direct-operational-stock-sleeve-validation-v1.0.0"


class AdaptivePortfolioDataError(ValueError):
    """Raised when the candidate set cannot support a fail-closed evaluation."""


@dataclass(frozen=True, slots=True)
class AdaptiveCandidate:
    """One quantitatively screened candidate with aligned daily simple returns."""

    symbol: str
    industry: str
    signal_score: float
    returns: pd.Series

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        industry = str(self.industry).strip()
        if not symbol:
            raise AdaptivePortfolioDataError("candidate symbol cannot be blank")
        if not industry:
            raise AdaptivePortfolioDataError(f"{symbol}: industry cannot be blank")
        if isinstance(self.signal_score, bool):
            raise AdaptivePortfolioDataError(f"{symbol}: signal_score must be numeric")
        try:
            score = float(self.signal_score)
        except (TypeError, ValueError) as exc:
            raise AdaptivePortfolioDataError(f"{symbol}: signal_score must be numeric") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise AdaptivePortfolioDataError(f"{symbol}: signal_score must be in [0, 1]")
        if not isinstance(self.returns, pd.Series):
            raise AdaptivePortfolioDataError(f"{symbol}: returns must be a pandas Series")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "industry", industry)
        object.__setattr__(self, "signal_score", score)


@dataclass(frozen=True, slots=True)
class AdaptiveRiskBudget:
    """Explicit research risk limits and sampling requirements.

    Defaults are conservative research settings, not guarantees.  Callers may
    tighten them, but the industry cap can never exceed 40% of account equity.
    ``holding_period_cost_rate`` is a total-account return deduction applied once
    to every non-overlapping holding-period sample.
    """

    max_annual_downside_volatility: float = 0.18
    max_rolling_drawdown_60_p90: float = 0.12
    max_es95_5d: float = 0.08
    max_down_period_correlation: float = 0.75
    max_position_downside_risk_contribution: float = 0.45
    industry_weight_limit: float = INDUSTRY_WEIGHT_HARD_CAP
    holding_period_sessions: int = 20
    holding_period_cost_rate: float = 0.0
    lcb_confidence: float = 0.90
    minimum_observations: int = 160
    minimum_down_periods: int = 20
    minimum_holding_period_samples: int = 8
    maximum_stock_exposure: float | None = None
    # Appended for positional compatibility.  ``None`` deliberately falls back
    # to the historical 60-session limit while changing only the measurement
    # window to ``holding_period_sessions``.
    max_horizon_rolling_drawdown_p90: float | None = None

    def __post_init__(self) -> None:
        positive: dict[str, object] = {
            "max_annual_downside_volatility": self.max_annual_downside_volatility,
            "max_rolling_drawdown_60_p90": self.max_rolling_drawdown_60_p90,
            "max_es95_5d": self.max_es95_5d,
            "max_position_downside_risk_contribution": (
                self.max_position_downside_risk_contribution
            ),
            "industry_weight_limit": self.industry_weight_limit,
        }
        for name, raw in positive.items():
            if isinstance(raw, bool):
                raise ValueError(f"{name} must be finite and positive")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be finite and positive") from exc
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if isinstance(self.max_down_period_correlation, bool):
            raise ValueError("max_down_period_correlation must be in [-1, 1]")
        try:
            max_correlation = float(self.max_down_period_correlation)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_down_period_correlation must be in [-1, 1]") from exc
        if not math.isfinite(max_correlation):
            raise ValueError("max_down_period_correlation must be in [-1, 1]")
        object.__setattr__(self, "max_down_period_correlation", max_correlation)
        if self.max_rolling_drawdown_60_p90 > 1.0:
            raise ValueError("max_rolling_drawdown_60_p90 cannot exceed 1")
        if self.max_es95_5d > 1.0:
            raise ValueError("max_es95_5d cannot exceed 1")
        if self.max_position_downside_risk_contribution > 1.0:
            raise ValueError("max_position_downside_risk_contribution cannot exceed 1")
        if not -1.0 <= self.max_down_period_correlation <= 1.0:
            raise ValueError("max_down_period_correlation must be in [-1, 1]")
        if self.industry_weight_limit > INDUSTRY_WEIGHT_HARD_CAP:
            raise ValueError("industry_weight_limit cannot exceed 40%")
        if self.maximum_stock_exposure is not None:
            if isinstance(self.maximum_stock_exposure, bool):
                raise ValueError("maximum_stock_exposure must be in (0, 0.85]")
            try:
                maximum_exposure = float(self.maximum_stock_exposure)
            except (TypeError, ValueError) as exc:
                raise ValueError("maximum_stock_exposure must be in (0, 0.85]") from exc
            if not math.isfinite(maximum_exposure) or not 0.0 < maximum_exposure <= 0.85:
                raise ValueError("maximum_stock_exposure must be in (0, 0.85]")
            object.__setattr__(self, "maximum_stock_exposure", maximum_exposure)
        if self.max_horizon_rolling_drawdown_p90 is not None:
            if isinstance(self.max_horizon_rolling_drawdown_p90, bool):
                raise ValueError("max_horizon_rolling_drawdown_p90 must be in (0, 1]")
            try:
                horizon_drawdown_limit = float(self.max_horizon_rolling_drawdown_p90)
            except (TypeError, ValueError) as exc:
                raise ValueError("max_horizon_rolling_drawdown_p90 must be in (0, 1]") from exc
            if not math.isfinite(horizon_drawdown_limit) or not 0.0 < horizon_drawdown_limit <= 1.0:
                raise ValueError("max_horizon_rolling_drawdown_p90 must be in (0, 1]")
            object.__setattr__(
                self,
                "max_horizon_rolling_drawdown_p90",
                horizon_drawdown_limit,
            )
        if isinstance(self.holding_period_sessions, bool) or not isinstance(
            self.holding_period_sessions, int
        ):
            raise ValueError("holding_period_sessions must be an integer of at least two")
        if self.holding_period_sessions < 2:
            raise ValueError("holding_period_sessions must be an integer of at least two")
        try:
            cost_rate = float(self.holding_period_cost_rate)
            confidence = float(self.lcb_confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("cost rate and confidence must be numeric") from exc
        if not math.isfinite(cost_rate) or not 0.0 <= cost_rate < 1.0:
            raise ValueError("holding_period_cost_rate must be in [0, 1)")
        if not math.isfinite(confidence) or not 0.50 < confidence < 1.0:
            raise ValueError("lcb_confidence must be between 0.50 and 1")
        object.__setattr__(self, "holding_period_cost_rate", cost_rate)
        object.__setattr__(self, "lcb_confidence", confidence)
        integer_minimums = {
            "minimum_observations": self.minimum_observations,
            "minimum_down_periods": self.minimum_down_periods,
            "minimum_holding_period_samples": self.minimum_holding_period_samples,
        }
        for name, value in integer_minimums.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")


@dataclass(frozen=True, slots=True)
class AdaptivePosition:
    symbol: str
    industry: str
    signal_score: float
    weight: float
    annual_downside_volatility: float
    downside_risk_contribution: float


@dataclass(frozen=True, slots=True)
class AdaptivePortfolioMetrics:
    observation_count: int
    down_period_count: int
    rolling_drawdown_window_count: int
    five_day_sample_count: int
    holding_period_sessions: int
    holding_period_sample_count: int
    annual_downside_volatility: float
    rolling_max_drawdown_60_p90: float
    es95_5d: float
    max_down_period_correlation: float
    max_position_downside_risk_contribution: float
    holding_period_return_mean: float
    holding_period_return_lcb: float
    lcb_confidence: float
    holding_period_cost_rate: float
    is_out_of_sample: bool = False
    # Appended defaults preserve older positional constructors.  New engine
    # results always populate these with the binding holding-horizon metric.
    horizon_rolling_drawdown_window_sessions: int = ROLLING_DRAWDOWN_SESSIONS
    horizon_rolling_drawdown_window_count: int = 0
    horizon_rolling_max_drawdown_p90: float = 0.0
    path_method_version: str = "fixed-shares-plus-cash-per-window-v1.0.0"


@dataclass(frozen=True, slots=True)
class AdaptiveRiskBudgetResult:
    passed: bool
    annual_downside_volatility_passed: bool
    rolling_drawdown_passed: bool
    es95_5d_passed: bool
    down_period_correlation_passed: bool
    position_risk_contribution_passed: bool
    industry_concentration_passed: bool
    violations: tuple[str, ...]
    # ``rolling_drawdown_passed`` above is retained as a compatibility alias
    # for the binding horizon result.  These appended fields make the active
    # window and effective (possibly fallback) limit explicit.
    horizon_rolling_drawdown_passed: bool = True
    horizon_rolling_drawdown_window_sessions: int = ROLLING_DRAWDOWN_SESSIONS
    horizon_rolling_drawdown_limit: float = 0.12


@dataclass(frozen=True, slots=True)
class AdaptivePortfolioEvaluation:
    positions: tuple[AdaptivePosition, ...]
    stock_exposure: float
    cash_weight: float
    borrowed_weight: float
    industry_weights: tuple[tuple[str, float], ...]
    metrics: AdaptivePortfolioMetrics
    risk_budget: AdaptiveRiskBudgetResult
    method: str = CONTINUOUS_TARGET_OPERATION_METHOD
    # Appended defaults preserve compatibility for older positional construction
    # while retaining the pre-quantization target for audit and reproduction.
    exact_target_weights: tuple[tuple[str, float], ...] = ()
    stock_sleeve_weight_step: float | None = None
    weight_quantization_method_version: str | None = None


DEFAULT_RISK_BUDGET = AdaptiveRiskBudget()


def optimize_adaptive_portfolio(
    candidates: Sequence[AdaptiveCandidate],
    *,
    budget: AdaptiveRiskBudget = DEFAULT_RISK_BUDGET,
) -> AdaptivePortfolioEvaluation:
    """Allocate and evaluate one 3--5 stock candidate set without side effects.

    The nearest structurally and industry-feasible grid is fixed before risk
    metrics are evaluated.  A risk-budget failure rejects this candidate set;
    no farther grid point is searched as a risk-aware rescue.
    """

    prepared, returns = _prepare_candidates(candidates, budget)
    stock_count = len(prepared)
    exposure, lower, upper = _position_limits(stock_count, budget)
    individual_downside = _individual_downside_volatility(returns)
    priorities = np.asarray(
        [
            (1.0 / individual_downside[index])
            * (
                SIGNAL_TILT_FLOOR
                + (SIGNAL_TILT_CEILING - SIGNAL_TILT_FLOOR) * candidate.signal_score
            )
            for index, candidate in enumerate(prepared)
        ],
        dtype=float,
    )
    exact_target_weights = _allocate_with_industry_caps(
        prepared,
        priorities,
        exposure=exposure,
        lower=lower,
        upper=upper,
        industry_cap=budget.industry_weight_limit,
    )
    operational_weights = _operationalize_weights(
        prepared,
        exact_target_weights,
        exposure=exposure,
        industry_cap=budget.industry_weight_limit,
    )
    return _evaluate_prepared(
        prepared,
        returns,
        operational_weights,
        individual_downside,
        budget,
        exact_target_weights=exact_target_weights,
    )


def evaluate_adaptive_portfolio(
    candidates: Sequence[AdaptiveCandidate],
    weights: Mapping[str, float],
    *,
    budget: AdaptiveRiskBudget = DEFAULT_RISK_BUDGET,
) -> AdaptivePortfolioEvaluation:
    """Evaluate a caller-supplied continuous target with the same pure risk engine.

    Symbols must exactly match the candidate set, target weights must sum to the
    exposure assigned to that candidate count, and every target must respect the
    same continuous per-position bounds used by :func:`optimize_adaptive_portfolio`.
    The target is retained for audit, then mapped to the nearest feasible 10%
    stock-sleeve grid.  Industry concentration and all reported metrics use the
    operational grid weights rather than the caller's continuous target.
    """

    prepared, returns = _prepare_candidates(candidates, budget)
    exposure, lower, upper = _position_limits(len(prepared), budget)
    ordered = _ordered_total_account_weights(prepared, weights)
    if bool((ordered < lower - 1e-12).any()) or bool((ordered > upper + 1e-12).any()):
        raise AdaptivePortfolioDataError(
            f"each {len(prepared)}-stock weight must be in [{lower:.2f}, {upper:.2f}]"
        )
    if not math.isclose(float(ordered.sum()), exposure, abs_tol=1e-10):
        raise AdaptivePortfolioDataError(
            f"{len(prepared)}-stock weights must sum to exposure {exposure:.2f}"
        )
    operational_weights = _operationalize_weights(
        prepared,
        ordered,
        exposure=exposure,
        industry_cap=budget.industry_weight_limit,
    )
    individual_downside = _individual_downside_volatility(returns)
    return _evaluate_prepared(
        prepared,
        returns,
        operational_weights,
        individual_downside,
        budget,
        exact_target_weights=ordered,
    )


def evaluate_operational_adaptive_portfolio(
    candidates: Sequence[AdaptiveCandidate],
    weights: Mapping[str, float],
    *,
    budget: AdaptiveRiskBudget = DEFAULT_RISK_BUDGET,
) -> AdaptivePortfolioEvaluation:
    """Validate and directly evaluate caller-supplied operational weights.

    ``weights`` are total-account weights.  Their stock-sleeve fractions must
    already be exact 10% units, sum to 100% of the count-specific stock
    exposure, satisfy the 3/4/5-name operational bounds, and respect the
    total-account industry cap.  This function never quantizes or substitutes
    another grid point.
    """

    prepared, returns = _prepare_candidates(candidates, budget)
    exposure = _position_limits(len(prepared), budget)[0]
    ordered = _ordered_total_account_weights(prepared, weights)
    _validate_operational_weights(
        prepared,
        ordered,
        exposure=exposure,
        industry_cap=budget.industry_weight_limit,
    )
    individual_downside = _individual_downside_volatility(returns)
    return _evaluate_prepared(
        prepared,
        returns,
        ordered,
        individual_downside,
        budget,
        exact_target_weights=ordered,
        evaluation_method=DIRECT_OPERATION_METHOD,
        weight_method_version=OPERATIONAL_WEIGHT_VALIDATION_METHOD_VERSION,
    )


def _position_limits(
    stock_count: int,
    budget: AdaptiveRiskBudget,
) -> tuple[float, float, float]:
    """Return exposure and a proportionally scaled per-name allocation box."""

    base_exposure, base_lower, base_upper = POSITION_LIMITS[stock_count]
    if budget.maximum_stock_exposure is None:
        return base_exposure, base_lower, base_upper
    exposure = min(base_exposure, budget.maximum_stock_exposure)
    scale = exposure / base_exposure
    return exposure, base_lower * scale, base_upper * scale


def _ordered_total_account_weights(
    candidates: tuple[AdaptiveCandidate, ...],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Normalize and order a complete total-account weight mapping."""

    if not isinstance(weights, Mapping):
        raise AdaptivePortfolioDataError("weights must be a symbol-to-weight mapping")
    normalized_weights: dict[str, float] = {}
    for raw_symbol, raw_weight in weights.items():
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in normalized_weights:
            raise AdaptivePortfolioDataError("weight symbols must be unique and non-blank")
        if isinstance(raw_weight, bool):
            raise AdaptivePortfolioDataError(f"{symbol}: weight must be numeric")
        try:
            value = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise AdaptivePortfolioDataError(f"{symbol}: weight must be numeric") from exc
        if not math.isfinite(value):
            raise AdaptivePortfolioDataError(f"{symbol}: weight must be finite")
        normalized_weights[symbol] = value
    symbols = {candidate.symbol for candidate in candidates}
    if set(normalized_weights) != symbols:
        raise AdaptivePortfolioDataError("weight symbols must exactly match candidate symbols")
    return np.asarray(
        [normalized_weights[candidate.symbol] for candidate in candidates],
        dtype=float,
    )


def _prepare_candidates(
    candidates: Sequence[AdaptiveCandidate],
    budget: AdaptiveRiskBudget,
) -> tuple[tuple[AdaptiveCandidate, ...], np.ndarray]:
    if not isinstance(budget, AdaptiveRiskBudget):
        raise TypeError("budget must be an AdaptiveRiskBudget")
    if isinstance(candidates, (str, bytes)):
        raise AdaptivePortfolioDataError("candidates must be a sequence of candidate objects")
    raw_candidates = tuple(candidates)
    if any(not isinstance(item, AdaptiveCandidate) for item in raw_candidates):
        raise AdaptivePortfolioDataError("every candidate must be an AdaptiveCandidate")
    prepared = tuple(sorted(raw_candidates, key=lambda item: item.symbol))
    if len(prepared) not in POSITION_LIMITS:
        raise AdaptivePortfolioDataError("exactly three, four, or five candidates are required")
    symbols = [item.symbol for item in prepared]
    if len(set(symbols)) != len(symbols):
        raise AdaptivePortfolioDataError("candidate symbols must be unique")

    reference_index = prepared[0].returns.index
    if reference_index.has_duplicates or not reference_index.is_monotonic_increasing:
        raise AdaptivePortfolioDataError(
            f"{prepared[0].symbol}: return index must be unique and increasing"
        )
    if len(reference_index) < budget.minimum_observations:
        raise AdaptivePortfolioDataError(
            f"only {len(reference_index)} common observations; "
            f"at least {budget.minimum_observations} required"
        )

    columns: list[np.ndarray] = []
    for candidate in prepared:
        series = candidate.returns
        if series.index.has_duplicates or not series.index.is_monotonic_increasing:
            raise AdaptivePortfolioDataError(
                f"{candidate.symbol}: return index must be unique and increasing"
            )
        if not series.index.equals(reference_index):
            raise AdaptivePortfolioDataError("candidate return indices must match exactly")
        numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        if not bool(np.isfinite(numeric).all()):
            raise AdaptivePortfolioDataError(f"{candidate.symbol}: returns contain missing data")
        if bool((numeric <= -1.0).any()):
            raise AdaptivePortfolioDataError(
                f"{candidate.symbol}: simple returns must be greater than -100%"
            )
        columns.append(numeric)

    matrix = np.column_stack(columns)
    if len(matrix) // FIVE_DAY_SESSIONS < MINIMUM_FIVE_DAY_SAMPLES:
        raise AdaptivePortfolioDataError(
            f"at least {MINIMUM_FIVE_DAY_SAMPLES} non-overlapping five-day samples required"
        )
    rolling_windows = len(matrix) - ROLLING_DRAWDOWN_SESSIONS + 1
    if rolling_windows < MINIMUM_ROLLING_DRAWDOWN_WINDOWS:
        raise AdaptivePortfolioDataError(
            f"at least {MINIMUM_ROLLING_DRAWDOWN_WINDOWS} rolling drawdown windows required"
        )
    horizon_rolling_windows = len(matrix) - budget.holding_period_sessions + 1
    if horizon_rolling_windows < MINIMUM_ROLLING_DRAWDOWN_WINDOWS:
        raise AdaptivePortfolioDataError(
            f"only {max(0, horizon_rolling_windows)} holding-horizon rolling drawdown windows; "
            f"at least {MINIMUM_ROLLING_DRAWDOWN_WINDOWS} required"
        )
    holding_samples = len(matrix) // budget.holding_period_sessions
    if holding_samples < budget.minimum_holding_period_samples:
        raise AdaptivePortfolioDataError(
            f"only {holding_samples} non-overlapping holding-period samples; "
            f"at least {budget.minimum_holding_period_samples} required"
        )
    return prepared, matrix


def _individual_downside_volatility(returns: np.ndarray) -> np.ndarray:
    downside = np.minimum(returns, 0.0)
    volatility = np.sqrt(np.mean(np.square(downside), axis=0)) * math.sqrt(ANNUAL_SESSIONS)
    if not bool(np.isfinite(volatility).all()) or bool((volatility <= 0.0).any()):
        raise AdaptivePortfolioDataError(
            "every candidate needs finite, non-zero historical downside volatility"
        )
    return volatility


def _project_box_simplex(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total: float,
) -> np.ndarray:
    """Euclidean projection onto ``sum(x)=total`` with elementwise bounds."""

    if float(lower.sum()) > total + 1e-12 or float(upper.sum()) < total - 1e-12:
        raise AdaptivePortfolioDataError("allocation bounds cannot reach required exposure")
    low_lambda = float(np.min(target - upper)) - 1.0
    high_lambda = float(np.max(target - lower)) + 1.0
    for _ in range(160):
        midpoint = (low_lambda + high_lambda) / 2.0
        projected = np.clip(target - midpoint, lower, upper)
        if float(projected.sum()) > total:
            low_lambda = midpoint
        else:
            high_lambda = midpoint
    result = np.clip(target - (low_lambda + high_lambda) / 2.0, lower, upper)
    residual = total - float(result.sum())
    if residual > 1e-12:
        for index in range(len(result)):
            addition = min(residual, float(upper[index] - result[index]))
            result[index] += addition
            residual -= addition
            if residual <= 1e-12:
                break
    elif residual < -1e-12:
        for index in range(len(result)):
            reduction = min(-residual, float(result[index] - lower[index]))
            result[index] -= reduction
            residual += reduction
            if residual >= -1e-12:
                break
    if not math.isclose(float(result.sum()), total, abs_tol=1e-10):
        raise AdaptivePortfolioDataError("bounded projection failed to preserve exposure")
    return result


def _allocate_with_industry_caps(
    candidates: tuple[AdaptiveCandidate, ...],
    priorities: np.ndarray,
    *,
    exposure: float,
    lower: float,
    upper: float,
    industry_cap: float,
) -> np.ndarray:
    if not bool(np.isfinite(priorities).all()) or bool((priorities <= 0.0).any()):
        raise AdaptivePortfolioDataError("allocation priorities must be finite and positive")
    industries = tuple(sorted({candidate.industry for candidate in candidates}))
    members = {
        industry: np.asarray(
            [index for index, candidate in enumerate(candidates) if candidate.industry == industry]
        )
        for industry in industries
    }
    group_priorities = np.asarray(
        [float(priorities[members[industry]].sum()) for industry in industries], dtype=float
    )
    group_lower = np.asarray(
        [len(members[industry]) * lower for industry in industries], dtype=float
    )
    group_upper = np.asarray(
        [min(len(members[industry]) * upper, industry_cap) for industry in industries],
        dtype=float,
    )
    if bool((group_lower > group_upper + 1e-12).any()):
        raise AdaptivePortfolioDataError(
            "candidate industry concentration is infeasible at the minimum position weights"
        )
    group_target = exposure * group_priorities / float(group_priorities.sum())
    group_weights = _project_box_simplex(group_target, group_lower, group_upper, exposure)

    result = np.zeros(len(candidates), dtype=float)
    for group_index, industry in enumerate(industries):
        indices = members[industry]
        local_priorities = priorities[indices]
        local_total = float(group_weights[group_index])
        local_target = local_total * local_priorities / float(local_priorities.sum())
        result[indices] = _project_box_simplex(
            local_target,
            np.full(len(indices), lower),
            np.full(len(indices), upper),
            local_total,
        )
    if bool((result < lower - 1e-10).any()) or bool((result > upper + 1e-10).any()):
        raise AdaptivePortfolioDataError("projected position weight breached its bounds")
    return result


def _operationalize_weights(
    candidates: tuple[AdaptiveCandidate, ...],
    exact_target_weights: np.ndarray,
    *,
    exposure: float,
    industry_cap: float,
) -> np.ndarray:
    """Return the nearest structurally and industry-feasible 10% sleeve grid."""

    stock_count = len(candidates)
    minimum_sleeve, maximum_sleeve = OPERATION_STOCK_SLEEVE_LIMITS[stock_count]
    if not math.isclose(float(exact_target_weights.sum()), exposure, abs_tol=1e-10):
        raise AdaptivePortfolioDataError("continuous target does not preserve stock exposure")
    exact_sleeve = exact_target_weights / exposure
    sleeve_industry_cap = min(1.0, industry_cap / exposure)
    try:
        operational_sleeve = quantize_stock_sleeve_weights(
            tuple(float(value) for value in exact_sleeve),
            step=OPERATION_STOCK_SLEEVE_STEP,
            minimum_weight=minimum_sleeve,
            maximum_weight=maximum_sleeve,
            group_labels=tuple(candidate.industry for candidate in candidates),
            maximum_group_weight=sleeve_industry_cap,
        )
    except ValueError as exc:
        raise AdaptivePortfolioDataError(
            "no structurally and industry-feasible 10% stock-sleeve grid"
        ) from exc

    operational = np.asarray(operational_sleeve, dtype=float) * exposure
    _validate_operational_weights(
        candidates,
        operational,
        exposure=exposure,
        industry_cap=industry_cap,
    )
    return operational


def _validate_operational_weights(
    candidates: tuple[AdaptiveCandidate, ...],
    weights: np.ndarray,
    *,
    exposure: float,
    industry_cap: float,
) -> None:
    """Fail closed unless weights already form the exact operational grid."""

    if weights.shape != (len(candidates),) or not bool(np.isfinite(weights).all()):
        raise AdaptivePortfolioDataError("operational weights must be finite and complete")
    if not math.isclose(float(weights.sum()), exposure, abs_tol=1e-10):
        raise AdaptivePortfolioDataError(
            f"{len(candidates)}-stock operational weights must sum to exposure {exposure:.2f}"
        )

    minimum_sleeve, maximum_sleeve = OPERATION_STOCK_SLEEVE_LIMITS[len(candidates)]
    sleeve = weights / exposure
    if not math.isclose(float(sleeve.sum()), 1.0, abs_tol=1e-10):
        raise AdaptivePortfolioDataError("operational stock-sleeve weights must sum to 1.0")
    if bool((sleeve < minimum_sleeve - 1e-12).any()) or bool(
        (sleeve > maximum_sleeve + 1e-12).any()
    ):
        raise AdaptivePortfolioDataError(
            f"each {len(candidates)}-stock operational sleeve weight must be in "
            f"[{minimum_sleeve:.2f}, {maximum_sleeve:.2f}]"
        )
    units = sleeve / OPERATION_STOCK_SLEEVE_STEP
    if not bool(np.isclose(units, np.rint(units), rtol=0.0, atol=1e-10).all()):
        raise AdaptivePortfolioDataError(
            "operational stock-sleeve weights must be exact 10% increments"
        )

    industry_totals: dict[str, float] = {}
    for candidate, weight in zip(candidates, weights, strict=True):
        industry_totals[candidate.industry] = industry_totals.get(candidate.industry, 0.0) + float(
            weight
        )
    if max(industry_totals.values()) > industry_cap + 1e-12:
        raise AdaptivePortfolioDataError("operational weights breached the industry cap")


def _non_overlapping_portfolio_returns(
    returns: np.ndarray,
    weights: np.ndarray,
    sessions: int,
) -> np.ndarray:
    sample_count = len(returns) // sessions
    usable = sample_count * sessions
    blocks = returns[-usable:].reshape(sample_count, sessions, returns.shape[1])
    stock_block_returns = np.prod(1.0 + blocks, axis=1) - 1.0
    return stock_block_returns @ weights


def _fixed_share_daily_returns(
    returns: np.ndarray, weights: np.ndarray, sessions: int
) -> np.ndarray:
    """Daily account returns with fixed shares/cash within each holding block."""

    usable = len(returns) // sessions * sessions
    blocks = returns[-usable:].reshape(-1, sessions, returns.shape[1])
    equity = (1.0 - float(weights.sum())) + np.cumprod(1.0 + blocks, axis=1) @ weights
    start = np.ones((len(blocks), 1))
    previous = np.concatenate((start, equity[:, :-1]), axis=1)
    return (equity / previous - 1.0).reshape(-1)


def _fixed_share_rolling_drawdown_magnitudes(
    returns: np.ndarray, weights: np.ndarray, sessions: int
) -> np.ndarray:
    """Each rolling window starts with the same shares allocation and cash.

    Reweighting daily would silently simulate a different strategy.  Each
    window instead allows weights to drift until its end, without any trades.
    """

    if sessions < 2 or sessions > len(returns):
        raise AdaptivePortfolioDataError("invalid fixed-share drawdown window")
    windows = np.lib.stride_tricks.sliding_window_view(returns, sessions, axis=0)
    stock_equity = np.cumprod(1.0 + windows, axis=2)
    equity = (1.0 - float(weights.sum())) + np.einsum("nst,s->nt", stock_equity, weights)
    equity = np.concatenate((np.ones((len(equity), 1)), equity), axis=1)
    return -(equity / np.maximum.accumulate(equity, axis=1) - 1.0).min(axis=1)


def _rolling_drawdown_magnitudes(
    portfolio_returns: np.ndarray,
    sessions: int = ROLLING_DRAWDOWN_SESSIONS,
) -> np.ndarray:
    """Return overlapping drawdowns for an explicit positive session window."""

    if isinstance(sessions, bool) or not isinstance(sessions, int) or sessions < 2:
        raise AdaptivePortfolioDataError("rolling drawdown sessions must be at least two")
    window_count = len(portfolio_returns) - sessions + 1
    if window_count <= 0:
        raise AdaptivePortfolioDataError("rolling drawdown window exceeds supplied history")
    magnitudes = np.empty(window_count, dtype=float)
    for start in range(window_count):
        sample = portfolio_returns[start : start + sessions]
        equity = np.concatenate(([1.0], np.cumprod(1.0 + sample)))
        running_peak = np.maximum.accumulate(equity)
        magnitudes[start] = max(0.0, -float((equity / running_peak - 1.0).min()))
    return magnitudes


def _down_period_max_correlation(
    returns: np.ndarray,
    minimum_down_periods: int,
) -> tuple[float, int]:
    equal_weight_proxy = returns.mean(axis=1)
    down_returns = returns[equal_weight_proxy < 0.0]
    if len(down_returns) < minimum_down_periods:
        raise AdaptivePortfolioDataError(
            f"only {len(down_returns)} down periods; at least {minimum_down_periods} required"
        )
    correlation = np.corrcoef(down_returns, rowvar=False)
    pair_values = correlation[np.triu_indices(returns.shape[1], k=1)]
    if not bool(np.isfinite(pair_values).all()):
        raise AdaptivePortfolioDataError("down-period correlation is undefined")
    return float(pair_values.max()), len(down_returns)


def _downside_risk_contributions(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    downside = np.minimum(returns, 0.0)
    second_moment = downside.T @ downside / len(downside)
    marginal = second_moment @ weights
    total = float(weights @ marginal)
    if not math.isfinite(total) or total <= 0.0:
        raise AdaptivePortfolioDataError("portfolio downside risk contribution is undefined")
    contributions = weights * marginal / total
    if not bool(np.isfinite(contributions).all()) or bool((contributions < -1e-12).any()):
        raise AdaptivePortfolioDataError("portfolio downside risk contribution is invalid")
    return contributions


def _evaluate_prepared(
    candidates: tuple[AdaptiveCandidate, ...],
    returns: np.ndarray,
    weights: np.ndarray,
    individual_downside: np.ndarray,
    budget: AdaptiveRiskBudget,
    *,
    exact_target_weights: np.ndarray,
    evaluation_method: str = CONTINUOUS_TARGET_OPERATION_METHOD,
    weight_method_version: str = WEIGHT_QUANTIZATION_METHOD_VERSION,
) -> AdaptivePortfolioEvaluation:
    portfolio_returns = _fixed_share_daily_returns(returns, weights, budget.holding_period_sessions)
    portfolio_downside = np.minimum(portfolio_returns, 0.0)
    annual_downside_volatility = float(
        np.sqrt(np.mean(np.square(portfolio_downside))) * math.sqrt(ANNUAL_SESSIONS)
    )

    drawdown_magnitudes = _fixed_share_rolling_drawdown_magnitudes(
        returns,
        weights,
        ROLLING_DRAWDOWN_SESSIONS,
    )
    drawdown_p90 = float(np.quantile(drawdown_magnitudes, 0.90))
    if budget.holding_period_sessions == ROLLING_DRAWDOWN_SESSIONS:
        horizon_drawdown_magnitudes = drawdown_magnitudes
    else:
        horizon_drawdown_magnitudes = _fixed_share_rolling_drawdown_magnitudes(
            returns,
            weights,
            budget.holding_period_sessions,
        )
    horizon_drawdown_p90 = float(np.quantile(horizon_drawdown_magnitudes, 0.90))

    five_day_returns = _non_overlapping_portfolio_returns(returns, weights, FIVE_DAY_SESSIONS)
    five_day_cutoff = float(np.quantile(five_day_returns, 0.05))
    five_day_tail = five_day_returns[five_day_returns <= five_day_cutoff + 1e-15]
    es95_5d = max(0.0, -float(five_day_tail.mean()))

    max_down_correlation, down_period_count = _down_period_max_correlation(
        returns, budget.minimum_down_periods
    )
    contributions = _downside_risk_contributions(returns, weights)
    max_contribution = float(contributions.max())

    holding_returns = _non_overlapping_portfolio_returns(
        returns, weights, budget.holding_period_sessions
    )
    net_holding_returns = holding_returns - budget.holding_period_cost_rate
    holding_mean = float(net_holding_returns.mean())
    holding_std = float(net_holding_returns.std(ddof=1))
    if not math.isfinite(holding_std):
        raise AdaptivePortfolioDataError("holding-period return uncertainty is undefined")
    z_score = NormalDist().inv_cdf(budget.lcb_confidence)
    holding_lcb = holding_mean - z_score * holding_std / math.sqrt(len(net_holding_returns))

    industry_totals: dict[str, float] = {}
    for candidate, weight in zip(candidates, weights, strict=True):
        industry_totals[candidate.industry] = industry_totals.get(candidate.industry, 0.0) + float(
            weight
        )

    metrics_values = (
        annual_downside_volatility,
        drawdown_p90,
        horizon_drawdown_p90,
        es95_5d,
        max_down_correlation,
        max_contribution,
        holding_mean,
        holding_lcb,
    )
    if not all(math.isfinite(value) for value in metrics_values):
        raise AdaptivePortfolioDataError("portfolio risk metrics are non-finite")

    metrics = AdaptivePortfolioMetrics(
        observation_count=len(returns),
        down_period_count=down_period_count,
        rolling_drawdown_window_count=len(drawdown_magnitudes),
        five_day_sample_count=len(five_day_returns),
        holding_period_sessions=budget.holding_period_sessions,
        holding_period_sample_count=len(net_holding_returns),
        annual_downside_volatility=annual_downside_volatility,
        rolling_max_drawdown_60_p90=drawdown_p90,
        es95_5d=es95_5d,
        max_down_period_correlation=max_down_correlation,
        max_position_downside_risk_contribution=max_contribution,
        holding_period_return_mean=holding_mean,
        holding_period_return_lcb=holding_lcb,
        lcb_confidence=budget.lcb_confidence,
        holding_period_cost_rate=budget.holding_period_cost_rate,
        horizon_rolling_drawdown_window_sessions=budget.holding_period_sessions,
        horizon_rolling_drawdown_window_count=len(horizon_drawdown_magnitudes),
        horizon_rolling_max_drawdown_p90=horizon_drawdown_p90,
    )
    risk_result = _risk_budget_result(metrics, industry_totals, budget)
    positions = tuple(
        AdaptivePosition(
            symbol=candidate.symbol,
            industry=candidate.industry,
            signal_score=candidate.signal_score,
            weight=float(weight),
            annual_downside_volatility=float(individual_downside[index]),
            downside_risk_contribution=float(contributions[index]),
        )
        for index, (candidate, weight) in enumerate(zip(candidates, weights, strict=True))
    )
    exposure = _position_limits(len(candidates), budget)[0]
    return AdaptivePortfolioEvaluation(
        positions=positions,
        stock_exposure=exposure,
        cash_weight=1.0 - exposure,
        borrowed_weight=0.0,
        industry_weights=tuple(sorted(industry_totals.items())),
        metrics=metrics,
        risk_budget=risk_result,
        method=evaluation_method,
        exact_target_weights=tuple(
            (candidate.symbol, float(weight))
            for candidate, weight in zip(candidates, exact_target_weights, strict=True)
        ),
        stock_sleeve_weight_step=OPERATION_STOCK_SLEEVE_STEP,
        weight_quantization_method_version=weight_method_version,
    )


def _risk_budget_result(
    metrics: AdaptivePortfolioMetrics,
    industry_weights: Mapping[str, float],
    budget: AdaptiveRiskBudget,
) -> AdaptiveRiskBudgetResult:
    horizon_drawdown_limit = (
        budget.max_rolling_drawdown_60_p90
        if budget.max_horizon_rolling_drawdown_p90 is None
        else budget.max_horizon_rolling_drawdown_p90
    )
    checks = {
        "annual_downside_volatility": (
            metrics.annual_downside_volatility <= budget.max_annual_downside_volatility + 1e-12
        ),
        "horizon_rolling_drawdown_p90": (
            metrics.horizon_rolling_max_drawdown_p90 <= horizon_drawdown_limit + 1e-12
        ),
        "es95_5d": metrics.es95_5d <= budget.max_es95_5d + 1e-12,
        "down_period_correlation": (
            metrics.max_down_period_correlation <= budget.max_down_period_correlation + 1e-12
        ),
        "position_downside_risk_contribution": (
            metrics.max_position_downside_risk_contribution
            <= budget.max_position_downside_risk_contribution + 1e-12
        ),
        "industry_concentration": (
            max(industry_weights.values()) <= budget.industry_weight_limit + 1e-12
        ),
    }
    violations = tuple(name for name, passed in checks.items() if not passed)
    return AdaptiveRiskBudgetResult(
        passed=not violations,
        annual_downside_volatility_passed=checks["annual_downside_volatility"],
        rolling_drawdown_passed=checks["horizon_rolling_drawdown_p90"],
        es95_5d_passed=checks["es95_5d"],
        down_period_correlation_passed=checks["down_period_correlation"],
        position_risk_contribution_passed=checks["position_downside_risk_contribution"],
        industry_concentration_passed=checks["industry_concentration"],
        violations=violations,
        horizon_rolling_drawdown_passed=checks["horizon_rolling_drawdown_p90"],
        horizon_rolling_drawdown_window_sessions=(metrics.horizon_rolling_drawdown_window_sessions),
        horizon_rolling_drawdown_limit=horizon_drawdown_limit,
    )
