"""Build three evidence-labelled weekly A-share research portfolios.

The service accepts already-normalized daily history and metadata.  It is a
research screen, not an execution engine: statistics are descriptive historical
proxies, financing remains off, and an ineligible universe is allowed to return
no portfolio instead of filling slots with weak candidates.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.analytics.allocation import allocate_four_stocks, get_default_profile
from ashare_lab.analytics.market_regime import (
    MarketRegimeAssessment,
    MarketRegimeState,
    assess_market_regime,
)
from ashare_lab.analytics.medium_term_stage import assess_medium_term_stage
from ashare_lab.analytics.portfolio_statistics import (
    PortfolioStatistics,
    calculate_portfolio_statistics,
)
from ashare_lab.analytics.probability import empirical_scenario
from ashare_lab.analytics.return_ambition import (
    ReturnAmbitionAssessment,
    ReturnAmbitionStatus,
    assess_return_ambition,
    validate_annual_return_ambition,
    validate_holding_weeks,
)
from ashare_lab.analytics.risk_metrics import risk_metrics
from ashare_lab.domain.models import PortfolioAllocation, RiskProfile, RiskProfileName

NON_PROMISE_NOTICE = (
    "当前收益、回撤、夏普、Sortino、Calmar及命中率均来自历史固定权重代理和"
    "重叠滚动窗口，不是walk-forward样本外结果、未来概率、收益承诺或最大回撤保证；"
    "也未包含实盘滑点、税费和融资成本。"
)

CORRELATION_LIMITS: dict[RiskProfileName, float] = {
    RiskProfileName.CONSERVATIVE: 0.65,
    RiskProfileName.BALANCED: 0.75,
    RiskProfileName.AGGRESSIVE: 0.85,
}


class WeeklyPortfolioStatus(StrEnum):
    READY = "ready"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ExcludedStock:
    symbol: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SelectedStock:
    symbol: str
    name: str
    industry: str
    score: float
    component_scores: tuple[tuple[str, float], ...]
    effective_weights: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class FactorCoverage:
    """Run-level availability of an optional point-in-time scoring factor."""

    factor: str
    provided: int
    eligible: int
    enabled: bool
    reason: str


HistoricalRiskSummary = PortfolioStatistics


@dataclass(frozen=True, slots=True)
class HistoricalScenarioRange:
    available: bool
    horizon_sessions: int
    sample_n: int
    return_p10: float | None = None
    return_p50: float | None = None
    return_p90: float | None = None
    historical_positive_rate: float | None = None
    method: str = "历史重叠滚动窗口；未经样本外概率校准"
    is_forecast_probability: bool = False
    is_promise: bool = False
    disclaimer: str = NON_PROMISE_NOTICE


@dataclass(frozen=True, slots=True)
class WeeklyPortfolioResult:
    profile: RiskProfileName
    status: WeeklyPortfolioStatus
    allocation: PortfolioAllocation | None = None
    selected: tuple[SelectedStock, ...] = ()
    historical_risk: HistoricalRiskSummary | None = None
    historical_scenario: HistoricalScenarioRange | None = None
    return_ambition_assessment: ReturnAmbitionAssessment | None = None
    reasons: tuple[str, ...] = ()
    risk_warnings: tuple[str, ...] = ()
    disclaimer: str = NON_PROMISE_NOTICE


@dataclass(frozen=True, slots=True)
class WeeklyPortfolioBatch:
    data_cutoff: pd.Timestamp | None
    portfolios: tuple[WeeklyPortfolioResult, ...]
    exclusions: tuple[ExcludedStock, ...]
    holding_weeks: int = 4
    annual_return_ambition_pct: int | None = None
    factor_coverage: tuple[FactorCoverage, ...] = ()
    market_regime: MarketRegimeAssessment | None = None
    disclaimer: str = NON_PROMISE_NOTICE

    def for_profile(self, profile: RiskProfileName | str) -> WeeklyPortfolioResult:
        wanted = RiskProfileName(profile)
        for result in self.portfolios:
            if result.profile == wanted:
                return result
        raise KeyError(wanted.value)


@dataclass(frozen=True, slots=True)
class _Candidate:
    symbol: str
    name: str
    industry: str
    returns: pd.Series
    fundamentals: float | None
    liquidity: float | None
    news: float | None
    sector_context: float | None
    raw: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    candidate: _Candidate
    components: tuple[tuple[str, float], ...]
    effective_weights: tuple[tuple[str, float], ...]
    base_score: float


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _optional_unit_metadata_score(
    metadata: Mapping[str, Any],
    key: str,
    *,
    legacy_key: str | None = None,
) -> float | None:
    """Read a supplied factor without inventing a neutral missing value."""

    available_keys = [candidate for candidate in (key, legacy_key) if candidate in metadata]
    if not available_keys:
        return None
    values = [_finite_or_none(metadata[candidate]) for candidate in available_keys]
    if all(value is None for value in values):
        return None
    if any(value is None or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"{key} must be null or a finite value between 0 and 1")
    assert all(value is not None for value in values)
    if len(values) == 2 and not np.isclose(values[0], values[1]):
        raise ValueError(f"{key} conflicts with legacy field {legacy_key}")
    value = values[0]
    assert value is not None
    return value


def _optional_boolean(metadata: Mapping[str, Any], key: str) -> bool | None:
    if key not in metadata or metadata[key] is None:
        return None
    value = metadata[key]
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} must be null or an explicit boolean")
    return bool(value)


def _as_daily_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("invalid trading date")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _normalized_close(frame: pd.DataFrame) -> pd.Series:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("history must be a pandas DataFrame")
    if "close" not in frame.columns:
        raise ValueError("history is missing the required close column")

    if "trade_date" in frame.columns:
        dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    elif "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
    else:
        dates = pd.to_datetime(frame.index, errors="coerce")
    if bool(pd.isna(dates).any()):
        raise ValueError("history contains invalid trading dates")

    normalized_dates = pd.DatetimeIndex(dates)
    if normalized_dates.tz is not None:
        normalized_dates = normalized_dates.tz_localize(None)
    normalized_dates = normalized_dates.normalize()
    if normalized_dates.has_duplicates:
        raise ValueError("history contains duplicate trading dates")

    values = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    close = pd.Series(values.to_numpy(), index=normalized_dates, name="close").sort_index()
    finite = np.isfinite(close.to_numpy())
    if not bool(finite.all()) or bool((close <= 0.0).any()):
        raise ValueError("close prices must be finite and positive")
    return close


def _normalise_inputs(
    histories: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, Mapping[str, Any]]]:
    normalized_histories: dict[str, pd.DataFrame] = {}
    for symbol, frame in histories.items():
        key = str(symbol).strip().upper()
        if not key or key in normalized_histories:
            raise ValueError("history symbols must be non-blank and unique after normalization")
        normalized_histories[key] = frame

    normalized_metadata: dict[str, Mapping[str, Any]] = {}
    for symbol, item in metadata.items():
        key = str(symbol).strip().upper()
        if not key or key in normalized_metadata:
            raise ValueError("metadata symbols must be non-blank and unique after normalization")
        normalized_metadata[key] = item
    return normalized_histories, normalized_metadata


def _choose_cutoff(
    histories: Mapping[str, pd.DataFrame], requested: object | None
) -> pd.Timestamp | None:
    if requested is not None:
        return _as_daily_timestamp(requested)
    latest: list[pd.Timestamp] = []
    for frame in histories.values():
        try:
            close = _normalized_close(frame)
        except ValueError:
            continue
        if not close.empty:
            latest.append(_as_daily_timestamp(close.index[-1]))
    return max(latest) if latest else None


def _momentum(close: pd.Series, sessions: int) -> float:
    if len(close) <= sessions:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-sessions - 1] - 1.0)


def _candidate_from_input(
    symbol: str,
    frame: pd.DataFrame,
    metadata: Mapping[str, Any] | None,
    cutoff: pd.Timestamp,
    minimum_sessions: int,
    exclude_formation_limit_up: bool,
    exclude_overheated_acceleration: bool,
) -> tuple[_Candidate | None, tuple[str, ...]]:
    reasons: list[str] = []
    if metadata is None:
        return None, ("metadata_missing",)

    for flag in ("is_st", "is_delisting"):
        if flag not in metadata or not isinstance(metadata[flag], (bool, np.bool_)):
            reasons.append(f"{flag}_must_be_explicit_boolean")
    if reasons:
        return None, tuple(reasons)

    name = str(metadata.get("name", symbol)).strip() or symbol
    compact_name = name.replace(" ", "").upper()
    if bool(metadata["is_st"]) or compact_name.startswith(("ST", "*ST")):
        reasons.append("st_stock_excluded")
    if bool(metadata["is_delisting"]) or "退" in compact_name:
        reasons.append("delisting_stock_excluded")
    if bool(metadata.get("is_suspended", False)):
        reasons.append("suspended_stock_excluded")

    try:
        is_limit_up = _optional_boolean(metadata, "is_limit_up_at_cutoff")
        is_buyable = _optional_boolean(metadata, "is_buyable_at_cutoff")
    except ValueError as exc:
        return None, (f"invalid_input:{exc}",)
    if is_buyable is False:
        reasons.append(
            "formation_limit_up_unbuyable"
            if is_limit_up is True
            else "formation_unbuyable_at_cutoff"
        )
    elif is_limit_up is True and exclude_formation_limit_up:
        # A daily close cannot reveal queue position or next-session execution.
        # The actionable portfolio therefore substitutes the next candidate
        # rather than pretending a locked or gapped entry is available.
        reasons.append("formation_limit_up_excluded_pending_execution_check")

    industry = str(metadata.get("industry", "")).strip()
    if not industry:
        reasons.append("industry_missing")
    if reasons:
        return None, tuple(reasons)

    try:
        close = _normalized_close(frame)
        fundamentals = _optional_unit_metadata_score(
            metadata,
            "fundamental_score",
            legacy_key="quality_score",
        )
        liquidity = _optional_unit_metadata_score(metadata, "liquidity_score")
        news = _optional_unit_metadata_score(
            metadata,
            "news_score",
            legacy_key="catalyst_score",
        )
        sector_context = _optional_unit_metadata_score(metadata, "sector_score")
    except ValueError as exc:
        return None, (f"invalid_input:{exc}",)

    close = close.loc[close.index <= cutoff]
    if close.empty or close.index[-1] != cutoff:
        return None, ("stale_or_missing_cutoff_close",)
    if len(close) < minimum_sessions:
        return None, (f"insufficient_history:{len(close)}<{minimum_sessions}",)
    stage = assess_medium_term_stage(close)
    if exclude_overheated_acceleration and stage.hard_freeze_new_entry:
        return None, ("overheated_acceleration_excluded",)

    returns = close.pct_change(fill_method=None).dropna()
    if len(returns) < minimum_sessions - 1:
        return None, ("insufficient_valid_returns",)
    metrics = risk_metrics(returns)
    raw = (
        ("max_drawdown", float(metrics["max_drawdown"])),
        ("cvar95", float(metrics["cvar95"])),
        ("sharpe", float(metrics["sharpe"])),
        ("sortino", float(metrics["sortino"])),
        ("momentum_20", _momentum(close, 20)),
        ("momentum_60", _momentum(close, 60)),
        ("momentum_120", _momentum(close, 120)),
        ("trend_stage_quality", stage.quality_score),
    )
    if any(not isfinite(value) for _, value in raw):
        return None, ("non_finite_derived_feature",)

    return (
        _Candidate(
            symbol=symbol,
            name=name,
            industry=industry,
            returns=returns,
            fundamentals=fundamentals,
            liquidity=liquidity,
            news=news,
            sector_context=sector_context,
            raw=raw,
        ),
        (),
    )


def _percentile_scores(values: list[float]) -> list[float]:
    if len(values) == 1:
        return [0.5]
    series = pd.Series(values, dtype=float)
    ranks = series.rank(method="average")
    return [float(value) for value in ((ranks - 1.0) / (len(series) - 1.0))]


_OPTIONAL_FACTOR_ATTRIBUTES = {
    "fundamentals": "fundamentals",
    "liquidity": "liquidity",
    "news": "news",
    "sector_context": "sector_context",
}


def _factor_coverage(candidates: list[_Candidate]) -> tuple[FactorCoverage, ...]:
    eligible = len(candidates)
    coverage: list[FactorCoverage] = []
    for factor, attribute in _OPTIONAL_FACTOR_ATTRIBUTES.items():
        provided = sum(getattr(candidate, attribute) is not None for candidate in candidates)
        enabled = eligible > 0 and provided == eligible
        if enabled:
            reason = "complete_eligible_universe_coverage"
        elif provided == 0:
            reason = "no_point_in_time_values;factor_disabled"
        else:
            reason = "partial_coverage;factor_disabled_to_avoid_missing_data_bias"
        coverage.append(
            FactorCoverage(
                factor=factor,
                provided=provided,
                eligible=eligible,
                enabled=enabled,
                reason=reason,
            )
        )
    return tuple(coverage)


def _score_candidates(
    candidates: list[_Candidate],
    profile: RiskProfile,
    enabled_factors: frozenset[str],
) -> list[_ScoredCandidate]:
    raw_by_name = {
        name: [dict(candidate.raw)[name] for candidate in candidates]
        for name, _ in candidates[0].raw
    }
    percentile = {name: _percentile_scores(values) for name, values in raw_by_name.items()}
    scored: list[_ScoredCandidate] = []
    weights = profile.scoring

    for index, candidate in enumerate(candidates):
        drawdown_cvar = (percentile["max_drawdown"][index] + percentile["cvar95"][index]) / 2.0
        risk_adjusted = (percentile["sharpe"][index] + percentile["sortino"][index]) / 2.0
        expected_return = (
            0.50 * percentile["momentum_20"][index]
            + 0.30 * percentile["momentum_60"][index]
            + 0.20 * percentile["momentum_120"][index]
        )
        components = {
            "drawdown_cvar": drawdown_cvar,
            "risk_adjusted_return": risk_adjusted,
            "expected_return": expected_return,
            "trend": percentile["trend_stage_quality"][index],
        }
        for factor in enabled_factors:
            value = getattr(candidate, _OPTIONAL_FACTOR_ATTRIBUTES[factor])
            assert value is not None
            components[factor] = value

        # Deprecated composite weights are split into their real ingredients;
        # no absent fundamental/news score is replaced by a neutral constant.
        configured_weights = {
            "drawdown_cvar": weights.drawdown_cvar,
            "risk_adjusted_return": weights.risk_adjusted_return,
            "expected_return": weights.expected_return,
            "fundamentals": weights.fundamentals + 0.50 * weights.quality_liquidity,
            "liquidity": weights.liquidity + 0.50 * weights.quality_liquidity,
            "trend": weights.trend + 0.70 * weights.trend_catalyst,
            "news": weights.news + 0.30 * weights.trend_catalyst,
            "sector_context": weights.market_sector,
        }
        active_weights = {
            name: value
            for name, value in configured_weights.items()
            if value > 0.0 and name in components
        }
        active_total = sum(active_weights.values())
        target_base_mass = 1.0 - weights.diversification
        if active_total <= 0.0 or target_base_mass <= 0.0:
            raise ValueError("profile has no active non-diversification scoring factors")
        effective_weights = {
            name: target_base_mass * value / active_total for name, value in active_weights.items()
        }
        base_score = 100.0 * sum(
            effective_weights[name] * components[name] for name in effective_weights
        )
        scored.append(
            _ScoredCandidate(
                candidate=candidate,
                components=tuple(sorted(components.items())),
                effective_weights=tuple(sorted(effective_weights.items())),
                base_score=base_score,
            )
        )
    return scored


def _pair_correlation(left: _Candidate, right: _Candidate, minimum_overlap: int) -> float | None:
    aligned = pd.concat((left.returns, right.returns), axis=1, join="inner").dropna()
    if len(aligned) < minimum_overlap:
        return None
    correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    return correlation if isfinite(correlation) else None


def _diversification_score(
    candidate: _ScoredCandidate,
    selected: tuple[_ScoredCandidate, ...],
    correlations: Mapping[tuple[str, str], float | None],
) -> float:
    if not selected:
        return 1.0
    correlation_scores: list[float] = []
    different_industry: list[float] = []
    for existing in selected:
        pair = tuple(sorted((candidate.candidate.symbol, existing.candidate.symbol)))
        correlation = correlations[pair]
        if correlation is None:
            return 0.0
        correlation_scores.append((1.0 - correlation) / 2.0)
        different_industry.append(
            float(candidate.candidate.industry != existing.candidate.industry)
        )
    return 0.70 * float(np.mean(correlation_scores)) + 0.30 * float(np.mean(different_industry))


def _select_diversified_four(
    scored: list[_ScoredCandidate],
    profile: RiskProfile,
    *,
    minimum_correlation_overlap: int,
    candidate_pool_size: int,
    beam_width: int,
) -> tuple[tuple[_ScoredCandidate, float, float], ...]:
    ranked = sorted(scored, key=lambda item: (-item.base_score, item.candidate.symbol))
    pool = ranked[:candidate_pool_size]
    if len(pool) < 4:
        return ()

    correlations: dict[tuple[str, str], float | None] = {}
    for left_index, left in enumerate(pool):
        for right in pool[left_index + 1 :]:
            pair = tuple(sorted((left.candidate.symbol, right.candidate.symbol)))
            correlations[pair] = _pair_correlation(
                left.candidate, right.candidate, minimum_correlation_overlap
            )

    # state: weighted portfolio score, selections with dynamic scores, sector exposure
    states: list[
        tuple[
            float,
            tuple[tuple[_ScoredCandidate, float, float], ...],
            dict[str, float],
        ]
    ] = [(0.0, (), {})]
    correlation_limit = CORRELATION_LIMITS[profile.name]

    for slot, slot_weight in enumerate(profile.position_weights):
        expanded: list[
            tuple[
                float,
                tuple[tuple[_ScoredCandidate, float, float], ...],
                dict[str, float],
            ]
        ] = []
        for portfolio_score, selected_rows, sector_weights in states:
            selected = tuple(row[0] for row in selected_rows)
            selected_symbols = {item.candidate.symbol for item in selected}
            for candidate in pool:
                if candidate.candidate.symbol in selected_symbols:
                    continue
                new_sector_weight = (
                    sector_weights.get(candidate.candidate.industry, 0.0) + slot_weight
                )
                if new_sector_weight > profile.risk.sector_exposure_cap + 1e-9:
                    continue

                valid_correlation = True
                for existing in selected:
                    pair = tuple(sorted((candidate.candidate.symbol, existing.candidate.symbol)))
                    correlation = correlations.get(pair)
                    if correlation is None or correlation > correlation_limit:
                        valid_correlation = False
                        break
                if not valid_correlation:
                    continue

                diversity = _diversification_score(candidate, selected, correlations)
                dynamic_score = candidate.base_score + 100.0 * (
                    profile.scoring.diversification * diversity
                )
                slot_fraction = slot_weight / profile.stock_exposure
                next_score = portfolio_score + slot_fraction * dynamic_score
                next_sectors = dict(sector_weights)
                next_sectors[candidate.candidate.industry] = new_sector_weight
                expanded.append(
                    (
                        next_score,
                        (*selected_rows, (candidate, dynamic_score, diversity)),
                        next_sectors,
                    )
                )
        if not expanded:
            return ()
        expanded.sort(
            key=lambda state: (
                -state[0],
                tuple(item[0].candidate.symbol for item in state[1]),
            )
        )
        states = expanded[:beam_width]
        if slot == 3:
            break
    return states[0][1] if states and len(states[0][1]) == 4 else ()


def _historical_outputs(
    selection: tuple[tuple[_ScoredCandidate, float, float], ...],
    allocation: PortfolioAllocation,
    *,
    scenario_horizon_sessions: int,
    minimum_scenario_samples: int,
    holding_weeks: int,
    annual_return_ambition_pct: int | None,
) -> tuple[
    HistoricalRiskSummary,
    HistoricalScenarioRange,
    ReturnAmbitionAssessment | None,
    tuple[str, ...],
]:
    series = []
    for row, position in zip(selection, allocation.positions, strict=True):
        series.append(row[0].candidate.returns.rename(position.ticker))
    aligned = pd.concat(series, axis=1, join="inner").dropna()
    weights = np.asarray([position.weight for position in allocation.positions], dtype=float)
    portfolio_returns = aligned.mul(weights, axis=1).sum(axis=1)
    profile = get_default_profile(allocation.profile)
    summary = calculate_portfolio_statistics(
        portfolio_returns,
        drawdown_budget=profile.risk.drawdown_alert,
        window_sessions=(20, 40, 60),
        minimum_window_samples=minimum_scenario_samples,
    )

    raw_scenario = empirical_scenario(
        portfolio_returns,
        scenario_horizon_sessions,
        minimum_samples=minimum_scenario_samples,
    )
    scenario = HistoricalScenarioRange(
        available=bool(raw_scenario["available"]),
        horizon_sessions=scenario_horizon_sessions,
        sample_n=int(raw_scenario["sample_n"]),
        return_p10=_finite_or_none(raw_scenario.get("return_p10")),
        return_p50=_finite_or_none(raw_scenario.get("return_p50")),
        return_p90=_finite_or_none(raw_scenario.get("return_p90")),
        historical_positive_rate=_finite_or_none(raw_scenario.get("historical_positive_rate")),
        method=str(raw_scenario.get("method", "历史滚动窗口不可用")),
    )
    ambition = None
    if annual_return_ambition_pct is not None:
        ambition = assess_return_ambition(
            portfolio_returns,
            annual_return_ambition_pct=annual_return_ambition_pct,
            holding_weeks=holding_weeks,
            minimum_samples=minimum_scenario_samples,
        )

    warnings: list[str] = []
    if (
        summary.historical_annual_volatility is not None
        and summary.historical_annual_volatility > profile.risk.target_annual_volatility_max
    ):
        warnings.append("历史代理波动率高于该档风险控制目标；不代表未来一定超限")
    if (
        summary.historical_max_drawdown is not None
        and abs(summary.historical_max_drawdown) > profile.risk.drawdown_de_risk
    ):
        warnings.append("历史代理回撤曾超过该档降风险阈值；不构成未来最大回撤保证")
    if not scenario.available:
        warnings.append("历史情景样本不足，未展示收益区间")
    if ambition is not None:
        if ambition.status == ReturnAmbitionStatus.INSUFFICIENT_EVIDENCE:
            warnings.append("收益期望缺少充分历史窗口；不判断目标是否可达")
        elif ambition.status == ReturnAmbitionStatus.UNSUPPORTED:
            warnings.append("所选收益期望未获当前历史重叠样本支持；不得据此承诺收益")
        elif ambition.status == ReturnAmbitionStatus.STRETCH:
            warnings.append("所选收益期望仅属历史拉伸情景，不是常态预期")
    return summary, scenario, ambition, tuple(warnings)


def _factor_warnings(
    profile: RiskProfile,
    coverage: tuple[FactorCoverage, ...],
) -> tuple[str, ...]:
    by_name = {item.factor: item for item in coverage}
    configured = {
        "fundamentals": profile.scoring.fundamentals + 0.50 * profile.scoring.quality_liquidity,
        "liquidity": profile.scoring.liquidity + 0.50 * profile.scoring.quality_liquidity,
        "news": profile.scoring.news + 0.30 * profile.scoring.trend_catalyst,
        "sector_context": profile.scoring.market_sector,
    }
    labels = {
        "fundamentals": "财务基本面",
        "liquidity": "流动性",
        "news": "公司新闻/公告",
        "sector_context": "板块环境",
    }
    warnings = []
    for factor, configured_weight in configured.items():
        status = by_name[factor]
        if configured_weight > 0.0 and not status.enabled:
            warnings.append(
                f"{labels[factor]}因子未启用（覆盖{status.provided}/{status.eligible}）；"
                "缺失权重已在真实可用因子间重新归一化，未填入中性假值"
            )
    return tuple(warnings)


def build_weekly_portfolios(
    histories: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    as_of: object | None = None,
    minimum_sessions: int = 252,
    holding_weeks: int = 4,
    annual_return_ambition_pct: int | None = None,
    scenario_horizon_sessions: int | None = None,
    minimum_scenario_samples: int = 30,
    minimum_correlation_overlap: int = 60,
    candidate_pool_size: int = 100,
    beam_width: int = 256,
    exclude_formation_limit_up: bool = True,
    exclude_overheated_acceleration: bool = True,
) -> WeeklyPortfolioBatch:
    """Build conservative, balanced and aggressive weekly research portfolios.

    Required metadata fields are ``industry``, ``is_st`` and ``is_delisting``.
    Optional normalized scores ``fundamental_score``, ``liquidity_score``,
    ``news_score`` and ``sector_score`` are in [0, 1].  The portfolio-level
    market regime is derived independently from common-cutoff cross-sectional
    breadth and is never treated as a per-stock ranking factor.
    An optional factor is enabled only with complete eligible-universe coverage;
    missing values are reported and never filled with an invented neutral score.
    Legacy ``quality_score``/``catalyst_score`` aliases remain accepted.

    ``is_limit_up_at_cutoff`` and ``is_buyable_at_cutoff`` may be supplied as
    explicit booleans by a licensed/current execution feed.  By default a stock
    that closes at its formation-date upper limit is removed from the actionable
    set, causing the constrained search to choose the next-best executable set.
    A separate conservative entry gate also freezes names with both a 20-session
    gain above 35% and a close more than 15% above their 20-session average.
    """

    if minimum_sessions < 121:
        raise ValueError("minimum_sessions must be at least 121")
    validated_holding_weeks = validate_holding_weeks(holding_weeks)
    validated_ambition = (
        None
        if annual_return_ambition_pct is None
        else validate_annual_return_ambition(annual_return_ambition_pct)
    )
    expected_horizon_sessions = validated_holding_weeks * 5
    if scenario_horizon_sessions is None:
        scenario_horizon_sessions = expected_horizon_sessions
    elif scenario_horizon_sessions != expected_horizon_sessions:
        raise ValueError("scenario_horizon_sessions must equal holding_weeks * 5")
    if minimum_scenario_samples < 1:
        raise ValueError("minimum_scenario_samples must be positive")
    if minimum_correlation_overlap < 20:
        raise ValueError("minimum_correlation_overlap must be at least 20")
    if candidate_pool_size < 4:
        raise ValueError("candidate_pool_size must be at least four")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")

    normalized_histories, normalized_metadata = _normalise_inputs(histories, metadata)
    cutoff = _choose_cutoff(normalized_histories, as_of)
    if cutoff is None:
        unavailable = tuple(
            WeeklyPortfolioResult(
                profile=profile,
                status=WeeklyPortfolioStatus.UNAVAILABLE,
                reasons=("no_valid_market_data_cutoff",),
            )
            for profile in RiskProfileName
        )
        return WeeklyPortfolioBatch(
            data_cutoff=None,
            portfolios=unavailable,
            exclusions=(),
            holding_weeks=validated_holding_weeks,
            annual_return_ambition_pct=validated_ambition,
        )

    market_regime = assess_market_regime(normalized_histories, cutoff)

    candidates: list[_Candidate] = []
    exclusions: list[ExcludedStock] = []
    for symbol, frame in sorted(normalized_histories.items()):
        candidate, reasons = _candidate_from_input(
            symbol,
            frame,
            normalized_metadata.get(symbol),
            cutoff,
            minimum_sessions,
            exclude_formation_limit_up,
            exclude_overheated_acceleration,
        )
        if candidate is None:
            exclusions.append(ExcludedStock(symbol=symbol, reasons=reasons))
        else:
            candidates.append(candidate)

    # Metadata without matching price history is also visible in the audit result.
    for symbol in sorted(set(normalized_metadata) - set(normalized_histories)):
        exclusions.append(ExcludedStock(symbol=symbol, reasons=("history_missing",)))

    factor_coverage = _factor_coverage(candidates)
    enabled_factors = frozenset(item.factor for item in factor_coverage if item.enabled)

    if market_regime.state == MarketRegimeState.RISK_OFF:
        unavailable = tuple(
            WeeklyPortfolioResult(
                profile=profile,
                status=WeeklyPortfolioStatus.UNAVAILABLE,
                reasons=("market_regime_risk_off_new_entries_paused",),
                risk_warnings=(
                    "全市场共同截止日宽度处于risk-off；低回撤规则暂停建立新的四股组合，保留现金。",
                ),
            )
            for profile in RiskProfileName
        )
        return WeeklyPortfolioBatch(
            data_cutoff=cutoff,
            portfolios=unavailable,
            exclusions=tuple(exclusions),
            holding_weeks=validated_holding_weeks,
            annual_return_ambition_pct=validated_ambition,
            factor_coverage=factor_coverage,
            market_regime=market_regime,
        )

    results: list[WeeklyPortfolioResult] = []
    for profile_name in RiskProfileName:
        if len(candidates) < 4:
            results.append(
                WeeklyPortfolioResult(
                    profile=profile_name,
                    status=WeeklyPortfolioStatus.UNAVAILABLE,
                    reasons=(f"only_{len(candidates)}_eligible_stocks;four_required",),
                )
            )
            continue

        profile = get_default_profile(profile_name)
        scored = _score_candidates(candidates, profile, enabled_factors)
        selection = _select_diversified_four(
            scored,
            profile,
            minimum_correlation_overlap=minimum_correlation_overlap,
            candidate_pool_size=candidate_pool_size,
            beam_width=beam_width,
        )
        if len(selection) != 4:
            results.append(
                WeeklyPortfolioResult(
                    profile=profile_name,
                    status=WeeklyPortfolioStatus.UNAVAILABLE,
                    reasons=("no_four_stock_set_passed_industry_and_correlation_constraints",),
                )
            )
            continue

        symbols = tuple(row[0].candidate.symbol for row in selection)
        allocation = allocate_four_stocks(symbols, profile)
        selected = tuple(
            SelectedStock(
                symbol=row[0].candidate.symbol,
                name=row[0].candidate.name,
                industry=row[0].candidate.industry,
                score=round(row[1], 6),
                component_scores=(
                    *row[0].components,
                    ("diversification", round(row[2], 12)),
                ),
                effective_weights=(
                    *row[0].effective_weights,
                    ("diversification", profile.scoring.diversification),
                ),
            )
            for row in selection
        )
        historical_risk, scenario, ambition, warnings = _historical_outputs(
            selection,
            allocation,
            scenario_horizon_sessions=scenario_horizon_sessions,
            minimum_scenario_samples=minimum_scenario_samples,
            holding_weeks=validated_holding_weeks,
            annual_return_ambition_pct=validated_ambition,
        )
        results.append(
            WeeklyPortfolioResult(
                profile=profile_name,
                status=WeeklyPortfolioStatus.READY,
                allocation=allocation,
                selected=selected,
                historical_risk=historical_risk,
                historical_scenario=scenario,
                return_ambition_assessment=ambition,
                risk_warnings=(*_factor_warnings(profile, factor_coverage), *warnings),
            )
        )

    return WeeklyPortfolioBatch(
        data_cutoff=cutoff,
        portfolios=tuple(results),
        exclusions=tuple(exclusions),
        holding_weeks=validated_holding_weeks,
        annual_return_ambition_pct=validated_ambition,
        factor_coverage=factor_coverage,
        market_regime=market_regime,
    )


def archive_weekly_portfolios(
    batch: WeeklyPortfolioBatch,
    histories: Mapping[str, pd.DataFrame],
    repository: SQLiteRepository,
) -> str | None:
    """Archive selected memberships; uncalibrated metrics remain NULL."""

    ready = [item for item in batch.portfolios if item.status == WeeklyPortfolioStatus.READY]
    if not ready or batch.data_cutoff is None:
        return None
    created_at = datetime.now(UTC)
    run_id = str(uuid4())
    frame_hashes: dict[str, str] = {}
    snapshots: list[dict[str, Any]] = []
    for symbol, frame in sorted(histories.items()):
        stable = frame.drop(columns=["retrieved_at"], errors="ignore")
        checksum = hashlib.sha256(
            stable.to_json(orient="split", date_format="iso").encode("utf-8")
        ).hexdigest()
        frame_hashes[symbol] = checksum
        dates = pd.to_datetime(frame.get("trade_date", frame.index))
        source = str(frame.iloc[-1].get("source", frame.attrs.get("provider", "unknown")))
        retrieved = frame.iloc[-1].get("retrieved_at", created_at)
        snapshots.append(
            {
                "id": str(uuid4()),
                "run_id": run_id,
                "source": source,
                "dataset": "daily_ohlcv",
                "symbol": symbol,
                "first_at": pd.Timestamp(dates.min()).date(),
                "last_at": pd.Timestamp(dates.max()).date(),
                "row_count": len(frame),
                "adjustment": "qfq",
                "unit_json": {"volume_shares": "share", "amount_cny": "CNY"},
                "checksum": checksum,
                "retrieved_at": retrieved,
                "is_stale": bool(frame.attrs.get("is_cache_fallback", False)),
            }
        )
    data_hash = hashlib.sha256(json.dumps(frame_hashes, sort_keys=True).encode("utf-8")).hexdigest()
    config_payload = (
        f"weekly-3-2-2-1-v0.3.0|holding_weeks={batch.holding_weeks}|"
        f"annual_ambition={batch.annual_return_ambition_pct or 'none'}"
    )
    config_hash = hashlib.sha256(config_payload.encode("utf-8")).hexdigest()
    sets: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    for portfolio in ready:
        assert portfolio.allocation is not None
        portfolio_id = str(uuid4())
        sets.append(
            {
                "id": portfolio_id,
                "run_id": run_id,
                "risk_profile": portfolio.profile.value,
                "cash_weight": portfolio.allocation.cash_ratio,
                "borrowed_weight": 0.0,
                # These stay null until genuine walk-forward calibration exists.
                "expected_return": None,
                "expected_vol": None,
                "expected_max_drawdown": None,
                "sharpe": None,
                "metric_window": "forecast_metrics_unavailable_until_walk_forward",
            }
        )
        selected = {item.symbol: item for item in portfolio.selected}
        for rank, position in enumerate(portfolio.allocation.positions, start=1):
            item = selected[position.ticker]
            members.append(
                {
                    "portfolio_id": portfolio_id,
                    "symbol": position.ticker,
                    "weight": position.weight,
                    "rank": rank,
                    "reason_json": {
                        "name": item.name,
                        "industry": item.industry,
                        "ranking_score_not_probability": item.score,
                    },
                }
            )
    repository.archive_run(
        {
            "id": run_id,
            "run_type": "weekly_portfolios",
            "as_of": batch.data_cutoff.date(),
            "data_cutoff": batch.data_cutoff.date(),
            "created_at": created_at,
            "strategy_version": "weekly-3-2-2-1-v0.2.0",
            "model_id": None,
            "config_hash": config_hash,
            "data_hash": data_hash,
            "status": "completed" if len(ready) == 3 else "partial",
            "warning_json": [
                NON_PROMISE_NOTICE,
                "当前组合统计为历史重叠描述；预测收益/波动/回撤字段保持空值，"
                "直到滚动样本外验证完成。",
            ],
        },
        data_snapshots=snapshots,
        portfolio_sets=sets,
        portfolio_members=members,
    )
    return run_id
