"""Descriptive portfolio statistics with explicit uncertainty labels.

The rolling windows in this module overlap.  They are useful for describing
the historical path, but they are neither independent observations nor a
walk-forward/out-of-sample forecast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import isfinite
from statistics import NormalDist

import numpy as np
import pandas as pd

from ashare_lab.analytics.risk_metrics import risk_metrics

HISTORICAL_OVERLAP_NOTICE = (
    "当前统计来自历史重叠滚动窗口，仅用于描述历史；不是walk-forward样本外结果、"
    "未来概率、收益承诺或最大回撤保证。Wilson区间也未消除重叠样本相关性。"
)


@dataclass(frozen=True, slots=True)
class WilsonInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    successes: int
    sample_n: int


@dataclass(frozen=True, slots=True)
class WindowDrawdownDistribution:
    window_sessions: int
    drawdown_budget: float
    available: bool
    sample_n: int
    effective_non_overlapping_n: int
    drawdown_magnitude_p10: float | None = None
    drawdown_magnitude_p50: float | None = None
    drawdown_magnitude_p90: float | None = None
    breach_probability: float | None = None
    breach_interval: WilsonInterval | None = None
    method: str = "历史重叠窗口内最大回撤幅度分布"
    is_out_of_sample: bool = False
    is_forecast_probability: bool = False
    is_promise: bool = False
    disclaimer: str = HISTORICAL_OVERLAP_NOTICE


@dataclass(frozen=True, slots=True)
class PortfolioStatistics:
    observation_count: int
    historical_cagr: float | None
    historical_annual_volatility: float | None
    historical_sharpe: float | None
    historical_sortino: float | None
    historical_calmar: float | None
    historical_max_drawdown: float | None
    historical_daily_cvar95: float | None
    drawdown_windows: tuple[WindowDrawdownDistribution, ...]
    method: str = "历史收盘收益的固定权重日频代理；包含重叠窗口，非样本外回测"
    is_out_of_sample: bool = False
    net_of_costs: bool = False
    is_promise: bool = False
    disclaimer: str = HISTORICAL_OVERLAP_NOTICE


def _clean_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    clean = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if clean.empty:
        return clean
    values = clean.to_numpy()
    if not bool(np.isfinite(values).all()):
        raise ValueError("returns must contain only finite values after missing data is removed")
    if bool((values <= -1.0).any()):
        raise ValueError("a simple return cannot be less than or equal to -100%")
    return clean


def _optional_finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def wilson_score_interval(
    successes: int,
    sample_n: int,
    *,
    confidence: float = 0.90,
) -> WilsonInterval:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if isinstance(successes, bool) or isinstance(sample_n, bool):
        raise TypeError("successes and sample_n must be integers")
    if not isinstance(successes, int) or not isinstance(sample_n, int):
        raise TypeError("successes and sample_n must be integers")
    if sample_n <= 0:
        raise ValueError("sample_n must be positive")
    if not 0 <= successes <= sample_n:
        raise ValueError("successes must be between zero and sample_n")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    estimate = successes / sample_n
    z_score = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z_squared = z_score * z_score
    denominator = 1.0 + z_squared / sample_n
    center = (estimate + z_squared / (2.0 * sample_n)) / denominator
    margin = (
        z_score
        * math.sqrt(
            estimate * (1.0 - estimate) / sample_n + z_squared / (4.0 * sample_n * sample_n)
        )
        / denominator
    )
    return WilsonInterval(
        estimate=estimate,
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
        confidence=confidence,
        successes=successes,
        sample_n=sample_n,
    )


def rolling_max_drawdown_magnitudes(
    returns: pd.Series,
    window_sessions: int,
) -> pd.Series:
    """Return positive maximum-drawdown magnitudes for overlapping windows."""

    if isinstance(window_sessions, bool) or not isinstance(window_sessions, int):
        raise TypeError("window_sessions must be an integer")
    if window_sessions < 2:
        raise ValueError("window_sessions must be at least two")
    clean = _clean_returns(returns)
    if len(clean) < window_sessions:
        return pd.Series(dtype=float, name=f"mdd_{window_sessions}")

    values = clean.to_numpy()
    magnitudes = np.empty(len(values) - window_sessions + 1, dtype=float)
    for index in range(len(magnitudes)):
        sample = values[index : index + window_sessions]
        equity = np.concatenate(([1.0], np.cumprod(1.0 + sample)))
        running_peak = np.maximum.accumulate(equity)
        drawdowns = equity / running_peak - 1.0
        magnitudes[index] = max(0.0, -float(drawdowns.min()))
    return pd.Series(magnitudes, name=f"mdd_{window_sessions}")


def calculate_portfolio_statistics(
    returns: pd.Series,
    *,
    drawdown_budget: float,
    window_sessions: tuple[int, ...] = (20, 40, 60),
    minimum_window_samples: int = 30,
    confidence: float = 0.90,
    annual_sessions: int = 252,
) -> PortfolioStatistics:
    """Calculate historical risk statistics without presenting them as forecasts."""

    if not 0.0 < drawdown_budget < 1.0:
        raise ValueError("drawdown_budget must be a positive fraction below one")
    if minimum_window_samples < 1:
        raise ValueError("minimum_window_samples must be positive")
    if annual_sessions < 1:
        raise ValueError("annual_sessions must be positive")
    if not window_sessions or len(set(window_sessions)) != len(window_sessions):
        raise ValueError("window_sessions must contain unique windows")

    clean = _clean_returns(returns)
    raw_metrics = risk_metrics(clean, annual_sessions=annual_sessions)
    cagr = _optional_finite(raw_metrics["cagr"])
    maximum_drawdown = _optional_finite(raw_metrics["max_drawdown"])
    calmar = None
    if cagr is not None and maximum_drawdown is not None and maximum_drawdown < 0.0:
        calmar = cagr / abs(maximum_drawdown)

    distributions: list[WindowDrawdownDistribution] = []
    for window in window_sessions:
        magnitudes = rolling_max_drawdown_magnitudes(clean, window)
        sample_n = len(magnitudes)
        effective_n = len(clean) // window
        if sample_n < minimum_window_samples:
            distributions.append(
                WindowDrawdownDistribution(
                    window_sessions=window,
                    drawdown_budget=drawdown_budget,
                    available=False,
                    sample_n=sample_n,
                    effective_non_overlapping_n=effective_n,
                )
            )
            continue

        p10, p50, p90 = magnitudes.quantile([0.10, 0.50, 0.90]).tolist()
        breaches = int((magnitudes > drawdown_budget).sum())
        interval = wilson_score_interval(breaches, sample_n, confidence=confidence)
        distributions.append(
            WindowDrawdownDistribution(
                window_sessions=window,
                drawdown_budget=drawdown_budget,
                available=True,
                sample_n=sample_n,
                effective_non_overlapping_n=effective_n,
                drawdown_magnitude_p10=float(p10),
                drawdown_magnitude_p50=float(p50),
                drawdown_magnitude_p90=float(p90),
                breach_probability=interval.estimate,
                breach_interval=interval,
            )
        )

    return PortfolioStatistics(
        observation_count=len(clean),
        historical_cagr=cagr,
        historical_annual_volatility=_optional_finite(raw_metrics["volatility"]),
        historical_sharpe=_optional_finite(raw_metrics["sharpe"]),
        historical_sortino=_optional_finite(raw_metrics["sortino"]),
        historical_calmar=_optional_finite(calmar),
        historical_max_drawdown=maximum_drawdown,
        historical_daily_cvar95=_optional_finite(raw_metrics["cvar95"]),
        drawdown_windows=tuple(distributions),
    )
