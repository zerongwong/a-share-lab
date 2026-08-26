from __future__ import annotations

import numpy as np
import pandas as pd


def _forward_returns(returns: pd.Series, horizon_sessions: int) -> pd.Series:
    clean = returns.dropna().astype(float)
    return (1 + clean).rolling(horizon_sessions).apply(np.prod, raw=True).shift(
        -(horizon_sessions - 1)
    ) - 1


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, float(centre - half)), min(1.0, float(centre + half))


def empirical_scenario(
    returns: pd.Series,
    horizon_sessions: int,
    *,
    minimum_samples: int = 30,
) -> dict[str, object]:
    """Estimate historical forward-return distribution without an LLM.

    Overlapping windows are intentional for descriptive scenarios; the result is
    labelled uncalibrated until walk-forward prediction outcomes exist.
    """
    samples = _forward_returns(returns, horizon_sessions).dropna()
    if len(samples) < minimum_samples:
        return {
            "available": False,
            "sample_n": int(len(samples)),
            "method": "历史滚动窗口（样本不足）",
        }
    q10, q50, q90 = samples.quantile([0.1, 0.5, 0.9]).tolist()
    return {
        "available": True,
        "sample_n": int(len(samples)),
        "return_p10": float(q10),
        "return_p50": float(q50),
        "return_p90": float(q90),
        "historical_positive_rate": float((samples > 0).mean()),
        "method": "同一标的历史滚动窗口；尚未做样本外概率校准",
        "confidence": "低" if len(samples) < 100 else "中",
    }


def empirical_three_way_scenarios(
    returns: pd.Series,
    horizon_sessions: int,
    *,
    minimum_samples: int = 30,
) -> list[dict[str, object]]:
    """Describe historical up/sideways/down frequencies with uncertainty.

    The neutral band scales with observed daily volatility.  These frequencies
    are evidence, not calibrated forecasts; the method field makes that
    limitation explicit in both the UI and immutable archive.
    """
    clean = returns.dropna().astype(float)
    samples = _forward_returns(clean, horizon_sessions).dropna()
    if len(samples) < minimum_samples:
        return []
    daily_vol = float(clean.tail(252).std(ddof=1))
    neutral = max(0.02, 0.35 * daily_vol * np.sqrt(horizon_sessions))
    masks = {
        "up": samples > neutral,
        "sideways": samples.abs() <= neutral,
        "down": samples < -neutral,
    }
    rows: list[dict[str, object]] = []
    for label, mask in masks.items():
        conditional = samples[mask]
        count = int(mask.sum())
        low, high = _wilson_interval(count, len(samples))
        quantiles = conditional.quantile([0.1, 0.5, 0.9]).tolist() if count else [None] * 3
        rows.append(
            {
                "label": label,
                "probability_low": low,
                "probability_mid": count / len(samples),
                "probability_high": high,
                "return_p10": quantiles[0],
                "return_p50": quantiles[1],
                "return_p90": quantiles[2],
                "sample_n": int(len(samples)),
                "method": (
                    "同一标的历史前向滚动窗口 + Wilson区间；未做样本外校准，不等同未来真实概率"
                ),
                "calibration_version": None,
            }
        )
    return rows
