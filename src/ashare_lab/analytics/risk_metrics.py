from __future__ import annotations

import math

import numpy as np
import pandas as pd


def max_drawdown(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return float("nan")
    equity = (1 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def risk_metrics(returns: pd.Series, *, annual_sessions: int = 252) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if len(clean) < 20:
        return {
            key: float("nan")
            for key in ("cagr", "volatility", "sharpe", "sortino", "max_drawdown", "cvar95")
        }
    years = len(clean) / annual_sessions
    cagr = float((1 + clean).prod() ** (1 / years) - 1) if years > 0 else float("nan")
    volatility = float(clean.std(ddof=1) * math.sqrt(annual_sessions))
    sharpe = (
        float(clean.mean() / clean.std(ddof=1) * math.sqrt(annual_sessions))
        if clean.std(ddof=1)
        else float("nan")
    )
    downside = clean[clean < 0].std(ddof=1)
    sortino = (
        float(clean.mean() / downside * math.sqrt(annual_sessions))
        if downside and not np.isnan(downside)
        else float("nan")
    )
    cutoff = clean.quantile(0.05)
    cvar = float(clean[clean <= cutoff].mean())
    return {
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown(clean),
        "cvar95": cvar,
    }
