from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_OHLC = ("open", "high", "low", "close")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"行情缺少字段: {', '.join(missing)}")


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.astype(float).ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(frame: pd.DataFrame) -> pd.Series:
    _require_columns(frame, REQUIRED_OHLC)
    previous_close = frame["close"].shift(1)
    parts = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return parts.max(axis=1)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR using alpha=1/period."""
    tr = true_range(frame)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI; zero-loss windows are reported as 100."""
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = average_gain / average_loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    result = result.mask((average_loss == 0) & (average_gain > 0), 100.0)
    return result.mask((average_loss == 0) & (average_gain == 0), 50.0)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    fast_line = ema(close, fast)
    slow_line = ema(close, slow)
    dif = fast_line - slow_line
    dea = dif.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = 2 * (dif - dea)
    return pd.DataFrame({"macd_dif": dif, "macd_dea": dea, "macd_hist": histogram})


def enrich_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, REQUIRED_OHLC)
    result = frame.sort_values("trade_date").copy()
    close = result["close"].astype(float)
    for window in (5, 10, 20, 60, 120, 250):
        result[f"ma{window}"] = close.rolling(window).mean()
    result["atr14"] = atr(result, 14)
    result["rsi14"] = rsi(close, 14)
    result = result.join(macd(close))
    if "volume_shares" in result:
        result["volume_ma20"] = result["volume_shares"].astype(float).rolling(20).mean()
        result["volume_median20"] = result["volume_shares"].astype(float).rolling(20).median()
    result["return_1d"] = close.pct_change()
    result["return_20d"] = close.pct_change(20)
    result["return_60d"] = close.pct_change(60)
    result["prior_high_20"] = result["high"].shift(1).rolling(20).max()
    result["prior_high_60"] = result["high"].shift(1).rolling(60).max()
    result["prior_low_20"] = result["low"].shift(1).rolling(20).min()
    result["prior_low_60"] = result["low"].shift(1).rolling(60).min()
    return result
