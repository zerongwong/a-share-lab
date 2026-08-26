from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

CANONICAL_DAILY_COLUMNS = (
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume_shares",
    "amount_cny",
    "turnover_pct",
    "source",
    "retrieved_at",
)


@runtime_checkable
class MarketDataPort(Protocol):
    def fetch_daily(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """Return canonical A-share daily bars, inclusive of start/end."""


def normalize_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    for suffix in (".SS", ".SH", ".SZ", ".BJ"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    if not (len(cleaned) == 6 and cleaned.isdigit()):
        raise ValueError("请输入6位A股代码，例如 600150")
    return cleaned
