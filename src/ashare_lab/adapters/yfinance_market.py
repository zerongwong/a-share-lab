"""Explicit Yahoo Finance price adapter used when the user selects it.

Yahoo is not treated as a Chinese news or fundamentals source.  It is a
clearly-labelled, replaceable daily-price fallback for the personal MVP while
the domestic AKShare endpoint is unavailable on the current network.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from ashare_lab.adapters.csv_market import canonicalize_daily_frame
from ashare_lab.domain.errors import DataUnavailableError
from ashare_lab.ports.market_data import normalize_symbol


def yahoo_symbol(symbol: str) -> str:
    code = normalize_symbol(symbol)
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    if code.startswith(("5", "6", "9")):
        return f"{code}.SS"
    return f"{code}.SZ"


class YFinanceMarketData:
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        allow_cache_fallback: bool = True,
        max_cache_stale_days: int = 10,
        module_loader: Callable[[], object] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.allow_cache_fallback = allow_cache_fallback
        self.max_cache_stale_days = max_cache_stale_days
        self.module_loader = module_loader or (lambda: importlib.import_module("yfinance"))
        self.clock = clock or (lambda: datetime.now(UTC))

    def fetch_daily(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        code = normalize_symbol(symbol)
        if adjust not in {"qfq", "none", ""}:
            raise ValueError("Yahoo价格适配器只支持qfq或none，不会静默改口径")
        adjustment = "none" if adjust in {"none", ""} else "qfq"
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("clock必须返回带时区的datetime")
        shanghai_now = now.astimezone(ZoneInfo("Asia/Shanghai"))
        effective_end = end
        current_session_excluded = False
        if (
            end >= shanghai_now.date()
            and shanghai_now.time() < datetime.strptime("15:30", "%H:%M").time()
        ):
            effective_end = shanghai_now.date() - timedelta(days=1)
            current_session_excluded = True
        try:
            yf = self.module_loader()
            raw = yf.download(
                yahoo_symbol(code),
                start=(start - timedelta(days=30)).isoformat(),
                end=(effective_end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                timeout=20,
            )
            if raw is None or raw.empty:
                raise DataUnavailableError("Yahoo返回空行情")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [column[0] for column in raw.columns]
            raw = raw.reset_index()
            if adjustment == "qfq" and "Adj Close" in raw and "Close" in raw:
                factor = raw["Adj Close"].astype(float) / raw["Close"].astype(float)
                for column in ("Open", "High", "Low", "Close"):
                    raw[column] = raw[column].astype(float) * factor
            retrieved_at = now.astimezone(UTC).isoformat().replace("+00:00", "Z")
            canonical = canonicalize_daily_frame(
                raw,
                start=start,
                end=effective_end,
                source="yahoo_finance:explicit_price_fallback",
                retrieved_at=retrieved_at,
                raw_volume_unit="shares",
            )
            self._write_cache(code, adjustment, canonical)
            canonical.attrs.update(
                {
                    "provider": "yahoo_finance",
                    "data_quality": "live",
                    "is_cache_fallback": False,
                    "warning": (
                        "EXPLICIT_YAHOO_PRICE_SOURCE: 仅用于日线价格备用；"
                        "不作为A股新闻、公告或基本面来源。"
                    ),
                    "symbol": code,
                    "adjustment": adjustment,
                    "as_of": effective_end.isoformat(),
                    "requested_as_of": end.isoformat(),
                    "current_session_excluded": current_session_excluded,
                }
            )
            return canonical
        except Exception as live_error:  # noqa: BLE001
            return self._fallback_or_raise(code, start, effective_end, adjustment, live_error)

    def _cache_path(self, symbol: str, adjustment: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"yahoo_{symbol}_{adjustment}_daily.csv"

    def _write_cache(self, symbol: str, adjustment: str, frame: pd.DataFrame) -> None:
        path = self._cache_path(symbol, adjustment)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)

    def _fallback_or_raise(
        self,
        symbol: str,
        start: date,
        end: date,
        adjustment: str,
        live_error: Exception,
    ) -> pd.DataFrame:
        path = self._cache_path(symbol, adjustment)
        reason = f"{type(live_error).__name__}: {str(live_error)[:200]}"
        if not self.allow_cache_fallback or path is None or not path.exists():
            raise DataUnavailableError(f"Yahoo价格请求失败且无同源缓存：{reason}") from live_error
        cached = canonicalize_daily_frame(
            pd.read_csv(path),
            start=start,
            end=end,
            source="yahoo_finance:explicit_cache",
            retrieved_at=None,
            raw_volume_unit="shares",
        )
        latest = pd.Timestamp(cached["trade_date"].max()).date()
        stale_days = (end - latest).days
        if stale_days > self.max_cache_stale_days:
            raise DataUnavailableError(
                f"Yahoo同源缓存已过期{stale_days}天，拒绝冒充最新数据"
            ) from live_error
        cached.attrs.update(
            {
                "provider": "yahoo_finance",
                "data_quality": "cached",
                "is_cache_fallback": True,
                "warning": f"LIVE_FETCH_FAILED_USING_EXPLICIT_YAHOO_CACHE: {reason}",
                "symbol": symbol,
                "adjustment": adjustment,
                "as_of": end.isoformat(),
            }
        )
        return cached
