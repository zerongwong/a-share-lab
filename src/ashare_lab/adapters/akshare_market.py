"""AKShare daily-market adapter with an explicit, auditable cache fallback."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pandas as pd

from ashare_lab.adapters.csv_market import (
    _normalize_adjust,
    _utc_iso,
    _validate_date_range,
    canonicalize_daily_frame,
)
from ashare_lab.domain.errors import DataUnavailableError
from ashare_lab.ports.market_data import normalize_symbol


def _load_akshare() -> ModuleType:
    """Import AKShare only when a live fetch is actually requested."""

    try:
        return importlib.import_module("akshare")
    except ImportError as exc:
        raise DataUnavailableError(
            "未安装AKShare。可安装项目的[akshare]可选依赖，或明确选择CSV离线数据源。"
        ) from exc


class AKShareMarketData:
    """Fetch A-share daily bars from AKShare and cache only successful results.

    This adapter never changes to Yahoo, Tushare, CSV, or another vendor.  If a
    live AKShare request fails, it may use its *own* prior cache, with every row
    marked ``source=akshare:cache`` and a warning attached to ``DataFrame.attrs``.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        allow_cache_fallback: bool = True,
        max_cache_stale_days: int = 10,
        module_loader: Callable[[], object] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_cache_stale_days < 0:
            raise ValueError("max_cache_stale_days不能为负数。")
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        self.allow_cache_fallback = allow_cache_fallback
        self.max_cache_stale_days = max_cache_stale_days
        self.module_loader = module_loader or _load_akshare
        self.clock = clock or (lambda: datetime.now(UTC))

    def fetch_daily(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        normalized_symbol = normalize_symbol(symbol)
        start_date, end_date = _validate_date_range(start, end)
        adjustment = _normalize_adjust(adjust)
        retrieved_at = _utc_iso(self.clock())

        try:
            # Fetch a calendar buffer so prev_close for the first requested row
            # can be computed without exposing the buffer to callers.
            fetch_start = start_date - timedelta(days=30)
            akshare = self.module_loader()
            fetcher = getattr(akshare, "stock_zh_a_hist", None)
            if not callable(fetcher):
                raise DataUnavailableError("当前AKShare版本没有stock_zh_a_hist接口。")
            raw = fetcher(
                symbol=normalized_symbol,
                period="daily",
                start_date=fetch_start.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust="" if adjustment == "none" else adjustment,
            )
            live = canonicalize_daily_frame(
                raw,
                start=fetch_start,
                end=end_date,
                source="akshare",
                retrieved_at=retrieved_at,
                raw_volume_unit="lots",
            )
            lookahead_rows_dropped = live.attrs.get("lookahead_rows_dropped", 0)
            cache_warning = self._write_cache(normalized_symbol, adjustment, live)
            result = _slice_canonical(live, start_date, end_date)
            result.attrs.update(
                {
                    "provider": "akshare",
                    "data_quality": "live",
                    "is_cache_fallback": False,
                    "warning": cache_warning,
                    "symbol": normalized_symbol,
                    "adjustment": adjustment,
                    "as_of": end_date.isoformat(),
                    "lookahead_rows_dropped": lookahead_rows_dropped,
                }
            )
            return result
        except Exception as live_error:  # noqa: BLE001 - normalize adapter boundary
            return self._fallback_or_raise(
                normalized_symbol,
                start_date,
                end_date,
                adjustment,
                live_error,
            )

    def _cache_path(self, symbol: str, adjustment: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{symbol}_{adjustment}_daily.csv"

    def _write_cache(
        self,
        symbol: str,
        adjustment: str,
        frame: pd.DataFrame,
    ) -> str | None:
        path = self._cache_path(symbol, adjustment)
        if path is None:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(path)
            return None
        except Exception as exc:  # noqa: BLE001 - live data remains usable
            return f"CACHE_WRITE_FAILED: {type(exc).__name__}: {str(exc)[:160]}"

    def _fallback_or_raise(
        self,
        symbol: str,
        start: date,
        end: date,
        adjustment: str,
        live_error: Exception,
    ) -> pd.DataFrame:
        live_reason = f"{type(live_error).__name__}: {str(live_error)[:240]}"
        path = self._cache_path(symbol, adjustment)
        if not self.allow_cache_fallback:
            raise DataUnavailableError(
                f"AKShare实时请求失败，且缓存降级已关闭：{live_reason}"
            ) from live_error
        if path is None or not path.is_file():
            raise DataUnavailableError(
                f"AKShare实时请求失败，且没有同源缓存可用：{live_reason}"
            ) from live_error

        try:
            cached_raw = pd.read_csv(path)
            cached = canonicalize_daily_frame(
                cached_raw,
                start=start,
                end=end,
                source="akshare:cache",
                # Preserve the original retrieval time stored in each row.
                retrieved_at=None,
                raw_volume_unit="shares",
            )
            latest = pd.Timestamp(cached["trade_date"].max()).date()
            stale_days = (end - latest).days
            if stale_days > self.max_cache_stale_days:
                raise DataUnavailableError(
                    f"同源缓存最后交易日为{latest.isoformat()}，"
                    f"距请求截止日{stale_days}天，超过允许的"
                    f"{self.max_cache_stale_days}天。"
                )
        except Exception as cache_error:  # noqa: BLE001 - combine both failures
            raise DataUnavailableError(
                "AKShare实时请求失败，同源缓存也不可用。"
                f"实时错误：{live_reason}；缓存错误："
                f"{type(cache_error).__name__}: {str(cache_error)[:240]}"
            ) from live_error

        cached.attrs.update(
            {
                "provider": "akshare",
                "data_quality": "cached",
                "is_cache_fallback": True,
                "warning": f"LIVE_FETCH_FAILED_USING_EXPLICIT_CACHE: {live_reason}",
                "cache_path": str(path),
                "symbol": symbol,
                "adjustment": adjustment,
                "as_of": end.isoformat(),
                "cache_last_trade_date": latest.isoformat(),
                "cache_stale_calendar_days": stale_days,
            }
        )
        return cached


def _slice_canonical(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    result = frame.loc[frame["trade_date"].between(start_timestamp, end_timestamp)].copy()
    if result.empty:
        raise DataUnavailableError(f"{start.isoformat()}至{end.isoformat()}没有可用交易日数据。")
    result.reset_index(drop=True, inplace=True)
    return result
