"""Offline CSV implementation of the market-data port.

The adapter never reaches the network and never falls through to another
provider.  It accepts either canonical English columns or a common AKShare
CSV export, then returns one strict schema with volume expressed in shares.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS, normalize_symbol

VolumeUnit = Literal["shares", "lots"]
Adjustment = Literal["none", "qfq", "hfq"]

_COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "trade_date": ("trade_date", "date", "Date", "日期", "交易日期"),
    "open": ("open", "Open", "开盘"),
    "high": ("high", "High", "最高"),
    "low": ("low", "Low", "最低"),
    "close": ("close", "Close", "收盘"),
    "prev_close": ("prev_close", "Preclose", "昨收", "前收盘"),
    "volume_shares": ("volume_shares", "volume", "Volume", "成交量"),
    "amount_cny": ("amount_cny", "amount", "Amount", "成交额"),
    "turnover_pct": ("turnover_pct", "turnover", "Turnover", "换手率"),
}


class CSVMarketData:
    """Read a deterministic local CSV file or a directory of symbol files."""

    def __init__(
        self,
        path: str | Path,
        *,
        adjustment: Adjustment = "qfq",
        volume_unit: VolumeUnit = "shares",
        source_label: str = "csv:offline",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.adjustment = _normalize_adjust(adjustment)
        self.volume_unit = _validate_volume_unit(volume_unit)
        self.source_label = source_label
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
        requested_adjustment = _normalize_adjust(adjust)
        if requested_adjustment != self.adjustment:
            raise DataUnavailableError(
                "CSV复权口径不匹配："
                f"文件声明为 {self.adjustment}，请求为 {requested_adjustment}。"
                "系统不会静默改用其他口径。"
            )

        csv_path = self._resolve_file(normalized_symbol, requested_adjustment)
        try:
            raw = pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise DataUnavailableError(f"无法读取离线行情文件 {csv_path}: {exc}") from exc

        raw = _filter_symbol_column(raw, normalized_symbol)
        retrieved_at = _utc_iso(self.clock())
        result = canonicalize_daily_frame(
            raw,
            start=start_date,
            end=end_date,
            source=self.source_label,
            retrieved_at=retrieved_at,
            raw_volume_unit=self.volume_unit,
        )
        result.attrs.update(
            {
                "provider": "csv",
                "data_quality": "offline",
                "is_cache_fallback": False,
                "warning": "OFFLINE_CSV_DATA: 这是明确选择的离线数据，不是实时行情。",
                "file_path": str(csv_path),
                "symbol": normalized_symbol,
                "adjustment": requested_adjustment,
                "as_of": end_date.isoformat(),
            }
        )
        return result

    def _resolve_file(self, symbol: str, adjustment: str) -> Path:
        if self.path.is_file():
            return self.path
        if not self.path.exists():
            raise DataUnavailableError(f"离线行情路径不存在：{self.path}")
        if not self.path.is_dir():
            raise DataUnavailableError(f"离线行情路径不是CSV文件或目录：{self.path}")

        candidates = (
            self.path / f"{symbol}_{adjustment}_daily.csv",
            self.path / f"{symbol}_daily.csv",
            self.path / f"{symbol}.csv",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        names = ", ".join(path.name for path in candidates)
        raise DataUnavailableError(
            f"没有找到 {symbol} 的离线行情文件；只检查了：{names}。系统不会静默切换到网络数据源。"
        )


def canonicalize_daily_frame(
    raw: pd.DataFrame,
    *,
    start: date,
    end: date,
    source: str,
    retrieved_at: str | None,
    raw_volume_unit: VolumeUnit,
) -> pd.DataFrame:
    """Normalize common Chinese/English daily bars into the port schema.

    ``prev_close`` is calculated before applying ``start`` so a provider may
    request a short lookback buffer without leaking rows into the result.
    Rows after ``end`` are always discarded before any value is returned.
    """

    start_date, end_date = _validate_date_range(start, end)
    volume_unit = _validate_volume_unit(raw_volume_unit)
    if raw is None or raw.empty:
        raise DataUnavailableError("行情数据为空。")
    if isinstance(raw.columns, pd.MultiIndex):
        raise DataQualityError("行情列是多层索引，无法确认字段含义。")

    date_column = _find_column(raw, "trade_date", required=True)
    parsed_dates = pd.to_datetime(raw[date_column], errors="coerce")
    invalid_dates = parsed_dates.isna()
    if invalid_dates.any():
        raise DataQualityError(f"行情中有 {int(invalid_dates.sum())} 行无效交易日期。")
    if parsed_dates.dt.tz is not None:
        parsed_dates = parsed_dates.dt.tz_localize(None)
    parsed_dates = parsed_dates.dt.normalize()

    end_timestamp = pd.Timestamp(end_date)
    future_mask = parsed_dates > end_timestamp
    working = raw.loc[~future_mask].copy()
    working["__trade_date"] = parsed_dates.loc[~future_mask]
    if working.empty:
        raise DataUnavailableError(f"{end_date.isoformat()}及以前没有可用行情。")

    output = pd.DataFrame(index=working.index)
    output["trade_date"] = working["__trade_date"]
    for field in ("open", "high", "low", "close"):
        column = _find_column(working, field, required=True)
        output[field] = _strict_numeric(working[column], field)

    volume_column = _find_column(working, "volume_shares", required=True)
    volume = _strict_numeric(working[volume_column], "volume_shares")
    # A canonical volume_shares column is authoritative.  Raw AKShare/Chinese
    # exports use lots (手), while generic CSV `volume` follows the adapter's
    # explicit volume_unit setting.
    if volume_column != "volume_shares" and volume_unit == "lots":
        volume = volume * 100.0
    output["volume_shares"] = volume.round().astype("Int64")

    amount_column = _find_column(working, "amount_cny", required=False)
    output["amount_cny"] = (
        _optional_numeric(working[amount_column], "amount_cny")
        if amount_column is not None
        else float("nan")
    )
    turnover_column = _find_column(working, "turnover_pct", required=False)
    output["turnover_pct"] = (
        _optional_numeric(working[turnover_column], "turnover_pct")
        if turnover_column is not None
        else float("nan")
    )

    previous_column = _find_column(working, "prev_close", required=False)
    if previous_column is None:
        output["prev_close"] = output["close"].shift(1)
    else:
        previous = pd.to_numeric(working[previous_column], errors="coerce")
        output["prev_close"] = previous.fillna(output["close"].shift(1))

    output = output.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    _validate_prices_and_units(output)

    start_timestamp = pd.Timestamp(start_date)
    output = output.loc[output["trade_date"].between(start_timestamp, end_timestamp)].copy()
    if output.empty:
        raise DataUnavailableError(
            f"{start_date.isoformat()}至{end_date.isoformat()}没有可用交易日数据。"
        )

    output["source"] = source
    if retrieved_at is None and "retrieved_at" in working.columns:
        retrieval_by_date = pd.Series(
            working["retrieved_at"].astype(str).to_numpy(),
            index=working["__trade_date"],
        )
        output["retrieved_at"] = output["trade_date"].map(retrieval_by_date)
    else:
        output["retrieved_at"] = retrieved_at or _utc_iso(datetime.now(UTC))

    output = output.loc[:, list(CANONICAL_DAILY_COLUMNS)].reset_index(drop=True)
    output.attrs["lookahead_rows_dropped"] = int(future_mask.sum())
    return output


def _filter_symbol_column(frame: pd.DataFrame, requested_symbol: str) -> pd.DataFrame:
    symbol_column = next(
        (column for column in ("symbol", "ticker", "代码", "股票代码") if column in frame.columns),
        None,
    )
    if symbol_column is None:
        return frame

    def normalize_or_none(value: object) -> str | None:
        try:
            # CSV type inference may turn 000001 into integer 1.
            text = str(value).strip()
            if text.endswith(".0") and text[:-2].isdigit():
                text = text[:-2]
            return normalize_symbol(text.zfill(6))
        except (TypeError, ValueError):
            return None

    normalized = frame[symbol_column].map(normalize_or_none)
    result = frame.loc[normalized == requested_symbol].copy()
    if result.empty:
        raise DataUnavailableError(
            f"CSV中没有股票 {requested_symbol}；系统不会返回其他股票的数据。"
        )
    return result


def _find_column(frame: pd.DataFrame, field: str, *, required: bool) -> str | None:
    column = next((name for name in _COLUMN_ALIASES[field] if name in frame.columns), None)
    if column is None and required:
        aliases = ", ".join(_COLUMN_ALIASES[field])
        raise DataQualityError(f"缺少字段 {field}；可识别列名：{aliases}")
    return column


def _strict_numeric(values: pd.Series, field: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = numeric.isna()
    if invalid.any():
        raise DataQualityError(f"字段 {field} 有 {int(invalid.sum())} 个非数值或空值。")
    return numeric.astype(float)


def _optional_numeric(values: pd.Series, field: str) -> pd.Series:
    """Parse an optional numeric column while preserving genuine missing data.

    Canonical cache files always contain ``amount_cny`` and ``turnover_pct``.
    Providers such as Yahoo do not supply those fields, so a cache round trip
    serializes them as empty cells.  Empty cells are valid for these optional
    columns; non-empty, non-numeric text remains a data-quality error.
    """

    numeric = pd.to_numeric(values, errors="coerce")
    missing = values.isna() | values.astype(str).str.strip().isin(("", "nan", "NaN", "None"))
    invalid = numeric.isna() & ~missing
    if invalid.any():
        raise DataQualityError(f"字段 {field} 有 {int(invalid.sum())} 个非数值。")
    return numeric.astype(float)


def _validate_prices_and_units(frame: pd.DataFrame) -> None:
    if (frame[["open", "high", "low", "close"]] <= 0).any(axis=None):
        raise DataQualityError("开高低收必须全部大于0。")
    if (frame["volume_shares"] < 0).any():
        raise DataQualityError("成交量不能为负数。")
    amount = frame["amount_cny"].dropna()
    if (amount < 0).any():
        raise DataQualityError("成交额不能为负数。")
    turnover = frame["turnover_pct"].dropna()
    if (turnover < 0).any():
        raise DataQualityError("换手率不能为负数。")

    price_tolerance = 1e-9
    if (frame["high"] + price_tolerance < frame[["open", "close", "low"]].max(axis=1)).any():
        raise DataQualityError("最高价低于开盘、收盘或最低价。")
    if (frame["low"] - price_tolerance > frame[["open", "close", "high"]].min(axis=1)).any():
        raise DataQualityError("最低价高于开盘、收盘或最高价。")


def _validate_date_range(start: date, end: date) -> tuple[date, date]:
    if isinstance(start, datetime):
        start = start.date()
    if isinstance(end, datetime):
        end = end.date()
    if not isinstance(start, date) or not isinstance(end, date):
        raise TypeError("start和end必须是date。")
    if start > end:
        raise ValueError("start不能晚于end。")
    return start, end


def _normalize_adjust(adjust: str) -> Adjustment:
    normalized = adjust.strip().lower() if isinstance(adjust, str) else ""
    if normalized == "":
        normalized = "none"
    if normalized not in {"none", "qfq", "hfq"}:
        raise ValueError("adjust只能是none、qfq或hfq。")
    return normalized  # type: ignore[return-value]


def _validate_volume_unit(value: str) -> VolumeUnit:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if normalized not in {"shares", "lots"}:
        raise ValueError("volume_unit只能是shares或lots。")
    return normalized  # type: ignore[return-value]


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("clock必须返回带时区的datetime。")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
