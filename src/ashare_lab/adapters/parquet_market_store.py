"""Provider-isolated Parquet storage for date-major full-market data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from ashare_lab.domain.data_sources import SourceId
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.bulk_market_data import CANONICAL_SECURITY_MASTER_COLUMNS
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS, normalize_symbol

CROSS_SECTION_MANIFEST_COLUMNS = (
    "source_id",
    "adjustment",
    "trade_date",
    "symbol_count",
    "checksum",
    "source_labels",
    "latest_retrieved_at",
    "file_path",
)


@dataclass(frozen=True, slots=True)
class StoredCrossSectionSummary:
    source_id: SourceId
    adjustment: str
    trade_date: date
    symbol_count: int
    checksum: str
    source_labels: tuple[str, ...]
    latest_retrieved_at: str
    file_path: str


class ParquetMarketStore:
    """Store one atomic Parquet file per provider and trading session.

    A full-market API response is written once, rather than split into 5,000
    tiny writes.  Provider and adjustment are explicit path partitions, so two
    vendors or price definitions can never be silently mixed.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def write_security_master(
        self,
        frame: pd.DataFrame,
        *,
        source_id: SourceId | str,
        as_of: date,
    ) -> Path:
        source = SourceId(source_id)
        normalized = normalize_security_master(frame, source_id=source, as_of=as_of)
        directory = self.root / "security_master" / f"source={source.value}"
        snapshot = directory / f"as_of={as_of.isoformat()}.parquet"
        _atomic_write_parquet(normalized, snapshot)
        _atomic_write_parquet(normalized, directory / "latest.parquet")
        return snapshot

    def read_security_master(
        self,
        source_id: SourceId | str,
        *,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        source = SourceId(source_id)
        directory = self.root / "security_master" / f"source={source.value}"
        path = directory / (f"as_of={as_of.isoformat()}.parquet" if as_of else "latest.parquet")
        if not path.is_file():
            raise DataUnavailableError(f"本地没有 {source.value} 的证券主表：{path}")
        return pd.read_parquet(path)

    def write_daily_cross_section(
        self,
        trade_date: date,
        frame: pd.DataFrame,
        *,
        source_id: SourceId | str,
        adjustment: str,
    ) -> StoredCrossSectionSummary:
        source = SourceId(source_id)
        normalized_adjustment = _normalize_adjustment(adjustment)
        normalized = normalize_daily_cross_section(frame, expected_date=trade_date)
        path = self.daily_cross_section_path(
            trade_date,
            source_id=source,
            adjustment=normalized_adjustment,
        )
        _atomic_write_parquet(normalized, path)
        return _summary_for_cross_section(
            normalized,
            path=path,
            source_id=source,
            adjustment=normalized_adjustment,
            trade_date=trade_date,
        )

    def read_daily_cross_section(
        self,
        trade_date: date,
        *,
        source_id: SourceId | str,
        adjustment: str,
    ) -> pd.DataFrame:
        source = SourceId(source_id)
        path = self.daily_cross_section_path(
            trade_date,
            source_id=source,
            adjustment=adjustment,
        )
        if not path.is_file():
            raise DataUnavailableError(
                f"本地没有 {source.value}/{adjustment}/{trade_date.isoformat()} 的全市场日线"
            )
        return normalize_daily_cross_section(pd.read_parquet(path), expected_date=trade_date)

    def daily_cross_section_path(
        self,
        trade_date: date,
        *,
        source_id: SourceId | str,
        adjustment: str,
    ) -> Path:
        source = SourceId(source_id)
        normalized_adjustment = _normalize_adjustment(adjustment)
        return (
            self.root
            / "daily"
            / f"source={source.value}"
            / f"adjust={normalized_adjustment}"
            / f"trade_date={trade_date.isoformat()}"
            / "data.parquet"
        )

    def available_trade_dates(
        self,
        *,
        source_id: SourceId | str,
        adjustment: str,
    ) -> tuple[date, ...]:
        source = SourceId(source_id)
        normalized_adjustment = _normalize_adjustment(adjustment)
        manifest = self.read_cross_section_manifest()
        selected = manifest.loc[
            (manifest["source_id"] == source.value)
            & (manifest["adjustment"] == normalized_adjustment)
        ]
        return tuple(sorted({pd.Timestamp(value).date() for value in selected["trade_date"]}))

    def read_daily_market(
        self,
        *,
        source_id: SourceId | str,
        adjustment: str,
        start: date,
        end: date,
        symbols: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        if start > end:
            raise ValueError("start不能晚于end")
        dates = [
            value
            for value in self.available_trade_dates(source_id=source_id, adjustment=adjustment)
            if start <= value <= end
        ]
        if not dates:
            raise DataUnavailableError(f"{start.isoformat()}至{end.isoformat()}没有本地日线")
        wanted = {normalize_symbol(symbol) for symbol in symbols} if symbols else None
        frames: list[pd.DataFrame] = []
        for trade_date in dates:
            path = self.daily_cross_section_path(
                trade_date,
                source_id=source_id,
                adjustment=adjustment,
            )
            frame = pd.read_parquet(path)
            if wanted is not None:
                frame = frame.loc[frame["symbol"].astype(str).isin(wanted)]
            if not frame.empty:
                frames.append(frame)
        if not frames:
            raise DataUnavailableError("本地日线中没有请求的股票")
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["symbol", "trade_date"])
            .reset_index(drop=True)
        )

    def build_symbol_coverage(
        self,
        *,
        source_id: SourceId | str,
        adjustment: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Aggregate only symbol/date columns for a lightweight quality gate."""

        dates = [
            value
            for value in self.available_trade_dates(source_id=source_id, adjustment=adjustment)
            if start <= value <= end
        ]
        if not dates:
            return pd.DataFrame(
                columns=["symbol", "first_trade_date", "last_trade_date", "row_count"]
            )
        frames = [
            pd.read_parquet(
                self.daily_cross_section_path(
                    trade_date,
                    source_id=source_id,
                    adjustment=adjustment,
                ),
                columns=["symbol", "trade_date"],
            )
            for trade_date in dates
        ]
        rows = pd.concat(frames, ignore_index=True)
        rows["trade_date"] = pd.to_datetime(rows["trade_date"]).dt.normalize()
        return (
            rows.groupby("symbol", as_index=False)
            .agg(
                first_trade_date=("trade_date", "min"),
                last_trade_date=("trade_date", "max"),
                row_count=("trade_date", "nunique"),
            )
            .sort_values("symbol")
            .reset_index(drop=True)
        )

    def read_cross_section_manifest(self) -> pd.DataFrame:
        path = self.root / "cross_section_manifest.parquet"
        if not path.is_file():
            return pd.DataFrame(columns=list(CROSS_SECTION_MANIFEST_COLUMNS))
        frame = pd.read_parquet(path)
        missing = set(CROSS_SECTION_MANIFEST_COLUMNS) - set(frame.columns)
        if missing:
            raise DataQualityError("日截面清单缺少字段：" + ", ".join(sorted(missing)))
        return frame.loc[:, list(CROSS_SECTION_MANIFEST_COLUMNS)].copy()

    def upsert_cross_section_manifest(self, summaries: Iterable[StoredCrossSectionSummary]) -> None:
        rows = [_manifest_row(summary) for summary in summaries]
        if not rows:
            return
        incoming = pd.DataFrame(rows, columns=list(CROSS_SECTION_MANIFEST_COLUMNS))
        existing = self.read_cross_section_manifest()
        combined = pd.concat((existing, incoming), ignore_index=True)
        combined = (
            combined.drop_duplicates(["source_id", "adjustment", "trade_date"], keep="last")
            .sort_values(["source_id", "adjustment", "trade_date"])
            .reset_index(drop=True)
        )
        _atomic_write_parquet(combined, self.root / "cross_section_manifest.parquet")

    def write_sync_report(
        self,
        summary: Mapping[str, Any] | Any,
        symbol_details: pd.DataFrame,
        date_details: pd.DataFrame,
        *,
        source_id: SourceId | str,
        completed_at: datetime,
    ) -> tuple[Path, Path, Path]:
        source = SourceId(source_id)
        timestamp = completed_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        directory = self.root / "reports" / f"source={source.value}"
        json_path = directory / f"sync_{timestamp}.json"
        symbol_path = directory / f"sync_{timestamp}_symbols.parquet"
        date_path = directory / f"sync_{timestamp}_dates.parquet"
        payload = json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True)
        _atomic_write_text(payload, json_path)
        _atomic_write_parquet(symbol_details, symbol_path)
        _atomic_write_parquet(date_details, date_path)
        _atomic_write_text(payload, directory / "latest.json")
        _atomic_write_parquet(symbol_details, directory / "latest_symbols.parquet")
        _atomic_write_parquet(date_details, directory / "latest_dates.parquet")
        return json_path, symbol_path, date_path


def normalize_security_master(
    frame: pd.DataFrame,
    *,
    source_id: SourceId,
    as_of: date,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise DataUnavailableError("证券主表为空")
    missing = set(CANONICAL_SECURITY_MASTER_COLUMNS) - set(frame.columns)
    if missing:
        raise DataQualityError("证券主表缺少字段：" + ", ".join(sorted(missing)))
    output = frame.loc[:, list(CANONICAL_SECURITY_MASTER_COLUMNS)].copy()
    try:
        output["symbol"] = output["symbol"].map(_normalize_symbol_cell)
    except ValueError as exc:
        raise DataQualityError(f"证券主表包含无效代码：{exc}") from exc
    if bool(output["symbol"].duplicated().any()):
        raise DataQualityError("证券主表包含重复代码")

    output["name"] = output["name"].astype(str).str.strip()
    output["exchange"] = output["exchange"].astype(str).str.upper().str.strip()
    output["board"] = output["board"].fillna("").astype(str).str.strip()
    output["industry"] = output["industry"].fillna("").astype(str).str.strip()
    if bool((output["name"] == "").any()):
        raise DataQualityError("证券主表存在空名称")
    if not bool(output["exchange"].isin({"SH", "SZ", "BJ"}).all()):
        raise DataQualityError("证券主表exchange只能是SH、SZ或BJ")

    output["list_date"] = pd.to_datetime(output["list_date"], errors="coerce").dt.normalize()
    if bool(output["list_date"].isna().any()):
        raise DataQualityError("证券主表存在无效上市日期")
    output["delist_date"] = pd.to_datetime(output["delist_date"], errors="coerce").dt.normalize()
    impossible = output["delist_date"].notna() & (output["delist_date"] < output["list_date"])
    if bool(impossible.any()):
        raise DataQualityError("证券主表存在退市日早于上市日")
    if bool((output["list_date"] > pd.Timestamp(as_of)).any()):
        raise DataQualityError("证券主表包含分析截止日之后才上市的证券")

    for column in ("is_st", "is_delisting", "is_suspended"):
        explicit = output[column].map(lambda value: isinstance(value, (bool, np.bool_)))
        if not bool(explicit.all()):
            raise DataQualityError(f"证券主表{column}必须为显式布尔值")
        output[column] = output[column].astype(bool)

    if not bool((output["source"].astype(str) == source_id.value).all()):
        raise DataQualityError(f"证券主表source必须全部为{source_id.value}，禁止混用供应商")
    retrieved = pd.to_datetime(output["retrieved_at"], errors="coerce", utc=True)
    if bool(retrieved.isna().any()):
        raise DataQualityError("证券主表存在无效retrieved_at")
    output["retrieved_at"] = retrieved.map(lambda value: value.isoformat().replace("+00:00", "Z"))
    return output.sort_values("symbol").reset_index(drop=True)


def normalize_daily_cross_section(
    frame: pd.DataFrame,
    *,
    expected_date: date,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise DataUnavailableError(f"{expected_date.isoformat()} 全市场日线为空")
    required = {"symbol", *CANONICAL_DAILY_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise DataQualityError("全市场日线缺少字段：" + ", ".join(sorted(missing)))
    output = frame.loc[:, ["symbol", *CANONICAL_DAILY_COLUMNS]].copy()
    try:
        output["symbol"] = output["symbol"].map(_normalize_symbol_cell)
    except ValueError as exc:
        raise DataQualityError(f"全市场日线包含无效代码：{exc}") from exc
    if bool(output["symbol"].duplicated().any()):
        raise DataQualityError(f"{expected_date.isoformat()} 日截面包含重复股票")

    dates = pd.to_datetime(output["trade_date"], errors="coerce")
    if bool(dates.isna().any()):
        raise DataQualityError("全市场日线存在无效交易日期")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    output["trade_date"] = dates.dt.normalize()
    if not bool((output["trade_date"] == pd.Timestamp(expected_date)).all()):
        raise DataQualityError("日截面包含目标交易日以外的数据")

    for column in ("open", "high", "low", "close", "prev_close"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    for column in ("volume_shares", "amount_cny", "turnover_pct"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if bool(output[["open", "high", "low", "close"]].isna().any(axis=None)):
        raise DataQualityError("全市场日线开高低收存在空值或非数值")
    if bool((output[["open", "high", "low", "close"]] <= 0).any(axis=None)):
        raise DataQualityError("全市场日线开高低收必须大于0")
    if bool((output["volume_shares"].dropna() < 0).any()):
        raise DataQualityError("全市场日线成交量不能为负")
    if bool((output["amount_cny"].dropna() < 0).any()):
        raise DataQualityError("全市场日线成交额不能为负")
    if bool((output["turnover_pct"].dropna() < 0).any()):
        raise DataQualityError("全市场日线换手率不能为负")
    if bool((output["high"] < output[["open", "low", "close"]].max(axis=1)).any()):
        raise DataQualityError("全市场日线最高价低于其他价格")
    if bool((output["low"] > output[["open", "high", "close"]].min(axis=1)).any()):
        raise DataQualityError("全市场日线最低价高于其他价格")
    if bool(output["source"].fillna("").astype(str).str.strip().eq("").any()):
        raise DataQualityError("全市场日线缺少source")
    retrieved = pd.to_datetime(output["retrieved_at"], errors="coerce", utc=True)
    if bool(retrieved.isna().any()):
        raise DataQualityError("全市场日线存在无效retrieved_at")
    output["retrieved_at"] = retrieved.map(lambda value: value.isoformat().replace("+00:00", "Z"))
    return output.sort_values("symbol").reset_index(drop=True)


def _summary_for_cross_section(
    frame: pd.DataFrame,
    *,
    path: Path,
    source_id: SourceId,
    adjustment: str,
    trade_date: date,
) -> StoredCrossSectionSummary:
    checksum_frame = frame.copy()
    checksum_frame["trade_date"] = checksum_frame["trade_date"].dt.strftime("%Y-%m-%d")
    payload = pd.util.hash_pandas_object(checksum_frame, index=False).values.tobytes()
    return StoredCrossSectionSummary(
        source_id=source_id,
        adjustment=adjustment,
        trade_date=trade_date,
        symbol_count=int(frame["symbol"].nunique()),
        checksum=hashlib.sha256(payload).hexdigest(),
        source_labels=tuple(sorted(set(frame["source"].astype(str)))),
        latest_retrieved_at=max(frame["retrieved_at"].astype(str)),
        file_path=str(path),
    )


def _manifest_row(summary: StoredCrossSectionSummary) -> dict[str, object]:
    return {
        "source_id": summary.source_id.value,
        "adjustment": summary.adjustment,
        "trade_date": summary.trade_date,
        "symbol_count": summary.symbol_count,
        "checksum": summary.checksum,
        "source_labels": "|".join(summary.source_labels),
        "latest_retrieved_at": summary.latest_retrieved_at,
        "file_path": summary.file_path,
    }


def _normalize_symbol_cell(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return normalize_symbol(text.zfill(6))


def _normalize_adjustment(value: str) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if normalized not in {"none", "qfq", "hfq"}:
        raise ValueError("adjustment只能是none、qfq或hfq")
    return normalized


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
