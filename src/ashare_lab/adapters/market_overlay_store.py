"""Provider-isolated, fail-closed storage for verified daily overlays.

The CSMAR history is deliberately outside this store.  A provider response is
first written to a run-specific staging directory.  Invalid or incomplete runs
are atomically moved to quarantine.  A successful run is moved to an immutable
verified directory and becomes visible only after one atomic manifest update.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd

from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS, normalize_symbol

AssetKind = Literal["stocks", "indices"]

VERIFIED_MANIFEST_COLUMNS = (
    "run_id",
    "source_id",
    "adjustment",
    "trade_date",
    "previous_trade_date",
    "stock_count",
    "expected_stock_count",
    "stock_coverage_ratio",
    "stock_checksum",
    "stock_source_labels",
    "index_count",
    "index_checksum",
    "index_source_labels",
    "core_index_symbols",
    "latest_retrieved_at",
    "receipt_json",
    "receipt_checksum",
    "stock_file",
    "index_file",
    "verified_at",
)

_SAFE_PROVIDER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class StagedOverlayRun:
    run_id: str
    source_id: str
    trade_date: date
    path: Path


@dataclass(frozen=True, slots=True)
class VerifiedOverlaySummary:
    run_id: str
    source_id: str
    trade_date: date
    previous_trade_date: date
    stock_count: int
    expected_stock_count: int
    stock_coverage_ratio: float
    stock_checksum: str
    index_count: int
    index_checksum: str
    core_index_symbols: tuple[str, ...]
    latest_retrieved_at: str
    receipt_checksum: str
    stock_file: str
    index_file: str
    verified_at: str
    unchanged: bool = False


class MarketOverlayStore:
    """Keep unadjusted daily increments separate from immutable base history."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def begin_staging(
        self,
        *,
        source_id: str | Enum,
        trade_date: date,
        receipt: Mapping[str, Any] | None = None,
    ) -> StagedOverlayRun:
        source = _normalize_source_id(source_id)
        run_id = uuid4().hex
        path = self._provider_root(source) / "staging" / f"run={run_id}"
        path.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(
            {
                "run_id": run_id,
                "source_id": source,
                "adjustment": "none",
                "trade_date": trade_date.isoformat(),
                "receipt": _jsonable(receipt or {}),
            },
            path / "receipt.json",
        )
        return StagedOverlayRun(run_id, source, trade_date, path)

    def stage_asset(
        self,
        run: StagedOverlayRun,
        kind: AssetKind,
        frame: pd.DataFrame,
    ) -> Path:
        _require_live_staging_run(run)
        _validate_kind(kind)
        if frame is None:
            raise DataUnavailableError(f"{kind} provider frame is missing")
        path = run.path / f"{kind}.parquet"
        _atomic_write_parquet(frame.copy(), path)
        return path

    def update_staging_receipt(
        self,
        run: StagedOverlayRun,
        receipt: Mapping[str, Any],
    ) -> None:
        _require_live_staging_run(run)
        _atomic_write_json(
            {
                "run_id": run.run_id,
                "source_id": run.source_id,
                "adjustment": "none",
                "trade_date": run.trade_date.isoformat(),
                "receipt": _jsonable(receipt),
            },
            run.path / "receipt.json",
        )

    def quarantine(
        self,
        run: StagedOverlayRun,
        *,
        reason: str,
        failed_at: datetime,
    ) -> Path:
        _require_live_staging_run(run)
        timestamp = _aware_utc(failed_at, "failed_at")
        _atomic_write_json(
            {
                "run_id": run.run_id,
                "source_id": run.source_id,
                "trade_date": run.trade_date.isoformat(),
                "reason": str(reason)[:2000],
                "failed_at": _iso_utc(timestamp),
            },
            run.path / "failure.json",
        )
        destination = self._provider_root(run.source_id) / "quarantine" / f"run={run.run_id}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        run.path.replace(destination)
        return destination

    def commit_verified(
        self,
        run: StagedOverlayRun,
        *,
        stocks: pd.DataFrame,
        indices: pd.DataFrame,
        previous_trade_date: date,
        expected_stock_count: int,
        stock_coverage_ratio: float,
        core_index_symbols: Sequence[str],
        receipt: Mapping[str, Any],
        verified_at: datetime,
    ) -> VerifiedOverlaySummary:
        """Publish both assets and one manifest row as a single visible unit.

        The immutable run directory is installed first.  The manifest is then
        replaced atomically.  A crash before that final replace can leave an
        orphan run directory, but it cannot advance the verified cutoff.
        """

        _require_live_staging_run(run)
        if previous_trade_date >= run.trade_date:
            raise DataQualityError("previous_trade_date must precede trade_date")
        if expected_stock_count <= 0:
            raise DataQualityError("expected_stock_count must be positive")
        if not 0 <= stock_coverage_ratio <= 1:
            raise DataQualityError("stock_coverage_ratio must be between zero and one")
        timestamp = _aware_utc(verified_at, "verified_at")
        canonical_stocks = normalize_overlay_daily(
            stocks,
            expected_date=run.trade_date,
            source_id=run.source_id,
            asset_kind="stocks",
        )
        canonical_indices = normalize_overlay_daily(
            indices,
            expected_date=run.trade_date,
            source_id=run.source_id,
            asset_kind="indices",
        )
        normalized_core = _normalize_symbols(core_index_symbols)
        if not normalized_core:
            raise DataQualityError("core_index_symbols cannot be empty")
        missing_indices = set(normalized_core) - set(canonical_indices["symbol"])
        if missing_indices:
            raise DataQualityError(
                "core index cross-section is incomplete: " + ", ".join(sorted(missing_indices))
            )

        stock_checksum = _frame_checksum(canonical_stocks)
        index_checksum = _frame_checksum(canonical_indices)
        receipt_payload = _jsonable(receipt)
        receipt_json = json.dumps(receipt_payload, ensure_ascii=False, sort_keys=True)
        receipt_checksum = hashlib.sha256(receipt_json.encode("utf-8")).hexdigest()
        existing = self._manifest_row(run.source_id, run.trade_date)
        if existing is not None:
            if _same_verified_payload(
                existing,
                previous_trade_date=previous_trade_date,
                stock_checksum=stock_checksum,
                index_checksum=index_checksum,
                expected_stock_count=expected_stock_count,
                stock_coverage_ratio=stock_coverage_ratio,
                core_index_symbols=normalized_core,
            ):
                summary = _summary_from_manifest_row(existing, unchanged=True)
                shutil.rmtree(run.path)
                return summary
            previous = pd.Timestamp(existing["previous_trade_date"]).date()
            if previous != previous_trade_date:
                raise DataQualityError("existing overlay has a different continuity predecessor")

        # Replace raw staging artifacts with canonical, deterministic payloads.
        _atomic_write_parquet(canonical_stocks, run.path / "stocks.parquet")
        _atomic_write_parquet(canonical_indices, run.path / "indices.parquet")
        self.update_staging_receipt(run, receipt_payload)
        destination = (
            self._provider_root(run.source_id)
            / "verified"
            / f"trade_date={run.trade_date.isoformat()}"
            / f"run={run.run_id}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        run.path.replace(destination)
        stock_path = destination / "stocks.parquet"
        index_path = destination / "indices.parquet"
        latest_retrieved = max(
            canonical_stocks["retrieved_at"].max(),
            canonical_indices["retrieved_at"].max(),
        )
        row = {
            "run_id": run.run_id,
            "source_id": run.source_id,
            "adjustment": "none",
            "trade_date": pd.Timestamp(run.trade_date),
            "previous_trade_date": pd.Timestamp(previous_trade_date),
            "stock_count": int(canonical_stocks["symbol"].nunique()),
            "expected_stock_count": int(expected_stock_count),
            "stock_coverage_ratio": float(stock_coverage_ratio),
            "stock_checksum": stock_checksum,
            "stock_source_labels": "|".join(sorted(set(canonical_stocks["source"]))),
            "index_count": int(canonical_indices["symbol"].nunique()),
            "index_checksum": index_checksum,
            "index_source_labels": "|".join(sorted(set(canonical_indices["source"]))),
            "core_index_symbols": "|".join(normalized_core),
            "latest_retrieved_at": latest_retrieved,
            "receipt_json": receipt_json,
            "receipt_checksum": receipt_checksum,
            "stock_file": str(stock_path),
            "index_file": str(index_path),
            "verified_at": _iso_utc(timestamp),
        }
        manifest = self.read_verified_manifest()
        # Parquet may restore text columns as pandas' extension StringDtype.
        # Concatenating those with a one-row Python-object frame can ask pandas
        # to union unordered dictionary categories.  Convert to plain objects
        # before the atomic manifest upsert to keep multi-day appends stable.
        text_columns = {
            "run_id",
            "source_id",
            "adjustment",
            "stock_checksum",
            "stock_source_labels",
            "index_checksum",
            "index_source_labels",
            "core_index_symbols",
            "latest_retrieved_at",
            "receipt_json",
            "receipt_checksum",
            "stock_file",
            "index_file",
            "verified_at",
        }
        for column in text_columns:
            if column in manifest:
                manifest[column] = manifest[column].astype(object)
        manifest = pd.concat(
            (manifest, pd.DataFrame([row], columns=list(VERIFIED_MANIFEST_COLUMNS))),
            ignore_index=True,
        )
        manifest = (
            manifest.drop_duplicates(["source_id", "adjustment", "trade_date"], keep="last")
            .sort_values(["source_id", "trade_date"])
            .reset_index(drop=True)
        )
        _atomic_write_parquet(manifest, self.root / "verified_manifest.parquet")
        return _summary_from_manifest_row(pd.Series(row), unchanged=False)

    def read_verified_manifest(
        self,
        *,
        source_id: str | Enum | None = None,
    ) -> pd.DataFrame:
        """Return verified rows only; dates are normalized pandas timestamps."""

        path = self.root / "verified_manifest.parquet"
        if not path.is_file():
            return pd.DataFrame(columns=list(VERIFIED_MANIFEST_COLUMNS))
        frame = pd.read_parquet(path)
        missing = set(VERIFIED_MANIFEST_COLUMNS) - set(frame.columns)
        if missing:
            raise DataQualityError("overlay manifest is missing: " + ", ".join(sorted(missing)))
        output = frame.loc[:, list(VERIFIED_MANIFEST_COLUMNS)].copy()
        output["trade_date"] = (
            pd.to_datetime(output["trade_date"]).dt.normalize().astype("datetime64[ns]")
        )
        output["previous_trade_date"] = (
            pd.to_datetime(output["previous_trade_date"]).dt.normalize().astype("datetime64[ns]")
        )
        if source_id is not None:
            source = _normalize_source_id(source_id)
            output = output.loc[output["source_id"] == source]
        return output.sort_values(["source_id", "trade_date"]).reset_index(drop=True)

    def latest_verified_date(self, source_id: str | Enum) -> date | None:
        manifest = self.read_verified_manifest(source_id=source_id)
        if manifest.empty:
            return None
        return pd.Timestamp(manifest["trade_date"].max()).date()

    def verified_dates_from(
        self,
        *,
        source_id: str | Enum,
        baseline_cutoff: date,
        through_date: date | None = None,
    ) -> tuple[date, ...]:
        """Return the continuous provider-confirmed chain after a base cutoff."""

        manifest = self.read_verified_manifest(source_id=source_id)
        current = baseline_cutoff
        chain: list[date] = []
        while True:
            candidate = manifest.loc[
                pd.to_datetime(manifest["previous_trade_date"]).dt.date == current
            ]
            if through_date is not None:
                candidate = candidate.loc[
                    pd.to_datetime(candidate["trade_date"]).dt.date <= through_date
                ]
            if candidate.empty:
                break
            if len(candidate) != 1:
                raise DataQualityError("overlay manifest contains a branched continuity chain")
            next_date = pd.Timestamp(candidate.iloc[0]["trade_date"]).date()
            if next_date <= current:
                raise DataQualityError("overlay manifest continuity is not strictly increasing")
            chain.append(next_date)
            current = next_date
        return tuple(chain)

    def read_verified_daily(
        self,
        trade_date: date,
        *,
        source_id: str | Enum,
        asset_kind: AssetKind,
    ) -> pd.DataFrame:
        """Read canonical rows; trade_date is datetime64[ns], retrieved_at is UTC text."""

        _validate_kind(asset_kind)
        source = _normalize_source_id(source_id)
        row = self._manifest_row(source, trade_date)
        if row is None:
            raise DataUnavailableError(f"no verified {source} overlay for {trade_date.isoformat()}")
        path = Path(str(row["stock_file"] if asset_kind == "stocks" else row["index_file"]))
        if not path.is_file():
            raise DataUnavailableError(f"verified overlay file is missing: {path}")
        frame = normalize_overlay_daily(
            pd.read_parquet(path),
            expected_date=trade_date,
            source_id=source,
            asset_kind=asset_kind,
        )
        expected_checksum = str(
            row["stock_checksum"] if asset_kind == "stocks" else row["index_checksum"]
        )
        if _frame_checksum(frame) != expected_checksum:
            raise DataQualityError("verified overlay checksum mismatch")
        return frame

    def _provider_root(self, source_id: str) -> Path:
        return self.root / f"source={source_id}" / "adjust=none"

    def _manifest_row(self, source_id: str, trade_date: date) -> pd.Series | None:
        manifest = self.read_verified_manifest(source_id=source_id)
        selected = manifest.loc[pd.to_datetime(manifest["trade_date"]).dt.date == trade_date]
        if selected.empty:
            return None
        if len(selected) != 1:
            raise DataQualityError("overlay manifest contains duplicate verified rows")
        return selected.iloc[0]


def normalize_overlay_daily(
    frame: pd.DataFrame,
    *,
    expected_date: date,
    source_id: str | Enum,
    asset_kind: AssetKind,
) -> pd.DataFrame:
    """Normalize one stock or core-index cross-section without mixing vendors."""

    _validate_kind(asset_kind)
    source = _normalize_source_id(source_id)
    if frame is None or frame.empty:
        raise DataUnavailableError(f"{expected_date.isoformat()} {asset_kind} overlay is empty")
    required = {"symbol", *CANONICAL_DAILY_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise DataQualityError("overlay is missing columns: " + ", ".join(sorted(missing)))
    output = frame.loc[:, ["symbol", *CANONICAL_DAILY_COLUMNS]].copy()
    try:
        output["symbol"] = output["symbol"].map(_normalize_symbol_cell)
    except ValueError as exc:
        raise DataQualityError(f"overlay contains invalid symbol: {exc}") from exc
    if bool(output["symbol"].duplicated().any()):
        raise DataQualityError(f"{asset_kind} overlay contains duplicate symbols")

    dates = pd.to_datetime(output["trade_date"], errors="coerce")
    if bool(dates.isna().any()):
        raise DataQualityError("overlay contains invalid trade_date")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    output["trade_date"] = dates.dt.normalize().astype("datetime64[ns]")
    if not bool((output["trade_date"] == pd.Timestamp(expected_date)).all()):
        raise DataQualityError("overlay contains rows outside the target trade_date")

    for column in ("open", "high", "low", "close", "prev_close"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    for column in ("volume_shares", "amount_cny", "turnover_pct"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    required_numeric = [
        "open",
        "high",
        "low",
        "close",
        "prev_close",
        "volume_shares",
        "amount_cny",
    ]
    if bool(output[required_numeric].isna().any(axis=None)):
        raise DataQualityError("overlay OHLC/previous-close/volume/amount must be explicit")
    if bool((output[["open", "high", "low", "close", "prev_close"]] <= 0).any(axis=None)):
        raise DataQualityError("overlay prices must be positive")
    if bool((output[["volume_shares", "amount_cny"]] < 0).any(axis=None)):
        raise DataQualityError("overlay volume and amount cannot be negative")
    if bool((output["turnover_pct"].dropna() < 0).any()):
        raise DataQualityError("overlay turnover_pct cannot be negative")
    if bool((output["high"] < output[["open", "low", "close"]].max(axis=1)).any()):
        raise DataQualityError("overlay high is below another price")
    if bool((output["low"] > output[["open", "high", "close"]].min(axis=1)).any()):
        raise DataQualityError("overlay low is above another price")

    labels = output["source"].fillna("").astype(str).str.strip()
    if bool(labels.eq("").any()):
        raise DataQualityError("overlay source is missing")
    if not bool(labels.map(lambda value: value == source or value.startswith(f"{source}:")).all()):
        raise DataQualityError(f"overlay source must belong to {source}")
    output["source"] = labels
    retrieved = pd.to_datetime(output["retrieved_at"], errors="coerce", utc=True)
    if bool(retrieved.isna().any()):
        raise DataQualityError("overlay contains invalid retrieved_at")
    output["retrieved_at"] = retrieved.map(_iso_utc)
    return output.sort_values("symbol").reset_index(drop=True)


def _same_verified_payload(
    row: pd.Series,
    *,
    previous_trade_date: date,
    stock_checksum: str,
    index_checksum: str,
    expected_stock_count: int,
    stock_coverage_ratio: float,
    core_index_symbols: Sequence[str],
) -> bool:
    return bool(
        pd.Timestamp(row["previous_trade_date"]).date() == previous_trade_date
        and str(row["stock_checksum"]) == stock_checksum
        and str(row["index_checksum"]) == index_checksum
        and int(row["expected_stock_count"]) == expected_stock_count
        and abs(float(row["stock_coverage_ratio"]) - stock_coverage_ratio) < 1e-12
        and tuple(str(row["core_index_symbols"]).split("|")) == tuple(core_index_symbols)
    )


def _summary_from_manifest_row(
    row: pd.Series,
    *,
    unchanged: bool,
) -> VerifiedOverlaySummary:
    return VerifiedOverlaySummary(
        run_id=str(row["run_id"]),
        source_id=str(row["source_id"]),
        trade_date=pd.Timestamp(row["trade_date"]).date(),
        previous_trade_date=pd.Timestamp(row["previous_trade_date"]).date(),
        stock_count=int(row["stock_count"]),
        expected_stock_count=int(row["expected_stock_count"]),
        stock_coverage_ratio=float(row["stock_coverage_ratio"]),
        stock_checksum=str(row["stock_checksum"]),
        index_count=int(row["index_count"]),
        index_checksum=str(row["index_checksum"]),
        core_index_symbols=tuple(str(row["core_index_symbols"]).split("|")),
        latest_retrieved_at=str(row["latest_retrieved_at"]),
        receipt_checksum=str(row["receipt_checksum"]),
        stock_file=str(row["stock_file"]),
        index_file=str(row["index_file"]),
        verified_at=str(row["verified_at"]),
        unchanged=unchanged,
    )


def _frame_checksum(frame: pd.DataFrame) -> str:
    # retrieved_at is audit metadata, not market content.  Excluding it makes a
    # byte-for-byte market rerun idempotent while the receipt still preserves
    # the time and trace of the first verified retrieval.
    payload_frame = frame.drop(columns=["retrieved_at"]).copy()
    payload_frame["trade_date"] = payload_frame["trade_date"].dt.strftime("%Y-%m-%d")
    payload = pd.util.hash_pandas_object(payload_frame, index=False).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _normalize_symbols(values: Sequence[str]) -> tuple[str, ...]:
    try:
        symbols = tuple(dict.fromkeys(_normalize_symbol_cell(value) for value in values))
    except ValueError as exc:
        raise DataQualityError(f"invalid configured symbol: {exc}") from exc
    return tuple(sorted(symbols))


def _normalize_symbol_cell(value: object) -> str:
    text = str(value).strip().upper()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return normalize_symbol(text.zfill(6))


def _normalize_source_id(value: str | Enum) -> str:
    raw = value.value if isinstance(value, Enum) else value
    normalized = str(raw).strip().lower()
    if _SAFE_PROVIDER.fullmatch(normalized) is None:
        raise ValueError("source_id must be a safe lowercase provider identifier")
    return normalized


def _validate_kind(value: str) -> None:
    if value not in {"stocks", "indices"}:
        raise ValueError("asset_kind must be stocks or indices")


def _require_live_staging_run(run: StagedOverlayRun) -> None:
    if not run.path.is_dir():
        raise DataUnavailableError(f"staging run is no longer available: {run.run_id}")


def _aware_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _iso_utc(value: datetime | pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)
    return timestamp.isoformat().replace("+00:00", "Z")


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(value: Mapping[str, Any], path: Path) -> None:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
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
