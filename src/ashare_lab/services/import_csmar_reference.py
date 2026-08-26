"""Resumable import of CSMAR balance-sheet snapshots and historical indices."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from ashare_lab.adapters.csmar_reference import (
    BALANCE_OUTPUT_COLUMNS,
    INDEX_OUTPUT_COLUMNS,
    CSMARReferenceData,
    CSMARReferenceEntry,
    CSMARReferenceReader,
    normalize_balance_sheet_snapshot,
    normalize_index_daily,
)
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError

REFERENCE_STATE_SCHEMA_VERSION = 1
CSMAR_REFERENCE_RETRIEVED_AT = date(2026, 8, 25)


@dataclass(frozen=True, slots=True)
class CSMARReferenceImportReport:
    status: str
    source_root: str
    output_root: str
    source_fingerprint: str
    common_cutoff_date: date
    retrieved_at: date
    completed_entries: int
    balance_sheet_rows: int
    balance_sheet_symbols: int
    index_daily_rows: int
    index_count: int
    index_invalid_price_rows_quarantined: int
    index_invalid_geometry_rows_quarantined: int
    index_invalid_activity_rows_quarantined: int
    first_index_date: date | None
    last_index_date: date | None
    balance_sheet_data_role: str
    balance_sheet_historical_backtest_eligible: bool
    index_historical_backtest_eligible: bool
    duckdb_path: str
    state_path: str
    warnings: tuple[str, ...]


def import_csmar_reference_data(
    source_root: str | Path,
    output_root: str | Path,
    *,
    common_cutoff_date: date,
    retrieved_at: date = CSMAR_REFERENCE_RETRIEVED_AT,
    batch_size: int = 50_000,
) -> CSMARReferenceImportReport:
    """Import only the new reference ZIPs into an independent local dataset.

    ``common_cutoff_date`` must be the same cutoff used by the stock-daily
    dataset.  This service never opens that dataset for writing and refuses an
    output directory that looks like the existing stock-daily import.
    """

    if batch_size <= 0:
        raise ValueError("batch_size必须大于0")
    if common_cutoff_date > retrieved_at:
        raise ValueError("common_cutoff_date不能晚于retrieved_at")

    reader = CSMARReferenceReader(source_root)
    destination = Path(output_root).expanduser().resolve()
    _guard_independent_output(destination)
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "reference_import_state.json"

    if state_path.is_file():
        state = _read_json(state_path)
        _validate_resume_state(
            state,
            reader=reader,
            destination=destination,
            common_cutoff_date=common_cutoff_date,
            retrieved_at=retrieved_at,
        )
    else:
        state = _new_state(
            reader=reader,
            destination=destination,
            common_cutoff_date=common_cutoff_date,
            retrieved_at=retrieved_at,
        )
        _atomic_write_json(state, state_path)

    state = _adopt_committed_entries(
        state,
        entries=reader.layout.entries,
        destination=destination,
        state_path=state_path,
    )
    for entry in reader.layout.entries:
        if entry.entry_id in state["completed_entries"]:
            continue
        state = _import_entry(
            reader,
            entry,
            destination=destination,
            state=state,
            state_path=state_path,
            common_cutoff_date=common_cutoff_date,
            retrieved_at=retrieved_at,
            batch_size=batch_size,
        )

    report = _build_catalog_and_report(
        reader=reader,
        destination=destination,
        state=state,
        state_path=state_path,
        common_cutoff_date=common_cutoff_date,
        retrieved_at=retrieved_at,
    )
    state["status"] = "complete"
    state["report"] = _jsonable(asdict(report))
    _atomic_write_json(state, state_path)
    return report


def open_csmar_reference_data(dataset_root: str | Path) -> CSMARReferenceData:
    """Return the read-only query adapter for a completed import."""

    return CSMARReferenceData(dataset_root)


def _import_entry(
    reader: CSMARReferenceReader,
    entry: CSMARReferenceEntry,
    *,
    destination: Path,
    state: dict[str, Any],
    state_path: Path,
    common_cutoff_date: date,
    retrieved_at: date,
    batch_size: int,
) -> dict[str, Any]:
    working_state = json.loads(json.dumps(state))
    dataset_name = "balance_sheet" if entry.kind == "balance_sheet" else "index_daily"
    dataset_root = destination / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    final_directory = dataset_root / f"entry={entry.entry_id}"
    if final_directory.exists():
        raise DataQualityError(f"发现未登记的已存在参考数据分片：{final_directory}；请检查断点清单")
    staging = dataset_root / f".entry={entry.entry_id}.{uuid4().hex}.tmp"
    staging.mkdir(parents=True, exist_ok=False)
    part_number = 0
    row_count = 0
    invalid_price_rows_quarantined = 0
    invalid_geometry_rows_quarantined = 0
    invalid_activity_rows_quarantined = 0
    try:
        for raw in reader.iter_raw_batches(entry, batch_size=batch_size):
            if entry.kind == "balance_sheet":
                normalized = normalize_balance_sheet_snapshot(
                    raw,
                    common_cutoff_date=common_cutoff_date,
                    retrieved_at=retrieved_at,
                )
            else:
                normalized = normalize_index_daily(
                    raw,
                    common_cutoff_date=common_cutoff_date,
                    retrieved_at=retrieved_at,
                )
                invalid_price_rows_quarantined += int(
                    normalized.attrs.get("invalid_price_rows_dropped", 0)
                )
                invalid_geometry_rows_quarantined += int(
                    normalized.attrs.get("invalid_geometry_rows_dropped", 0)
                )
                invalid_activity_rows_quarantined += int(
                    normalized.attrs.get("invalid_activity_rows_dropped", 0)
                )
            if normalized.empty:
                continue
            _atomic_write_parquet(normalized, staging / f"part-{part_number:05d}.parquet")
            row_count += int(len(normalized))
            part_number += 1

        if part_number == 0:
            columns = (
                BALANCE_OUTPUT_COLUMNS if entry.kind == "balance_sheet" else INDEX_OUTPUT_COLUMNS
            )
            _atomic_write_parquet(
                pd.DataFrame(columns=list(columns)),
                staging / "part-00000.parquet",
            )
            part_number = 1

        working_state["completed_entries"].append(entry.entry_id)
        working_state["entry_reports"][entry.entry_id] = {
            "kind": entry.kind,
            "display_name": entry.display_name,
            "archive_sha256": entry.archive_sha256,
            "crc32": entry.crc32,
            "parts": part_number,
            "rows": row_count,
            "invalid_price_rows_quarantined": invalid_price_rows_quarantined,
            "invalid_geometry_rows_quarantined": invalid_geometry_rows_quarantined,
            "invalid_activity_rows_quarantined": invalid_activity_rows_quarantined,
        }
        commit_record = {
            "schema_version": REFERENCE_STATE_SCHEMA_VERSION,
            "source_fingerprint": working_state["source_fingerprint"],
            "entry_id": entry.entry_id,
            "state_after": working_state,
        }
        _atomic_write_json(commit_record, staging / "entry_commit.json")
        staging.replace(final_directory)
        _atomic_write_json(working_state, state_path)
        return working_state
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _build_catalog_and_report(
    *,
    reader: CSMARReferenceReader,
    destination: Path,
    state: dict[str, Any],
    state_path: Path,
    common_cutoff_date: date,
    retrieved_at: date,
) -> CSMARReferenceImportReport:
    try:
        import duckdb
    except ImportError as exc:
        raise DataUnavailableError("缺少duckdb依赖，无法建立CSMAR参考数据库") from exc

    balance_glob = destination / "balance_sheet" / "entry=*" / "part-*.parquet"
    index_glob = destination / "index_daily" / "entry=*" / "part-*.parquet"
    database_path = destination / "csmar_reference.duckdb"
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute(
            f"CREATE OR REPLACE VIEW balance_sheet_raw AS "
            f"SELECT * FROM read_parquet('{_sql_literal(balance_glob)}', union_by_name=true)"
        )
        connection.execute(
            f"CREATE OR REPLACE VIEW index_daily_raw AS "
            f"SELECT * FROM read_parquet('{_sql_literal(index_glob)}', union_by_name=true)"
        )
        _validate_catalog_metadata(
            connection,
            common_cutoff_date=common_cutoff_date,
            retrieved_at=retrieved_at,
        )
        _fail_on_conflicting_balance_rows(connection)
        _fail_on_conflicting_index_rows(connection)
        connection.execute(
            "CREATE OR REPLACE VIEW balance_sheet_snapshot AS "
            "SELECT * EXCLUDE (_dedupe_rank) FROM ("
            "SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol, report_period "
            "ORDER BY source, retrieved_at) AS _dedupe_rank FROM balance_sheet_raw"
            ") WHERE _dedupe_rank = 1"
        )
        connection.execute(
            "CREATE OR REPLACE VIEW index_daily AS "
            "SELECT * EXCLUDE (_dedupe_rank) FROM ("
            "SELECT *, ROW_NUMBER() OVER (PARTITION BY index_code, trade_date "
            "ORDER BY source, retrieved_at) AS _dedupe_rank FROM index_daily_raw"
            ") WHERE _dedupe_rank = 1"
        )
        connection.execute(
            "CREATE OR REPLACE TABLE reference_dataset_metadata AS SELECT "
            "?::DATE AS common_cutoff_date, ?::DATE AS retrieved_at, "
            "'current_snapshot' AS balance_sheet_data_role, "
            "false AS balance_sheet_historical_backtest_eligible, "
            "true AS index_historical_backtest_eligible",
            [common_cutoff_date, retrieved_at],
        )
        balance_rows, balance_symbols = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol) FROM balance_sheet_snapshot"
        ).fetchone()
        index_rows, index_count, first_date, last_date = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT index_code), MIN(trade_date), MAX(trade_date) "
            "FROM index_daily"
        ).fetchone()
    finally:
        connection.close()

    if not balance_rows:
        raise DataUnavailableError("CSMAR资产负债表在共同截止日前没有可用快照")
    if not index_rows:
        raise DataUnavailableError("CSMAR指数在共同截止日前没有可用日线")

    index_invalid_price_rows_quarantined = sum(
        int(item.get("invalid_price_rows_quarantined", 0))
        for item in state["entry_reports"].values()
        if item.get("kind") == "index_daily"
    )
    index_invalid_geometry_rows_quarantined = sum(
        int(item.get("invalid_geometry_rows_quarantined", 0))
        for item in state["entry_reports"].values()
        if item.get("kind") == "index_daily"
    )
    index_invalid_activity_rows_quarantined = sum(
        int(item.get("invalid_activity_rows_quarantined", 0))
        for item in state["entry_reports"].values()
        if item.get("kind") == "index_daily"
    )
    warnings = [
        "FS_Combas缺少普通财报公告日，只能作为"
        f"{retrieved_at.isoformat()}取得的当前快照，禁止用于历史回测。",
        "DeclareDate仅是差错更正披露日，不能替代普通财报公告日。",
        "本模块只导入资产负债表原始字段，不生成或冒充完整fundamental_score。",
        "指数日线可按交易日做PIT研究，但必须与个股日线使用同一common_cutoff_date。",
        "原始CSMAR ZIP受许可约束，本模块只读访问且不上传或再分发。",
    ]
    if index_invalid_geometry_rows_quarantined:
        warnings.append(
            f"有{index_invalid_geometry_rows_quarantined}条指数记录因原始OHLC几何不可能被隔离；"
            "未修复、未进入本地指数库。"
        )
    if index_invalid_price_rows_quarantined:
        warnings.append(
            f"有{index_invalid_price_rows_quarantined}条指数记录因原始价格缺失或为负被隔离；"
            "未填补、未进入本地指数库。"
        )
    if index_invalid_activity_rows_quarantined:
        warnings.append(
            f"有{index_invalid_activity_rows_quarantined}条指数记录因原始成交量或成交额为负"
            "被隔离；未修复、未进入本地指数库。"
        )

    return CSMARReferenceImportReport(
        status="complete",
        source_root=str(reader.layout.root),
        output_root=str(destination),
        source_fingerprint=reader.layout.fingerprint,
        common_cutoff_date=common_cutoff_date,
        retrieved_at=retrieved_at,
        completed_entries=len(state["completed_entries"]),
        balance_sheet_rows=int(balance_rows),
        balance_sheet_symbols=int(balance_symbols),
        index_daily_rows=int(index_rows),
        index_count=int(index_count),
        index_invalid_price_rows_quarantined=index_invalid_price_rows_quarantined,
        index_invalid_geometry_rows_quarantined=index_invalid_geometry_rows_quarantined,
        index_invalid_activity_rows_quarantined=index_invalid_activity_rows_quarantined,
        first_index_date=_as_date(first_date),
        last_index_date=_as_date(last_date),
        balance_sheet_data_role="current_snapshot",
        balance_sheet_historical_backtest_eligible=False,
        index_historical_backtest_eligible=True,
        duckdb_path=str(database_path),
        state_path=str(state_path),
        warnings=tuple(warnings),
    )


def _validate_catalog_metadata(
    connection,
    *,
    common_cutoff_date: date,
    retrieved_at: date,
) -> None:
    invalid_balance = connection.execute(
        "SELECT COUNT(*) FROM balance_sheet_raw WHERE "
        "data_role <> 'current_snapshot' OR historical_backtest_eligible "
        "OR common_cutoff_date <> ? OR retrieved_at <> ?",
        [common_cutoff_date, retrieved_at.isoformat()],
    ).fetchone()[0]
    if invalid_balance:
        raise DataQualityError("资产负债表快照的PIT安全元数据不一致")
    invalid_index = connection.execute(
        "SELECT COUNT(*) FROM index_daily_raw WHERE "
        "data_role <> 'historical_point_in_time' OR NOT historical_backtest_eligible "
        "OR common_cutoff_date <> ? OR retrieved_at <> ? OR knowledge_date <> trade_date",
        [common_cutoff_date, retrieved_at.isoformat()],
    ).fetchone()[0]
    if invalid_index:
        raise DataQualityError("指数日线的PIT元数据或共同截止日不一致")


def _fail_on_conflicting_balance_rows(connection) -> None:
    conflicts = connection.execute(
        "SELECT symbol, report_period FROM balance_sheet_raw "
        "GROUP BY symbol, report_period HAVING COUNT(DISTINCT struct_pack("
        "name := name, is_correction := is_correction, "
        "correction_disclosure_dates := correction_disclosure_dates, "
        "cash_cny := cash_cny, accounts_receivable_cny := accounts_receivable_cny, "
        "inventory_cny := inventory_cny, current_assets_cny := current_assets_cny, "
        "total_assets_cny := total_assets_cny, current_liabilities_cny := current_liabilities_cny, "
        "total_liabilities_cny := total_liabilities_cny, parent_equity_cny := parent_equity_cny, "
        "total_equity_cny := total_equity_cny)) > 1 LIMIT 5"
    ).fetchall()
    if conflicts:
        raise DataQualityError(f"CSMAR资产负债表存在冲突重复记录：{conflicts}")


def _fail_on_conflicting_index_rows(connection) -> None:
    conflicts = connection.execute(
        "SELECT index_code, trade_date FROM index_daily_raw "
        "GROUP BY index_code, trade_date HAVING COUNT(DISTINCT struct_pack("
        "open := open, high := high, low := low, close := close, "
        "component_volume := component_volume, component_amount_cny := component_amount_cny, "
        "index_return := index_return)) > 1 LIMIT 5"
    ).fetchall()
    if conflicts:
        raise DataQualityError(f"CSMAR指数日线存在冲突重复记录：{conflicts}")


def _new_state(
    *,
    reader: CSMARReferenceReader,
    destination: Path,
    common_cutoff_date: date,
    retrieved_at: date,
) -> dict[str, Any]:
    return {
        "schema_version": REFERENCE_STATE_SCHEMA_VERSION,
        "status": "in_progress",
        "source_root": str(reader.layout.root),
        "output_root": str(destination),
        "source_fingerprint": reader.layout.fingerprint,
        "common_cutoff_date": common_cutoff_date.isoformat(),
        "retrieved_at": retrieved_at.isoformat(),
        "completed_entries": [],
        "entry_reports": {},
    }


def _validate_resume_state(
    state: dict[str, Any],
    *,
    reader: CSMARReferenceReader,
    destination: Path,
    common_cutoff_date: date,
    retrieved_at: date,
) -> None:
    if state.get("schema_version") != REFERENCE_STATE_SCHEMA_VERSION:
        raise DataQualityError("CSMAR参考数据断点版本不受支持")
    if state.get("source_fingerprint") != reader.layout.fingerprint:
        raise DataQualityError("CSMAR参考数据原始ZIP哈希已改变，拒绝续传")
    if Path(str(state.get("source_root"))).resolve() != reader.layout.root:
        raise DataQualityError("CSMAR参考数据断点属于另一个原始目录")
    if Path(str(state.get("output_root"))).resolve() != destination:
        raise DataQualityError("CSMAR参考数据断点属于另一个输出目录")
    if state.get("common_cutoff_date") != common_cutoff_date.isoformat():
        raise DataQualityError("本次共同截止日与断点不一致")
    if state.get("retrieved_at") != retrieved_at.isoformat():
        raise DataQualityError("本次取得日期与断点不一致")


def _adopt_committed_entries(
    state: dict[str, Any],
    *,
    entries: tuple[CSMARReferenceEntry, ...],
    destination: Path,
    state_path: Path,
) -> dict[str, Any]:
    completed = set(state["completed_entries"])
    for entry in entries:
        dataset_name = "balance_sheet" if entry.kind == "balance_sheet" else "index_daily"
        directory = destination / dataset_name / f"entry={entry.entry_id}"
        if entry.entry_id in completed:
            if not directory.is_dir():
                raise DataQualityError(f"断点已登记但参考数据分片缺失：{directory}")
            continue
        if not directory.is_dir():
            continue
        record_path = directory / "entry_commit.json"
        if not record_path.is_file():
            raise DataQualityError(f"未登记参考数据分片没有提交记录：{directory}")
        record = _read_json(record_path)
        if (
            record.get("source_fingerprint") != state["source_fingerprint"]
            or record.get("entry_id") != entry.entry_id
        ):
            raise DataQualityError(f"未登记参考数据分片提交记录不匹配：{directory}")
        state = record["state_after"]
        _atomic_write_json(state, state_path)
        completed = set(state["completed_entries"])
    return state


def _guard_independent_output(destination: Path) -> None:
    if (destination / "daily").exists() or (destination / "csmar.duckdb").exists():
        raise DataQualityError("参考数据必须写入独立目录；拒绝修改现有个股日线导入目录")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataQualityError(f"无法读取CSMAR参考数据断点：{path}") from exc
    if not isinstance(value, dict):
        raise DataQualityError("CSMAR参考数据断点必须是JSON对象")
    return value


def _atomic_write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (date, Path)):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _sql_literal(path: str | Path) -> str:
    return str(path).replace("'", "''")


def _as_date(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date()
