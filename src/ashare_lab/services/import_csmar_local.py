"""Resumable, offline CSMAR-to-Parquet/DuckDB import service."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from ashare_lab.adapters.csmar_local import (
    CSMARLocalReader,
    CSMARWorkbookEntry,
    infer_a_share_exchange,
)
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS

STATE_SCHEMA_VERSION = 1
DAILY_OUTPUT_COLUMNS = (
    "symbol",
    *CANONICAL_DAILY_COLUMNS,
    "free_float_market_cap_cny",
    "total_market_cap_cny",
    "after_hours_volume_shares",
)


@dataclass(frozen=True, slots=True)
class CSMARImportReport:
    status: str
    source_root: str
    output_root: str
    source_fingerprint: str
    as_of: date
    started_at: str
    completed_at: str
    workbook_entries: int
    completed_entries: int
    master_symbol_count: int
    daily_symbol_count: int
    symbol_coverage_ratio: float
    daily_row_count: int
    first_trade_date: date | None
    last_trade_date: date | None
    duplicate_rows_dropped: int
    non_a_share_rows_dropped: int
    rows_missing_from_master_dropped: int
    future_rows_dropped: int
    invalid_price_rows_dropped: int
    full_day_volume_available: bool
    security_master_path: str
    daily_dataset_glob: str
    duckdb_path: str
    coverage_json_path: str
    symbol_coverage_path: str
    date_coverage_path: str
    warnings: tuple[str, ...]


def import_csmar_local(
    source_root: str | Path,
    output_root: str | Path,
    *,
    as_of: date,
    batch_size: int = 100_000,
    clock: Any = None,
) -> CSMARImportReport:
    """Convert a CSMAR local export without modifying or replacing its ZIPs.

    The durable checkpoint is committed after each nested XLSX member.  An
    interrupted member is safely re-read; already committed members are not.
    Conflicting duplicate ``(symbol, trade_date)`` rows fail loudly instead of
    silently choosing a price.  Workbooks may be split by CSMAR in an order
    unrelated to symbol/date; global de-duplication and ``prev_close`` are
    therefore computed by DuckDB only after every independent part is present.
    """

    if batch_size <= 0:
        raise ValueError("batch_size必须大于0")
    now = clock or (lambda: datetime.now(UTC))
    reader = CSMARLocalReader(source_root)
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "import_state.json"

    if state_path.is_file():
        state = _read_json(state_path)
        _validate_resume_state(
            state,
            fingerprint=reader.layout.fingerprint,
            source_root=reader.layout.root,
            as_of=as_of,
        )
    else:
        started = _aware_utc(now())
        master = reader.read_security_master(as_of=as_of, retrieved_at=started)
        _atomic_write_parquet(master, destination / "security_master.parquet")
        state = _new_state(
            reader=reader,
            as_of=as_of,
            started_at=started,
            master_symbol_count=int(master["symbol"].nunique()),
        )
        _atomic_write_json(state, state_path)

    master_path = destination / "security_master.parquet"
    if not master_path.is_file():
        raise DataQualityError("导入状态存在，但security_master.parquet缺失")
    master = pd.read_parquet(master_path)
    master_symbols = frozenset(master["symbol"].astype(str))
    state = _adopt_committed_entries(
        state,
        entries=reader.layout.daily_entries,
        destination=destination,
        state_path=state_path,
    )

    for entry in reader.layout.daily_entries:
        if entry.entry_id in state["completed_entries"]:
            continue
        state = _import_entry(
            reader,
            entry,
            destination=destination,
            state=state,
            state_path=state_path,
            master_symbols=master_symbols,
            as_of=as_of,
            batch_size=batch_size,
        )

    completed_at = _aware_utc(now())
    report = _write_coverage_and_catalog(
        reader=reader,
        destination=destination,
        state=state,
        master=master,
        completed_at=completed_at,
    )
    state["status"] = "complete"
    state["completed_at"] = report.completed_at
    state["report"] = _jsonable(asdict(report))
    _atomic_write_json(state, state_path)
    return report


def _import_entry(
    reader: CSMARLocalReader,
    entry: CSMARWorkbookEntry,
    *,
    destination: Path,
    state: dict[str, Any],
    state_path: Path,
    master_symbols: frozenset[str],
    as_of: date,
    batch_size: int,
) -> dict[str, Any]:
    working_state = json.loads(json.dumps(state))
    daily_root = destination / "daily"
    daily_root.mkdir(parents=True, exist_ok=True)
    final_directory = daily_root / f"entry={entry.entry_id}"
    if final_directory.exists():
        raise DataQualityError(
            f"发现未登记的已存在分片目录：{final_directory}；请保留现场并检查导入状态"
        )
    staging = daily_root / f".entry={entry.entry_id}.{uuid4().hex}.tmp"
    staging.mkdir(parents=True, exist_ok=False)
    part_number = 0
    try:
        for raw in reader.iter_daily_batches(entry, batch_size=batch_size):
            normalized = _normalize_daily_batch(
                raw,
                state=working_state,
                master_symbols=master_symbols,
                as_of=as_of,
            )
            if normalized.empty:
                continue
            _atomic_write_parquet(
                normalized,
                staging / f"part-{part_number:05d}.parquet",
            )
            part_number += 1

        working_state["completed_entries"].append(entry.entry_id)
        working_state["entry_reports"][entry.entry_id] = {
            "display_name": entry.display_name,
            "part_count": part_number,
            "rows_imported_total": working_state["stats"]["rows_imported"],
        }
        commit_record = {
            "schema_version": STATE_SCHEMA_VERSION,
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


def _normalize_daily_batch(
    raw: pd.DataFrame,
    *,
    state: dict[str, Any],
    master_symbols: frozenset[str],
    as_of: date,
) -> pd.DataFrame:
    stats = state["stats"]
    stats["rows_seen"] += int(len(raw))
    frame = raw.copy()
    frame["symbol"] = frame["Stkcd"].map(_normalize_code)
    valid_code = frame["symbol"].str.fullmatch(r"\d{6}", na=False)
    if not bool(valid_code.all()):
        examples = frame.loc[~valid_code, "Stkcd"].head(5).tolist()
        raise DataQualityError(f"CSMAR日线存在无效证券代码：{examples}")

    is_a_share = frame["symbol"].map(infer_a_share_exchange).notna()
    stats["non_a_share_rows_dropped"] += int((~is_a_share).sum())
    frame = frame.loc[is_a_share].copy()
    in_master = frame["symbol"].isin(master_symbols)
    stats["rows_missing_from_master_dropped"] += int((~in_master).sum())
    frame = frame.loc[in_master].copy()
    if frame.empty:
        return _empty_daily_frame()

    frame["trade_date"] = pd.to_datetime(frame["Trddt"], errors="coerce").dt.normalize()
    if bool(frame["trade_date"].isna().any()):
        examples = frame.loc[frame["trade_date"].isna(), ["symbol", "Trddt"]].head(5)
        raise DataQualityError(f"CSMAR日线存在无效交易日期：{examples.to_dict('records')}")
    future = frame["trade_date"] > pd.Timestamp(as_of)
    stats["future_rows_dropped"] += int(future.sum())
    frame = frame.loc[~future].copy()
    if frame.empty:
        return _empty_daily_frame()

    numeric_map = {
        "Opnprc": "open",
        "Hiprc": "high",
        "Loprc": "low",
        "Clsprc": "close",
        "Dnvaltrd": "amount_cny",
        "Dsmvosd": "free_float_market_cap_thousand_cny",
        "Dsmvtll": "total_market_cap_thousand_cny",
        "Ahshrtrd_D": "after_hours_volume_shares",
    }
    for raw_name, canonical_name in numeric_map.items():
        frame[canonical_name] = pd.to_numeric(frame[raw_name], errors="coerce")

    invalid_price = frame[["open", "high", "low", "close"]].isna().any(axis=1) | (
        frame[["open", "high", "low", "close"]] <= 0
    ).any(axis=1)
    stats["invalid_price_rows_dropped"] += int(invalid_price.sum())
    frame = frame.loc[~invalid_price].copy()
    if frame.empty:
        return _empty_daily_frame()
    if bool((frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any()):
        raise DataQualityError("CSMAR日线存在最高价低于开盘/最低/收盘价")
    if bool((frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any()):
        raise DataQualityError("CSMAR日线存在最低价高于开盘/最高/收盘价")
    if bool((frame["amount_cny"].dropna() < 0).any()):
        raise DataQualityError("CSMAR日线存在负成交额")
    for column in (
        "free_float_market_cap_thousand_cny",
        "total_market_cap_thousand_cny",
        "after_hours_volume_shares",
    ):
        if bool((frame[column].dropna() < 0).any()):
            raise DataQualityError(f"CSMAR日线存在负值字段：{column}")

    frame = frame.reset_index(drop=True)
    keys = pd.MultiIndex.from_frame(frame[["symbol", "trade_date"]])
    if not keys.is_monotonic_increasing:
        raise DataQualityError("CSMAR单批日线未按证券代码、交易日期升序排列")

    duplicate_mask = frame.duplicated(["symbol", "trade_date"], keep=False)
    if bool(duplicate_mask.any()):
        comparison = [
            "open",
            "high",
            "low",
            "close",
            "amount_cny",
            "free_float_market_cap_thousand_cny",
            "total_market_cap_thousand_cny",
            "after_hours_volume_shares",
        ]
        conflicts = (
            frame.loc[duplicate_mask]
            .groupby(["symbol", "trade_date"], dropna=False)[comparison]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if bool(conflicts.any()):
            raise DataQualityError("CSMAR日线存在同代码同日期但价格或金额不同的冲突记录")
        before = len(frame)
        frame = frame.drop_duplicates(["symbol", "trade_date"], keep="first")
        stats["duplicate_rows_dropped"] += before - len(frame)

    frame["prev_close"] = math.nan
    frame["volume_shares"] = math.nan
    frame["turnover_pct"] = math.nan
    frame["source"] = "csmar:local:unadjusted"
    frame["retrieved_at"] = state["retrieved_at"]
    frame["free_float_market_cap_cny"] = frame["free_float_market_cap_thousand_cny"] * 1000.0
    frame["total_market_cap_cny"] = frame["total_market_cap_thousand_cny"] * 1000.0

    output = frame.loc[:, list(DAILY_OUTPUT_COLUMNS)].copy()
    output["symbol"] = output["symbol"].astype(str)
    output["trade_date"] = pd.to_datetime(output["trade_date"]).dt.normalize()
    output = output.sort_values(["symbol", "trade_date"], kind="stable").reset_index(drop=True)

    batch_first = output["trade_date"].min().date().isoformat()
    batch_last = output["trade_date"].max().date().isoformat()
    stats["first_trade_date"] = (
        batch_first
        if stats["first_trade_date"] is None
        else min(stats["first_trade_date"], batch_first)
    )
    stats["last_trade_date"] = (
        batch_last
        if stats["last_trade_date"] is None
        else max(stats["last_trade_date"], batch_last)
    )
    stats["rows_imported"] += int(len(output))
    return output


def _update_coverage(state: dict[str, Any], frame: pd.DataFrame) -> None:
    symbol_coverage = state["symbol_coverage"]
    grouped_symbols = frame.groupby("symbol", sort=False)["trade_date"].agg(["min", "max", "size"])
    for symbol, row in grouped_symbols.iterrows():
        first = row["min"].date().isoformat()
        last = row["max"].date().isoformat()
        current = symbol_coverage.get(str(symbol))
        if current is None:
            symbol_coverage[str(symbol)] = {"first": first, "last": last, "rows": int(row["size"])}
        else:
            current["first"] = min(current["first"], first)
            current["last"] = max(current["last"], last)
            current["rows"] += int(row["size"])

    date_coverage = state["date_coverage"]
    grouped_dates = frame.groupby("trade_date", sort=False).agg(
        rows=("symbol", "size"), symbols=("symbol", "nunique")
    )
    for trade_date, row in grouped_dates.iterrows():
        key = trade_date.date().isoformat()
        current = date_coverage.setdefault(key, {"rows": 0, "symbols": 0})
        current["rows"] += int(row["rows"])
        current["symbols"] += int(row["symbols"])

    batch_first = frame["trade_date"].min().date().isoformat()
    batch_last = frame["trade_date"].max().date().isoformat()
    stats = state["stats"]
    stats["first_trade_date"] = (
        batch_first
        if stats["first_trade_date"] is None
        else min(stats["first_trade_date"], batch_first)
    )
    stats["last_trade_date"] = (
        batch_last
        if stats["last_trade_date"] is None
        else max(stats["last_trade_date"], batch_last)
    )


def _adopt_committed_entries(
    state: dict[str, Any],
    *,
    entries: tuple[CSMARWorkbookEntry, ...],
    destination: Path,
    state_path: Path,
) -> dict[str, Any]:
    completed = set(state["completed_entries"])
    for entry in entries:
        directory = destination / "daily" / f"entry={entry.entry_id}"
        if entry.entry_id in completed:
            if not directory.is_dir():
                raise DataQualityError(f"断点清单已登记但分片目录缺失：{directory}")
            continue
        if not directory.is_dir():
            break
        record_path = directory / "entry_commit.json"
        if not record_path.is_file():
            raise DataQualityError(f"未登记分片没有提交记录：{directory}")
        record = _read_json(record_path)
        if (
            record.get("source_fingerprint") != state["source_fingerprint"]
            or record.get("entry_id") != entry.entry_id
        ):
            raise DataQualityError(f"未登记分片提交记录不匹配：{directory}")
        state = record["state_after"]
        _atomic_write_json(state, state_path)
        completed = set(state["completed_entries"])
    return state


def _write_coverage_and_catalog(
    *,
    reader: CSMARLocalReader,
    destination: Path,
    state: dict[str, Any],
    master: pd.DataFrame,
    completed_at: datetime,
) -> CSMARImportReport:
    reports = destination / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    symbol_path = reports / "symbol_coverage.parquet"
    date_path = reports / "date_coverage.parquet"
    daily_glob = str(destination / "daily" / "entry=*" / "part-*.parquet")
    duckdb_path = destination / "csmar.duckdb"
    symbol_detail, date_detail, raw_row_count, daily_row_count = _build_duckdb_catalog(
        database_path=duckdb_path,
        master_path=destination / "security_master.parquet",
        daily_glob=daily_glob,
    )
    _atomic_write_parquet(symbol_detail, symbol_path)
    _atomic_write_parquet(date_detail, date_path)
    _attach_coverage_views(
        database_path=duckdb_path,
        symbol_coverage_path=symbol_path,
        date_coverage_path=date_path,
    )

    stats = state["stats"]
    master_count = int(master["symbol"].nunique())
    daily_count = int(len(symbol_detail))
    warnings = (
        "CSMAR本批日线没有全日成交量；volume_shares保持为空，Ahshrtrd_D仅保存为盘后成交量。",
        "价格为CSMAR导出的未复权原始价格；不得与前复权或后复权序列静默拼接。",
        "公司文件没有退市日期和历史ST/停牌状态；相关字段只能视为当前名称推断或未知。",
        "本地导出受原许可限制，只用于已授权研究，不上传或再分发原始数据。",
    )
    report_path = reports / "coverage.json"
    report = CSMARImportReport(
        status="complete",
        source_root=str(reader.layout.root),
        output_root=str(destination),
        source_fingerprint=reader.layout.fingerprint,
        as_of=date.fromisoformat(state["as_of"]),
        started_at=state["started_at"],
        completed_at=completed_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        workbook_entries=len(reader.layout.daily_entries),
        completed_entries=len(state["completed_entries"]),
        master_symbol_count=master_count,
        daily_symbol_count=daily_count,
        symbol_coverage_ratio=(daily_count / master_count if master_count else 0.0),
        daily_row_count=daily_row_count,
        first_trade_date=(
            date.fromisoformat(stats["first_trade_date"]) if stats["first_trade_date"] else None
        ),
        last_trade_date=(
            date.fromisoformat(stats["last_trade_date"]) if stats["last_trade_date"] else None
        ),
        duplicate_rows_dropped=(
            int(stats["duplicate_rows_dropped"]) + raw_row_count - daily_row_count
        ),
        non_a_share_rows_dropped=int(stats["non_a_share_rows_dropped"]),
        rows_missing_from_master_dropped=int(stats["rows_missing_from_master_dropped"]),
        future_rows_dropped=int(stats["future_rows_dropped"]),
        invalid_price_rows_dropped=int(stats["invalid_price_rows_dropped"]),
        full_day_volume_available=False,
        security_master_path=str(destination / "security_master.parquet"),
        daily_dataset_glob=daily_glob,
        duckdb_path=str(duckdb_path),
        coverage_json_path=str(report_path),
        symbol_coverage_path=str(symbol_path),
        date_coverage_path=str(date_path),
        warnings=warnings,
    )
    _atomic_write_json(_jsonable(asdict(report)), report_path)
    return report


def _build_duckdb_catalog(
    *,
    database_path: Path,
    master_path: Path,
    daily_glob: str,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    try:
        import duckdb
    except ImportError as exc:
        raise DataUnavailableError("缺少duckdb依赖，无法建立本地查询目录") from exc
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute(
            f"CREATE OR REPLACE VIEW security_master AS "
            f"SELECT * FROM read_parquet('{_sql_literal(master_path)}')"
        )
        connection.execute(
            f"CREATE OR REPLACE VIEW daily_raw AS "
            f"SELECT * FROM read_parquet('{_sql_literal(daily_glob)}', union_by_name=true)"
        )
        conflicts = connection.execute(
            "SELECT symbol, trade_date FROM daily_raw "
            "GROUP BY symbol, trade_date HAVING COUNT(DISTINCT struct_pack("
            "open := open, high := high, low := low, close := close, "
            "amount_cny := amount_cny, free_float_market_cap_cny := free_float_market_cap_cny, "
            "total_market_cap_cny := total_market_cap_cny, "
            "after_hours_volume_shares := after_hours_volume_shares)) > 1 LIMIT 5"
        ).fetchall()
        if conflicts:
            raise DataQualityError(f"CSMAR跨分片存在冲突重复日线：{conflicts}")
        connection.execute(
            "CREATE OR REPLACE VIEW daily_deduplicated AS "
            "SELECT * EXCLUDE (_dedupe_rank) FROM ("
            "SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol, trade_date "
            "ORDER BY source, retrieved_at) AS _dedupe_rank FROM daily_raw"
            ") WHERE _dedupe_rank = 1"
        )
        connection.execute(
            "CREATE OR REPLACE VIEW daily_bars AS SELECT symbol, trade_date, open, high, low, "
            "close, LAG(close) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_close, "
            "volume_shares, amount_cny, turnover_pct, source, retrieved_at, "
            "free_float_market_cap_cny, total_market_cap_cny, after_hours_volume_shares "
            "FROM daily_deduplicated"
        )
        connection.execute(
            "CREATE OR REPLACE TABLE dataset_metadata AS "
            "SELECT 'csmar' AS source, 'none' AS adjustment, "
            "false AS full_day_volume_available"
        )
        symbol_detail = connection.execute(
            "SELECT symbol, MIN(trade_date) AS first_trade_date, "
            "MAX(trade_date) AS last_trade_date, COUNT(*) AS row_count "
            "FROM daily_bars GROUP BY symbol ORDER BY symbol"
        ).fetchdf()
        date_detail = connection.execute(
            "SELECT trade_date, COUNT(*) AS row_count, COUNT(DISTINCT symbol) AS symbol_count "
            "FROM daily_bars GROUP BY trade_date ORDER BY trade_date"
        ).fetchdf()
        raw_row_count = int(connection.execute("SELECT COUNT(*) FROM daily_raw").fetchone()[0])
        daily_row_count = int(connection.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0])
        return symbol_detail, date_detail, raw_row_count, daily_row_count
    finally:
        connection.close()


def _attach_coverage_views(
    *,
    database_path: Path,
    symbol_coverage_path: Path,
    date_coverage_path: Path,
) -> None:
    import duckdb

    connection = duckdb.connect(str(database_path))
    try:
        connection.execute(
            f"CREATE OR REPLACE VIEW symbol_coverage AS "
            f"SELECT * FROM read_parquet('{_sql_literal(symbol_coverage_path)}')"
        )
        connection.execute(
            f"CREATE OR REPLACE VIEW date_coverage AS "
            f"SELECT * FROM read_parquet('{_sql_literal(date_coverage_path)}')"
        )
    finally:
        connection.close()


def _new_state(
    *,
    reader: CSMARLocalReader,
    as_of: date,
    started_at: datetime,
    master_symbol_count: int,
) -> dict[str, Any]:
    timestamp = started_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": "in_progress",
        "source_root": str(reader.layout.root),
        "source_fingerprint": reader.layout.fingerprint,
        "as_of": as_of.isoformat(),
        "started_at": timestamp,
        "retrieved_at": timestamp,
        "master_symbol_count": master_symbol_count,
        "completed_entries": [],
        "entry_reports": {},
        "last_key": None,
        "last_symbol": None,
        "last_close": None,
        "last_signature": None,
        "symbol_coverage": {},
        "date_coverage": {},
        "stats": {
            "rows_seen": 0,
            "rows_imported": 0,
            "duplicate_rows_dropped": 0,
            "non_a_share_rows_dropped": 0,
            "rows_missing_from_master_dropped": 0,
            "future_rows_dropped": 0,
            "invalid_price_rows_dropped": 0,
            "first_trade_date": None,
            "last_trade_date": None,
        },
    }


def _validate_resume_state(
    state: dict[str, Any],
    *,
    fingerprint: str,
    source_root: Path,
    as_of: date,
) -> None:
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise DataQualityError("CSMAR断点清单版本不受支持")
    if state.get("source_fingerprint") != fingerprint:
        raise DataQualityError("CSMAR原始文件与断点清单不一致；拒绝混入另一批数据")
    if Path(str(state.get("source_root"))).resolve() != source_root:
        raise DataQualityError("CSMAR断点清单属于另一个原始目录")
    if state.get("as_of") != as_of.isoformat():
        raise DataQualityError("本次as_of与已有断点不一致；请使用相同截止日或新输出目录")


def _signature_for_row(row: pd.Series) -> list[float | None]:
    columns = (
        "open",
        "high",
        "low",
        "close",
        "amount_cny",
        "free_float_market_cap_thousand_cny",
        "total_market_cap_thousand_cny",
        "after_hours_volume_shares",
    )
    return [_finite_or_none(row[column]) for column in columns]


def _finite_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _normalize_code(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def _empty_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(DAILY_OUTPUT_COLUMNS))


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock必须返回包含时区的datetime")
    return value.astimezone(UTC)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataQualityError(f"无法读取CSMAR断点清单：{path}") from exc
    if not isinstance(value, dict):
        raise DataQualityError("CSMAR断点清单必须是JSON对象")
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
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _sql_literal(path: str | Path) -> str:
    return str(path).replace("'", "''")
