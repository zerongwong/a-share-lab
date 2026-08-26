from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from ashare_lab.adapters.csmar_local import (
    CSMARLocalReader,
    CSMARParquetMarketData,
)
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.services.import_csmar_local import import_csmar_local

NOW = datetime(2026, 8, 25, 13, 0, tzinfo=UTC)

MASTER_HEADERS = (
    "Stkcd",
    "Stknme",
    "Listdt",
    "Indnme",
    "Nindnme",
    "Nnindnme",
    "IndnmeZX",
    "Estbdt",
    "OWNERSHIPTYPE",
    "Parval",
)
DAILY_HEADERS = (
    "Stkcd",
    "Trddt",
    "Opnprc",
    "Hiprc",
    "Loprc",
    "Clsprc",
    "Dnvaltrd",
    "Dsmvosd",
    "Dsmvtll",
    "Ahshrtrd_D",
)


def _workbook_bytes(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(headers)
    sheet.append(tuple(f"中文-{value}" for value in headers))
    sheet.append(tuple("没有单位" for _ in headers))
    for row in rows:
        sheet.append(row)
    output = io.BytesIO()
    workbook.save(output)
    return _force_bad_dimension(output.getvalue())


def _force_bad_dimension(value: bytes) -> bytes:
    """Mirror CSMAR's incorrect A1:A... metadata to guard the workaround."""

    source = zipfile.ZipFile(io.BytesIO(value))
    output = io.BytesIO()
    with source, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "xl/worksheets/sheet1.xml":
                payload = payload.replace(b'ref="A1:J', b'ref="A1:A')
            target.writestr(info, payload)
    return output.getvalue()


def _daily_row(
    symbol: str,
    trade_date: str,
    close: float,
    *,
    amount: float = 1_000_000.0,
) -> tuple[object, ...]:
    return (
        symbol,
        trade_date,
        close - 0.2,
        close + 0.3,
        close - 0.4,
        close,
        amount,
        10_000.0,
        20_000.0,
        None,
    )


def _make_bundle(root: Path, *, conflict: bool = False) -> Path:
    source = root / "CSMAR-export"
    company = source / "公司文件(仅供测试使用)"
    company.mkdir(parents=True)
    master_rows = [
        (
            "000001",
            "平安银行",
            "1991-04-03",
            "金融",
            "银行",
            "银行",
            "银行",
            "1987-12-22",
            "私营企业",
            1,
        ),
        (
            "600150",
            "中国船舶",
            "1998-05-20",
            "工业",
            "制造业",
            "船舶制造",
            "船舶制造",
            "1998-03-18",
            "国营或国有控股",
            1,
        ),
        (
            "900901",
            "云赛B股",
            "1990-12-19",
            "工业",
            "制造业",
            "制造业",
            "制造业",
            "1986-01-01",
            "国营或国有控股",
            1,
        ),
    ]
    (company / "TRD_Co.xlsx").write_bytes(_workbook_bytes(MASTER_HEADERS, master_rows))

    first = [
        _daily_row("000001", "2026-08-20", 10.0),
        _daily_row("000001", "2026-08-21", 11.0),
        _daily_row("600150", "2026-08-20", 20.0),
    ]
    boundary_close = 20.5 if conflict else 20.0
    second = [
        _daily_row("600150", "2026-08-20", boundary_close),
        _daily_row("600150", "2026-08-21", 21.0),
        _daily_row("900901", "2026-08-21", 5.0),
    ]
    with zipfile.ZipFile(source / "日个股回报率文件1.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("TRD_Dalyr.xlsx", _workbook_bytes(DAILY_HEADERS, first))
    with zipfile.ZipFile(source / "日个股回报率文件2.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("TRD_Dalyr.xlsx", _workbook_bytes(DAILY_HEADERS, second))
    return source


def _tree_hash(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_import_is_read_only_resumable_and_queryable(tmp_path: Path) -> None:
    source = _make_bundle(tmp_path)
    before = _tree_hash(source)
    output = tmp_path / "local-db"

    report = import_csmar_local(
        source,
        output,
        as_of=date(2026, 8, 25),
        batch_size=2,
        clock=lambda: NOW,
    )

    assert _tree_hash(source) == before
    assert report.status == "complete"
    assert report.master_symbol_count == 2
    assert report.daily_symbol_count == 2
    assert report.daily_row_count == 4
    assert report.duplicate_rows_dropped == 1
    assert report.non_a_share_rows_dropped == 1
    assert report.full_day_volume_available is False
    assert report.first_trade_date == date(2026, 8, 20)
    assert report.last_trade_date == date(2026, 8, 21)
    assert Path(report.duckdb_path).is_file()

    adapter = CSMARParquetMarketData(output)
    assert adapter.read_security_master()["symbol"].tolist() == ["000001", "600150"]
    bars = adapter.fetch_daily(
        "600150.SH",
        date(2026, 8, 20),
        date(2026, 8, 21),
        adjust="none",
    )
    assert bars["close"].tolist() == [20.0, 21.0]
    assert pd.isna(bars.loc[0, "prev_close"])
    assert bars.loc[1, "prev_close"] == pytest.approx(20.0)
    assert bars["volume_shares"].isna().all()
    assert bars.attrs["full_day_volume_available"] is False
    cross_section = adapter.read_full_market_for_date(date(2026, 8, 21))
    assert cross_section["symbol"].tolist() == ["000001", "600150"]

    part_paths = sorted(output.glob("daily/entry=*/part-*.parquet"))
    rerun = import_csmar_local(
        source,
        output,
        as_of=date(2026, 8, 25),
        batch_size=1,
        clock=lambda: NOW,
    )
    assert rerun.daily_row_count == 4
    assert sorted(output.glob("daily/entry=*/part-*.parquet")) == part_paths


def test_reader_recovers_all_columns_from_bad_csmar_dimension(tmp_path: Path) -> None:
    source = _make_bundle(tmp_path)
    reader = CSMARLocalReader(source)
    master = reader.read_security_master(as_of=date(2026, 8, 25), retrieved_at=NOW)
    assert master.loc[master["symbol"] == "600150", "name"].item() == "中国船舶"
    first = next(reader.iter_daily_batches(reader.layout.daily_entries[0], batch_size=10))
    assert tuple(first.columns) == DAILY_HEADERS
    assert first.loc[0, "Trddt"] == "2026-08-20"
    assert first.loc[0, "Clsprc"] == pytest.approx(10.0)


def test_conflicting_duplicate_fails_without_altering_source(tmp_path: Path) -> None:
    source = _make_bundle(tmp_path, conflict=True)
    before = _tree_hash(source)
    with pytest.raises(DataQualityError, match="冲突"):
        import_csmar_local(
            source,
            tmp_path / "local-db",
            as_of=date(2026, 8, 25),
            batch_size=2,
            clock=lambda: NOW,
        )
    assert _tree_hash(source) == before


def test_adjustment_and_resume_identity_are_fail_closed(tmp_path: Path) -> None:
    source = _make_bundle(tmp_path)
    output = tmp_path / "local-db"
    import_csmar_local(
        source,
        output,
        as_of=date(2026, 8, 25),
        batch_size=10,
        clock=lambda: NOW,
    )
    adapter = CSMARParquetMarketData(output)
    with pytest.raises(DataUnavailableError, match="未复权"):
        adapter.fetch_daily("600150", date(2026, 8, 20), date(2026, 8, 21), adjust="qfq")
    with pytest.raises(DataQualityError, match="as_of"):
        import_csmar_local(
            source,
            output,
            as_of=date(2026, 8, 24),
            batch_size=10,
            clock=lambda: NOW,
        )
