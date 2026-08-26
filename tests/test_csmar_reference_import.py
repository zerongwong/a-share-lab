from __future__ import annotations

import hashlib
import io
import re
import zipfile
from datetime import date
from pathlib import Path

import pytest
from openpyxl import Workbook

from ashare_lab.adapters.csmar_reference import (
    BALANCE_SHEET_FIELDS,
    INDEX_DAILY_FIELDS,
    CSMARReferenceData,
)
from ashare_lab.domain.errors import DataQualityError
from ashare_lab.services.import_csmar_reference import import_csmar_reference_data

RETRIEVED_AT = date(2026, 8, 25)


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
    source = zipfile.ZipFile(io.BytesIO(value))
    output = io.BytesIO()
    with source, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "xl/worksheets/sheet1.xml":
                payload = re.sub(rb'ref="A1:[A-Z]+(\d+)"', rb'ref="A1:A\1"', payload)
            target.writestr(info, payload)
    return output.getvalue()


def _balance_row(
    symbol: str,
    period: str,
    report_type: str,
    total_assets: float,
) -> tuple[object, ...]:
    return (
        symbol,
        f"股票{symbol}",
        period,
        report_type,
        0,
        None,
        100.0,
        50.0,
        30.0,
        500.0,
        total_assets,
        250.0,
        400.0,
        550.0,
        600.0,
    )


def _index_row(
    index_code: str,
    trade_date: str,
    close: float,
) -> tuple[object, ...]:
    return (
        index_code,
        trade_date,
        close - 5,
        close + 10,
        close - 10,
        close,
        1_000_000,
        2_000_000,
        0.01,
    )


def _make_reference_bundle(root: Path, *, missing_balance_field: bool = False) -> Path:
    source = root / "CSMAR-reference"
    index_root = source / "大盘数据"
    index_root.mkdir(parents=True)

    headers = BALANCE_SHEET_FIELDS
    if missing_balance_field:
        headers = tuple(field for field in headers if field != "A003000000")
    rows = [
        _balance_row("600150", "2026-06-30", "A", 1_000.0),
        _balance_row("600150", "2026-06-30", "B", 900.0),
        _balance_row("000001", "2026-01-01", "A", 800.0),
        _balance_row("000001", "2026-03-31", "A", 850.0),
        _balance_row("000001", "2026-09-30", "A", 950.0),
        _balance_row("900901", "2026-06-30", "A", 700.0),
    ]
    if missing_balance_field:
        drop_at = BALANCE_SHEET_FIELDS.index("A003000000")
        rows = [tuple(value for index, value in enumerate(row) if index != drop_at) for row in rows]
    with zipfile.ZipFile(source / "财务数据.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("FS_Combas.xlsx", _workbook_bytes(headers, rows))
        bundle.writestr("FS_Combas[DES][xlsx].txt", "synthetic fixture")

    with zipfile.ZipFile(index_root / "大盘1.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "IDX_Idxtrd.xlsx",
            _workbook_bytes(
                INDEX_DAILY_FIELDS,
                [
                    ("000001", "2026-08-21", 3_700.0, 3_900.0, 3_600.0, 3_800.0, 1, -1, 0.0),
                    ("000001", "2026-08-22", None, 3_900.0, 3_700.0, 3_800.0, 1, 1, 0.0),
                    ("000001", "2026-08-23", 3_800.0, 3_700.0, 3_750.0, 3_800.0, 1, 1, 0.0),
                    _index_row("000001", "2026-08-24", 3_800.0),
                    _index_row("000001", "2026-08-26", 3_900.0),
                ],
            ),
        )
    with zipfile.ZipFile(index_root / "大盘2.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "IDX_Idxtrd1.xlsx",
            _workbook_bytes(
                INDEX_DAILY_FIELDS,
                [_index_row("399001", "2026-08-25", 12_000.0)],
            ),
        )
    return source


def _tree_hash(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_reference_import_is_read_only_resumable_and_pit_safe(tmp_path: Path) -> None:
    source = _make_reference_bundle(tmp_path)
    before = _tree_hash(source)
    output = tmp_path / "reference-db"

    report = import_csmar_reference_data(
        source,
        output,
        common_cutoff_date=RETRIEVED_AT,
        retrieved_at=RETRIEVED_AT,
        batch_size=2,
    )

    assert _tree_hash(source) == before
    assert report.status == "complete"
    assert report.completed_entries == 3
    assert report.balance_sheet_rows == 2
    assert report.balance_sheet_symbols == 2
    assert report.index_daily_rows == 2
    assert report.index_count == 2
    assert report.index_invalid_price_rows_quarantined == 1
    assert report.index_invalid_geometry_rows_quarantined == 1
    assert report.index_invalid_activity_rows_quarantined == 1
    assert report.first_index_date == date(2026, 8, 24)
    assert report.last_index_date == RETRIEVED_AT
    assert report.balance_sheet_data_role == "current_snapshot"
    assert report.balance_sheet_historical_backtest_eligible is False
    assert report.index_historical_backtest_eligible is True
    assert not (output / "daily").exists()
    assert not (output / "csmar.duckdb").exists()

    adapter = CSMARReferenceData(output)
    balance = adapter.read_balance_sheet_snapshot()
    assert balance["symbol"].tolist() == ["000001", "600150"]
    assert balance["report_type"].eq("A").all()
    assert balance["report_period"].dt.strftime("%m-%d").ne("01-01").all()
    assert balance["data_role"].eq("current_snapshot").all()
    assert not bool(balance["historical_backtest_eligible"].any())
    assert balance["retrieved_at"].eq("2026-08-25").all()
    assert balance["ordinary_announcement_date"].isna().all()
    assert "fundamental_score" not in balance.columns

    index = adapter.read_index_daily("000001", date(2026, 8, 1), RETRIEVED_AT)
    assert index["trade_date"].dt.date.tolist() == [date(2026, 8, 24)]
    assert index["knowledge_date"].equals(index["trade_date"])
    assert bool(index["historical_backtest_eligible"].all())
    assert index["common_cutoff_date"].dt.date.eq(RETRIEVED_AT).all()

    part_paths = sorted(output.glob("**/part-*.parquet"))
    rerun = import_csmar_reference_data(
        source,
        output,
        common_cutoff_date=RETRIEVED_AT,
        retrieved_at=RETRIEVED_AT,
        batch_size=1,
    )
    assert rerun.balance_sheet_rows == 2
    assert sorted(output.glob("**/part-*.parquet")) == part_paths
    assert _tree_hash(source) == before


def test_reference_import_validates_workbook_fields(tmp_path: Path) -> None:
    source = _make_reference_bundle(tmp_path, missing_balance_field=True)
    before = _tree_hash(source)

    with pytest.raises(DataQualityError, match="A003000000"):
        import_csmar_reference_data(
            source,
            tmp_path / "reference-db",
            common_cutoff_date=RETRIEVED_AT,
            retrieved_at=RETRIEVED_AT,
            batch_size=10,
        )

    assert _tree_hash(source) == before


def test_reference_resume_rejects_changed_source_hash(tmp_path: Path) -> None:
    source = _make_reference_bundle(tmp_path)
    output = tmp_path / "reference-db"
    import_csmar_reference_data(
        source,
        output,
        common_cutoff_date=RETRIEVED_AT,
        retrieved_at=RETRIEVED_AT,
        batch_size=10,
    )

    with zipfile.ZipFile(source / "财务数据.zip", "a", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("audit-marker.txt", "the archive changed")

    with pytest.raises(DataQualityError, match="哈希"):
        import_csmar_reference_data(
            source,
            output,
            common_cutoff_date=RETRIEVED_AT,
            retrieved_at=RETRIEVED_AT,
            batch_size=10,
        )


def test_reference_import_refuses_stock_daily_output_directory(tmp_path: Path) -> None:
    source = _make_reference_bundle(tmp_path)
    output = tmp_path / "existing-stock-daily"
    (output / "daily").mkdir(parents=True)
    marker = output / "daily" / "keep.parquet"
    marker.write_bytes(b"do-not-touch")

    with pytest.raises(DataQualityError, match="独立目录"):
        import_csmar_reference_data(
            source,
            output,
            common_cutoff_date=RETRIEVED_AT,
            retrieved_at=RETRIEVED_AT,
        )

    assert marker.read_bytes() == b"do-not-touch"
