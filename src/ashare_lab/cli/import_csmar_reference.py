"""CLI for the independent CSMAR balance-sheet and index import."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from ashare_lab.bootstrap import application_data_dir
from ashare_lab.services.import_csmar_reference import import_csmar_reference_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "导入CSMAR资产负债表当前快照与指数日线；原ZIP保持只读，并与个股日线采用同一截止日"
        )
    )
    parser.add_argument("source_root", type=Path, help="包含财务数据.zip与大盘数据目录的路径")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=application_data_dir() / "cache" / "csmar_reference",
        help="独立参考数据库输出目录",
    )
    parser.add_argument(
        "--common-cutoff",
        type=date.fromisoformat,
        required=True,
        help="与个股日线一致的截止日，格式YYYY-MM-DD",
    )
    parser.add_argument(
        "--retrieved-at",
        type=date.fromisoformat,
        default=date.today(),
        help="本次实际取得数据的日期，格式YYYY-MM-DD",
    )
    parser.add_argument("--batch-size", type=int, default=50_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = import_csmar_reference_data(
        args.source_root,
        args.output_root,
        common_cutoff_date=args.common_cutoff,
        retrieved_at=args.retrieved_at,
        batch_size=args.batch_size,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
