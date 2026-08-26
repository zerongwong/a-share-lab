"""CLI for the offline, resumable CSMAR import."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from ashare_lab.bootstrap import application_data_dir
from ashare_lab.services.import_csmar_local import import_csmar_local


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将用户已取得的CSMAR本地导出转换为Parquet和DuckDB（不修改原ZIP）"
    )
    parser.add_argument("source_root", type=Path, help="包含TRD_Co.xlsx和日个股回报率ZIP的目录")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=application_data_dir() / "cache" / "csmar",
        help="本地数据库输出目录",
    )
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = import_csmar_local(
        args.source_root,
        args.output_root,
        as_of=args.as_of,
        batch_size=args.batch_size,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
