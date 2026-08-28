"""CLI entrypoint for a Keychain-authenticated completed-session update."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ashare_lab.bootstrap import application_data_dir
from ashare_lab.domain.errors import AShareLabError
from ashare_lab.services.daily_update_lock import daily_update_lock
from ashare_lab.services.run_daily_update import run_daily_update


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "从macOS钥匙串读取Infoway密钥，将缺失的已完成沪深收盘数据写入独立overlay；"
            "不会修改CSMAR，也不会读取交易账户。"
        )
    )
    parser.add_argument(
        "--csmar-root",
        type=Path,
        default=application_data_dir() / "cache" / "csmar",
        help="只读CSMAR DuckDB目录",
    )
    parser.add_argument(
        "--overlay-root",
        type=Path,
        default=application_data_dir() / "cache" / "market_overlay",
        help="Infoway已验证收盘增量目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with daily_update_lock() as acquired:
            if not acquired:
                print("自动数据更新未执行：另一个收盘数据更新正在运行。", file=sys.stderr)
                return 75
            report = run_daily_update(
                csmar_root=args.csmar_root,
                overlay_root=args.overlay_root,
            )
    except (AShareLabError, ValueError) as exc:
        print(f"自动数据更新未完成：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2, default=str))
    return 0 if report.current_through_latest_complete_session else 1


if __name__ == "__main__":
    raise SystemExit(main())
