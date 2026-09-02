"""Read-only CLI for the completed-calendar-month model review."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date, datetime
from typing import Any

from ashare_lab.bootstrap import build_repository
from ashare_lab.services.build_monthly_model_review import (
    build_monthly_model_review,
    render_monthly_model_review_message,
)

EXIT_OK = 0
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "只读汇总一个完整自然月内到期的推荐组合；三个观察总体和六期限分别统计，"
            "不发送通知、不改旧档案、不自动调整模型。"
        )
    )
    parser.add_argument(
        "--month",
        type=_month,
        default=None,
        help="复盘月份（YYYY-MM）；默认复盘当前日期之前的完整自然月",
    )
    parser.add_argument(
        "--as-of",
        type=_iso_date,
        default=None,
        help="复盘生成日（YYYY-MM-DD）；默认今天",
    )
    parser.add_argument(
        "--format",
        choices=("json", "message"),
        default="json",
        help="输出完整机器可读JSON，或输出不发送的简洁消息预览",
    )
    parser.add_argument(
        "--report-limit",
        type=int,
        default=1000,
        help="最多扫描的最近不可变报告数量（1–1000）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        as_of = args.as_of or date.today()
        review_month = args.month or _previous_month(as_of)
        review = build_monthly_model_review(
            build_repository(),
            review_month=review_month,
            as_of=as_of,
            report_limit=args.report_limit,
        )
        if args.format == "message":
            message = render_monthly_model_review_message(review)
            print(message.title)
            print(message.body)
        else:
            print(
                json.dumps(
                    asdict(review),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_json_default,
                )
            )
        return EXIT_OK
    except Exception as exc:  # noqa: BLE001 - final CLI boundary fails closed
        print(f"月度模型复盘未完成：{type(exc).__name__}", file=sys.stderr)
        return EXIT_ERROR


def _previous_month(as_of: date) -> date:
    if as_of.month == 1:
        return date(as_of.year - 1, 12, 1)
    return date(as_of.year, as_of.month - 1, 1)


def _month(value: str) -> date:
    try:
        parsed = datetime.strptime(value, "%Y-%m").date()
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("月份必须使用YYYY-MM") from exc
    return parsed.replace(day=1)


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("日期必须使用YYYY-MM-DD") from exc


def _json_default(value: object) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    raise SystemExit(main())
