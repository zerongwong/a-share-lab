"""Explicit, local candidate financial/announcement document-review commands."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from ashare_lab.bootstrap import application_data_dir
from ashare_lab.services.candidate_review_workflow import (
    ReviewWorkflowError,
    create_review_template,
    download_review_document,
    list_review_status,
    register_review,
    validate_review,
)


def main(argv=None, *, _now=None, _client=None) -> int:
    parser = argparse.ArgumentParser(description="候选财务公告正文审阅：不会自动勾选或生成通过结论")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--review-dir", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="查看最新取证与正文待审阅状态")
    status.add_argument("--symbol")
    template = sub.add_parser("template", help="创建全部未确认的pending审阅草稿")
    template.add_argument("symbol")
    download = sub.add_parser("download", help="仅下载当前清单中的巨潮官方PDF，不代表已研读")
    download.add_argument("symbol")
    download.add_argument("announcement_id")
    for name in ("validate", "register"):
        command = sub.add_parser(name)
        command.add_argument("symbol")
        command.add_argument("--file", type=Path, required=True)
        if name == "register":
            command.add_argument("--confirm-reviewed", action="store_true")
            command.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args(argv)
    root = application_data_dir()
    shared = {
        "cache_dir": args.cache_dir or root / "cache" / "candidate_evidence",
        "review_dir": args.review_dir or root / "candidate_reviews",
        "known_at": _now or datetime.now(UTC),
        "symbol": args.symbol,
    }
    try:
        if args.command == "status":
            result = list_review_status(**shared)
        elif args.command == "template":
            result = create_review_template(**shared)
        elif args.command == "download":
            result = download_review_document(
                **shared, announcement_id=args.announcement_id, client=_client
            )
        elif args.command == "validate":
            result = validate_review(**shared, review_path=args.file)
        else:
            result = register_review(
                **shared,
                review_path=args.file,
                confirm_reviewed=args.confirm_reviewed,
                replace_existing=args.replace_existing,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("gate") != "unknown" else 3
    except (ReviewWorkflowError, OSError) as exc:
        reason = str(exc) if isinstance(exc, ReviewWorkflowError) else "LOCAL_FILE_OPERATION_FAILED"
        print(json.dumps({"error": reason, "registered": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
