"""Inspect/run local Marks shadow research or import reviewed public evidence."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

from ashare_lab.bootstrap import build_repository
from ashare_lab.services.marks_cycle_shadow import import_evidence, run_marks_cycle_shadow


def main(argv=None):
    parser = argparse.ArgumentParser(description="马克斯周期影子研究：不改生产仓位，不发送通知")
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import-evidence")
    imp.add_argument("path", type=Path)
    run = sub.add_parser("run")
    run.add_argument("--price-cutoff", type=date.fromisoformat, required=True)
    run.add_argument("--incumbent-cap", type=float, required=True)
    args = parser.parse_args(argv)
    repo = build_repository()
    if args.command == "import-evidence":
        records = import_evidence(repo, json.loads(args.path.read_text()))
        result = {"imported_count": len(records), "production_decision_input": False}
    else:
        result = run_marks_cycle_shadow(
            repo,
            price_cutoff=args.price_cutoff,
            known_at=datetime.now(UTC),
            incumbent_cap=args.incumbent_cap,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
