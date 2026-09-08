"""Stable module name for the 21:00 continuous-portfolio evening report."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ashare_lab.cli.evening_digest import (
        EXIT_ERROR,
        EXIT_OK,
        EXIT_RETRY,
        EveningDigestOutcome,
        EveningNotificationSummary,
        build_parser,
        latest_verified_overlay_cutoff,
        resolve_next_infoway_trading_day,
        resolve_next_zero_budget_trading_day,
        run_evening_digest,
        send_evening_digest,
        send_serverchan_digest,
    )

__all__ = (
    "EXIT_ERROR",
    "EXIT_OK",
    "EXIT_RETRY",
    "EveningDigestOutcome",
    "EveningNotificationSummary",
    "build_parser",
    "latest_verified_overlay_cutoff",
    "main",
    "resolve_next_infoway_trading_day",
    "resolve_next_zero_budget_trading_day",
    "run_evening_digest",
    "send_evening_digest",
    "send_serverchan_digest",
)


def __getattr__(name: str):
    # Preserve legacy imports without loading analytics before the supervised
    # child starts. Import errors in that child become a bounded failed attempt.
    if name in __all__ and name != "main":
        from ashare_lab.cli import evening_digest

        return getattr(evening_digest, name)
    raise AttributeError(name)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(value in {"--send-now", "--help", "-h"} for value in arguments):
        from ashare_lab.cli.evening_digest import main as manual_main

        return manual_main(arguments)
    from ashare_lab.cli.evening_report_worker import supervise_evening_report

    code, event = supervise_evening_report(arguments)
    print(json.dumps(event, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
