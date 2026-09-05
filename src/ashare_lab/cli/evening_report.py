"""Stable module name for the 21:00 continuous-portfolio evening report."""

from __future__ import annotations

from ashare_lab.cli.evening_digest import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_RETRY,
    EveningDigestOutcome,
    EveningNotificationSummary,
    build_parser,
    latest_verified_overlay_cutoff,
    main,
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


if __name__ == "__main__":
    raise SystemExit(main())
