"""Local consent and refresh CLI for official company-action evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.bootstrap import build_repository
from ashare_lab.services.company_action_evidence import (
    AUTHORIZATION_SCOPE,
    AUTHORIZED_FIELDS,
    authorize_company_actions,
    company_action_config_path,
    is_company_action_authorized,
    refresh_and_load_company_action_clearances,
    revoke_company_actions,
)
from ashare_lab.services.holding_ledger import get_active_holding_portfolio

EXIT_OK = 0
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "巨潮公司行动核验：默认关闭，只可发送当前持仓的六位股票代码；"
            "不会发送名称、成本、股数、金额或权重，也不会连接券商或下单。"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    authorize = commands.add_parser("authorize", help="授权仅发送当前持仓股票代码")
    authorize.add_argument("--yes", action="store_true", help="明确确认这一最小披露授权")

    revoke = commands.add_parser("revoke", help="停止后续巨潮联网核验")
    revoke.add_argument("--yes", action="store_true", help="明确确认撤销授权")

    commands.add_parser("status", help="查看本机授权状态（不显示持仓）")

    refresh = commands.add_parser("refresh", help="刷新并缓存当前持仓公司行动证据")
    refresh.add_argument("--as-of", type=date.fromisoformat, required=True)
    refresh.add_argument("--phase", choices=("intraday", "eod"), required=True)
    refresh.add_argument("--reviewed-at", type=_aware_datetime, default=None)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    _repository: SQLiteRepository | None = None,
    _config_path: Path | None = None,
    _fetcher: Any | None = None,
    _now: datetime | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    path = company_action_config_path() if _config_path is None else _config_path
    try:
        if args.command == "authorize":
            authorize_company_actions(confirmed=bool(args.yes), config_path=path)
            payload = _status_payload(path)
        elif args.command == "revoke":
            revoke_company_actions(confirmed=bool(args.yes), config_path=path)
            payload = _status_payload(path)
        elif args.command == "status":
            payload = _status_payload(path)
        else:
            repository = _repository or build_repository()
            reviewed_at = args.reviewed_at or _now or datetime.now(UTC)
            clearances = refresh_and_load_company_action_clearances(
                repository,
                as_of=args.as_of,
                reviewed_at=reviewed_at,
                phase=args.phase,
                fetcher=_fetcher,
                config_path=path,
                allow_noncanonical_repository=_repository is not None,
            )
            portfolio = get_active_holding_portfolio(repository)
            holding_count = (
                0 if portfolio is None or portfolio.status != "active" else len(portfolio.positions)
            )
            if holding_count == 0:
                refresh_status = "no_active_holdings"
            elif len(clearances) == holding_count:
                refresh_status = "decision_grade_evidence_ready"
            elif clearances:
                refresh_status = "partial_decision_grade_evidence"
            else:
                refresh_status = "no_decision_grade_evidence"
            payload = {
                **_status_payload(path),
                "refresh_status": refresh_status,
                "known_clearance_count": len(clearances),
                "as_of": args.as_of.isoformat(),
                "phase": args.phase,
            }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return EXIT_OK
    except Exception as exc:  # noqa: BLE001 - do not leak local configuration details
        print(f"公司行动设置未更改：{type(exc).__name__}", file=sys.stderr)
        return EXIT_ERROR


def _status_payload(path: Path) -> dict[str, object]:
    enabled = is_company_action_authorized(config_path=path)
    return {
        "enabled": enabled,
        "scope": AUTHORIZATION_SCOPE if enabled else None,
        "authorized_fields": list(AUTHORIZED_FIELDS) if enabled else [],
        "orders_enabled": False,
        "privacy": (
            "the only holding-derived application field sent is the six-digit symbol; "
            "public query dates and normal HTTPS connection metadata are also visible"
        ),
    }


def _aware_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("reviewed-at必须使用ISO时间") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("reviewed-at必须包含时区")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
