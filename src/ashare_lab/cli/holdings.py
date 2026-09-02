"""Local-only CLI for explicit current-holding declarations.

The command itself never sends holdings to a notification channel, network
service, or broker.  ``replace`` reads a local JSON file so symbols, costs and
weights do not have to be placed in shell history.  Optional disclosure flags
are explicit, per-provider consent for later summaries; both default off.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.bootstrap import build_repository
from ashare_lab.services.holding_ledger import (
    HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY,
    HoldingPositionInput,
    clear_active_holdings,
    get_active_holding_portfolio,
    holding_summary_delivery_channels,
    replace_active_holdings,
)

CN = ZoneInfo("Asia/Shanghai")
EXIT_OK = 0
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "只管理本机持仓账本；本命令不会联网、不会连接券商或自动下单。"
            "持仓摘要外发默认关闭，必须逐通道显式授权。"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="查看本机当前持仓声明")

    replace = commands.add_parser("replace", help="用本机JSON文件整组替换当前持仓")
    replace.add_argument("--file", type=Path, required=True, help="UTF-8 JSON持仓文件")
    replace.add_argument(
        "--holding-weeks",
        type=int,
        choices=(1, 2, 4, 13, 26, 52),
        required=True,
        help="计划周期：1/2/4/13/26/52周",
    )
    replace.add_argument("--effective-at", type=_local_datetime, default=None)
    replace.add_argument("--change-id", default=None, help="可选的本次显式变更幂等编号")
    replace.add_argument(
        "--allow-holding-summary-serverchan",
        action="store_true",
        help="明确允许后续Server酱晚报包含持仓摘要（不含成本、金额或账户权重）",
    )
    replace.add_argument(
        "--allow-holding-summary-bark",
        action="store_true",
        help="明确允许后续Bark晚报包含持仓摘要（不含成本、金额或账户权重）",
    )
    replace.add_argument("--yes", action="store_true", help="确认整组替换本机持仓")

    clear = commands.add_parser("clear", help="明确记录当前已无持仓")
    clear.add_argument("--effective-at", type=_local_datetime, default=None)
    clear.add_argument("--yes", action="store_true", help="确认清空本机当前持仓")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    _repository: SQLiteRepository | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    repository = _repository or build_repository()
    try:
        if args.command == "list":
            payload = _portfolio_payload(get_active_holding_portfolio(repository))
        elif args.command == "replace":
            if not args.yes:
                raise ValueError("replace_requires_explicit_yes")
            positions = _read_positions(args.file)
            portfolio = replace_active_holdings(
                repository,
                positions,
                holding_weeks=args.holding_weeks,
                effective_at=args.effective_at or datetime.now(CN),
                source="user_confirmed_local_json",
                change_id=args.change_id,
                metadata={HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY: _delivery_channels(args)},
            )
            payload = _portfolio_payload(portfolio)
        else:
            if not args.yes:
                raise ValueError("clear_requires_explicit_yes")
            portfolio = clear_active_holdings(
                repository,
                effective_at=args.effective_at or datetime.now(CN),
                source="user_confirmed_local_cli",
                metadata={HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY: []},
            )
            payload = _portfolio_payload(portfolio)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return EXIT_OK
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(f"本机持仓未更改：{type(exc).__name__}", file=sys.stderr)
        return EXIT_ERROR


def _read_positions(path: Path) -> tuple[HoldingPositionInput, ...]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    rows = payload.get("positions") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("JSON must contain a non-empty positions list")
    results: list[HoldingPositionInput] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every positions item must be an object")
        metadata = _company_action_metadata(row)
        results.append(
            HoldingPositionInput(
                symbol=str(row.get("symbol", "")),
                name=str(row.get("name", "")),
                entry_date=_iso_date(row.get("entry_date")),
                cost_price=_optional_float(row.get("cost_price")),
                stock_sleeve_weight=float(row["stock_sleeve_weight"]),
                account_weight=_optional_float(row.get("account_weight")),
                source="user_confirmed_local_json",
                metadata=metadata,
            )
        )
    return tuple(results)


def _portfolio_payload(portfolio: object) -> dict[str, Any]:
    if portfolio is None:
        return {
            "status": "no_holding_statement",
            HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY: [],
            "positions": [],
        }
    channels = sorted(holding_summary_delivery_channels(portfolio))
    return {
        "status": portfolio.status,
        "holding_portfolio_id": portfolio.id,
        "holding_portfolio_version": portfolio.version,
        "holding_weeks": portfolio.holding_weeks,
        "effective_at": portfolio.effective_at.isoformat(),
        HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY: channels,
        "positions": [
            {
                "symbol": item.symbol,
                "name": item.name,
                "entry_date": item.entry_date.isoformat(),
                "cost_price": item.cost_price,
                "stock_sleeve_weight": item.stock_sleeve_weight,
                "account_weight": item.account_weight,
                "status": item.status,
                "source": item.source,
                "version": item.version,
            }
            for item in portfolio.positions
        ],
        "privacy": (
            "local ledger; only explicitly listed providers may receive a redacted holding summary"
        ),
        "network_used": False,
        "orders_enabled": False,
    }


def _delivery_channels(args: argparse.Namespace) -> list[str]:
    channels: list[str] = []
    if bool(args.allow_holding_summary_serverchan):
        channels.append("serverchan")
    if bool(args.allow_holding_summary_bark):
        channels.append("bark")
    return channels


def _iso_date(value: object) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("entry_date must use YYYY-MM-DD") from exc


def _optional_float(value: object) -> float | None:
    return None if value is None or value == "" else float(value)


def _company_action_metadata(row: dict[str, object]) -> dict[str, object]:
    """Keep only complete, explicit local company-action evidence."""

    clear = row.get("company_action_clear")
    fields = (
        row.get("company_action_clear_through"),
        row.get("company_action_evidence_source"),
        row.get("company_action_evidence_id"),
    )
    if clear is None and all(value in (None, "") for value in fields):
        return {}
    if not isinstance(clear, bool) or any(value in (None, "") for value in fields):
        raise ValueError("Company-action evidence must be complete and explicit")
    through = _iso_date(row["company_action_clear_through"])
    return {
        "company_action_clear": clear,
        "company_action_clear_through": through.isoformat(),
        "company_action_evidence_source": str(row["company_action_evidence_source"]).strip(),
        "company_action_evidence_id": str(row["company_action_evidence_id"]).strip(),
    }


def _local_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("时间必须使用ISO格式") from exc
    return parsed.replace(tzinfo=CN) if parsed.tzinfo is None else parsed


if __name__ == "__main__":
    raise SystemExit(main())
