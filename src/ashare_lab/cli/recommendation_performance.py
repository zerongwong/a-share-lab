"""Local CLI for recommendation maturity settlement and explicit reconstruction.

``settle`` reads only the local SQLite audit log and provider-verified overlay,
then uses the already configured scheduled notification channels.  ``reconstruct``
is deliberately offline: it requires explicit historical dates, verifies them
against the local overlay, and can archive only a visibly reconstructed cohort.
Neither command accepts credentials or has brokerage/order capability.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ashare_lab.adapters.market_overlay_store import MarketOverlayStore
from ashare_lab.bootstrap import application_data_dir, build_repository
from ashare_lab.cli.scheduled_sync import send_scheduled_notification
from ashare_lab.domain.data_sources import DEFAULT_MARKET_OVERLAY_SOURCE_ID
from ashare_lab.services.archive_recommendation_report import (
    archive_recommendation_report,
)
from ashare_lab.services.build_evening_digest import build_evening_research_digest
from ashare_lab.services.run_recommendation_performance import (
    load_available_local_corporate_action_evidence,
    run_recommendation_performance,
)

EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "使用本机不可变推荐档案和已验证收盘数据进行到期复盘；"
            "也可离线重建明确标记的历史观察档案。不会连接券商或自动下单。"
        )
    )
    commands = parser.add_subparsers(dest="command")

    settle = commands.add_parser(
        "settle",
        help="结算所有已到期批次，并通过本机已配置通知通道发送新增复盘（默认）",
    )
    settle.add_argument(
        "--overlay-root",
        type=Path,
        default=None,
        help="本机已验证、未复权收盘增量目录",
    )
    settle.add_argument(
        "--as-of",
        type=_iso_date,
        default=None,
        help="只使用截至该日的本地已验证数据（YYYY-MM-DD）",
    )

    reconstruct = commands.add_parser(
        "reconstruct",
        help="从本地历史数据重建研究档案；不联网、不通知、不能标记为原始档案",
    )
    reconstruct.add_argument(
        "--decision-date",
        type=_iso_date,
        required=True,
        help="历史报告生成日（YYYY-MM-DD）",
    )
    reconstruct.add_argument(
        "--plan-for-date",
        type=_iso_date,
        required=True,
        help="本地交易日链已验证的计划适用日（YYYY-MM-DD）",
    )
    reconstruct.add_argument("--csmar-root", type=Path, default=None, help="本地只读CSMAR目录")
    reconstruct.add_argument(
        "--overlay-root",
        type=Path,
        default=None,
        help="本机已验证、未复权收盘增量目录",
    )
    reconstruct.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="本地只读CSMAR证券主表目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "reconstruct":
            payload = _run_reconstruct(args)
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default))
            return EXIT_OK

        summary = _run_settle(args)
        print(
            json.dumps(
                _mapping_payload(summary),
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            )
        )
        if summary.failed_batch_ids or summary.notification_failed_batches:
            return EXIT_INCOMPLETE
        return EXIT_OK
    except Exception as exc:  # noqa: BLE001 - final CLI boundary must fail closed
        # Do not echo provider payloads, paths, rendered reports or credentials.
        print(f"推荐复盘未完成：{type(exc).__name__}", file=sys.stderr)
        return EXIT_ERROR


def _run_settle(args: argparse.Namespace):
    data_root = application_data_dir()
    overlay_root = (
        Path(getattr(args, "overlay_root", None) or data_root / "cache" / "market_overlay")
        .expanduser()
        .resolve()
    )
    repository = build_repository()
    store = MarketOverlayStore(overlay_root)
    return run_recommendation_performance(
        repository=repository,
        overlay_store=store,
        notifier=send_scheduled_notification,
        as_of=getattr(args, "as_of", None),
        corporate_action_loader=load_available_local_corporate_action_evidence,
    )


def _run_reconstruct(args: argparse.Namespace) -> dict[str, Any]:
    decision_date = args.decision_date
    plan_for_date = args.plan_for_date
    if plan_for_date <= decision_date:
        raise ValueError("plan_for_date must be later than decision_date")

    data_root = application_data_dir()
    csmar_root = Path(args.csmar_root or data_root / "cache" / "csmar").expanduser().resolve()
    overlay_root = (
        Path(args.overlay_root or data_root / "cache" / "market_overlay").expanduser().resolve()
    )
    reference_root = (
        Path(args.reference_root or data_root / "cache" / "csmar_reference").expanduser().resolve()
    )

    digest = build_evening_research_digest(
        dataset_root=csmar_root,
        overlay_root=overlay_root,
        reference_dataset_root=reference_root,
        decision_date=decision_date,
    )
    if digest.common_cutoff >= plan_for_date:
        raise ValueError("historical data cutoff must precede plan_for_date")
    _validate_local_plan_session(
        MarketOverlayStore(overlay_root),
        plan_for_date=plan_for_date,
        common_cutoff=digest.common_cutoff,
    )
    digest = replace(digest, plan_for_date=plan_for_date)
    bundle = archive_recommendation_report(
        digest,
        build_repository(),
        archive_nature="reconstructed",
    )
    return {
        "status": "archived_reconstructed",
        "archive_nature": "reconstructed",
        "report_id": bundle.report_id,
        "content_hash": bundle.content_hash,
        "decision_date": decision_date,
        "plan_for_date": plan_for_date,
        "common_cutoff": digest.common_cutoff,
        "batch_count": len(bundle.batches),
        "member_count": len(bundle.members),
        "network_used": False,
        "notification_sent": False,
        "orders_enabled": False,
    }


def _validate_local_plan_session(
    store: MarketOverlayStore,
    *,
    plan_for_date: date,
    common_cutoff: date,
) -> None:
    manifest = store.read_verified_manifest(source_id=DEFAULT_MARKET_OVERLAY_SOURCE_ID)
    required = {"trade_date", "previous_trade_date", "adjustment"}
    if required - set(manifest.columns):
        raise ValueError("local verified manifest is incomplete")
    rows = manifest.copy()
    rows["trade_date"] = rows["trade_date"].map(_manifest_date)
    rows["previous_trade_date"] = rows["previous_trade_date"].map(_manifest_date)
    selected = rows.loc[rows["trade_date"] == plan_for_date]
    if len(selected) != 1:
        raise ValueError("plan_for_date is not a unique locally verified session")
    row = selected.iloc[0]
    if str(row["adjustment"]) != "none":
        raise ValueError("reconstruction requires the unadjusted verified overlay")
    if row["previous_trade_date"] != common_cutoff:
        raise ValueError("plan_for_date is not the verified session after common_cutoff")


def _iso_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("日期必须使用YYYY-MM-DD") from exc
    return parsed


def _manifest_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ValueError("local verified manifest contains an invalid date") from exc


def _mapping_payload(value: object) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return value
    raise TypeError("performance runner returned an invalid summary")


def _json_default(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    raise SystemExit(main())
