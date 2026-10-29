"""Opt-in, idempotent completion notices for recommendation price references.

No account amounts, actual fills, or holdings are read here. An accepted
provider receipt is not proof of phone delivery. Existing EOD sync invokes
this service; it installs no separate timer and sends no empty status pushes.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from ashare_lab.ports.notifications import NotificationMessage
from ashare_lab.services.daily_update_lock import daily_update_lock
from ashare_lab.services.intraday_stop_monitor import write_private_json

CN = ZoneInfo("Asia/Shanghai")


def authorized(root: Path) -> bool:
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        return (
            isinstance(config, dict)
            and config.get("enabled") is True
            and config.get("authorized_channels") == ["serverchan"]
            and config.get("allow_reference_exit_notifications") is True
        )
    except (OSError, ValueError):
        return False


def _completed_record(episode: dict, through: date) -> dict:
    """Validate arithmetic independently at the public notification boundary."""
    record = {
        key: episode[key]
        for key in (
            "episode_id",
            "symbol",
            "name",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "profit_per_share",
            "return_fraction",
            "trading_days",
            "daily_average_return",
            "daily_average_profit_per_share",
        )
    }
    entry, exit_day = (
        date.fromisoformat(record["entry_date"]),
        date.fromisoformat(record["exit_date"]),
    )
    if not date(2026, 10, 8) <= entry <= exit_day <= through:
        raise ValueError("invalid reference interval")
    n = record["trading_days"]
    if type(n) is not int or not 1 <= n <= (exit_day - entry).days + 1:
        raise ValueError("invalid reference session count")
    for key in (
        "entry_price",
        "exit_price",
        "profit_per_share",
        "return_fraction",
        "daily_average_return",
        "daily_average_profit_per_share",
    ):
        value = record[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError("invalid reference number")
    buy, sell = record["entry_price"], record["exit_price"]
    if buy <= 0 or sell <= 0:
        raise ValueError("nonpositive reference price")
    expected = {
        "profit_per_share": sell - buy,
        "return_fraction": sell / buy - 1,
        "daily_average_return": (sell / buy - 1) / n,
        "daily_average_profit_per_share": (sell - buy) / n,
    }
    if any(
        not math.isclose(record[key], value, rel_tol=1e-7, abs_tol=1e-9)
        for key, value in expected.items()
    ):
        raise ValueError("reference arithmetic mismatch")
    if not all(
        isinstance(record[key], str) and record[key] for key in ("episode_id", "symbol", "name")
    ):
        raise ValueError("reference identity missing")
    return record


def render_reference_exits(records: list[dict], *, through: date) -> str:
    sections = [f"🌷 推荐生命周期 · 价格基准\n\n行情截至 {through:%Y-%m-%d}"]
    for row in records:
        name = " ".join(row["name"].split())[:24]
        sections.append(
            f"**{name} · 已触发退出条件**\n\n"
            f"{row['entry_date']} 开盘 {row['entry_price']:.2f} 元"
            f" → {row['exit_date']} 收盘 {row['exit_price']:.2f} 元\n\n"
            f"持有 {row['trading_days']} 个交易日\n\n"
            f"累计 {row['return_fraction']:+.2%} ｜ {row['profit_per_share']:+.4f} 元/股\n\n"
            f"日均 {row['daily_average_return']:+.2%} ｜ "
            f"{row['daily_average_profit_per_share']:+.4f} 元/股/日"
        )
    sections.append(
        "🌱 推荐日开盘至退出信号日收盘的模拟价格表现，未计费用分红；"
        "不代表买点条件已触发、可按该价成交或实际账户收益。日均为累计÷交易日数。"
    )
    return "\n\n---\n\n".join(sections)


def dispatch_reference_exits(*, summary: dict, root: Path, notifier, now: datetime) -> dict:
    """At most five newly closed references per sync, with durable retry state."""
    event = {"status": "disabled", "accepted": 0, "delivery_confirmed": False}
    if not authorized(root):
        return event
    if now.tzinfo is None or now.utcoffset() is None:
        return {**event, "status": "invalid_clock"}
    now = now.astimezone(CN)
    from ashare_lab.services.recommendation_reference import (
        REFERENCE_METHOD,
        VERSION_ISOLATED_REFERENCE_METHOD,
    )

    if summary.get("method_version") not in {REFERENCE_METHOD, VERSION_ISOLATED_REFERENCE_METHOD}:
        return {**event, "status": "reference_method_mismatch"}
    # V2 changes cohort grouping, not episode identity. Keep the receipt
    # namespace stable so previously announced V1 exits are not sent again.
    receipt_method = REFERENCE_METHOD
    try:
        through = date.fromisoformat(summary["through"])
        if through > now.date():
            raise ValueError("future reference")
        episodes = summary["episodes"]
        if not isinstance(episodes, list):
            raise ValueError("invalid episodes")
        records = [
            _completed_record(row, through) for row in episodes if row.get("status") == "closed"
        ]
        if len({row["episode_id"] for row in records}) != len(records):
            raise ValueError("duplicate reference episode")
        if now.time() < time(15) and any(
            row["exit_date"] == now.date().isoformat() for row in records
        ):
            raise ValueError("today's close is not available before market close")
    except (KeyError, TypeError, ValueError, AttributeError):
        return {**event, "status": "reference_integrity_error"}
    path = root / "delivery-state.json"
    with daily_update_lock(root / "delivery.lock") as acquired:
        if not acquired:
            return {**event, "status": "already_running"}
        try:
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            if not isinstance(state, dict):
                raise ValueError("invalid receipt")
            closed_keys = {f"{receipt_method}:{row['episode_id']}" for row in records}
            for key, receipt in state.items():
                if not isinstance(receipt, dict):
                    raise ValueError("invalid receipt")
                # New evidence may invalidate a formerly completed episode.
                # Do not silently retain published results as if still valid,
                # or auto-replace them with a different performance claim.
                if (
                    key.startswith(f"{receipt_method}:")
                    and receipt.get("accepted") is True
                    and key not in closed_keys
                ):
                    return {**event, "status": "accepted_reference_changed_needs_review"}
            pending = []
            for row in sorted(records, key=lambda r: (r["exit_date"], r["episode_id"])):
                key = f"{receipt_method}:{row['episode_id']}"
                encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                digest = hashlib.sha256(encoded.encode()).hexdigest()
                prior = state.get(key, {})
                if not isinstance(prior, dict):
                    raise ValueError("invalid receipt")
                if prior.get("accepted") is True:
                    if prior.get("payload_hash") != digest:
                        return {**event, "status": "accepted_reference_changed_needs_review"}
                    continue
                attempts = prior.get("attempts", 0)
                if type(attempts) is not int or attempts < 0:
                    raise ValueError("invalid attempt count")
                if prior.get("attempted_at"):
                    last = datetime.fromisoformat(prior["attempted_at"])
                    if last.tzinfo is None or (now - last).total_seconds() < (
                        180 if attempts < 3 else 3600
                    ):
                        continue
                pending.append((key, row, digest, attempts))
            pending = pending[:5]
        except (OSError, ValueError, TypeError, AttributeError):
            return {**event, "status": "delivery_state_needs_review"}
        if not pending:
            return {**event, "status": "no_new_completed_references"}

        def allowed(channel="serverchan"):
            return channel == "serverchan" and authorized(root)

        message = NotificationMessage(
            title="A股推荐生命周期 · 退出复盘",
            body=render_reference_exits([row for _, row, _, _ in pending], through=through),
            group="A股研究室·推荐生命周期",
            holding_authorization_guard=allowed,
            unauthorized_body="推荐生命周期通知授权已关闭，本条不含研究结果。",
        )
        if not allowed():
            return {**event, "status": "authorization_changed"}
        for key, _, digest, attempts in pending:
            state[key] = {
                "payload_hash": digest,
                "attempts": attempts + 1,
                "attempted_at": now.isoformat(),
                "accepted": False,
                "delivery_confirmed": False,
            }
        write_private_json(path, state)
        try:
            accepted = notifier(message) is True
        except Exception:
            accepted = False
        if accepted:
            for key, _, _, _ in pending:
                state[key]["accepted"] = True
            write_private_json(path, state)
        return {
            **event,
            "status": "provider_accepted" if accepted else "provider_not_accepted",
            "accepted": len(pending) if accepted else 0,
        }
