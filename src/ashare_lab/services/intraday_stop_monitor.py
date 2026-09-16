"""Minute-polled holding-risk alerts, isolated from full-market updates.

No screening, stop ratcheting, trade execution, or membership changes occur.
Structural touches are provisional until the existing completed-close review.
"""

from __future__ import annotations

import json
import math
import os
from datetime import date, time
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from ashare_lab.adapters.free_intraday_quotes import crosscheck, fresh_quotes
from ashare_lab.ports.notifications import NotificationMessage, NotificationUrgency
from ashare_lab.services.holding_ledger import get_active_holding_portfolio
from ashare_lab.services.intraday_alert_store import (
    mark_attempt,
    mark_delivery,
    pending_alerts,
    put_alert,
)

CN = ZoneInfo("Asia/Shanghai")
METHOD_VERSION = "intraday-holding-stop-monitor-v1"


def in_monitor_hours(now):
    now = now.astimezone(CN)
    return now.weekday() < 5 and (
        time(9, 25) <= now.time() <= time(11, 30, 59) or time(13) <= now.time() <= time(15, 0, 59)
    )


def read_config(root: Path):
    try:
        config = json.loads((root / "config.json").read_text())
        return (
            config
            if config.get("enabled") is True and config.get("authorized_channels") == ["serverchan"]
            else None
        )
    except (OSError, ValueError, AttributeError):
        return None


def write_private_json(path: Path, document):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, default=str)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _corporate_clear(
    holding,
    now,
    evidence=None,
    *,
    allow_manual_fallback=True,
    require_knowledge_time=False,
):
    today = now.date()
    if evidence is not None:
        knowledge_time = getattr(evidence, "knowledge_time", None)
        return bool(
            evidence.symbol == holding.symbol
            and evidence.clear
            and evidence.from_date is not None
            and evidence.from_date <= holding.entry_date
            and evidence.through_date >= today
            and evidence.source
            and evidence.evidence_id
            and (
                (knowledge_time is None and not require_knowledge_time)
                or (
                    knowledge_time is not None
                    and knowledge_time.tzinfo is not None
                    and knowledge_time <= now
                )
            )
        )
    if not allow_manual_fallback:
        return False
    m = holding.metadata
    try:
        return (
            m.get("company_action_clear") is True
            and date.fromisoformat(m["company_action_clear_from"]) <= holding.entry_date
            and date.fromisoformat(m["company_action_clear_through"]) == today
            and bool(m["company_action_evidence_source"])
            and bool(m["company_action_evidence_id"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def run_monitor(
    repository,
    *,
    root,
    now,
    quote_fetcher,
    calendar,
    notifier,
    clock=None,
    company_action_clear_by_symbol=None,
    company_action_clearance_loader=None,
    allow_manual_company_action_fallback=True,
    company_action_authorization_checker=None,
):
    now = now.astimezone(CN)
    event = {
        "method": METHOD_VERSION,
        "checked_at": now.isoformat(),
        "status": "disabled",
        "accepted": 0,
        "failed": 0,
        "orders_enabled": False,
        "interval_seconds": 60,
    }
    if read_config(root) is None:
        return event
    portfolio = get_active_holding_portfolio(repository)
    if portfolio is None or not portfolio.positions or portfolio.status != "active":
        event["status"] = "no_holdings"
        return event
    if portfolio.effective_at > now:
        event["status"] = "holding_not_yet_effective"
        return event
    if not in_monitor_hours(now):
        event["status"] = "outside_session"
        return event

    def allowed(channel="serverchan"):
        current = get_active_holding_portfolio(repository)
        return (
            channel == "serverchan"
            and read_config(root) is not None
            and current is not None
            and (current.id, current.version) == (portfolio.id, portfolio.version)
        )

    def health(reason):
        put_alert(
            repository,
            incident=f"{portfolio.id}:health:{now.date()}:{reason}",
            position_key=None,
            portfolio=portfolio,
            kind="health",
            now=now,
            confirmed=False,
            payload={
                "body": f"{now:%m-%d %H:%M}｜盘中监控需注意\n{reason}\n请核对券商行情；本程序不自动下单。"
            },
        )

    try:
        open_today = calendar(now.date())
    except Exception:
        open_today = None
    if open_today is False:
        event["status"] = "market_closed"
        return event
    if open_today is not True:
        health("交易日历未核验，暂不能可靠监控。")
        event["status"] = "calendar_unavailable"
    elif len(portfolio.positions) > 8:
        health("持仓数量超出已验证的8只监控范围，请核验登记。")
        event["status"] = "holding_scope_unavailable"
    else:
        clearances = company_action_clear_by_symbol
        if clearances is None and company_action_clearance_loader is not None:
            try:
                clearances = company_action_clearance_loader(
                    repository,
                    as_of=now.date(),
                    reviewed_at=now,
                    phase="intraday",
                )
                if not isinstance(clearances, dict):
                    clearances = {}
            except Exception:  # noqa: BLE001 - UNKNOWN must not stop risk monitoring
                clearances = {}
            if clock is not None:
                now = clock().astimezone(CN)
            if not allowed():
                event["status"] = "holding_or_authorization_changed"
                return event

        def manual_company_action_fallback_allowed():
            if not allow_manual_company_action_fallback:
                return False
            if company_action_authorization_checker is None:
                return True
            try:
                return company_action_authorization_checker() is False
            except Exception:  # noqa: BLE001 - a broken grant check fails closed
                return False

        try:
            batches = quote_fetcher(tuple(h.symbol for h in portfolio.positions))
        except Exception:
            batches = {}
        if clock is not None:
            now = clock().astimezone(CN)
        if not allowed():
            event["status"] = "holding_or_authorization_changed"
            return event
        issues = set()
        for holding in portfolio.positions:
            if holding.entry_date > now.date():
                issues.add("入场日期晚于今天，无法核验该笔持仓。")
                continue
            quotes = fresh_quotes(
                tuple(
                    batch[holding.symbol] for batch in batches.values() if holding.symbol in batch
                ),
                now=now,
            )
            paired = crosscheck(quotes)
            if not paired:
                issues.add("部分持仓行情过期、缺失或双源不一致；不能当作安全持有。")
            if not quotes:
                continue
            clearance = (clearances or {}).get(holding.symbol)
            clear = _corporate_clear(
                holding,
                now,
                clearance,
                allow_manual_fallback=manual_company_action_fallback_allowed(),
                require_knowledge_time=(
                    company_action_clearance_loader is not None
                    and company_action_clear_by_symbol is None
                ),
            )
            if not clear:
                issues.add(
                    "已发现公司行动；触线会优先核对除权影响，不伪造确认卖出。"
                    if clearance is not None and clearance.clear is False
                    else "除权/分红区间证据未齐；触线会提示优先核验，不伪造确认卖出。"
                )
            stored = repository.get_holding_protective_stop(holding.position_key)
            if stored is not None and str(stored["data_cutoff"]) >= now.date().isoformat():
                stored = None  # Today's closing line was not knowable at the open.
            details = {} if stored is None else stored.get("details_json", {})
            if stored is None:
                issues.add("部分持仓尚无已核验结构保护线；成本检查仍独立执行。")
            cost_line = None
            if (
                holding.cost_price is not None
                and math.isfinite(holding.cost_price)
                and holding.cost_price > 0
            ):
                cost_line = max(
                    Decimal(str(holding.cost_price)) * Decimal(".92"),
                    Decimal(str(details.get("cost_stop") or 0)),
                )
            else:
                issues.add("部分持仓未登记有效成本，8%成本止损无法核验。")
            # A day's low may precede today's fill/cost change; use only fresh
            # last prices then. Older unchanged holdings may use the day's low.
            use_low = (
                holding.entry_date < now.date()
                and portfolio.effective_at.astimezone(CN).date() < now.date()
            )
            tested = tuple(Decimal(str(q.low if use_low else q.price)) for q in quotes)
            cost_touch = cost_line is not None and min(tested) <= cost_line
            structural_line = None if stored is None else float(stored["effective_stop"])
            structural_touch = (
                structural_line is not None and min(q.price for q in quotes) <= structural_line
            )
            if not cost_touch and not structural_touch:
                continue
            # Agreement on last prices alone does not validate a source's low.
            # Both fresh sources must independently show the cost-line touch.
            confirmed = bool(
                cost_touch and paired and clear and all(value <= cost_line for value in tested)
            )
            kind = "cost_exit" if confirmed else "cost_review" if cost_touch else "structure_touch"
            line = float(cost_line) if cost_touch else structural_line
            latest = max(quotes, key=lambda q: q.quoted_at)
            label = (
                "🔴 已触及8%成本止损｜退出风险提示"
                if confirmed
                else "⚠️ 疑似触及8%成本止损｜立即核验"
                if cost_touch
                else "⚠️ 盘中触及结构保护线｜等待收盘确认"
            )
            body = (
                f"{holding.name}（{holding.symbol}）\n{label}\n"
                f"行情 {latest.quoted_at:%m-%d %H:%M:%S}｜现价{latest.price:.2f}｜触发线{line:.2f}\n"
                + (
                    "除权/分红或双源行情待核验，请立即查看券商。\n"
                    if cost_touch and not confirmed
                    else ""
                )
            )
            incident = f"{holding.position_key}:{kind}"
            if kind != "cost_exit":
                incident += f":{now.date()}:{line:.4f}"
            put_alert(
                repository,
                incident=incident,
                position_key=holding.position_key,
                portfolio=portfolio,
                kind=kind,
                now=now,
                confirmed=confirmed,
                payload={
                    "body": body,
                    "line": line,
                    "quote_sources": [q.source for q in quotes],
                    "quote_times": [q.quoted_at.isoformat() for q in quotes],
                    "company_action_clear": clear,
                    "paired": paired,
                    "comparison": "day_low" if use_low else "last_price",
                    "compared_prices": [float(value) for value in tested],
                    "method": METHOD_VERSION,
                },
            )
        if issues:
            health("\n".join(sorted(issues)))
        event["status"] = "degraded" if issues else "checked"
    if not allowed():
        event["status"] = "holding_or_authorization_changed"
        return event
    keys = {h.position_key for h in portfolio.positions}
    pending = pending_alerts(repository, position_keys=keys, portfolio_id=portfolio.id, now=now)
    if pending and allowed():
        message = NotificationMessage(
            title="A股盘中风险提醒"
            if any(i["kind"] != "health" for i in pending)
            else "A股盘中监控状态",
            body="\n\n".join(i["payload"]["body"].strip() for i in pending)
            + "\n\n仅预警，不自动下单；T+1、停牌、跌停可能影响卖出。确认成交后再评估替补，无合格项留现金。",
            urgency=NotificationUrgency.TIME_SENSITIVE,
            holding_authorization_guard=allowed,
            unauthorized_body="持仓或监控授权已变化，本次旧持仓提醒取消。",
        )
        for incident in pending:
            mark_attempt(repository, incident["incident_key"], now)
        try:
            accepted = notifier(message) is True
        except Exception:
            accepted = False
        for incident in pending:
            mark_delivery(repository, incident["incident_key"], accepted)
        event["alert_count"] = len(pending)
        event["accepted" if accepted else "failed"] += 1
    return event
