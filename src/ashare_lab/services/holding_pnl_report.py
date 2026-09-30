"""Current-position floating P&L estimates, separate from account NAV and orders.

Only explicit quantities/costs and same-day two-source closing snapshots may
enter an amount. Company-action coverage is mandatory. Invested principal is
disclosed only with its own private authorization; quantities, market value,
cash and whole-account profit are never disclosed by this report.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

from ashare_lab.ports.notifications import NotificationMessage
from ashare_lab.services.daily_update_lock import daily_update_lock
from ashare_lab.services.holding_ledger import get_active_holding_portfolio
from ashare_lab.services.intraday_stop_monitor import _corporate_clear, write_private_json

CN = ZoneInfo("Asia/Shanghai")
METHOD_VERSION = "holding-floating-pnl-v2"
_CENT = Decimal("0.01")
_FINAL_RETRY = time(15, 55)


@dataclass(frozen=True)
class HoldingPnlRow:
    symbol: str
    name: str
    pnl_percent: Decimal | None
    pnl_amount: Decimal | None
    issues: tuple[str, ...]


@dataclass(frozen=True)
class HoldingPnlReport:
    as_of: date
    rows: tuple[HoldingPnlRow, ...]
    pnl_percent: Decimal | None
    pnl_amount: Decimal | None
    invested_principal: Decimal | None
    daily_percent: Decimal | None
    daily_amount: Decimal | None
    held_trading_days: int | None
    complete: bool


def in_report_hours(now: datetime) -> bool:
    local = now.astimezone(CN)
    return local.weekday() < 5 and time(15, 30) <= local.time() < time(16)


def read_pnl_config(root: Path) -> dict | None:
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if (
            isinstance(config, dict)
            and config.get("enabled") is True
            and config.get("authorized_channels") == ["serverchan"]
            and config.get("allow_pnl_amounts") is True
        ):
            return config
    except (OSError, ValueError):
        pass
    return None


def _known_market_closed(root: Path, day: date) -> bool:
    try:
        state = json.loads((root / "calendar.json").read_text(encoding="utf-8"))
        return (
            isinstance(state, dict)
            and state.get("date") == day.isoformat()
            and state.get("open") is False
        )
    except (OSError, ValueError):
        return False


def _known_previous_session(root: Path, day: date) -> date | None:
    """Read only the already-verified current-day calendar receipt."""
    try:
        state = json.loads((root / "calendar.json").read_text(encoding="utf-8"))
        if (
            isinstance(state, dict)
            and state.get("date") == day.isoformat()
            and state.get("open") is True
        ):
            prior = date.fromisoformat(state["previous_session"])
            if day - timedelta(days=20) <= prior < day:
                return prior
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return None


def _positive_decimal(value) -> Decimal | None:
    if isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
        return result if result.is_finite() and result > 0 else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def _closing_snapshot(
    symbol: str, batches: Mapping, now: datetime
) -> tuple[Decimal, Decimal] | None:
    """Both completed-session closes and previous closes must agree."""
    prices, previous = [], []
    try:
        for source in ("tencent", "sina"):
            quote = batches[source][symbol]
            if quote.symbol != symbol or quote.source != source:
                return None
            stamp = quote.quoted_at
            if not isinstance(stamp, datetime) or stamp.tzinfo is None:
                return None
            local = stamp.astimezone(CN)
            if local.date() != now.date() or local.time() < time(15) or local > now:
                return None
            price, prior = _positive_decimal(quote.price), _positive_decimal(quote.previous_close)
            low, high = _positive_decimal(quote.low), _positive_decimal(quote.high)
            if (
                price is None
                or prior is None
                or low is None
                or high is None
                or not low <= price <= high
            ):
                return None
            prices.append(price)
            previous.append(prior)
        if abs(prices[0] - prices[1]) > Decimal("0.011"):
            return None
        if prices[0].quantize(_CENT) != prices[1].quantize(_CENT):
            return None
        if previous[0] != previous[1]:
            return None
        return prices[0].quantize(_CENT, rounding=ROUND_HALF_UP), previous[0]
    except (KeyError, AttributeError, TypeError, ValueError, InvalidOperation):
        return None


def _verified_sessions(values, *, start: date, end: date) -> tuple[date, ...] | None:
    try:
        sessions = tuple(values)
        if (
            not sessions
            or any(type(day) is not date or not start <= day <= end for day in sessions)
            or sessions != tuple(sorted(set(sessions)))
            or end not in sessions
        ):
            return None
        return sessions
    except (TypeError, ValueError):
        return None


def _calendar_start(_as_of: date) -> date:
    # Public A-share era lower bound, independent of the private entry date.
    # The production loader queries only a recent fixed window and extends a
    # locally verified calendar cache; this is only the validation lower bound.
    return date(1990, 1, 1)


def build_holding_pnl_report(
    portfolio,
    *,
    batches: Mapping,
    clearances: Mapping,
    now: datetime,
    trading_sessions: tuple[date, ...] | None = None,
    previous_session: date | None = None,
    previous_portfolio=None,
):
    now = now.astimezone(CN)
    rows = []
    total_amount, total_cost = Decimal(0), Decimal(0)
    daily_amount, prior_value = Decimal(0), Decimal(0)
    common_entry = max((holding.entry_date for holding in portfolio.positions), default=None)
    sessions = (
        _verified_sessions(
            trading_sessions,
            start=_calendar_start(now.date()),
            end=now.date(),
        )
        if common_entry is not None and trading_sessions is not None
        else None
    )
    held_days = (
        sum(day >= common_entry for day in sessions)
        if sessions is not None and common_entry in sessions
        else None
    )
    if previous_session is None:
        previous_session = max((day for day in (sessions or ()) if day < now.date()), default=None)
    elif type(previous_session) is not date or not previous_session < now.date():
        previous_session = None
    unchanged_since_previous_close = bool(
        previous_session is not None
        and previous_portfolio is not None
        and (previous_portfolio.id, previous_portfolio.version)
        == (portfolio.id, portfolio.version)
        and portfolio.effective_at.astimezone(CN)
        <= datetime.combine(previous_session, time(15), tzinfo=CN)
        and all(holding.entry_date <= previous_session for holding in portfolio.positions)
    )
    for holding in portfolio.positions:
        issues = []
        cost = _positive_decimal(holding.cost_price)
        quantity = holding.metadata.get("quantity")
        quantity_confirmed = (
            type(quantity) is int
            and quantity > 0
            and holding.metadata.get("quantity_user_confirmed") is True
        )
        snapshot = _closing_snapshot(holding.symbol, batches, now)
        if cost is None:
            issues.append("成本待补齐")
        if not quantity_confirmed:
            issues.append("股数待补齐")
        if snapshot is None:
            issues.append("今日双源收盘快照待核验")
        evidence = clearances.get(holding.symbol)
        corporate_clear = False
        with suppress(AttributeError, TypeError, ValueError):
            corporate_clear = (
                evidence is not None
                and evidence.clear is True
                and evidence.through_date == now.date()
                and _corporate_clear(
                    holding,
                    now,
                    evidence,
                    allow_manual_fallback=False,
                    require_knowledge_time=True,
                )
            )
        if not corporate_clear:
            issues.append("除权分红核验未完成")
        if holding.entry_date > now.date():
            issues.append("入场日期待核验")
        can_value = (
            cost is not None
            and snapshot is not None
            and corporate_clear
            and holding.entry_date <= now.date()
        )
        close, previous_close = snapshot if snapshot is not None else (None, None)
        percent = ((close - cost) / cost * Decimal(100)) if can_value else None
        amount = (close - cost) * quantity if can_value and quantity_confirmed else None
        if amount is not None:
            total_amount += amount
            total_cost += cost * quantity
            if unchanged_since_previous_close:
                daily_amount += (close - previous_close) * quantity
                prior_value += previous_close * quantity
        rows.append(HoldingPnlRow(holding.symbol, holding.name, percent, amount, tuple(issues)))
    complete = bool(rows) and all(not row.issues for row in rows)
    return HoldingPnlReport(
        now.date(),
        tuple(rows),
        total_amount / total_cost * Decimal(100) if complete and total_cost > 0 else None,
        total_amount if complete else None,
        total_cost if complete else None,
        daily_amount / prior_value * Decimal(100)
        if complete and unchanged_since_previous_close and prior_value > 0
        else None,
        daily_amount if complete and unchanged_since_previous_close else None,
        held_days,
        complete,
    )


def _signed(value: Decimal) -> str:
    rounded = value.quantize(_CENT, rounding=ROUND_HALF_UP)
    if rounded == 0:
        rounded = abs(rounded)
    return f"{rounded:+,.2f}"


def render_holding_pnl_report(
    report: HoldingPnlReport, *, include_principal: bool = False
) -> str:
    # Server酱仅支持 Markdown；彩色 emoji、加粗和空行无需 HTML/CSS。
    sections = [f"🌷 {report.as_of:%Y-%m-%d} · 收盘"]
    days = "待核验" if report.held_trading_days is None else f"{report.held_trading_days} 个交易日"
    sections.append(f"⏳ 现组合共同持有：{days}")
    if include_principal:
        principal = (
            "待核验"
            if report.invested_principal is None
            else f"{report.invested_principal.quantize(_CENT, rounding=ROUND_HALF_UP):,.2f} 元"
        )
        sections.append(f"💼 持仓投入本金：{principal}")
    daily = (
        "待核验（持仓变动或昨收基准不足）"
        if report.daily_percent is None or report.daily_amount is None
        else f"{_signed(report.daily_percent)}% ｜ {_signed(report.daily_amount)} 元"
    )
    cumulative = (
        "待核验"
        if report.pnl_percent is None or report.pnl_amount is None
        else f"{_signed(report.pnl_percent)}% ｜ {_signed(report.pnl_amount)} 元"
    )
    sections.append(f"🩵 **当日收益（价差）**：{daily}")
    sections.append(f"💜 **累计收益（价差）**：{cumulative}")
    sections.append("🌸 持仓明细 · 累计价差")
    for row in report.rows:
        percent = "—" if row.pnl_percent is None else f"{_signed(row.pnl_percent)}%"
        amount = "金额待核验" if row.pnl_amount is None else f"{_signed(row.pnl_amount)} 元"
        name = " ".join(str(row.name).split())[:24]
        suffix = "；" + "、".join(row.issues) if row.issues else ""
        sections.append(f"{name}：{percent} ｜ {amount}{suffix}")
    sections.append("🌱 数据来源：腾讯＋新浪收盘快照 · AKShare等交易日历")
    return "\n\n".join(sections)


def run_holding_pnl(
    repository,
    *,
    root: Path,
    now: datetime,
    quote_fetcher: Callable,
    calendar: Callable,
    trading_sessions_loader: Callable[[date, date], tuple[date, ...]] | None = None,
    notifier: Callable,
    company_action_clearance_loader: Callable,
    clock: Callable[[], datetime] | None = None,
) -> dict:
    """One private, revision-guarded report; no ledger or NAV writes."""
    clock = clock or (lambda: datetime.now(CN))
    now = now.astimezone(CN)
    event = {
        "method": METHOD_VERSION,
        "checked_at": now.isoformat(),
        "status": "disabled",
        "delivery_confirmed": False,
        "orders_enabled": False,
    }

    def finish(status):
        return {**event, "status": status}

    config = read_pnl_config(root)
    if config is None:
        return event
    principal_authorized = config.get("allow_invested_principal") is True
    if not in_report_hours(now):
        return finish("outside_report_window")
    with daily_update_lock(root / "report.lock") as acquired:
        if not acquired:
            return finish("already_running")
        portfolio = get_active_holding_portfolio(repository)
        if portfolio is None or portfolio.status != "active" or not portfolio.positions:
            return finish("no_holdings")
        if portfolio.effective_at > now:
            return finish("holding_not_yet_effective")
        receipt_path = root / "delivery-state.json"
        try:
            accepted = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            accepted = {}
        if isinstance(accepted, dict) and accepted.get("accepted_date") == now.date().isoformat():
            return finish("already_provider_accepted")
        identity = (portfolio.id, portfolio.version)

        def allowed(channel="serverchan"):
            try:
                fresh = clock().astimezone(CN)
                current = get_active_holding_portfolio(repository)
                fresh_config = read_pnl_config(root)
                return (
                    channel == "serverchan"
                    and fresh_config is not None
                    and (
                        not principal_authorized
                        or fresh_config.get("allow_invested_principal") is True
                    )
                    and fresh.date() == now.date()
                    and in_report_hours(fresh)
                    and fresh.time() < time(15, 59)
                    and current is not None
                    and current.status == "active"
                    and bool(current.positions)
                    and (current.id, current.version) == identity
                    and current.effective_at <= fresh
                )
            except Exception:
                return False

        if not allowed():
            return finish("holding_or_authorization_changed")
        try:
            open_today = calendar(now.date())
        except Exception:
            open_today = None
        if open_today is False:
            return finish("market_closed")
        report = None
        if open_today is True:
            if not allowed():
                return finish("holding_or_authorization_changed")
            try:
                clearances = company_action_clearance_loader(
                    repository, as_of=now.date(), reviewed_at=clock(), phase="intraday"
                )
                if not isinstance(clearances, Mapping):
                    clearances = {}
            except Exception:
                clearances = {}
            if not allowed():
                return finish("holding_or_authorization_changed")
            try:
                # These public data services receive stock identities only.
                batches = quote_fetcher(tuple(h.symbol for h in portfolio.positions))
                if not isinstance(batches, Mapping):
                    batches = {}
            except Exception:
                batches = {}
            if not allowed():
                return finish("holding_or_authorization_changed")
            sessions = None
            if trading_sessions_loader is not None:
                start = _calendar_start(now.date())
                try:
                    sessions = _verified_sessions(
                        trading_sessions_loader(start, now.date()), start=start, end=now.date()
                    )
                except Exception:
                    sessions = None
            if not allowed():
                return finish("holding_or_authorization_changed")
            previous_session = max(
                (day for day in (sessions or ()) if day < now.date()), default=None
            )
            previous_session = previous_session or _known_previous_session(root, now.date())
            previous_portfolio = (
                get_active_holding_portfolio(repository, as_of=previous_session)
                if previous_session is not None
                else None
            )
            report = build_holding_pnl_report(
                portfolio,
                batches=batches,
                clearances=clearances,
                now=clock(),
                trading_sessions=sessions,
                previous_session=previous_session,
                previous_portfolio=previous_portfolio,
            )
        fresh = clock().astimezone(CN)
        if (report is None or not report.complete) and fresh.time() < _FINAL_RETRY:
            return finish("awaiting_verified_pnl")
        if not allowed():
            return finish("holding_or_authorization_changed")
        public = "收盘收益核验未完成；持仓或授权可能已变化。本条不含收益结论，请勿据此交易。"
        if report is None:
            title = "A股收盘收益核验未完成"
            body = "交易日及收盘数据核验尚未完成；暂无法提供可靠收益。本条仅为系统状态提醒，不代表今日一定开市。"
        else:
            title = "A股持仓收盘日报" if report.complete else "A股收盘盈亏待核验"
            body = render_holding_pnl_report(report, include_principal=principal_authorized)
        message = NotificationMessage(
            title=title,
            body=body,
            group="A股研究室·持仓收盘日报",
            holding_authorization_guard=allowed,
            unauthorized_body=public,
        )
        # The adapter repeats this same guard at its final disclosure boundary.
        if not allowed():
            return finish("holding_or_authorization_changed")
        try:
            provider_accepted = notifier(message) is True
        except Exception:
            provider_accepted = False
        if not provider_accepted:
            return finish("provider_not_accepted")
        if not allowed():
            return finish("holding_or_authorization_changed")
        write_private_json(
            receipt_path,
            {
                "accepted_date": now.date().isoformat(),
                "accepted_channels": ["serverchan"],
                "report_complete": bool(report and report.complete),
                "delivery_confirmed": False,
            },
        )
        return finish(
            "provider_accepted" if report and report.complete else "pending_provider_accepted"
        )


def send_unverified_holding_pnl(
    repository, *, root: Path, clock: Callable[[], datetime], notifier: Callable
) -> dict:
    """The same daily report's final fallback, without quotes or private body."""
    now = clock().astimezone(CN)
    event = {
        "method": METHOD_VERSION,
        "checked_at": now.isoformat(),
        "status": "disabled",
        "delivery_confirmed": False,
        "orders_enabled": False,
    }
    if read_pnl_config(root) is None:
        return event
    if not in_report_hours(now) or not _FINAL_RETRY <= now.time() < time(15, 59):
        return {**event, "status": "outside_report_window"}
    if _known_market_closed(root, now.date()):
        return {**event, "status": "market_closed"}
    with daily_update_lock(root / "report.lock") as acquired:
        if not acquired:
            return {**event, "status": "already_running"}
        portfolio = get_active_holding_portfolio(repository)
        if portfolio is None or portfolio.status != "active" or not portfolio.positions:
            return {**event, "status": "no_holdings"}
        if portfolio.effective_at > now:
            return {**event, "status": "holding_not_yet_effective"}
        path = root / "delivery-state.json"
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            receipt = {}
        if isinstance(receipt, dict) and receipt.get("accepted_date") == now.date().isoformat():
            return {**event, "status": "already_provider_accepted"}
        identity = (portfolio.id, portfolio.version)

        def allowed(channel="serverchan"):
            try:
                fresh = clock().astimezone(CN)
                current = get_active_holding_portfolio(repository)
                return (
                    channel == "serverchan"
                    and read_pnl_config(root) is not None
                    and fresh.date() == now.date()
                    and _FINAL_RETRY <= fresh.time() < time(15, 59)
                    and not _known_market_closed(root, fresh.date())
                    and current is not None
                    and current.status == "active"
                    and bool(current.positions)
                    and (current.id, current.version) == identity
                    and current.effective_at <= fresh
                )
            except Exception:
                return False

        public = "收盘盈亏计算或数据核验尚未完成，今天暂不能给出可靠收益数字。本条是本次日报的待核验状态，不是盈亏为零，也不是买卖信号。"
        message = NotificationMessage(
            title="A股收盘盈亏待核验",
            body=public,
            group="A股研究室·持仓收盘日报",
            holding_authorization_guard=allowed,
            unauthorized_body="持仓或授权已变化，本条不含收益数据。",
        )
        if not allowed():
            return {**event, "status": "holding_or_authorization_changed"}
        try:
            accepted = notifier(message) is True
        except Exception:
            accepted = False
        if not accepted:
            return {**event, "status": "provider_not_accepted"}
        if not allowed():
            return {**event, "status": "holding_or_authorization_changed"}
        write_private_json(
            path,
            {
                "accepted_date": now.date().isoformat(),
                "accepted_channels": ["serverchan"],
                "report_complete": False,
                "delivery_confirmed": False,
            },
        )
        return {**event, "status": "pending_provider_accepted"}
