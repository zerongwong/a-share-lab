"""Render one continuous-holding plan without granting execution permission.

This pure renderer never loads a holding ledger, prices, credentials, or a
network client. Its caller supplies already-authorized holding lines and must
set ``entry_qualified`` to the actual boolean ``True`` only after every entry
gate passes. Charts remain in the existing authorized R2 attachment path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from ashare_lab.ports.market_data import normalize_symbol

_FOOTNOTE = "仅信号退出，不设到期卖出｜不自动下单，不保证收益。"


def render_continuous_report(
    *,
    as_of: date,
    plan_date: date | None,
    market_summary: str,
    holding_lines: Sequence[str],
    entries: Sequence[Mapping[str, Any]],
    cash_weight: float | None,
    status_note: str,
    chart_markdown: str | None = None,
    max_bytes: int = 4096,
) -> str:
    """Render a single concise plan with total-account, not sleeve, weights.

    Qualified entries require ``symbol``, ``name``, ``account_weight``, a
    nonempty ``entry_label`` and a positive numeric ``protection_line``.
    Unqualified entries never expose a stock name, price, or allocation as a
    new-buy recommendation. Cash is the planned remaining account allocation;
    ``None`` means unknown, never an inferred empty portfolio.

    Critical holding lines, status, entry conditions and numbers are never
    shortened. If needed, only the entire optional market narrative is replaced
    with an explicit omission label; if still too large, rendering fails closed.
    ``chart_markdown`` must remain ``None``: attach charts after rendering via
    the existing independently authorized, bounded R2 delivery mechanism.
    """

    _require_date(as_of, "as_of")
    if plan_date is not None:
        _require_date(plan_date, "plan_date")
        if plan_date <= as_of:
            raise ValueError("plan_date must follow the verified data cutoff")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if chart_markdown is not None:
        raise ValueError("Attach charts through the authorized R2 delivery path, not Markdown")
    if isinstance(holding_lines, (str, bytes)) or not isinstance(holding_lines, Sequence):
        raise TypeError("holding_lines must be a sequence of authorized text lines")
    if isinstance(entries, (str, bytes, Mapping)) or not isinstance(entries, Sequence):
        raise TypeError("entries must be a sequence of mappings")

    market = _text(market_summary, "market_summary", allow_empty=True)
    status = _text(status_note, "status_note", allow_empty=True)
    holdings = tuple(_text(line, "holding_line") for line in holding_lines)
    qualified: list[str] = []
    symbols: set[str] = set()
    total_new_weight = Decimal(0)
    rejected_count = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("each entry must be a mapping")
        allowed = entry.get("entry_qualified", False)
        if not isinstance(allowed, bool):
            raise TypeError("entry_qualified must be a boolean")
        if not allowed:
            rejected_count += 1
            continue
        if plan_date is None:
            raise ValueError("qualified entries require a verified plan_date")
        symbol = normalize_symbol(_required_string(entry, "symbol"))
        if symbol in symbols:
            raise ValueError("qualified entries must have unique stock symbols")
        symbols.add(symbol)
        name = _text(_required_string(entry, "name"), "name")
        condition = _text(_required_string(entry, "entry_label"), "entry_label")
        weight = _fraction(entry.get("account_weight"), "account_weight", positive=True)
        protection = _number(entry.get("protection_line"), "protection_line")
        if protection <= 0:
            raise ValueError("protection_line must be positive")
        total_new_weight += weight
        qualified.append(
            f"- {name}({symbol})｜总资金{_percent(weight)}｜{condition}｜保护{_decimal(protection)}"
        )
    if len(qualified) > 5:
        raise ValueError("a continuous portfolio supports at most five qualified new entries")
    cash = None if cash_weight is None else _fraction(cash_weight, "cash_weight")
    if total_new_weight > 1 or (cash is not None and total_new_weight + cash > 1 + Decimal("1e-9")):
        raise ValueError("new entries and planned cash cannot exceed the total account")

    title = "持仓核验" if plan_date is None else f"{plan_date.isoformat()} 次日交易计划"
    prefix = [f"# 🪻 {title}", f"数据截至 {as_of.isoformat()}", "", "## 🩷 持仓优先"]
    prefix.extend(line if line.startswith("- ") else f"- {line}" for line in holdings)
    if not holdings:
        prefix.append("- 持仓信息未提供｜待核验")
    if status:
        prefix.extend(("", f"📌 {status}"))
    suffix = ["", "## 🩵 条件新买"]
    if qualified:
        suffix.extend(qualified)
        if rejected_count:
            suffix.append("其余未过门：暂不新买。")
    else:
        suffix.append("⏸ 暂不新买｜等待合格信号")
    suffix.extend(
        (
            "现金：未核定（不代表空仓）" if cash is None else f"计划现金：总资金{_percent(cash)}",
            "",
            _FOOTNOTE,
        )
    )
    market_options = [f"🟦 市场：{market}" if market else "🟦 市场：待核验"]
    if market:
        market_options.append("🟦 市场摘要略；优先保留风险与价格条件")
    for market_line in market_options:
        body = "\n".join((*prefix, "", market_line, *suffix))
        if len(body.encode("utf-8")) <= max_bytes:
            return body
    raise ValueError("continuous report exceeds byte budget without dropping critical information")


def _require_date(value: object, field: str) -> None:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError(f"{field} must be a calendar date")


def _required_string(entry: Mapping[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"qualified entry requires nonempty {key}")
    return value


def _text(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    normalized = " ".join(value.split())
    if not normalized and not allow_empty:
        raise ValueError(f"{field} cannot be empty")
    # Caller text remains visible, but cannot become an embedded URL, HTML,
    # image, or nested Markdown instruction in a notification renderer.
    for character in ("\\", "`", "*", "_", "[", "]", "<", ">"):
        normalized = normalized.replace(character, "\\" + character)
    return normalized


def _number(value: object, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise TypeError(f"{field} must be a finite number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be a finite number")
    return result


def _fraction(value: object, field: str, *, positive: bool = False) -> Decimal:
    result = _number(value, field)
    if result < 0 or result > 1 or (positive and result == 0):
        raise ValueError(f"{field} must be an account fraction between zero and one")
    return result


def _decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _percent(value: Decimal) -> str:
    return f"{_decimal(value * 100)}%"


__all__ = ["render_continuous_report"]
