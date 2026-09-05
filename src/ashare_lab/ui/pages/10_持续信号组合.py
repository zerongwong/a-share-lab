"""Local continuous-plan page: explicit generation, no automatic delivery."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PAGE_TITLE = "🪻 持续信号组合"
_STATE_KEY = "continuous_signal_local_view"


@dataclass(frozen=True)
class LocalContinuousView:
    digest: Any
    holding_review: Any
    data_root: Path
    holding_identity: tuple[str, int] | None
    plan_date: date | None = None


def _view_state(view: LocalContinuousView) -> dict[str, Any]:
    # Streamlit re-executes page classes on every click. Store plain state so a
    # subsequent run does not reject yesterday's class identity by accident.
    return {
        "digest": view.digest,
        "holding_review": view.holding_review,
        "data_root": view.data_root,
        "holding_identity": view.holding_identity,
        "plan_date": view.plan_date,
    }


def _restore_view(value: object) -> LocalContinuousView | None:
    if isinstance(value, LocalContinuousView):
        return value
    if isinstance(value, Mapping) and set(value) == {
        "digest",
        "holding_review",
        "data_root",
        "holding_identity",
        "plan_date",
    }:
        return LocalContinuousView(**value)
    return None


def _load_local_view(*, decision_date: date) -> LocalContinuousView:
    from ashare_lab.adapters.sqlite_repository import SQLiteRepository
    from ashare_lab.bootstrap import application_data_dir, project_root
    from ashare_lab.services.build_continuous_digest import build_continuous_research_digest
    from ashare_lab.services.holding_ledger import (
        get_active_holding_portfolio,
        holding_knowledge_context,
    )
    from ashare_lab.services.run_active_holding_review import build_evening_holding_review

    root = application_data_dir()
    repository = SQLiteRepository(root / "research.db", project_root() / "migrations")
    known_at = datetime.now(UTC)
    paths = {
        "dataset_root": root / "cache" / "csmar",
        "overlay_root": root / "cache" / "market_overlay",
    }
    digest = build_continuous_research_digest(
        **paths,
        reference_dataset_root=root / "cache" / "csmar_reference",
        decision_date=decision_date,
        repository=repository,
        known_at=known_at,
    )
    portfolio = get_active_holding_portfolio(repository)
    plan = getattr(digest, "continuous_plan", None)
    plan_identity = plan.get("holding_identity") if isinstance(plan, Mapping) else None
    current_plan_identity = (
        (portfolio.id, portfolio.version) if portfolio is not None and portfolio.positions else None
    )
    if (None if plan_identity is None else tuple(plan_identity)) != current_plan_identity:
        raise ValueError("holding membership changed while the plan was being generated")
    context = None if portfolio is None else holding_knowledge_context(portfolio, known_at=known_at)
    review = build_evening_holding_review(
        repository,
        **paths,
        decision_date=digest.common_cutoff,
        reviewed_at=known_at,
        persist=False,
        holding_context=context,
    )
    identity = None if portfolio is None else (portfolio.id, portfolio.version)
    if identity is not None and (review.portfolio_id, review.holding_version) != identity:
        raise ValueError("holding review identity changed")
    return LocalContinuousView(digest, review, root, identity)


def _verify_next_session(cutoff: date) -> date:
    # Network use exists only behind the explicitly labelled calendar button;
    # ordinary page loads and generation never call this resolver.
    from ashare_lab.cli.evening_digest import resolve_next_zero_budget_trading_day

    return resolve_next_zero_budget_trading_day(cutoff)


def _current_holding_identity_matches(view: LocalContinuousView) -> bool:
    from ashare_lab.adapters.sqlite_repository import SQLiteRepository
    from ashare_lab.bootstrap import project_root
    from ashare_lab.services.holding_ledger import get_active_holding_portfolio

    repository = SQLiteRepository(view.data_root / "research.db", project_root() / "migrations")
    portfolio = get_active_holding_portfolio(repository)
    identity = None if portfolio is None else (portfolio.id, portfolio.version)
    return identity == view.holding_identity


def _checked_plan_date(value: object, *, cutoff: date, today: date) -> date:
    if not isinstance(value, date) or isinstance(value, datetime) or value <= cutoff:
        raise ValueError("next session is not verified")
    if value <= today:
        raise ValueError("next-session evidence is already in the past or today")
    return value


def _load_local_chart(view: LocalContinuousView):
    from ashare_lab.adapters.sqlite_repository import SQLiteRepository
    from ashare_lab.bootstrap import project_root
    from ashare_lab.services.build_holding_chart_report import build_holding_chart_report
    from ashare_lab.services.holding_ledger import (
        get_active_holding_portfolio,
        holding_knowledge_context,
    )

    repository = SQLiteRepository(view.data_root / "research.db", project_root() / "migrations")
    portfolio = get_active_holding_portfolio(repository)
    identity = None if portfolio is None else (portfolio.id, portfolio.version)
    if identity != view.holding_identity:
        raise ValueError("holding membership changed; regenerate the report")
    known_at = datetime.now(UTC)
    context = None if portfolio is None else holding_knowledge_context(portfolio, known_at=known_at)
    return build_holding_chart_report(
        repository,
        dataset_root=view.data_root / "cache" / "csmar",
        overlay_root=view.data_root / "cache" / "market_overlay",
        as_of=view.digest.common_cutoff,
        reviewed_at=known_at,
        archive_directory=None,
        holding_context=context,
    )


def _plan(view: LocalContinuousView) -> Mapping[str, Any]:
    value = getattr(view.digest, "continuous_plan", None)
    return value if isinstance(value, Mapping) else {}


def _has_pending_exits(plan: Mapping[str, Any]) -> bool:
    return bool(plan.get("pending_exit_symbols"))


def _research_rows(plan: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for entry in plan.get("entries", ()):
        if not isinstance(entry, Mapping):
            continue
        protection = entry.get("protection_line")
        valid_protection = (
            isinstance(protection, (int, float))
            and not isinstance(protection, bool)
            and math.isfinite(protection)
            and protection > 0
        )
        condition = str(entry.get("entry_label") or "条件待核验")
        line = str(protection) if valid_protection else "待核验"
        rows.append(
            {
                "股票": f"{entry.get('name', '未命名')} ({entry.get('symbol', '待核验')})",
                "仓位": "—（仅研究，不作买入配置）",
                "参考条件 / 保护线": f"{condition}；保护参考 {line}",
            }
        )
    return rows


def _report_text(view: LocalContinuousView) -> str:
    from ashare_lab.services.build_continuous_report import render_continuous_report
    from ashare_lab.services.build_evening_digest import _holding_review_lines

    plan = _plan(view)
    formal = view.plan_date is not None and not _has_pending_exits(plan)
    note = str(plan.get("status_note") or "组合证据未完整，暂不新买。")
    if view.plan_date is None:
        note = "下一交易日未核验：仅作研究观察，不提供可买仓位。 " + note
    if _has_pending_exits(plan):
        note = "卖出尚未确认：仅作补位预案，不作正式新买。 " + note
    return render_continuous_report(
        as_of=view.digest.common_cutoff,
        plan_date=view.plan_date,
        market_summary=str(getattr(view.digest, "cycle_label", "市场证据待核验")),
        holding_lines=_holding_review_lines(view.holding_review, name_bytes=36, reason_bytes=180),
        entries=plan.get("entries", ()) if formal else (),
        cash_weight=plan.get("cash_weight") if formal else None,
        status_note=note,
    )


def render(
    ui: Any | None = None,
    *,
    decision_date: date | None = None,
    _view_loader: Callable[..., LocalContinuousView] = _load_local_view,
    _calendar_resolver: Callable[[date], date] = _verify_next_session,
    _chart_loader: Callable[[LocalContinuousView], Any] = _load_local_chart,
    _identity_checker: Callable[[LocalContinuousView], bool] | None = None,
) -> None:
    if ui is None:
        import streamlit as ui  # type: ignore[no-redef]
    today = decision_date or datetime.now(ZoneInfo("Asia/Shanghai")).date()
    ui.set_page_config(page_title="持续信号组合", page_icon="🪻", layout="wide")
    ui.title(PAGE_TITLE)
    ui.caption("一组组合，持续跟踪｜先处理弱仓，再择机补位｜不设到期卖出")
    ui.info("本页仅在本机研究和看图，不发送微信、不上传图片、不自动改动持仓。")
    if ui.button("生成持续组合", type="primary"):
        try:
            ui.session_state[_STATE_KEY] = _view_state(_view_loader(decision_date=today))
        except Exception:  # noqa: BLE001 - do not disclose provider/ledger exceptions
            ui.session_state.pop(_STATE_KEY, None)
            ui.error("本地研究生成失败，数据或持仓证据待核验；未形成新买计划。")
    view = _restore_view(ui.session_state.get(_STATE_KEY))
    if not isinstance(view, LocalContinuousView):
        ui.caption("点击生成后查看当前持仓结论与这一组研究计划。")
        return
    try:
        current = (_identity_checker or _current_holding_identity_matches)(view)
    except Exception:  # noqa: BLE001 - a failed ledger read cannot preserve buy permission
        current = False
    if not current:
        ui.session_state.pop(_STATE_KEY, None)
        ui.warning("持仓版本已变化或无法核验，请重新生成；不继续沿用旧仓位方案。")
        return
    if view.plan_date is not None and view.plan_date <= today:
        view = replace(view, plan_date=None)
        ui.session_state[_STATE_KEY] = _view_state(view)
    ui.caption("日历核验为单独的免费只读联网请求；失败时保持研究观察，不猜交易日期。")
    if ui.button("核验下一交易日"):
        try:
            verified = _checked_plan_date(
                _calendar_resolver(view.digest.common_cutoff),
                cutoff=view.digest.common_cutoff,
                today=today,
            )
            view = replace(view, plan_date=verified)
            ui.session_state[_STATE_KEY] = _view_state(view)
        except Exception:  # noqa: BLE001 - no raw provider URLs or credentials
            view = replace(view, plan_date=None)
            ui.session_state[_STATE_KEY] = _view_state(view)
            ui.warning("下一交易日尚未核验；继续显示研究观察，不显示可买仓位。")
    try:
        ui.markdown(_report_text(view))
    except (ValueError, TypeError, KeyError):
        ui.warning("条目资格、条件或保护线不完整；暂不形成可买计划，请先核验。")
    plan = _plan(view)
    if view.plan_date is None or _has_pending_exits(plan):
        rows = _research_rows(plan)
        if rows:
            ui.subheader("🩵 单组合研究观察")
            ui.dataframe(rows, hide_index=True, width="stretch")
    if ui.button("生成持仓日 / 周 K线图（仅本机）"):
        try:
            charts = _chart_loader(view)
            rendered = getattr(charts, "rendered", None)
            if rendered is None:
                ui.warning("当前无可用持仓图：可能为空仓或价格证据不足，不会上传或发送。")
            else:
                ui.image(
                    rendered.composite_png,
                    caption="当前登记持仓 · 日线 / 完整周线 · 仅本机",
                    width="stretch",
                )
        except Exception:  # noqa: BLE001 - preserve the existing text and private state
            ui.warning("持仓图暂不可用，或持仓版本已变化；请重新生成核验，未上传或发送。")
    ui.caption("信号是研究结论，不保证收益，也不保证保护线能够成交。")


if __name__ == "__main__":
    render()
