from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import streamlit as st

from ashare_lab.adapters.macos_keychain import (
    save_tushare_token,
    tushare_token_is_configured,
)
from ashare_lab.adapters.market_overlay_store import MarketOverlayStore
from ashare_lab.bootstrap import application_data_dir
from ashare_lab.domain.errors import AShareLabError, DataUnavailableError
from ashare_lab.services.daily_update_lock import daily_update_lock
from ashare_lab.services.run_daily_update import (
    DailyUpdateReport,
    latest_complete_cn_candidate,
    read_csmar_baseline_cutoff,
)
from ashare_lab.services.run_zero_budget_daily_update import run_zero_budget_daily_update

CSMAR_ROOT = application_data_dir() / "cache" / "csmar"
OVERLAY_ROOT = application_data_dir() / "cache" / "market_overlay"
SCHEDULED_SYNC_LOCK = application_data_dir() / "scheduler" / "daily-sync.lock"
_FLASH_KEY = "daily_update_flash"


@dataclass(frozen=True, slots=True)
class _LocalUpdateStatus:
    requested_complete_date: date
    baseline_cutoff: date | None
    automatic_increment_cutoff: date | None
    common_cutoff: date | None
    quarantine_count: int


def _local_status(now: datetime) -> _LocalUpdateStatus:
    candidate = latest_complete_cn_candidate(now)
    baseline: date | None = None
    automatic: date | None = None
    try:
        baseline = read_csmar_baseline_cutoff(CSMAR_ROOT, through_date=candidate)
        chain = MarketOverlayStore(OVERLAY_ROOT).verified_dates_from(
            source_id="zero_budget_eod",
            baseline_cutoff=baseline,
            through_date=candidate,
        )
        automatic = chain[-1] if chain else None
    except AShareLabError:
        pass
    quarantine_root = (
        OVERLAY_ROOT / "source=zero_budget_eod" / "adjust=none" / "quarantine"
    )
    quarantine_count = len(tuple(quarantine_root.glob("run=*")))
    return _LocalUpdateStatus(
        requested_complete_date=candidate,
        baseline_cutoff=baseline,
        automatic_increment_cutoff=automatic,
        common_cutoff=automatic or baseline,
        quarantine_count=quarantine_count,
    )


def _save_key(value: str) -> None:
    save_tushare_token(value)
    st.session_state.pop("tushare_token_input", None)
    st.session_state[_FLASH_KEY] = (
        "Tushare Token已安全保存到macOS钥匙串；页面不会回显。"
    )
    st.rerun()


def _run_update() -> None:
    with daily_update_lock(SCHEDULED_SYNC_LOCK) as acquired:
        if not acquired:
            raise DataUnavailableError("收盘数据更新正在另一个进程中运行，请稍后再试。")
        report = run_zero_budget_daily_update(
            csmar_root=CSMAR_ROOT,
            overlay_root=OVERLAY_ROOT,
        )
    st.session_state["latest_daily_update_report"] = report


def _render_report(report: DailyUpdateReport) -> None:
    st.subheader("本次更新结果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("历史基线截止", report.historical_baseline_cutoff.isoformat())
    c2.metric(
        "自动增量截止",
        report.automatic_increment_cutoff.isoformat()
        if report.automatic_increment_cutoff is not None
        else "尚无",
    )
    c3.metric("共同截止", report.common_cutoff.isoformat())
    c4.metric("本次隔离失败", len(report.quarantined_failures))
    st.caption(
        f"交易日历确认的最近完整收盘：{report.latest_complete_session.isoformat()}｜"
        f"新增{len(report.updated_sessions)}日｜未变化{len(report.unchanged_sessions)}日｜"
        f"单位合同：{report.unit_contract_version}"
    )
    if report.current_through_latest_complete_session:
        st.success("本地共同截止已经追平最近完整交易日，可供研究流程读取。")
    else:
        st.error("本地共同截止尚未追平最近完整交易日；失败日之后的数据没有被登记。")
    if report.provider_contract_changed:
        st.error(
            "三源字段、单位或交叉核验合同可能发生变化。系统已停止并隔离该批数据，"
            "没有猜测单位，也没有用AKShare替换Tushare数据。"
        )
    if report.quarantined_failures:
        with st.expander("查看隔离失败", expanded=True):
            for item in report.quarantined_failures:
                st.write(f"{item.trade_date.isoformat()}：{item.reason}")
                if item.path:
                    st.caption(f"本地隔离目录：{item.path}")


def render() -> None:
    st.title("自动补齐收盘数据")
    st.caption(
        "CSMAR只作为不可改写的历史基线；缺失交易日写入零预算三源独立overlay。"
        "系统不会连接券商、不会读取持仓，也不会自动下单。"
    )
    flash = st.session_state.pop(_FLASH_KEY, None)
    if flash:
        st.success(str(flash))

    st.info(
        "盘中不会把今天的累计行情当成完整日线：上海时间15:30前只补到上一日期。"
        "15:30后当日只进入收盘候选，仍需通过BaoStock交易日历、Tushare覆盖和"
        "AKShare独立核验；"
        "不完整数据会被隔离，18:30复核，20:00晚报前再预检。"
    )
    st.warning(
        "当前免费链只覆盖沪深A股，不含北交所。北交所股票不会被伪造或静默补齐，"
        "在混合研究样本中会明确排除。"
    )
    st.caption(
        "本页提供一键追平和状态检查。项目另附独立LaunchAgent安装脚本；"
        "只有使用者主动运行安装脚本后，才会在每日15:30首次同步、18:30质量复核、"
        "20:00晚报前预检。"
        "该任务不依赖本网页，不会常驻、连接券商或自动下单。"
    )

    try:
        configured = tushare_token_is_configured()
        key_status = "🟢 已安全配置" if configured else "🟡 尚未配置"
    except AShareLabError as exc:
        key_status = "🔴 macOS钥匙串暂时无法读取"
        st.error(str(exc))

    with st.container(border=True):
        st.subheader("Tushare Token · 仅存macOS钥匙串")
        st.write("状态：" + key_status)
        st.caption(
            "免费注册后在Tushare个人中心复制Token。请直接粘贴到本机页面，"
            "不要发到聊天、截图或GitHub。"
        )
        token = st.text_input(
            "Tushare Token",
            type="password",
            key="tushare_token_input",
            help="Token不会写入项目、数据库、日志、命令行参数或GitHub。",
        )
        if st.button("保存Token到钥匙串", type="secondary", width="stretch"):
            try:
                _save_key(token)
            except (AShareLabError, ValueError) as exc:
                st.error(str(exc))

    status = _local_status(datetime.now(UTC))
    st.subheader("本地数据状态")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "历史基线截止",
        status.baseline_cutoff.isoformat() if status.baseline_cutoff else "未找到",
    )
    c2.metric(
        "自动增量截止",
        status.automatic_increment_cutoff.isoformat()
        if status.automatic_increment_cutoff
        else "尚无",
    )
    c3.metric(
        "共同截止",
        status.common_cutoff.isoformat() if status.common_cutoff else "不可用",
    )
    c4.metric("累计隔离失败", status.quarantine_count)
    st.caption(
        f"当前候选完整日期：{status.requested_complete_date.isoformat()}（仍须交易日历确认）"
    )

    if st.button("立即自动补齐缺失收盘数据", type="primary", width="stretch"):
        try:
            with st.spinner(
                "正在用BaoStock核对交易日历、Tushare取得日线、AKShare交叉核验…"
            ):
                _run_update()
        except (AShareLabError, ValueError) as exc:
            st.error(f"自动数据更新未完成：{exc}")

    report = st.session_state.get("latest_daily_update_report")
    if isinstance(report, DailyUpdateReport):
        _render_report(report)


render()
