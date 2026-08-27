from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import streamlit as st

from ashare_lab.adapters.macos_keychain import (
    infoway_key_is_configured,
    save_infoway_api_key,
)
from ashare_lab.adapters.market_overlay_store import MarketOverlayStore
from ashare_lab.bootstrap import application_data_dir
from ashare_lab.domain.errors import AShareLabError
from ashare_lab.services.run_daily_update import (
    DailyUpdateReport,
    latest_complete_cn_candidate,
    read_csmar_baseline_cutoff,
    run_daily_update,
)

CSMAR_ROOT = application_data_dir() / "cache" / "csmar"
OVERLAY_ROOT = application_data_dir() / "cache" / "market_overlay"
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
            source_id="infoway",
            baseline_cutoff=baseline,
            through_date=candidate,
        )
        automatic = chain[-1] if chain else None
    except AShareLabError:
        pass
    quarantine_root = OVERLAY_ROOT / "source=infoway" / "adjust=none" / "quarantine"
    quarantine_count = len(tuple(quarantine_root.glob("run=*")))
    return _LocalUpdateStatus(
        requested_complete_date=candidate,
        baseline_cutoff=baseline,
        automatic_increment_cutoff=automatic,
        common_cutoff=automatic or baseline,
        quarantine_count=quarantine_count,
    )


def _save_key(value: str) -> None:
    save_infoway_api_key(value)
    st.session_state.pop("infoway_api_key_input", None)
    st.session_state[_FLASH_KEY] = (
        "新的Infoway API密钥已安全保存到macOS钥匙串；页面不会回显该密钥。"
    )
    st.rerun()


def _run_update() -> None:
    report = run_daily_update(
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
            "供应商字段契约变化，请更新适配器。系统已停止并隔离该批数据，"
            "没有自动尝试vm、替换vw或猜测成交量/成交额倍数。"
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
        "CSMAR只作为不可改写的历史基线；缺失交易日写入Infoway独立overlay。"
        "系统不会连接券商、不会读取持仓，也不会自动下单。"
    )
    flash = st.session_state.pop(_FLASH_KEY, None)
    if flash:
        st.success(str(flash))

    st.info(
        "盘中不会把今天的累计行情当成完整日线：上海时间16:15前只补到上一日期，"
        "再由Infoway中国交易日历排除周末和法定休市日。"
    )
    st.warning(
        "Infoway当前股票清单只覆盖沪深A股，不含北交所。北交所股票不会被伪造或静默补齐，"
        "在混合研究样本中会明确排除。"
    )
    st.caption(
        "本页提供一键追平和状态检查；当前不会安装LaunchAgent或后台常驻任务。"
        "命令行入口可供后续经过确认的定时调度使用。"
    )

    try:
        configured = infoway_key_is_configured()
        key_status = "🟢 已安全配置" if configured else "🟡 尚未配置"
    except AShareLabError as exc:
        key_status = "🔴 macOS钥匙串暂时无法读取"
        st.error(str(exc))

    with st.container(border=True):
        st.subheader("Infoway密钥 · 仅存macOS钥匙串")
        st.write("状态：" + key_status)
        st.error(
            "曾经粘贴到聊天、截图或其他公开位置的旧密钥应先在Infoway后台轮换，"
            "这里只保存轮换后的新密钥。"
        )
        api_key = st.text_input(
            "新的Infoway API密钥",
            type="password",
            key="infoway_api_key_input",
            help="密钥不会写入项目、数据库、日志、命令行参数或GitHub。",
        )
        if st.button("保存新密钥到钥匙串", type="secondary", width="stretch"):
            try:
                _save_key(api_key)
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
            with st.spinner("正在核对交易日历、分批取得沪深收盘数据并验证核心指数…"):
                _run_update()
        except (AShareLabError, ValueError) as exc:
            st.error(f"自动数据更新未完成：{exc}")

    report = st.session_state.get("latest_daily_update_report")
    if isinstance(report, DailyUpdateReport):
        _render_report(report)


render()
