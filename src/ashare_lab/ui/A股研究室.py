"""Single-purpose entrypoint for the medium-horizon A-share research app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

APP_DIRECTORY = Path(__file__).resolve().parent
PAGE_DIRECTORY = APP_DIRECTORY / "pages"

st.set_page_config(page_title="A股研究室", page_icon="📊", layout="wide")


def _existing_page(prefix: str) -> Path:
    """Resolve a retained implementation page without exposing its old UI name."""

    matches = tuple(PAGE_DIRECTORY.glob(f"{prefix}_*.py"))
    if len(matches) != 1:
        raise RuntimeError(f"无法定位模块页面：{prefix}")
    return matches[0]


# Streamlit's explicit router suppresses automatic discovery of every file under
# pages/.  The closing-strength and position-monitor implementations stay in the
# codebase for a future separate project, but are intentionally not registered.
pages = [
    st.Page(
        _existing_page("08"),
        title="中期主升组合",
        icon="📈",
        url_path="midterm-maintrend-portfolio",
        default=True,
    ),
    st.Page(
        _existing_page("07"),
        title="通知设置",
        icon="🔔",
        url_path="notification-settings",
    ),
    st.Page(
        _existing_page("09"),
        title="自动数据更新",
        icon="🔄",
        url_path="automatic-daily-update",
    ),
]

current_page = st.navigation(pages, position="sidebar", expanded=True)

with st.sidebar:
    st.caption("本地个人研究版 · 不连接券商 · 不自动下单")
    with st.expander("当前主模块", expanded=False):
        st.markdown(
            "**中期主升组合：研究版。** 先筛主升趋势与明确介入点，再自动比较"
            "3、4、5只组合并计算风险约束权重。默认研究3个月；"
            "1周只观察入场节奏，1–3个月是核心持有期，6–12个月用于长期验证。"
        )
    with st.expander("证据与研究留档", expanded=False):
        st.write(
            "授权状态与历史研究档案仍保留在本地系统中，但不作为独立入口。"
            "结果页面会继续显示证据出处、截止时间和档案编号。"
        )

st.info(
    "当前状态：主升趋势、量价介入门和3–5股下行风险优化已进入主流程。"
    "财务与公告仍有证据缺口时会醒目标注；结果仍基于未复权日线，且尚未完成严格"
    "walk-forward样本外验证。"
)

current_page.run()
