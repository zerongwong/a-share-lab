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
        _existing_page("02"),
        title="中线四股组合",
        icon="🧺",
        url_path="four-stock-portfolio",
        default=True,
    ),
]

current_page = st.navigation(pages, position="sidebar", expanded=True)

with st.sidebar:
    st.caption("本地个人研究版 · 不连接券商 · 不自动下单")
    with st.expander("当前主模块", expanded=False):
        st.markdown(
            "**中线四股组合：历史研究可用。** 已接入本机CSMAR全市场日线，"
            "可从共同截止日的合格股票中生成四股研究组合。默认研究3个月；"
            "1周只观察入场节奏，1–3个月是核心持有期，6–12个月用于长期验证。"
        )
    with st.expander("证据与研究留档", expanded=False):
        st.write(
            "授权状态与历史研究档案仍保留在本地系统中，但不作为独立入口。"
            "结果页面会继续显示证据出处、截止时间和档案编号。"
        )

st.info(
    "当前状态：CSMAR全市场日线、核心指数风险确认和资产负债表当前快照已接入。"
    "财务快照只有通过决策日/价格截止时点门后才作为窄因子使用；结果仍基于未复权日线，"
    "尚缺利润表、现金流、普通财报公告日、公司新闻和严格walk-forward样本外验证。"
)

current_page.run()
