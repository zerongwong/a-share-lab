"""Honest status screen for the not-yet-real-time position monitor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PAGE_TITLE = "持仓实时监控"
STATUS = "实时监控尚未就绪"
AVAILABLE_NOW = (
    "当前可用的是单只股票手动检查：只有点击分析按钮时才读取日线数据，"
    "并根据用户输入的持仓成本生成研究计划。"
)
NOT_READY = (
    "没有后台常驻任务，不会自动刷新价格。",
    "没有主动通知、到价提醒或券商连接。",
    "尚未提供多只持仓的统一清单与组合级风险监控。",
    "没有授权实时行情时，不会把延迟日线称作实时数据。",
)


def _manual_check_path() -> Path:
    matches = tuple(Path(__file__).resolve().parent.glob("01_*.py"))
    if len(matches) != 1:
        raise RuntimeError("无法定位单股手动检查页面")
    return Path("pages") / matches[0].name


def render(ui: Any | None = None) -> None:
    """Render with an injectable UI so status claims remain regression-tested."""

    if ui is None:
        import streamlit as ui  # type: ignore[no-redef]

    ui.title(PAGE_TITLE)
    ui.warning(STATUS)
    ui.info(AVAILABLE_NOW)

    ui.subheader("尚未接通的能力")
    ui.markdown("\n".join(f"- {item}" for item in NOT_READY))

    ui.subheader("现在可以这样用")
    ui.markdown(
        "1. 打开下面的手动检查。\n"
        "2. 输入一只股票代码、分析截止日和持仓成本。\n"
        "3. 查看不同持有期的计划区、减仓区和结构失效位。\n"
        "4. 需要更新时再次手动运行；页面不会在后台替你盯盘。"
    )
    ui.page_link(
        _manual_check_path(),
        label="打开单股手动检查",
        icon="📋",
        width="stretch",
    )
    ui.caption("仅用于个人研究；不自动下单，不承诺收益，也不保证止损能够成交。")


if __name__ == "__main__":
    render()
