"""Simple, fail-closed status page for the closing-strength shortlist.

实时授权数据、成交仿真与样本外概率校准未完成前，不调用筛选服务，
也不展示股票代码或模拟概率。
"""

from __future__ import annotations

from typing import Any

PAGE_TITLE = "尾盘五强"
DISABLED_MESSAGE = "当前未启用"
DISABLED_DETAIL = (
    "授权实时行情、成交仿真和样本外概率校准尚未完成，因此现在不会生成候选股票，也不会展示推测概率。"
)
AT_A_GLANCE = (
    "14:40预检 → 14:45扫描 → 14:50输出0–5只。启用后，每只将展示排名、综合分、"
    "连板概率区间和主要风险；没有合格标的时显示“今日无候选”。"
)

PREREQUISITES = (
    "取得可用于本地研究的授权实时行情；候选成交性判断需要逐笔/L2数据。",
    "交易所时钟、交易日历、板块、ST/退市、停复牌和当日涨跌停价均已核验。",
    "14:49:55冻结的快照完整且数据延迟不超过3秒；任一关键字段缺失即停止输出。",
    "按主板、创业板、科创板和北交所制度分层完成历史逐笔成交仿真。",
    "完成无前视的滚动样本外回测、概率校准和影子运行，并达到预设验收门槛。",
    "公告、新闻和行情具备合法的数据授权，且每条证据保留来源与发布时间。",
)

SAFETY_BOUNDARIES = (
    "每天允许输出0只，最多5只；没有合格标的时必须明确显示“今日无候选”。",
    "5%是该高风险试验模块的资金仓上限，不是止损线，也不是最大亏损保证。",
    "每只股票的资金仓上限是1%；不足5只时剩余资金保留现金，不向前几名追加。",
    "同一题材最多2只，避免把五个名额集中到同一个交易逻辑。",
    "融资比例固定为0，不借钱参与涨停或连板研究。",
    "只生成研究观察名单，不自动下单，也不连接券商执行交易。",
    "A股实行T+1；预设退出研究窗口为次一交易日09:35–09:45。",
    "跌停无买盘时止损不能成交；预设退出价不是成交保证。",
    "封住涨停不代表可以买到；没有可靠成交概率或盘口数据时只能显示观察。",
)

TRACEABLE_SCORE = (
    "市场环境10分、板块协同15分、个股量价结构15分。",
    "封板行为20分、催化证据10分、真实可成交性15分。",
    "次日净收益分布15分，再扣除最高30分尾部风险惩罚。",
    "每个组件都必须关联证据标识；输入不全时该股票直接失去资格。",
)

RESEARCH_TIMELINE = (
    "14:40开始预检，14:44完成数据、授权、交易日历与时钟检查。",
    "14:45开始扫描，14:49:55冻结特征，14:50:05输出研究观察。",
    "未成交委托应在14:56:45前人工撤回；14:57进入收盘集合竞价后不可撤单。",
    "次一交易日09:35–09:45按预设退出规则研究，并记录无法成交的真实情况。",
)


def render(ui: Any | None = None) -> None:
    """Render the disabled-by-default status page.

    ``ui`` is injectable for deterministic tests. The normal app calls
    ``render()`` without arguments and Streamlit is imported lazily.
    """

    interactive = ui is None
    if ui is None:
        import streamlit as ui  # type: ignore[no-redef]

    ui.title(PAGE_TITLE)
    ui.warning(DISABLED_MESSAGE)
    ui.markdown(f"**{AT_A_GLANCE}**")
    ui.info(DISABLED_DETAIL)

    if interactive:
        _render_infoway_connection(ui)

    with ui.expander("启用前必须完成", expanded=False):
        ui.markdown("\n".join(f"- {item}" for item in PREREQUISITES))

    with ui.expander("资金与交易安全边界", expanded=False):
        ui.markdown("\n".join(f"- {item}" for item in SAFETY_BOUNDARIES))

    with ui.expander("评分依据与完整时间线", expanded=False):
        ui.markdown("**固定可追溯评分**")
        ui.markdown("\n".join(f"- {item}" for item in TRACEABLE_SCORE))
        ui.markdown("**预设研究时间线**")
        ui.markdown("\n".join(f"- {item}" for item in RESEARCH_TIMELINE))

    with ui.expander("当前不会做什么", expanded=False):
        ui.markdown(
            "- 不展示股票候选、涨停概率或预期收益。\n"
            "- 不根据演示数据生成买卖信号。\n"
            "- 不创建系统自动任务或桌面定时任务。\n"
            "- 不承诺涨停、连板、止损成交或本金安全。"
        )


def _render_infoway_connection(ui: Any) -> None:
    """Render local-only secret setup without changing the safety status."""

    from ashare_lab.adapters.infoway_realtime import InfowayRealtimeMarketData
    from ashare_lab.adapters.macos_keychain import (
        infoway_key_is_configured,
        load_infoway_api_key,
        save_infoway_api_key,
    )

    with ui.expander("Infoway实时数据连接", expanded=not infoway_key_is_configured()):
        if infoway_key_is_configured():
            ui.success("API Key 已保存在这台Mac的系统钥匙串中。")
        else:
            ui.warning("尚未配置API Key；请从Infoway后台复制后粘贴到下方。")
        ui.caption("密钥不会写入项目、数据库、日志或聊天记录。")
        secret = ui.text_input("Infoway API Key", type="password", key="infoway-api-key")
        save_clicked = ui.button("保存到Mac系统钥匙串", key="save-infoway-key")
        if save_clicked:
            if not secret.strip():
                ui.error("请先粘贴API Key。")
            else:
                try:
                    save_infoway_api_key(secret)
                    ui.success("保存成功。")
                    ui.rerun()
                except Exception as exc:
                    ui.error(f"保存失败：{exc}")

        if ui.button("测试A股接口", key="test-infoway-key"):
            try:
                api_key = load_infoway_api_key()
                if not api_key:
                    raise RuntimeError("尚未保存API Key")
                with InfowayRealtimeMarketData(api_key) as client:
                    symbols = client.fetch_symbol_list()
                ui.success(f"连接成功：接口返回 {len(symbols)} 个 STOCK_CN 标的。")
                ui.caption("是否包含北交所将以实际返回代码为准，当前不会自行补代码。")
            except Exception as exc:
                ui.error(f"接口测试失败：{exc}")


if __name__ == "__main__":
    render()
