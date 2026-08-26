from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from ashare_lab.bootstrap import build_market_provider, build_repository
from ashare_lab.domain.errors import AShareLabError
from ashare_lab.services.analyze_stock import analyze_stock, archive_stock_analysis
from ashare_lab.ui.charts import price_chart

st.set_page_config(page_title="个股四周期", page_icon="🔎", layout="wide")
st.title("个股四周期研究")
st.caption(
    "艾德华兹–麦吉负责趋势、支撑阻力、收盘与量价确认；利弗摩尔负责关键点、等待和只向盈利仓加码。"
)

with st.form("stock-analysis-form"):
    col1, col2, col3, col4, col5 = st.columns([1.1, 1.1, 1.2, 0.8, 1])
    with col1:
        symbol = st.text_input("A股代码", value="600150", help="输入6位代码，不需要 .SS/.SZ")
    with col2:
        as_of = st.date_input("分析截止日", value=date.today(), max_value=date.today())
    with col3:
        source_name = st.selectbox(
            "日线来源",
            options=("yahoo", "akshare"),
            format_func=lambda value: (
                "Yahoo价格备用（当前可用）" if value == "yahoo" else "东财/AKShare（国内优先）"
            ),
        )
    with col4:
        has_position = st.checkbox("我已经持有", value=False)
    with col5:
        cost_price = st.number_input(
            "持仓成本（元）",
            min_value=0.0,
            value=0.0,
            step=0.01,
            disabled=not has_position,
        )
    submitted = st.form_submit_button("读取数据并分析", type="primary", width="stretch")

if submitted:
    try:
        with st.status("正在读取A股日线并计算结构…", expanded=True) as status:
            provider = build_market_provider(source_name)
            st.write("校验截止日，防止使用未来K线")
            result = analyze_stock(
                provider,
                symbol,
                as_of,
                cost_price=cost_price if has_position and cost_price > 0 else None,
            )
            st.write("计算四周期关键价位和历史情景")
            run_id = archive_stock_analysis(result, build_repository())
            result["run_id"] = run_id
            st.session_state["latest_stock_result"] = result
            status.update(label="分析完成并已留档", state="complete", expanded=False)
    except (AShareLabError, ValueError, ImportError) as exc:
        st.error(f"本次没有生成结论：{exc}")
    except Exception as exc:  # Streamlit must remain usable when a public endpoint changes.
        st.error("数据接口暂时不可用，程序没有用旧数据冒充最新结果。")
        st.exception(exc)

result = st.session_state.get("latest_stock_result")
if result:
    st.success(
        f"{result['symbol']}｜数据截止 {result['data_cutoff']}｜来源 {result['source']}｜"
        f"档案 {result['run_id'][:8]}"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新收盘", f"¥{result['latest_price']:.2f}")
    c2.metric("趋势状态", result["trend"])
    volatility = result["metrics"]["volatility"]
    drawdown = result["metrics"]["max_drawdown"]
    c3.metric("近一年年化波动", "—" if pd.isna(volatility) else f"{volatility:.1%}")
    c4.metric("近一年历史最大回撤", "—" if pd.isna(drawdown) else f"{drawdown:.1%}")

    st.plotly_chart(
        price_chart(result["frame"], title=f"{result['symbol']} · 最近250个交易日"),
        width="stretch",
    )

    st.subheader("关键价位与持股动作")
    rows = []
    for level in result["levels"]:
        probabilities = {item["label"]: item["probability_mid"] for item in level["scenarios"]}
        rows.append(
            {
                "周期": level["horizon"],
                "计划持有": f"约{level['sessions']}个交易日",
                "回踩计划区": f"{level['pullback_entry_low']:.2f}–{level['pullback_entry_high']:.2f}",
                "回踩转强触发": f"{level['entry_trigger']:.2f}",
                "突破确认线": f"{level['breakout_trigger']:.2f}",
                "第一减仓区": f"{level['reduce_low']:.2f}–{level['reduce_high']:.2f}",
                "第二减仓区": (
                    f"{level['second_reduce_low']:.2f}–{level['second_reduce_high']:.2f}"
                ),
                "结构失效位": f"{level['invalidation']:.2f}",
                "计划风险收益比": (
                    "—"
                    if level["reward_risk_ratio"] is None
                    else f"{level['reward_risk_ratio']:.2f}"
                ),
                "空仓动作": level["action_for_empty"],
                "持仓动作": level["action_for_holder"],
                "历史上涨情景频率": (
                    f"{probabilities['up']:.0%}" if "up" in probabilities else "样本不足"
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("买点、卖点怎样确认"):
        first_level = result["levels"][0]
        st.markdown(
            "- **回踩买点不是碰到价格就买：**进入计划区后，还要观察收盘重新转强。\n"
            "- **突破买点使用收盘确认：**盘中瞬间越线不算，成交量还要达到规则阈值。\n"
            "- **卖点分两类：**到达阻力/测量目标分批减仓；跌破结构失效位执行风控。\n"
            "- **时间止损：**到计划持有期仍未按预期发展，应退出或重新评估。\n"
            f"- **成交限制：**{first_level['stop_execution_rule']}。"
        )
        st.caption(first_level["level_method"])

    with st.expander("查看三情景历史分布（不是未来真实概率）"):
        scenario_rows = []
        label_names = {"up": "上涨", "sideways": "震荡", "down": "下跌"}
        for level in result["levels"]:
            for item in level["scenarios"]:
                scenario_rows.append(
                    {
                        "周期": level["horizon"],
                        "情景": label_names[item["label"]],
                        "历史频率": item["probability_mid"],
                        "95%区间下限": item["probability_low"],
                        "95%区间上限": item["probability_high"],
                        "样本数": item["sample_n"],
                        "情景收益P10": item["return_p10"],
                        "情景收益中位": item["return_p50"],
                        "情景收益P90": item["return_p90"],
                    }
                )
        st.dataframe(
            pd.DataFrame(scenario_rows),
            hide_index=True,
            width="stretch",
            column_config={
                name: st.column_config.NumberColumn(format="%.1f%%")
                for name in (
                    "历史频率",
                    "95%区间下限",
                    "95%区间上限",
                    "情景收益P10",
                    "情景收益中位",
                    "情景收益P90",
                )
            },
        )

    st.info(
        f"利弗摩尔关键点：试仓线 {result['livermore']['pivotal_buy_above']:.2f}；"
        f"盈利仓加仓线 {result['livermore']['add_only_above']:.2f}；"
        f"失效参考 {result['livermore']['invalidation_below']:.2f}。"
    )
    for warning in result["warnings"]:
        st.warning(warning)

if not result:
    st.info(
        "当前网络访问东财/新浪会出现TLS中断，所以默认使用已明确标注的Yahoo日线价格备用源。"
        "它不参与国内新闻、公告或基本面分析；国内源恢复后可在上方切换。"
    )
