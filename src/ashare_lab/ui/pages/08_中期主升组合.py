from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from ashare_lab.bootstrap import application_data_dir
from ashare_lab.services.build_midterm_portfolio import (
    MidtermPortfolioResult,
    MidtermPortfolioStatus,
    build_midterm_portfolio,
)
from ashare_lab.services.load_hybrid_universe import load_hybrid_universe

HOLDING_PERIOD_OPTIONS = (1, 4, 13, 26, 52)
HOLDING_PERIOD_LABELS = {
    1: "1周（最短观察）",
    4: "1个月（短中期）",
    13: "3个月（默认）",
    26: "6个月（延长复核）",
    52: "1年（长期复核）",
}
STATUS_LABELS = {
    MidtermPortfolioStatus.DATA_NOT_READY: "数据未就绪",
    MidtermPortfolioStatus.VALIDATION_NOT_READY: "验证未就绪",
    MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO: "本轮没有合格组合",
    MidtermPortfolioStatus.RESEARCH_ONLY: "研究组合已生成",
}
PATTERN_LABELS = {
    "volume_confirmed_breakout": "近期放量突破",
    "healthy_breakout_pullback": "突破后健康回踩",
    "breakout_line_reclaim": "回踩后重新站回突破线",
}


def _period_label(weeks: int) -> str:
    return HOLDING_PERIOD_LABELS[weeks]


st.set_page_config(page_title="中期主升组合", page_icon="📈", layout="wide")
st.title("中期主升组合")
st.caption(
    "先用主升趋势与明确介入点筛全市场，再自动比较3、4、5只组合和权重。"
    "4只为常态；只有3只合格时增加现金，第5只确有分散价值时才加入。"
)

st.info(
    "价格和成交额负责选股与介入；资产负债表和正式公告只负责否决风险，不用利好新闻抬分。"
    "组合必须同时通过下行波动、60日滚动回撤、5日尾部损失、下跌相关性和单股风险贡献预算。"
)
st.warning(
    "当前版本的收益下界仍是历史代理，并非严格walk-forward样本外结果。"
    "若财务或官方公告证据未接齐，入选股票会明确标成待复核，不能据此直接交易。"
)

CSMAR_ROOT = application_data_dir() / "cache" / "csmar"
REFERENCE_ROOT = application_data_dir() / "cache" / "csmar_reference"
OVERLAY_ROOT = application_data_dir() / "cache" / "market_overlay"

with st.form("midterm-maintrend-form"):
    mode_label = st.radio(
        "研究模式",
        ("当前研究", "历史回放"),
        horizontal=True,
        help="历史回放不会使用今天才取得的财务快照。",
    )
    mode = "live" if mode_label == "当前研究" else "historical"
    as_of = st.date_input(
        "决策日",
        value=date.today(),
        max_value=date.today(),
        help="程序会另行核验真正完整的共同数据截止日。",
    )
    holding_weeks = st.selectbox(
        "计划持有与研究周期",
        HOLDING_PERIOD_OPTIONS,
        index=2,
        format_func=_period_label,
    )
    submitted = st.form_submit_button(
        "运行全市场主升组合研究",
        type="primary",
        width="stretch",
    )

if submitted:
    try:
        if not (REFERENCE_ROOT / "csmar_reference.duckdb").is_file():
            raise FileNotFoundError(
                "缺少核心指数参考库。请先导入CSMAR指数参考数据；没有指数确认时系统不会降级凑组合。"
            )
        holding_sessions = holding_weeks * 5
        minimum_sessions = max(252, holding_sessions * 8 + 1)
        history_sessions = minimum_sessions + 70
        with st.spinner("正在读取共同截止日全市场，识别主升突破并计算3–5股风险组合…"):
            hybrid = load_hybrid_universe(
                CSMAR_ROOT,
                overlay_root=OVERLAY_ROOT,
                as_of=as_of,
                minimum_sessions=minimum_sessions,
                history_sessions=history_sessions,
                reference_dataset_root=REFERENCE_ROOT,
                decision_date=as_of,
                mode=mode,
            )
            snapshot = hybrid.snapshot
            result = build_midterm_portfolio(
                snapshot.histories,
                snapshot.metadata,
                as_of=snapshot.data_cutoff,
                holding_weeks=holding_weeks,
                market_index_histories=snapshot.market_index_histories,
            )
            st.session_state["latest_midterm_result"] = result
            st.session_state["latest_midterm_universe"] = {
                "master": snapshot.master_symbols,
                "active": snapshot.active_symbols,
                "eligible": snapshot.eligible_symbols,
                "cutoff": snapshot.data_cutoff.isoformat(),
                "historical_baseline_cutoff": hybrid.historical_baseline_cutoff.isoformat(),
                "automatic_increment_cutoff": (
                    hybrid.automatic_increment_cutoff.isoformat()
                    if hybrid.automatic_increment_cutoff is not None
                    else "—"
                ),
                "common_cutoff": hybrid.common_cutoff.isoformat(),
                "sources": " + ".join(hybrid.sources),
                "mode": mode,
            }
    except Exception as exc:
        st.error(f"本轮研究没有运行：{exc}")

result: MidtermPortfolioResult | None = st.session_state.get("latest_midterm_result")
if result is not None:
    universe = st.session_state.get("latest_midterm_universe", {})
    st.subheader(STATUS_LABELS[result.status])
    st.caption(
        f"共同截止日：{result.data_cutoff.date() if result.data_cutoff is not None else '—'}｜"
        f"证券主表：{universe.get('master', '—')}｜当日有行情：{universe.get('active', '—')}｜"
        f"数据资格门后：{universe.get('eligible', '—')}"
    )
    st.caption(
        f"历史基线截止：{universe.get('historical_baseline_cutoff', '—')}｜"
        f"自动增量截止：{universe.get('automatic_increment_cutoff', '—')}｜"
        f"共同截止：{universe.get('common_cutoff', universe.get('cutoff', '—'))}｜"
        f"来源：{universe.get('sources', '—')}"
    )
    if result.reasons:
        st.warning("；".join(result.reasons))
    for warning in result.warnings:
        st.warning(warning)

    c1, c2, c3 = st.columns(3)
    c1.metric("达到主升介入门", result.entry_ready_count)
    c2.metric("进入机器组合池", result.search_pool_count)
    c3.metric("完成风险评估的组合", result.evaluated_portfolio_count)

    if result.positions and result.evaluation is not None:
        assert result.evaluation is not None
        positions = pd.DataFrame(
            {
                "顺序": item.rank,
                "代码": item.symbol,
                "名称": item.name,
                "行业": item.industry,
                "自动权重": item.weight,
                "介入结构": PATTERN_LABELS.get(item.entry_pattern.value, item.entry_pattern.value),
                "突破线": item.breakout_line,
                "距突破日": item.days_since_breakout,
                "量价信号分（非概率）": item.signal_score,
                "下行风险贡献": item.downside_risk_contribution,
                "证据状态": "待补财务/公告" if item.evidence_unknown else "否决证据已核验",
            }
            for item in result.positions
        )
        st.dataframe(
            positions.style.format(
                {
                    "自动权重": "{:.1%}",
                    "突破线": "{:.2f}",
                    "量价信号分（非概率）": "{:.3f}",
                    "下行风险贡献": "{:.1%}",
                }
            ),
            hide_index=True,
            width="stretch",
        )

        m = result.evaluation.metrics
        st.markdown("#### 组合风险与历史保守收益证据")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("股票 / 现金", f"{result.stock_exposure:.0%} / {result.cash_weight:.0%}")
        r2.metric("年化下行波动", f"{m.annual_downside_volatility:.1%}")
        r3.metric("60日回撤严重度P90", f"{m.rolling_max_drawdown_60_p90:.1%}")
        r4.metric("5日ES95", f"{m.es95_5d:.1%}")
        r5, r6, r7, r8 = st.columns(4)
        r5.metric("下跌期最高相关", f"{m.max_down_period_correlation:.2f}")
        r6.metric("最大单股风险贡献", f"{m.max_position_downside_risk_contribution:.1%}")
        r7.metric("持有期历史均值", f"{m.holding_period_return_mean:.1%}")
        r8.metric(
            f"{m.lcb_confidence:.0%}历史下界",
            f"{m.holding_period_return_lcb:.1%}",
        )
        st.caption(
            f"使用{m.holding_period_sample_count}个不重叠的{m.holding_period_sessions}交易日历史窗口；"
            f"成本假设{m.holding_period_cost_rate:.2%}。is_out_of_sample=false。"
        )
        if result.evidence_review_required:
            st.error(
                "这只是机器生成的临时候选组合：财务、正式公告或次日可买性尚未全部确认，"
                "状态为验证未就绪，不能作为最终买入清单。"
            )
        else:
            st.success("融资为0；权重已按实际占比从大到小排列。")
    else:
        st.info("系统保留现金，没有降低主升、介入点或风险标准来凑股票。")

    with st.expander("查看排除审计", expanded=False):
        if result.exclusions:
            st.dataframe(
                pd.DataFrame(
                    {
                        "代码": item.symbol,
                        "排除原因": "；".join(item.reasons),
                    }
                    for item in result.exclusions
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.write("本轮没有记录到个股排除项。")

    st.caption(result.disclaimer)
