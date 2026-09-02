from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd
import streamlit as st

from ashare_lab.analytics.indicators import enrich_indicators
from ashare_lab.analytics.levels import HorizonLevels, build_horizon_levels
from ashare_lab.bootstrap import application_data_dir, build_repository
from ashare_lab.services.build_weekly_portfolios import (
    WeeklyPortfolioStatus,
    archive_weekly_portfolios,
    build_weekly_portfolios,
)
from ashare_lab.services.load_csmar_universe import load_csmar_universe

HOLDING_PERIOD_OPTIONS = (1, 2, 4, 13, 26, 52)
HOLDING_PERIOD_LABELS = {
    1: "1周（入场节奏）",
    2: "2周（短中期确认）",
    4: "1个月（核心持有）",
    13: "3个月（默认核心）",
    26: "6个月（长期验证）",
    52: "1年（长期验证）",
}
HOLDING_PERIOD_GUIDANCE = {
    1: "1周仅用于观察入场节奏和结构确认，不把它当作日内或高频策略。",
    2: "2周用于确认10个交易日的短期强度是否得到1个月趋势支持，不追逐5日急涨。",
    4: "1个月属于核心持有区间，重点观察趋势延续与风险失效条件。",
    13: "3个月是默认核心周期，组合以中线趋势、分散和回撤控制为主。",
    26: "6个月用于检验中长期趋势是否持续，期间仍按风险触发条件复核。",
    52: "1年用于长期验证；历史样本不足时程序会明确不展示区间，不补造概率。",
}


def _holding_period_label(holding_weeks: int) -> str:
    return HOLDING_PERIOD_LABELS.get(holding_weeks, f"{holding_weeks}周")


def _nearest_level_plan(frame: pd.DataFrame, holding_weeks: int) -> HorizonLevels:
    prepared = frame.copy()
    if "trade_date" not in prepared.columns:
        if "date" in prepared.columns:
            prepared["trade_date"] = pd.to_datetime(prepared["date"])
        else:
            prepared["trade_date"] = pd.to_datetime(prepared.index)
    enriched = enrich_indicators(prepared)
    target_sessions = holding_weeks * 5
    return min(
        build_horizon_levels(enriched),
        key=lambda item: (abs(item.sessions - target_sessions), item.sessions),
    )


st.set_page_config(page_title="中线四股组合", page_icon="🧺", layout="wide")
st.title("中线四股组合")
st.caption(
    "默认持有3个月：1周只观察入场节奏，1–3个月是核心周期，"
    "6个月至1年用于长期验证；不做打板、日内交易或实时盯盘。"
)

st.info(
    "历史主源已切换为本机 CSMAR 全市场数据库。程序先取共同截止日仍有行情的股票，"
    "再过滤 ST/退市标记、历史不足、低流动性和异常原始价格跳变。"
)
st.warning(
    "当前个股日线仍是未复权价格且缺少全日成交量。资产负债表参考库只有当前快照，"
    "没有利润表、现金流量表和普通财报实际公告日；只有在决策日与价格截止均通过时点门后，"
    "才会以较低权重启用“资产负债表稳健度”，绝不冒充完整基本面或历史PIT因子。"
)

CSMAR_ROOT = application_data_dir() / "cache" / "csmar"
CSMAR_REFERENCE_ROOT = application_data_dir() / "cache" / "csmar_reference"

with st.form("weekly-portfolio-form"):
    mode_label = st.radio(
        "研究模式",
        options=("当前研究", "历史回放"),
        horizontal=True,
        help="历史回放永远不会使用今天取得的资产负债表快照。",
    )
    mode = "live" if mode_label == "当前研究" else "historical"
    as_of = st.date_input(
        "决策日（系统另行显示实际完整数据截止日）", value=date.today(), max_value=date.today()
    )
    holding_weeks = st.selectbox(
        "研究与计划持有周期",
        options=HOLDING_PERIOD_OPTIONS,
        index=3,
        format_func=_holding_period_label,
    )
    st.info(HOLDING_PERIOD_GUIDANCE[holding_weeks])
    submitted = st.form_submit_button(
        "读取全市场并生成四股研究组合",
        type="primary",
        width="stretch",
    )

if submitted:
    try:
        with st.spinner("正在读取本机 CSMAR 全市场数据库并计算风险收益排序…"):
            snapshot = load_csmar_universe(
                CSMAR_ROOT,
                as_of=as_of,
                reference_dataset_root=CSMAR_REFERENCE_ROOT,
                decision_date=as_of,
                mode=mode,
            )
            histories = snapshot.histories
            metadata = snapshot.metadata
            batch = build_weekly_portfolios(
                histories,
                metadata,
                as_of=snapshot.data_cutoff,
                holding_weeks=holding_weeks,
                market_index_histories=snapshot.market_index_histories,
            )
            batch = replace(batch, portfolios=(batch.for_profile("balanced"),))
            selected_symbols = {
                item.symbol for portfolio in batch.portfolios for item in portfolio.selected
            }
            selected_histories = {
                symbol: histories[symbol] for symbol in selected_symbols if symbol in histories
            }
            run_id = archive_weekly_portfolios(
                batch,
                histories,
                build_repository(),
                requested_as_of=as_of,
            )
            level_plans: dict[str, HorizonLevels] = {}
            level_failures: dict[str, str] = {}
            for symbol in sorted(selected_symbols):
                try:
                    level_plans[symbol] = _nearest_level_plan(
                        selected_histories[symbol],
                        holding_weeks,
                    )
                except Exception as exc:
                    level_failures[symbol] = str(exc)
            st.session_state["latest_weekly_batch"] = batch
            st.session_state["latest_weekly_histories"] = selected_histories
            st.session_state["latest_weekly_level_plans"] = level_plans
            st.session_state["latest_weekly_level_failures"] = level_failures
            st.session_state["latest_weekly_run_id"] = run_id
            st.session_state["latest_weekly_universe"] = {
                "data_cutoff": snapshot.data_cutoff.isoformat(),
                "master_symbols": snapshot.master_symbols,
                "active_symbols": snapshot.active_symbols,
                "eligible_symbols": snapshot.eligible_symbols,
                "excluded_symbols": snapshot.excluded_symbols,
                "minimum_median_amount_cny": snapshot.minimum_median_amount_cny,
                "requested_as_of": as_of.isoformat(),
                "mode": mode,
                "reference_common_cutoff": (
                    None
                    if snapshot.reference_common_cutoff is None
                    else snapshot.reference_common_cutoff.isoformat()
                ),
                "balance_sheet_strength_available": snapshot.balance_sheet_strength_available,
                "balance_sheet_strength_symbols": snapshot.balance_sheet_strength_symbols,
                "balance_sheet_strength_excluded_symbols": snapshot.balance_sheet_strength_excluded_symbols,
                "balance_sheet_snapshot_retrieved_at": (
                    None
                    if snapshot.balance_sheet_snapshot_retrieved_at is None
                    else snapshot.balance_sheet_snapshot_retrieved_at.isoformat()
                ),
                "balance_sheet_strength_reason": snapshot.balance_sheet_strength_reason,
                "reference_warnings": snapshot.reference_warnings,
            }
    except Exception as exc:
        st.error(f"全市场研究未生成：{exc}")

batch = st.session_state.get("latest_weekly_batch")
if batch:
    universe_stats = st.session_state.get("latest_weekly_universe", {})
    st.success(
        f"本次本地全市场截面：证券主表 {universe_stats.get('master_symbols', '—')} 只，"
        f"共同截止日有行情 {universe_stats.get('active_symbols', '—')} 只，"
        f"通过历史/流动性/风险资格门 {universe_stats.get('eligible_symbols', '—')} 只。"
    )
    st.caption(
        "这里的“全市场”指本机 CSMAR 导出可覆盖且共同截止日有行情的沪深A股研究截面；"
        "当前免费三源链覆盖沪深A股，不含北交所；停牌缺行受98%覆盖门约束，"
        "最新ST状态仍须开盘前人工复核。"
    )
    factor_labels = {
        "fundamentals": "财务基本面",
        "balance_sheet_strength": "资产负债表稳健度（当前快照）",
        "liquidity": "流动性",
        "news": "公司新闻/公告",
        "sector_context": "板块环境",
    }
    regime = batch.market_regime
    if regime is not None and regime.score is not None:
        regime_labels = {"risk_on": "偏多", "neutral": "中性", "risk_off": "风险规避"}
        st.markdown("#### 全市场环境")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("市场状态", regime_labels.get(regime.state.value, regime.state.value))
        r2.metric("站上20日均线", f"{regime.breadth_above_ma20:.1%}")
        r3.metric("站上60日均线", f"{regime.breadth_above_ma60:.1%}")
        r4.metric("站上120日均线", f"{regime.breadth_above_ma120:.1%}")
        st.caption(
            f"共同截止日覆盖{regime.eligible_symbols}只；全市场20日收益中位数"
            f"{regime.median_return_20:.1%}，60日中位数{regime.median_return_60:.1%}。"
            "这是组合级风险门，不作为单只股票加分项。"
        )
    index_regime = batch.index_regime
    if index_regime is not None:
        index_labels = {
            "risk_on": "偏多",
            "neutral": "中性",
            "risk_off": "风险规避",
            "unavailable": "不可用",
        }
        st.markdown("#### 核心指数确认")
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("指数状态", index_labels.get(index_regime.state.value, index_regime.state.value))
        i2.metric(
            "站上60日均线",
            "—"
            if index_regime.breadth_above_ma60 is None
            else f"{index_regime.breadth_above_ma60:.1%}",
        )
        i3.metric(
            "60日收益中位数",
            "—"
            if index_regime.median_return_60 is None
            else f"{index_regime.median_return_60:.1%}",
        )
        i4.metric(
            "120日最差回撤",
            "—"
            if index_regime.worst_max_drawdown_120 is None
            else f"{index_regime.worst_max_drawdown_120:.1%}",
        )
        st.caption(
            f"核心指数覆盖 {index_regime.eligible_indices}/{index_regime.required_indices}；"
            "它只作组合级风险确认，不给任何单只股票加分。"
        )
    st.markdown("#### 财务快照时点")
    finance_status = (
        f"已启用，覆盖{universe_stats.get('balance_sheet_strength_symbols', 0)}只"
        if universe_stats.get("balance_sheet_strength_available")
        else "未进入本轮评分"
    )
    st.write(
        f"资产负债表取得日：{universe_stats.get('balance_sheet_snapshot_retrieved_at') or '—'}；"
        f"状态：{finance_status}；原因：{universe_stats.get('balance_sheet_strength_reason', '—')}。"
    )
    for warning in universe_stats.get("reference_warnings", ()):
        st.warning(warning)
    if batch.factor_coverage:
        st.markdown("#### 本轮评分因子覆盖")
        st.dataframe(
            pd.DataFrame(
                {
                    "因子": factor_labels.get(item.factor, item.factor),
                    "覆盖": f"{item.provided}/{item.eligible}",
                    "是否进入评分": "启用" if item.enabled else "禁用",
                    "说明": item.reason,
                }
                for item in batch.factor_coverage
            ),
            hide_index=True,
            width="stretch",
        )
        disabled = [
            factor_labels.get(item.factor, item.factor)
            for item in batch.factor_coverage
            if not item.enabled
        ]
        if disabled:
            st.warning(
                "以下因子本轮没有完整的点时覆盖，未进入排序："
                + "、".join(disabled)
                + "。缺失值没有被伪造成中性分。"
            )
    execution_exclusions = [
        item
        for item in batch.exclusions
        if any(
            reason.startswith(("formation_", "overheated_acceleration")) for reason in item.reasons
        )
    ]
    if execution_exclusions:
        preview = "、".join(item.symbol for item in execution_exclusions[:12])
        suffix = "…" if len(execution_exclusions) > 12 else ""
        st.warning(
            f"本轮有{len(execution_exclusions)}只股票因形成日涨停/成交不可确认或走势过热"
            f"被移出可执行组合，程序已在剩余股票中重新满足行业与相关性约束：{preview}{suffix}"
        )
    run_id = st.session_state.get("latest_weekly_run_id")
    archive_label = f"｜档案 {run_id[:8]}" if run_id else "｜没有合格组合可留档"
    st.success(
        f"组合数据共同截止：{batch.data_cutoff.date() if batch.data_cutoff is not None else '不可用'}"
        f"{archive_label}"
    )
    st.subheader(f"本轮研究周期：{_holding_period_label(batch.holding_weeks)}")
    st.caption(
        "页面只展示所选周期的历史情景与风险证据，不预设年化收益目标，也不把历史频率表述为未来概率。"
    )
    level_plans = st.session_state.get("latest_weekly_level_plans", {})
    tabs = st.tabs(["唯一组合（CSMAR全市场研究）"])
    for tab, portfolio in zip(tabs, batch.portfolios, strict=True):
        with tab:
            if portfolio.status != WeeklyPortfolioStatus.READY:
                st.info("本档没有四只同时通过行业和相关性约束，保留现金。")
                st.code("\n".join(portfolio.reasons))
                continue
            assert portfolio.allocation is not None
            allocation_rows = []
            selected_by_symbol = {item.symbol: item for item in portfolio.selected}
            for position in portfolio.allocation.positions:
                selected = selected_by_symbol[position.ticker]
                allocation_rows.append(
                    {
                        "角色": position.role.value,
                        "代码": position.ticker,
                        "名称": selected.name,
                        "行业": selected.industry,
                        "组合权重": position.weight,
                        "排序分": selected.score,
                    }
                )
            st.dataframe(
                pd.DataFrame(allocation_rows),
                hide_index=True,
                width="stretch",
                column_config={"组合权重": st.column_config.NumberColumn(format="%.2f%%")},
            )
            st.metric("保留现金", f"{portfolio.allocation.cash_ratio:.0%}")

            risk = portfolio.historical_risk
            if risk:
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "历史代理CAGR",
                    "—" if risk.historical_cagr is None else f"{risk.historical_cagr:.1%}",
                )
                c2.metric(
                    "历史年化波动",
                    "—"
                    if risk.historical_annual_volatility is None
                    else f"{risk.historical_annual_volatility:.1%}",
                )
                c3.metric(
                    "历史最大回撤",
                    "—"
                    if risk.historical_max_drawdown is None
                    else f"{risk.historical_max_drawdown:.1%}",
                )
                c4, c5, c6 = st.columns(3)
                c4.metric(
                    "历史Sharpe",
                    "—" if risk.historical_sharpe is None else f"{risk.historical_sharpe:.2f}",
                )
                c5.metric(
                    "历史Sortino",
                    "—" if risk.historical_sortino is None else f"{risk.historical_sortino:.2f}",
                )
                c6.metric(
                    "历史Calmar",
                    "—" if risk.historical_calmar is None else f"{risk.historical_calmar:.2f}",
                )
                drawdown_rows = []
                for distribution in risk.drawdown_windows:
                    interval = distribution.breach_interval
                    drawdown_rows.append(
                        {
                            "窗口": f"{distribution.window_sessions}日",
                            "历史回撤P50": distribution.drawdown_magnitude_p50,
                            "历史回撤P90": distribution.drawdown_magnitude_p90,
                            "超预算历史频率": distribution.breach_probability,
                            "Wilson下界": None if interval is None else interval.lower,
                            "Wilson上界": None if interval is None else interval.upper,
                            "重叠样本": distribution.sample_n,
                            "近似非重叠样本": distribution.effective_non_overlapping_n,
                        }
                    )
                st.dataframe(
                    pd.DataFrame(drawdown_rows),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        column: st.column_config.NumberColumn(format="%.1f%%")
                        for column in (
                            "历史回撤P50",
                            "历史回撤P90",
                            "超预算历史频率",
                            "Wilson下界",
                            "Wilson上界",
                        )
                    },
                )
            scenario = portfolio.historical_scenario
            if scenario and scenario.available:
                st.write(
                    f"历史{scenario.horizon_sessions}日组合收益情景："
                    f"P10 {scenario.return_p10:.1%} / "
                    f"中位 {scenario.return_p50:.1%} / P90 {scenario.return_p90:.1%}；"
                    f"历史正收益频率 {scenario.historical_positive_rate:.0%}（样本 {scenario.sample_n}）"
                )
            for warning in (item for item in portfolio.risk_warnings if "收益期望" not in item):
                st.warning(warning)

            st.markdown("#### 四只股票的关键计划")
            for position in portfolio.allocation.positions:
                selected = selected_by_symbol[position.ticker]
                plan = level_plans.get(position.ticker)
                with st.container(border=True):
                    st.markdown(
                        f"**{selected.name}（{position.ticker}）** · "
                        f"{selected.industry} · 权重 {position.weight:.2%}"
                    )
                    if plan is None:
                        st.warning("本轮历史字段不足，无法可靠计算关键价位；不生成替代数字。")
                        continue
                    p1, p2, p3, p4, p5 = st.columns(5)
                    p1.metric(
                        "回踩计划区",
                        f"{plan.pullback_entry_low:.2f}–{plan.pullback_entry_high:.2f}",
                    )
                    p2.metric("收盘突破确认线", f"{plan.breakout_trigger:.2f}")
                    p3.metric("结构失效位", f"{plan.invalidation:.2f}")
                    p4.metric("第一减仓区", f"{plan.reduce_low:.2f}–{plan.reduce_high:.2f}")
                    p5.metric(
                        "风险收益比",
                        "—" if plan.reward_risk_ratio is None else f"{plan.reward_risk_ratio:.2f}",
                    )
                    st.write(
                        f"第二减仓区：{plan.second_reduce_low:.2f}–"
                        f"{plan.second_reduce_high:.2f}；采用最接近所选周期的"
                        f"{plan.horizon}（{plan.sessions}个交易日）结构。"
                    )
                    st.caption(plan.breakout_confirmation_rule)
                    st.caption(plan.stop_execution_rule)
            st.caption(portfolio.disclaimer)

    failures = st.session_state.get("latest_weekly_failures", {})
    if failures:
        with st.expander(f"{len(failures)}只读取失败或被跳过"):
            st.json(failures)
    level_failures = st.session_state.get("latest_weekly_level_failures", {})
    if level_failures:
        with st.expander(f"{len(level_failures)}只关键价位计算失败"):
            st.json(level_failures)

st.info(
    "当前预览使用平衡型综合评分作为过渡规则：四股按3:2:2:1分配，"
    "合计股票仓80%+现金20%，融资为0。建议每周或风险条件触发时复核，"
    "不因盘中波动频繁换手；正式全市场版仍需通过点时数据和滚动样本外验证。"
)
