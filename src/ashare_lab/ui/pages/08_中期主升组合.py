from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from ashare_lab.bootstrap import application_data_dir
from ashare_lab.services.build_midterm_portfolio import (
    MIDTERM_METHOD_VERSION,
    CandidateAction,
    MidtermPortfolioResult,
    MidtermPortfolioStatus,
    build_midterm_portfolio,
    horizon_history_requirements,
)
from ashare_lab.services.load_hybrid_universe import load_hybrid_universe
from ashare_lab.ui.midterm_position_view import build_midterm_position_views

HOLDING_PERIOD_OPTIONS = (1, 2, 4, 13, 26, 52)
HOLDING_PERIOD_LABELS = {
    1: "1周（最短观察）",
    2: "2周（短中期确认）",
    4: "1个月（短中期）",
    13: "3个月（默认）",
    26: "6个月（延长复核）",
    52: "1年（长期复核）",
}
STATUS_LABELS = {
    MidtermPortfolioStatus.DATA_NOT_READY: "数据未就绪",
    MidtermPortfolioStatus.VALIDATION_NOT_READY: "研究候选已生成｜证据待核验",
    MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO: "研究候选已生成｜暂无可介入组合",
    MidtermPortfolioStatus.RESEARCH_ONLY: "研究组合已生成",
}
PATTERN_LABELS = {
    "volume_confirmed_breakout": "近期放量突破",
    "healthy_breakout_pullback": "突破后健康回踩",
    "breakout_line_reclaim": "回踩后重新站回突破线",
}
ACTION_LABELS = {
    CandidateAction.CONDITIONAL_ENTRY: "可条件介入",
    CandidateAction.WAIT_CONFIRMATION: "等待确认",
    CandidateAction.OBSERVE_ONLY: "仅观察",
}
ENTRY_STRICTNESS_LABELS = {
    "standard": "标准确认",
    "tight": "加强确认",
    "defensive": "防守确认",
    "exception_only": "例外级确认",
    "unavailable": "不可用",
}
REASON_LABELS = {
    "research_candidates_generated_but_fewer_than_three_pass_current_entry_policy": (
        "候选已经生成，但当前周期的介入门后不足3只，暂不组成持仓组合"
    ),
    "no_3_to_5_stock_set_passed_grid_industry_and_risk_budgets": (
        "候选已经生成，但没有3至5股组合同时通过10%操作档、行业与下行风险约束"
    ),
    "price_cycle_evidence_unavailable": "价格周期证据不完整",
    "core_index_regime_unavailable": "核心指数证据不完整",
    "annual_downside_volatility": "年化下行波动超过预算",
    "rolling_drawdown_60_p90": "60日滚动回撤压力超过预算",
    "horizon_rolling_drawdown_p90": "所选持有期窗口的滚动回撤压力超过预算",
    "es95_5d": "5日尾部损失超过预算",
    "down_period_correlation": "下跌期相关性超过预算",
    "position_downside_risk_contribution": "单只股票下行风险贡献超过预算",
    "industry_concentration": "行业集中度超过预算",
    "holding_period_return_lcb_below_minimum": "历史持有期收益下界未达最低门槛",
}
ACTION_REASON_LABELS = {
    "standard_multi_timeframe_and_daily_entry_confirmed": "多周期结构与日线介入已确认",
    "standard_entry_structure_confirmed": "标准介入结构已确认",
    "tight_entry_confirmed": "加强介入条件已确认",
    "defensive_entry_confirmed": "防守介入条件已确认",
    "exception_only_entry_confirmed": "例外级介入条件已确认",
    "fundamental_announcement_or_execution_evidence_requires_review": (
        "财务、公告或可成交性仍待复核"
    ),
    "sixty_session_absolute_return_not_positive": "60日绝对收益尚未为正",
    "relative_strength_not_top_decile": "相对强度未进入前10%",
    "ma20_not_above_ma60": "MA20尚未高于MA60",
    "downside_capture_above_0_80_or_unavailable": "下跌捕获率高于0.80或数据不足",
    "breakout_amount_ratio_below_1_20": "突破量额不足1.20倍",
    "breakout_amount_ratio_below_1_30": "突破量额不足1.30倍",
    "distance_ma20_ratio_above_8pct": "距MA20超过8%",
    "distance_ma20_ratio_above_6pct": "距MA20超过6%",
    "distance_ma20_atr_above_2": "距MA20超过2ATR",
    "distance_ma20_atr_above_1_5": "距MA20超过1.5ATR",
    "downtrend_pressure_requires_pullback_or_reclaim": "下行压力期只接受回踩或重新站回",
    "horizon_execution_activity_ratio_below_1_20": "本期限日线成交活跃度不足1.20倍",
    "horizon_execution_activity_ratio_below_1_30": "本期限日线成交活跃度不足1.30倍",
    "horizon_execution_average_distance_above_8pct": "价格距本期限执行均线超过8%",
    "horizon_execution_average_distance_above_6pct": "价格距本期限执行均线超过6%",
}
TIMEFRAME_LABELS = {
    "daily": "日线",
    "weekly_completed": "完整周线",
    "monthly_completed": "完整月线",
}
STRUCTURE_LABELS = {
    "insufficient": "证据不足",
    "failed": "结构失效",
    "trend_continuation_without_entry_structure": "趋势延续但无介入结构",
    "base_not_yet_near_breakout": "底座形成中",
    "near_breakout": "临近突破",
    "healthy_post_breakout_pullback": "突破后健康回踩",
    "volume_confirmed_breakout": "量能确认突破",
}
EXECUTION_LABELS = {
    "insufficient": "日线证据不足",
    "failed": "日线失效",
    "extended_do_not_chase": "过度延伸不追",
    "wait_for_daily_confirmation": "等待日线确认",
    "daily_healthy_pullback_ready": "日线健康回踩",
    "daily_volume_breakout_ready": "日线放量突破",
}
LATEST_MIDTERM_RUN_KEY = "latest_midterm_run"


def _period_label(weeks: int) -> str:
    return HOLDING_PERIOD_LABELS[weeks]


def _reason_label(reason: str) -> str:
    return REASON_LABELS.get(reason, reason)


def _action_reason_label(reason: str) -> str:
    return ACTION_REASON_LABELS.get(reason, reason)


def _weight_label(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _timeframe_value(item: object, path: str, default: object = None) -> object:
    value = getattr(item, "timeframe", None)
    for name in path.split("."):
        value = getattr(value, name, default)
        if value is default:
            return default
    return getattr(value, "value", value)


st.set_page_config(page_title="中期主升组合", page_icon="📈", layout="wide")
st.title("中期主升组合")
st.caption(
    "无论市场强弱都先筛主升趋势与明确介入点，再自动比较3、4、5只组合和权重。"
    "4只为常态；只有3只合格时增加现金，第5只确有分散价值时才加入。"
)

st.info(
    "价格和成交额负责选股与介入；资产负债表和正式公告只负责否决风险，不用利好新闻抬分。"
    "市场周期只调整仓位、介入难度和风险预算，不再停止全市场筛选。"
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
        index=3,
        format_func=_period_label,
    )
    submitted = st.form_submit_button(
        "运行全市场主升组合研究",
        type="primary",
        width="stretch",
    )

current_request = {
    "mode": mode,
    "as_of": as_of.isoformat(),
    "holding_weeks": holding_weeks,
    "method_version": MIDTERM_METHOD_VERSION,
}

if submitted:
    st.session_state.pop(LATEST_MIDTERM_RUN_KEY, None)
    try:
        if not (REFERENCE_ROOT / "csmar_reference.duckdb").is_file():
            raise FileNotFoundError(
                "缺少核心指数参考库。请先导入CSMAR指数参考数据；没有指数确认时系统不会降级凑组合。"
            )
        history_requirements = horizon_history_requirements(holding_weeks)
        with st.spinner("正在读取共同截止日全市场，识别主升突破并计算3–5股风险组合…"):
            hybrid = load_hybrid_universe(
                CSMAR_ROOT,
                overlay_root=OVERLAY_ROOT,
                as_of=as_of,
                minimum_sessions=history_requirements.qualification_minimum_sessions,
                history_sessions=history_requirements.history_read_sessions,
                minimum_qualification_sessions=(
                    history_requirements.qualification_minimum_sessions
                ),
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
            universe = {
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
            st.session_state[LATEST_MIDTERM_RUN_KEY] = {
                "request": current_request,
                "result": result,
                "universe": universe,
            }
    except Exception as exc:
        st.error(f"本轮研究没有运行：{exc}")

latest_run = st.session_state.get(LATEST_MIDTERM_RUN_KEY)
result: MidtermPortfolioResult | None = None
universe: dict[str, object] = {}
if (
    isinstance(latest_run, dict)
    and latest_run.get("request") == current_request
    and isinstance(latest_run.get("result"), MidtermPortfolioResult)
    and isinstance(latest_run.get("universe"), dict)
):
    result = latest_run["result"]
    universe = latest_run["universe"]

if result is not None:
    title = STATUS_LABELS[result.status]
    if (
        result.status == MidtermPortfolioStatus.NO_ELIGIBLE_PORTFOLIO
        and not result.research_candidates
    ):
        title = "本轮没有达到主升结构的研究候选"
    st.subheader(title)
    st.caption(
        f"中央多周期实现状态：{getattr(result, 'central_implementation_status', '—')}｜"
        f"分析组件状态：{getattr(result, 'multi_timeframe_component_status', '—')}。"
        "当前仍是部分实现的研究输出，不代表中央合同或样本外验证已完成。"
    )
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
    st.caption("计划适用：共同截止日之后的下一交易日；开盘前还需复核停牌、涨停或跳空等可买性。")
    try:
        position_views = build_midterm_position_views(result)
    except ValueError:
        position_views = ()
        st.error("仓位数据一致性校验失败；本轮不展示权重，行动层保持现金。")
        st.stop()
    if position_views:
        has_research_allocation = result.research_evaluation is not None
        # A Streamlit hot reload may retain a pre-upgrade result object in
        # session state.  Treat the appended observation layer as absent until
        # the next complete run instead of crashing the page.
        has_observation_allocation = getattr(result, "observation_evaluation", None) is not None
        if has_research_allocation:
            st.markdown("#### 本轮组合简表（股票仓内10%操作档）")
            st.caption(
                "系统保留精确权重作为审计目标，再从股票仓内10%整数档中选择最接近且"
                "重新通过风险与行业约束的操作权重。这不是账户实仓、下单指令或未来最优。"
            )
        elif has_observation_allocation:
            st.markdown("#### 候选观察方案｜风险门未通过")
            st.caption(
                "系统保留了一组已完成结构计算但未通过研究风险门的10%档观察配比。"
                "它只用于比较候选，不是资金仓位、买入建议或可执行组合。"
            )
            observation_rejection_reasons = getattr(
                result,
                "observation_rejection_reasons",
                (),
            )
            if observation_rejection_reasons:
                st.warning(
                    "本观察组合未通过："
                    + "；".join(_reason_label(reason) for reason in observation_rejection_reasons)
                )
        else:
            st.markdown("#### 研究候选｜尚未形成风险合格权重")
            st.caption(
                "本轮有候选股，但没有3–5股组合通过研究风险预算；—表示未算出权重，不是0%仓位方案。"
            )
        allocation_summary, action_summary = st.columns(2)
        allocation_summary.metric(
            "研究层测算：股票 / 现金",
            (
                f"{result.research_stock_exposure:.0%} / {result.research_cash_weight:.0%}"
                if has_research_allocation
                else "尚未形成"
            ),
        )
        action_summary.metric(
            "行动层测算：股票 / 现金",
            f"{result.stock_exposure:.0%} / {result.cash_weight:.0%}",
        )
        for item in position_views:
            with st.container(border=True):
                stock_col, weight_col, entry_col = st.columns([2.0, 2.0, 2.8])
                stock_col.caption("股票")
                stock_col.markdown(item.stock_label)
                weight_col.caption(
                    "计划仓位" if has_research_allocation else "候选观察配比（非资金仓位）"
                )
                weight_col.markdown(item.planned_weight_label)
                entry_col.caption("介入价格（条件）" if has_research_allocation else "价格观察条件")
                entry_col.markdown(item.entry_price_condition)
        if has_research_allocation:
            st.caption(
                "计划仓位以10%为一档，在股票仓内合计100%；占总资金比例="
                "股票仓内比例×本轮股票敞口，其余为现金。介入价格是基于"
                "所选持有期的慢周期/主结构和完整日线生成，未触发条件时保持等待；"
                "不是无条件买入价。"
            )
        elif has_observation_allocation:
            st.caption(
                "候选观察配比仅在观察池内合计100%，不对应总资金，也不会进入行动层。"
                "价格观察线基于共同截止日生成；触及不等于可以买，仍需复核周期、财务、"
                "公告、停牌、涨停和跳空。"
            )
    if result.price_cycle is not None:
        cycle = result.price_cycle
        st.markdown("#### 市场价格周期与风险姿态")
        confidence = (
            f"规则一致度 {cycle.confidence:.1%}"
            if cycle.confidence is not None
            else "规则一致度不可用"
        )
        st.info(
            f"{cycle.label}｜{confidence}。一致度不是未来涨跌概率；"
            "周期决定进攻或防守程度，不决定是否继续选股。"
        )
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("股票敞口上限", f"{cycle.policy.max_stock_exposure:.0%}")
        p2.metric("最低现金", f"{cycle.policy.minimum_cash_weight:.0%}")
        p3.metric(
            "介入严格度",
            ENTRY_STRICTNESS_LABELS.get(
                cycle.policy.entry_strictness.value,
                cycle.policy.entry_strictness.value,
            ),
        )
        p4.metric("融资", "0%")
        with st.expander("查看周期证据与本轮介入要求", expanded=False):
            st.markdown("**周期证据**")
            for item in cycle.evidence:
                st.write(f"- {item}")
            st.markdown("**本轮介入要求**")
            for item in cycle.policy.entry_requirements:
                st.write(f"- {item}")
            st.caption("当前只实现价格周期代理，尚未实现完整的经济、信贷和投资者心理周期。")
    if result.reasons:
        st.warning("；".join(_reason_label(reason) for reason in result.reasons))
    for warning in result.warnings:
        st.warning(warning)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("通过期限结构门", result.horizon_candidate_count)
    c2.metric("日线介入结构就绪", result.entry_ready_count)
    c3.metric("通过当前介入门", result.actionable_candidate_count)
    c4.metric("完成研究风险评估的组合", result.evaluated_portfolio_count)
    st.caption(
        f"本轮行动层测算（非账户实仓）：股票{result.stock_exposure:.0%}｜"
        f"现金{result.cash_weight:.0%}｜融资{result.borrowed_weight:.0%}。"
        "研究权重是假设证据补齐后的风险测算，不是当前持仓。"
    )

    if result.research_candidates:
        st.markdown("#### 3–5只研究候选")
        st.caption(
            "这是全市场量价筛选后的机器候选，不等于当前可以买；"
            "行动状态还会受周期、财务、公告和可成交性约束。"
        )
        candidate_rows = pd.DataFrame(
            {
                "顺序": item.rank,
                "代码": item.symbol,
                "名称": item.name,
                "精确目标·占总资金（审计）": item.research_weight,
                "操作权重·占总资金": item.operational_account_weight,
                "操作比例·股票仓内（10%档）": (item.operational_stock_sleeve_weight),
                "行业": item.industry,
                "慢周期方向": (
                    TIMEFRAME_LABELS.get(
                        str(_timeframe_value(item, "slow_direction.timeframe", "")),
                        str(_timeframe_value(item, "slow_direction.timeframe", "—")),
                    )
                    + "｜"
                    + str(_timeframe_value(item, "slow_direction.direction", "—"))
                ),
                "主周期结构": (
                    TIMEFRAME_LABELS.get(
                        str(_timeframe_value(item, "structure.timeframe", "")),
                        str(_timeframe_value(item, "structure.timeframe", "—")),
                    )
                    + "｜"
                    + STRUCTURE_LABELS.get(
                        str(_timeframe_value(item, "structure.state", "")),
                        str(_timeframe_value(item, "structure.state", "—")),
                    )
                ),
                "日线执行": EXECUTION_LABELS.get(
                    str(_timeframe_value(item, "execution.state", "")),
                    str(_timeframe_value(item, "execution.state", "—")),
                ),
                "当前行动": ACTION_LABELS[item.action],
                "介入结构": PATTERN_LABELS.get(
                    item.entry_pattern.value,
                    item.entry_pattern.value,
                ),
                "主结构线": item.breakout_line,
                "期限绝对收益": item.horizon_absolute_return,
                "期限相对强度分位": item.relative_strength_percentile,
                "下跌捕获率": item.downside_capture_ratio,
                "研究下行风险贡献": item.downside_risk_contribution,
                "研究分（非概率）": item.signal_score,
                "待确认事项": "；".join(
                    _action_reason_label(reason) for reason in item.action_reasons
                ),
            }
            for item in result.research_candidates
        )
        st.dataframe(
            candidate_rows.style.format(
                {
                    "主结构线": "{:.2f}",
                    "精确目标·占总资金（审计）": "{:.2%}",
                    "操作权重·占总资金": "{:.1%}",
                    "操作比例·股票仓内（10%档）": "{:.0%}",
                    "期限绝对收益": "{:.1%}",
                    "期限相对强度分位": "{:.1%}",
                    "下跌捕获率": lambda value: "—" if pd.isna(value) else f"{value:.2f}",
                    "研究下行风险贡献": "{:.1%}",
                    "研究分（非概率）": "{:.3f}",
                },
                na_rep="—",
            ),
            hide_index=True,
            width="stretch",
        )

    if result.research_evaluation is not None and not result.positions:
        research_metrics = result.research_evaluation.metrics
        st.markdown("#### 候选组合历史风险压力测试")
        st.warning(
            "这只是对研究候选做的组合风险测算，不代表现在应按该权重买入。当前行动层仍为100%现金。"
        )
        q1, q2, q3, q4 = st.columns(4)
        q1.metric(
            "候选测算股票 / 现金",
            f"{result.research_stock_exposure:.0%} / {result.research_cash_weight:.0%}",
        )
        q2.metric("年化下行波动", f"{research_metrics.annual_downside_volatility:.1%}")
        q3.metric(
            f"{research_metrics.horizon_rolling_drawdown_window_sessions}日回撤严重度P90",
            f"{research_metrics.horizon_rolling_max_drawdown_p90:.1%}",
        )
        q4.metric("5日ES95", f"{research_metrics.es95_5d:.1%}")
        st.caption(
            f"已评估{result.evaluated_portfolio_count}个候选组合；"
            f"历史持有期下界{research_metrics.holding_period_return_lcb:.1%}，"
            "is_out_of_sample=false，只用于候选组合比较。"
        )

    if result.positions and result.evaluation is not None:
        assert result.evaluation is not None
        st.markdown("#### 当前通过介入门与组合风险预算的研究组合")
        positions = pd.DataFrame(
            {
                "顺序": item.rank,
                "代码": item.symbol,
                "名称": item.name,
                "行业": item.industry,
                "主周期结构": (
                    TIMEFRAME_LABELS.get(
                        str(_timeframe_value(item, "structure.timeframe", "")),
                        str(_timeframe_value(item, "structure.timeframe", "—")),
                    )
                    + "｜"
                    + STRUCTURE_LABELS.get(
                        str(_timeframe_value(item, "structure.state", "")),
                        str(_timeframe_value(item, "structure.state", "—")),
                    )
                ),
                "精确目标·占总资金（审计）": item.weight,
                "行动操作权重·占总资金": item.operational_account_weight,
                "行动操作比例·股票仓内（10%档）": (item.operational_stock_sleeve_weight),
                "介入结构": PATTERN_LABELS.get(item.entry_pattern.value, item.entry_pattern.value),
                "主结构线": item.breakout_line,
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
                    "精确目标·占总资金（审计）": "{:.2%}",
                    "行动操作权重·占总资金": "{:.1%}",
                    "行动操作比例·股票仓内（10%档）": "{:.0%}",
                    "主结构线": "{:.2f}",
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
        r3.metric(
            f"{m.horizon_rolling_drawdown_window_sessions}日回撤严重度P90",
            f"{m.horizon_rolling_max_drawdown_p90:.1%}",
        )
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
            st.success("融资为0；行动层权重已按股票仓内10%整数档计算并重新复核风险，不是账户实仓。")
    elif result.research_candidates:
        st.info(
            "研究候选已经保留，但当前没有形成可介入组合。"
            "系统保持现金，没有降低主升、介入点、证据或风险标准来凑股票。"
        )
    else:
        st.info("本轮没有股票达到主升研究门，系统保持现金，不为凑数降低标准。")

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
