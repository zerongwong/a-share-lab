from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = PROJECT_ROOT / "src" / "ashare_lab" / "ui"


def test_main_router_registers_research_and_notification_settings_only() -> None:
    source = (UI_ROOT / "A股研究室.py").read_text(encoding="utf-8")

    assert '_existing_page("10")' in source
    assert '_existing_page("08")' not in source
    assert '_existing_page("07")' in source
    assert '_existing_page("03")' not in source
    assert '_existing_page("06")' not in source
    assert "closing-strength-five" not in source
    assert "stock-position-monitor" not in source
    assert "notification-settings" in source
    assert (UI_ROOT / "pages" / "03_涨停实验室.py").exists()
    assert (UI_ROOT / "pages" / "06_持仓监控.py").exists()
    assert (UI_ROOT / "pages" / "07_通知设置.py").exists()
    assert (UI_ROOT / "pages" / "08_中期主升组合.py").exists()


def test_portfolio_page_uses_named_horizons_and_no_visible_return_target() -> None:
    source = (UI_ROOT / "pages" / "08_中期主升组合.py").read_text(encoding="utf-8")

    for label in (
        "1周（最短观察）",
        "2周（短中期确认）",
        "1个月（短中期）",
        "3个月（默认）",
        "6个月（延长复核）",
        "1年（长期复核）",
    ):
        assert label in source
    assert "HOLDING_PERIOD_OPTIONS = (1, 2, 4, 13, 26, 52)" in source
    assert "index=3" in source
    assert "年化收益期望（仅用于检验历史支持度，不是收益承诺）" not in source
    assert "收益门槛" not in source
    assert "收益期望状态" not in source
    assert "3、4、5只组合" in source
    assert "融资为0" in source
    assert "历史代理" in source


def test_legacy_portfolio_page_also_exposes_two_week_option_without_changing_default() -> None:
    source = (UI_ROOT / "pages" / "02_每周三档组合.py").read_text(encoding="utf-8")

    assert "HOLDING_PERIOD_OPTIONS = (1, 2, 4, 13, 26, 52)" in source
    assert "2周（短中期确认）" in source
    assert "10个交易日" in source
    assert "index=3" in source


def test_portfolio_page_uses_hybrid_baseline_and_verified_increment() -> None:
    source = (UI_ROOT / "pages" / "08_中期主升组合.py").read_text(encoding="utf-8")

    assert "load_hybrid_universe" in source
    assert 'OVERLAY_ROOT = application_data_dir() / "cache" / "market_overlay"' in source
    for label in (
        "历史基线截止",
        "自动增量截止",
        "共同截止",
        "来源",
        "共同截止日之后的下一交易日",
    ):
        assert label in source


def test_portfolio_page_exposes_each_weight_before_the_wide_audit_table() -> None:
    source = (UI_ROOT / "pages" / "08_中期主升组合.py").read_text(encoding="utf-8")

    allocation = source.index("本轮组合简表（股票仓内10%操作档）")
    cycle = source.index("市场价格周期与风险姿态")
    candidate_table = source.index("candidate_rows = pd.DataFrame(")

    assert allocation < cycle < candidate_table
    for label in (
        'caption("股票")',
        '"计划仓位"',
        '"介入价格（条件）"',
        '"候选观察配比（非资金仓位）"',
        '"价格观察条件"',
        "研究层测算：股票 / 现金",
        "行动层测算：股票 / 现金",
    ):
        assert label in source
    simple_table = source[allocation:cycle]
    assert "st.columns([2.0, 2.0, 2.8])" in simple_table
    assert "个股介入状态" not in simple_table
    assert "行动层测算·占总资金" not in simple_table
    assert "breakout_line" not in simple_table
    assert "在股票仓内合计100%" in source
    assert "重新通过风险与行业约束" in source
    assert "精确目标·占总资金（审计）" in source
    assert "st.stop()" in source
    assert "非账户实仓" in source
    assert "研究候选｜尚未形成风险合格权重" in source
    assert "—表示未算出权重，不是0%仓位方案" in source
    assert "候选观察方案｜风险门未通过" in source
    assert "触及不等于可以买" in source
    assert "本轮没有股票达到主升研究门" in source
