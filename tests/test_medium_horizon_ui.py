from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = PROJECT_ROOT / "src" / "ashare_lab" / "ui"


def test_main_router_registers_research_and_notification_settings_only() -> None:
    source = (UI_ROOT / "A股研究室.py").read_text(encoding="utf-8")

    assert '_existing_page("08")' in source
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
        "1个月（短中期）",
        "3个月（默认）",
        "6个月（延长复核）",
        "1年（长期复核）",
    ):
        assert label in source
    assert "HOLDING_PERIOD_OPTIONS = (1, 4, 13, 26, 52)" in source
    assert "index=2" in source
    assert "年化收益期望（仅用于检验历史支持度，不是收益承诺）" not in source
    assert "收益门槛" not in source
    assert "收益期望状态" not in source
    assert "3、4、5只组合" in source
    assert "融资为0" in source
    assert "历史代理" in source


def test_portfolio_page_uses_hybrid_baseline_and_verified_increment() -> None:
    source = (UI_ROOT / "pages" / "08_中期主升组合.py").read_text(encoding="utf-8")

    assert "load_hybrid_universe" in source
    assert 'OVERLAY_ROOT = application_data_dir() / "cache" / "market_overlay"' in source
    for label in ("历史基线截止", "自动增量截止", "共同截止", "来源"):
        assert label in source
