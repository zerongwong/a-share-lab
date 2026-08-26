from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = PROJECT_ROOT / "src" / "ashare_lab" / "ui"


def test_main_router_registers_only_the_medium_horizon_portfolio() -> None:
    source = (UI_ROOT / "A股研究室.py").read_text(encoding="utf-8")

    assert '_existing_page("02")' in source
    assert '_existing_page("03")' not in source
    assert '_existing_page("06")' not in source
    assert "closing-strength-five" not in source
    assert "stock-position-monitor" not in source
    assert (UI_ROOT / "pages" / "03_涨停实验室.py").exists()
    assert (UI_ROOT / "pages" / "06_持仓监控.py").exists()


def test_portfolio_page_uses_named_horizons_and_no_visible_return_target() -> None:
    source = (UI_ROOT / "pages" / "02_每周三档组合.py").read_text(encoding="utf-8")

    for label in (
        "1周（入场节奏）",
        "1个月（核心持有）",
        "3个月（默认核心）",
        "6个月（长期验证）",
        "1年（长期验证）",
    ):
        assert label in source
    assert "HOLDING_PERIOD_OPTIONS = (1, 4, 13, 26, 52)" in source
    assert "index=2" in source
    assert "年化收益期望（仅用于检验历史支持度，不是收益承诺）" not in source
    assert "收益门槛" not in source
    assert "收益期望状态" not in source
    assert "不做打板、日内交易或实时盯盘" in source
