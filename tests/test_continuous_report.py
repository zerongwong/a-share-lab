from __future__ import annotations

from datetime import date

import pytest

from ashare_lab.services.build_continuous_report import render_continuous_report


def _entry(**changes):
    return {
        "symbol": "600000.SH",
        "name": "合成测试",
        "account_weight": 0.125,
        "entry_label": "确认≥10.01，买≤10.30+量",
        "protection_line": 9.4812,
        "entry_qualified": True,
        **changes,
    }


def _render(**changes):
    options = {
        "as_of": date(2026, 9, 4),
        "plan_date": date(2026, 9, 7),
        "market_summary": "防守，等待确认",
        "holding_lines": ["🔴 优先退出：合成持仓(600001)，保护9.87"],
        "entries": [_entry()],
        "cash_weight": 0.5,
        "status_note": "卖出确认后再择机补位",
    }
    return render_continuous_report(**(options | changes))


def test_one_portfolio_uses_account_weights_and_keeps_risk_conditions():
    body = _render()
    assert "2026-09-07 次日交易计划" in body
    assert body.index("优先退出") < body.index("市场：") < body.index("条件新买")
    assert "合成测试(600000)｜总资金12.5%" in body
    assert "确认≥10.01，买≤10.30+量｜保护9.4812" in body
    assert "计划现金：总资金50%" in body
    assert "卖出确认后再择机补位" in body
    assert "仅信号退出，不设到期卖出" in body
    assert "不保证收益" in body
    for absent in ("六期限", "重合审计", "股票仓内", "一个月", "三个月"):
        assert absent not in body


@pytest.mark.parametrize("qualification", [False, None])
def test_observation_entries_never_become_new_buys(qualification):
    entry = _entry(name="不能透露的观察股", entry_qualified=False)
    if qualification is None:
        del entry["entry_qualified"]
    body = _render(entries=[entry], cash_weight=None, status_note="数据不足，未形成新计划")
    assert "不能透露的观察股" not in body
    assert "12.5%" not in body
    assert "10.30" not in body
    assert "暂不新买" in body
    assert "数据不足，未形成新计划" in body
    assert "未核定（不代表空仓）" in body


@pytest.mark.parametrize("qualification", [1, 0, "true", "false", None])
def test_entry_gate_requires_actual_boolean(qualification):
    with pytest.raises(TypeError, match="entry_qualified"):
        _render(entries=[_entry(entry_qualified=qualification)])


@pytest.mark.parametrize(
    "field,value",
    [
        ("entry_label", ""),
        ("entry_label", None),
        ("protection_line", None),
        ("protection_line", 0),
        ("protection_line", float("nan")),
        ("account_weight", None),
        ("account_weight", True),
        ("account_weight", -0.1),
    ],
)
def test_qualified_missing_or_invalid_critical_fields_fail_closed(field, value):
    with pytest.raises((ValueError, TypeError)):
        _render(entries=[_entry(**{field: value})])


def test_undated_report_is_holding_verification_not_a_new_buy_plan():
    body = _render(plan_date=None, entries=[], cash_weight=None, holding_lines=[])
    assert "# 🪻 持仓核验" in body
    assert "次日交易计划" not in body
    assert "持仓信息未提供｜待核验" in body
    assert "暂不新买" in body
    with pytest.raises(ValueError, match="plan_date"):
        _render(plan_date=None)


def test_market_narrative_may_be_omitted_but_risk_lines_and_numbers_never_truncated():
    risk = "🔴 优先退出：合成持仓(600001)，已核验保护9.8712，不要误认成持有"
    body = _render(holding_lines=[risk], market_summary="非关键市场描述" * 800, max_bytes=1000)
    assert len(body.encode()) <= 1000
    assert "市场摘要略" in body
    assert risk in body
    assert "确认≥10.01，买≤10.30+量｜保护9.4812" in body
    assert "卖出确认后再择机补位" in body
    with pytest.raises(ValueError, match="critical information"):
        _render(holding_lines=[risk * 100], max_bytes=1000)


def test_five_entries_fit_default_budget_without_hidden_or_sensitive_fields():
    entries = [
        _entry(symbol=f"60000{index}", account_weight=0.075, cost=987654.32) for index in range(5)
    ]
    body = _render(entries=entries)
    assert body.count("总资金7.5%") == 5
    assert body.count("保护9.4812") == 5
    assert "987654.32" not in body
    assert len(body.encode()) <= 4096


def test_rejected_entries_do_not_hide_qualified_ones_or_acquire_weights():
    body = _render(
        entries=[_entry(), _entry(symbol="600002", name="观察股", entry_qualified=False)]
    )
    assert body.count("总资金12.5%") == 1
    assert "观察股" not in body
    assert "其余未过门：暂不新买" in body


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("inf"), True])
def test_invalid_cash_or_overallocated_account_fail_closed(weight):
    with pytest.raises((ValueError, TypeError)):
        _render(cash_weight=weight)
    with pytest.raises(ValueError, match="exceed"):
        _render(cash_weight=0.9)


def test_duplicate_stocks_and_past_plan_dates_fail_closed():
    with pytest.raises(ValueError, match="unique"):
        _render(entries=[_entry(), _entry(symbol="600000")])
    with pytest.raises(ValueError, match="follow"):
        _render(plan_date=date(2026, 9, 4))


def test_more_than_five_qualified_entries_fail_closed():
    with pytest.raises(ValueError, match="at most five"):
        _render(entries=[_entry(symbol=f"60000{index}", account_weight=0.05) for index in range(6)])


def test_caller_formatted_holding_bullet_is_not_duplicated():
    body = _render(holding_lines=["- 🟣 疑似破位｜待核验，不作已确认退出"])
    assert "- 🟣 疑似破位｜待核验，不作已确认退出" in body
    assert "- - " not in body


def test_arbitrary_chart_urls_rejected_and_text_cannot_embed_remote_images():
    with pytest.raises(ValueError, match="R2"):
        _render(chart_markdown="![图](https://untrusted.example/private.png)")
    body = _render(market_summary="![远程图](https://untrusted.example/chart.png)")
    assert "![远程图](" not in body
    assert r"!\[远程图\]" in body
