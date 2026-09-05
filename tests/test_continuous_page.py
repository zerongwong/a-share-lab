from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

page = importlib.import_module("ashare_lab.ui.pages.10_持续信号组合")
TODAY = date(2026, 9, 5)


@pytest.fixture(autouse=True)
def local_identity_fake(monkeypatch):
    monkeypatch.setattr(page, "_current_holding_identity_matches", lambda _view: True)


@dataclass
class FakeUI:
    buttons: set[str] = field(default_factory=set)
    session_state: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)

    def button(self, label, **_kwargs):
        return label in self.buttons

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.messages.append((name, args, kwargs))


def _view(pending=False):
    digest = SimpleNamespace(
        common_cutoff=date(2026, 9, 4),
        cycle_label="防守观察",
        continuous_plan={
            "entries": [
                {
                    "name": "合成甲",
                    "symbol": "600000",
                    "entry_qualified": True,
                    "account_weight": 0.2,
                    "entry_label": "确认≥10，买≤10.30+量",
                    "protection_line": 9.48,
                }
            ],
            "cash_weight": 0.8,
            "status_note": "先处理旧仓",
            "pending_exit_symbols": ["600001"] if pending else [],
        },
    )
    return page.LocalContinuousView(digest, None, Path("unused"), None)


def _text(ui):
    return "\n".join(str(args) for _kind, args, _kwargs in ui.messages)


def test_import_and_initial_page_do_not_load_data_or_call_network():
    ui = FakeUI()

    def forbidden(**_kwargs):
        pytest.fail("page load must not start data generation")

    page.render(ui, decision_date=TODAY, _view_loader=forbidden)
    assert "不发送微信" in _text(ui)
    assert "一组组合" in _text(ui)
    assert not any(kind in {"selectbox", "radio"} for kind, _args, _kwargs in ui.messages)


def test_unverified_date_shows_observation_without_buy_weights():
    ui = FakeUI(buttons={"生成持续组合"})

    def forbidden(_cutoff):
        pytest.fail("only the calendar button may request calendar data")

    page.render(
        ui,
        decision_date=TODAY,
        _view_loader=lambda **_kwargs: _view(),
        _calendar_resolver=forbidden,
    )
    text = _text(ui)
    assert "下一交易日未核验" in text
    assert "仅研究，不作买入配置" in text
    assert "总资金20%" not in text
    assert "确认≥10，买≤10.30+量" in text


def test_explicit_calendar_button_allows_future_qualified_plan():
    ui = FakeUI(buttons={"生成持续组合", "核验下一交易日"})
    page.render(
        ui,
        decision_date=TODAY,
        _view_loader=lambda **_kwargs: _view(),
        _calendar_resolver=lambda _cutoff: date(2026, 9, 7),
    )
    assert "2026-09-07 次日交易计划" in _text(ui)
    assert "总资金20%" in _text(ui)


def test_pending_exit_never_becomes_a_formal_buy_even_after_calendar_verification():
    ui = FakeUI(buttons={"生成持续组合", "核验下一交易日"})
    page.render(
        ui,
        decision_date=TODAY,
        _view_loader=lambda **_kwargs: _view(pending=True),
        _calendar_resolver=lambda _cutoff: date(2026, 9, 7),
    )
    assert "卖出尚未确认" in _text(ui)
    assert "总资金20%" not in _text(ui)


@pytest.mark.parametrize("day", [date(2026, 9, 3), date(2026, 9, 4), TODAY, None])
def test_invalid_or_expired_calendar_evidence_never_produces_buy_weights(day):
    ui = FakeUI(buttons={"生成持续组合", "核验下一交易日"})
    page.render(
        ui,
        decision_date=TODAY,
        _view_loader=lambda **_kwargs: _view(),
        _calendar_resolver=lambda _cutoff: day,
    )
    assert "总资金20%" not in _text(ui)
    assert "尚未核验" in _text(ui)


def test_local_chart_runs_only_on_its_button_and_does_not_send():
    calls = []
    ui = FakeUI(buttons={"生成持续组合"})

    def chart(view):
        calls.append(view)
        return SimpleNamespace(rendered=SimpleNamespace(composite_png=b"synthetic-png"))

    page.render(
        ui, decision_date=TODAY, _view_loader=lambda **_kwargs: _view(), _chart_loader=chart
    )
    assert calls == []
    ui.buttons = {"生成持仓日 / 周 K线图（仅本机）"}
    page.render(ui, decision_date=TODAY, _chart_loader=chart)
    assert len(calls) == 1
    assert any(kind == "image" for kind, _args, _kwargs in ui.messages)


def test_generation_error_is_sanitized_and_discards_previous_result():
    ui = FakeUI(buttons={"生成持续组合"}, session_state={page._STATE_KEY: _view()})

    def failure(**_kwargs):
        raise RuntimeError("PRIVATE_TOKEN_OR_HOLDING_DETAIL")

    page.render(ui, decision_date=TODAY, _view_loader=failure)
    assert page._STATE_KEY not in ui.session_state
    assert "PRIVATE_TOKEN_OR_HOLDING_DETAIL" not in _text(ui)
    assert "未形成新买计划" in _text(ui)


def test_changed_holding_identity_discards_the_cached_plan_before_display():
    ui = FakeUI(session_state={page._STATE_KEY: _view()})
    page.render(ui, decision_date=TODAY, _identity_checker=lambda _view: False)
    assert page._STATE_KEY not in ui.session_state
    assert "持仓版本已变化或无法核验" in _text(ui)
    assert "确认≥10" not in _text(ui)


def test_generated_view_survives_streamlit_page_class_reload_as_plain_state():
    ui = FakeUI(buttons={"生成持续组合"})
    page.render(ui, decision_date=TODAY, _view_loader=lambda **_kwargs: _view())
    assert isinstance(ui.session_state[page._STATE_KEY], dict)
    ui.buttons = {"核验下一交易日"}
    page.render(ui, decision_date=TODAY, _calendar_resolver=lambda _cutoff: date(2026, 9, 7))
    assert "总资金20%" in _text(ui)


def test_page_source_keeps_ledger_writes_and_delivery_outside_this_page():
    source = Path(page.__file__).read_text(encoding="utf-8")
    assert "persist=False" in source
    assert "archive_directory=None" in source
    for forbidden in (
        "replace_active_holdings(",
        "clear_active_holdings(",
        "publish_png(",
        ".send(",
    ):
        assert forbidden not in source
