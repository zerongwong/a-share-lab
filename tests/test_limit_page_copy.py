from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class FakeUI:
    messages: list[tuple[str, str]] = field(default_factory=list)

    def _record(self, kind: str, body: object) -> None:
        self.messages.append((kind, str(body)))

    def title(self, body: object) -> None:
        self._record("title", body)

    def warning(self, body: object) -> None:
        self._record("warning", body)

    def info(self, body: object) -> None:
        self._record("info", body)

    def subheader(self, body: object) -> None:
        self._record("subheader", body)

    def markdown(self, body: object) -> None:
        self._record("markdown", body)

    @contextmanager
    def expander(self, label: str, *, expanded: bool) -> Iterator[FakeUI]:
        self._record("expander", f"{label}|expanded={expanded}")
        yield self


def load_page():
    return importlib.import_module("ashare_lab.ui.pages.03_涨停实验室")


def rendered_copy() -> tuple[FakeUI, str]:
    page = load_page()
    ui = FakeUI()
    page.render(ui)
    return ui, "\n".join(body for _, body in ui.messages)


def test_page_is_disabled_and_lists_every_activation_prerequisite() -> None:
    ui, copy = rendered_copy()
    assert ("title", "尾盘五强") in ui.messages
    assert ("warning", "当前未启用") in ui.messages
    assert "授权实时行情" in copy
    assert "逐笔/L2数据" in copy
    assert "延迟不超过3秒" in copy
    assert "滚动样本外回测" in copy
    assert "概率校准" in copy
    assert "数据授权" in copy
    assert "14:49:55冻结" in copy


def test_first_screen_is_a_simple_timeline_and_output_contract() -> None:
    ui, copy = rendered_copy()
    assert "14:40预检 → 14:45扫描 → 14:50输出0–5只" in copy
    assert "排名" in copy
    assert "综合分" in copy
    assert "连板概率区间" in copy
    assert "主要风险" in copy
    assert "今日无候选" in copy

    expanders = [body for kind, body in ui.messages if kind == "expander"]
    assert expanders == [
        "启用前必须完成|expanded=False",
        "资金与交易安全边界|expanded=False",
        "评分依据与完整时间线|expanded=False",
        "当前不会做什么|expanded=False",
    ]


def test_copy_distinguishes_capital_cap_from_stop_and_forbids_financing() -> None:
    _, copy = rendered_copy()
    assert "5%是该高风险试验模块的资金仓上限，不是止损线" in copy
    assert "融资比例固定为0" in copy
    assert "每天允许输出0只，最多5只" in copy
    assert "每只股票的资金仓上限是1%" in copy
    assert "不足5只时剩余资金保留现金" in copy
    assert "同一题材最多2只" in copy
    assert "不自动下单" in copy
    assert "T+1" in copy
    assert "09:35–09:45" in copy
    assert "止损不能成交" in copy


def test_page_documents_fixed_score_timeline_and_no_system_task() -> None:
    _, copy = rendered_copy()
    assert "市场环境10分" in copy
    assert "封板行为20分" in copy
    assert "最高30分尾部风险惩罚" in copy
    assert "输入不全时该股票直接失去资格" in copy
    assert "14:40开始预检" in copy
    assert "14:44完成" in copy
    assert "14:50:05输出" in copy
    assert "14:56:45前人工撤回" in copy
    assert "14:57进入收盘集合竞价后不可撤单" in copy
    assert "不创建系统自动任务" in copy


def test_disabled_page_does_not_render_candidates_or_fake_probabilities() -> None:
    ui, copy = rendered_copy()
    rendered_kinds = {kind for kind, _ in ui.messages}
    assert "dataframe" not in rendered_kinds
    assert "metric" not in rendered_kinds
    assert "不会生成候选股票" in copy
    assert "不会展示推测概率" in copy
    assert "不展示股票候选、涨停概率或预期收益" in copy
