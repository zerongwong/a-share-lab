from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path


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

    def caption(self, body: object) -> None:
        self._record("caption", body)

    def page_link(self, page: Path, **options: object) -> None:
        self._record("page_link", f"{page}|{options}")


def rendered_copy() -> tuple[FakeUI, str]:
    page = importlib.import_module("ashare_lab.ui.pages.06_持仓监控")
    ui = FakeUI()
    page.render(ui)
    return ui, "\n".join(body for _, body in ui.messages)


def test_page_leads_with_real_time_not_ready_state() -> None:
    ui, copy = rendered_copy()
    assert ("title", "持仓实时监控") in ui.messages
    assert ("warning", "实时监控尚未就绪") in ui.messages
    assert "没有后台常驻任务" in copy
    assert "不会自动刷新价格" in copy
    assert "没有主动通知" in copy
    assert "尚未提供多只持仓" in copy
    assert "不会把延迟日线称作实时数据" in copy


def test_page_reuses_manual_single_stock_workflow_without_fake_monitoring() -> None:
    ui, copy = rendered_copy()
    assert "当前可用的是单只股票手动检查" in copy
    assert "打开单股手动检查" in copy
    assert "pages/01_" in copy
    assert "再次手动运行" in copy
    rendered_kinds = {kind for kind, _ in ui.messages}
    assert "metric" not in rendered_kinds
    assert "dataframe" not in rendered_kinds
