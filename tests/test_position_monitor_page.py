from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.services.holding_ledger import (
    HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY,
    HoldingPositionInput,
    holding_chart_delivery_channels,
    holding_chart_publisher_id,
    holding_summary_delivery_channels,
    replace_active_holdings,
)


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


def test_page_leads_with_close_based_local_holding_state() -> None:
    ui, copy = rendered_copy()
    assert ("title", "我的持仓与每日修枝") in ui.messages
    assert ("warning", "收盘后持仓复核可用；盘中实时监控尚未接通") in ui.messages
    assert "除非再次明确替换或清空" in copy
    assert "不会因候选排名变化自动换股" in copy
    assert "不会自动下单" in copy
    assert "默认只保存在这台Mac" in copy
    assert "持仓摘要外发默认关闭" in copy
    assert "必须分别勾选Server酱或Bark" in copy
    assert "不会把延迟日线称作实时数据" in copy


def test_page_reuses_manual_single_stock_workflow_without_fake_monitoring() -> None:
    ui, copy = rendered_copy()
    assert "还可以做单股手动检查" in copy
    assert "打开单股手动检查" in copy
    assert "pages/01_" in copy
    assert "盘中仍需手动运行" in copy
    rendered_kinds = {kind for kind, _ in ui.messages}
    assert "metric" not in rendered_kinds
    assert "dataframe" not in rendered_kinds


def test_page_keeps_chart_consent_separate_and_serverchan_only() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "ashare_lab"
        / "ui"
        / "pages"
        / "06_持仓监控.py"
    ).read_text(encoding="utf-8")

    assert "持仓文字摘要授权 ≠ 持仓K线图授权" in source
    assert "只允许Server酱" in source
    assert "不会开启Bark图片" in source
    assert "不会改动现有Server酱/Bark文字摘要授权" in source
    assert "私有存储桶" in source
    assert "1日内自动删除" in source
    assert "最长1小时的签名HTTPS地址" in source
    assert "本次操作没有上传或发送图片" in source
    assert 'options=("disabled", "serverchan")' in source
    assert '"cloudflare_r2" if allow_serverchan else None' in source
    assert "expected_current_revision_id=current.id" in source
    assert "expected_current_version=current.version" in source


@pytest.fixture
def repository(tmp_path: Path) -> SQLiteRepository:
    result = SQLiteRepository(
        tmp_path / "research.db",
        Path(__file__).resolve().parents[1] / "migrations",
    )
    result.initialize()
    return result


def test_chart_authorization_copies_positions_and_preserves_text_channels(
    repository: SQLiteRepository,
) -> None:
    page = importlib.import_module("ashare_lab.ui.pages.06_持仓监控")
    original = replace_active_holdings(
        repository,
        (
            HoldingPositionInput(
                symbol="600919",
                name="江苏银行",
                entry_date=date(2026, 8, 28),
                cost_price=12.34,
                stock_sleeve_weight=0.6,
                account_weight=0.3,
                metadata={"company_action_evidence_id": "keep-position-metadata"},
            ),
            HoldingPositionInput(
                symbol="601919",
                name="中远海控",
                entry_date=date(2026, 8, 28),
                stock_sleeve_weight=0.4,
            ),
        ),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 28, 21, tzinfo=UTC),
        metadata={
            HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY: ["serverchan", "bark"],
            "unrelated_portfolio_metadata": "keep-me",
        },
    )

    enabled = page._save_chart_delivery_authorization(
        repository,
        original,
        allow_serverchan=True,
        effective_at=datetime(2026, 9, 1, 9, tzinfo=UTC),
    )

    assert enabled.version == original.version + 1
    assert holding_summary_delivery_channels(enabled) == frozenset({"serverchan", "bark"})
    assert holding_chart_delivery_channels(enabled) == frozenset({"serverchan"})
    assert holding_chart_publisher_id(enabled) == "cloudflare_r2"
    assert enabled.metadata["unrelated_portfolio_metadata"] == "keep-me"
    assert [
        (
            item.symbol,
            item.entry_date,
            item.cost_price,
            item.stock_sleeve_weight,
            item.account_weight,
            dict(item.metadata),
        )
        for item in enabled.positions
    ] == [
        (
            item.symbol,
            item.entry_date,
            item.cost_price,
            item.stock_sleeve_weight,
            item.account_weight,
            dict(item.metadata),
        )
        for item in original.positions
    ]

    disabled = page._save_chart_delivery_authorization(
        repository,
        enabled,
        allow_serverchan=False,
        effective_at=datetime(2026, 9, 1, 10, tzinfo=UTC),
    )
    assert holding_summary_delivery_channels(disabled) == frozenset({"serverchan", "bark"})
    assert holding_chart_delivery_channels(disabled) == frozenset()
    assert holding_chart_publisher_id(disabled) is None


def test_chart_authorization_copy_fails_closed_after_concurrent_holding_change(
    repository: SQLiteRepository,
) -> None:
    page = importlib.import_module("ashare_lab.ui.pages.06_持仓监控")
    stale = replace_active_holdings(
        repository,
        (
            HoldingPositionInput(
                symbol="600919",
                name="江苏银行",
                entry_date=date(2026, 8, 28),
                stock_sleeve_weight=1.0,
            ),
        ),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 28, 21, tzinfo=UTC),
    )
    current = replace_active_holdings(
        repository,
        (
            HoldingPositionInput(
                symbol="601919",
                name="中远海控",
                entry_date=date(2026, 8, 29),
                stock_sleeve_weight=1.0,
            ),
        ),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 29, 21, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="changed; reload"):
        page._save_chart_delivery_authorization(
            repository,
            stale,
            allow_serverchan=True,
            effective_at=datetime(2026, 9, 1, 9, tzinfo=UTC),
        )

    from ashare_lab.services.holding_ledger import get_active_holding_portfolio

    assert get_active_holding_portfolio(repository) == current
