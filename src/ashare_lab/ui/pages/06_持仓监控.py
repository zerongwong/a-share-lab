"""Local current-holding ledger and honest close-based review status."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PAGE_TITLE = "我的持仓与每日修枝"
STATUS = "收盘后持仓复核可用；盘中实时监控尚未接通"
AVAILABLE_NOW = (
    "你可以在本机保存整组持仓和计划周期；除非再次明确替换或清空，"
    "这份持仓声明会持续有效。每日复核只使用已验证完整收盘，不会因候选排名变化自动换股。"
)
NOT_READY = (
    "盘中实时行情、到价提醒和券商连接仍未接通。",
    "研究结果只会提示持有、收紧、减仓、退出或复核，不会自动下单。",
    "退出或减仓后的资金默认留在现金，不会自动补入新股票。",
    "没有授权实时行情时，不会把延迟日线称作实时数据。",
)
PRIVACY_NOTICE = (
    "持仓、成本和权重默认只保存在这台Mac的本地数据库。"
    "不会自动写入GitHub、日志或微信/Bark/邮件；持仓摘要外发默认关闭，"
    "必须分别勾选Server酱或Bark。"
)
CN = ZoneInfo("Asia/Shanghai")


def _manual_check_path() -> Path:
    matches = tuple(Path(__file__).resolve().parent.glob("01_*.py"))
    if len(matches) != 1:
        raise RuntimeError("无法定位单股手动检查页面")
    return Path("pages") / matches[0].name


def render(ui: Any | None = None) -> None:
    """Render with an injectable UI so status claims remain regression-tested."""

    if ui is None:
        import streamlit as ui  # type: ignore[no-redef]

    ui.title(PAGE_TITLE)
    ui.warning(STATUS)
    ui.info(AVAILABLE_NOW)
    ui.caption(PRIVACY_NOTICE)

    if hasattr(ui, "file_uploader"):
        _render_local_ledger(ui)

    ui.subheader("边界说明")
    ui.markdown("\n".join(f"- {item}" for item in NOT_READY))

    ui.subheader("还可以做单股手动检查")
    ui.markdown(
        "输入一只股票代码、分析截止日和持仓成本，可查看不同持有期的计划区、"
        "减仓区和结构失效位。盘中仍需手动运行；本页不会假装实时盯盘。"
    )
    ui.page_link(
        _manual_check_path(),
        label="打开单股手动检查",
        icon="📋",
        width="stretch",
    )
    ui.caption("仅用于个人研究；不自动下单，不承诺收益，也不保证止损能够成交。")


def _render_local_ledger(ui: Any) -> None:
    from ashare_lab.bootstrap import build_repository
    from ashare_lab.services.holding_ledger import (
        HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY,
        HoldingPositionInput,
        clear_active_holdings,
        get_active_holding_portfolio,
        holding_chart_delivery_channels,
        holding_chart_publisher_id,
        holding_summary_delivery_channels,
        replace_active_holdings,
    )

    repository = build_repository()
    current = get_active_holding_portfolio(repository)
    ui.subheader("本机当前持仓")
    if current is None or current.status != "active":
        ui.info("目前没有有效的持仓声明。")
    else:
        ui.markdown(f"**计划周期：{current.holding_weeks}周｜本机版本：{current.version}**")
        allowed_channels = holding_summary_delivery_channels(current)
        ui.caption(
            "持仓摘要外发："
            + ("、".join(sorted(allowed_channels)) if allowed_channels else "未授权任何通道")
        )
        chart_channels = holding_chart_delivery_channels(current)
        chart_publisher = holding_chart_publisher_id(current)
        ui.caption(
            "持仓K线图外发："
            + (
                "仅Server酱 · Cloudflare R2"
                if chart_channels == frozenset({"serverchan"})
                and chart_publisher == "cloudflare_r2"
                else "未授权"
            )
        )
        ui.dataframe(
            [
                {
                    "代码": item.symbol,
                    "名称": item.name,
                    "入场日": item.entry_date.isoformat(),
                    "成本": item.cost_price,
                    "股票仓内": f"{item.stock_sleeve_weight:.0%}",
                    "总资金": (
                        "未知" if item.account_weight is None else f"{item.account_weight:.0%}"
                    ),
                }
                for item in current.positions
            ],
            hide_index=True,
            width="stretch",
        )

        ui.subheader("持仓K线图单独授权")
        ui.warning(
            "持仓文字摘要授权 ≠ 持仓K线图授权。本处只允许Server酱，"
            "不会开启Bark图片，也不会改动现有Server酱/Bark文字摘要授权。"
        )
        ui.caption(
            "启用后，Cloudflare R2只使用私有存储桶，图片对象必须在1日内自动删除；"
            "Server酱及其下游微信服务会接触最长1小时的签名HTTPS地址，"
            "并可能在到期前拉取或缓存图片。保存授权本身不会立即上传或发送。"
        )
        chart_choice = ui.selectbox(
            "持仓图外发范围",
            options=("disabled", "serverchan"),
            index=(
                1
                if chart_channels == frozenset({"serverchan"})
                and chart_publisher == "cloudflare_r2"
                else 0
            ),
            format_func=lambda value: {
                "disabled": "不外发（默认）",
                "serverchan": "仅Server酱（Cloudflare R2私有发布）",
            }[value],
            key=f"holding_chart_delivery_choice_v{current.version}",
        )
        chart_confirmed = ui.checkbox(
            "我确认上述第三方暴露范围、1小时签名地址和1日自动删除要求",
            key=f"holding_chart_delivery_confirm_v{current.version}",
        )
        if ui.button(
            "保存持仓图授权（生成新的持仓版本）",
            type="primary",
            disabled=not chart_confirmed,
            key=f"save_holding_chart_delivery_v{current.version}",
        ):
            try:
                _save_chart_delivery_authorization(
                    repository,
                    current,
                    allow_serverchan=chart_choice == "serverchan",
                )
                ui.success("持仓图授权已保存为新版本；本次操作没有上传或发送图片。")
                ui.rerun()
            except ValueError:
                ui.error("未保存：当前持仓已变化，请刷新页面后重新确认。")
            except Exception as exc:  # noqa: BLE001 - nontechnical UI boundary
                ui.error(f"未保存持仓图授权（{type(exc).__name__}）。")

    ui.subheader("明确更新整组持仓")
    ui.caption("上传UTF-8 JSON；成本和总资金权重可写null，系统不会猜。股票仓内权重必须合计100%。")
    uploaded = ui.file_uploader("选择本机持仓JSON", type=["json"], key="holding_json")
    weeks = ui.selectbox(
        "计划持有周期",
        options=(1, 2, 4, 13, 26, 52),
        index=2,
        format_func=lambda value: {
            1: "1周",
            2: "2周",
            4: "1个月",
            13: "3个月",
            26: "6个月",
            52: "1年",
        }[value],
    )
    confirmed = ui.checkbox("我确认这是完整的当前持仓，将整组替换本机记录")
    ui.caption(
        "下面两个选项默认不勾选。授权后也只外发代码/名称、周期、动作、保护线和简短理由；"
        "成本、总金额及账户权重永不外发。"
    )
    allow_serverchan = ui.checkbox(
        "允许后续Server酱晚报包含持仓摘要",
        key="holding_summary_serverchan_consent",
    )
    allow_bark = ui.checkbox(
        "允许后续Bark晚报包含持仓摘要",
        key="holding_summary_bark_consent",
    )
    if ui.button("保存本机持仓", type="primary", disabled=not confirmed):
        try:
            payload = json.loads(uploaded.getvalue().decode("utf-8")) if uploaded else None
            rows = payload.get("positions") if isinstance(payload, dict) else payload
            if not isinstance(rows, list) or not rows:
                raise ValueError("持仓JSON为空")
            positions = tuple(
                HoldingPositionInput(
                    symbol=str(row.get("symbol", "")),
                    name=str(row.get("name", "")),
                    entry_date=datetime.strptime(row["entry_date"], "%Y-%m-%d").date(),
                    cost_price=_optional_number(row.get("cost_price")),
                    stock_sleeve_weight=float(row["stock_sleeve_weight"]),
                    account_weight=_optional_number(row.get("account_weight")),
                    source="user_confirmed_local_ui",
                    metadata=_company_action_metadata(row),
                )
                for row in rows
            )
            replace_active_holdings(
                repository,
                positions,
                holding_weeks=weeks,
                effective_at=datetime.now(CN),
                source="user_confirmed_local_ui",
                metadata={
                    HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY: [
                        channel
                        for channel, allowed in (
                            ("serverchan", allow_serverchan),
                            ("bark", allow_bark),
                        )
                        if allowed
                    ]
                },
            )
            ui.success("已保存到本机。后续会持续沿用，直到你再次明确修改。")
            ui.rerun()
        except Exception as exc:  # noqa: BLE001 - nontechnical UI boundary
            ui.error(f"未保存，请检查JSON内容（{type(exc).__name__}）。")

    clear_confirmed = ui.checkbox("我确认当前已经没有持仓", key="clear_holdings_confirm")
    if ui.button("清空本机持仓", disabled=not clear_confirmed):
        try:
            clear_active_holdings(
                repository,
                effective_at=datetime.now(CN),
                source="user_confirmed_local_ui",
                metadata={HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY: []},
            )
            ui.success("已明确记录当前无持仓。")
            ui.rerun()
        except Exception as exc:  # noqa: BLE001 - nontechnical UI boundary
            ui.error(f"未清空（{type(exc).__name__}）。")


def _optional_number(value: object) -> float | None:
    return None if value is None or value == "" else float(value)


def _save_chart_delivery_authorization(
    repository: Any,
    current: Any,
    *,
    allow_serverchan: bool,
    effective_at: datetime | None = None,
) -> Any:
    """Copy the whole displayed snapshot and CAS only its chart consent."""

    from ashare_lab.services.holding_ledger import (
        HOLDING_CHART_DELIVERY_CHANNELS_KEY,
        HOLDING_CHART_PUBLISHER_ID_KEY,
        HoldingPositionInput,
        replace_active_holdings,
    )

    if not isinstance(allow_serverchan, bool):
        raise TypeError("allow_serverchan must be a bool")
    metadata = dict(current.metadata)
    metadata[HOLDING_CHART_DELIVERY_CHANNELS_KEY] = ["serverchan"] if allow_serverchan else []
    metadata[HOLDING_CHART_PUBLISHER_ID_KEY] = "cloudflare_r2" if allow_serverchan else None
    positions = tuple(
        HoldingPositionInput(
            symbol=item.symbol,
            name=item.name,
            entry_date=item.entry_date,
            cost_price=item.cost_price,
            stock_sleeve_weight=item.stock_sleeve_weight,
            account_weight=item.account_weight,
            source=item.source,
            metadata=dict(item.metadata),
        )
        for item in current.positions
    )
    return replace_active_holdings(
        repository,
        positions,
        holding_weeks=current.holding_weeks,
        effective_at=effective_at or datetime.now(CN),
        source="user_confirmed_chart_delivery_authorization",
        metadata=metadata,
        expected_current_revision_id=current.id,
        expected_current_version=current.version,
    )


def _company_action_metadata(row: dict[str, object]) -> dict[str, object]:
    """Accept dated evidence only when every field is explicitly supplied."""

    clear = row.get("company_action_clear")
    through = row.get("company_action_clear_through")
    source = str(row.get("company_action_evidence_source", "")).strip()
    evidence_id = str(row.get("company_action_evidence_id", "")).strip()
    if clear is None and through in (None, "") and not source and not evidence_id:
        return {}
    if not isinstance(clear, bool) or through in (None, "") or not source or not evidence_id:
        raise ValueError("公司行动核验信息不完整")
    through_date = datetime.strptime(str(through), "%Y-%m-%d").date()
    return {
        "company_action_clear": clear,
        "company_action_clear_through": through_date.isoformat(),
        "company_action_evidence_source": source,
        "company_action_evidence_id": evidence_id,
    }


if __name__ == "__main__":
    render()
