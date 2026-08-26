from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ashare_lab.analytics.balance_sheet_strength import (
    assess_balance_sheet_strength_snapshot,
)


def _row(
    symbol: str,
    *,
    name: str,
    assets: float,
    liabilities: float,
    cash: float,
    current_assets: float | None,
    current_liabilities: float | None,
    receivables: float | None,
    inventory: float | None,
    report_period: str = "2026-06-30",
    data_role: str = "current_snapshot",
    historical_backtest_eligible: bool = False,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "name": name,
        "report_period": report_period,
        "cash_cny": cash,
        "accounts_receivable_cny": receivables,
        "inventory_cny": inventory,
        "current_assets_cny": current_assets,
        "total_assets_cny": assets,
        "current_liabilities_cny": current_liabilities,
        "total_liabilities_cny": liabilities,
        "total_equity_cny": assets - liabilities,
        "data_role": data_role,
        "historical_backtest_eligible": historical_backtest_eligible,
        "common_cutoff_date": "2026-08-24",
        "retrieved_at": "2026-08-25",
    }


def _non_financial_rows(count: int = 4) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        strength = index + 1
        assets = 100.0
        liabilities = 82.0 - 12.0 * strength
        rows.append(
            _row(
                f"{600000 + index}",
                name=f"样本制造{index}",
                assets=assets,
                liabilities=liabilities,
                cash=4.0 + 4.0 * strength,
                current_assets=35.0 + 6.0 * strength,
                current_liabilities=43.0 - 5.0 * strength,
                receivables=18.0 - 2.0 * strength,
                inventory=16.0 - 2.0 * strength,
            )
        )
    return rows


def _financial_rows(count: int = 3) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        strength = index + 1
        assets = 1_000.0
        liabilities = 950.0 - 20.0 * strength
        rows.append(
            _row(
                f"{601000 + index}",
                name=f"样本银行{index}",
                assets=assets,
                liabilities=liabilities,
                cash=60.0 + 20.0 * strength,
                current_assets=None,
                current_liabilities=None,
                receivables=None,
                inventory=None,
            )
        )
    return rows


def _frame(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_non_financial_strength_is_a_zero_to_one_group_rank() -> None:
    raw_result = assess_balance_sheet_strength_snapshot(
        _frame(_non_financial_rows()), minimum_group_size=4
    )
    assert raw_result.attrs["factor_name"] == "balance_sheet_strength_current_snapshot"
    assert raw_result.attrs["historical_backtest_eligible"] is False
    assert raw_result.attrs["is_complete_company_quality_assessment"] is False
    result = raw_result.set_index("symbol")

    assert result["available"].all()
    assert result["coverage"].eq(1.0).all()
    assert result["score"].between(0.0, 1.0, inclusive="both").all()
    assert result.loc["600000", "score"] == 0.0
    assert result.loc["600003", "score"] == 1.0
    assert result["accounting_group"].eq("non_financial").all()
    assert not bool(result["historical_backtest_eligible"].any())


def test_financial_and_non_financial_firms_are_ranked_separately() -> None:
    rows = [*_non_financial_rows(3), *_financial_rows(3)]
    result = assess_balance_sheet_strength_snapshot(_frame(rows), minimum_group_size=3)
    financial = result.loc[result["accounting_group"].eq("financial")].set_index("symbol")
    non_financial = result.loc[result["accounting_group"].eq("non_financial")]

    assert financial["available"].all()
    assert non_financial["available"].all()
    assert financial.loc["601000", "score"] == 0.0
    assert financial.loc["601002", "score"] == 1.0
    assert all(
        "financial_common_field_proxy_has_limited_scope" in reasons
        for reasons in financial["reasons"]
    )


def test_a_missing_required_value_makes_only_that_security_unavailable() -> None:
    rows = _non_financial_rows(4)
    rows[0]["inventory_cny"] = None
    result = assess_balance_sheet_strength_snapshot(_frame(rows), minimum_group_size=3).set_index(
        "symbol"
    )

    assert not bool(result.loc["600000", "available"])
    assert pd.isna(result.loc["600000", "score"])
    assert result.loc["600000", "coverage"] == 0.9
    assert "missing_metric:operating_asset_lockup" in result.loc["600000", "reasons"]
    assert result.loc["600003", "available"]


def test_missing_schema_returns_an_explicit_unavailable_result() -> None:
    snapshot = _frame(_non_financial_rows(3)).drop(columns="total_equity_cny")
    result = assess_balance_sheet_strength_snapshot(snapshot, minimum_group_size=3)

    assert len(result) == 3
    assert not bool(result["available"].any())
    assert result["score"].isna().all()
    assert result["coverage"].eq(0.0).all()
    assert result.attrs["available"] is False
    assert "missing_required_columns:total_equity_cny" in result.attrs["reasons"]


def test_small_group_never_invents_a_rank() -> None:
    result = assess_balance_sheet_strength_snapshot(
        _frame(_non_financial_rows(3)), minimum_group_size=4
    )

    assert not bool(result["available"].any())
    assert result["score"].isna().all()
    assert result["coverage"].eq(0.0).all()
    assert all(
        "non_financial_group_sample_below_minimum:3<4" in reasons for reasons in result["reasons"]
    )


def test_non_snapshot_metadata_is_rejected_and_never_becomes_historical_pit() -> None:
    rows = _non_financial_rows(3)
    rows[0]["data_role"] = "historical_point_in_time"
    rows[0]["historical_backtest_eligible"] = True
    result = assess_balance_sheet_strength_snapshot(_frame(rows), minimum_group_size=3).set_index(
        "symbol"
    )

    assert not bool(result.loc["600000", "available"])
    assert pd.isna(result.loc["600000", "score"])
    assert "data_role_is_not_current_snapshot" in result.loc["600000", "reasons"]
    assert "historical_backtest_eligibility_must_be_false" in result.loc["600000", "reasons"]
    assert result.loc["600000", "data_role"] == "current_snapshot"
    assert not bool(result.loc["600000", "historical_backtest_eligible"])


def test_latest_period_is_used_only_as_a_current_snapshot() -> None:
    rows = _non_financial_rows(3)
    older = dict(rows[0])
    older["report_period"] = "2026-03-31"
    older["cash_cny"] = 1.0
    result = assess_balance_sheet_strength_snapshot(
        _frame([older, *reversed(rows)]), minimum_group_size=3
    ).set_index("symbol")

    assert len(result) == 3
    assert result.loc["600000", "report_period"] == pd.Timestamp("2026-06-30")
    assert _CURRENT_REASON in result.loc["600000", "reasons"]


def test_stale_or_unclassified_rows_are_unavailable() -> None:
    rows = _non_financial_rows(3)
    rows[0]["report_period"] = "2025-12-31"
    rows[1]["current_assets_cny"] = None
    rows[1]["current_liabilities_cny"] = None
    result = assess_balance_sheet_strength_snapshot(_frame(rows), minimum_group_size=3).set_index(
        "symbol"
    )

    assert not bool(result.loc["600000", "available"])
    assert "stale_report_period" in result.loc["600000", "reasons"]
    assert not bool(result.loc["600001", "available"])
    assert "unclassified_accounting_group" in result.loc["600001", "reasons"]


_CURRENT_REASON = "current_snapshot_only_not_historical_point_in_time"
