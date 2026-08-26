"""Deterministic current-snapshot balance-sheet strength ranks.

The CSMAR ``FS_Combas`` extract available to this project has no ordinary
announcement date.  Consequently this module deliberately produces only a
current-snapshot cross-sectional rank.  Its output is not point-in-time
historical data and must never be joined to an earlier formation date.

This is a narrow balance-sheet-strength factor, not a complete assessment of
company quality.  It uses no growth, profitability, cash-flow, valuation,
regulatory-capital, asset-quality, announcement, or news information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

BALANCE_SHEET_STRENGTH_OUTPUT_COLUMNS: Final = (
    "symbol",
    "name",
    "report_period",
    "common_cutoff_date",
    "accounting_group",
    "available",
    "score",
    "coverage",
    "reasons",
    "data_role",
    "historical_backtest_eligible",
)

_REQUIRED_COLUMNS: Final = (
    "symbol",
    "name",
    "report_period",
    "cash_cny",
    "accounts_receivable_cny",
    "inventory_cny",
    "current_assets_cny",
    "total_assets_cny",
    "current_liabilities_cny",
    "total_liabilities_cny",
    "total_equity_cny",
    "data_role",
    "historical_backtest_eligible",
    "common_cutoff_date",
    "retrieved_at",
)

_FINANCIAL_NAME_PATTERN: Final = (
    r"(?:银行|证券|保险|财险|人寿|信托|期货|金控|金融|消费金融|融资租赁)"
)
_CURRENT_SNAPSHOT_REASON: Final = "current_snapshot_only_not_historical_point_in_time"


@dataclass(frozen=True, slots=True)
class _Metric:
    name: str
    weight: float
    higher_is_stronger: bool


_NON_FINANCIAL_METRICS: Final = (
    _Metric("capital_buffer", 0.40, True),
    _Metric("working_capital_buffer", 0.30, True),
    _Metric("cash_buffer", 0.20, True),
    _Metric("operating_asset_lockup", 0.10, False),
)
_FINANCIAL_METRICS: Final = (
    _Metric("capital_buffer", 0.75, True),
    _Metric("cash_buffer", 0.25, True),
)


def assess_balance_sheet_strength_snapshot(
    snapshot: pd.DataFrame,
    *,
    minimum_group_size: int = 20,
    maximum_report_age_days: int = 190,
    accounting_identity_tolerance: float = 0.02,
) -> pd.DataFrame:
    """Rank latest balance-sheet strength within financial/non-financial groups.

    Parameters
    ----------
    snapshot:
        A frame compatible with
        :meth:`CSMARReferenceData.read_balance_sheet_snapshot`.  If it contains
        more than one report period per security, only the latest period is
        assessed.  This selection is valid only for the current snapshot.
    minimum_group_size:
        Minimum number of valid observations required for every metric rank.
        A smaller group or metric sample produces an unavailable score.
    maximum_report_age_days:
        Maximum age of a report period relative to ``common_cutoff_date``.
    accounting_identity_tolerance:
        Maximum relative difference between total assets and liabilities plus
        equity before the row is rejected.

    Returns
    -------
    pandas.DataFrame
        One row per security. ``score`` is in ``[0, 1]`` when ``available`` is
        true, otherwise it is missing. ``coverage`` is the weight share of
        usable group-relative ranks. ``reasons`` is a deterministic tuple.

    Notes
    -----
    Financial firms are identified conservatively from explicit financial
    words in the existing ``name`` field.  A row without such a word is treated
    as non-financial only when both current assets and current liabilities are
    present; otherwise it remains unclassified and unavailable.  No external
    industry data or inferred replacement values are introduced.
    """

    if not isinstance(snapshot, pd.DataFrame):
        raise TypeError("snapshot must be a pandas DataFrame")
    if isinstance(minimum_group_size, bool) or not isinstance(minimum_group_size, int):
        raise TypeError("minimum_group_size must be an integer")
    if minimum_group_size < 3:
        raise ValueError("minimum_group_size must be at least three")
    if isinstance(maximum_report_age_days, bool) or not isinstance(maximum_report_age_days, int):
        raise TypeError("maximum_report_age_days must be an integer")
    if maximum_report_age_days < 0:
        raise ValueError("maximum_report_age_days cannot be negative")
    if not 0.0 <= accounting_identity_tolerance <= 0.10:
        raise ValueError("accounting_identity_tolerance must be between zero and 0.10")

    missing_columns = tuple(sorted(set(_REQUIRED_COLUMNS) - set(snapshot.columns)))
    if missing_columns:
        return _schema_unavailable(snapshot, missing_columns)
    if snapshot.empty:
        return _empty_result(("empty_snapshot",))

    frame = snapshot.loc[:, list(_REQUIRED_COLUMNS)].copy()
    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    frame["name"] = frame["name"].fillna("").astype(str).str.strip()
    frame["report_period"] = pd.to_datetime(frame["report_period"], errors="coerce")
    frame["common_cutoff_date"] = pd.to_datetime(frame["common_cutoff_date"], errors="coerce")
    frame["retrieved_at"] = pd.to_datetime(frame["retrieved_at"], errors="coerce")

    numeric_columns = (
        "cash_cny",
        "accounts_receivable_cny",
        "inventory_cny",
        "current_assets_cny",
        "total_assets_cny",
        "current_liabilities_cny",
        "total_liabilities_cny",
        "total_equity_cny",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values(["symbol", "report_period"], kind="stable", na_position="first")
    duplicate_latest = _conflicting_latest_symbols(frame)
    latest = frame.groupby("symbol", sort=True, dropna=False).tail(1).copy()
    latest = latest.sort_values("symbol", kind="stable").reset_index(drop=True)

    reasons: list[list[str]] = [[_CURRENT_SNAPSHOT_REASON] for _ in latest.index]
    rank_eligible = pd.Series(True, index=latest.index, dtype=bool)

    def reject(mask: pd.Series, reason: str) -> None:
        rejected = mask.fillna(True).astype(bool)
        for position in latest.index[rejected]:
            reasons[position].append(reason)
        rank_eligible.loc[rejected] = False

    reject(~latest["symbol"].str.fullmatch(r"\d{6}", na=False), "invalid_symbol")
    reject(latest["report_period"].isna(), "missing_report_period")
    reject(latest["common_cutoff_date"].isna(), "missing_common_cutoff_date")
    reject(latest["retrieved_at"].isna(), "missing_retrieved_at")
    reject(latest["data_role"].ne("current_snapshot"), "data_role_is_not_current_snapshot")
    reject(
        ~latest["historical_backtest_eligible"].map(_is_strict_false),
        "historical_backtest_eligibility_must_be_false",
    )
    reject(latest["symbol"].isin(duplicate_latest), "conflicting_latest_report_rows")

    valid_dates = latest["report_period"].notna() & latest["common_cutoff_date"].notna()
    reject(
        valid_dates & (latest["report_period"] > latest["common_cutoff_date"]),
        "report_period_after_common_cutoff",
    )
    report_age = (latest["common_cutoff_date"] - latest["report_period"]).dt.days
    reject(valid_dates & report_age.gt(maximum_report_age_days), "stale_report_period")
    reject(
        latest["retrieved_at"].notna()
        & latest["common_cutoff_date"].notna()
        & (latest["retrieved_at"] < latest["common_cutoff_date"]),
        "retrieval_precedes_common_cutoff",
    )

    explicit_financial = latest["name"].str.contains(_FINANCIAL_NAME_PATTERN, regex=True, na=False)
    has_current_structure = (
        latest["current_assets_cny"].notna() & latest["current_liabilities_cny"].notna()
    )
    accounting_group = pd.Series("unclassified", index=latest.index, dtype="object")
    accounting_group.loc[has_current_structure] = "non_financial"
    accounting_group.loc[explicit_financial] = "financial"
    reject(accounting_group.eq("unclassified"), "unclassified_accounting_group")

    assets = latest["total_assets_cny"]
    liabilities = latest["total_liabilities_cny"]
    equity = latest["total_equity_cny"]
    reject(assets.isna() | assets.le(0), "missing_or_nonpositive_total_assets")
    reject(liabilities.isna() | liabilities.lt(0), "missing_or_negative_total_liabilities")
    reject(equity.isna(), "missing_total_equity")

    identity_gap = (assets - liabilities - equity).abs() / assets.abs()
    identity_inputs_present = assets.gt(0) & liabilities.ge(0) & equity.notna()
    reject(
        identity_inputs_present & identity_gap.gt(accounting_identity_tolerance),
        "accounting_identity_outside_tolerance",
    )

    non_financial = accounting_group.eq("non_financial")
    reject(
        non_financial
        & (latest["current_assets_cny"].lt(0) | latest["current_liabilities_cny"].lt(0)),
        "negative_current_balance",
    )
    reject(
        latest["cash_cny"].notna() & latest["cash_cny"].lt(0),
        "negative_cash_balance",
    )
    reject(
        non_financial
        & (
            (latest["accounts_receivable_cny"].notna() & latest["accounts_receivable_cny"].lt(0))
            | (latest["inventory_cny"].notna() & latest["inventory_cny"].lt(0))
        ),
        "negative_operating_asset_balance",
    )

    metrics = pd.DataFrame(index=latest.index, dtype=float)
    metrics["capital_buffer"] = equity / assets
    metrics["working_capital_buffer"] = (
        latest["current_assets_cny"] - latest["current_liabilities_cny"]
    ) / assets
    metrics["cash_buffer"] = latest["cash_cny"] / assets
    metrics["operating_asset_lockup"] = (
        latest["accounts_receivable_cny"] + latest["inventory_cny"]
    ) / assets
    metrics = metrics.replace([np.inf, -np.inf], np.nan)

    score_numerator = pd.Series(0.0, index=latest.index, dtype=float)
    coverage = pd.Series(0.0, index=latest.index, dtype=float)
    minimum_metric_sample = minimum_group_size

    for group, specifications in (
        ("non_financial", _NON_FINANCIAL_METRICS),
        ("financial", _FINANCIAL_METRICS),
    ):
        group_mask = accounting_group.eq(group) & rank_eligible
        group_size = int(group_mask.sum())
        if group_size < minimum_group_size:
            for position in latest.index[accounting_group.eq(group)]:
                reasons[position].append(
                    f"{group}_group_sample_below_minimum:{group_size}<{minimum_group_size}"
                )
            continue

        for metric in specifications:
            metric_mask = group_mask & metrics[metric.name].notna()
            sample_size = int(metric_mask.sum())
            if sample_size < minimum_metric_sample:
                for position in latest.index[accounting_group.eq(group)]:
                    reasons[position].append(
                        f"{metric.name}_sample_below_minimum:{sample_size}<{minimum_metric_sample}"
                    )
                continue

            percentile = _percentile_rank(
                metrics.loc[metric_mask, metric.name],
                higher_is_stronger=metric.higher_is_stronger,
            )
            score_numerator.loc[metric_mask] += percentile * metric.weight
            coverage.loc[metric_mask] += metric.weight

            missing_metric = accounting_group.eq(group) & rank_eligible & ~metric_mask
            for position in latest.index[missing_metric]:
                reasons[position].append(f"missing_metric:{metric.name}")

    coverage = coverage.clip(lower=0.0, upper=1.0).round(12)
    complete_coverage = np.isclose(coverage.to_numpy(), 1.0, rtol=0.0, atol=1e-12)
    available = rank_eligible & pd.Series(complete_coverage, index=latest.index)
    score = pd.Series(pd.NA, index=latest.index, dtype="Float64")
    score.loc[available] = score_numerator.loc[available].clip(lower=0.0, upper=1.0).round(12)

    for position in latest.index:
        if accounting_group.loc[position] == "financial":
            reasons[position].append("financial_common_field_proxy_has_limited_scope")
        else:
            reasons[position].append("non_financial_group_cross_sectional_rank")
        if not bool(available.loc[position]):
            reasons[position].append("score_unavailable")

    result = pd.DataFrame(
        {
            "symbol": latest["symbol"],
            "name": latest["name"],
            "report_period": latest["report_period"],
            "common_cutoff_date": latest["common_cutoff_date"],
            "accounting_group": accounting_group,
            "available": available.astype(bool),
            "score": score,
            "coverage": coverage.astype(float),
            "reasons": [tuple(dict.fromkeys(items)) for items in reasons],
            "data_role": "current_snapshot",
            "historical_backtest_eligible": False,
        }
    ).loc[:, list(BALANCE_SHEET_STRENGTH_OUTPUT_COLUMNS)]
    return _set_result_metadata(result)


def _percentile_rank(values: pd.Series, *, higher_is_stronger: bool) -> pd.Series:
    count = len(values)
    if count < 2:
        raise ValueError("at least two observations are required for a percentile rank")
    ranks = values.rank(method="average", ascending=higher_is_stronger)
    return ((ranks - 1.0) / (count - 1.0)).astype(float)


def _conflicting_latest_symbols(frame: pd.DataFrame) -> set[str]:
    clean = frame.dropna(subset=["report_period"])
    if clean.empty:
        return set()
    latest_period = clean.groupby("symbol", dropna=False)["report_period"].transform("max")
    latest = clean.loc[clean["report_period"].eq(latest_period)]
    return set(latest.loc[latest.duplicated("symbol", keep=False), "symbol"].astype(str))


def _normalize_symbol(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip().upper()
    for suffix in (".SH", ".SZ", ".BJ"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def _is_strict_false(value: object) -> bool:
    return isinstance(value, (bool, np.bool_)) and not bool(value)


def _schema_unavailable(snapshot: pd.DataFrame, missing_columns: tuple[str, ...]) -> pd.DataFrame:
    reason = "missing_required_columns:" + ",".join(missing_columns)
    if "symbol" not in snapshot.columns or snapshot.empty:
        return _empty_result((reason,))

    symbols = snapshot["symbol"].map(_normalize_symbol)
    names = (
        snapshot["name"].fillna("").astype(str).str.strip()
        if "name" in snapshot.columns
        else pd.Series("", index=snapshot.index)
    )
    fallback = pd.DataFrame({"symbol": symbols, "name": names}).drop_duplicates("symbol")
    result = pd.DataFrame(
        {
            "symbol": fallback["symbol"],
            "name": fallback["name"],
            "report_period": pd.NaT,
            "common_cutoff_date": pd.NaT,
            "accounting_group": "unclassified",
            "available": False,
            "score": pd.Series(pd.NA, index=fallback.index, dtype="Float64"),
            "coverage": 0.0,
            "reasons": [(reason, "score_unavailable") for _ in fallback.index],
            "data_role": "current_snapshot",
            "historical_backtest_eligible": False,
        }
    ).loc[:, list(BALANCE_SHEET_STRENGTH_OUTPUT_COLUMNS)]
    return _set_result_metadata(result, dataset_reasons=(reason,))


def _empty_result(dataset_reasons: tuple[str, ...]) -> pd.DataFrame:
    result = pd.DataFrame(columns=list(BALANCE_SHEET_STRENGTH_OUTPUT_COLUMNS))
    result = result.astype(
        {
            "symbol": "object",
            "name": "object",
            "accounting_group": "object",
            "available": "bool",
            "score": "Float64",
            "coverage": "float64",
            "reasons": "object",
            "data_role": "object",
            "historical_backtest_eligible": "bool",
        }
    )
    return _set_result_metadata(result, dataset_reasons=dataset_reasons)


def _set_result_metadata(
    result: pd.DataFrame,
    *,
    dataset_reasons: tuple[str, ...] = (),
) -> pd.DataFrame:
    result.attrs.update(
        {
            "factor_name": "balance_sheet_strength_current_snapshot",
            "data_role": "current_snapshot",
            "historical_backtest_eligible": False,
            "is_complete_company_quality_assessment": False,
            "available": bool(result["available"].any()) if not result.empty else False,
            "reasons": dataset_reasons,
            "method": (
                "latest current-snapshot balance sheet; conservative financial/non-financial "
                "grouping; within-group percentile ranks; no historical PIT use"
            ),
        }
    )
    return result
