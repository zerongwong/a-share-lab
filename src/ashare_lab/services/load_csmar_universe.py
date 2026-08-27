"""Load a current, auditable A-share research universe from the local CSMAR catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ashare_lab.adapters.csmar_reference import CSMARReferenceData
from ashare_lab.analytics.balance_sheet_strength import (
    assess_balance_sheet_strength_snapshot,
)
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.domain.market_rules import calculate_price_band


@dataclass(frozen=True, slots=True)
class CSMARUniverseSnapshot:
    histories: dict[str, pd.DataFrame]
    metadata: dict[str, dict[str, object]]
    data_cutoff: date
    master_symbols: int
    active_symbols: int
    eligible_symbols: int
    excluded_symbols: int
    minimum_median_amount_cny: float
    price_adjustment: str = "none"
    full_day_volume_available: bool = False
    fundamental_scores_available: bool = False
    news_scores_available: bool = False
    balance_sheet_strength_available: bool = False
    balance_sheet_strength_symbols: int = 0
    balance_sheet_strength_excluded_symbols: int = 0
    balance_sheet_snapshot_retrieved_at: date | None = None
    balance_sheet_strength_reason: str = "reference_data_not_requested"
    market_index_histories: dict[str, pd.DataFrame] = field(default_factory=dict)
    reference_common_cutoff: date | None = None
    reference_warnings: tuple[str, ...] = ()


DEFAULT_CORE_INDEX_CODES = (
    "000001",
    "000300",
    "000852",
    "000905",
    "399001",
    "399006",
)


def _minimum_live_price_cutoff(decision_date: date) -> date:
    """Conservative weekday fallback until an official calendar is connected."""

    weekday = decision_date.weekday()
    if weekday == 0:  # Monday -> previous Friday
        return decision_date - timedelta(days=3)
    if weekday == 6:  # Sunday -> previous Friday
        return decision_date - timedelta(days=2)
    return decision_date - timedelta(days=1)


def _price_limit_rate(board: object) -> Decimal:
    label = str(board).strip()
    if label in {"科创板", "创业板", "star", "chinext"}:
        return Decimal("0.20")
    if label in {"北京证券交易所", "bse"}:
        return Decimal("0.30")
    return Decimal("0.10")


def _closed_at_formation_limit(frame: pd.DataFrame, board: object) -> bool:
    """Identify a price-band close; this does not claim an order was fillable."""

    latest = frame.iloc[-1]
    previous_close = pd.to_numeric(pd.Series([latest.get("prev_close")]), errors="coerce").iloc[0]
    if not np.isfinite(previous_close) and len(frame) >= 2:
        previous_close = float(frame.iloc[-2]["close"])
    latest_close = float(latest["close"])
    if not np.isfinite(previous_close) or previous_close <= 0.0:
        return False
    upper, _ = calculate_price_band(
        Decimal(str(float(previous_close))),
        _price_limit_rate(board),
    )
    rounded_close = Decimal(str(latest_close)).quantize(Decimal("0.01"))
    return rounded_close == upper


def load_csmar_universe(
    dataset_root: str | Path,
    *,
    as_of: date,
    minimum_sessions: int = 252,
    history_sessions: int = 320,
    minimum_active_symbols: int = 4_500,
    minimum_active_master_coverage: float = 0.85,
    minimum_eligible_symbols: int = 1_000,
    minimum_eligible_active_coverage: float = 0.70,
    minimum_median_amount_cny: float = 20_000_000.0,
    raw_return_outlier_limit: float = 0.45,
    reference_dataset_root: str | Path | None = None,
    decision_date: date | None = None,
    mode: Literal["live", "historical"] = "live",
    minimum_balance_sheet_coverage: float = 0.90,
    balance_sheet_minimum_group_size: int = 20,
    core_index_codes: tuple[str, ...] = DEFAULT_CORE_INDEX_CODES,
) -> CSMARUniverseSnapshot:
    """Return active stocks with sufficient recent history from local DuckDB.

    Presence on the common cutoff session is used as the current trading-status
    gate because this CSMAR export does not include a dated suspension table.
    Names carrying ST/delisting markers remain excluded.  The export is raw,
    unadjusted price data; extreme one-day raw moves are excluded to reduce the
    risk of treating an ex-right adjustment as investment performance.
    """

    if minimum_sessions < 121:
        raise ValueError("minimum_sessions must be at least 121")
    if history_sessions < minimum_sessions:
        raise ValueError("history_sessions cannot be shorter than minimum_sessions")
    if minimum_active_symbols < 4 or minimum_eligible_symbols < 3:
        raise ValueError("full-market symbol minimums are too small")
    for name, value in {
        "minimum_active_master_coverage": minimum_active_master_coverage,
        "minimum_eligible_active_coverage": minimum_eligible_active_coverage,
    }.items():
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    if minimum_median_amount_cny < 0:
        raise ValueError("minimum_median_amount_cny cannot be negative")
    if not 0 < raw_return_outlier_limit < 1:
        raise ValueError("raw_return_outlier_limit must be between zero and one")
    if mode not in {"live", "historical"}:
        raise ValueError("mode must be live or historical")
    if not 0.0 < minimum_balance_sheet_coverage <= 1.0:
        raise ValueError("minimum_balance_sheet_coverage must be in (0, 1]")
    if balance_sheet_minimum_group_size < 3:
        raise ValueError("balance_sheet_minimum_group_size must be at least three")
    if len(core_index_codes) < 3 or len(set(core_index_codes)) != len(core_index_codes):
        raise ValueError("core_index_codes must contain at least three unique codes")
    resolved_decision_date = as_of if decision_date is None else decision_date
    if resolved_decision_date < as_of:
        raise ValueError("decision_date cannot be earlier than as_of")

    root = Path(dataset_root).expanduser().resolve()
    database_path = root / "csmar.duckdb"
    if not database_path.is_file():
        raise DataUnavailableError(f"CSMAR本地数据库不存在：{database_path}")

    try:
        import duckdb
    except ImportError as exc:
        raise DataUnavailableError("缺少duckdb依赖，无法读取CSMAR本地库") from exc

    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        cutoff_value = connection.execute(
            "SELECT max(trade_date) FROM daily_bars WHERE trade_date <= ?",
            [as_of],
        ).fetchone()[0]
        if cutoff_value is None:
            raise DataUnavailableError(f"CSMAR在{as_of.isoformat()}及之前没有日线数据")
        cutoff = pd.Timestamp(cutoff_value).date()
        if mode == "live":
            minimum_live_cutoff = _minimum_live_price_cutoff(resolved_decision_date)
            if cutoff < minimum_live_cutoff:
                raise DataUnavailableError(
                    "当前研究的数据不是最新完整截面："
                    f"决策日{resolved_decision_date.isoformat()}至少需要"
                    f"{minimum_live_cutoff.isoformat()}收盘数据，实际只有{cutoff.isoformat()}。"
                    "请更新本地日线，或切换历史回放。"
                )
        master_symbols = int(
            connection.execute("SELECT count(DISTINCT symbol) FROM security_master").fetchone()[0]
        )
        active = connection.execute(
            """
            SELECT m.symbol, m.name, m.exchange, m.board,
                   coalesce(nullif(m.industry_listing_association, ''),
                            nullif(m.industry_csrc_2012, ''),
                            nullif(m.industry, ''), '未分类') AS industry,
                   coalesce(m.is_st, false) AS is_st,
                   coalesce(m.is_delisting, false) AS is_delisting,
                   m.list_date
            FROM security_master m
            INNER JOIN (
                SELECT DISTINCT symbol FROM daily_bars WHERE trade_date = ?
            ) d USING (symbol)
            ORDER BY m.symbol
            """,
            [cutoff],
        ).fetchdf()
        if active.empty:
            raise DataUnavailableError(f"CSMAR在{cutoff.isoformat()}没有可交易证券截面")
        active_symbols = int(active["symbol"].nunique())
        active_master_coverage = active_symbols / master_symbols if master_symbols else 0.0
        if (
            active_symbols < minimum_active_symbols
            or active_master_coverage < minimum_active_master_coverage
        ):
            raise DataUnavailableError(
                "CSMAR当前截面不足以称为全市场："
                f"当日有行情{active_symbols}只/证券主表{master_symbols}只，"
                f"最低要求{minimum_active_symbols}只且覆盖率"
                f"{minimum_active_master_coverage:.0%}。"
            )

        # A calendar buffer is cheaper than one query per stock.  The final
        # per-symbol tail is trimmed below, so every stock gets the same maximum
        # number of sessions without loading the full 18-million-row history.
        start = cutoff - timedelta(days=max(550, history_sessions * 2))
        bars = connection.execute(
            """
            SELECT d.symbol, d.trade_date, d.open, d.high, d.low, d.close,
                   d.prev_close, d.volume_shares, d.amount_cny, d.turnover_pct,
                   d.source, d.retrieved_at
            FROM daily_bars d
            INNER JOIN (
                SELECT DISTINCT symbol FROM daily_bars WHERE trade_date = ?
            ) current_symbols USING (symbol)
            WHERE d.trade_date BETWEEN ? AND ?
            ORDER BY d.symbol, d.trade_date
            """,
            [cutoff, start, cutoff],
        ).fetchdf()
    finally:
        connection.close()

    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    histories: dict[str, pd.DataFrame] = {}
    liquidity_values: dict[str, float] = {}
    active_by_symbol = active.set_index("symbol", drop=False)
    for symbol, group in bars.groupby("symbol", sort=False):
        frame = group.tail(history_sessions).reset_index(drop=True)
        if len(frame) < minimum_sessions:
            continue
        close = pd.to_numeric(frame["close"], errors="coerce")
        if bool(close.isna().any()) or bool((close <= 0).any()):
            continue
        raw_returns = close.pct_change(fill_method=None).dropna()
        if bool((raw_returns.abs() > raw_return_outlier_limit).any()):
            continue
        amount = pd.to_numeric(frame["amount_cny"], errors="coerce")
        median_amount = float(amount.tail(20).median())
        if not np.isfinite(median_amount) or median_amount < minimum_median_amount_cny:
            continue
        row = active_by_symbol.loc[str(symbol)]
        compact_name = str(row["name"]).replace(" ", "").upper()
        if bool(row["is_st"]) or compact_name.startswith(("ST", "*ST")):
            continue
        if bool(row["is_delisting"]) or "退" in compact_name:
            continue
        histories[str(symbol)] = frame
        liquidity_values[str(symbol)] = median_amount

    eligible_active_coverage = len(histories) / active_symbols
    if (
        len(histories) < minimum_eligible_symbols
        or eligible_active_coverage < minimum_eligible_active_coverage
    ):
        raise DataUnavailableError(
            "CSMAR通过资格门的样本不足以支持全市场研究："
            f"合格{len(histories)}只/当日有行情{active_symbols}只，"
            f"最低要求{minimum_eligible_symbols}只且覆盖率"
            f"{minimum_eligible_active_coverage:.0%}。"
        )

    liquidity = pd.Series(liquidity_values, dtype=float)
    liquidity_scores = liquidity.rank(method="average", pct=True)
    metadata: dict[str, dict[str, object]] = {}
    for symbol in histories:
        row = active_by_symbol.loc[symbol]
        metadata[symbol] = {
            "name": str(row["name"]),
            "industry": str(row["industry"]) or "未分类",
            "is_st": False,
            "is_delisting": False,
            "is_suspended": False,
            # This export has no point-in-time financial statements, licensed
            # news or market/sector sentiment factors.  Null means unavailable;
            # the portfolio scorer disables these fields rather than inventing
            # a neutral score.
            "fundamental_score": None,
            "balance_sheet_strength_score": None,
            "liquidity_score": float(liquidity_scores.loc[symbol]),
            "news_score": None,
            "market_regime_score": None,
            "sector_score": None,
            # A daily limit close is observable, but queue/fillability is not.
            # The actionable screen uses this formation flag to choose a
            # substitute instead of assuming the stock can be bought.
            "is_limit_up_at_cutoff": _closed_at_formation_limit(histories[symbol], row["board"]),
            "is_buyable_at_cutoff": None,
            "median_amount_20d_cny": liquidity_values[symbol],
            "exchange": str(row["exchange"]),
            "board": str(row["board"]),
        }

    balance_sheet_strength_available = False
    balance_sheet_strength_symbols = 0
    balance_sheet_strength_excluded_symbols = 0
    balance_sheet_snapshot_retrieved_at: date | None = None
    balance_sheet_strength_reason = "reference_data_not_requested"
    market_index_histories: dict[str, pd.DataFrame] = {}
    reference_common_cutoff: date | None = None
    reference_warnings: list[str] = []

    if reference_dataset_root is not None:
        reference = CSMARReferenceData(reference_dataset_root)
        for index_code in core_index_codes:
            market_index_histories[index_code] = reference.read_index_daily(
                index_code,
                cutoff - timedelta(days=550),
                cutoff,
            )
        declared_cutoffs = {
            pd.Timestamp(value).date()
            for frame in market_index_histories.values()
            for value in frame["common_cutoff_date"].dropna().unique()
        }
        if len(declared_cutoffs) != 1:
            raise DataQualityError("CSMAR核心指数没有唯一的共同截止日")
        reference_common_cutoff = next(iter(declared_cutoffs))
        if reference_common_cutoff < cutoff:
            raise DataQualityError("CSMAR核心指数参考库落后于个股共同截止日")

        balance = reference.read_balance_sheet_snapshot()
        retrieved = pd.to_datetime(balance["retrieved_at"], errors="coerce").dt.date
        retrieved_values = set(retrieved.dropna().unique())
        if len(retrieved_values) != 1 or bool(retrieved.isna().any()):
            raise DataQualityError("CSMAR资产负债表快照没有唯一、完整的取得日期")
        balance_sheet_snapshot_retrieved_at = next(iter(retrieved_values))
        safe_live_date = balance_sheet_snapshot_retrieved_at + timedelta(days=1)

        if mode == "historical":
            balance_sheet_strength_reason = "historical_mode_rejects_current_snapshot"
        elif resolved_decision_date < safe_live_date:
            balance_sheet_strength_reason = "decision_precedes_safe_current_snapshot_use"
        elif cutoff < balance_sheet_snapshot_retrieved_at:
            balance_sheet_strength_reason = "price_cutoff_precedes_current_snapshot_retrieval"
        else:
            eligible_balance = balance.loc[balance["symbol"].astype(str).isin(histories)].copy()
            assessed = assess_balance_sheet_strength_snapshot(
                eligible_balance,
                minimum_group_size=balance_sheet_minimum_group_size,
            )
            available = assessed.loc[assessed["available"]].copy()
            available_symbols = set(available["symbol"].astype(str))
            coverage = len(available_symbols) / len(histories)
            if coverage < minimum_balance_sheet_coverage:
                balance_sheet_strength_reason = (
                    "balance_sheet_strength_coverage_below_minimum:"
                    f"{len(available_symbols)}/{len(histories)}"
                )
            else:
                scores = available.set_index("symbol")["score"].astype(float).to_dict()
                missing = set(histories) - available_symbols
                for symbol in missing:
                    histories.pop(symbol, None)
                    metadata.pop(symbol, None)
                for symbol, score in scores.items():
                    if symbol in metadata:
                        metadata[symbol]["balance_sheet_strength_score"] = float(score)
                balance_sheet_strength_available = True
                balance_sheet_strength_symbols = len(histories)
                balance_sheet_strength_excluded_symbols = len(missing)
                balance_sheet_strength_reason = (
                    "current_snapshot_balance_sheet_strength_enabled;"
                    "not_complete_fundamental_quality;not_historical_pit"
                )
                reference_warnings.append(
                    "资产负债表稳健度仅为当前快照窄因子，不含利润、现金流、估值或公告时点，"
                    "不得用于历史回测。"
                )

    final_eligible_coverage = len(histories) / active_symbols
    if (
        len(histories) < minimum_eligible_symbols
        or final_eligible_coverage < minimum_eligible_active_coverage
    ):
        raise DataUnavailableError(
            "参考证据合并后样本不足以支持全市场研究："
            f"合格{len(histories)}只/当日有行情{active_symbols}只。"
        )
    if active_symbols > master_symbols:
        raise DataQualityError("CSMAR当前截面证券数超过证券主表，数据库关系异常")
    return CSMARUniverseSnapshot(
        histories=histories,
        metadata=metadata,
        data_cutoff=cutoff,
        master_symbols=master_symbols,
        active_symbols=active_symbols,
        eligible_symbols=len(histories),
        excluded_symbols=active_symbols - len(histories),
        minimum_median_amount_cny=minimum_median_amount_cny,
        balance_sheet_strength_available=balance_sheet_strength_available,
        balance_sheet_strength_symbols=balance_sheet_strength_symbols,
        balance_sheet_strength_excluded_symbols=balance_sheet_strength_excluded_symbols,
        balance_sheet_snapshot_retrieved_at=balance_sheet_snapshot_retrieved_at,
        balance_sheet_strength_reason=balance_sheet_strength_reason,
        market_index_histories=market_index_histories,
        reference_common_cutoff=reference_common_cutoff,
        reference_warnings=tuple(reference_warnings),
    )
