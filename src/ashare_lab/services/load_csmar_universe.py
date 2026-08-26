"""Load a current, auditable A-share research universe from the local CSMAR catalog."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

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
    minimum_median_amount_cny: float = 20_000_000.0,
    raw_return_outlier_limit: float = 0.45,
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
    if minimum_median_amount_cny < 0:
        raise ValueError("minimum_median_amount_cny cannot be negative")
    if not 0 < raw_return_outlier_limit < 1:
        raise ValueError("raw_return_outlier_limit must be between zero and one")

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

    if len(histories) < 4:
        raise DataUnavailableError(f"CSMAR当前只有{len(histories)}只通过全市场资格门")

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

    active_symbols = int(active["symbol"].nunique())
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
    )
