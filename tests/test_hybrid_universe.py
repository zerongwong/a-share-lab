from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import pytest

from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.services import load_hybrid_universe as hybrid_module
from ashare_lab.services.load_csmar_universe import (
    DEFAULT_CORE_INDEX_CODES,
    CSMARUniverseSnapshot,
)
from ashare_lab.services.load_hybrid_universe import load_hybrid_universe

BASELINE_CUTOFF = date(2026, 8, 24)
STOCK_SYMBOLS = ("600000", "000002", "300750")
INFOWAY_CORE_INDEX_CODES = tuple(
    "399905" if code == "000905" else code for code in DEFAULT_CORE_INDEX_CODES
)
SMALL_UNIVERSE_GATES = {
    "minimum_eligible_symbols": 3,
    "minimum_eligible_active_coverage": 0.70,
}


@dataclass
class FakeOverlayStore:
    manifest: pd.DataFrame
    stock_days: dict[date, pd.DataFrame]
    index_days: dict[date, pd.DataFrame]

    def __post_init__(self) -> None:
        self.calls: list[tuple[date, str]] = []

    def read_verified_manifest(self, *, source_id: str | None = None) -> pd.DataFrame:
        if source_id is not None:
            assert source_id == "infoway"
        return self.manifest.copy()

    def read_verified_daily(
        self,
        trade_date: date,
        *,
        source_id: str,
        asset_kind: str,
    ) -> pd.DataFrame:
        assert source_id == "infoway"
        self.calls.append((trade_date, asset_kind))
        frames = self.stock_days if asset_kind == "stocks" else self.index_days
        return frames[trade_date].copy()


def _baseline_snapshot() -> CSMARUniverseSnapshot:
    sessions = pd.to_datetime(["2026-08-20", "2026-08-21", "2026-08-24"])
    histories: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, object]] = {}
    for offset, symbol in enumerate(STOCK_SYMBOLS):
        close = pd.Series([10.0 + offset, 10.1 + offset, 10.2 + offset])
        histories[symbol] = pd.DataFrame(
            {
                "symbol": symbol,
                "trade_date": sessions,
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "prev_close": close.shift(1).fillna(close.iloc[0] - 0.1),
                "volume_shares": 1_000_000 + offset,
                "amount_cny": 100_000_000 + offset,
                "turnover_pct": 1.0,
                "source": "CSMAR",
                "retrieved_at": pd.Timestamp("2026-08-25T00:00:00Z"),
            }
        )
        metadata[symbol] = {
            "name": f"测试{offset}",
            "industry": f"行业{offset}",
            "board": "主板" if symbol != "300750" else "创业板",
            "median_amount_20d_cny": 100_000_000.0,
            "liquidity_score": 0.5,
            "is_limit_up_at_cutoff": False,
        }

    indices: dict[str, pd.DataFrame] = {}
    for offset, code in enumerate(DEFAULT_CORE_INDEX_CODES):
        close = pd.Series([3_000.0 + offset, 3_010.0 + offset, 3_020.0 + offset])
        indices[code] = pd.DataFrame(
            {
                "index_code": code,
                "trade_date": sessions,
                "open": close - 5.0,
                "high": close + 10.0,
                "low": close - 10.0,
                "close": close,
                "component_volume": 10_000_000,
                "component_amount_cny": 1_000_000_000,
                "index_return": close.pct_change(fill_method=None),
                "knowledge_date": BASELINE_CUTOFF,
                "data_role": "market_regime_only",
                "historical_backtest_eligible": True,
                "common_cutoff_date": BASELINE_CUTOFF,
                "source": "CSMAR",
                "retrieved_at": pd.Timestamp("2026-08-25T00:00:00Z"),
            }
        )
    return CSMARUniverseSnapshot(
        histories=histories,
        metadata=metadata,
        data_cutoff=BASELINE_CUTOFF,
        master_symbols=len(STOCK_SYMBOLS),
        active_symbols=len(STOCK_SYMBOLS),
        eligible_symbols=len(STOCK_SYMBOLS),
        excluded_symbols=0,
        minimum_median_amount_cny=20_000_000.0,
        market_index_histories=indices,
        reference_common_cutoff=BASELINE_CUTOFF,
    )


def _stock_day(trade_day: date, *, include_new_stock: bool = False) -> pd.DataFrame:
    symbols = list(STOCK_SYMBOLS)
    if include_new_stock:
        symbols.append("920001")
    rows = []
    for offset, symbol in enumerate(symbols):
        close = 10.3 + offset
        rows.append(
            {
                "symbol": symbol,
                "trade_date": trade_day,
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "prev_close": close - 0.1,
                "volume_shares": 1_100_000 + offset,
                "amount_cny": 110_000_000 + offset,
                "turnover_pct": 1.1,
                "source": "infoway:eod",
                "retrieved_at": pd.Timestamp(f"{trade_day.isoformat()}T08:10:00Z"),
            }
        )
    return pd.DataFrame(rows)


def _index_day(trade_day: date) -> pd.DataFrame:
    rows = []
    for offset, code in enumerate(DEFAULT_CORE_INDEX_CODES):
        provider_code = "399905" if code == "000905" else code
        close = 3_030.0 + offset
        rows.append(
            {
                "symbol": provider_code,
                "trade_date": trade_day,
                "open": close - 5.0,
                "high": close + 10.0,
                "low": close - 10.0,
                "close": close,
                "prev_close": close - 10.0,
                "volume_shares": 11_000_000 + offset,
                "amount_cny": 1_100_000_000 + offset,
                "turnover_pct": 1.0,
                "source": "infoway:eod",
                "retrieved_at": pd.Timestamp(f"{trade_day.isoformat()}T08:10:00Z"),
            }
        )
    return pd.DataFrame(rows)


def _manifest(
    stock_days: dict[date, pd.DataFrame],
    index_days: dict[date, pd.DataFrame],
    *,
    retrieved_at: str | None = None,
    adjustment: str = "none",
) -> pd.DataFrame:
    trade_days = list(stock_days)
    previous_days: list[date] = []
    previous = BASELINE_CUTOFF
    for trade_day in trade_days:
        if trade_day <= BASELINE_CUTOFF:
            previous_days.append(date(2026, 8, 21))
        else:
            previous_days.append(previous)
        previous = trade_day
    retrieval_times = [
        retrieved_at or f"{trade_day.isoformat()}T08:30:00Z" for trade_day in trade_days
    ]
    stock_counts = [len(frame) for frame in stock_days.values()]
    return pd.DataFrame(
        {
            "source_id": "infoway",
            "adjustment": adjustment,
            "trade_date": trade_days,
            "previous_trade_date": previous_days,
            "stock_count": stock_counts,
            "expected_stock_count": stock_counts,
            "stock_coverage_ratio": 1.0,
            "index_count": [len(index_days[trade_day]) for trade_day in stock_days],
            "core_index_symbols": "|".join(INFOWAY_CORE_INDEX_CODES),
            "latest_retrieved_at": retrieval_times,
            "verified_at": retrieval_times,
            "verified": True,
        }
    )


@pytest.fixture(autouse=True)
def _replace_csmar_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hybrid_module, "load_csmar_universe", lambda *args, **kwargs: _baseline_snapshot()
    )
    monkeypatch.setattr(
        hybrid_module,
        "_read_csmar_master_symbols",
        lambda dataset_root: set(STOCK_SYMBOLS),
    )


def test_appends_only_verified_post_baseline_rows_and_advances_common_cutoff() -> None:
    overlap = BASELINE_CUTOFF
    day_one = date(2026, 8, 25)
    day_two = date(2026, 8, 26)
    stock_days = {day: _stock_day(day) for day in (overlap, day_one, day_two)}
    stock_days[overlap].loc[:, "close"] = 999.0
    index_days = {day: _index_day(day) for day in stock_days}
    store = FakeOverlayStore(_manifest(stock_days, index_days), stock_days, index_days)

    loaded = load_hybrid_universe(
        "/unused/csmar",
        overlay_root="/unused/overlay",
        overlay_source_id="infoway",
        as_of=day_two,
        decision_date=date(2026, 8, 27),
        mode="live",
        overlay_store=store,
        **SMALL_UNIVERSE_GATES,
    )

    assert loaded.historical_baseline_cutoff == overlap
    assert loaded.automatic_increment_cutoff == day_two
    assert loaded.common_cutoff == day_two
    assert loaded.overlay_trading_days == (day_one, day_two)
    assert loaded.sources == ("CSMAR只读历史基线", "infoway:eod已验证收盘增量")
    assert (overlap, "stocks") not in store.calls
    assert loaded.snapshot.data_cutoff == day_two
    assert loaded.snapshot.reference_common_cutoff == day_two
    assert loaded.snapshot.histories["600000"].iloc[2]["close"] == pytest.approx(10.2)
    assert loaded.snapshot.histories["600000"].iloc[-1]["source"] == "infoway:eod"
    assert len(loaded.snapshot.histories["600000"]) == 5
    for code, history in loaded.snapshot.market_index_histories.items():
        assert code in DEFAULT_CORE_INDEX_CODES
        assert history.iloc[-1]["source"] == "infoway:eod"
        assert pd.Timestamp(history.iloc[-1]["trade_date"]).date() == day_two
        assert set(history["common_cutoff_date"]) == {day_two}
    assert "399905" not in loaded.snapshot.market_index_histories
    assert loaded.snapshot.market_index_histories["000905"].iloc[-1]["source"] == "infoway:eod"


def test_long_risk_history_request_does_not_become_full_market_coverage_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def capture_loader(*_args: object, **kwargs: object) -> CSMARUniverseSnapshot:
        captured.append(dict(kwargs))
        return _baseline_snapshot()

    monkeypatch.setattr(hybrid_module, "load_csmar_universe", capture_loader)
    store = FakeOverlayStore(pd.DataFrame(), {}, {})
    loaded = load_hybrid_universe(
        "/unused/csmar",
        overlay_root="/unused/overlay",
        overlay_source_id="infoway",
        as_of=BASELINE_CUTOFF,
        decision_date=BASELINE_CUTOFF,
        mode="historical",
        overlay_store=store,
        minimum_sessions=2017,
        history_sessions=2087,
        **SMALL_UNIVERSE_GATES,
    )

    assert len(captured) == 1
    assert captured[0]["minimum_sessions"] == 252
    assert captured[0]["history_sessions"] == 2087
    assert set(loaded.snapshot.histories) == set(STOCK_SYMBOLS)
    assert any(
        "2017个价格点的持有期风险历史仅对各期限结构合格候选逐股核验" in warning
        for warning in loaded.snapshot.reference_warnings
    )


def test_new_stock_outside_csmar_master_is_isolated() -> None:
    trade_day = date(2026, 8, 25)
    stock_days = {trade_day: _stock_day(trade_day, include_new_stock=True)}
    index_days = {trade_day: _index_day(trade_day)}
    store = FakeOverlayStore(_manifest(stock_days, index_days), stock_days, index_days)

    loaded = load_hybrid_universe(
        "/unused/csmar",
        overlay_root="/unused/overlay",
        overlay_source_id="infoway",
        as_of=trade_day,
        decision_date=trade_day,
        mode="historical",
        overlay_store=store,
        **SMALL_UNIVERSE_GATES,
    )

    assert loaded.isolated_overlay_symbols == ("920001",)
    assert "920001" not in loaded.snapshot.histories
    assert any("已隔离" in warning for warning in loaded.snapshot.reference_warnings)


def test_missing_baseline_stock_is_excluded_when_coverage_gate_passes() -> None:
    trade_day = date(2026, 8, 25)
    stocks = _stock_day(trade_day)
    stocks = stocks.loc[stocks["symbol"] != "300750"].reset_index(drop=True)
    stock_days = {trade_day: stocks}
    index_days = {trade_day: _index_day(trade_day)}
    store = FakeOverlayStore(_manifest(stock_days, index_days), stock_days, index_days)

    loaded = load_hybrid_universe(
        "/unused/csmar",
        overlay_root="/unused/overlay",
        overlay_source_id="infoway",
        as_of=trade_day,
        decision_date=trade_day,
        mode="historical",
        overlay_store=store,
        minimum_eligible_symbols=2,
        minimum_eligible_active_coverage=0.60,
    )

    assert set(loaded.snapshot.histories) == {"600000", "000002"}
    assert loaded.snapshot.eligible_symbols == 2
    assert loaded.snapshot.excluded_symbols == 1
    assert any("未覆盖全部自动增量" in warning for warning in loaded.snapshot.reference_warnings)


def test_missing_baseline_stock_fails_when_coverage_gate_is_breached() -> None:
    trade_day = date(2026, 8, 25)
    stocks = _stock_day(trade_day)
    stocks = stocks.loc[stocks["symbol"] != "300750"].reset_index(drop=True)
    stock_days = {trade_day: stocks}
    index_days = {trade_day: _index_day(trade_day)}
    store = FakeOverlayStore(_manifest(stock_days, index_days), stock_days, index_days)

    with pytest.raises(DataUnavailableError, match="不再满足全市场研究资格门"):
        load_hybrid_universe(
                "/unused/csmar",
                overlay_root="/unused/overlay",
                overlay_source_id="infoway",
            as_of=trade_day,
            decision_date=trade_day,
            mode="historical",
            overlay_store=store,
            **SMALL_UNIVERSE_GATES,
        )


def test_missing_core_index_fails_closed() -> None:
    trade_day = date(2026, 8, 25)
    stock_days = {trade_day: _stock_day(trade_day)}
    indices = _index_day(trade_day).iloc[:-1].reset_index(drop=True)
    index_days = {trade_day: indices}
    store = FakeOverlayStore(_manifest(stock_days, index_days), stock_days, index_days)

    with pytest.raises(DataQualityError, match="六核心指数不完整"):
        load_hybrid_universe(
                "/unused/csmar",
                overlay_root="/unused/overlay",
                overlay_source_id="infoway",
            as_of=trade_day,
            decision_date=trade_day,
            mode="historical",
            overlay_store=store,
            **SMALL_UNIVERSE_GATES,
        )


def test_manifest_index_alias_collision_fails_closed() -> None:
    trade_day = date(2026, 8, 25)
    stock_days = {trade_day: _stock_day(trade_day)}
    index_days = {trade_day: _index_day(trade_day)}
    manifest = _manifest(stock_days, index_days)
    manifest.loc[:, "core_index_symbols"] = "|".join(INFOWAY_CORE_INDEX_CODES) + "|000905"
    store = FakeOverlayStore(manifest, stock_days, index_days)

    with pytest.raises(DataQualityError, match="别名映射后发生指数代码碰撞"):
        load_hybrid_universe(
                "/unused/csmar",
                overlay_root="/unused/overlay",
                overlay_source_id="infoway",
            as_of=trade_day,
            decision_date=trade_day,
            mode="historical",
            overlay_store=store,
            **SMALL_UNIVERSE_GATES,
        )


def test_index_rows_alias_collision_fails_closed() -> None:
    trade_day = date(2026, 8, 25)
    stock_days = {trade_day: _stock_day(trade_day)}
    indices = _index_day(trade_day)
    canonical_collision = indices.loc[indices["symbol"] == "399905"].copy()
    canonical_collision.loc[:, "symbol"] = "000905"
    index_days = {trade_day: pd.concat([indices, canonical_collision], ignore_index=True)}
    manifest = _manifest(stock_days, index_days)
    manifest.loc[:, "index_count"] = len(indices)
    store = FakeOverlayStore(manifest, stock_days, index_days)

    with pytest.raises(DataQualityError, match="别名映射后发生指数代码碰撞"):
        load_hybrid_universe(
                "/unused/csmar",
                overlay_root="/unused/overlay",
                overlay_source_id="infoway",
            as_of=trade_day,
            decision_date=trade_day,
            mode="historical",
            overlay_store=store,
            **SMALL_UNIVERSE_GATES,
        )


def test_manifest_trading_day_gap_fails_closed() -> None:
    day_one = date(2026, 8, 25)
    day_two = date(2026, 8, 26)
    stock_days = {day: _stock_day(day) for day in (day_one, day_two)}
    index_days = {day: _index_day(day) for day in stock_days}
    manifest = _manifest(stock_days, index_days)
    manifest.loc[manifest["trade_date"] == day_two, "previous_trade_date"] = BASELINE_CUTOFF
    store = FakeOverlayStore(manifest, stock_days, index_days)

    with pytest.raises(DataQualityError, match="交易日断链"):
        load_hybrid_universe(
                "/unused/csmar",
                overlay_root="/unused/overlay",
                overlay_source_id="infoway",
            as_of=day_two,
            decision_date=day_two,
            mode="historical",
            overlay_store=store,
            **SMALL_UNIVERSE_GATES,
        )


@pytest.mark.parametrize("audit_column", ["latest_retrieved_at", "verified_at"])
def test_historical_replay_rejects_manifest_known_after_decision(audit_column: str) -> None:
    trade_day = date(2026, 8, 25)
    stock_days = {trade_day: _stock_day(trade_day)}
    index_days = {trade_day: _index_day(trade_day)}
    manifest = _manifest(stock_days, index_days, retrieved_at="2026-08-25T08:30:00Z")
    manifest.loc[:, audit_column] = "2026-08-26T00:00:00Z"
    store = FakeOverlayStore(manifest, stock_days, index_days)

    with pytest.raises(DataQualityError, match="历史回放禁止使用"):
        load_hybrid_universe(
                "/unused/csmar",
                overlay_root="/unused/overlay",
                overlay_source_id="infoway",
            as_of=trade_day,
            decision_date=trade_day,
            mode="historical",
            overlay_store=store,
            **SMALL_UNIVERSE_GATES,
        )


def test_adjusted_overlay_is_rejected() -> None:
    trade_day = date(2026, 8, 25)
    stock_days = {trade_day: _stock_day(trade_day)}
    index_days = {trade_day: _index_day(trade_day)}
    store = FakeOverlayStore(
        _manifest(stock_days, index_days, adjustment="qfq"),
        stock_days,
        index_days,
    )

    with pytest.raises(DataQualityError, match="必须为none未复权"):
        load_hybrid_universe(
                "/unused/csmar",
                overlay_root="/unused/overlay",
                overlay_source_id="infoway",
            as_of=trade_day,
            decision_date=trade_day,
            mode="historical",
            overlay_store=store,
            **SMALL_UNIVERSE_GATES,
        )
