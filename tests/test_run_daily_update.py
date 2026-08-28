from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.daily_increment import AssetKind, DailyIncrementBatch
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS, normalize_symbol
from ashare_lab.services.run_daily_update import (
    DEFAULT_INFOWAY_CORE_INDICES,
    INFOWAY_EOD_UNIT_CONTRACT,
    INFOWAY_EOD_UNIT_CONTRACT_VERSION,
    INFOWAY_EOD_UNIT_RESOLUTION_METHOD_VERSION,
    latest_complete_cn_candidate,
    run_daily_update,
)

BASELINE = date(2026, 8, 24)
DAY_25 = date(2026, 8, 25)
DAY_26 = date(2026, 8, 26)
NOW = datetime(2026, 8, 27, 4, 30, tzinfo=UTC)  # 12:30 in Shanghai
SECRET = "keychain-secret-never-report-this"
STOCKS = tuple(f"{value:06d}.SZ" for value in range(1, 101))


def _csmar_root(tmp_path: Path) -> Path:
    root = tmp_path / "csmar"
    root.mkdir()
    connection = duckdb.connect(str(root / "csmar.duckdb"))
    try:
        connection.execute("CREATE TABLE daily_bars(trade_date DATE)")
        connection.execute("INSERT INTO daily_bars VALUES (?)", [BASELINE])
    finally:
        connection.close()
    return root


def _frame(symbols: tuple[str, ...], target_date: date) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, raw_symbol in enumerate(symbols):
        close = 10.0 + index / 100
        rows.append(
            {
                "symbol": normalize_symbol(raw_symbol),
                "trade_date": target_date,
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "prev_close": close - 0.05,
                "volume_shares": 1_000_000 + index,
                "amount_cny": close * (1_000_000 + index),
                "turnover_pct": None,
                "source": "infoway:eod_unadjusted",
                "retrieved_at": NOW.isoformat().replace("+00:00", "Z"),
            }
        )
    return pd.DataFrame(rows, columns=["symbol", *CANONICAL_DAILY_COLUMNS])


class FakeProvider:
    provider = "infoway"

    def __init__(self, api_key: str, *, unit_contract) -> None:
        self.received_secret = api_key
        self.unit_contract = unit_contract
        self.closed = False
        self.asset_kinds: list[AssetKind] = []

    def fetch_cn_trading_days(self, start: date, end: date) -> tuple[date, ...]:
        return tuple(value for value in (DAY_25, DAY_26) if start <= value <= end)

    def fetch_cn_stock_symbols(self) -> tuple[str, ...]:
        return STOCKS

    def fetch_daily_increment(
        self,
        symbols: tuple[str, ...],
        target_date: date,
        *,
        cutoff_timestamp: int | None = None,
        asset_kind: AssetKind = "stocks",
    ) -> DailyIncrementBatch:
        requested = tuple(symbols)
        self.asset_kinds.append(asset_kind)
        return DailyIncrementBatch(
            frame=_frame(requested, target_date),
            target_date=target_date,
            requested_symbols=requested,
            received_symbols=requested,
            fetched_at=NOW,
            trace_ids=("safe-trace",),
            provider="infoway",
            cutoff_timestamp=cutoff_timestamp or 1_777_777_777,
        )

    def close(self) -> None:
        self.closed = True


def test_intraday_never_treats_today_as_a_completed_daily_bar() -> None:
    assert latest_complete_cn_candidate(NOW) == DAY_26
    just_before_ready = datetime(2026, 8, 27, 7, 29, 59, tzinfo=UTC)
    at_ready = datetime(2026, 8, 27, 7, 30, tzinfo=UTC)  # 15:30 Shanghai
    assert latest_complete_cn_candidate(just_before_ready) == DAY_26
    assert latest_complete_cn_candidate(at_ready) == date(2026, 8, 27)


def test_explicit_versioned_public_unit_contract_never_guesses_vm() -> None:
    assert INFOWAY_EOD_UNIT_CONTRACT_VERSION == "infoway-cn-eod-v3-2026-08-27"
    assert INFOWAY_EOD_UNIT_RESOLUTION_METHOD_VERSION == "batch-price-band-v1"
    assert INFOWAY_EOD_UNIT_CONTRACT.volume_multiplier_to_shares == 100
    assert INFOWAY_EOD_UNIT_CONTRACT.amount_field == "vw"
    assert INFOWAY_EOD_UNIT_CONTRACT.amount_multiplier_to_cny == 1
    assert INFOWAY_EOD_UNIT_CONTRACT.provisional_amount_multiplier_to_cny == 100
    assert (
        INFOWAY_EOD_UNIT_CONTRACT.resolution_method_version
        == INFOWAY_EOD_UNIT_RESOLUTION_METHOD_VERSION
    )
    assert "low*volume_shares<=amount_cny<=high*volume_shares" in (
        INFOWAY_EOD_UNIT_CONTRACT.verified_reference
    )
    assert "raw rows are deliberately not stored" in (INFOWAY_EOD_UNIT_CONTRACT.verified_reference)
    assert "600000.SH" not in INFOWAY_EOD_UNIT_CONTRACT.verified_reference
    assert "100-share lots" in INFOWAY_EOD_UNIT_CONTRACT.verified_reference


def test_infoway_core_indices_use_provider_csi500_symbol() -> None:
    assert "399905.SZ" in DEFAULT_INFOWAY_CORE_INDICES
    assert "000905.SH" not in DEFAULT_INFOWAY_CORE_INDICES


def test_keychain_only_update_appends_25_and_26_without_mutating_csmar(tmp_path: Path) -> None:
    csmar_root = _csmar_root(tmp_path)
    overlay_root = tmp_path / "overlay"
    providers: list[FakeProvider] = []

    def factory(api_key: str, *, unit_contract):
        provider = FakeProvider(api_key, unit_contract=unit_contract)
        providers.append(provider)
        return provider

    report = run_daily_update(
        csmar_root=csmar_root,
        overlay_root=overlay_root,
        now=NOW,
        _api_key_loader=lambda: SECRET,
        _provider_factory=factory,
    )

    assert report.historical_baseline_cutoff == BASELINE
    assert report.requested_complete_date == DAY_26
    assert report.latest_complete_session == DAY_26
    assert report.automatic_increment_cutoff == DAY_26
    assert report.common_cutoff == DAY_26
    assert report.updated_sessions == (DAY_25, DAY_26)
    assert report.quarantined_failures == ()
    assert report.current_through_latest_complete_session is True
    assert report.csmar_mutated is False
    assert report.unit_contract_version == INFOWAY_EOD_UNIT_CONTRACT_VERSION
    assert report.unit_resolution_method_version == INFOWAY_EOD_UNIT_RESOLUTION_METHOD_VERSION
    assert "不含北交所" in report.market_scope
    assert SECRET not in repr(report)
    assert providers[0].received_secret == SECRET
    assert providers[0].unit_contract is INFOWAY_EOD_UNIT_CONTRACT
    assert providers[0].asset_kinds == ["stocks", "indices", "stocks", "indices"]
    assert providers[0].closed is True

    connection = duckdb.connect(str(csmar_root / "csmar.duckdb"), read_only=True)
    try:
        assert connection.execute("SELECT count(*) FROM daily_bars").fetchone()[0] == 1
        assert (
            connection.execute("SELECT max(trade_date) FROM daily_bars").fetchone()[0] == BASELINE
        )
    finally:
        connection.close()


def test_missing_key_fails_before_provider_construction(tmp_path: Path) -> None:
    called = False

    def factory(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("must not construct provider")

    import pytest

    with pytest.raises(DataUnavailableError, match="macOS钥匙串"):
        run_daily_update(
            csmar_root=_csmar_root(tmp_path),
            overlay_root=tmp_path / "overlay",
            now=NOW,
            _api_key_loader=lambda: None,
            _provider_factory=factory,
        )
    assert called is False


def test_unit_contract_change_is_quarantined_and_does_not_advance_cutoff(
    tmp_path: Path,
) -> None:
    class ChangedFieldProvider(FakeProvider):
        def fetch_daily_increment(self, *args, **kwargs):
            raise DataQualityError("Infoway 日K成交额字段与已验证单位合同不一致。")

    report = run_daily_update(
        csmar_root=_csmar_root(tmp_path),
        overlay_root=tmp_path / "overlay",
        now=NOW,
        _api_key_loader=lambda: SECRET,
        _provider_factory=ChangedFieldProvider,
    )

    assert report.provider_contract_changed is True
    assert report.automatic_increment_cutoff is None
    assert report.common_cutoff == BASELINE
    assert report.current_through_latest_complete_session is False
    assert len(report.quarantined_failures) == 1
    failure = report.quarantined_failures[0]
    assert failure.trade_date == DAY_25
    assert failure.path is not None and Path(failure.path).is_dir()
    assert SECRET not in failure.reason
