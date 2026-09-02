from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

import ashare_lab.services.run_zero_budget_daily_update as update_module
from ashare_lab.adapters.market_overlay_store import MarketOverlayStore
from ashare_lab.adapters.tushare_daily import TushareDailyClient
from ashare_lab.adapters.zero_budget_eod import (
    ZERO_BUDGET_AMOUNT_MULTIPLIER_TO_CNY,
    ZERO_BUDGET_EOD_PROVIDER,
    ZERO_BUDGET_INDEX_SOURCE,
    ZERO_BUDGET_STOCK_SOURCE,
    ZERO_BUDGET_UNIT_CONTRACT_VERSION,
    ZERO_BUDGET_UNIT_RESOLUTION_METHOD_VERSION,
    ZeroBudgetEodMarketData,
)
from ashare_lab.domain.data_sources import (
    DataAction,
    RightsViolationError,
    SourceId,
)
from ashare_lab.domain.errors import DataUnavailableError
from ashare_lab.ports.daily_increment import AssetKind, DailyIncrementBatch
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS, normalize_symbol

BASELINE = date(2026, 8, 24)
DAY_25 = date(2026, 8, 25)
DAY_26 = date(2026, 8, 26)
NOW = datetime(2026, 8, 27, 4, 30, tzinfo=UTC)  # 12:30 Asia/Shanghai
TOKEN = "tushare-secret-never-report-this"
STOCKS = tuple(f"{value:06d}.SZ" for value in range(1, 101))
CORE_INDICES = (
    "000001.SH",
    "000300.SH",
    "000852.SH",
    "000905.SH",
    "399001.SZ",
    "399006.SZ",
)


class CloseableComponent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.close_calls = 0

    @property
    def closed(self) -> bool:
        return self.close_calls > 0

    def close(self) -> None:
        self.close_calls += 1


class RecordingRightsPolicy:
    def __init__(self, deny: SourceId | None = None) -> None:
        self.deny = deny
        self.calls: list[tuple[SourceId, DataAction]] = []

    def require(
        self,
        source_id: SourceId | str,
        action: DataAction | str,
    ) -> object:
        source = source_id if isinstance(source_id, SourceId) else SourceId(source_id)
        normalized = action if isinstance(action, DataAction) else DataAction(action)
        self.calls.append((source, normalized))
        if source is self.deny:
            raise RightsViolationError(f"denied test source: {source.value}")
        return object()


def _daily_frame(
    symbols: tuple[str, ...],
    target_date: date,
    *,
    asset_kind: AssetKind,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, raw_symbol in enumerate(symbols):
        close = 10.0 + index / 100.0
        volume = 1_000_000 + index * 100
        rows.append(
            {
                "symbol": normalize_symbol(raw_symbol),
                "trade_date": target_date,
                "open": close - 0.02,
                "high": close + 0.10,
                "low": close - 0.10,
                "close": close,
                "prev_close": close - 0.05,
                "volume_shares": volume,
                "amount_cny": close * volume,
                "turnover_pct": None,
                "source": (
                    ZERO_BUDGET_STOCK_SOURCE
                    if asset_kind == "stocks"
                    else ZERO_BUDGET_INDEX_SOURCE
                ),
                "retrieved_at": NOW.isoformat().replace("+00:00", "Z"),
            }
        )
    return pd.DataFrame(rows, columns=["symbol", *CANONICAL_DAILY_COLUMNS])


class FakeCompositeProvider(CloseableComponent):
    provider = ZERO_BUDGET_EOD_PROVIDER
    source_id = ZERO_BUDGET_EOD_PROVIDER

    def __init__(
        self,
        *,
        missing_stock_count: int = 0,
        fail_text: str | None = None,
    ) -> None:
        super().__init__("provider")
        self.missing_stock_count = missing_stock_count
        self.fail_text = fail_text
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
        if self.fail_text is not None:
            raise RuntimeError(self.fail_text)
        self.asset_kinds.append(asset_kind)
        requested = tuple(symbols)
        received = (
            requested[: len(requested) - self.missing_stock_count]
            if asset_kind == "stocks"
            else requested
        )
        return DailyIncrementBatch(
            frame=_daily_frame(received, target_date, asset_kind=asset_kind),
            target_date=target_date,
            requested_symbols=requested,
            received_symbols=received,
            fetched_at=NOW,
            trace_ids=(f"safe-{asset_kind}-{target_date.isoformat()}",),
            provider=ZERO_BUDGET_EOD_PROVIDER,
            cutoff_timestamp=cutoff_timestamp or 1_777_777_777,
            unit_contract_version=ZERO_BUDGET_UNIT_CONTRACT_VERSION,
            unit_resolution_method_version=(
                ZERO_BUDGET_UNIT_RESOLUTION_METHOD_VERSION
            ),
            amount_multiplier_to_cny=ZERO_BUDGET_AMOUNT_MULTIPLIER_TO_CNY,
        )


def _patch_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        update_module,
        "read_csmar_baseline_cutoff",
        lambda _root, *, through_date: BASELINE,
    )


def _factories(
    *,
    missing_stock_count: int = 0,
) -> tuple[dict[str, object], dict[str, object]]:
    created: dict[str, object] = {}
    received: dict[str, object] = {}

    def baostock_factory(**kwargs: object) -> CloseableComponent:
        received["baostock_kwargs"] = kwargs
        component = CloseableComponent("baostock")
        created["baostock"] = component
        return component

    def tushare_factory(token: str, **kwargs: object) -> CloseableComponent:
        received["token"] = token
        received["tushare_kwargs"] = kwargs
        component = CloseableComponent("tushare")
        created["tushare"] = component
        return component

    def verifier_factory(**kwargs: object) -> CloseableComponent:
        received["verifier_kwargs"] = kwargs
        component = CloseableComponent("verifier")
        created["verifier"] = component
        return component

    def provider_factory(**kwargs: object) -> FakeCompositeProvider:
        received["provider_kwargs"] = kwargs
        provider = FakeCompositeProvider(missing_stock_count=missing_stock_count)
        created["provider"] = provider
        return provider

    received["baostock_factory"] = baostock_factory
    received["tushare_factory"] = tushare_factory
    received["verifier_factory"] = verifier_factory
    received["provider_factory"] = provider_factory
    return created, received


def _run_with_factories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing_stock_count: int = 0,
    rights: RecordingRightsPolicy | None = None,
):
    _patch_baseline(monkeypatch)
    created, seams = _factories(missing_stock_count=missing_stock_count)
    report = update_module.run_zero_budget_daily_update(
        csmar_root=tmp_path / "csmar",
        overlay_root=tmp_path / "overlay",
        now=NOW,
        core_index_symbols=CORE_INDICES,
        _token_loader=lambda: f"  {TOKEN}  ",
        _baostock_factory=seams["baostock_factory"],
        _tushare_factory=seams["tushare_factory"],
        _verifier_factory=seams["verifier_factory"],
        _provider_factory=seams["provider_factory"],
        _rights_policy=rights or RecordingRightsPolicy(),
    )
    return report, created, seams


def _assert_components_closed(created: dict[str, object]) -> None:
    assert set(created) == {"baostock", "tushare", "verifier", "provider"}
    assert all(component.close_calls == 1 for component in created.values())


def _assert_secret_absent_from_tree(root: Path) -> None:
    encoded = TOKEN.encode()
    for path in root.rglob("*"):
        if path.is_file():
            assert encoded not in path.read_bytes(), f"secret leaked to {path}"


def test_missing_token_fails_before_any_provider_component_is_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_baseline(monkeypatch)
    factory_calls = 0

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("provider components must not be constructed")

    with pytest.raises(DataUnavailableError, match="Tushare Token") as error:
        update_module.run_zero_budget_daily_update(
            csmar_root=tmp_path / "csmar",
            overlay_root=tmp_path / "overlay",
            now=NOW,
            _token_loader=lambda: "   ",
            _baostock_factory=forbidden_factory,
            _tushare_factory=forbidden_factory,
            _verifier_factory=forbidden_factory,
            _provider_factory=forbidden_factory,
            _rights_policy=RecordingRightsPolicy(),
        )

    assert factory_calls == 0
    assert TOKEN not in str(error.value)


@pytest.mark.parametrize(
    "denied_source",
    [
        SourceId.TUSHARE,
        SourceId.BAOSTOCK,
        SourceId.AKSHARE,
        SourceId.ZERO_BUDGET_EOD,
    ],
)
def test_each_of_four_source_rights_is_required_before_component_creation(
    denied_source: SourceId,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_baseline(monkeypatch)
    rights = RecordingRightsPolicy(deny=denied_source)
    constructed = False

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        nonlocal constructed
        constructed = True
        raise AssertionError("rights must be checked first")

    with pytest.raises(RightsViolationError, match=denied_source.value):
        update_module.run_zero_budget_daily_update(
            csmar_root=tmp_path / "csmar",
            overlay_root=tmp_path / "overlay",
            now=NOW,
            _token_loader=lambda: TOKEN,
            _baostock_factory=forbidden_factory,
            _tushare_factory=forbidden_factory,
            _verifier_factory=forbidden_factory,
            _provider_factory=forbidden_factory,
            _rights_policy=rights,
        )

    assert any(source is denied_source for source, _ in rights.calls)
    assert constructed is False


def test_two_missing_stock_rows_pass_the_98_percent_gate_and_publish_own_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rights = RecordingRightsPolicy()
    report, created, seams = _run_with_factories(
        tmp_path,
        monkeypatch,
        missing_stock_count=2,
        rights=rights,
    )

    assert report.source_id == ZERO_BUDGET_EOD_PROVIDER
    assert report.range_report.source_id == ZERO_BUDGET_EOD_PROVIDER
    assert report.historical_baseline_cutoff == BASELINE
    assert report.requested_complete_date == DAY_26
    assert report.latest_complete_session == DAY_26
    assert report.automatic_increment_cutoff == DAY_26
    assert report.common_cutoff == DAY_26
    assert report.updated_sessions == (DAY_25, DAY_26)
    assert report.quarantined_failures == ()
    assert report.current_through_latest_complete_session is True
    assert report.csmar_mutated is False
    assert report.unit_contract_version == ZERO_BUDGET_UNIT_CONTRACT_VERSION
    assert (
        report.unit_resolution_method_version
        == ZERO_BUDGET_UNIT_RESOLUTION_METHOD_VERSION
    )
    assert seams["token"] == TOKEN
    _assert_components_closed(created)

    initial_rights = (
        (SourceId.TUSHARE, DataAction.MARKET_DATA_READ),
        (SourceId.TUSHARE, DataAction.MARKET_DATA_CACHE),
        (SourceId.BAOSTOCK, DataAction.MARKET_DATA_READ),
        (SourceId.BAOSTOCK, DataAction.MARKET_DATA_CACHE),
        (SourceId.BAOSTOCK, DataAction.METADATA_READ),
        (SourceId.AKSHARE, DataAction.MARKET_DATA_READ),
        (SourceId.ZERO_BUDGET_EOD, DataAction.MARKET_DATA_READ),
        (SourceId.ZERO_BUDGET_EOD, DataAction.MARKET_DATA_CACHE),
    )
    assert tuple(rights.calls[: len(initial_rights)]) == initial_rights

    overlay_root = tmp_path / "overlay"
    store = MarketOverlayStore(overlay_root)
    manifest = store.read_verified_manifest()
    assert manifest["source_id"].tolist() == [
        ZERO_BUDGET_EOD_PROVIDER,
        ZERO_BUDGET_EOD_PROVIDER,
    ]
    assert pd.to_datetime(manifest["trade_date"]).dt.date.tolist() == [DAY_25, DAY_26]
    assert manifest["stock_count"].tolist() == [98, 98]
    assert manifest["expected_stock_count"].tolist() == [100, 100]
    assert manifest["stock_coverage_ratio"].tolist() == pytest.approx([0.98, 0.98])
    assert all(
        f"source={ZERO_BUDGET_EOD_PROVIDER}" in str(path)
        for path in (*manifest["stock_file"], *manifest["index_file"])
    )
    assert not (overlay_root / "source=infoway").exists()
    assert TOKEN not in repr(report)
    assert not manifest["receipt_json"].astype(str).str.contains(TOKEN, regex=False).any()
    _assert_secret_absent_from_tree(overlay_root)


def test_three_missing_stock_rows_are_quarantined_below_the_98_percent_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, created, _ = _run_with_factories(
        tmp_path,
        monkeypatch,
        missing_stock_count=3,
    )

    assert report.automatic_increment_cutoff is None
    assert report.common_cutoff == BASELINE
    assert report.current_through_latest_complete_session is False
    assert report.updated_sessions == ()
    assert len(report.quarantined_failures) == 1
    failure = report.quarantined_failures[0]
    assert failure.trade_date == DAY_25
    assert "coverage 0.9700 is below 0.9800" in failure.reason
    assert failure.path is not None and Path(failure.path).is_dir()
    assert TOKEN not in failure.reason
    assert MarketOverlayStore(tmp_path / "overlay").read_verified_manifest().empty
    _assert_components_closed(created)
    _assert_secret_absent_from_tree(tmp_path / "overlay")


def test_upstream_error_text_is_sanitized_in_report_and_all_components_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_baseline(monkeypatch)
    created: dict[str, CloseableComponent] = {}

    class FakeBaoStock(CloseableComponent):
        def fetch_cn_trading_days(self, start: date, end: date) -> tuple[date, ...]:
            return tuple(value for value in (DAY_25, DAY_26) if start <= value <= end)

        def fetch_cn_stock_symbols(self) -> tuple[str, ...]:
            return STOCKS

    class EchoingTushare(CloseableComponent):
        def fetch_daily(self, _target_date: date) -> object:
            raise RuntimeError(f"upstream echoed credential {TOKEN}")

    def baostock_factory(**_kwargs: object) -> FakeBaoStock:
        value = FakeBaoStock("baostock")
        created["baostock"] = value
        return value

    def tushare_factory(_token: str, **_kwargs: object) -> EchoingTushare:
        value = EchoingTushare("tushare")
        created["tushare"] = value
        return value

    def verifier_factory(**_kwargs: object) -> CloseableComponent:
        value = CloseableComponent("verifier")
        created["verifier"] = value
        return value

    report = update_module.run_zero_budget_daily_update(
        csmar_root=tmp_path / "csmar",
        overlay_root=tmp_path / "overlay",
        now=NOW,
        core_index_symbols=CORE_INDICES,
        _token_loader=lambda: TOKEN,
        _baostock_factory=baostock_factory,
        _tushare_factory=tushare_factory,
        _verifier_factory=verifier_factory,
        _provider_factory=ZeroBudgetEodMarketData,
        _rights_policy=RecordingRightsPolicy(),
    )

    assert len(report.quarantined_failures) == 1
    assert TOKEN not in repr(report)
    assert TOKEN not in report.quarantined_failures[0].reason
    assert "原始错误已脱敏" in report.quarantined_failures[0].reason
    assert all(component.close_calls == 1 for component in created.values())
    _assert_secret_absent_from_tree(tmp_path / "overlay")


def test_partial_construction_failure_is_sanitized_and_closes_prior_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: cleanup must start before the second factory is called."""

    _patch_baseline(monkeypatch)
    baostock = CloseableComponent("baostock")

    def baostock_factory(**_kwargs: object) -> CloseableComponent:
        return baostock

    def tushare_factory(token: str, **kwargs: object) -> TushareDailyClient:
        def echoing_client_factory(_token: str) -> object:
            raise RuntimeError(f"SDK echoed credential {token}")

        return TushareDailyClient(
            token,
            client_factory=echoing_client_factory,
            clock=kwargs["clock"],
        )

    with pytest.raises(DataUnavailableError) as error:
        update_module.run_zero_budget_daily_update(
            csmar_root=tmp_path / "csmar",
            overlay_root=tmp_path / "overlay",
            now=NOW,
            _token_loader=lambda: TOKEN,
            _baostock_factory=baostock_factory,
            _tushare_factory=tushare_factory,
            _verifier_factory=lambda **_kwargs: pytest.fail("must not be constructed"),
            _provider_factory=lambda **_kwargs: pytest.fail("must not be constructed"),
            _rights_policy=RecordingRightsPolicy(),
        )

    assert TOKEN not in str(error.value)
    assert baostock.close_calls == 1
