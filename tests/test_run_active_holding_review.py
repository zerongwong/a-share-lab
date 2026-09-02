from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.domain.errors import DataQualityError
from ashare_lab.services.holding_ledger import (
    HoldingPositionInput,
    holding_knowledge_context,
    replace_active_holdings,
)
from ashare_lab.services.review_active_holdings import HoldingReviewSummaryStatus
from ashare_lab.services.run_active_holding_review import (
    build_evening_holding_review,
    load_active_holding_histories,
    run_active_holding_review,
)


@pytest.fixture
def repository(tmp_path: Path) -> SQLiteRepository:
    repo = SQLiteRepository(
        tmp_path / "research.db",
        Path(__file__).resolve().parents[1] / "migrations",
    )
    repo.initialize()
    replace_active_holdings(
        repo,
        (
            HoldingPositionInput(
                symbol="600919",
                name="江苏银行",
                entry_date=date(2026, 8, 24),
                stock_sleeve_weight=1.0,
                metadata={
                    "company_action_clear": True,
                    "company_action_clear_through": "2026-08-28",
                    "company_action_evidence_source": "test_local_announcement_review",
                    "company_action_evidence_id": "ca:600919:2026-08-28",
                },
            ),
        ),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 24, 21, 0, tzinfo=UTC),
    )
    return repo


class FakeBaseline:
    def __init__(self) -> None:
        self.fetches: list[str] = []

    def latest_trade_date(self, *, on_or_before: date | None = None) -> date:
        assert on_or_before == date(2026, 8, 28)
        return date(2026, 8, 24)

    def fetch_daily(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        adjust: str = "none",
    ) -> pd.DataFrame:
        self.fetches.append(symbol)
        assert adjust == "none"
        assert end == date(2026, 8, 24)
        dates = pd.bdate_range(end=end, periods=300)
        index = np.arange(len(dates), dtype=float)
        close = 10.0 * np.exp(0.001 * index + 0.02 * np.sin(index / 13.0))
        return pd.DataFrame(
            {
                "trade_date": dates,
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "amount_cny": np.full(len(dates), 80_000_000.0),
                "source": "csmar",
            }
        )


class FakeOverlay:
    dates = (
        date(2026, 8, 25),
        date(2026, 8, 26),
        date(2026, 8, 27),
        date(2026, 8, 28),
    )

    def read_verified_manifest(self, *, source_id: str | None = None) -> pd.DataFrame:
        assert source_id == "zero_budget_eod"
        return pd.DataFrame(
            {
                "trade_date": pd.to_datetime(self.dates),
                "previous_trade_date": pd.to_datetime((date(2026, 8, 24), *self.dates[:-1])),
                "adjustment": ["none"] * 4,
                "source_id": ["zero_budget_eod"] * 4,
            }
        )

    def verified_dates_from(
        self,
        *,
        source_id: str,
        baseline_cutoff: date,
        through_date: date | None = None,
    ) -> tuple[date, ...]:
        assert source_id == "zero_budget_eod"
        assert baseline_cutoff == date(2026, 8, 24)
        assert through_date == date(2026, 8, 28)
        return self.dates

    def read_verified_daily(
        self,
        trade_date: date,
        *,
        source_id: str,
        asset_kind: str,
    ) -> pd.DataFrame:
        assert source_id == "zero_budget_eod"
        assert asset_kind == "stocks"
        close = 13.5 + self.dates.index(trade_date) * 0.05
        return pd.DataFrame(
            {
                "symbol": ["600919", "000001"],
                "trade_date": pd.to_datetime([trade_date, trade_date]),
                "open": [close - 0.02, 10.0],
                "high": [close + 0.10, 10.1],
                "low": [close - 0.10, 9.9],
                "close": [close, 10.0],
                "amount_cny": [90_000_000.0, 80_000_000.0],
                "source": ["zero_budget_eod", "zero_budget_eod"],
            }
        )


def test_loader_reads_only_current_holdings_and_joins_continuous_overlay(
    repository: SQLiteRepository,
) -> None:
    baseline = FakeBaseline()
    overlay = FakeOverlay()

    loaded = load_active_holding_histories(
        repository,
        dataset_root="unused",
        overlay_root="unused",
        as_of=date(2026, 8, 28),
        _baseline_market=baseline,
        _overlay_store=overlay,
    )

    assert loaded is not None
    assert baseline.fetches == ["600919"]
    assert set(loaded.histories) == {"600919"}
    assert loaded.baseline_cutoff == date(2026, 8, 24)
    assert loaded.data_cutoff == date(2026, 8, 28)
    assert loaded.adjustment == "none"
    assert tuple(loaded.histories["600919"].tail(4)["trade_date"].dt.date) == overlay.dates


def test_evening_entrypoint_reuses_verified_cutoff_and_core(
    repository: SQLiteRepository,
) -> None:
    result = build_evening_holding_review(
        repository,
        dataset_root="unused",
        overlay_root="unused",
        decision_date=date(2026, 8, 28),
        persist=False,
        _baseline_market=FakeBaseline(),
        _overlay_store=FakeOverlay(),
    )

    assert result.status is HoldingReviewSummaryStatus.READY
    assert result.data_cutoff == date(2026, 8, 28)
    assert result.holding_version == 1
    assert result.rows[0].symbol == "600919"


def test_live_holding_context_uses_latest_revision_on_older_market_cutoff(
    repository: SQLiteRepository,
) -> None:
    latest = replace_active_holdings(
        repository,
        (
            HoldingPositionInput(
                symbol="600919",
                name="江苏银行",
                entry_date=date(2026, 8, 24),
                stock_sleeve_weight=1.0,
                metadata={
                    "company_action_clear": True,
                    "company_action_clear_through": "2026-08-28",
                    "company_action_evidence_source": "test_local_announcement_review",
                    "company_action_evidence_id": "ca:600919:2026-08-28",
                },
            ),
        ),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 29, 21, 0, tzinfo=UTC),
    )
    known_at = datetime(2026, 8, 29, 22, 0, tzinfo=UTC)
    context = holding_knowledge_context(latest, known_at=known_at)

    historical = load_active_holding_histories(
        repository,
        dataset_root="unused",
        overlay_root="unused",
        as_of=date(2026, 8, 28),
        _baseline_market=FakeBaseline(),
        _overlay_store=FakeOverlay(),
    )
    live = run_active_holding_review(
        repository,
        dataset_root="unused",
        overlay_root="unused",
        as_of=date(2026, 8, 28),
        reviewed_at=known_at,
        persist=False,
        _baseline_market=FakeBaseline(),
        _overlay_store=FakeOverlay(),
        holding_context=context,
    )

    assert historical is not None
    assert historical.holding_version == 1
    assert historical.data_cutoff == date(2026, 8, 28)
    assert live.status is HoldingReviewSummaryStatus.READY
    assert live.holding_version == latest.version == 2
    assert live.data_cutoff == date(2026, 8, 28)


def test_latest_manifest_row_must_be_on_continuous_chain(
    repository: SQLiteRepository,
) -> None:
    class GappedOverlay(FakeOverlay):
        def verified_dates_from(self, **kwargs: object) -> tuple[date, ...]:
            return self.dates[:-1]

    with pytest.raises(DataQualityError, match="continuous chain"):
        load_active_holding_histories(
            repository,
            dataset_root="unused",
            overlay_root="unused",
            as_of=date(2026, 8, 28),
            _baseline_market=FakeBaseline(),
            _overlay_store=GappedOverlay(),
        )


def test_future_manifest_adjustment_cannot_poison_historical_replay(
    repository: SQLiteRepository,
) -> None:
    class FutureAdjustedOverlay(FakeOverlay):
        def read_verified_manifest(self, *, source_id: str | None = None) -> pd.DataFrame:
            manifest = super().read_verified_manifest(source_id=source_id)
            return pd.concat(
                [
                    manifest,
                    pd.DataFrame(
                        {
                            "trade_date": [pd.Timestamp("2026-09-01")],
                            "previous_trade_date": [pd.Timestamp("2026-08-31")],
                            "adjustment": ["qfq"],
                            "source_id": ["zero_budget_eod"],
                        }
                    ),
                ],
                ignore_index=True,
            )

    loaded = load_active_holding_histories(
        repository,
        dataset_root="unused",
        overlay_root="unused",
        as_of=date(2026, 8, 28),
        _baseline_market=FakeBaseline(),
        _overlay_store=FutureAdjustedOverlay(),
    )

    assert loaded is not None
    assert loaded.data_cutoff == date(2026, 8, 28)
