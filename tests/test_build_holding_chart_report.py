from __future__ import annotations

import stat
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.domain.errors import DataQualityError
from ashare_lab.services.build_holding_chart_report import (
    HoldingChartBuildStatus,
    OriginOverlayNature,
    _archive_holding_chart_report,
    _entry_overlays,
    _OriginArchive,
    _review_is_chart_eligible,
    build_holding_chart_report,
)
from ashare_lab.services.holding_ledger import (
    HoldingPositionInput,
    get_active_holding_portfolio,
    holding_chart_delivery_channels,
    holding_knowledge_context,
    replace_active_holdings,
)
from ashare_lab.services.render_holding_chart_report import EntryOverlayNature
from ashare_lab.services.review_active_holdings import (
    HoldingAction,
    HoldingReviewRowStatus,
    HoldingReviewSummaryStatus,
    HoldingTreeReviewRow,
    HoldingTreeReviewSummary,
)
from ashare_lab.services.run_active_holding_review import ActiveHoldingHistoryLoad

CUTOFF = date(2026, 8, 28)
PLAN_CUTOFF = date(2026, 8, 27)
SYMBOLS = ("600919", "601919", "002142", "601156")


def _repository(
    tmp_path: Path,
    *,
    linked_origin: bool = True,
    archive_nature: str = "original",
) -> SQLiteRepository:
    repository = SQLiteRepository(
        tmp_path / "research.db",
        Path(__file__).resolve().parents[1] / "migrations",
    )
    repository.initialize()
    if linked_origin:
        report = {
            "id": "report-origin",
            "content_hash": f"origin-{archive_nature}",
            "archive_nature": archive_nature,
            "decision_date": PLAN_CUTOFF.isoformat(),
            "plan_for_date": CUTOFF.isoformat(),
            "common_cutoff": PLAN_CUTOFF.isoformat(),
            "method_version": "test-digest-v1",
            "cycle_label": "test",
            "entry_strictness": "standard",
            "max_stock_exposure": 0.4,
            "minimum_cash_weight": 0.6,
            "created_at": "2026-08-27T21:00:00+08:00",
            "metadata_json": {"orders_enabled": False},
        }
        batch = {
            "id": "batch-origin-4w",
            "report_id": "report-origin",
            "holding_weeks": 4,
            "holding_sessions": 20,
            "label": "1个月",
            "data_cutoff": PLAN_CUTOFF.isoformat(),
            "source_status": "ready",
            "allocation_nature": "action_research",
            "action_stock_exposure": 0.4,
            "action_cash_weight": 0.6,
            "member_count": len(SYMBOLS),
            "status": "pending",
        }
        members = [
            {
                "id": f"member-{symbol}",
                "batch_id": "batch-origin-4w",
                "rank": rank,
                "symbol": f"{symbol}.SH" if symbol.startswith("6") else f"{symbol}.SZ",
                "name": f"样本{rank}",
                "action": "条件介入研究",
                "allocation_nature": "action_research",
                "operational_stock_sleeve_weight": 0.25,
                "operational_account_weight": 0.10,
                "price_nature": "conditional_entry",
                "plan_kind": "breakout_or_pullback",
                "price_low": 14.8 + rank,
                "price_high": 15.1 + rank,
                "trigger_price": 15.3 + rank,
                "evaluation_price": 15.0 + rank,
                "confirmation_rule": "completed_close_confirmation",
                "invalidation_price": 14.2 + rank,
                "plan_cutoff": PLAN_CUTOFF.isoformat(),
                "plan_sessions": 20,
                "plan_method_version": "structured-plan-v1",
                "price_condition": "archived structured condition",
                "evidence_pending": False,
                "primary_timeframe": "daily",
                "primary_structure": "near_breakout",
                "entry_rule_json": {"kind": "close_confirmation"},
            }
            for rank, symbol in enumerate(SYMBOLS, start=1)
        ]
        repository.archive_recommendation_report(
            report,
            batches=[batch],
            members=members,
        )

    portfolio_metadata = (
        {
            "origin_recommendation_report_id": "report-origin",
            "origin_recommendation_batch_id": "batch-origin-4w",
        }
        if linked_origin
        else {}
    )
    replace_active_holdings(
        repository,
        tuple(
            HoldingPositionInput(
                symbol=symbol,
                name=f"样本{rank}",
                entry_date=CUTOFF,
                cost_price=987.65 + rank,
                stock_sleeve_weight=0.25,
                account_weight=0.10,
            )
            for rank, symbol in enumerate(SYMBOLS, start=1)
        ),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 28, 1, 0, tzinfo=UTC),
        metadata=portfolio_metadata,
    )
    return repository


def _loaded(repository: SQLiteRepository) -> ActiveHoldingHistoryLoad:
    portfolio = get_active_holding_portfolio(repository, as_of=CUTOFF)
    assert portfolio is not None
    frames = {
        symbol: pd.DataFrame(
            {
                "trade_date": pd.to_datetime([CUTOFF]),
                "open": [10.0],
                "high": [10.2],
                "low": [9.9],
                "close": [10.1],
                "amount_cny": [123_456_789.0],
            }
        )
        for symbol in SYMBOLS
    }
    return ActiveHoldingHistoryLoad(
        portfolio_id=portfolio.id,
        holding_version=portfolio.version,
        holding_weeks=portfolio.holding_weeks,
        histories=frames,
        baseline_cutoff=PLAN_CUTOFF,
        verified_overlay_dates=(CUTOFF,),
        data_cutoff=CUTOFF,
        unavailable_symbols=(),
        sources=("CSMAR", "infoway"),
    )


def _review(
    repository: SQLiteRepository,
    *,
    cutoff: date = CUTOFF,
) -> HoldingTreeReviewSummary:
    portfolio = get_active_holding_portfolio(repository, as_of=CUTOFF)
    assert portfolio is not None
    rows = tuple(
        HoldingTreeReviewRow(
            symbol=holding.symbol,
            name=holding.name,
            holding_weeks=portfolio.holding_weeks,
            holding_version=portfolio.version,
            position_key=holding.position_key,
            status=HoldingReviewRowStatus.READY,
            action=HoldingAction.HOLD,
            latest_close=10.1,
            cost_price=holding.cost_price,
            stock_sleeve_weight=holding.stock_sleeve_weight,
            account_weight=holding.account_weight,
            candidate_stop=9.2,
            previous_stop=9.0,
            effective_stop=9.2,
            stop_raised=True,
            close_below_stop=False,
            source_timeframe="daily",
            evidence_date=cutoff,
            slow_direction="up",
            primary_structure="trend_up",
            daily_execution="healthy",
            reasons=(),
        )
        for holding in portfolio.positions
    )
    return HoldingTreeReviewSummary(
        status=HoldingReviewSummaryStatus.READY,
        portfolio_id=portfolio.id,
        holding_version=portfolio.version,
        holding_weeks=portfolio.holding_weeks,
        reviewed_at=datetime(2026, 8, 28, 8, 0, tzinfo=UTC),
        data_cutoff=cutoff,
        rows=rows,
    )


def _build_with_spies(
    repository: SQLiteRepository,
    *,
    review_cutoff: date = CUTOFF,
) -> tuple[object, dict[str, object]]:
    captured: dict[str, object] = {}

    def reviewer(*args: object, **kwargs: object) -> HoldingTreeReviewSummary:
        captured["review_persist"] = kwargs.get("persist")
        return _review(repository, cutoff=review_cutoff)

    def renderer(request: object) -> object:
        captured["request"] = request
        return SimpleNamespace(
            composite_png=b"\x89PNG\r\n\x1a\ncomposite",
            individual_pngs={symbol: b"\x89PNG\r\n\x1a\n" + symbol.encode() for symbol in SYMBOLS},
            metadata=SimpleNamespace(local_only=True, raw_rows_embedded=False),
        )

    result = build_holding_chart_report(
        repository,
        dataset_root="unused",
        overlay_root="unused",
        as_of=CUTOFF,
        _history_loader=lambda *args, **kwargs: _loaded(repository),
        _reviewer=reviewer,
        _renderer=renderer,
    )
    return result, captured


def test_origin_plan_maps_only_structured_prices_and_keeps_cutoff_identity(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)

    result, captured = _build_with_spies(repository)

    assert result.status is HoldingChartBuildStatus.READY
    assert result.data_cutoff == CUTOFF
    assert result.origin_report_id == "report-origin"
    assert result.origin_batch_id == "batch-origin-4w"
    assert result.entry_overlay_count == 4
    assert captured["review_persist"] is False
    request = captured["request"]
    assert request.expected_identity.data_cutoff == CUTOFF
    first = next(item for item in request.entry_overlays if item.symbol == SYMBOLS[0])
    assert first.symbol == SYMBOLS[0]
    assert first.zone_low == pytest.approx(15.8)
    assert first.zone_high == pytest.approx(16.1)
    assert first.trigger_price == pytest.approx(16.3)
    assert first.reference_price == pytest.approx(16.0)
    assert first.source_cutoff == PLAN_CUTOFF
    # An origin plan on a current-holdings chart is history, not renewed action.
    assert first.nature is EntryOverlayNature.HISTORICAL_OBSERVATION


def test_reconstructed_origin_is_explicitly_historical_observation(tmp_path: Path) -> None:
    repository = _repository(tmp_path, archive_nature="reconstructed")

    result, captured = _build_with_spies(repository)

    assert result.origin_nature == "historical_observation"
    assert all(
        item.nature is EntryOverlayNature.HISTORICAL_OBSERVATION
        for item in captured["request"].entry_overlays
    )


def test_missing_origin_plan_is_graceful_and_never_inferred(tmp_path: Path) -> None:
    repository = _repository(tmp_path, linked_origin=False)

    result, captured = _build_with_spies(repository)

    assert result.status is HoldingChartBuildStatus.READY
    assert result.entry_overlay_count == 0
    assert result.origin_report_id is None
    assert captured["request"].entry_overlays == ()
    assert "origin_recommendation_not_linked" in result.reasons


def test_linked_member_without_structured_entry_prices_is_gracefully_omitted(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path, linked_origin=False)
    portfolio = get_active_holding_portfolio(repository, as_of=CUTOFF)
    assert portfolio is not None
    origin = _OriginArchive(
        report_id="report-without-structured-plan",
        batch_id="batch-without-structured-plan",
        archive_nature="reconstructed",
        evaluation_mode="reconstructed_observation",
        overlay_nature=OriginOverlayNature.HISTORICAL_OBSERVATION,
        members=(
            {
                "symbol": SYMBOLS[0],
                "plan_cutoff": PLAN_CUTOFF.isoformat(),
                # Deliberately only prose: the builder must not parse it.
                "price_condition": "大概回踩后观察",
            },
        ),
    )

    assert _entry_overlays(origin, portfolio) == ()


def test_mismatched_review_cutoff_is_rejected_before_render(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    with pytest.raises(DataQualityError, match="identity"):
        _build_with_spies(repository, review_cutoff=PLAN_CUTOFF)


def test_only_company_action_evidence_partial_is_chart_eligible(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    ready = _review(repository)
    blocked_row = replace(
        ready.rows[0],
        status=HoldingReviewRowStatus.DATA_NOT_READY,
        reasons=(
            "company_action_evidence_blocks_exit:clearance_missing",
            "candidate_stop_not_persisted_without_company_action_clearance",
        ),
    )
    allowed = replace(
        ready,
        status=HoldingReviewSummaryStatus.PARTIAL,
        rows=(blocked_row, *ready.rows[1:]),
    )
    ordinary_failure = replace(
        allowed,
        rows=(
            replace(blocked_row, reasons=("holding_history_missing",)),
            *ready.rows[1:],
        ),
    )

    assert _review_is_chart_eligible(allowed)
    assert not _review_is_chart_eligible(ordinary_failure)


def test_private_archive_is_deterministic_idempotent_and_prunes_only_owned_files(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "holding-charts"
    archive.mkdir(mode=0o755)
    old_owned = archive / "holding-chart-20260701-v000001-composite.png"
    old_owned.write_bytes(b"old")
    unrelated = archive / "my-family-photo-20260701.png"
    unrelated.write_bytes(b"keep")
    payload = b"\x89PNG\r\n\x1a\nprivate-composite"
    details = {symbol: b"\x89PNG\r\n\x1a\n" + symbol.encode() for symbol in SYMBOLS}

    first = _archive_holding_chart_report(
        payload,
        directory=archive,
        data_cutoff=CUTOFF,
        portfolio_version=7,
        retention_days=10,
        individual_pngs=details,
    )
    first_mtime = first.composite_path.stat().st_mtime_ns
    second = _archive_holding_chart_report(
        payload,
        directory=archive,
        data_cutoff=CUTOFF,
        portfolio_version=7,
        retention_days=10,
        individual_pngs=details,
    )

    assert first.composite_path == second.composite_path
    assert first.individual_paths == second.individual_paths
    assert first.composite_path.read_bytes() == payload
    assert first.composite_path.stat().st_mtime_ns == first_mtime
    assert stat.S_IMODE(archive.stat().st_mode) == 0o700
    assert stat.S_IMODE(first.composite_path.stat().st_mode) == 0o600
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in first.individual_paths)
    assert not old_owned.exists()
    assert unrelated.read_bytes() == b"keep"
    assert not tuple(archive.glob("*.tmp"))


def test_build_result_and_archive_do_not_serialize_raw_frames_or_account_fields(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)

    result, _ = _build_with_spies(repository)
    text = repr(result).lower()

    assert "dataframe" not in text
    assert "amount_cny" not in text
    assert "cost_price" not in text
    assert "account_weight" not in text
    assert "987.65" not in text


def test_live_chart_uses_latest_holding_authorization_without_advancing_market_cutoff(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    market_cutoff = date(2026, 9, 1)
    authorization_effective_at = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    holding_known_at = datetime(2026, 9, 2, 13, 0, tzinfo=UTC)

    historical = get_active_holding_portfolio(repository, as_of=market_cutoff)
    assert historical is not None
    assert holding_chart_delivery_channels(historical) == frozenset()

    live = replace_active_holdings(
        repository,
        tuple(
            HoldingPositionInput(
                symbol=holding.symbol,
                name=holding.name,
                entry_date=holding.entry_date,
                cost_price=holding.cost_price,
                stock_sleeve_weight=holding.stock_sleeve_weight,
                account_weight=holding.account_weight,
                source=holding.source,
                metadata=holding.metadata,
            )
            for holding in historical.positions
        ),
        holding_weeks=historical.holding_weeks,
        effective_at=authorization_effective_at,
        metadata={
            **historical.metadata,
            "holding_chart_delivery_channels": ["serverchan"],
            "holding_chart_publisher_id": "cloudflare_r2",
        },
        expected_current_revision_id=historical.id,
        expected_current_version=historical.version,
    )
    context = holding_knowledge_context(live, known_at=holding_known_at)
    assert live.version == historical.version + 1
    assert holding_chart_delivery_channels(live) == frozenset({"serverchan"})

    # The new authorization is current knowledge, but it must not be backdated
    # into an ordinary historical holding lookup for the market cutoff.
    historical_replay = get_active_holding_portfolio(repository, as_of=market_cutoff)
    assert historical_replay is not None
    assert (historical_replay.id, historical_replay.version) == (
        historical.id,
        historical.version,
    )
    assert holding_chart_delivery_channels(historical_replay) == frozenset()

    histories = {
        symbol: pd.DataFrame(
            {
                "trade_date": pd.to_datetime([market_cutoff]),
                "open": [10.0],
                "high": [10.2],
                "low": [9.9],
                "close": [10.1],
                "amount_cny": [123_456_789.0],
            }
        )
        for symbol in SYMBOLS
    }
    loaded = ActiveHoldingHistoryLoad(
        portfolio_id=live.id,
        holding_version=live.version,
        holding_weeks=live.holding_weeks,
        histories=histories,
        baseline_cutoff=CUTOFF,
        verified_overlay_dates=(market_cutoff,),
        data_cutoff=market_cutoff,
        unavailable_symbols=(),
        sources=("CSMAR", "verified-overlay"),
    )
    historical_review = _review(repository, cutoff=market_cutoff)
    live_review = replace(
        historical_review,
        portfolio_id=live.id,
        holding_version=live.version,
        reviewed_at=holding_known_at,
        data_cutoff=market_cutoff,
        rows=tuple(
            replace(
                row,
                holding_version=live.version,
                evidence_date=market_cutoff,
            )
            for row in historical_review.rows
        ),
    )
    captured: dict[str, object] = {}

    def history_loader(*args: object, **kwargs: object) -> ActiveHoldingHistoryLoad:
        assert kwargs["as_of"] == market_cutoff
        assert kwargs["_holding_context"] == context
        return loaded

    def reviewer(*args: object, **kwargs: object) -> HoldingTreeReviewSummary:
        assert kwargs["as_of"] == market_cutoff
        assert kwargs["verified_data_cutoff"] == market_cutoff
        assert kwargs["reviewed_at"] == holding_known_at
        assert kwargs["holding_context"] == context
        return live_review

    def renderer(request: object) -> object:
        captured["request"] = request
        return SimpleNamespace(
            composite_png=b"\x89PNG\r\n\x1a\nlive-context",
            individual_pngs={},
            metadata=SimpleNamespace(local_only=True, raw_rows_embedded=False),
        )

    result = build_holding_chart_report(
        repository,
        dataset_root="unused",
        overlay_root="unused",
        as_of=market_cutoff,
        reviewed_at=holding_known_at,
        holding_context=context,
        _history_loader=history_loader,
        _reviewer=reviewer,
        _renderer=renderer,
    )

    assert result.status is HoldingChartBuildStatus.READY
    assert (result.portfolio_id, result.holding_version) == (live.id, live.version)
    assert result.data_cutoff == market_cutoff
    request = captured["request"]
    assert (request.expected_identity.portfolio_id, request.expected_identity.holding_version) == (
        live.id,
        live.version,
    )
    assert request.expected_identity.data_cutoff == market_cutoff

    historical_loaded = replace(
        loaded,
        portfolio_id=historical.id,
        holding_version=historical.version,
    )

    def historical_loader(*args: object, **kwargs: object) -> ActiveHoldingHistoryLoad:
        assert kwargs["as_of"] == market_cutoff
        assert kwargs["_holding_context"] is None
        return historical_loaded

    def historical_reviewer(*args: object, **kwargs: object) -> HoldingTreeReviewSummary:
        assert kwargs["as_of"] == market_cutoff
        assert kwargs["verified_data_cutoff"] == market_cutoff
        assert kwargs["holding_context"] is None
        return historical_review

    historical_result = build_holding_chart_report(
        repository,
        dataset_root="unused",
        overlay_root="unused",
        as_of=market_cutoff,
        reviewed_at=holding_known_at,
        _history_loader=historical_loader,
        _reviewer=historical_reviewer,
        _renderer=renderer,
    )

    assert historical_result.status is HoldingChartBuildStatus.READY
    assert (historical_result.portfolio_id, historical_result.holding_version) == (
        historical.id,
        historical.version,
    )
    assert historical_result.data_cutoff == market_cutoff
