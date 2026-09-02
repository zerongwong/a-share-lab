from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from ashare_lab.services.render_holding_chart_report import (
    COMPOSITE_HEIGHT,
    COMPOSITE_WIDTH,
    HOLDING_CHART_COLORS,
    EntryOverlayNature,
    HoldingChartConfirmedPivot,
    HoldingChartEntryOverlay,
    HoldingChartReportRequest,
    HoldingChartReviewIdentity,
    ProtectiveLineNature,
    _fit_text,
    _label_height,
    _LabelSpec,
    _load_font,
    _merge_entry_label_specs,
    _resolve_label_positions,
    render_holding_chart_report,
)
from ashare_lab.services.review_active_holdings import (
    HoldingAction,
    HoldingReviewRowStatus,
    HoldingReviewSummaryStatus,
    HoldingTreeReviewRow,
    HoldingTreeReviewSummary,
)

CUTOFF = date(2026, 8, 26)  # Wednesday: the current week must not enter weekly structure.
SYMBOLS = ("600919", "601156", "601919", "002142")
NAMES = ("江苏银行", "东航物流", "中远海控", "宁波银行")


def _history(offset: float) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-08-28", periods=240)
    index = np.arange(len(dates), dtype=float)
    close = offset + 0.018 * index + 0.22 * np.sin(index / 9.0)
    open_price = close * (1.0 + 0.004 * np.sin(index / 5.0))
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": open_price,
            "high": np.maximum(open_price, close) * 1.012,
            "low": np.minimum(open_price, close) * 0.988,
            "close": close,
            "volume_shares": 2_000_000.0 + index * 1_000.0,
            "amount_cny": 40_000_000.0 + index * 20_000.0,
        }
    )


def _histories() -> dict[str, pd.DataFrame]:
    return {symbol: _history(8.0 + position) for position, symbol in enumerate(SYMBOLS)}


def _close_at_cutoff(frame: pd.DataFrame) -> float:
    dates = pd.to_datetime(frame["trade_date"]).dt.date
    return float(frame.loc[dates == CUTOFF, "close"].iloc[-1])


def _review(histories: dict[str, pd.DataFrame]) -> HoldingTreeReviewSummary:
    actions = (
        HoldingAction.HOLD,
        HoldingAction.TIGHTEN,
        HoldingAction.REDUCE,
        HoldingAction.EXIT,
    )
    rows = []
    for index, (symbol, name, action) in enumerate(zip(SYMBOLS, NAMES, actions, strict=True)):
        close = _close_at_cutoff(histories[symbol])
        rows.append(
            HoldingTreeReviewRow(
                symbol=symbol,
                name=name,
                holding_weeks=13,
                holding_version=7,
                position_key=f"position:{symbol}:v7",
                status=HoldingReviewRowStatus.READY,
                action=action,
                latest_close=close,
                # Distinctive sensitive values prove the renderer uses an allow-list.
                cost_price=987_654.32 + index,
                stock_sleeve_weight=0.25,
                account_weight=0.731,
                candidate_stop=close * 0.91,
                previous_stop=close * 0.89,
                effective_stop=close * 0.91,
                stop_raised=True,
                close_below_stop=False,
                source_timeframe="weekly_completed",
                evidence_date=date(2026, 7, 20),
                slow_direction="up",
                primary_structure="healthy_pullback",
                daily_execution="confirmed",
                reasons=("no_completed_close_exit_or_reduce_signal", "ACCOUNT_SECRET_42"),
                company_action_clear=True,
            )
        )
    return HoldingTreeReviewSummary(
        status=HoldingReviewSummaryStatus.READY,
        portfolio_id="holding-review:exact-v7",
        holding_version=7,
        holding_weeks=13,
        reviewed_at=datetime(2026, 8, 26, 16, 0, tzinfo=UTC),
        data_cutoff=CUTOFF,
        rows=tuple(rows),
    )


def _identity() -> HoldingChartReviewIdentity:
    return HoldingChartReviewIdentity(
        portfolio_id="holding-review:exact-v7",
        holding_version=7,
        holding_weeks=13,
        data_cutoff=CUTOFF,
    )


def _request(
    histories: dict[str, pd.DataFrame] | None = None,
    review: HoldingTreeReviewSummary | None = None,
) -> HoldingChartReportRequest:
    histories = _histories() if histories is None else histories
    review = _review(histories) if review is None else review
    first_close = _close_at_cutoff(histories[SYMBOLS[0]])
    second_close = _close_at_cutoff(histories[SYMBOLS[1]])
    return HoldingChartReportRequest(
        review=review,
        histories=histories,
        expected_identity=_identity(),
        entry_overlays=(
            HoldingChartEntryOverlay(
                symbol=SYMBOLS[0],
                trigger_price=first_close * 1.01,
                reference_price=first_close * 0.99,
                zone_low=first_close * 0.97,
                zone_high=first_close * 0.985,
                nature=EntryOverlayNature.HISTORICAL_OBSERVATION,
                source_cutoff=date(2026, 8, 10),
            ),
            # Compatibility alias: the renderer resolves the omitted current
            # source cutoff to the exact review cutoff rather than backfilling it.
            HoldingChartEntryOverlay(symbol=SYMBOLS[1], line_price=second_close * 1.01),
        ),
        confirmed_pivots=(
            HoldingChartConfirmedPivot(
                symbol=SYMBOLS[0],
                price=first_close * 0.95,
                confirmed_on=date(2026, 8, 5),
            ),
        ),
    )


def test_render_valid_deterministic_png_with_eight_audited_panels() -> None:
    histories = _histories()
    result = render_holding_chart_report(_request(histories))

    with Image.open(BytesIO(result.composite_png)) as image:
        assert image.format == "PNG"
        assert image.size == (COMPOSITE_WIDTH, COMPOSITE_HEIGHT)
        assert image.mode == "RGB"
        # No text/source-path/account payload is attached as a PNG metadata chunk.
        assert not image.info
        pixels = np.asarray(image.convert("RGB"))
        assert bool(np.all(pixels == HOLDING_CHART_COLORS["entry_zone"], axis=2).any())
        assert bool(np.all(pixels == HOLDING_CHART_COLORS["stop"], axis=2).any())
        assert bool(np.all(pixels == HOLDING_CHART_COLORS["pivot"], axis=2).any())
        panel_top = 112
        panel_gap = 12
        panel_height = (COMPOSITE_HEIGHT - panel_top - 24 - 7 * panel_gap) // 8
        for index in range(8):
            y0 = panel_top + index * (panel_height + panel_gap)
            assert tuple(pixels[y0 + 5, COMPOSITE_WIDTH // 2]) == HOLDING_CHART_COLORS["panel"]
            if index < 7:
                assert (
                    tuple(pixels[y0 + panel_height + panel_gap // 2, COMPOSITE_WIDTH // 2])
                    == HOLDING_CHART_COLORS["canvas"]
                )

    assert result.metadata.mime_type == "image/png"
    assert result.metadata.panel_count == 8
    assert result.metadata.layout == "single_column_8_panels"
    assert result.metadata.width == 1200
    assert 3200 <= result.metadata.height <= 3800
    assert result.metadata.symbols == SYMBOLS
    assert len(result.metadata.panels) == 8
    assert {panel.timeframe for panel in result.metadata.panels} == {
        "daily",
        "weekly_completed",
    }
    assert all(panel.protective_stop_drawn for panel in result.metadata.panels)
    assert result.metadata.individual_image_count == 4
    assert set(result.individual_pngs) == set(SYMBOLS)
    for png in result.individual_pngs.values():
        with Image.open(BytesIO(png)) as image:
            assert image.format == "PNG"
            assert image.size == (1800, 500)

    # Same completed data and identity always produce identical output bytes.
    assert render_holding_chart_report(_request(histories)).composite_png == result.composite_png


def test_cutoff_excludes_future_daily_rows_and_incomplete_week() -> None:
    histories = _histories()
    full = render_holding_chart_report(_request(histories))
    clipped = {
        symbol: frame.loc[pd.to_datetime(frame["trade_date"]).dt.date <= CUTOFF].copy()
        for symbol, frame in histories.items()
    }
    without_future = render_holding_chart_report(_request(clipped, _review(histories)))

    assert full.composite_png == without_future.composite_png
    daily = [panel for panel in full.metadata.panels if panel.timeframe == "daily"]
    weekly = [panel for panel in full.metadata.panels if panel.timeframe == "weekly_completed"]
    assert all(panel.series_end == CUTOFF for panel in daily)
    assert all(panel.bar_count == 150 for panel in daily)
    assert all(panel.incomplete_current_week_excluded for panel in weekly)
    assert all(panel.series_end == date(2026, 8, 21) for panel in weekly)
    assert all(panel.bar_count == 47 for panel in weekly)


def test_overlay_and_protective_lines_start_only_at_evidence_dates() -> None:
    result = render_holding_chart_report(_request())
    first_daily = result.metadata.panels[0]
    first_weekly = result.metadata.panels[1]
    second_daily = result.metadata.panels[2]

    assert first_daily.entry_nature == EntryOverlayNature.HISTORICAL_OBSERVATION.value
    assert first_daily.entry_overlay_start == date(2026, 8, 10)
    assert first_weekly.entry_overlay_start == date(2026, 8, 10)
    assert first_daily.pivot_start == date(2026, 8, 5)
    assert first_daily.protective_line_start == date(2026, 7, 20)
    assert second_daily.entry_overlay_start == CUTOFF

    # The historical observation zone begins near its source date, not at the
    # left edge of the 150-bar history.  This pixel assertion guards the actual
    # drawing behaviour in addition to the returned audit metadata.
    with Image.open(BytesIO(result.composite_png)) as image:
        pixels = np.asarray(image.convert("RGB"))
    first_panel = pixels[112:536, 24:1176]
    zone_mask = np.all(first_panel == HOLDING_CHART_COLORS["entry_zone"], axis=2)
    zone_x = np.flatnonzero(zone_mask.any(axis=0))
    assert zone_x.size > 0
    assert int(zone_x.min()) > 900
    assert first_daily.historical_warning_count == 1
    assert first_weekly.historical_warning_count == 1
    assert second_daily.historical_warning_count == 0


def test_label_merge_and_collision_resolution_are_bounded_and_deterministic() -> None:
    image = Image.new("RGB", (800, 320), HOLDING_CHART_COLORS["panel"])
    draw = ImageDraw.Draw(image)
    font = _load_font(16)
    entry_specs = (
        _LabelSpec(100, "原观察区 10.00–10.10", HOLDING_CHART_COLORS["entry"]),
        _LabelSpec(108, "原观察参考 10.08", HOLDING_CHART_COLORS["entry"]),
        _LabelSpec(116, "原观察触发 10.12", HOLDING_CHART_COLORS["entry"]),
    )
    merged = _merge_entry_label_specs(entry_specs, prefix="原观察")
    assert len(merged) == 1
    assert merged[0].label.count("原观察") == 1
    assert all(word in merged[0].label for word in ("区", "参考", "触发"))

    specs = (
        *merged,
        _LabelSpec(104, "确认基准点 9.90", HOLDING_CHART_COLORS["pivot"]),
        _LabelSpec(110, "保护参考（公司行动待核验） 9.80", HOLDING_CHART_COLORS["stop"]),
    )
    first = _resolve_label_positions(specs, top=40, bottom=260, draw=draw, font=font)
    second = _resolve_label_positions(specs, top=40, bottom=260, draw=draw, font=font)
    assert first == second
    centers = [center for _spec, center in first]
    height = _label_height(draw, font)
    assert centers == sorted(centers)
    assert centers[0] - (height + 1) // 2 >= 40
    assert centers[-1] + (height + 1) // 2 <= 260
    assert all(
        right - left >= height + 5 for left, right in zip(centers, centers[1:], strict=False)
    )

    fitted = _fit_text(draw, "保护参考（公司行动待核验）" * 8, font, max_width=180)
    assert fitted.endswith("…")
    assert draw.textlength(fitted, font=font) <= 180


def test_company_action_only_partial_renders_unverified_reference_and_review_badge() -> None:
    histories = _histories()
    review = _review(histories)
    blocked = replace(
        review.rows[0],
        status=HoldingReviewRowStatus.DATA_NOT_READY,
        action=HoldingAction.REVIEW,
        reasons=(
            "company_action_evidence_blocks_exit:company_action_clearance_missing_or_stale",
            "candidate_stop_not_persisted_without_company_action_clearance",
        ),
        company_action_clear=None,
    )
    hold_without_clearance = replace(
        review.rows[1],
        action=HoldingAction.HOLD,
        reasons=(
            "company_action_clearance_missing_non_destructive_hold_only",
            "candidate_stop_not_persisted_without_company_action_clearance",
        ),
        company_action_clear=None,
    )
    partial = replace(
        review,
        status=HoldingReviewSummaryStatus.PARTIAL,
        rows=(blocked, hold_without_clearance, *review.rows[2:]),
    )
    result = render_holding_chart_report(_request(histories, partial))

    assert result.metadata.panels[0].action == HoldingAction.REVIEW.value
    assert (
        result.metadata.panels[0].protective_line_nature
        == ProtectiveLineNature.UNVERIFIED_REFERENCE.value
    )
    assert (
        result.metadata.panels[2].protective_line_nature
        == ProtectiveLineNature.UNVERIFIED_REFERENCE.value
    )
    with Image.open(BytesIO(result.composite_png)) as image:
        pixels = np.asarray(image.convert("RGB"))
    assert bool(np.all(pixels == HOLDING_CHART_COLORS["action_review"], axis=2).any())


def test_output_metadata_and_png_never_embed_account_fields() -> None:
    result = render_holding_chart_report(_request())
    serialized = json.dumps(asdict(result.metadata), ensure_ascii=False, default=str)
    lowered = serialized.lower()

    for forbidden_field in ("cost", "shares", "account", "weight", "amount", "secret"):
        assert forbidden_field not in lowered
    for forbidden_value in ("987654.32", "73.1%", "ACCOUNT_SECRET_42"):
        assert forbidden_value not in serialized
        assert forbidden_value.encode() not in result.composite_png
    assert result.metadata.local_only is True
    assert result.metadata.raw_rows_embedded is False
    assert result.metadata.sensitive_fields_embedded is False


def test_invalid_identity_count_symbols_and_stale_history_fail_closed() -> None:
    histories = _histories()
    review = _review(histories)

    with pytest.raises(ValueError, match="identity mismatch"):
        render_holding_chart_report(
            replace(
                _request(histories, review),
                expected_identity=replace(_identity(), holding_version=8),
            )
        )
    with pytest.raises(ValueError, match="exactly 4"):
        render_holding_chart_report(_request(histories, replace(review, rows=review.rows[:3])))

    duplicate_histories = dict(histories)
    duplicate_histories["600919.SH"] = histories["600919"]
    with pytest.raises(ValueError, match="duplicate normalized history"):
        render_holding_chart_report(_request(duplicate_histories, review))

    stale = dict(histories)
    stale[SYMBOLS[0]] = stale[SYMBOLS[0]].loc[
        pd.to_datetime(stale[SYMBOLS[0]]["trade_date"]).dt.date != CUTOFF
    ]
    with pytest.raises(ValueError, match="cutoff mismatch"):
        render_holding_chart_report(replace(_request(histories, review), histories=stale))


def test_non_company_action_partial_and_future_evidence_fail_closed() -> None:
    histories = _histories()
    review = _review(histories)
    missing_price = replace(
        review.rows[0],
        status=HoldingReviewRowStatus.DATA_NOT_READY,
        action=HoldingAction.REVIEW,
        latest_close=None,
        reasons=("holding_data_not_ready:missing_close",),
    )
    invalid_partial = replace(
        review,
        status=HoldingReviewSummaryStatus.PARTIAL,
        rows=(missing_price, *review.rows[1:]),
    )
    with pytest.raises(ValueError, match="not price-ready"):
        render_holding_chart_report(_request(histories, invalid_partial))

    future_pivot = HoldingChartConfirmedPivot(
        symbol=SYMBOLS[0],
        price=10.0,
        confirmed_on=date(2026, 8, 27),
    )
    with pytest.raises(ValueError, match="future confirmed pivot"):
        render_holding_chart_report(
            replace(_request(histories, review), confirmed_pivots=(future_pivot,))
        )

    historical_without_cutoff = HoldingChartEntryOverlay(
        symbol=SYMBOLS[0],
        trigger_price=10.0,
        nature=EntryOverlayNature.HISTORICAL_OBSERVATION,
    )
    with pytest.raises(ValueError, match="requires source_cutoff"):
        render_holding_chart_report(
            replace(_request(histories, review), entry_overlays=(historical_without_cutoff,))
        )
