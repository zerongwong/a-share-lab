"""Render local-only holding charts without uploading licensed market data.

The renderer deliberately has no notification or network dependency.  It turns
already-authorized local daily bars and an exact holding-review snapshot into a
pastel PNG report.  The resulting image still depicts licensed prices and must
therefore remain on the user's machine unless a separate redistribution right
has been established.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from enum import StrEnum
from io import BytesIO
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ashare_lab.analytics.multi_timeframe import (
    MultiTimeframeDataError,
    build_completed_timeframes,
)
from ashare_lab.ports.market_data import normalize_symbol
from ashare_lab.services.review_active_holdings import (
    HoldingAction,
    HoldingReviewRowStatus,
    HoldingReviewSummaryStatus,
    HoldingTreeReviewSummary,
)

COMPOSITE_WIDTH: Final = 1200
# Preserve the existing maximum external-image dimensions, including for five
# holdings. Four holdings retain their original 1200-by-3600 layout.
COMPOSITE_HEIGHT: Final = 3600
INDIVIDUAL_WIDTH: Final = 1800
INDIVIDUAL_HEIGHT: Final = 500
MIN_HOLDING_COUNT: Final = 1
MAX_HOLDING_COUNT: Final = 5
DAILY_BAR_LIMIT: Final = 150
WEEKLY_BAR_LIMIT: Final = 52

# Stable semantic colours.  Red is an advancing A-share candle, green is a
# declining candle, blue denotes a model condition, amber a confirmed pivot,
# and coral the effective protective line.
HOLDING_CHART_COLORS: Final = MappingProxyType(
    {
        "canvas": (245, 245, 252),
        "panel": (252, 253, 255),
        "grid": (225, 229, 239),
        "text": (50, 54, 63),
        "muted_text": (112, 116, 124),
        "up": (214, 112, 120),
        "down": (105, 164, 145),
        "ma_fast": (214, 169, 82),
        "ma_slow": (116, 139, 196),
        "entry": (101, 143, 204),
        "entry_zone": (230, 239, 251),
        "pivot": (147, 122, 184),
        "stop": (211, 92, 99),
        "volume_up": (231, 177, 181),
        "volume_down": (170, 207, 194),
        "white": (255, 255, 255),
        "action_hold": (83, 151, 147),
        "action_tighten": (210, 150, 72),
        "action_reduce": (219, 132, 111),
        "action_exit": (208, 86, 96),
        "action_review": (145, 112, 181),
    }
)


class EntryOverlayNature(StrEnum):
    """Provenance label shown on the chart, never an instruction to trade."""

    CURRENT_CONDITION = "current_condition"
    HISTORICAL_OBSERVATION = "historical_observation"


class ProtectiveLineNature(StrEnum):
    EFFECTIVE_STOP = "effective_stop"
    UNVERIFIED_REFERENCE = "unverified_reference"


@dataclass(frozen=True, slots=True)
class HoldingChartReviewIdentity:
    portfolio_id: str
    holding_version: int
    holding_weeks: int
    data_cutoff: date

    def __post_init__(self) -> None:
        portfolio_id = self.portfolio_id.strip()
        if not portfolio_id:
            raise ValueError("portfolio_id must not be empty")
        if isinstance(self.holding_version, bool) or self.holding_version < 1:
            raise ValueError("holding_version must be a positive integer")
        if isinstance(self.holding_weeks, bool) or self.holding_weeks < 1:
            raise ValueError("holding_weeks must be a positive integer")
        if not isinstance(self.data_cutoff, date):
            raise TypeError("data_cutoff must be a date")
        object.__setattr__(self, "portfolio_id", portfolio_id)


@dataclass(frozen=True, slots=True)
class HoldingChartEntryOverlay:
    """Optional structured model condition for one holding.

    ``line_price`` is retained as a compatibility alias for
    ``trigger_price``.  Free-form labels are intentionally forbidden so costs,
    quantities, account values, or unsupported action language cannot leak into
    the image.
    """

    symbol: str
    line_price: float | None = None
    trigger_price: float | None = None
    reference_price: float | None = None
    zone_low: float | None = None
    zone_high: float | None = None
    nature: EntryOverlayNature | str = EntryOverlayNature.CURRENT_CONDITION
    source_cutoff: date | None = None

    def __post_init__(self) -> None:
        symbol = normalize_symbol(self.symbol)
        try:
            nature = EntryOverlayNature(self.nature)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported entry overlay nature") from exc
        values = {
            "line_price": self.line_price,
            "trigger_price": self.trigger_price,
            "reference_price": self.reference_price,
            "zone_low": self.zone_low,
            "zone_high": self.zone_high,
        }
        for label, value in values.items():
            if value is not None and not _positive_finite(value):
                raise ValueError(f"{label} must be a positive finite price")
        if (self.zone_low is None) != (self.zone_high is None):
            raise ValueError("entry zone requires both zone_low and zone_high")
        if (
            self.zone_low is not None
            and self.zone_high is not None
            and float(self.zone_low) > float(self.zone_high)
        ):
            raise ValueError("zone_low cannot exceed zone_high")
        if (
            self.line_price is not None
            and self.trigger_price is not None
            and abs(float(self.line_price) - float(self.trigger_price)) > 1e-9
        ):
            raise ValueError("line_price and trigger_price disagree")
        if not any(value is not None for value in values.values()):
            raise ValueError("entry overlay must contain a line, reference, or zone")
        if self.source_cutoff is not None and not isinstance(self.source_cutoff, date):
            raise TypeError("source_cutoff must be a date or None")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "nature", nature)

    @property
    def effective_trigger_price(self) -> float | None:
        value = self.trigger_price if self.trigger_price is not None else self.line_price
        return None if value is None else float(value)


@dataclass(frozen=True, slots=True)
class HoldingChartConfirmedPivot:
    symbol: str
    price: float
    confirmed_on: date
    kind: Literal["support", "resistance"] = "support"

    def __post_init__(self) -> None:
        symbol = normalize_symbol(self.symbol)
        if not _positive_finite(self.price):
            raise ValueError("pivot price must be positive and finite")
        if not isinstance(self.confirmed_on, date):
            raise TypeError("confirmed_on must be a date")
        if self.kind not in {"support", "resistance"}:
            raise ValueError("pivot kind must be support or resistance")
        object.__setattr__(self, "symbol", symbol)


@dataclass(frozen=True, slots=True)
class HoldingChartReportRequest:
    review: HoldingTreeReviewSummary
    histories: Mapping[str, pd.DataFrame]
    expected_identity: HoldingChartReviewIdentity
    entry_overlays: tuple[HoldingChartEntryOverlay, ...] = ()
    confirmed_pivots: tuple[HoldingChartConfirmedPivot, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.histories, Mapping):
            raise TypeError("histories must be a mapping")
        object.__setattr__(self, "entry_overlays", tuple(self.entry_overlays))
        object.__setattr__(self, "confirmed_pivots", tuple(self.confirmed_pivots))


@dataclass(frozen=True, slots=True)
class HoldingChartPanelMetadata:
    symbol: str
    timeframe: Literal["daily", "weekly_completed"]
    bar_count: int
    series_start: date
    series_end: date
    activity_source: Literal["volume", "traded_value"] | None
    incomplete_current_week_excluded: bool
    action: str
    entry_overlay_drawn: bool
    entry_nature: str | None
    entry_overlay_start: date | None
    historical_warning_count: int
    pivot_drawn: bool
    pivot_start: date | None
    protective_stop_drawn: bool
    protective_line_nature: str
    protective_line_start: date


@dataclass(frozen=True, slots=True)
class HoldingChartReportMetadata:
    mime_type: str
    width: int
    height: int
    panel_count: int
    review_identity: HoldingChartReviewIdentity
    symbols: tuple[str, ...]
    panels: tuple[HoldingChartPanelMetadata, ...]
    individual_image_count: int
    layout: str = "single_column_8_panels"
    local_only: bool = True
    raw_rows_embedded: bool = False
    sensitive_fields_embedded: bool = False


@dataclass(frozen=True, slots=True)
class RenderedHoldingChartReport:
    composite_png: bytes
    metadata: HoldingChartReportMetadata
    individual_pngs: Mapping[str, bytes] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _PreparedHolding:
    symbol: str
    name: str
    daily: pd.DataFrame
    weekly: pd.DataFrame
    action: HoldingAction
    effective_stop: float
    protective_line_nature: ProtectiveLineNature
    protective_line_start: date
    entry_overlay: HoldingChartEntryOverlay | None
    pivot: HoldingChartConfirmedPivot | None
    incomplete_week_excluded: bool


@dataclass(frozen=True, slots=True)
class _LabelSpec:
    target_y: int
    label: str
    color: tuple[int, int, int]


def render_holding_chart_report(
    request: HoldingChartReportRequest,
) -> RenderedHoldingChartReport:
    """Return a local composite, one detail PNG per holding, and safe metadata.

    The function fails closed unless the request identifies exactly the same
    ready one-to-five-position snapshot and every history reaches that snapshot's
    verified cutoff.  Future rows are sliced away before every calculation.
    """

    if not isinstance(request, HoldingChartReportRequest):
        raise TypeError("request must be a HoldingChartReportRequest")
    rows = _validated_review_rows(request.review, request.expected_identity)
    histories = _normalized_history_mapping(request.histories)
    entries = _overlay_mapping(request.entry_overlays, request.expected_identity)
    pivots = _pivot_mapping(request.confirmed_pivots, request.expected_identity)
    symbols = tuple(normalize_symbol(row.symbol) for row in rows)
    unknown_entries = set(entries).difference(symbols)
    unknown_pivots = set(pivots).difference(symbols)
    if unknown_entries:
        raise ValueError(f"entry overlay does not belong to review: {sorted(unknown_entries)!r}")
    if unknown_pivots:
        raise ValueError(f"pivot does not belong to review: {sorted(unknown_pivots)!r}")

    prepared: list[_PreparedHolding] = []
    for row in rows:
        symbol = normalize_symbol(row.symbol)
        frame = histories.get(symbol)
        if frame is None:
            raise ValueError(f"history missing for reviewed symbol: {symbol}")
        try:
            completed = build_completed_timeframes(
                frame,
                as_of=request.expected_identity.data_cutoff,
            )
        except (MultiTimeframeDataError, TypeError, ValueError) as exc:
            raise ValueError(f"history is not chart-ready for {symbol}: {exc}") from exc
        if completed.data_cutoff.date() != request.expected_identity.data_cutoff:
            raise ValueError(f"history cutoff mismatch for {symbol}")
        if len(completed.daily) < 60:
            raise ValueError(f"at least 60 completed daily bars required for {symbol}")
        if len(completed.weekly) < 26:
            raise ValueError(f"at least 26 completed weekly bars required for {symbol}")
        latest_close = float(completed.daily.iloc[-1]["close"])
        if row.latest_close is None or not _same_price(latest_close, row.latest_close):
            raise ValueError(f"review close does not match history cutoff for {symbol}")
        daily = _with_averages(completed.daily, (20, 60), prefix="ma").tail(DAILY_BAR_LIMIT)
        weekly = _with_averages(completed.weekly, (8, 26), prefix="w").tail(WEEKLY_BAR_LIMIT)
        protective_nature = _protective_line_nature(row)
        if row.evidence_date is None:
            raise ValueError(f"protective-line evidence date unavailable for {symbol}")
        prepared.append(
            _PreparedHolding(
                symbol=symbol,
                name=_safe_name(row.name),
                daily=daily.reset_index(drop=True),
                weekly=weekly.reset_index(drop=True),
                action=row.action,
                effective_stop=float(row.effective_stop),
                protective_line_nature=protective_nature,
                protective_line_start=row.evidence_date,
                entry_overlay=entries.get(symbol),
                pivot=pivots.get(symbol),
                incomplete_week_excluded=completed.incomplete_week_excluded,
            )
        )

    font_book = _FontBook()
    composite = _render_composite(tuple(prepared), request.expected_identity, font_book)
    individual = MappingProxyType(
        {
            holding.symbol: _encode_png(
                _render_individual(holding, request.expected_identity, font_book)
            )
            for holding in prepared
        }
    )
    panels = tuple(
        panel
        for holding in prepared
        for panel in (
            _panel_metadata(holding, holding.daily, timeframe="daily"),
            _panel_metadata(holding, holding.weekly, timeframe="weekly_completed"),
        )
    )
    metadata = HoldingChartReportMetadata(
        mime_type="image/png",
        width=COMPOSITE_WIDTH,
        height=holding_chart_composite_height(len(prepared)),
        panel_count=len(panels),
        review_identity=request.expected_identity,
        symbols=tuple(item.symbol for item in prepared),
        panels=panels,
        individual_image_count=len(individual),
        layout=f"single_column_{len(panels)}_panels",
    )
    return RenderedHoldingChartReport(
        composite_png=_encode_png(composite),
        metadata=metadata,
        individual_pngs=individual,
    )


def _validated_review_rows(
    review: HoldingTreeReviewSummary,
    expected: HoldingChartReviewIdentity,
) -> tuple[object, ...]:
    if not isinstance(review, HoldingTreeReviewSummary):
        raise TypeError("review must be a HoldingTreeReviewSummary")
    actual = HoldingChartReviewIdentity(
        portfolio_id=review.portfolio_id or "",
        holding_version=review.holding_version or 0,
        holding_weeks=review.holding_weeks or 0,
        data_cutoff=review.data_cutoff or date.min,
    )
    if actual != expected:
        raise ValueError("holding review identity mismatch")
    if review.status not in {
        HoldingReviewSummaryStatus.READY,
        HoldingReviewSummaryStatus.PARTIAL,
    }:
        raise ValueError("holding review must be ready or company-action-only partial")
    if not MIN_HOLDING_COUNT <= len(review.rows) <= MAX_HOLDING_COUNT:
        raise ValueError("one to five reviewed holdings required")
    symbols: set[str] = set()
    position_keys: set[str] = set()
    for row in review.rows:
        symbol = normalize_symbol(row.symbol)
        if symbol in symbols:
            raise ValueError("review symbols must be unique")
        symbols.add(symbol)
        if not row.position_key or row.position_key in position_keys:
            raise ValueError("review position identities must be unique and non-empty")
        position_keys.add(row.position_key)
        company_action_block = _company_action_blocked(row)
        if row.status is not HoldingReviewRowStatus.READY and not company_action_block:
            raise ValueError(f"holding row is not price-ready: {symbol}")
        if row.holding_version != expected.holding_version:
            raise ValueError(f"holding version mismatch for {symbol}")
        if row.holding_weeks != expected.holding_weeks:
            raise ValueError(f"holding horizon mismatch for {symbol}")
        if not _positive_finite(row.effective_stop):
            raise ValueError(f"effective protective stop unavailable for {symbol}")
        if row.evidence_date is not None and row.evidence_date > expected.data_cutoff:
            raise ValueError(f"future stop evidence rejected for {symbol}")
    return tuple(review.rows)


def _company_action_blocked(row: object) -> bool:
    reasons = tuple(getattr(row, "reasons", ()))
    return any(str(reason).startswith("company_action_evidence_blocks_") for reason in reasons)


def _protective_line_nature(row: object) -> ProtectiveLineNature:
    reasons = tuple(str(reason) for reason in getattr(row, "reasons", ()))
    if _company_action_blocked(row) or any(
        reason == "candidate_stop_not_persisted_without_company_action_clearance"
        for reason in reasons
    ):
        return ProtectiveLineNature.UNVERIFIED_REFERENCE
    return ProtectiveLineNature.EFFECTIVE_STOP


def _normalized_history_mapping(
    histories: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    normalized: dict[str, pd.DataFrame] = {}
    for raw_symbol, frame in histories.items():
        symbol = normalize_symbol(str(raw_symbol))
        if symbol in normalized:
            raise ValueError(f"duplicate normalized history symbol: {symbol}")
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"history for {symbol} must be a DataFrame")
        normalized[symbol] = frame
    return normalized


def _overlay_mapping(
    overlays: tuple[HoldingChartEntryOverlay, ...],
    identity: HoldingChartReviewIdentity,
) -> dict[str, HoldingChartEntryOverlay]:
    result: dict[str, HoldingChartEntryOverlay] = {}
    for overlay in overlays:
        if not isinstance(overlay, HoldingChartEntryOverlay):
            raise TypeError("entry_overlays must contain HoldingChartEntryOverlay")
        if overlay.symbol in result:
            raise ValueError(f"duplicate entry overlay: {overlay.symbol}")
        source_cutoff = overlay.source_cutoff
        if overlay.nature is EntryOverlayNature.HISTORICAL_OBSERVATION:
            if source_cutoff is None:
                raise ValueError("historical observation requires source_cutoff")
            if source_cutoff > identity.data_cutoff:
                raise ValueError("future historical observation rejected")
        elif source_cutoff is not None and source_cutoff != identity.data_cutoff:
            raise ValueError("current condition source_cutoff must match review cutoff")
        elif source_cutoff is None:
            overlay = replace(overlay, source_cutoff=identity.data_cutoff)
        result[overlay.symbol] = overlay
    return result


def _pivot_mapping(
    pivots: tuple[HoldingChartConfirmedPivot, ...],
    identity: HoldingChartReviewIdentity,
) -> dict[str, HoldingChartConfirmedPivot]:
    result: dict[str, HoldingChartConfirmedPivot] = {}
    for pivot in pivots:
        if not isinstance(pivot, HoldingChartConfirmedPivot):
            raise TypeError("confirmed_pivots must contain HoldingChartConfirmedPivot")
        if pivot.symbol in result:
            raise ValueError(f"duplicate confirmed pivot: {pivot.symbol}")
        if pivot.confirmed_on > identity.data_cutoff:
            raise ValueError("future confirmed pivot rejected")
        result[pivot.symbol] = pivot
    return result


def _with_averages(
    frame: pd.DataFrame,
    windows: tuple[int, int],
    *,
    prefix: str,
) -> pd.DataFrame:
    result = frame.copy()
    close = result["close"].astype(float)
    for window in windows:
        result[f"{prefix}{window}"] = close.rolling(window, min_periods=window).mean()
    return result


class _FontBook:
    def __init__(self) -> None:
        self.title = _load_font(34)
        self.panel_title = _load_font(28)
        self.body = _load_font(20)
        self.small = _load_font(18)
        self.tiny = _load_font(16)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "DejaVuSans.ttf",
    )
    for candidate in candidates:
        if candidate == "DejaVuSans.ttf" or Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def holding_chart_composite_height(holding_count: int) -> int:
    """Size a nonempty holding report without increasing its existing cap."""

    if (
        isinstance(holding_count, bool)
        or not isinstance(holding_count, int)
        or not MIN_HOLDING_COUNT <= holding_count <= MAX_HOLDING_COUNT
    ):
        raise ValueError("one to five reviewed holdings required")
    return min(900 * holding_count, COMPOSITE_HEIGHT)


def _render_composite(
    holdings: tuple[_PreparedHolding, ...],
    identity: HoldingChartReviewIdentity,
    fonts: _FontBook,
) -> Image.Image:
    height = holding_chart_composite_height(len(holdings))
    panel_count = 2 * len(holdings)
    image = Image.new("RGB", (COMPOSITE_WIDTH, height), HOLDING_CHART_COLORS["canvas"])
    draw = ImageDraw.Draw(image)
    _draw_report_header(draw, identity, fonts, width=COMPOSITE_WIDTH)
    outer_x = 24
    panel_gap = 12
    top = 112
    bottom = 24
    panel_height = (height - top - bottom - (panel_count - 1) * panel_gap) // panel_count
    panel_index = 0
    for holding_index, holding in enumerate(holdings, start=1):
        for timeframe, frame in (
            ("daily", holding.daily),
            ("weekly_completed", holding.weekly),
        ):
            y0 = top + panel_index * (panel_height + panel_gap)
            box = (outer_x, y0, COMPOSITE_WIDTH - outer_x, y0 + panel_height)
            _draw_panel(image, holding, frame, box, timeframe, holding_index, fonts)
            panel_index += 1
    return image


def _render_individual(
    holding: _PreparedHolding,
    identity: HoldingChartReviewIdentity,
    fonts: _FontBook,
) -> Image.Image:
    image = Image.new("RGB", (INDIVIDUAL_WIDTH, INDIVIDUAL_HEIGHT), HOLDING_CHART_COLORS["canvas"])
    draw = ImageDraw.Draw(image)
    _draw_report_header(draw, identity, fonts, width=INDIVIDUAL_WIDTH)
    outer_x = 30
    gap = 20
    top = 96
    panel_width = (INDIVIDUAL_WIDTH - 2 * outer_x - gap) // 2
    left = (outer_x, top, outer_x + panel_width, INDIVIDUAL_HEIGHT - 25)
    right_x = outer_x + panel_width + gap
    right = (right_x, top, right_x + panel_width, INDIVIDUAL_HEIGHT - 25)
    _draw_panel(image, holding, holding.daily, left, "daily", 1, fonts)
    _draw_panel(image, holding, holding.weekly, right, "weekly_completed", 1, fonts)
    return image


def _draw_report_header(
    draw: ImageDraw.ImageDraw,
    identity: HoldingChartReviewIdentity,
    fonts: _FontBook,
    *,
    width: int,
) -> None:
    draw.text(
        (32, 18),
        "持仓趋势与保护线｜收盘复核图",
        font=fonts.title,
        fill=HOLDING_CHART_COLORS["text"],
    )
    note = (
        f"数据截至 {identity.data_cutoff.isoformat()}｜仅使用完整日线与完整周线｜"
        "条件观察，不构成交易指令"
    )
    draw.text(
        (32, 64),
        note,
        font=fonts.body,
        fill=HOLDING_CHART_COLORS["muted_text"],
    )


def _draw_panel(
    image: Image.Image,
    holding: _PreparedHolding,
    frame: pd.DataFrame,
    box: tuple[int, int, int, int],
    timeframe: Literal["daily", "weekly_completed"],
    ordinal: int,
    fonts: _FontBook,
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=HOLDING_CHART_COLORS["panel"])
    period_label = "日线（最多150根）" if timeframe == "daily" else "完整周线（最多52根）"
    title = f"{ordinal}. {holding.name} ({holding.symbol})"
    draw.text((x0 + 18, y0 + 12), title, font=fonts.panel_title, fill=HOLDING_CHART_COLORS["text"])
    draw.text(
        (x0 + 19, y0 + 47),
        period_label,
        font=fonts.small,
        fill=HOLDING_CHART_COLORS["muted_text"],
    )
    _draw_action_badge(draw, holding.action, x1 - 18, y0 + 12, fonts)
    if timeframe == "weekly_completed" and holding.incomplete_week_excluded:
        note = "本周未完成，已排除"
        note_box = draw.textbbox((0, 0), note, font=fonts.small)
        draw.text(
            (x1 - 18 - (note_box[2] - note_box[0]), y0 + 49),
            note,
            font=fonts.small,
            fill=HOLDING_CHART_COLORS["muted_text"],
        )
    entry = holding.entry_overlay
    if entry is not None and entry.nature is EntryOverlayNature.HISTORICAL_OBSERVATION:
        source = entry.source_cutoff.isoformat() if entry.source_cutoff is not None else "待核验"
        draw.text(
            (x0 + 19, y0 + 72),
            f"原观察条件（非行动/非买入许可）｜来源 {source}",
            font=fonts.small,
            fill=HOLDING_CHART_COLORS["entry"],
        )

    plot_left = x0 + 54
    plot_right = x1 - 20
    price_top = y0 + 103
    price_bottom = y1 - 100
    volume_top = y1 - 78
    volume_bottom = y1 - 31
    prices = _price_domain(frame, holding)
    price_min, price_max = prices

    def y_price(value: float) -> int:
        return _map_y(value, price_min, price_max, price_top, price_bottom)

    for step in range(5):
        y = round(price_top + step * (price_bottom - price_top) / 4)
        draw.line((plot_left, y, plot_right, y), fill=HOLDING_CHART_COLORS["grid"], width=1)
        value = price_max - step * (price_max - price_min) / 4
        draw.text(
            (x0 + 6, y - 7),
            f"{value:.2f}",
            font=fonts.tiny,
            fill=HOLDING_CHART_COLORS["muted_text"],
        )

    label_specs: list[_LabelSpec] = []
    if entry is not None:
        label_specs.extend(
            _draw_entry_overlay(
                draw,
                entry,
                frame,
                plot_left,
                plot_right,
                y_price,
            )
        )
    if holding.pivot is not None:
        pivot_y = y_price(holding.pivot.price)
        _draw_horizontal_line(
            draw,
            _date_start_x(frame, holding.pivot.confirmed_on, plot_left, plot_right),
            plot_right,
            pivot_y,
            HOLDING_CHART_COLORS["pivot"],
            dashed=True,
        )
        label_specs.append(
            _LabelSpec(
                target_y=pivot_y,
                label=f"确认基准点 {holding.pivot.price:.2f}",
                color=HOLDING_CHART_COLORS["pivot"],
            )
        )
    protection_label = (
        f"有效保护线 {holding.effective_stop:.2f}"
        if holding.protective_line_nature is ProtectiveLineNature.EFFECTIVE_STOP
        else f"保护参考（公司行动待核验） {holding.effective_stop:.2f}"
    )
    protection_y = y_price(holding.effective_stop)
    _draw_horizontal_line(
        draw,
        _date_start_x(frame, holding.protective_line_start, plot_left, plot_right),
        plot_right,
        protection_y,
        HOLDING_CHART_COLORS["stop"],
        dashed=False,
    )
    label_specs.append(
        _LabelSpec(
            target_y=protection_y,
            label=protection_label,
            color=HOLDING_CHART_COLORS["stop"],
        )
    )

    _draw_candles(draw, frame, plot_left, plot_right, y_price)
    if timeframe == "daily":
        _draw_series(draw, frame, "ma20", plot_left, plot_right, y_price, "ma_fast")
        _draw_series(draw, frame, "ma60", plot_left, plot_right, y_price, "ma_slow")
        legend = "MA20 / MA60"
    else:
        _draw_series(draw, frame, "w8", plot_left, plot_right, y_price, "ma_fast")
        _draw_series(draw, frame, "w26", plot_left, plot_right, y_price, "ma_slow")
        legend = "W8 / W26"
    draw.text(
        (plot_left + 4, price_top + 4),
        legend,
        font=fonts.tiny,
        fill=HOLDING_CHART_COLORS["muted_text"],
    )
    for spec, label_y in _resolve_label_positions(
        tuple(label_specs),
        top=price_top + 8,
        bottom=price_bottom - 8,
        draw=draw,
        font=fonts.tiny,
    ):
        _draw_label(
            draw,
            plot_left,
            plot_right,
            label_y,
            spec.label,
            spec.color,
            fonts,
        )
    activity_source = _activity_source(frame)
    _draw_volume(
        draw,
        frame,
        activity_source,
        plot_left,
        plot_right,
        volume_top,
        volume_bottom,
        fonts,
    )
    start = _frame_date(frame.iloc[0]["trade_date"])
    end = _frame_date(frame.iloc[-1]["trade_date"])
    draw.text(
        (plot_left, y1 - 22),
        start.isoformat(),
        font=fonts.tiny,
        fill=HOLDING_CHART_COLORS["muted_text"],
    )
    end_label = end.isoformat()
    end_box = draw.textbbox((0, 0), end_label, font=fonts.tiny)
    draw.text(
        (plot_right - (end_box[2] - end_box[0]), y1 - 22),
        end_label,
        font=fonts.tiny,
        fill=HOLDING_CHART_COLORS["muted_text"],
    )


def _price_domain(frame: pd.DataFrame, holding: _PreparedHolding) -> tuple[float, float]:
    values = [
        *frame["low"].astype(float).tolist(),
        *frame["high"].astype(float).tolist(),
        holding.effective_stop,
    ]
    entry = holding.entry_overlay
    if entry is not None:
        values.extend(
            value
            for value in (
                entry.effective_trigger_price,
                entry.reference_price,
                entry.zone_low,
                entry.zone_high,
            )
            if value is not None
        )
    if holding.pivot is not None:
        values.append(holding.pivot.price)
    low = min(float(value) for value in values)
    high = max(float(value) for value in values)
    span = max(high - low, max(high, 1.0) * 0.02)
    return max(0.0, low - 0.06 * span), high + 0.08 * span


def _draw_entry_overlay(
    draw: ImageDraw.ImageDraw,
    entry: HoldingChartEntryOverlay,
    frame: pd.DataFrame,
    left: int,
    right: int,
    y_price,
) -> tuple[_LabelSpec, ...]:
    historical = entry.nature is EntryOverlayNature.HISTORICAL_OBSERVATION
    prefix = "原观察" if historical else "当前条件"
    source_cutoff = entry.source_cutoff or _frame_date(frame.iloc[-1]["trade_date"])
    start_x = _date_start_x(frame, source_cutoff, left, right)
    specs: list[_LabelSpec] = []
    if entry.zone_low is not None and entry.zone_high is not None:
        y0 = y_price(float(entry.zone_high))
        y1 = y_price(float(entry.zone_low))
        draw.rectangle((start_x, y0, right, y1), fill=HOLDING_CHART_COLORS["entry_zone"])
        specs.append(
            _LabelSpec(
                target_y=(y0 + y1) // 2,
                label=f"{prefix}区 {float(entry.zone_low):.2f}–{float(entry.zone_high):.2f}",
                color=HOLDING_CHART_COLORS["entry"],
            )
        )
    if entry.reference_price is not None:
        reference_y = y_price(float(entry.reference_price))
        _draw_horizontal_line(
            draw,
            start_x,
            right,
            reference_y,
            HOLDING_CHART_COLORS["entry"],
            dashed=True,
        )
        specs.append(
            _LabelSpec(
                target_y=reference_y,
                label=f"{prefix}参考 {float(entry.reference_price):.2f}",
                color=HOLDING_CHART_COLORS["entry"],
            )
        )
    trigger = entry.effective_trigger_price
    if trigger is not None:
        trigger_y = y_price(trigger)
        _draw_horizontal_line(
            draw,
            start_x,
            right,
            trigger_y,
            HOLDING_CHART_COLORS["entry"],
            dashed=False,
        )
        specs.append(
            _LabelSpec(
                target_y=trigger_y,
                label=f"{prefix}触发 {trigger:.2f}",
                color=HOLDING_CHART_COLORS["entry"],
            )
        )
    return _merge_entry_label_specs(tuple(specs), prefix=prefix)


def _draw_horizontal_line(
    draw: ImageDraw.ImageDraw,
    left: int,
    right: int,
    y: int,
    color: tuple[int, int, int],
    *,
    dashed: bool,
) -> None:
    if dashed:
        for start in range(left, right, 14):
            draw.line((start, y, min(start + 8, right), y), fill=color, width=2)
    else:
        draw.line((left, y, right, y), fill=color, width=2)


def _merge_entry_label_specs(
    specs: tuple[_LabelSpec, ...],
    *,
    prefix: str,
    proximity_pixels: int = 26,
) -> tuple[_LabelSpec, ...]:
    """Combine adjacent entry labels before the shared collision pass."""

    if not specs:
        return ()
    ordered = sorted(specs, key=lambda item: (item.target_y, item.label))
    clusters: list[list[_LabelSpec]] = []
    for spec in ordered:
        if not clusters or spec.target_y - clusters[-1][-1].target_y > proximity_pixels:
            clusters.append([spec])
        else:
            clusters[-1].append(spec)
    merged: list[_LabelSpec] = []
    for cluster in clusters:
        if len(cluster) == 1:
            merged.append(cluster[0])
            continue
        details = [item.label.removeprefix(prefix) for item in cluster]
        merged.append(
            _LabelSpec(
                target_y=round(sum(item.target_y for item in cluster) / len(cluster)),
                label=f"{prefix}：{'｜'.join(details)}",
                color=HOLDING_CHART_COLORS["entry"],
            )
        )
    return tuple(merged)


def _resolve_label_positions(
    specs: tuple[_LabelSpec, ...],
    *,
    top: int,
    bottom: int,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    gap: int = 5,
) -> tuple[tuple[_LabelSpec, int], ...]:
    """Place labels inside a price panel without overlap or nondeterminism."""

    if not specs:
        return ()
    if top >= bottom or gap < 0:
        raise ValueError("invalid label layout bounds")
    label_height = _label_height(draw, font)
    available = bottom - top
    required = len(specs) * label_height + (len(specs) - 1) * gap
    if required > available:
        raise ValueError("price panel is too short for non-overlapping labels")
    half = (label_height + 1) // 2
    lower = top + half
    upper = bottom - half
    ordered = sorted(specs, key=lambda item: (item.target_y, item.label))
    centers: list[int] = []
    for spec in ordered:
        center = min(max(spec.target_y, lower), upper)
        if centers:
            center = max(center, centers[-1] + label_height + gap)
        centers.append(center)
    if centers[-1] > upper:
        centers[-1] = upper
        for index in range(len(centers) - 2, -1, -1):
            centers[index] = min(
                centers[index],
                centers[index + 1] - label_height - gap,
            )
    if centers[0] < lower:
        centers[0] = lower
        for index in range(1, len(centers)):
            centers[index] = max(
                centers[index],
                centers[index - 1] + label_height + gap,
            )
    if centers[-1] > upper:
        raise ValueError("label collision resolution exceeded panel bounds")
    return tuple(zip(ordered, centers, strict=True))


def _label_height(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> int:
    bbox = draw.textbbox((0, 0), "价位Ag09", font=font)
    return max(18, bbox[3] - bbox[1] + 10)


def _draw_action_badge(
    draw: ImageDraw.ImageDraw,
    action: HoldingAction,
    right: int,
    top: int,
    fonts: _FontBook,
) -> int:
    label, color_key = {
        HoldingAction.HOLD: ("持有", "action_hold"),
        HoldingAction.TIGHTEN: ("收紧", "action_tighten"),
        HoldingAction.REDUCE: ("减仓", "action_reduce"),
        HoldingAction.EXIT: ("退出", "action_exit"),
        HoldingAction.REVIEW: ("复核", "action_review"),
    }[action]
    bbox = draw.textbbox((0, 0), label, font=fonts.small)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    left = right - width - 18
    draw.rounded_rectangle(
        (left, top, right, top + height + 10),
        radius=9,
        fill=HOLDING_CHART_COLORS[color_key],
    )
    draw.text(
        (left + 9, top + 4),
        label,
        font=fonts.small,
        fill=HOLDING_CHART_COLORS["white"],
    )
    return left


def _date_start_x(frame: pd.DataFrame, start: date, left: int, right: int) -> int:
    dates = tuple(_frame_date(value) for value in frame["trade_date"])
    count = len(dates)
    step = (right - left) / max(count, 1)
    if start <= dates[0]:
        return left
    for index, value in enumerate(dates):
        if value >= start:
            return round(left + (index + 0.5) * step)
    # A current daily condition can post-date the last completed weekly bar.
    # Keep it visible only as a small right-edge marker; never backfill it.
    return max(left, right - 2)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    left: int,
    right: int,
    y: int,
    label: str,
    color: tuple[int, int, int],
    fonts: _FontBook,
) -> None:
    label = _fit_text(draw, label, fonts.tiny, max_width=right - left - 18)
    bbox = draw.textbbox((0, 0), label, font=fonts.tiny)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = max(left + 6, right - width - 8)
    draw.rounded_rectangle(
        (x - 3, y - height // 2 - 3, right, y + height // 2 + 3),
        radius=4,
        fill=HOLDING_CHART_COLORS["white"],
        outline=color,
        width=1,
    )
    draw.text((x, y - height // 2 - 1), label, font=fonts.tiny, fill=color)


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    *,
    max_width: int,
) -> str:
    if max_width <= 0:
        raise ValueError("label width must be positive")
    if draw.textlength(text, font=font) <= max_width:
        return text
    suffix = "…"
    candidate = text
    while candidate and draw.textlength(candidate + suffix, font=font) > max_width:
        candidate = candidate[:-1]
    if not candidate:
        raise ValueError("label cannot fit inside panel")
    return candidate + suffix


def _draw_candles(
    draw: ImageDraw.ImageDraw,
    frame: pd.DataFrame,
    left: int,
    right: int,
    y_price,
) -> None:
    count = len(frame)
    step = (right - left) / max(count, 1)
    body_half = max(1, min(6, int(step * 0.30)))
    for index, row in frame.iterrows():
        x = round(left + (index + 0.5) * step)
        open_price = float(row["open"])
        close = float(row["close"])
        color = HOLDING_CHART_COLORS["up"] if close >= open_price else HOLDING_CHART_COLORS["down"]
        high_y = y_price(float(row["high"]))
        low_y = y_price(float(row["low"]))
        open_y = y_price(open_price)
        close_y = y_price(close)
        draw.line((x, high_y, x, low_y), fill=color, width=1)
        top = min(open_y, close_y)
        bottom = max(open_y, close_y)
        if bottom == top:
            draw.line((x - body_half, top, x + body_half, top), fill=color, width=2)
        else:
            draw.rectangle((x - body_half, top, x + body_half, bottom), fill=color)


def _draw_series(
    draw: ImageDraw.ImageDraw,
    frame: pd.DataFrame,
    column: str,
    left: int,
    right: int,
    y_price,
    color_key: str,
) -> None:
    count = len(frame)
    step = (right - left) / max(count, 1)
    points: list[tuple[int, int]] = []
    for index, value in enumerate(frame[column]):
        if pd.isna(value) or not isfinite(float(value)):
            if len(points) > 1:
                draw.line(points, fill=HOLDING_CHART_COLORS[color_key], width=2)
            points = []
            continue
        points.append((round(left + (index + 0.5) * step), y_price(float(value))))
    if len(points) > 1:
        draw.line(points, fill=HOLDING_CHART_COLORS[color_key], width=2)


def _draw_volume(
    draw: ImageDraw.ImageDraw,
    frame: pd.DataFrame,
    source: Literal["volume_shares", "amount_cny"] | None,
    left: int,
    right: int,
    top: int,
    bottom: int,
    fonts: _FontBook,
) -> None:
    draw.line((left, top, right, top), fill=HOLDING_CHART_COLORS["grid"], width=1)
    if source is None:
        draw.text(
            (left + 4, top + 8),
            "成交活跃度不可用",
            font=fonts.tiny,
            fill=HOLDING_CHART_COLORS["muted_text"],
        )
        return
    values = frame[source].astype(float)
    maximum = float(values.max())
    if maximum <= 0.0:
        return
    count = len(frame)
    step = (right - left) / max(count, 1)
    half = max(1, min(6, int(step * 0.30)))
    for index, row in frame.iterrows():
        x = round(left + (index + 0.5) * step)
        height = round((bottom - top) * float(row[source]) / maximum)
        color = (
            HOLDING_CHART_COLORS["volume_up"]
            if float(row["close"]) >= float(row["open"])
            else HOLDING_CHART_COLORS["volume_down"]
        )
        draw.rectangle((x - half, bottom - height, x + half, bottom), fill=color)
    label = "量" if source == "volume_shares" else "额"
    draw.text((left + 4, top + 2), label, font=fonts.tiny, fill=HOLDING_CHART_COLORS["muted_text"])


def _activity_source(
    frame: pd.DataFrame,
) -> Literal["volume_shares", "amount_cny"] | None:
    for column in ("volume_shares", "amount_cny"):
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if not bool(numeric.isna().any()) and all(
            isfinite(float(value)) and float(value) >= 0.0 for value in numeric
        ):
            return column
    return None


def _panel_metadata(
    holding: _PreparedHolding,
    frame: pd.DataFrame,
    *,
    timeframe: Literal["daily", "weekly_completed"],
) -> HoldingChartPanelMetadata:
    entry = holding.entry_overlay
    raw_activity_source = _activity_source(frame)
    safe_activity_source: Literal["volume", "traded_value"] | None = {
        "volume_shares": "volume",
        "amount_cny": "traded_value",
        None: None,
    }[raw_activity_source]
    return HoldingChartPanelMetadata(
        symbol=holding.symbol,
        timeframe=timeframe,
        bar_count=len(frame),
        series_start=_frame_date(frame.iloc[0]["trade_date"]),
        series_end=_frame_date(frame.iloc[-1]["trade_date"]),
        activity_source=safe_activity_source,
        incomplete_current_week_excluded=(
            holding.incomplete_week_excluded if timeframe == "weekly_completed" else False
        ),
        action=holding.action.value,
        entry_overlay_drawn=entry is not None,
        entry_nature=(None if entry is None else entry.nature.value),
        entry_overlay_start=(None if entry is None else entry.source_cutoff),
        historical_warning_count=int(
            entry is not None and entry.nature is EntryOverlayNature.HISTORICAL_OBSERVATION
        ),
        pivot_drawn=holding.pivot is not None,
        pivot_start=(None if holding.pivot is None else holding.pivot.confirmed_on),
        protective_stop_drawn=True,
        protective_line_nature=holding.protective_line_nature.value,
        protective_line_start=holding.protective_line_start,
    )


def _map_y(
    value: float,
    minimum: float,
    maximum: float,
    top: int,
    bottom: int,
) -> int:
    ratio = (float(value) - minimum) / (maximum - minimum)
    return round(bottom - ratio * (bottom - top))


def _frame_date(value: object) -> date:
    return pd.Timestamp(value).date()


def _positive_finite(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(number) and number > 0.0


def _same_price(left: object, right: object) -> bool:
    if not _positive_finite(left) or not _positive_finite(right):
        return False
    lhs = float(left)
    rhs = float(right)
    return abs(lhs - rhs) <= max(0.005, abs(lhs) * 1e-9)


def _safe_name(value: object) -> str:
    name = " ".join(str(value).split())
    if not name or len(name) > 32 or any(ord(character) < 32 for character in name):
        raise ValueError("holding name is invalid")
    return name


def _encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    # No PngInfo is attached: the PNG carries pixels only, never source paths,
    # raw rows, review objects, account values, or credentials.
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


__all__ = [
    "COMPOSITE_HEIGHT",
    "COMPOSITE_WIDTH",
    "DAILY_BAR_LIMIT",
    "EntryOverlayNature",
    "MIN_HOLDING_COUNT",
    "MAX_HOLDING_COUNT",
    "holding_chart_composite_height",
    "HOLDING_CHART_COLORS",
    "HoldingChartConfirmedPivot",
    "HoldingChartEntryOverlay",
    "HoldingChartPanelMetadata",
    "HoldingChartReportMetadata",
    "HoldingChartReportRequest",
    "HoldingChartReviewIdentity",
    "ProtectiveLineNature",
    "RenderedHoldingChartReport",
    "WEEKLY_BAR_LIMIT",
    "render_holding_chart_report",
]
