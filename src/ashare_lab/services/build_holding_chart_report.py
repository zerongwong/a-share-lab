"""Assemble a private, local chart report for the current holding snapshot.

This module is deliberately side-effect bounded.  It reads only the explicit
holding ledger, verified local price history, persisted protective-stop state,
and an explicitly linked recommendation archive.  The holding review always
runs with ``persist=False``; this builder neither sends a notification nor
changes a holding, stop, recommendation, or order state.

Only structured archived entry fields may become chart overlays.  A
reconstructed or observation-only archive is always described as a historical
observation, never as contemporaneous action permission.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Any, Final, Protocol

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.domain.data_sources import DEFAULT_MARKET_OVERLAY_SOURCE_ID
from ashare_lab.domain.errors import DataQualityError
from ashare_lab.ports.market_data import normalize_symbol
from ashare_lab.services.holding_ledger import (
    ActiveHoldingPortfolio,
    HoldingKnowledgeContext,
    get_active_holding_portfolio,
    resolve_current_holding_context,
)
from ashare_lab.services.render_holding_chart_report import (
    MAX_HOLDING_COUNT,
    MIN_HOLDING_COUNT,
    EntryOverlayNature,
    HoldingChartEntryOverlay,
    HoldingChartReportRequest,
    HoldingChartReviewIdentity,
    RenderedHoldingChartReport,
    render_holding_chart_report,
)
from ashare_lab.services.review_active_holdings import (
    HoldingReviewRowStatus,
    HoldingReviewSummaryStatus,
    HoldingTreeReviewSummary,
    review_active_holdings,
)
from ashare_lab.services.run_active_holding_review import (
    ActiveHoldingHistoryLoad,
    _local_company_action_clearances,
    load_active_holding_histories,
)

HOLDING_CHART_BUILD_METHOD_VERSION: Final = "holding-chart-builder-v0.1.0"
_ARCHIVE_STEM: Final = "holding-chart"
_ARCHIVE_NAME = re.compile(
    rf"^{_ARCHIVE_STEM}-(?P<cutoff>\d{{8}})-v(?P<version>\d+)-"
    r"(?P<panel>composite|[0-9]{6})\.png$"
)
_ORIGIN_REPORT_KEYS: Final = (
    "origin_recommendation_report_id",
    "source_recommendation_report_id",
    "recommendation_report_id",
    "origin_report_id",
)
_ORIGIN_BATCH_KEYS: Final = (
    "origin_recommendation_batch_id",
    "source_recommendation_batch_id",
    "recommendation_batch_id",
    "origin_batch_id",
)
_ORIGIN_CONTAINER_KEYS: Final = (
    "recommendation_origin",
    "origin_recommendation",
)


class HoldingChartBuildStatus(StrEnum):
    READY = "ready"
    NO_HOLDINGS = "no_holdings"
    DATA_NOT_READY = "data_not_ready"


class OriginOverlayNature(StrEnum):
    ARCHIVED_ENTRY_PLAN = "archived_entry_plan"
    HISTORICAL_OBSERVATION = "historical_observation"


@dataclass(frozen=True, slots=True)
class HoldingChartArchiveReceipt:
    directory: Path
    composite_path: Path
    individual_paths: tuple[Path, ...]
    pruned_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class HoldingChartBuildResult:
    """Sanitized build result; it intentionally contains no market frames.

    The rendered object contains PNG bytes plus non-sensitive renderer
    metadata.  Cost, shares, amount, and account or sleeve weights are not
    copied out of the holding-review object.
    """

    status: HoldingChartBuildStatus
    portfolio_id: str | None
    holding_version: int | None
    holding_weeks: int | None
    data_cutoff: date | None
    rendered: RenderedHoldingChartReport | None
    archive: HoldingChartArchiveReceipt | None
    origin_report_id: str | None = None
    origin_batch_id: str | None = None
    origin_nature: str | None = None
    entry_overlay_count: int = 0
    reasons: tuple[str, ...] = ()
    method_version: str = HOLDING_CHART_BUILD_METHOD_VERSION


@dataclass(frozen=True, slots=True)
class _OriginArchive:
    report_id: str | None = None
    batch_id: str | None = None
    archive_nature: str | None = None
    evaluation_mode: str | None = None
    overlay_nature: OriginOverlayNature | None = None
    members: tuple[Mapping[str, Any], ...] = ()
    reasons: tuple[str, ...] = ()


class _RecommendationArchiveReader(Protocol):
    def get_recommendation_report(self, report_id: str) -> dict[str, Any] | None: ...

    def get_recommendation_batch(self, batch_id: str) -> dict[str, Any] | None: ...

    def list_recommendation_batches(self, report_id: str) -> list[dict[str, Any]]: ...

    def list_recommendation_members(self, batch_id: str) -> list[dict[str, Any]]: ...


def build_holding_chart_report(
    repository: SQLiteRepository,
    *,
    dataset_root: str | Path,
    overlay_root: str | Path,
    as_of: date,
    overlay_source_id: str = DEFAULT_MARKET_OVERLAY_SOURCE_ID,
    reviewed_at: datetime | None = None,
    archive_directory: str | Path | None = None,
    archive_retention_days: int = 30,
    holding_context: HoldingKnowledgeContext | None = None,
    _history_loader: Callable[..., ActiveHoldingHistoryLoad | None] = (
        load_active_holding_histories
    ),
    _reviewer: Callable[..., HoldingTreeReviewSummary] = review_active_holdings,
    _renderer: Callable[[HoldingChartReportRequest], RenderedHoldingChartReport] = (
        render_holding_chart_report
    ),
) -> HoldingChartBuildResult:
    """Build one current-holdings chart from verified local completed bars."""

    if not isinstance(as_of, date):
        raise TypeError("as_of must be a date")
    resolved_reviewed_at = reviewed_at or datetime.now(UTC)
    context = holding_context
    if context is None:
        # Keep direct and historical calls point-in-time safe.  Only a live
        # caller (the evening digest) may explicitly supply a frozen current
        # revision whose knowledge time is later than the market-data cutoff.
        current_portfolio = get_active_holding_portfolio(repository, as_of=as_of)
    else:
        current_portfolio = resolve_current_holding_context(repository, context)
    loaded = _history_loader(
        repository,
        dataset_root=dataset_root,
        overlay_root=overlay_root,
        as_of=as_of,
        overlay_source_id=overlay_source_id,
        _holding_context=context,
    )
    if loaded is None:
        review = _reviewer(
            repository,
            {},
            as_of=as_of,
            verified_data_cutoff=as_of,
            reviewed_at=resolved_reviewed_at,
            persist=False,
            holding_context=context,
        )
        return HoldingChartBuildResult(
            status=HoldingChartBuildStatus.NO_HOLDINGS,
            portfolio_id=review.portfolio_id,
            holding_version=review.holding_version,
            holding_weeks=review.holding_weeks,
            data_cutoff=review.data_cutoff,
            rendered=None,
            archive=None,
            reasons=review.reasons or ("no_user_confirmed_active_holdings",),
        )

    portfolio = (
        current_portfolio
        if current_portfolio is not None
        else get_active_holding_portfolio(repository, as_of=loaded.data_cutoff)
    )
    _require_loaded_portfolio_identity(loaded, portfolio)
    assert portfolio is not None  # narrowed by the identity check
    review = _reviewer(
        repository,
        loaded.histories,
        as_of=loaded.data_cutoff,
        verified_data_cutoff=loaded.data_cutoff,
        verified_close=loaded.verified_close,
        reviewed_at=resolved_reviewed_at,
        persist=False,
        company_action_clear_by_symbol=_local_company_action_clearances(
            repository,
            cutoff=loaded.data_cutoff,
            holding_context=context,
        ),
        holding_context=context,
        continuous_profile=True,
    )
    identity = HoldingChartReviewIdentity(
        portfolio_id=loaded.portfolio_id,
        holding_version=loaded.holding_version,
        holding_weeks=loaded.holding_weeks,
        data_cutoff=loaded.data_cutoff,
    )
    _require_review_identity(review, identity)
    if tuple(row.symbol for row in review.rows) != tuple(
        position.symbol for position in portfolio.positions
    ):
        raise DataQualityError("Holding review membership does not match registered holdings")

    origin = _read_origin_archive(repository, portfolio)
    entry_overlays = _entry_overlays(origin, portfolio)
    if not _review_is_chart_eligible(review):
        return HoldingChartBuildResult(
            status=HoldingChartBuildStatus.DATA_NOT_READY,
            portfolio_id=loaded.portfolio_id,
            holding_version=loaded.holding_version,
            holding_weeks=loaded.holding_weeks,
            data_cutoff=loaded.data_cutoff,
            rendered=None,
            archive=None,
            origin_report_id=origin.report_id,
            origin_batch_id=origin.batch_id,
            origin_nature=(None if origin.overlay_nature is None else origin.overlay_nature.value),
            entry_overlay_count=len(entry_overlays),
            reasons=(
                *loaded.unavailable_symbols,
                *review.reasons,
                *origin.reasons,
                "holding_chart_requires_one_to_five_ready_holdings",
            ),
        )
    request = HoldingChartReportRequest(
        review=review,
        histories=loaded.histories,
        expected_identity=identity,
        entry_overlays=entry_overlays,
        confirmed_pivots=(),
    )
    rendered = _renderer(request)
    archive = None
    if archive_directory is not None:
        archive = _archive_holding_chart_report(
            rendered.composite_png,
            directory=archive_directory,
            data_cutoff=loaded.data_cutoff,
            portfolio_version=loaded.holding_version,
            retention_days=archive_retention_days,
            individual_pngs=rendered.individual_pngs,
        )
    return HoldingChartBuildResult(
        status=HoldingChartBuildStatus.READY,
        portfolio_id=loaded.portfolio_id,
        holding_version=loaded.holding_version,
        holding_weeks=loaded.holding_weeks,
        data_cutoff=loaded.data_cutoff,
        rendered=rendered,
        archive=archive,
        origin_report_id=origin.report_id,
        origin_batch_id=origin.batch_id,
        origin_nature=(None if origin.overlay_nature is None else origin.overlay_nature.value),
        entry_overlay_count=len(entry_overlays),
        reasons=(*loaded.unavailable_symbols, *origin.reasons),
    )


def _read_origin_archive(
    repository: _RecommendationArchiveReader,
    portfolio: ActiveHoldingPortfolio,
) -> _OriginArchive:
    report_id, report_reason = _one_origin_id(portfolio.metadata, _ORIGIN_REPORT_KEYS)
    batch_id, batch_reason = _one_origin_id(portfolio.metadata, _ORIGIN_BATCH_KEYS)
    nested_report, nested_batch, nested_reason = _nested_origin_ids(portfolio.metadata)
    reasons = tuple(
        reason for reason in (report_reason, batch_reason, nested_reason) if reason is not None
    )
    if reasons:
        return _OriginArchive(reasons=reasons)
    report_id = report_id or nested_report
    batch_id = batch_id or nested_batch
    if report_id is not None and nested_report is not None and report_id != nested_report:
        return _OriginArchive(reasons=("conflicting_origin_report_ids",))
    if batch_id is not None and nested_batch is not None and batch_id != nested_batch:
        return _OriginArchive(reasons=("conflicting_origin_batch_ids",))
    if report_id is None and batch_id is None:
        return _OriginArchive(reasons=("origin_recommendation_not_linked",))

    batch: Mapping[str, Any] | None = None
    if batch_id is not None:
        batch = repository.get_recommendation_batch(batch_id)
        if batch is None:
            return _OriginArchive(
                report_id=report_id,
                batch_id=batch_id,
                reasons=("origin_recommendation_batch_missing",),
            )
        batch_report_id = _optional_text(batch.get("report_id"))
        if batch_report_id is None:
            return _OriginArchive(
                report_id=report_id,
                batch_id=batch_id,
                reasons=("origin_recommendation_batch_has_no_report",),
            )
        if report_id is not None and batch_report_id != report_id:
            return _OriginArchive(
                report_id=report_id,
                batch_id=batch_id,
                reasons=("origin_recommendation_report_batch_mismatch",),
            )
        report_id = batch_report_id

    assert report_id is not None
    report = repository.get_recommendation_report(report_id)
    if report is None:
        return _OriginArchive(
            report_id=report_id,
            batch_id=batch_id,
            reasons=("origin_recommendation_report_missing",),
        )
    if batch is None:
        matching = [
            item
            for item in repository.list_recommendation_batches(report_id)
            if _optional_int(item.get("holding_weeks")) == portfolio.holding_weeks
        ]
        if len(matching) != 1:
            return _OriginArchive(
                report_id=report_id,
                reasons=(
                    "origin_recommendation_horizon_batch_missing"
                    if not matching
                    else "origin_recommendation_horizon_batch_ambiguous",
                ),
            )
        batch = matching[0]
        batch_id = _optional_text(batch.get("id"))
    if batch_id is None:
        return _OriginArchive(
            report_id=report_id,
            reasons=("origin_recommendation_batch_has_no_id",),
        )
    if _optional_int(batch.get("holding_weeks")) != portfolio.holding_weeks:
        return _OriginArchive(
            report_id=report_id,
            batch_id=batch_id,
            reasons=("origin_recommendation_horizon_mismatch",),
        )

    archive_nature = _optional_text(batch.get("archive_nature") or report.get("archive_nature"))
    evaluation_mode = _optional_text(batch.get("evaluation_mode"))
    if archive_nature not in {"original", "reconstructed"}:
        return _OriginArchive(
            report_id=report_id,
            batch_id=batch_id,
            archive_nature=archive_nature,
            evaluation_mode=evaluation_mode,
            reasons=("origin_recommendation_archive_nature_unknown",),
        )
    overlay_nature = (
        OriginOverlayNature.ARCHIVED_ENTRY_PLAN
        if archive_nature == "original" and evaluation_mode == "action_simulation"
        else OriginOverlayNature.HISTORICAL_OBSERVATION
    )
    members = tuple(repository.list_recommendation_members(batch_id))
    return _OriginArchive(
        report_id=report_id,
        batch_id=batch_id,
        archive_nature=archive_nature,
        evaluation_mode=evaluation_mode,
        overlay_nature=overlay_nature,
        members=members,
        reasons=(() if members else ("origin_recommendation_has_no_members",)),
    )


def _entry_overlays(
    origin: _OriginArchive,
    portfolio: ActiveHoldingPortfolio,
) -> tuple[HoldingChartEntryOverlay, ...]:
    if origin.overlay_nature is None or not origin.members:
        return ()
    by_symbol: dict[str, Mapping[str, Any]] = {}
    ambiguous_symbols: set[str] = set()
    for member in origin.members:
        try:
            symbol = normalize_symbol(str(member.get("symbol", "")))
        except ValueError:
            continue
        if symbol in ambiguous_symbols:
            continue
        if symbol in by_symbol:
            # Ambiguous archived members fail closed instead of selecting one.
            by_symbol.pop(symbol, None)
            ambiguous_symbols.add(symbol)
            continue
        by_symbol[symbol] = member

    overlays: list[HoldingChartEntryOverlay] = []
    for holding in portfolio.positions:
        member = by_symbol.get(holding.symbol)
        plan_cutoff = None if member is None else _plan_cutoff(member)
        if member is None or plan_cutoff is None or plan_cutoff > holding.entry_date:
            continue
        low = _positive_price(member.get("price_low"))
        high = _positive_price(member.get("price_high"))
        if (low is None) != (high is None) or (low is not None and high is not None and low > high):
            low = None
            high = None
        trigger = _positive_price(member.get("trigger_price"))
        reference = _positive_price(
            member.get("reference_price", member.get("entry_reference_price"))
        )
        if low is None and trigger is None and reference is None:
            continue
        overlays.append(
            HoldingChartEntryOverlay(
                symbol=holding.symbol,
                line_price=trigger,
                trigger_price=trigger,
                reference_price=reference,
                zone_low=low,
                zone_high=high,
                # Every origin-plan overlay is historical relative to an
                # already-confirmed holding.  Even an original action archive
                # must not become renewed buy permission on a later chart.
                nature=EntryOverlayNature.HISTORICAL_OBSERVATION,
                source_cutoff=plan_cutoff,
            )
        )
    return tuple(overlays)


def _archive_holding_chart_report(
    composite_png: bytes,
    *,
    directory: str | Path,
    data_cutoff: date,
    portfolio_version: int,
    retention_days: int = 30,
    individual_pngs: Mapping[str, bytes] | None = None,
) -> HoldingChartArchiveReceipt:
    """Atomically store only this service's PNG artifacts with private modes."""

    if not isinstance(data_cutoff, date):
        raise TypeError("data_cutoff must be a date")
    if portfolio_version <= 0:
        raise ValueError("portfolio_version must be positive")
    if retention_days < 0:
        raise ValueError("retention_days cannot be negative")
    root = Path(directory).expanduser()
    if root.is_symlink():
        raise ValueError("Holding chart archive directory cannot be a symlink")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError("Holding chart archive path is not a directory")
    root.chmod(0o700)

    cutoff_text = data_cutoff.strftime("%Y%m%d")
    version_text = f"v{portfolio_version:06d}"
    composite_path = root / f"{_ARCHIVE_STEM}-{cutoff_text}-{version_text}-composite.png"
    _atomic_private_write(composite_path, _as_bytes(composite_png, "composite_png"))

    individual_paths: list[Path] = []
    for raw_symbol, payload in sorted((individual_pngs or {}).items()):
        symbol = normalize_symbol(raw_symbol)
        target = root / f"{_ARCHIVE_STEM}-{cutoff_text}-{version_text}-{symbol}.png"
        _atomic_private_write(target, _as_bytes(payload, f"individual_pngs[{symbol!r}]"))
        individual_paths.append(target)
    pruned = _prune_owned_archives(
        root,
        expire_before=data_cutoff - timedelta(days=retention_days),
        protected={composite_path, *individual_paths},
    )
    return HoldingChartArchiveReceipt(
        directory=root,
        composite_path=composite_path,
        individual_paths=tuple(individual_paths),
        pruned_paths=pruned,
    )


def _atomic_private_write(target: Path, payload: bytes) -> None:
    if target.exists() and not target.is_symlink():
        try:
            if target.read_bytes() == payload:
                target.chmod(0o600)
                return
        except OSError:
            pass
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        target.chmod(0o600)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _prune_owned_archives(
    directory: Path,
    *,
    expire_before: date,
    protected: set[Path],
) -> tuple[Path, ...]:
    pruned: list[Path] = []
    for item in directory.iterdir():
        if item in protected or item.is_symlink() or not item.is_file():
            continue
        match = _ARCHIVE_NAME.fullmatch(item.name)
        if match is None:
            continue
        try:
            artifact_cutoff = datetime.strptime(match.group("cutoff"), "%Y%m%d").date()
        except ValueError:
            continue
        if artifact_cutoff < expire_before:
            item.unlink()
            pruned.append(item)
    return tuple(sorted(pruned))


def _require_loaded_portfolio_identity(
    loaded: ActiveHoldingHistoryLoad,
    portfolio: ActiveHoldingPortfolio | None,
) -> None:
    if portfolio is None or portfolio.status != "active":
        raise DataQualityError("Loaded holding histories have no matching active portfolio")
    if (
        portfolio.id != loaded.portfolio_id
        or portfolio.version != loaded.holding_version
        or portfolio.holding_weeks != loaded.holding_weeks
    ):
        raise DataQualityError("Holding portfolio changed while chart evidence was loading")


def _require_review_identity(
    review: HoldingTreeReviewSummary,
    expected: HoldingChartReviewIdentity,
) -> None:
    actual = (
        review.portfolio_id,
        review.holding_version,
        review.holding_weeks,
        review.data_cutoff,
    )
    wanted = (
        expected.portfolio_id,
        expected.holding_version,
        expected.holding_weeks,
        expected.data_cutoff,
    )
    if actual != wanted:
        raise DataQualityError("Holding review identity does not match loaded chart evidence")


def _review_is_chart_eligible(review: HoldingTreeReviewSummary) -> bool:
    """Mirror the renderer's narrow allowance for evidence-blocked references."""

    if not MIN_HOLDING_COUNT <= len(review.rows) <= MAX_HOLDING_COUNT:
        return False
    if review.status is HoldingReviewSummaryStatus.READY:
        return True
    if review.status is not HoldingReviewSummaryStatus.PARTIAL:
        return False
    return all(
        row.status is HoldingReviewRowStatus.READY
        or any(str(reason).startswith("company_action_evidence_blocks_") for reason in row.reasons)
        for row in review.rows
    )


def _one_origin_id(
    metadata: Mapping[str, Any],
    keys: tuple[str, ...],
) -> tuple[str | None, str | None]:
    values = {value for key in keys if (value := _optional_text(metadata.get(key))) is not None}
    if len(values) > 1:
        return None, "conflicting_origin_recommendation_ids"
    return (next(iter(values)) if values else None), None


def _nested_origin_ids(
    metadata: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None]:
    containers = [metadata.get(key) for key in _ORIGIN_CONTAINER_KEYS]
    mappings = [item for item in containers if isinstance(item, Mapping)]
    reports = {
        value for item in mappings if (value := _optional_text(item.get("report_id"))) is not None
    }
    batches = {
        value for item in mappings if (value := _optional_text(item.get("batch_id"))) is not None
    }
    if len(reports) > 1 or len(batches) > 1:
        return None, None, "conflicting_nested_origin_recommendation_ids"
    return (
        next(iter(reports)) if reports else None,
        next(iter(batches)) if batches else None,
        None,
    )


def _plan_cutoff(member: Mapping[str, Any]) -> date | None:
    raw = member.get("plan_cutoff")
    if raw is None:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def _positive_price(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) and number > 0 else None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _as_bytes(value: bytes, field: str) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"{field} must contain non-empty PNG bytes")
    return value
