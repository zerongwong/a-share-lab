"""Create and persist an immutable, derived recommendation-report archive.

Only compact decision evidence enters this bundle.  Raw OHLCV frames,
provider credentials, notification keys and rendered notification bodies are
intentionally outside the API.  The content hash is therefore reproducible
from one :class:`EveningResearchDigest` and suitable for idempotent storage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal, Protocol
from uuid import NAMESPACE_URL, uuid5

from ashare_lab.services.build_evening_digest import (
    EveningDigestCandidate,
    EveningPeriodDigest,
    EveningResearchDigest,
)

ArchiveNature = Literal["original", "reconstructed"]
ARCHIVE_METHOD_VERSION = "recommendation-performance-archive-v2.0.0"

_HORIZON_KEYS = {1: "1w", 2: "2w", 4: "1m", 13: "3m", 26: "6m", 52: "1y"}


class RecommendationArchiveRepository(Protocol):
    """Narrow persistence boundary used by the archive service."""

    def archive_recommendation_report(
        self,
        report: Mapping[str, Any],
        *,
        batches: Iterable[Mapping[str, Any]] = (),
        members: Iterable[Mapping[str, Any]] = (),
        delivery_events: Iterable[Mapping[str, Any]] = (),
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class RecommendationArchiveBundle:
    """One complete immutable report ready for an atomic repository write."""

    report: Mapping[str, Any]
    batches: tuple[Mapping[str, Any], ...]
    members: tuple[Mapping[str, Any], ...]
    content_hash: str

    @property
    def report_id(self) -> str:
        return str(self.report["id"])


def build_recommendation_archive_bundle(
    digest: EveningResearchDigest,
    *,
    archive_nature: ArchiveNature = "original",
    created_at: datetime | None = None,
) -> RecommendationArchiveBundle:
    """Build a deterministic archive without persisting or exposing raw data.

    ``original`` means the digest is archived from the provider-accepted run.
    ``reconstructed`` is reserved for a later, explicitly labelled backfill;
    it can never be classified as a primary action simulation.
    """

    if not isinstance(digest, EveningResearchDigest):
        raise TypeError("digest must be an EveningResearchDigest")
    if archive_nature not in {"original", "reconstructed"}:
        raise ValueError("archive_nature must be original or reconstructed")
    if digest.plan_for_date is None:
        raise ValueError("plan_for_date must be verified before recommendation archiving")
    created = _utc_datetime(created_at)

    canonical_periods = tuple(
        _canonical_period(period)
        for period in sorted(digest.periods, key=lambda item: item.holding_weeks)
    )
    canonical_report = {
        "archive_method_version": ARCHIVE_METHOD_VERSION,
        "archive_nature": archive_nature,
        "decision_date": digest.decision_date.isoformat(),
        "plan_for_date": (
            None if digest.plan_for_date is None else digest.plan_for_date.isoformat()
        ),
        "common_cutoff": digest.common_cutoff.isoformat(),
        "method_version": digest.method_version,
        "cycle_label": digest.cycle_label,
        "entry_strictness": digest.entry_strictness,
        "max_stock_exposure": digest.max_stock_exposure,
        "minimum_cash_weight": digest.minimum_cash_weight,
        "central_implementation_status": digest.central_implementation_status,
        "multi_timeframe_component_status": digest.multi_timeframe_component_status,
        "periods": canonical_periods,
    }
    if digest.continuous_plan is not None:
        if digest.periods:
            raise ValueError("continuous decisions must not create fixed-maturity batches")
        canonical_report["continuous_plan"] = digest.continuous_plan
    content_hash = _canonical_sha256(canonical_report)
    report_uuid = uuid5(NAMESPACE_URL, f"a-share-lab:recommendation-report:{content_hash}")
    report_id = str(report_uuid)

    batches: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    for period in sorted(digest.periods, key=lambda item: item.holding_weeks):
        batch_id = str(uuid5(report_uuid, f"batch:{period.holding_weeks}"))
        semantics = _batch_semantics(
            period,
            archive_nature=archive_nature,
            has_plan_date=digest.plan_for_date is not None,
        )
        primary_action_eligible_count = sum(
            _member_primary_action_eligible(candidate, semantics["evaluation_mode"])
            for candidate in period.candidates
        )
        observation_eligible_count = sum(
            _member_observation_eligible(candidate, semantics["evaluation_mode"])
            for candidate in period.candidates
        )
        performance_candidates = tuple(
            candidate
            for candidate in period.candidates
            if _member_primary_action_eligible(candidate, semantics["evaluation_mode"])
            or _member_observation_eligible(candidate, semantics["evaluation_mode"])
        )
        batch_status = (
            "pending"
            if semantics["evaluation_mode"] != "unavailable" and performance_candidates
            else "unavailable"
        )
        batches.append(
            {
                "id": batch_id,
                "report_id": report_id,
                "horizon_key": _HORIZON_KEYS[period.holding_weeks],
                "holding_weeks": period.holding_weeks,
                "holding_sessions": period.holding_sessions,
                "label": period.label,
                "data_cutoff": period.data_cutoff,
                "source_status": period.source_status,
                "evaluation_mode": semantics["evaluation_mode"],
                "actionability": semantics["actionability"],
                "cohort_nature": semantics["cohort_nature"],
                "allocation_nature": semantics["allocation_nature"],
                "action_stock_exposure": period.action_stock_exposure,
                "action_cash_weight": period.action_cash_weight,
                "stock_exposure": period.action_stock_exposure,
                "cash_weight": period.action_cash_weight,
                "member_count": len(performance_candidates),
                "status": batch_status,
                "metadata_json": {
                    "performance_nature": period.performance_nature,
                    "risk_nature": period.risk_nature,
                    "action_nature": period.action_nature,
                    "failure_code": period.failure_code,
                    "performance_eligible_member_count": (
                        primary_action_eligible_count + observation_eligible_count
                    ),
                    "primary_action_eligible_member_count": primary_action_eligible_count,
                    "observation_eligible_member_count": observation_eligible_count,
                    "unadjusted_price_return_only": True,
                },
            }
        )
        for candidate in performance_candidates:
            plan = _entry_plan(candidate)
            primary_action_eligible = _member_primary_action_eligible(
                candidate, semantics["evaluation_mode"]
            )
            observation_eligible = _member_observation_eligible(
                candidate, semantics["evaluation_mode"]
            )
            is_observation_mode = semantics["evaluation_mode"] in {
                "observation_simulation",
                "reconstructed_observation",
            }
            sleeve_weight = candidate.operational_stock_sleeve_weight
            account_weight = candidate.operational_account_weight
            if is_observation_mode:
                sleeve_weight = candidate.stock_sleeve_weight
                account_weight = None
            observation_anchor = "none"
            if is_observation_mode:
                observation_anchor = (
                    "archived_reference_price"
                    if candidate.price_plan_evaluation_price is not None
                    else "plan_session_close"
                )
            member_uuid = uuid5(
                uuid5(report_uuid, f"batch:{period.holding_weeks}"),
                f"member:{candidate.rank}:{candidate.symbol}",
            )
            members.append(
                {
                    "id": str(member_uuid),
                    "batch_id": batch_id,
                    "rank": candidate.rank,
                    "symbol": candidate.symbol,
                    "name": candidate.name,
                    "action": candidate.action,
                    "allocation_nature": candidate.allocation_nature,
                    "operational_stock_sleeve_weight": sleeve_weight,
                    "operational_account_weight": account_weight,
                    "price_nature": candidate.price_nature,
                    "plan_kind": candidate.price_plan_kind,
                    "price_low": candidate.price_plan_low,
                    "price_high": candidate.price_plan_high,
                    "trigger_price": candidate.price_plan_trigger,
                    "entry_plan_json": plan,
                    "entry_rule_json": plan,
                    "reference_price": candidate.price_plan_evaluation_price,
                    "evaluation_price": candidate.price_plan_evaluation_price,
                    "observation_anchor": observation_anchor,
                    "confirmation_rule": candidate.price_plan_confirmation_rule,
                    "invalidation_price": candidate.price_plan_invalidation_price,
                    "plan_cutoff": candidate.price_plan_cutoff,
                    "plan_sessions": candidate.price_plan_sessions,
                    "plan_method_version": candidate.price_plan_method_version,
                    "price_condition": candidate.price_condition,
                    "evidence_pending": candidate.evidence_pending,
                    "primary_timeframe": candidate.primary_timeframe,
                    "primary_structure": candidate.primary_structure,
                    "metadata_json": {
                        # Kept as the primary-action flag for backwards
                        # compatibility. Observation scorecards use their own
                        # explicit flag and can never enter the action series.
                        "performance_eligible": primary_action_eligible,
                        "primary_action_performance_eligible": primary_action_eligible,
                        "observation_performance_eligible": observation_eligible,
                        "display_stock_sleeve_weight": candidate.stock_sleeve_weight,
                        "display_account_weight": candidate.account_weight,
                        "evidence_pending": candidate.evidence_pending,
                        "primary_timeframe": candidate.primary_timeframe,
                        "primary_structure": candidate.primary_structure,
                        "multi_timeframe_method_version": (
                            candidate.multi_timeframe_method_version
                        ),
                    },
                }
            )

    report = {
        "id": report_id,
        "content_hash": content_hash,
        "archive_nature": archive_nature,
        "decision_date": digest.decision_date,
        "plan_for_date": digest.plan_for_date,
        "common_cutoff": digest.common_cutoff,
        "method_version": digest.method_version,
        "cycle_label": digest.cycle_label,
        "entry_strictness": digest.entry_strictness,
        "max_stock_exposure": digest.max_stock_exposure,
        "minimum_cash_weight": digest.minimum_cash_weight,
        "created_at": created,
        "metadata_json": {
            "archive_method_version": ARCHIVE_METHOD_VERSION,
            "central_implementation_status": digest.central_implementation_status,
            "multi_timeframe_component_status": digest.multi_timeframe_component_status,
            "raw_data_exposed": False,
            "brokerage_connected": False,
            "orders_enabled": False,
            "unadjusted_price_return_only": True,
        },
    }
    return RecommendationArchiveBundle(
        report=report,
        batches=tuple(batches),
        members=tuple(members),
        content_hash=content_hash,
    )


def archive_recommendation_report(
    digest: EveningResearchDigest,
    repository: RecommendationArchiveRepository,
    *,
    archive_nature: ArchiveNature = "original",
    created_at: datetime | None = None,
) -> RecommendationArchiveBundle:
    """Atomically persist one digest and return its deterministic receipt."""

    bundle = build_recommendation_archive_bundle(
        digest,
        archive_nature=archive_nature,
        created_at=created_at,
    )
    repository.archive_recommendation_report(
        bundle.report,
        batches=bundle.batches,
        members=bundle.members,
    )
    return bundle


def _canonical_period(period: EveningPeriodDigest) -> dict[str, Any]:
    return {
        "holding_weeks": period.holding_weeks,
        "holding_sessions": period.holding_sessions,
        "label": period.label,
        "data_cutoff": None if period.data_cutoff is None else period.data_cutoff.isoformat(),
        "source_status": period.source_status,
        "performance_nature": period.performance_nature,
        "action_nature": period.action_nature,
        "risk_nature": period.risk_nature,
        "action_stock_exposure": period.action_stock_exposure,
        "action_cash_weight": period.action_cash_weight,
        "failure_code": period.failure_code,
        "candidates": tuple(
            _canonical_candidate(candidate)
            for candidate in sorted(period.candidates, key=lambda item: (item.rank, item.symbol))
        ),
    }


def _canonical_candidate(candidate: EveningDigestCandidate) -> dict[str, Any]:
    return {
        "rank": candidate.rank,
        "symbol": candidate.symbol,
        "name": candidate.name,
        "action": candidate.action,
        "allocation_nature": candidate.allocation_nature,
        "display_stock_sleeve_weight": candidate.stock_sleeve_weight,
        "display_account_weight": candidate.account_weight,
        "operational_stock_sleeve_weight": candidate.operational_stock_sleeve_weight,
        "operational_account_weight": candidate.operational_account_weight,
        "price_nature": candidate.price_nature,
        "price_condition": candidate.price_condition,
        "entry_plan": _entry_plan(candidate),
        "evidence_pending": candidate.evidence_pending,
        "primary_timeframe": candidate.primary_timeframe,
        "primary_structure": candidate.primary_structure,
        "multi_timeframe_method_version": candidate.multi_timeframe_method_version,
    }


def _entry_plan(candidate: EveningDigestCandidate) -> dict[str, Any] | None:
    if candidate.price_plan_kind is None:
        return None
    plan = {
        "kind": candidate.price_plan_kind,
        "evaluation_rule": _plan_evaluation_rule(candidate.price_plan_kind),
        "price_low": candidate.price_plan_low,
        "price_high": candidate.price_plan_high,
        "trigger_price": candidate.price_plan_trigger,
        "evaluation_price": candidate.price_plan_evaluation_price,
        "confirmation_rule": candidate.price_plan_confirmation_rule,
        "confirmation_activity_metric": candidate.price_plan_confirmation_activity_metric,
        "confirmation_activity_min": candidate.price_plan_confirmation_activity_min,
        "invalidation_price": candidate.price_plan_invalidation_price,
        "cutoff": (
            None if candidate.price_plan_cutoff is None else candidate.price_plan_cutoff.isoformat()
        ),
        "sessions": candidate.price_plan_sessions,
        "method_version": candidate.price_plan_method_version,
        "evaluation_price_is_fill_price": False,
    }
    # Do not add invented cap/stop evidence to legacy records: only newly
    # assessed plans carry these optional fields.  The canonical content hash
    # and immutable entry_plan_json include every supplied risk constraint.
    for key in (
        "initial_risk_reference_price",
        "initial_risk_fraction",
        "maximum_entry_price",
        "initial_risk_qualified",
        "initial_risk_reason",
        "initial_protection_support",
        "initial_protection_atr",
        "initial_protection_evidence_date",
        "initial_protection_atr_cutoff",
        "initial_protection_method_version",
    ):
        value = getattr(candidate, f"price_plan_{key}", None)
        if value is not None:
            plan[key] = value.isoformat() if isinstance(value, (date, datetime)) else value
    return plan


def _plan_evaluation_rule(kind: str) -> str:
    if kind == "healthy_pullback_range":
        return "complete_daily_bar_trades_within_archived_range"
    if kind in {
        "reclaim_close_confirmation",
        "volume_breakout_close_confirmation",
    }:
        return "complete_daily_close_at_or_above_archived_trigger"
    return "unsupported_fail_closed"


def _batch_semantics(
    period: EveningPeriodDigest,
    *,
    archive_nature: ArchiveNature,
    has_plan_date: bool,
) -> dict[str, str]:
    if (
        archive_nature == "original"
        and has_plan_date
        and period.performance_nature == "official_action"
    ):
        return {
            "evaluation_mode": "action_simulation",
            "actionability": "primary_action",
            "cohort_nature": "action_qualified",
            "allocation_nature": "action_research",
        }
    if archive_nature == "reconstructed" and period.performance_nature in {
        "official_action",
        "risk_qualified_observation",
        "observation_only",
    }:
        return {
            "evaluation_mode": "reconstructed_observation",
            "actionability": "research_only",
            "cohort_nature": "observation_only",
            "allocation_nature": "observation_only",
        }
    if period.performance_nature == "risk_qualified_observation":
        return {
            "evaluation_mode": "observation_simulation",
            "actionability": "research_only",
            "cohort_nature": "risk_qualified",
            "allocation_nature": "risk_qualified_research",
        }
    if period.performance_nature == "observation_only":
        return {
            "evaluation_mode": "observation_simulation",
            "actionability": "research_only",
            "cohort_nature": "observation_only",
            "allocation_nature": "observation_only",
        }
    return {
        "evaluation_mode": "unavailable",
        "actionability": "unavailable",
        "cohort_nature": "unavailable",
        "allocation_nature": "unavailable",
    }


def _member_primary_action_eligible(candidate: EveningDigestCandidate, mode: str) -> bool:
    return bool(
        mode == "action_simulation"
        and candidate.allocation_nature == "action_research"
        and candidate.price_nature == "conditional_entry"
        and candidate.operational_stock_sleeve_weight is not None
        and candidate.operational_account_weight is not None
        and candidate.price_plan_kind is not None
        and not candidate.evidence_pending
    )


def _member_observation_eligible(candidate: EveningDigestCandidate, mode: str) -> bool:
    return bool(
        mode in {"observation_simulation", "reconstructed_observation"}
        and candidate.stock_sleeve_weight is not None
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_datetime(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    return value.astimezone(UTC)
