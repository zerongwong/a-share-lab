"""Local-only orchestration for two Edwards--Magee stop counterfactuals.

This service is intentionally outside the production holding decision state
machine.  It archives immutable observations for later comparison, receives no
notification channel, and never writes a production protective stop or action.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import pandas as pd

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.analytics.magee_shadow import (
    MAGEE_SHADOW_METHOD_VERSION,
    MageeShadowDataError,
    MageeShadowEvent,
    MageeShadowResult,
    MageeShadowVariant,
    evaluate_magee_shadow,
)
from ashare_lab.domain.data_sources import DEFAULT_MARKET_OVERLAY_SOURCE_ID
from ashare_lab.services.holding_ledger import ActiveHolding, get_active_holding_portfolio
from ashare_lab.services.run_active_holding_review import (
    ActiveHoldingHistoryLoad,
    load_active_holding_histories,
)

HOLDING_STOP_SHADOW_RUNNER_VERSION: Final = "holding-stop-shadow-runner-v0.1.0"
HOLDING_STOP_SHADOW_VARIANTS: Final = (
    MageeShadowVariant.THREE_DAY_ESCAPE_6PCT,
    MageeShadowVariant.NEW_HIGH_3PCT_6PCT,
)


class HoldingStopShadowRunStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    DATA_NOT_READY = "data_not_ready"
    NO_HOLDINGS = "no_holdings"


@dataclass(frozen=True, slots=True)
class HoldingStopShadowRunSummary:
    status: HoldingStopShadowRunStatus
    portfolio_id: str | None
    holding_version: int | None
    data_cutoff: date | None
    observation_count: int
    inserted_count: int
    variant_count: int
    method_version: str = HOLDING_STOP_SHADOW_RUNNER_VERSION


@dataclass(frozen=True, slots=True)
class _CompanyActionEvidence:
    clear: bool | None
    evidence_id: str | None
    source: str | None
    covered_from: date | None
    clear_through: date | None
    knowledge_time: datetime | None
    complete_for_interval: bool


HistoryLoader = Callable[..., ActiveHoldingHistoryLoad | None]
ShadowEvaluator = Callable[..., MageeShadowResult]


def run_holding_stop_shadows(
    repository: SQLiteRepository,
    *,
    dataset_root: str | Path,
    overlay_root: str | Path,
    as_of: date,
    evaluated_at: datetime | None = None,
    persist: bool = True,
    overlay_source_id: str = DEFAULT_MARKET_OVERLAY_SOURCE_ID,
    _history_loader: HistoryLoader = load_active_holding_histories,
    _evaluator: ShadowEvaluator = evaluate_magee_shadow,
) -> HoldingStopShadowRunSummary:
    """Evaluate and archive both variants for the explicit current holdings."""

    if not isinstance(as_of, date) or isinstance(as_of, datetime):
        raise TypeError("as_of must be a date")
    timestamp = datetime.now(UTC) if evaluated_at is None else evaluated_at
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
        raise ValueError("evaluated_at must be timezone-aware")

    loaded = _history_loader(
        repository,
        dataset_root=dataset_root,
        overlay_root=overlay_root,
        as_of=as_of,
        overlay_source_id=overlay_source_id,
    )
    if loaded is None:
        return HoldingStopShadowRunSummary(
            status=HoldingStopShadowRunStatus.NO_HOLDINGS,
            portfolio_id=None,
            holding_version=None,
            data_cutoff=None,
            observation_count=0,
            inserted_count=0,
            variant_count=0,
        )
    portfolio = get_active_holding_portfolio(repository, as_of=loaded.data_cutoff)
    if (
        portfolio is None
        or portfolio.status != "active"
        or portfolio.id != loaded.portfolio_id
        or portfolio.version != loaded.holding_version
        or portfolio.holding_weeks != loaded.holding_weeks
    ):
        raise ValueError("Holding shadow loader identity does not match the explicit ledger")

    unavailable = set(loaded.unavailable_symbols)
    records: list[dict[str, object]] = []
    ready_count = 0
    for holding in portfolio.positions:
        history = loaded.histories.get(holding.symbol)
        action_evidence = _company_action_evidence(
            holding,
            cutoff=loaded.data_cutoff,
            evaluated_at=timestamp,
        )
        for variant in HOLDING_STOP_SHADOW_VARIANTS:
            record = _build_shadow_record(
                repository,
                portfolio_id=portfolio.id,
                holding_weeks=portfolio.holding_weeks,
                holding=holding,
                frame=history,
                cutoff=loaded.data_cutoff,
                evaluated_at=timestamp,
                variant=variant,
                company_action=action_evidence,
                history_unavailable=holding.symbol in unavailable,
                evaluator=_evaluator,
            )
            records.append(record)
            if record["status"] in {"ready", "no_confirmed_baseline"}:
                ready_count += 1

    inserted = repository.archive_holding_shadow_events(records) if persist else 0
    if ready_count == len(records):
        status = HoldingStopShadowRunStatus.COMPLETED
    elif ready_count:
        status = HoldingStopShadowRunStatus.PARTIAL
    else:
        status = HoldingStopShadowRunStatus.DATA_NOT_READY
    return HoldingStopShadowRunSummary(
        status=status,
        portfolio_id=portfolio.id,
        holding_version=portfolio.version,
        data_cutoff=loaded.data_cutoff,
        observation_count=len(records),
        inserted_count=inserted,
        variant_count=len(HOLDING_STOP_SHADOW_VARIANTS),
    )


def _build_shadow_record(
    repository: SQLiteRepository,
    *,
    portfolio_id: str,
    holding_weeks: int,
    holding: ActiveHolding,
    frame: pd.DataFrame | None,
    cutoff: date,
    evaluated_at: datetime,
    variant: MageeShadowVariant,
    company_action: _CompanyActionEvidence,
    history_unavailable: bool,
    evaluator: ShadowEvaluator,
) -> dict[str, object]:
    parameters = _variant_parameters(variant)
    parameter_hash = _hash_payload(parameters)
    input_data_hash = _observed_input_hash(
        frame,
        entry_date=holding.entry_date,
        cutoff=cutoff,
    )
    result: MageeShadowResult | None = None
    failure_reason: str | None = None
    if frame is None or history_unavailable:
        failure_reason = "holding_history_not_current_at_verified_cutoff"
    else:
        try:
            result = evaluator(
                frame,
                entry_date=holding.entry_date,
                as_of=cutoff,
                variant=variant,
            )
        except (MageeShadowDataError, TypeError, ValueError):
            failure_reason = "magee_shadow_evidence_failed_closed"

    if result is not None and result.variant is not variant:
        raise ValueError("Magee shadow evaluator returned another variant")
    if result is not None and result.method_version != MAGEE_SHADOW_METHOD_VERSION:
        raise ValueError("Magee shadow evaluator method version mismatch")
    if result is not None and result.entry_date != holding.entry_date:
        raise ValueError("Magee shadow evaluator entry date mismatch")

    cutoff_matches = result is not None and result.data_cutoff == cutoff
    prior = repository.get_latest_holding_shadow_state(
        position_key=holding.position_key,
        holding_weeks=holding_weeks,
        variant_key=variant.value,
        method_version=MAGEE_SHADOW_METHOD_VERSION,
        parameter_hash=parameter_hash,
        before_cutoff=cutoff,
    )
    previous_stop = None if prior is None else _optional_float(prior.get("effective_shadow_stop"))
    engine_stop = None if result is None else result.effective_stop
    effective_stop = _maximum_optional(previous_stop, engine_stop)
    next_stop = _maximum_optional(
        effective_stop,
        None if result is None else result.next_effective_stop,
    )
    close_breach = (
        None
        if result is None or effective_stop is None or result.latest_close is None
        else result.latest_close < effective_stop
    )
    intraday_touch = (
        None
        if result is None or effective_stop is None or result.latest_low is None
        else result.latest_low <= effective_stop
    )

    latest_event = None if result is None or not result.events else result.events[-1]
    active_event = _active_event(result, effective_stop)
    reasons = list(result.reasons if result is not None else ())
    if failure_reason is not None:
        reasons.extend((failure_reason, "shadow_observation_not_used_for_comparison"))
    if result is not None and not cutoff_matches:
        reasons.extend(
            (
                "shadow_data_cutoff_mismatch",
                "shadow_observation_not_used_for_comparison",
            )
        )
    if not company_action.complete_for_interval:
        reasons.append("company_action_interval_or_knowledge_coverage_incomplete")
    # The current hybrid loader proves a verified common cutoff, but it does
    # not yet provide point-in-time official-calendar evidence that every
    # expected session between entry and cutoff is present.  Keep comparison
    # eligibility fail-closed until that separate contract is integrated.
    official_session_chain_verified = False
    reasons.append("official_session_chain_unverified_shadow_not_comparison_eligible")
    reasons.append(f"input_data_hash:{input_data_hash}")
    if previous_stop is not None and engine_stop is not None and engine_stop < previous_stop:
        reasons.append("archived_shadow_floor_prevented_line_from_moving_down")

    if failure_reason is not None or not cutoff_matches:
        status = "data_not_ready"
    elif not company_action.complete_for_interval or not official_session_chain_verified:
        status = "needs_review"
    elif effective_stop is None and next_stop is None:
        status = "no_confirmed_baseline"
    else:
        status = "ready"
    comparison_eligible = bool(
        status == "ready"
        and company_action.complete_for_interval
        and company_action.clear is True
        and official_session_chain_verified
    )
    evidence_payload = {
        "position_key": holding.position_key,
        "holding_weeks": holding_weeks,
        "entry_date": holding.entry_date,
        "cutoff": cutoff,
        "variant": variant.value,
        "parameter_hash": parameter_hash,
        "input_data_hash": input_data_hash,
        "result": None if result is None else asdict(result),
        "company_action": asdict(company_action),
        "loader_cutoff_matches": cutoff_matches,
        "official_session_chain_verified": official_session_chain_verified,
    }
    evidence_hash = _hash_payload(evidence_payload)
    identity = hashlib.sha256(
        (
            f"{holding.position_key}|{holding_weeks}|{cutoff}|{variant.value}|"
            f"{MAGEE_SHADOW_METHOD_VERSION}|{parameter_hash}|{evidence_hash}"
        ).encode()
    ).hexdigest()[:32]
    return {
        "id": f"holding-shadow:{identity}",
        "revision_id": portfolio_id,
        "position_id": holding.id,
        "position_key": holding.position_key,
        "symbol": holding.symbol,
        "holding_weeks": holding_weeks,
        "holding_version": holding.version,
        "entry_date": holding.entry_date,
        "data_cutoff": cutoff,
        "evaluated_at": evaluated_at,
        "archive_nature": "live_shadow",
        "variant_key": variant.value,
        "status": status,
        "source_timeframe": "daily",
        "baseline_kind": None if latest_event is None else latest_event.kind.value,
        "baseline_date": None if latest_event is None else latest_event.baseline_date,
        "confirmation_date": None if latest_event is None else latest_event.confirmed_on,
        "baseline_price": None if latest_event is None else latest_event.baseline_price,
        "latest_close": None if result is None else result.latest_close,
        "latest_low": None if result is None else result.latest_low,
        "candidate_stop": None if latest_event is None else latest_event.candidate_stop,
        "previous_shadow_stop": previous_stop,
        "effective_shadow_stop": effective_stop,
        "effective_from_date": None if active_event is None else active_event.effective_on,
        "next_effective_shadow_stop": next_stop,
        "latest_intraday_touch_observed": intraday_touch,
        "intraday_touch_observed": (None if result is None else result.ever_low_touched_stop),
        "intraday_touch_date": None if result is None else result.first_low_touch_on,
        "latest_close_breach_observed": close_breach,
        "close_breach_observed": (None if result is None else result.ever_close_below_stop),
        "close_breach_date": None if result is None else result.first_close_breach_on,
        "company_action_clear": company_action.clear,
        "company_action_evidence_id": company_action.evidence_id,
        "company_action_evidence_source": company_action.source,
        "company_action_covered_from": company_action.covered_from,
        "company_action_clear_through": company_action.clear_through,
        "company_action_knowledge_time": company_action.knowledge_time,
        "evaluation_eligible": comparison_eligible,
        "parameters_json": parameters,
        "reason_json": reasons,
        "input_data_hash": input_data_hash,
        "evidence_hash": evidence_hash,
        "parameter_hash": parameter_hash,
        "method_version": MAGEE_SHADOW_METHOD_VERSION,
        "created_at": evaluated_at,
    }


def _active_event(
    result: MageeShadowResult | None,
    effective_stop: float | None,
) -> MageeShadowEvent | None:
    if result is None or effective_stop is None:
        return None
    active = [
        event
        for event in result.events
        if event.effective_on is not None
        and event.effective_on <= result.as_of
        and event.effective_stop == effective_stop
    ]
    return active[-1] if active else None


def _variant_parameters(variant: MageeShadowVariant) -> dict[str, object]:
    parameters: dict[str, object] = {
        "sampling_interval": "completed_daily",
        "entry_date_inclusive": True,
        "stop_distance_fraction": 0.06,
        "confirmed_line_effective": "next_observed_trading_bar",
        "primary_comparison_trigger": "complete_close_strictly_below_line",
        "intraday_touch_nature": "diagnostic_only",
        "official_session_chain_evidence": "not_integrated_fail_closed",
    }
    if variant is MageeShadowVariant.THREE_DAY_ESCAPE_6PCT:
        parameters.update(
            {
                "escape_bar_count": 3,
                "escape_rule": "each_full_range_strictly_above_candidate_day_high",
            }
        )
    else:
        parameters.update(
            {
                "new_high_threshold_fraction": 0.03,
                "baseline": "accepted_new_high_day_low",
            }
        )
    return parameters


def _company_action_evidence(
    holding: ActiveHolding,
    *,
    cutoff: date,
    evaluated_at: datetime,
) -> _CompanyActionEvidence:
    metadata = dict(holding.metadata)
    explicit = metadata.get("company_action_clear")
    clear = explicit if isinstance(explicit, bool) else None
    evidence_id = _optional_text(metadata.get("company_action_evidence_id"))
    source = _optional_text(metadata.get("company_action_evidence_source"))
    covered_from = _optional_date(metadata.get("company_action_covered_from"))
    clear_through = _optional_date(metadata.get("company_action_clear_through"))
    knowledge_time = _optional_datetime(metadata.get("company_action_knowledge_time"))
    complete = bool(
        clear is True
        and evidence_id
        and source
        and covered_from is not None
        and covered_from <= holding.entry_date
        and clear_through is not None
        and clear_through >= cutoff
        and knowledge_time is not None
        and knowledge_time <= evaluated_at
    )
    return _CompanyActionEvidence(
        clear=clear,
        evidence_id=evidence_id,
        source=source,
        covered_from=covered_from,
        clear_through=clear_through,
        knowledge_time=knowledge_time,
        complete_for_interval=complete,
    )


def _maximum_optional(*values: float | None) -> float | None:
    supplied = [float(value) for value in values if value is not None]
    return max(supplied) if supplied else None


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_date(value: object) -> date | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return timestamp.date()


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        return None
    return timestamp.to_pydatetime()


def _hash_payload(value: Mapping[str, Any] | object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _observed_input_hash(
    frame: pd.DataFrame | None,
    *,
    entry_date: date,
    cutoff: date,
) -> str:
    """Hash the exact entry-through-cutoff OHLC evidence without archiving it."""

    required = ("trade_date", "open", "high", "low", "close")
    if frame is None:
        return _hash_payload({"status": "missing_frame"})
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return _hash_payload(
            {
                "status": "missing_columns",
                "missing": missing,
                "row_count": len(frame),
            }
        )
    selected = frame.loc[:, required].copy()
    parsed = pd.to_datetime(selected["trade_date"], errors="coerce")
    if parsed.isna().any():
        return _hash_payload(
            {
                "status": "invalid_trade_date",
                "rows": selected.to_dict(orient="records"),
            }
        )
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    selected["trade_date"] = parsed.dt.date
    for column in ("open", "high", "low", "close"):
        numeric = pd.to_numeric(selected[column], errors="coerce")
        if numeric.isna().any():
            return _hash_payload(
                {
                    "status": f"invalid_{column}",
                    "rows": selected.to_dict(orient="records"),
                }
            )
        selected[column] = numeric.astype(float)
    selected = selected.loc[
        (selected["trade_date"] >= entry_date) & (selected["trade_date"] <= cutoff)
    ].sort_values("trade_date")
    return _hash_payload(
        {
            "status": "observed_entry_through_cutoff",
            "rows": selected.to_dict(orient="records"),
        }
    )
