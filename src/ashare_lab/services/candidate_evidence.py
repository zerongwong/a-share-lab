"""Evidence enrichment after technical screening and before portfolio search.

The automatic financial gate is a deliberately narrow, versioned basic-quality
screen, not a guarantee of "blue-chip" status or future returns. The official
announcement gate requires a content-grounded review artifact; successful HTTP
requests and an absence of alarming title words can never make it pass.

No current snapshot is usable for historical replay. Announcement publication
time is distinct from the price cutoff and actual retrieval time is archived.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ashare_lab.adapters.candidate_financial_evidence import (
    METHOD_VERSION,
    _validate_request,
    content_hash,
    read_candidate_evidence,
    required_report_period,
)

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_CHECKLIST = frozenset({
    "latest_financial_report_read", "latest_annual_audit_read", "nonrecurring_profit_reviewed",
    "financial_metrics_cross_checked",
    "all_manifest_items_triaged", "material_risk_disclosures_reviewed", "no_unresolved_material_risk",
})
_FINANCIAL_INDUSTRIES = ("银行", "保险", "证券", "多元金融", "金融", "bank", "insurance", "financial")
_MATERIAL_TITLE = re.compile("立案|调查|处罚|诉讼|仲裁|担保|债务|违约|减值|更正|会计差错|审计|退市|风险警示|业绩预|重大资产|重大事项|关联交易")


@dataclass(frozen=True)
class CandidateEvidenceResult:
    symbol: str
    fundamental_gate: str
    announcement_gate: str
    execution_gate: str
    fundamental_reasons: tuple[str, ...]
    announcement_reasons: tuple[str, ...]
    execution_reasons: tuple[str, ...]
    financial_hash: str
    announcement_manifest_hash: str
    execution_hash: str
    retrieved_at: str | None
    financial_period: str | None
    publication_dates: tuple[str, ...]
    metrics: dict[str, float]
    announcement_items: tuple[dict[str, Any], ...]
    evidence_path: str | None
    method_version: str = METHOD_VERSION
    data_role: str = "current_candidate_evidence_not_historical_pit"


@dataclass(frozen=True)
class CandidateEvidenceBatch:
    metadata: dict[str, dict[str, Any]]
    results: tuple[CandidateEvidenceResult, ...]
    diagnostics: dict[str, Any]


def _aware(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("evidence timestamp must be timezone-aware")
    return parsed


def _number(row: dict, field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or value in (None, ""):
        raise ValueError("missing financial value")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite financial value")
    return number


def evaluate_financial_quality(
    payload: Mapping[str, Any], *, symbol: str, industry: str, knowledge_time: datetime
) -> tuple[str, tuple[str, ...], dict[str, float], tuple[str, ...]]:
    """Minimum quality: profitability, solvency and positive cash conversion.

    Thresholds are unvalidated research controls: current and prior-year same-
    period net profits and current ROE > 0; 0 <= liabilities/assets < 1;
    assets/equity > 0; operating cash flow/net profit > 0. These are *not*
    industry-relative valuation or a promise of stable profits. Banks/insurers
    cannot inherit industrial cash-flow/debt thresholds and remain unsupported.
    """
    if not isinstance(payload, Mapping):
        return "unknown", ("FINANCIAL_FIELDS_INCOMPLETE_OR_INVALID",), {}, ()
    if payload.get("error"):
        return "unknown", (str(payload["error"]),), {}, ()
    if any(token in industry.lower() for token in _FINANCIAL_INDUSTRIES):
        return "unknown", ("FINANCIAL_SECTOR_SPECIFIC_REVIEW_REQUIRED",), {}, ()
    if not industry or industry in {"未知", "unknown", "其他"}:
        return "unknown", ("INDUSTRY_CLASSIFICATION_REQUIRED_FOR_FINANCIAL_POLICY",), {}, ()
    try:
        selected = date.fromisoformat(str(payload["selected_period"]))
        known_date = knowledge_time.astimezone(_SHANGHAI).date()
        if selected < required_report_period(known_date) or selected >= known_date:
            return "unknown", ("LATEST_REQUIRED_FINANCIAL_REPORT_UNAVAILABLE",), {}, ()
        code, exchange = symbol.split(".")
        external = f"{exchange.lower()}.{code}"
        profit = payload["profit"]
        others = [payload[field] for field in ("balance", "cash_flow", "prior_year_profit")]
        if not isinstance(profit, dict) or any(not isinstance(rows, list) or len(rows) != 1 for rows in others):
            return "unknown", ("FINANCIAL_PERIOD_ROWS_INCOMPLETE_OR_DUPLICATE",), {}, ()
        balance, cash, prior = (rows[0] for rows in others)
        rows = [profit, balance, cash, prior]
        publication_dates = []
        for index, row in enumerate(rows):
            period = selected if index != 3 else selected.replace(year=selected.year - 1)
            published = date.fromisoformat(row["pubDate"])
            if row["code"] != external or row["statDate"] != period.isoformat():
                return "unknown", ("FINANCIAL_IDENTITY_OR_PERIOD_MISMATCH",), {}, ()
            if not period <= published <= known_date:
                return "unknown", ("FINANCIAL_PUBLICATION_OUTSIDE_KNOWLEDGE_BOUNDARY",), {}, ()
            publication_dates.append(published.isoformat())
        metrics = {
            "net_profit": _number(profit, "netProfit"),
            "prior_year_same_period_net_profit": _number(prior, "netProfit"),
            "roe": _number(profit, "roeAvg"),
            "liability_to_asset": _number(balance, "liabilityToAsset"),
            "asset_to_equity": _number(balance, "assetToEquity"),
            "operating_cash_flow_to_net_profit": _number(cash, "CFOToNP"),
        }
    except (TypeError, ValueError, KeyError, AttributeError):
        return "unknown", ("FINANCIAL_FIELDS_INCOMPLETE_OR_INVALID",), {}, ()
    reasons = []
    if metrics["net_profit"] <= 0 or metrics["prior_year_same_period_net_profit"] <= 0:
        reasons.append("CURRENT_OR_PRIOR_YEAR_PERIOD_NOT_PROFITABLE")
    if metrics["roe"] <= 0:
        reasons.append("NONPOSITIVE_RETURN_ON_EQUITY")
    if metrics["asset_to_equity"] <= 0 or metrics["liability_to_asset"] >= 1:
        reasons.append("NONPOSITIVE_EQUITY_OR_EXCESS_BALANCE_SHEET_LEVERAGE")
    if metrics["liability_to_asset"] < 0:
        return "unknown", ("FINANCIAL_RATIO_UNIT_OR_VALUE_INVALID",), metrics, tuple(publication_dates)
    if metrics["operating_cash_flow_to_net_profit"] <= 0:
        reasons.append("OPERATING_CASH_CONVERSION_NOT_POSITIVE")
    return (
        "veto" if reasons else "pass",
        tuple(reasons) if reasons else ("BASIC_PROFIT_SOLVENCY_CASH_QUALITY_CONFIRMED",),
        metrics, tuple(publication_dates),
    )


def evaluate_announcement_review(
    manifest: Mapping[str, Any], *, symbol: str, financial_hash: str,
    knowledge_time: datetime, review_dir: Path | None, financial_period: str | None = None,
    review_path: Path | None = None,
) -> tuple[str, tuple[str, ...]]:
    if not isinstance(manifest, Mapping):
        return "unknown", ("OFFICIAL_MANIFEST_SCHEMA_INVALID",)
    if manifest.get("error"):
        return "unknown", (str(manifest["error"]),)
    try:
        items = manifest["items"]
        if manifest["complete"] is not True or not isinstance(items, list) or not items:
            return "unknown", ("OFFICIAL_MANIFEST_INCOMPLETE_OR_EMPTY",)
        through = _aware(manifest["coverage_through"])
        if through > knowledge_time or knowledge_time - through > timedelta(hours=24):
            return "unknown", ("OFFICIAL_MANIFEST_STALE_OR_FUTURE",)
        if manifest["content_hash"] != content_hash(items):
            return "unknown", ("OFFICIAL_MANIFEST_HASH_MISMATCH",)
        indexed = {item["announcement_id"]: item for item in items}
        if len(indexed) != len(items) or any(_aware(item["published_at"]) > knowledge_time for item in items):
            return "unknown", ("OFFICIAL_MANIFEST_IDENTITY_OR_TIME_INVALID",)
        if review_dir is None:
            return "unknown", ("OFFICIAL_DOCUMENT_CONTENT_REVIEW_REQUIRED",)
        root = Path(review_dir).resolve()
        path = Path(review_path).resolve() if review_path is not None else root / f"{symbol}.json"
        if not path.resolve().is_relative_to(root):
            return "unknown", ("OFFICIAL_REVIEW_PATH_OUTSIDE_REVIEW_DIRECTORY",)
        if not path.is_file() or path.stat().st_size > 2_000_000:
            return "unknown", ("OFFICIAL_DOCUMENT_CONTENT_REVIEW_REQUIRED",)
        review = json.loads(path.read_text())
        if (
            review["method_version"] != METHOD_VERSION or review["symbol"] != symbol
            or review["manifest_hash"] != manifest["content_hash"]
            or review["financial_hash"] != financial_hash
        ):
            return "unknown", ("OFFICIAL_REVIEW_CONTENT_CHANGED_REVIEW_REQUIRED",)
        reviewed_at = _aware(review["reviewed_at"])
        if reviewed_at > knowledge_time or any(_aware(item["published_at"]) > reviewed_at for item in items) or not review.get("reviewer") or not review.get("reason"):
            return "unknown", ("OFFICIAL_REVIEW_PROVENANCE_INVALID",)
        if review["status"] == "veto":
            return "veto", ("DOCUMENT_REVIEW_IDENTIFIED_MATERIAL_RISK",)
        if review["status"] != "pass" or any(review["checklist"].get(key) is not True for key in _CHECKLIST):
            return "unknown", ("OFFICIAL_REVIEW_CHECKLIST_INCOMPLETE",)
        triage = review["triage"]
        if not isinstance(triage, list) or len(triage) != len(items) or {r["announcement_id"] for r in triage} != set(indexed):
            return "unknown", ("OFFICIAL_REVIEW_MANIFEST_COVERAGE_INCOMPLETE",)
        reviewed_ids, roles = set(), set()
        for document in review["documents"]:
            item = indexed[document["announcement_id"]]
            if document["url"] != item["url"] or not document.get("findings"):
                return "unknown", ("OFFICIAL_REVIEW_DOCUMENT_PROVENANCE_INVALID",)
            file = (root / document["file"]).resolve()
            if not file.is_relative_to(root) or not file.is_file() or file.stat().st_size > 50_000_000:
                return "unknown", ("OFFICIAL_REVIEW_DOCUMENT_FILE_INVALID",)
            binary = file.read_bytes()
            if not binary.startswith(b"%PDF-") or hashlib.sha256(binary).hexdigest() != document["sha256"]:
                return "unknown", ("OFFICIAL_REVIEW_DOCUMENT_HASH_MISMATCH",)
            reviewed_ids.add(document["announcement_id"])
            roles.add(document["role"])
            title = item["title"]
            if document["role"] == "annual_audit" and ("审计" not in title and "年度报告" not in title or "摘要" in title):
                return "unknown", ("OFFICIAL_REVIEW_AUDIT_DOCUMENT_ROLE_MISMATCH",)
            if document["role"] == "annual_audit":
                known_date = knowledge_time.astimezone(_SHANGHAI).date()
                minimum_year = known_date.year - (1 if known_date >= date(known_date.year, 5, 1) else 2)
                if financial_period is not None:
                    selected = date.fromisoformat(financial_period)
                    if selected.month == 12:
                        # An early-published new annual supersedes the statutory
                        # floor; its own audit cannot be replaced by last year's.
                        minimum_year = max(minimum_year, selected.year)
                years = {int(value) for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", title)}
                if not any(minimum_year <= year < known_date.year for year in years):
                    return "unknown", ("OFFICIAL_REVIEW_LATEST_ANNUAL_AUDIT_REQUIRED",)
            if document["role"] == "latest_financial_report" and financial_period is not None:
                period = date.fromisoformat(financial_period)
                period_words = {3: ("一季度", "第一季度"), 6: ("半年度", "中期"), 9: ("三季度", "第三季度"), 12: ("年度",)}
                if str(period.year) not in title or not any(word in title for word in period_words[period.month]) or "摘要" in title:
                    return "unknown", ("OFFICIAL_REVIEW_LATEST_FINANCIAL_DOCUMENT_ROLE_MISMATCH",)
        if not {"latest_financial_report", "annual_audit"} <= roles:
            return "unknown", ("OFFICIAL_REVIEW_FINANCIAL_AND_AUDIT_DOCUMENTS_REQUIRED",)
        for row in triage:
            if not row.get("reason") or row["decision"] not in {"document_reviewed", "routine_nonmaterial"}:
                return "unknown", ("OFFICIAL_REVIEW_TRIAGE_INVALID",)
            if row["decision"] == "document_reviewed" and row["announcement_id"] not in reviewed_ids:
                return "unknown", ("OFFICIAL_REVIEW_TRIAGED_DOCUMENT_MISSING",)
            if row["decision"] == "routine_nonmaterial" and _MATERIAL_TITLE.search(indexed[row["announcement_id"]]["title"]):
                return "unknown", ("POTENTIALLY_MATERIAL_DOCUMENT_REQUIRES_CONTENT_REVIEW",)
        return "pass", ("CONTENT_GROUNDED_OFFICIAL_REVIEW_CONFIRMED",)
    except (TypeError, ValueError, KeyError, OSError, AttributeError):
        return "unknown", ("OFFICIAL_REVIEW_INVALID",)


def evaluate_execution_status(
    payload: Mapping[str, Any], *, symbol: str, cutoff: date, metadata: Mapping[str, Any]
) -> tuple[str, tuple[str, ...]]:
    """Formation-close tradability only; not a next-session fill guarantee."""
    if not isinstance(payload, Mapping) or payload.get("error"):
        return "unknown", (str(payload.get("error") or "EXECUTION_PROVIDER_UNAVAILABLE") if isinstance(payload, Mapping) else "EXECUTION_RESPONSE_INVALID",)
    try:
        rows = payload["rows"]
        if payload["cutoff"] != cutoff.isoformat() or not isinstance(rows, list) or len(rows) != 1:
            return "unknown", ("EXECUTION_CUTOFF_OR_CARDINALITY_INVALID",)
        if content_hash(rows) != payload["content_hash"]:
            return "unknown", ("EXECUTION_CONTENT_HASH_MISMATCH",)
        row = rows[0]
        code, exchange = symbol.split(".")
        if row["code"] != f"{exchange.lower()}.{code}" or row["date"] != cutoff.isoformat():
            return "unknown", ("EXECUTION_IDENTITY_OR_DATE_MISMATCH",)
        if row["tradestatus"] not in {"0", "1"} or row["isST"] not in {"0", "1"}:
            return "unknown", ("EXECUTION_STATUS_VALUE_UNKNOWN",)
        if row["tradestatus"] == "0" or row["isST"] == "1" or metadata.get("is_suspended") is True:
            return "veto", ("FORMATION_SUSPENDED_OR_ST",)
        if metadata.get("is_limit_up_at_cutoff") is True:
            return "veto", ("FORMATION_LIMIT_UP",)
        if metadata.get("is_limit_up_at_cutoff") is not False:
            return "unknown", ("FORMATION_LIMIT_STATUS_UNCONFIRMED",)
        return "pass", ("FORMATION_DATE_TRADED_NON_ST_NOT_LIMIT_UP",)
    except (TypeError, ValueError, KeyError):
        return "unknown", ("EXECUTION_RESPONSE_INVALID",)


def _cached_receipt(path: Path, *, symbol: str, knowledge_time: datetime) -> dict | None:
    try:
        if path.stat().st_size > 2_000_000:
            return None
        payload = json.loads(path.read_text())
        fetched = _aware(payload["retrieved_at"])
        if (
            payload["symbol"] != symbol or payload["method_version"] != METHOD_VERSION
            or not timedelta(0) <= knowledge_time - fetched <= timedelta(hours=24)
            or fetched.astimezone(_SHANGHAI).date() != knowledge_time.astimezone(_SHANGHAI).date()
        ):
            return None
        return payload
    except (OSError, TypeError, ValueError, KeyError):
        return None


def _reusable_financial_receipt(payload: Mapping[str, Any], *, cutoff: date) -> bool:
    return (
        isinstance(payload.get("financials"), Mapping)
        and bool(payload["financials"].get("profit")) and not payload["financials"].get("error")
        and isinstance(payload.get("execution"), Mapping)
        and not payload["execution"].get("error")
        and payload["execution"].get("cutoff") == cutoff.isoformat()
        and isinstance(payload["execution"].get("rows"), list)
        and len(payload["execution"]["rows"]) == 1
    )


def _valid_cached(path: Path, *, symbol: str, knowledge_time: datetime, cutoff: date) -> dict | None:
    payload = _cached_receipt(path, symbol=symbol, knowledge_time=knowledge_time)
    # Outages/partial manifests are retried. Same-day advancement to a newer
    # price cutoff also requires a fresh execution receipt rather than repeatedly
    # reusing yesterday's status until midnight.
    if payload is None or not _reusable_financial_receipt(payload, cutoff=cutoff):
        return None
    announcements = payload.get("announcements")
    if not isinstance(announcements, Mapping) or announcements.get("error") or announcements.get("complete") is not True:
        return None
    return payload


def _attempts(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 1_000_000 else 0


def enrich_candidate_evidence(
    metadata: Mapping[str, Mapping[str, Any]], *, symbols: Sequence[str], cutoff: date,
    knowledge_time: datetime, cache_dir: Path | None = None, review_dir: Path | None = None,
    total_timeout_seconds: float = 75, provider=None, clock=None,
) -> CandidateEvidenceBatch:
    """Enrich only a bounded, technically screened list; never weaken unknown gates.

    Online reads are current-only. They preserve the original publication boundary
    and expose ``effective_knowledge_time`` >= actual retrieval time for the caller
    to archive. Errors are diagnostic outcomes, not "no stocks qualify".
    """
    requested = _validate_request(symbols, cutoff, knowledge_time, total_timeout_seconds)
    metadata_copy = {symbol: dict(item) for symbol, item in metadata.items()}
    now = (clock or (lambda: datetime.now(UTC)))()
    if now.utcoffset() is None:
        raise ValueError("clock must return an aware timestamp")
    # Never inject a current revised snapshot into a historical request.
    historical = knowledge_time.astimezone(_SHANGHAI).date() != now.astimezone(_SHANGHAI).date()
    cache = Path(cache_dir) if cache_dir is not None else None
    payloads = {}
    cached_receipts = {}
    if cache is not None:
        for symbol in requested:
            path = cache / f"{symbol}.json"
            receipt = _cached_receipt(path, symbol=symbol, knowledge_time=knowledge_time)
            if receipt is not None:
                cached_receipts[symbol] = receipt
            cached = _valid_cached(path, symbol=symbol, knowledge_time=knowledge_time, cutoff=cutoff)
            if cached is not None:
                payloads[symbol] = cached
    missing = [symbol for symbol in requested if symbol not in payloads]
    reusable = {symbol: cached_receipts[symbol] for symbol in missing if symbol in cached_receipts and _reusable_financial_receipt(cached_receipts[symbol], cutoff=cutoff)}
    # Fair retries: unprocessed financial candidates first; for already-known
    # financial candidates retry the least-attempted announcement fetch first.
    # A permanently slow first name cannot always consume the entire deadline.
    missing.sort(key=lambda symbol: (
        symbol in reusable,
        _attempts(cached_receipts.get(symbol, {}), "_announcement_attempts" if symbol in reusable else "_financial_attempts"),
        requested.index(symbol),
    ))
    if missing and not historical:
        reader = provider or read_candidate_evidence
        try:
            kwargs = {"reuse_receipts": reusable} if provider is None and reusable else {}
            fresh = reader(
                missing, cutoff=cutoff, knowledge_time=knowledge_time,
                timeout_seconds=total_timeout_seconds, **kwargs,
            )
            if isinstance(fresh, Mapping):
                payloads.update({symbol: value for symbol, value in fresh.items() if symbol in missing})
        except Exception:
            # No provider exception text enters reports, logs or notification bodies.
            pass
    results = []
    effective_knowledge_time = knowledge_time
    for symbol in requested:
        payload = payloads.get(symbol, {})
        if not isinstance(payload, Mapping):
            payload = {}
        payload = dict(payload)
        if symbol in missing:
            old = cached_receipts.get(symbol, {})
            payload["_financial_attempts"] = _attempts(old, "_financial_attempts") + int(payload.get("financial_attempted") is True)
            payload["_announcement_attempts"] = _attempts(old, "_announcement_attempts") + int(payload.get("announcement_attempted") is True)
        trusted_time = False
        try:
            fetched = _aware(payload["retrieved_at"])
            trusted_time = (
                payload["symbol"] == symbol and payload["method_version"] == METHOD_VERSION
                and abs(fetched - knowledge_time) <= timedelta(hours=24)
                and fetched.astimezone(_SHANGHAI).date() == knowledge_time.astimezone(_SHANGHAI).date()
                and fetched <= (clock or (lambda: datetime.now(UTC)))() + timedelta(seconds=5)
                and not historical
            )
            if trusted_time:
                effective_knowledge_time = max(effective_knowledge_time, fetched)
        except (TypeError, ValueError, KeyError):
            pass
        financials = payload.get("financials", {}) if trusted_time else {}
        announcements = payload.get("announcements", {}) if trusted_time else {}
        execution = payload.get("execution", {}) if trusted_time else {}
        financials = financials if isinstance(financials, Mapping) else {}
        announcements = announcements if isinstance(announcements, Mapping) else {}
        execution = execution if isinstance(execution, Mapping) else {}
        financial_hash = content_hash(financials)
        item = metadata_copy.setdefault(symbol, {})
        if not trusted_time:
            gate, reasons, metrics, publication_dates = "unknown", (
                "CURRENT_SNAPSHOT_FORBIDDEN_IN_HISTORICAL_REPLAY" if historical
                else str(payload.get("error") or "PROVIDER_RECEIPT_INVALID_OR_UNAVAILABLE"),
            ), {}, ()
            announcement_gate, announcement_reasons = "unknown", reasons
            execution_gate, execution_reasons = "unknown", reasons
        else:
            gate, reasons, metrics, publication_dates = evaluate_financial_quality(
                financials, symbol=symbol, industry=str(item.get("industry") or ""),
                knowledge_time=knowledge_time,
            )
            announcement_gate, announcement_reasons = evaluate_announcement_review(
                announcements, symbol=symbol, financial_hash=financial_hash,
                knowledge_time=knowledge_time, review_dir=review_dir,
                financial_period=financials.get("selected_period"),
            )
            execution_gate, execution_reasons = evaluate_execution_status(
                execution, symbol=symbol, cutoff=cutoff, metadata=item,
            )
        evidence_path = None
        if cache is not None and trusted_time:
            try:
                cache.mkdir(parents=True, exist_ok=True)
                path = cache / f"{symbol}.json"
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                temporary.replace(path)
                evidence_path = str(path)
            except OSError:
                pass
        result = CandidateEvidenceResult(
            symbol=symbol, fundamental_gate=gate, announcement_gate=announcement_gate,
            execution_gate=execution_gate,
            fundamental_reasons=reasons, announcement_reasons=announcement_reasons,
            execution_reasons=execution_reasons,
            financial_hash=financial_hash,
            announcement_manifest_hash=str(announcements.get("content_hash") or ""),
            execution_hash=str(execution.get("content_hash") or ""),
            retrieved_at=payload.get("retrieved_at") if trusted_time else None,
            financial_period=financials.get("selected_period"), publication_dates=publication_dates,
            metrics=metrics, announcement_items=tuple(announcements.get("items") or ()),
            evidence_path=evidence_path,
        )
        results.append(result)
        # Sticky explicit vetoes are never downgraded by an unavailable provider.
        item["fundamental_gate"] = "veto" if item.get("fundamental_gate") == "veto" else gate
        item["announcement_gate"] = "veto" if item.get("announcement_gate") == "veto" else announcement_gate
        if execution_gate == "veto":
            item["is_buyable_at_cutoff"] = False
        elif item.get("is_buyable_at_cutoff") is not False:
            item["is_buyable_at_cutoff"] = True if execution_gate == "pass" else None
        item["candidate_evidence_method_version"] = METHOD_VERSION
        item["candidate_evidence_reasons"] = (*reasons, *announcement_reasons, *execution_reasons)
        item["candidate_evidence_financial_hash"] = financial_hash
        item["candidate_evidence_announcement_manifest_hash"] = result.announcement_manifest_hash
        item["candidate_evidence_retrieved_at"] = result.retrieved_at
    return CandidateEvidenceBatch(
        metadata=metadata_copy, results=tuple(results),
        diagnostics={
            "method_version": METHOD_VERSION, "requested_count": len(requested),
            "financial_pass_count": sum(r.fundamental_gate == "pass" for r in results),
            "announcement_pass_count": sum(r.announcement_gate == "pass" for r in results),
            "execution_pass_count": sum(r.execution_gate == "pass" for r in results),
            "double_confirmation_count": sum(r.fundamental_gate == r.announcement_gate == "pass" for r in results),
            "unknown_count": sum("unknown" in (r.fundamental_gate, r.announcement_gate) for r in results),
            "veto_count": sum("veto" in (r.fundamental_gate, r.announcement_gate) for r in results),
            "effective_knowledge_time": effective_knowledge_time.isoformat(),
            "price_cutoff": cutoff.isoformat(),
            "financial_policy_scope": "basic_profit_solvency_cash_quality_not_blue_chip_certification",
            "official_review_scope": "annual_coverage_aware_manifest_and_explicit_content_review_not_all_risks_guaranteed",
            "historical_replay_allowed": False,
        },
    )
