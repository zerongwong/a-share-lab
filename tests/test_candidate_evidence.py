from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from ashare_lab.adapters.candidate_financial_evidence import (
    METHOD_VERSION,
    REPORTING_WINDOW_SOURCE,
    announcement_window_start,
    collect_announcements,
    collect_execution_status,
    collect_financials,
    collect_worker_batch,
    content_hash,
    read_candidate_evidence,
    required_report_period,
)
from ashare_lab.services.candidate_evidence import (
    _CHECKLIST,
    enrich_candidate_evidence,
    evaluate_announcement_review,
    evaluate_execution_status,
    evaluate_financial_quality,
)

NOW = datetime(2026, 9, 22, 8, 40, tzinfo=ZoneInfo("Asia/Shanghai"))
SYMBOL = "601298.SH"
CUTOFF = date(2026, 9, 21)


def financials():
    base = {"code": "sh.601298", "pubDate": "2026-08-29", "statDate": "2026-06-30"}
    return {
        "provider": "baostock", "required_period": "2026-06-30", "selected_period": "2026-06-30",
        "profit": {**base, "netProfit": "100", "roeAvg": "0.05"},
        "balance": [{**base, "liabilityToAsset": "0.3", "assetToEquity": "1.4"}],
        "cash_flow": [{**base, "CFOToNP": "1.2"}],
        "prior_year_profit": [{**base, "pubDate": "2025-08-29", "statDate": "2025-06-30", "netProfit": "80"}],
    }


def manifest():
    items = [
        {"announcement_id": "annual", "title": "2025年年度报告", "published_at": "2026-03-20T00:00:00+08:00", "url": "https://static.cninfo.com.cn/finalpage/2026-03-20/annual.PDF"},
        {"announcement_id": "latest", "title": "2026年半年度报告", "published_at": "2026-08-29T00:00:00+08:00", "url": "https://static.cninfo.com.cn/finalpage/2026-08-29/latest.PDF"},
    ]
    return {
        "provider": "cninfo", "source_url": "https://www.cninfo.com.cn/new/hisAnnouncement/query",
        "symbol": SYMBOL,
        "coverage_from": "2025-09-22", "coverage_through": NOW.isoformat(),
        "complete": True, "record_count": len(items), "total_provider_records": len(items),
        "items": items, "content_hash": content_hash(items),
    }


def payload():
    return {"symbol": SYMBOL, "method_version": METHOD_VERSION, "retrieved_at": NOW.isoformat(), "financials": financials(), "announcements": manifest(), "execution": execution()}


def execution():
    rows = [{"date": CUTOFF.isoformat(), "code": "sh.601298", "tradestatus": "1", "isST": "0", "peTTM": "12.5", "pbMRQ": "1.3"}]
    return {"cutoff": CUTOFF.isoformat(), "rows": rows, "content_hash": content_hash(rows)}


def check_financial(data=None, industry="港口"):
    return evaluate_financial_quality(data or financials(), symbol=SYMBOL, industry=industry, knowledge_time=NOW)


@pytest.mark.parametrize(("day", "expected"), [
    (date(2026, 4, 30), date(2025, 9, 30)),
    (date(2026, 5, 1), date(2026, 3, 31)),
    (date(2026, 9, 1), date(2026, 6, 30)),
    (date(2026, 11, 1), date(2026, 9, 30)),
])
def test_required_period_is_freshness_floor_not_publication(day, expected):
    assert required_report_period(day) == expected


def test_reporting_rule_points_to_stock_listing_rule_not_convertible_bond_rule():
    assert "/stocks/mainipo/" in REPORTING_WINDOW_SOURCE
    assert "/bond/convertible/" not in REPORTING_WINDOW_SOURCE


def test_financials_require_all_three_statement_families_and_prior_profit():
    gate, reasons, metrics, dates = check_financial()
    assert gate == "pass"
    assert metrics["operating_cash_flow_to_net_profit"] == 1.2
    assert "2026-08-29" in dates
    for key in ("balance", "cash_flow", "prior_year_profit"):
        data = financials()
        data[key] = []
        assert check_financial(data)[0] == "unknown"


@pytest.mark.parametrize(("family", "field", "value"), [
    ("profit", "netProfit", "-1"), ("profit", "roeAvg", "0"),
    ("cash_flow", "CFOToNP", "-0.5"), ("balance", "liabilityToAsset", "1.1"),
    ("balance", "assetToEquity", "-2"), ("prior_year_profit", "netProfit", "0"),
])
def test_financial_quality_vetoes_are_not_synthesized_passes(family, field, value):
    data = financials()
    row = data[family] if family == "profit" else data[family][0]
    row[field] = value
    assert check_financial(data)[0] == "veto"


@pytest.mark.parametrize("value", ["", "NaN", "Infinity", None, True])
def test_missing_and_nonfinite_are_not_zero(value):
    data = financials()
    data["profit"]["netProfit"] = value
    assert check_financial(data)[0] == "unknown"


def test_banks_do_not_inherit_industrial_cashflow_tests():
    gate, reasons, metrics, dates = check_financial(industry="银行")
    assert (gate, reasons) == ("unknown", ("FINANCIAL_SECTOR_SPECIFIC_REVIEW_REQUIRED",))
    assert metrics == {"net_profit": 100.0, "prior_year_same_period_net_profit": 80.0, "roe": 0.05}
    assert dates == ("2026-08-29", "2025-08-29")
    assert check_financial(industry="")[0] == "unknown"


def test_financial_sector_clear_profit_failure_vetoes_but_positive_profit_does_not_auto_pass():
    data = financials()
    data["cash_flow"] = []  # industrial CFO ratio is not applicable
    data["balance"] = []  # industrial liabilities/assets cutoff is not applicable
    assert check_financial(data, industry="保险")[0] == "unknown"
    data["profit"]["netProfit"] = "-1"
    assert check_financial(data, industry="保险")[0:2] == (
        "veto", ("CURRENT_OR_PRIOR_YEAR_PERIOD_NOT_PROFITABLE",)
    )
    data["profit"]["pubDate"] = "2026-09-23"
    assert check_financial(data, industry="保险")[0] == "unknown"


def test_future_publication_stale_period_and_identity_fail_closed():
    for field, value in (("pubDate", "2026-09-23"), ("code", "sh.600000"), ("statDate", "2026-03-31")):
        data = financials()
        data["profit"][field] = value
        assert check_financial(data)[0] == "unknown"
    data = financials()
    data["selected_period"] = "2026-03-31"
    assert check_financial(data)[0] == "unknown"


def test_date_only_publication_on_decision_day_cannot_pass_premarket():
    data = financials()
    data["profit"]["pubDate"] = "2026-09-22"
    assert check_financial(data)[0:2] == (
        "unknown", ("FINANCIAL_PUBLICATION_TIME_UNVERIFIED_ON_DECISION_DATE",)
    )


def test_complete_current_official_manifest_can_pass_narrow_auto_title_screen():
    assert evaluate_announcement_review(
        manifest(), symbol=SYMBOL, financial_hash="abc", knowledge_time=NOW,
        review_dir=None, financial_period="2026-06-30",
    ) == ("pass", ("AUTOMATIC_BASIC_OFFICIAL_TITLE_SCREEN_PASSED",))


@pytest.mark.parametrize(("mutation", "reason"), [
    ("short_window", "OFFICIAL_MANIFEST_WINDOW_INCOMPLETE"),
    ("count_mismatch", "OFFICIAL_MANIFEST_COUNT_UNVERIFIED"),
    ("total_mismatch", "OFFICIAL_MANIFEST_COUNT_UNVERIFIED"),
    ("wrong_source", "OFFICIAL_MANIFEST_SOURCE_UNVERIFIED"),
    ("wrong_issuer", "OFFICIAL_MANIFEST_ISSUER_MISMATCH"),
    ("missing_issuer", "OFFICIAL_MANIFEST_ISSUER_MISMATCH"),
    ("stale_same_day", "OFFICIAL_MANIFEST_STALE_OR_FUTURE"),
])
def test_auto_screen_requires_complete_current_official_listing(mutation, reason):
    data = manifest()
    if mutation == "short_window":
        data["coverage_from"] = "2026-01-01"
    elif mutation == "count_mismatch":
        data["record_count"] = 1
    elif mutation == "total_mismatch":
        data["total_provider_records"] = 3
    elif mutation == "wrong_source":
        data["source_url"] = "https://example.com/not-official"
    elif mutation == "wrong_issuer":
        data["symbol"] = "600000.SH"
    elif mutation == "missing_issuer":
        del data["symbol"]
    else:
        data["coverage_through"] = (NOW - timedelta(minutes=10)).isoformat()
    assert evaluate_announcement_review(
        data, symbol=SYMBOL, financial_hash="abc", knowledge_time=NOW,
        review_dir=None, financial_period="2026-06-30",
    ) == ("unknown", (reason,))


@pytest.mark.parametrize(("title", "expected_reason"), [
    ("2026年半年度报告摘要", "LATEST_FULL_OFFICIAL_FINANCIAL_REPORT_REQUIRED"),
    ("2026年半年度报告更正公告", "LATEST_FULL_OFFICIAL_FINANCIAL_REPORT_REQUIRED"),
    ("2026年半年度业绩预告", "LATEST_FULL_OFFICIAL_FINANCIAL_REPORT_REQUIRED"),
])
def test_summary_correction_or_forecast_cannot_replace_full_latest_report(title, expected_reason):
    data = manifest()
    data["items"][1]["title"] = title
    data["content_hash"] = content_hash(data["items"])
    assert evaluate_announcement_review(
        data, symbol=SYMBOL, financial_hash="abc", knowledge_time=NOW,
        review_dir=None, financial_period="2026-06-30",
    ) == ("unknown", (expected_reason,))


def test_title_screen_is_not_content_review_and_material_risk_remains_unknown():
    data = manifest()
    data["items"].append({
        "announcement_id": "risk", "title": "关于收到立案告知书的公告",
        "published_at": "2026-09-21T00:00:00+08:00",
        "url": "https://static.cninfo.com.cn/finalpage/risk.PDF",
    })
    data["record_count"] = data["total_provider_records"] = len(data["items"])
    data["content_hash"] = content_hash(data["items"])
    assert evaluate_announcement_review(
        data, symbol=SYMBOL, financial_hash="abc", knowledge_time=NOW,
        review_dir=None, financial_period="2026-06-30",
    ) == ("unknown", ("POTENTIALLY_MATERIAL_DOCUMENT_REQUIRES_CONTENT_REVIEW",))


def test_ordinary_audit_title_is_not_a_nonstandard_audit_opinion():
    data = manifest()
    data["items"].append({
        "announcement_id": "audit", "title": "2025年度审计报告",
        "published_at": "2026-03-20T00:01:00+08:00",
        "url": "https://static.cninfo.com.cn/finalpage/audit.PDF",
    })
    data["record_count"] = data["total_provider_records"] = len(data["items"])
    data["content_hash"] = content_hash(data["items"])
    assert evaluate_announcement_review(
        data, symbol=SYMBOL, financial_hash="abc", knowledge_time=NOW,
        review_dir=None, financial_period="2026-06-30",
    )[0] == "pass"
    data["items"][-1]["title"] = "2025年度非标准审计意见"
    data["content_hash"] = content_hash(data["items"])
    assert evaluate_announcement_review(
        data, symbol=SYMBOL, financial_hash="abc", knowledge_time=NOW,
        review_dir=None, financial_period="2026-06-30",
    )[0] == "unknown"


def test_combined_full_annual_and_audit_title_can_supply_annual_report_presence():
    data = manifest()
    data["items"][0]["title"] = "2025年度报告及审计报告"
    data["content_hash"] = content_hash(data["items"])
    assert evaluate_announcement_review(
        data, symbol=SYMBOL, financial_hash="abc", knowledge_time=NOW,
        review_dir=None, financial_period="2026-06-30",
    )[0] == "pass"


def test_newer_official_report_blocks_lagging_structured_financial_period():
    known = datetime(2026, 10, 25, 8, 40, tzinfo=ZoneInfo("Asia/Shanghai"))
    data = manifest()
    data["items"].append({
        "announcement_id": "q3", "title": "2026年第三季度报告",
        "published_at": "2026-10-23T17:00:00+08:00",
        "url": "https://static.cninfo.com.cn/finalpage/q3.PDF",
    })
    data["coverage_through"] = known.isoformat()
    data["content_hash"] = content_hash(data["items"])
    data["record_count"] = data["total_provider_records"] = len(data["items"])
    assert evaluate_announcement_review(
        data, symbol=SYMBOL, financial_hash="abc", knowledge_time=known,
        review_dir=None, financial_period="2026-06-30",
    ) == ("unknown", ("STRUCTURED_FINANCIAL_LAGS_LATEST_OFFICIAL_REPORT",))


def make_review(tmp_path, *, data=None):
    data = data or manifest()
    docs = []
    for item, role in zip(data["items"], ("annual_audit", "latest_financial_report"), strict=True):
        file = tmp_path / (item["announcement_id"] + ".pdf")
        binary = b"%PDF-1.7 verified fixture content"
        file.write_bytes(binary)
        docs.append({"announcement_id": item["announcement_id"], "url": item["url"], "sha256": hashlib.sha256(binary).hexdigest(), "file": file.name, "role": role, "findings": "Reviewed relevant report/audit sections; fixture only"})
    review = {
        "method_version": METHOD_VERSION, "symbol": SYMBOL,
        "manifest_hash": data["content_hash"], "financial_hash": content_hash(financials()),
        "reviewed_at": (NOW - timedelta(hours=1)).isoformat(), "reviewer": "test reviewer",
        "status": "pass", "reason": "Document-grounded review fixture, not a live opinion",
        "checklist": dict.fromkeys(_CHECKLIST, True), "documents": docs,
        "triage": [{"announcement_id": item["announcement_id"], "decision": "document_reviewed", "reason": "Read relevant sections"} for item in data["items"]],
    }
    (tmp_path / f"{SYMBOL}.json").write_text(json.dumps(review))
    return review


def review_gate(tmp_path, data=None, time=NOW):
    return evaluate_announcement_review(data or manifest(), symbol=SYMBOL, financial_hash=content_hash(financials()), knowledge_time=time, review_dir=tmp_path)


def test_content_grounded_review_passes_and_survives_unchanged_manifest_refresh(tmp_path):
    make_review(tmp_path)
    assert review_gate(tmp_path)[0] == "pass"
    next_manifest = manifest()
    next_manifest["coverage_through"] = (NOW + timedelta(days=3)).isoformat()
    assert review_gate(tmp_path, next_manifest, NOW + timedelta(days=3))[0] == "pass"


def test_new_announcement_or_financial_restatement_invalidates_review(tmp_path):
    review = make_review(tmp_path)
    data = manifest()
    data["items"][1]["title"] = "半年度报告更正"
    data["content_hash"] = content_hash(data["items"])
    assert review_gate(tmp_path, data)[0] == "unknown"
    review["financial_hash"] = "changed"
    (tmp_path / f"{SYMBOL}.json").write_text(json.dumps(review))
    assert review_gate(tmp_path)[0] == "unknown"


@pytest.mark.parametrize("mutation", ["path_escape", "hash_change", "missing_audit", "missing_triage", "incomplete_checklist", "future_review"])
def test_review_provenance_and_coverage_cannot_be_skipped(tmp_path, mutation):
    review = make_review(tmp_path)
    if mutation == "path_escape":
        review["documents"][0]["file"] = "../outside.pdf"
    elif mutation == "hash_change":
        review["documents"][0]["sha256"] = "0" * 64
    elif mutation == "missing_audit":
        review["documents"][0]["role"] = "latest_financial_report"
    elif mutation == "missing_triage":
        review["triage"].pop()
    elif mutation == "incomplete_checklist":
        review["checklist"]["nonrecurring_profit_reviewed"] = False
    else:
        review["reviewed_at"] = (NOW + timedelta(seconds=1)).isoformat()
    (tmp_path / f"{SYMBOL}.json").write_text(json.dumps(review))
    assert review_gate(tmp_path)[0] == "unknown"


def test_bounded_adapter_never_transmits_metadata_and_preserves_timeout_partial_results():
    def runner(command, **kwargs):
        request = json.loads(kwargs["input"])
        assert set(request) == {"symbols", "cutoff", "knowledge_time"}
        assert "cost" not in kwargs["input"]
        raise subprocess.TimeoutExpired(command, 2, output=(json.dumps(payload()) + "\n").encode())
    out = read_candidate_evidence([SYMBOL, "002292.SZ"], cutoff=CUTOFF, knowledge_time=NOW, timeout_seconds=2, runner=runner)
    assert out[SYMBOL]["financials"]["profit"]["netProfit"] == "100"
    assert out["002292.SZ"]["error"] == "PROVIDER_DEADLINE_EXCEEDED"


def test_historical_current_snapshot_is_not_networked_or_backfilled():
    def forbidden(*args, **kwargs):
        raise AssertionError("must not call provider")
    result = enrich_candidate_evidence({SYMBOL: {"industry": "港口"}}, symbols=[SYMBOL], cutoff=CUTOFF, knowledge_time=NOW, provider=forbidden, clock=lambda: NOW + timedelta(days=1))
    assert result.results[0].fundamental_reasons == ("CURRENT_SNAPSHOT_FORBIDDEN_IN_HISTORICAL_REPLAY",)


def test_enrichment_is_copied_cached_and_labels_narrow_auto_review(tmp_path):
    original = {SYMBOL: {"industry": "港口", "private_cost": 12}}
    seen = []
    def provider(symbols, **kwargs):
        seen.append((symbols, kwargs))
        return {SYMBOL: payload()}
    batch = enrich_candidate_evidence(original, symbols=[SYMBOL], cutoff=CUTOFF, knowledge_time=NOW, cache_dir=tmp_path, provider=provider, clock=lambda: NOW)
    assert original == {SYMBOL: {"industry": "港口", "private_cost": 12}}
    assert batch.metadata[SYMBOL]["fundamental_gate"] == "pass"
    assert batch.metadata[SYMBOL]["announcement_gate"] == "pass"
    assert batch.metadata[SYMBOL]["candidate_evidence_announcement_review_mode"] == "automatic_basic_title_screen"
    assert batch.results[0].announcement_review_mode == "automatic_basic_title_screen"
    assert batch.diagnostics["automatic_basic_announcement_pass_count"] == 1
    assert batch.diagnostics["manual_content_review_pass_count"] == 0
    assert batch.diagnostics["fundamental_risk_review_pass_count"] == 1
    assert batch.diagnostics["unknown_count"] == 0
    again = enrich_candidate_evidence(original, symbols=[SYMBOL], cutoff=CUTOFF, knowledge_time=NOW, cache_dir=tmp_path, provider=provider, clock=lambda: NOW)
    assert len(seen) == 1
    assert again.results[0].financial_hash == batch.results[0].financial_hash
    assert "private_cost" not in (tmp_path / f"{SYMBOL}.json").read_text()


def test_same_day_cached_official_manifest_is_refreshed_at_new_decision_time(tmp_path):
    (tmp_path / f"{SYMBOL}.json").write_text(json.dumps(payload()))
    later = NOW + timedelta(minutes=10)
    calls = []
    def provider(symbols, **kwargs):
        calls.append(symbols)
        fresh = payload()
        fresh["retrieved_at"] = later.isoformat()
        fresh["announcements"]["coverage_through"] = later.isoformat()
        return {SYMBOL: fresh}
    batch = enrich_candidate_evidence(
        {SYMBOL: {"industry": "港口", "is_limit_up_at_cutoff": False}},
        symbols=[SYMBOL], cutoff=CUTOFF, knowledge_time=later,
        cache_dir=tmp_path, provider=provider, clock=lambda: later,
    )
    assert calls == [[SYMBOL]]
    assert batch.results[0].announcement_gate == "pass"


def test_provider_failure_is_unknown_not_no_eligible_and_explicit_veto_sticks():
    batch = enrich_candidate_evidence({SYMBOL: {"industry": "港口", "fundamental_gate": "veto"}}, symbols=[SYMBOL], cutoff=CUTOFF, knowledge_time=NOW, provider=lambda *a, **k: {}, clock=lambda: NOW)
    assert batch.results[0].fundamental_gate == "unknown"
    assert batch.metadata[SYMBOL]["fundamental_gate"] == "veto"
    assert batch.diagnostics["unknown_count"] == 1


class FakeResponse:
    def __init__(self, data):
        self.data = data
        self.content = json.dumps(data).encode()
    def raise_for_status(self):
        pass
    def json(self):
        return self.data


def announcement(ident="a", code="601298"):
    return {"announcementId": ident, "secCode": code, "announcementTime": int((NOW - timedelta(days=1)).timestamp() * 1000), "adjunctUrl": f"finalpage/2026-09-21/{ident}.PDF", "announcementTitle": "关于经营业绩的公告"}


def test_cninfo_pagination_must_be_complete_and_issuer_exact():
    class Client:
        def __init__(self, pages):
            self.pages = iter(pages)
        def post(self, *args, **kwargs):
            return FakeResponse(next(self.pages))
    rows = [{"totalAnnouncement": 2, "announcements": [announcement()], "hasMore": True}, {"totalAnnouncement": 2, "announcements": [announcement("b")], "hasMore": False}]
    data = collect_announcements(Client(rows), SYMBOL, NOW, {"601298": "issuer"})
    assert data["complete"] is True
    assert data["record_count"] == 2
    bad = [{"totalAnnouncement": 1, "announcements": [announcement(code="600000")], "hasMore": False}]
    assert collect_announcements(Client(bad), SYMBOL, NOW, {"601298": "issuer"})["error"] == "OFFICIAL_MANIFEST_IDENTITY_OR_PAGINATION_INVALID"
    truncated = [{"totalAnnouncement": 2, "announcements": [announcement()], "hasMore": False}]
    assert collect_announcements(Client(truncated), SYMBOL, NOW, {"601298": "issuer"})["error"] == "OFFICIAL_MANIFEST_TRUNCATED"


def test_future_same_day_announcement_not_used():
    row = announcement()
    row["announcementTime"] = int((NOW + timedelta(hours=2)).timestamp() * 1000)
    client = SimpleNamespace(post=lambda *a, **k: FakeResponse({"totalAnnouncement": 1, "announcements": [row], "hasMore": False}))
    result = collect_announcements(client, SYMBOL, NOW, {"601298": "issuer"})
    assert result["complete"] is True
    assert result["items"] == []


def test_adapter_rejects_private_or_invalid_symbols_before_process():
    with pytest.raises(ValueError):
        read_candidate_evidence(["601298.SH cost=100"], cutoff=CUTOFF, knowledge_time=NOW)
    with pytest.raises(ValueError):
        read_candidate_evidence([SYMBOL] * 37, cutoff=CUTOFF, knowledge_time=NOW)


class FakeRows:
    error_code = "0"
    def __init__(self, rows):
        self.fields = list(rows[0]) if rows else []
        self.rows = iter(rows)
        self.row = None
    def next(self):
        self.row = next(self.rows, None)
        return self.row is not None
    def get_row_data(self):
        return [self.row[key] for key in self.fields]


def test_financial_collector_probes_newer_period_not_only_minimum():
    data = financials()
    class Module:
        def query_profit_data(self, **kwargs):
            row = data["profit"] if kwargs["year"] == 2026 else data["prior_year_profit"][0]
            return FakeRows([row] if kwargs["quarter"] == 2 else [])
        def query_balance_data(self, **kwargs):
            return FakeRows(data["balance"])
        def query_cash_flow_data(self, **kwargs):
            return FakeRows(data["cash_flow"])
    assert collect_financials(Module(), SYMBOL, NOW)["selected_period"] == "2026-06-30"


def test_execution_receipt_collects_dated_valuation_context_without_using_it_as_a_gate():
    requested = []
    class Module:
        def query_history_k_data_plus(self, symbol, fields, **kwargs):
            requested.append((symbol, fields, kwargs))
            return FakeRows(execution()["rows"])
    result = collect_execution_status(Module(), SYMBOL, CUTOFF)
    assert requested[0][0] == "sh.601298"
    assert requested[0][1].endswith("peTTM,pbMRQ")
    assert result["rows"][0]["peTTM"] == "12.5"
    assert result["cutoff"] == CUTOFF.isoformat()


def test_review_hash_is_based_on_financial_content_not_fetch_time():
    one = payload()
    two = copy.deepcopy(one)
    two["retrieved_at"] = (NOW + timedelta(hours=1)).isoformat()
    assert content_hash(one["financials"]) == content_hash(two["financials"])


@pytest.mark.parametrize(("status", "st", "limit", "expected"), [
    ("1", "0", False, "pass"), ("0", "0", False, "veto"),
    ("1", "1", False, "veto"), ("1", "0", True, "veto"),
    ("1", "0", None, "unknown"), ("?", "0", False, "unknown"),
])
def test_formation_execution_requires_independent_limit_status(status, st, limit, expected):
    data = execution()
    data["rows"][0].update(tradestatus=status, isST=st)
    data["content_hash"] = content_hash(data["rows"])
    assert evaluate_execution_status(data, symbol=SYMBOL, cutoff=CUTOFF, metadata={"is_limit_up_at_cutoff": limit})[0] == expected


def test_execution_cannot_reuse_wrong_date_or_duplicate_or_mismatched_symbol():
    for mutation in ("date", "code", "duplicate"):
        data = execution()
        if mutation == "date":
            data["rows"][0]["date"] = "2026-09-18"
        elif mutation == "code":
            data["rows"][0]["code"] = "sh.600000"
        else:
            data["rows"].append(data["rows"][0].copy())
        data["content_hash"] = content_hash(data["rows"])
        assert evaluate_execution_status(data, symbol=SYMBOL, cutoff=CUTOFF, metadata={"is_limit_up_at_cutoff": False})[0] == "unknown"


def test_evidence_fills_only_verified_formation_execution_metadata():
    batch = enrich_candidate_evidence({SYMBOL: {"industry": "港口", "is_limit_up_at_cutoff": False}}, symbols=[SYMBOL], cutoff=CUTOFF, knowledge_time=NOW, provider=lambda *a, **k: {SYMBOL: payload()}, clock=lambda: NOW)
    assert batch.metadata[SYMBOL]["is_buyable_at_cutoff"] is True
    assert batch.results[0].execution_gate == "pass"
    assert batch.results[0].execution_hash == execution()["content_hash"]
    assert batch.diagnostics["execution_pass_count"] == 1
    assert batch.results[0].required_financial_period == "2026-06-30"
    assert batch.results[0].latest_official_financial_period == "2026-06-30"
    assert batch.results[0].valuation_asof == CUTOFF.isoformat()
    assert batch.results[0].metrics["pe_ttm_at_price_cutoff"] == 12.5
    assert batch.results[0].metrics["pb_mrq_at_price_cutoff"] == 1.3
    assert batch.metadata[SYMBOL]["candidate_evidence_valuation_asof"] == CUTOFF.isoformat()
    assert batch.diagnostics["required_financial_period"] == "2026-06-30"


def test_missing_or_bad_valuation_does_not_become_a_quality_pass():
    data = payload()
    data["execution"]["rows"][0]["peTTM"] = "NaN"
    data["execution"]["content_hash"] = content_hash(data["execution"]["rows"])
    batch = enrich_candidate_evidence(
        {SYMBOL: {"industry": "银行", "is_limit_up_at_cutoff": False}},
        symbols=[SYMBOL], cutoff=CUTOFF, knowledge_time=NOW,
        provider=lambda *a, **k: {SYMBOL: data}, clock=lambda: NOW,
    )
    assert batch.results[0].execution_gate == "pass"
    assert batch.results[0].fundamental_gate == "unknown"
    assert "pe_ttm_at_price_cutoff" not in batch.results[0].metrics
    assert batch.diagnostics["financial_sector_special_review_count"] == 1


def test_malformed_nested_provider_payload_does_not_crash_report():
    data = payload()
    data["financials"] = "bad"
    data["announcements"] = ["bad"]
    data["execution"] = None
    batch = enrich_candidate_evidence({SYMBOL: {"industry": "港口"}}, symbols=[SYMBOL], cutoff=CUTOFF, knowledge_time=NOW, provider=lambda *a, **k: {SYMBOL: data}, clock=lambda: NOW)
    assert batch.results[0].fundamental_gate == "unknown"
    assert batch.results[0].announcement_gate == "unknown"


def test_material_announcement_cannot_be_triaged_away_as_routine(tmp_path):
    data = manifest()
    data["items"].append({"announcement_id": "risk", "title": "关于收到立案告知书的公告", "published_at": "2026-09-21T00:00:00+08:00", "url": "https://static.cninfo.com.cn/finalpage/risk.PDF"})
    data["content_hash"] = content_hash(data["items"])
    data["record_count"] = data["total_provider_records"] = len(data["items"])
    review = make_review(tmp_path)
    review["manifest_hash"] = data["content_hash"]
    review["triage"].append({"announcement_id": "risk", "decision": "routine_nonmaterial", "reason": "Not important"})
    (tmp_path / f"{SYMBOL}.json").write_text(json.dumps(review))
    assert review_gate(tmp_path, data)[1] == ("POTENTIALLY_MATERIAL_DOCUMENT_REQUIRES_CONTENT_REVIEW",)


def test_draft_review_path_inside_root_supported_and_escape_rejected(tmp_path):
    make_review(tmp_path)
    draft = tmp_path / "drafts" / "draft.json"
    draft.parent.mkdir()
    (tmp_path / f"{SYMBOL}.json").rename(draft)
    kwargs = {"symbol": SYMBOL, "financial_hash": content_hash(financials()), "knowledge_time": NOW, "review_dir": tmp_path}
    assert evaluate_announcement_review(manifest(), review_path=draft, **kwargs)[0] == "pass"
    assert evaluate_announcement_review(manifest(), review_path=tmp_path.parent / "outside.json", **kwargs)[1] == ("OFFICIAL_REVIEW_PATH_OUTSIDE_REVIEW_DIRECTORY",)


def test_failed_provider_cache_does_not_suppress_same_day_retry(tmp_path):
    data = payload()
    data["announcements"] = {"error": "OFFICIAL_MANIFEST_PENDING"}
    (tmp_path / f"{SYMBOL}.json").write_text(json.dumps(data))
    calls = []
    def provider(symbols, **kwargs):
        calls.append(symbols)
        return {SYMBOL: payload()}
    result = enrich_candidate_evidence({SYMBOL: {"industry": "港口"}}, symbols=[SYMBOL], cutoff=CUTOFF, knowledge_time=NOW, cache_dir=tmp_path, provider=provider, clock=lambda: NOW)
    assert calls == [[SYMBOL]]
    assert result.results[0].announcement_manifest_hash == manifest()["content_hash"]


def test_worker_partial_is_replaced_by_complete_receipt():
    partial = payload()
    partial["announcements"] = {"error": "OFFICIAL_MANIFEST_PENDING"}
    completed = payload()
    def runner(*a, **k):
        return SimpleNamespace(returncode=0, stdout=json.dumps(partial) + "\n" + json.dumps(completed) + "\n")
    assert read_candidate_evidence([SYMBOL], cutoff=CUTOFF, knowledge_time=NOW, runner=runner)[SYMBOL]["announcements"]["complete"] is True


def test_request_date_boundary_uses_shanghai_not_utc():
    time = datetime(2026, 9, 22, 1, tzinfo=ZoneInfo("Asia/Shanghai")).astimezone(ZoneInfo("UTC"))
    def runner(*a, **k):
        return SimpleNamespace(returncode=0, stdout="")
    assert read_candidate_evidence([SYMBOL], cutoff=date(2026, 9, 22), knowledge_time=time, runner=runner)[SYMBOL]["error"] == "PROVIDER_RESULT_MISSING"


def test_same_day_cutoff_advance_refreshes_execution_instead_of_reusing_cache(tmp_path):
    (tmp_path / f"{SYMBOL}.json").write_text(json.dumps(payload()))
    calls = []
    new_cutoff = date(2026, 9, 22)
    later = NOW + timedelta(hours=8)
    def provider(symbols, **kwargs):
        calls.append(kwargs["cutoff"])
        data = payload()
        data["retrieved_at"] = later.isoformat()
        data["execution"]["cutoff"] = new_cutoff.isoformat()
        data["execution"]["rows"][0]["date"] = new_cutoff.isoformat()
        data["execution"]["content_hash"] = content_hash(data["execution"]["rows"])
        return {SYMBOL: data}
    batch = enrich_candidate_evidence({SYMBOL: {"industry": "港口", "is_limit_up_at_cutoff": False}}, symbols=[SYMBOL], cutoff=new_cutoff, knowledge_time=later, cache_dir=tmp_path, provider=provider, clock=lambda: later)
    assert calls == [new_cutoff]
    assert batch.results[0].execution_gate == "pass"


def test_annual_window_in_april_keeps_prior_march_published_latest_annual():
    start = announcement_window_start(date(2027, 4, 20))
    assert start == date(2026, 1, 1)
    assert start <= date(2026, 3, 20)
    assert announcement_window_start(date(2027, 5, 1)) == date(2026, 5, 1)


def test_early_new_annual_report_requires_its_own_audit_not_statutory_old_floor(tmp_path):
    make_review(tmp_path)
    known = datetime(2027, 2, 20, 9, tzinfo=ZoneInfo("Asia/Shanghai"))
    data = manifest()
    data["coverage_through"] = known.isoformat()
    gate, reasons = evaluate_announcement_review(data, symbol=SYMBOL, financial_hash=content_hash(financials()), knowledge_time=known, review_dir=tmp_path, financial_period="2026-12-31")
    assert gate == "unknown"
    assert reasons == ("OFFICIAL_REVIEW_LATEST_ANNUAL_AUDIT_REQUIRED",)


def test_worker_completes_first_candidate_before_starting_next(monkeypatch):
    import ashare_lab.adapters.candidate_financial_evidence as adapter
    events, receipts = [], []
    def financial(module, symbol, time):
        events.append(("financial", symbol))
        return financials()
    def status(module, symbol, cutoff):
        events.append(("execution", symbol))
        return execution()
    def announcements(client, symbol, time, identities):
        events.append(("announcements", symbol))
        if symbol == SYMBOL:
            return manifest()
        raise TimeoutError("slow official source")
    monkeypatch.setattr(adapter, "collect_financials", financial)
    monkeypatch.setattr(adapter, "collect_execution_status", status)
    monkeypatch.setattr(adapter, "collect_announcements", announcements)
    client = SimpleNamespace(get=lambda *a: FakeResponse({"stockList": []}))
    collect_worker_batch([SYMBOL, "002292.SZ"], cutoff=CUTOFF, knowledge_time=NOW, module=object(), client=client, emit=lambda r: receipts.append(copy.deepcopy(r)))
    assert events[:4] == [
        ("financial", SYMBOL), ("execution", SYMBOL),
        ("announcements", SYMBOL), ("financial", "002292.SZ"),
    ]
    assert any(
        r["symbol"] == SYMBOL and r.get("announcements", {}).get("complete") is True
        for r in receipts
    )
    assert any(r["symbol"] == "002292.SZ" and r.get("financials", {}).get("profit") and not r["announcement_attempted"] for r in receipts)


def test_partial_financial_receipt_is_reused_without_transmitting_financial_values():
    old = payload()
    old["announcements"] = {"error": "OFFICIAL_MANIFEST_PENDING"}
    def runner(command, **kwargs):
        request = json.loads(kwargs["input"])
        assert request["skip_financial_symbols"] == [SYMBOL]
        assert "netProfit" not in kwargs["input"]
        new = {"symbol": SYMBOL, "method_version": METHOD_VERSION, "retrieved_at": NOW.isoformat(), "announcements": manifest()}
        return SimpleNamespace(returncode=0, stdout=json.dumps(new) + "\n")
    result = read_candidate_evidence([SYMBOL], cutoff=CUTOFF, knowledge_time=NOW, reuse_receipts={SYMBOL: old}, runner=runner)[SYMBOL]
    assert result["financials"] == old["financials"]
    assert result["financial_retrieved_at"] == old["retrieved_at"]
    assert result["announcements"]["complete"] is True


def test_same_failed_financial_candidate_does_not_always_run_first(tmp_path):
    slow = payload()
    slow["financials"] = {"error": "FINANCIAL_PROVIDER_PENDING"}
    slow["_financial_attempts"] = 1
    (tmp_path / f"{SYMBOL}.json").write_text(json.dumps(slow))
    observed = []
    def provider(symbols, **kwargs):
        observed.append(symbols)
        return {}
    enrich_candidate_evidence({SYMBOL: {"industry": "港口"}, "002292.SZ": {"industry": "传媒"}}, symbols=[SYMBOL, "002292.SZ"], cutoff=CUTOFF, knowledge_time=NOW, cache_dir=tmp_path, provider=provider, clock=lambda: NOW)
    assert observed == [["002292.SZ", SYMBOL]]


def test_slow_announcement_retries_rotate_and_successful_financials_reuse(tmp_path, monkeypatch):
    import ashare_lab.services.candidate_evidence as service
    first = payload()
    first["announcements"] = {"error": "OFFICIAL_MANIFEST_PENDING"}
    first["_announcement_attempts"] = 1
    second = copy.deepcopy(first)
    second["symbol"] = "002292.SZ"
    second["_announcement_attempts"] = 0
    (tmp_path / f"{SYMBOL}.json").write_text(json.dumps(first))
    (tmp_path / "002292.SZ.json").write_text(json.dumps(second))
    calls = []
    def reader(symbols, **kwargs):
        calls.append((symbols, kwargs["reuse_receipts"]))
        return {}
    monkeypatch.setattr(service, "read_candidate_evidence", reader)
    enrich_candidate_evidence({SYMBOL: {"industry": "港口"}, "002292.SZ": {"industry": "传媒"}}, symbols=[SYMBOL, "002292.SZ"], cutoff=CUTOFF, knowledge_time=NOW, cache_dir=tmp_path, clock=lambda: NOW)
    assert calls[0][0] == ["002292.SZ", SYMBOL]
    assert set(calls[0][1]) == {SYMBOL, "002292.SZ"}
