"""Synthetic documents exercise risk semantics, not investment validity."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import timedelta

import pytest
from test_candidate_evidence import CUTOFF, NOW, SYMBOL, manifest, payload

from ashare_lab.adapters.candidate_financial_evidence import content_hash
from ashare_lab.services.candidate_evidence import (
    enrich_candidate_evidence,
    evaluate_announcement_review,
)
from ashare_lab.services.material_risk_review import (
    classify_document_title,
    full_report_period,
    required_document_items,
)


def evaluate(data, **kwargs):
    return evaluate_announcement_review(
        data, symbol=SYMBOL, financial_hash="fixture", knowledge_time=NOW,
        review_dir=None, financial_period="2026-06-30", **kwargs,
    )


def add_document(data, title, text=None):
    item = {
        "announcement_id": "new", "title": title,
        "published_at": (NOW - timedelta(days=1)).isoformat(),
        "url": "https://static.cninfo.com.cn/finalpage/new.PDF",
    }
    data["items"].append(item)
    data["record_count"] = data["total_provider_records"] = len(data["items"])
    data["content_hash"] = content_hash(data["items"])
    if text is not None:
        text += " 本文件由公司发布，所述数据及事项以本次公开披露为准，详细披露依据及背景载于本报告其他章节。" * 2
        data["documents"].append({
            **item, "symbol": SYMBOL, "retrieved_at": NOW.isoformat(),
            "sha256": hashlib.sha256(b"%PDF-risk-fixture").hexdigest(),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text,
            "page_count": 3, "extraction_status": "complete", "extraction_complete": True,
        })
    return data


def test_manifest_without_pdf_contents_never_passes():
    data = manifest()
    del data["documents"]
    assert evaluate(data) == ("unknown", ("TARGETED_OFFICIAL_DOCUMENTS_REQUIRED",))


@pytest.mark.parametrize("title", ["关于修订公司章程的公告", "关联交易管理制度（修订）", "对子公司担保管理办法", "防范控股股东及关联方资金占用制度"])
def test_routine_governance_is_not_an_all_item_approval_queue(title):
    data = add_document(manifest(), title)
    assert classify_document_title(title) == "routine_governance"
    assert len(required_document_items(data["items"], "2026-06-30", NOW)) == 2
    assert evaluate(data)[0] == "pass"


@pytest.mark.parametrize(("title", "text"), [
    ("关于为控股子公司提供担保的公告", "公司在已批准的担保额度范围内为控股子公司提供担保5000万元，公司不存在逾期担保。累计担保总额占公司净资产的4.24%。"),
    ("2026年度日常关联交易预计公告", "本次日常关联交易金额5000万元，定价公允，以市场价格为依据，不会损害股东利益。已履行决策程序，董事会审议通过。"),
])
def test_routine_transaction_requires_positive_body_evidence_not_just_name(title, text):
    data = add_document(manifest(), title)
    assert evaluate(data)[0] == "unknown"
    data = add_document(manifest(), title, text)
    assert evaluate(data)[0] == "pass"


@pytest.mark.parametrize(("title", "text", "reason"), [
    ("关于为控股子公司提供担保的公告", "本公司收到中国证监会立案告知书，因公司涉嫌财务造假。", "OFFICIAL_DISCLOSURE_INVESTIGATION_OR_FRAUD"),
    ("重大债务事项公告", "本公司未能按期偿还到期债务，流动资金严重不足。", "OFFICIAL_DISCLOSURE_DEBT_DEFAULT"),
    ("风险警示公告", "公司股票被实施退市风险警示。", "OFFICIAL_DISCLOSURE_DELISTING_OR_INSOLVENCY"),
    ("资产冻结事项公告", "公司主要资产被司法冻结，经营受到重大影响。", "OFFICIAL_DISCLOSURE_MAJOR_ASSET_FREEZE"),
    ("审计意见公告", "审计师已对本公司年度财务报告出具保留意见。", "OFFICIAL_DISCLOSURE_NONSTANDARD_AUDIT"),
])
def test_verified_material_fact_vetoes_even_with_ordinary_title(title, text, reason):
    data = add_document(manifest(), title, text)
    gate, reasons = evaluate(data)
    assert gate == "veto"
    assert reason in reasons


def test_unresolved_nonstandard_or_ambiguous_risk_is_unknown_not_pass():
    data = add_document(manifest(), "关于诉讼事项的公告", "公司涉及若干诉讼，其金额、进展以及经营影响需要进一步核对。")
    assert evaluate(data) == ("unknown", ("MATERIAL_FACT_CONTENT_NEEDS_REVIEW",))


@pytest.mark.parametrize("text", [
    "如果公司未能按期偿还债务，将会影响未来融资，此为风险因素假设。",
    "公司不存在债务违约，未发生重大资产冻结，公司未收到立案告知书。",
    "公司曾未能按期偿还贷款，该款项目前已全部结清，不再存在逾期。",
    "本公司曾于2019年被立案调查，2020年结案。",
])
def test_hypothetical_negated_and_resolved_risks_are_not_confirmed_vetoes(text):
    data = add_document(manifest(), "关于债务事项的说明", text)
    assert evaluate(data)[0] == "unknown"


def test_routine_title_does_not_waive_material_dependency_or_missing_guarantee_context():
    data = add_document(manifest(), "日常关联交易公告", "交易金额5000万元，定价公允，不会损害股东利益。董事会审议通过。但公司对关联方形成较大依赖。")
    assert evaluate(data)[0] == "unknown"


@pytest.mark.parametrize("title", [
    "关于修订关联交易管理制度及新增关联交易的公告",
    "关于担保管理办法及新增对外担保事项的公告",
    "关于修改公司章程并出售重大资产的公告",
])
def test_mixed_policy_and_substantive_transaction_titles_require_body(title):
    assert evaluate(add_document(manifest(), title))[0] == "unknown"


@pytest.mark.parametrize("text", [
    "本公司未能按期\n偿还到期债务。",
    "本公司不存在逾期担保，但本公司未能按期偿还到期债务。",
])
def test_soft_line_wrap_and_unrelated_negation_do_not_hide_default(text):
    assert evaluate(add_document(manifest(), "关于债务事项的公告", text))[0] == "veto"


@pytest.mark.parametrize("text", [
    "上述事项表明存在可能导致对本公司持续经营能力产生重大疑虑的重大不确定性。",
    "本公司主要银行账户被冻结，经营受到重大影响。",
    "本公司未能偿还到期债务。",
    "本公司控股股东已违规占用公司资金100亿元。",
])
def test_explicit_material_fact_variants_in_reports_cannot_pass(text):
    data = manifest()
    doc = data["documents"][0]
    doc["text"] += text
    doc["text_sha256"] = hashlib.sha256(doc["text"].encode()).hexdigest()
    assert evaluate(data)[0] in {"unknown", "veto"}


def test_unchecked_annual_report_template_is_not_a_delisting_fact():
    data = manifest()
    doc = data["documents"][0]
    # The literal filing template, with unrelated previous section too, is not
    # an affirmative report that the issuer has been risk-warned.
    doc["text"] += "C.报告期内公司股票被实施退市风险警示或其他风险警示\n前五名销售客户□适用√不适用\n前五名供应商□适用√不适用。"
    doc["text_sha256"] = hashlib.sha256(doc["text"].encode()).hexdigest()
    assert evaluate(data)[0] == "pass"


def test_extreme_guarantee_ratio_never_counts_as_routine_clearance():
    text = "本公司在已批准的担保额度范围内为控股子公司提供担保50000000万元，公司不存在逾期担保。本次对外担保总额超过公司最近一期经审计净资产的500%，可能产生重大偿债风险。"
    assert evaluate(add_document(manifest(), "关于控股子公司担保的公告", text))[0] == "unknown"
    data = add_document(manifest(), "控股子公司担保公告", "已批准担保额度5000万元，控股子公司不存在逾期担保。未披露累计担保占净资产比例。")
    assert evaluate(data)[0] == "unknown"


@pytest.mark.parametrize("mutation", ["missing", "hash", "issuer", "future", "partial", "empty"])
def test_document_evidence_cannot_be_faked_or_reused_from_other_issuer(mutation):
    data = manifest()
    doc = data["documents"][0]
    if mutation == "missing":
        data["documents"].pop()
    elif mutation == "hash":
        doc["text"] += "changed"
    elif mutation == "issuer":
        doc["symbol"] = "600000.SH"
    elif mutation == "future":
        doc["retrieved_at"] = (NOW + timedelta(days=1)).isoformat()
    elif mutation == "partial":
        doc["extraction_complete"] = False
    else:
        doc["text"] = ""
    assert evaluate(data)[0] == "unknown"


def test_real_fetch_after_requested_boundary_uses_verified_receipt_time():
    data = manifest()
    later = NOW + timedelta(seconds=10)
    data["documents"][0]["retrieved_at"] = later.isoformat()
    assert evaluate(data)[0] == "unknown"
    assert evaluate(data, document_retrieval_time=later)[0] == "pass"
    data["items"][0]["published_at"] = (NOW + timedelta(seconds=1)).isoformat()
    data["content_hash"] = content_hash(data["items"])
    assert evaluate(data, document_retrieval_time=later)[0] == "unknown"


def test_latest_audit_content_is_required_even_if_title_looks_normal():
    data = manifest()
    text = data["documents"][0]["text"].replace("审计机构已出具标准无保留意见的审计报告。", "该年报未明确说明审计结论。")
    data["documents"][0].update(text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest())
    assert evaluate(data) == ("unknown", ("LATEST_ANNUAL_AUDIT_OPINION_UNCONFIRMED",))


def test_one_candidate_financial_veto_is_not_double_counted_as_unknown():
    value = payload()
    value["financials"]["profit"]["netProfit"] = "-1"
    value["announcements"].pop("documents")
    batch = enrich_candidate_evidence(
        {SYMBOL: {"industry": "港口"}}, symbols=[SYMBOL], cutoff=CUTOFF,
        knowledge_time=NOW, provider=lambda *a, **k: {SYMBOL: value}, clock=lambda: NOW,
    )
    assert batch.results[0].fundamental_risk_gate == "veto"
    assert batch.diagnostics["veto_count"] == 1
    assert batch.diagnostics["unknown_count"] == 0


def test_old_method_receipt_cannot_be_promoted_to_combined_pass():
    value = payload()
    value["method_version"] = "candidate-evidence-v1.2.0"
    batch = enrich_candidate_evidence(
        {SYMBOL: {"industry": "港口"}}, symbols=[SYMBOL], cutoff=CUTOFF,
        knowledge_time=NOW, provider=lambda *a, **k: {SYMBOL: value}, clock=lambda: NOW,
    )
    assert batch.results[0].fundamental_risk_gate == "unknown"


def test_stale_manual_pass_does_not_block_new_independent_targeted_review(tmp_path):
    review = {
        "method_version": "candidate-evidence-v1.2.0", "symbol": SYMBOL, "status": "pass",
        "manifest_hash": "old", "financial_hash": "old",
    }
    (tmp_path / f"{SYMBOL}.json").write_text(json.dumps(review))
    args = {"symbol": SYMBOL, "financial_hash": "new", "knowledge_time": NOW, "review_dir": tmp_path, "financial_period": "2026-06-30"}
    assert evaluate_announcement_review(manifest(), **args)[0] == "pass"
    review["status"] = "veto"
    (tmp_path / f"{SYMBOL}.json").write_text(json.dumps(review))
    assert evaluate_announcement_review(manifest(), **args)[0] == "unknown"


def test_bad_first_issuer_does_not_prevent_independent_second_issuer_pass():
    bad = payload()
    bad["announcements"]["items"] = 3
    other = "002292.SZ"
    good = copy.deepcopy(payload())
    # The second receipt's identity must be consistent in every evidence layer.
    serialized = json.dumps(good).replace(SYMBOL, other).replace("sh.601298", "sz.002292")
    good = json.loads(serialized)
    batch = enrich_candidate_evidence(
        {SYMBOL: {"industry": "港口"}, other: {"industry": "传媒"}},
        symbols=[SYMBOL, other], cutoff=CUTOFF, knowledge_time=NOW,
        provider=lambda *a, **k: {SYMBOL: bad, other: good}, clock=lambda: NOW,
    )
    assert [r.fundamental_risk_gate for r in batch.results] == ["unknown", "pass"]


def test_full_report_recognition_handles_year_suffix_without_accepting_summary():
    assert full_report_period("泉阳泉2026年半年度报告正文").isoformat() == "2026-06-30"
    assert full_report_period("2026年半年度报告摘要") is None
    assert full_report_period("2026年半年度报告更正公告") is None
