"""Synthetic official-receipt fixtures; no investment/real-issuer validation."""

from __future__ import annotations

import copy
import hashlib
from datetime import UTC, datetime

import pytest

from ashare_lab.services.sector_financial_review import evaluate_sector_financial_risk

SYMBOL = "601328.SH"
PERIOD = "2026-06-30"
KNOWN = datetime(2026, 10, 11, 8, tzinfo=UTC)
BANK = (
    "截至2026年6月30日，本集团核心一级资本充足率为12.50%，"
    "一级资本充足率为14.00%，资本充足率为16.00%，均满足监管要求。"
    "报告期末，本集团不良贷款率为1.20%，拨备覆盖率为210.00%。"
)
INSURANCE = (
    "截至2026年6月30日，本公司核心偿付能力充足率为150.00%，"
    "综合偿付能力充足率为250.00%，偿付能力风险综合评级为A类。"
)


def receipt(text=BANK, *, title="测试机构2026年半年度报告"):
    item = {
        "announcement_id": "test-latest-report",
        "title": title,
        "url": "https://static.cninfo.com.cn/finalpage/2026-08-28/test.pdf",
        "published_at": "2026-08-28T08:00:00+08:00",
    }
    return {
        "items": [item],
        "documents": [{
            **item, "symbol": SYMBOL,
            "retrieved_at": "2026-10-11T07:50:00+00:00",
            "sha256": "a" * 64,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text": text, "extraction_complete": True,
            "extraction_status": "complete", "page_count": 100,
        }],
    }


def run(data=None, *, industry="银行", period=PERIOD):
    return evaluate_sector_financial_risk(
        receipt() if data is None else data, industry=industry,
        symbol=SYMBOL, period=period, knowledge_time=KNOWN,
    )


def test_bank_has_reachable_pass_using_current_group_official_report():
    gate, reasons, metrics = run()
    assert gate == "pass"
    assert reasons == ("BANK_OFFICIAL_CAPITAL_AND_ASSET_QUALITY_BASELINE_CONFIRMED",)
    assert metrics["core_tier1_capital_ratio"] == .125
    assert metrics["nonperforming_loan_ratio"] == .012
    assert metrics["provision_coverage_ratio"] == 2.1


def test_explicit_period_end_group_subject_does_not_require_optional_prefix():
    text = BANK.replace("报告期末，本集团不良", "报告期末，集团不良")
    assert run(receipt(text))[0] == "pass"


def test_positive_profit_and_roe_cannot_replace_specialist_evidence():
    assert run(receipt("报告期末，本集团净利润1000亿元，ROE10%。"))[0] == "unknown"


def test_bank_parallel_metric_order_is_explicit_not_sorted_by_magnitude():
    text = (
        "报告期末，本集团核心一级资本充足率、一级资本充足率和资本充足率"
        "分别为12.50%、14.00%和16.00%，均符合监管要求。"
        "报告期末，本集团不良贷款率1.20%，拨备覆盖率210.00%。"
    )
    assert run(receipt(text))[0] == "pass"
    assert run(receipt(text.replace("12.50%、14.00%", "15.00%、14.00%")))[0] == "unknown"


@pytest.mark.parametrize("source,replacement", [
    ("12.50%", "7.49%"), ("14.00%", "8.49%"),
    ("16.00%", "10.49%"),
])
def test_bank_regulatory_floor_breaches_veto(source, replacement):
    # Keep capital hierarchy internally consistent while testing each floor.
    text = BANK.replace("12.50%", "7.50%").replace("14.00%", "8.50%").replace("16.00%", "10.50%")
    original = {"12.50%": "7.50%", "14.00%": "8.50%", "16.00%": "10.50%"}[source]
    assert run(receipt(text.replace(original, replacement)))[0] == "veto"


def test_generic_floors_alone_do_not_certify_bank_specific_surcharges():
    gate, reasons, _ = run(receipt(BANK.replace("，均满足监管要求", "")))
    assert gate == "unknown"
    assert reasons == ("BANK_APPLICABLE_CAPITAL_REQUIREMENTS_CONFIRMATION_REQUIRED",)


def test_aspirational_or_qualified_compliance_is_not_current_compliance():
    assert run(receipt(BANK.replace("均满足", "计划满足")))[0] == "unknown"
    assert run(receipt(BANK.replace("均满足", "基本满足")))[0] == "unknown"


def test_capital_noncompliance_not_confused_with_positive_phrase():
    assert run(receipt(BANK.replace("均满足", "不满足")))[0] == "veto"


@pytest.mark.parametrize("text", [
    BANK.replace("本集团", "本行"),
    BANK.replace("本集团", "本集团旗下子公司"),
    BANK.replace("本集团", "本集团和本行"),
    BANK.replace("2026年6月30日", "2025年6月30日").replace("报告期末", "2025年6月30日"),
    BANK.replace("2026年6月30日", "2025年12月31日与2026年6月30日"),
    BANK.replace("12.50%", "0.125"),
    BANK.replace("核心一级资本充足率为", "核心一级资本充足率最低要求为"),
    BANK.replace("报告期末，本集团不良", "报告期末，本集团上年末不良"),
])
def test_parent_subsidiary_prior_year_implicit_units_and_regulatory_text_do_not_pass(text):
    assert run(receipt(text))[0] == "unknown"


def test_differing_current_values_cannot_choose_optimistic_value():
    duplicate = BANK + "报告期末，本集团核心一级资本充足率7.00%。"
    assert run(receipt(duplicate))[1] == ("SECTOR_CURRENT_GROUP_VALUES_CONFLICT",)


def test_simple_dated_table_selects_exact_current_column():
    text = (
        "本集团 2025年12月31日 2026年6月30日\n"
        "核心一级资本充足率(%) 11.00%|12.50%\n"
        "一级资本充足率(%) 13.00%|14.00%\n"
        "资本充足率(%) 15.00%|16.00%\n"
        "不良贷款率(%) 1.50%|1.20%\n"
        "拨备覆盖率(%) 190.00%|210.00%\n"
        "。报告期末，本集团资本充足率均满足监管要求。"
    )
    gate, _, metrics = run(receipt(text))
    assert gate == "pass"
    assert metrics["core_tier1_capital_ratio"] == .125
    assert metrics["nonperforming_loan_ratio"] == .012
    assert run(receipt(text.replace("本集团 2025", "本集团 本行 2025")))[0] == "unknown"
    assert run(receipt(text.replace("(%)", "")))[0] == "unknown"


@pytest.mark.parametrize("field,value", [
    ("symbol", "600000.SH"), ("text_sha256", "a" * 64),
    ("sha256", ""), ("extraction_status", "partial"), ("extraction_complete", False),
    ("url", "https://example.com/test.pdf"),
    ("published_at", "2026-10-12T00:00:00+00:00"),
    ("retrieved_at", "2026-10-12T00:00:00+00:00"),
    ("retrieved_at", "2026-10-11T07:00:00"),
])
def test_evidence_integrity_failures_are_unknown(field, value):
    data = receipt()
    data["documents"][0][field] = value
    assert run(data)[0] == "unknown"


def test_old_report_or_summary_cannot_replace_latest_full_report():
    assert run(receipt(title="2025年年度报告"))[0] == "unknown"
    assert run(receipt(title="2026年半年度报告摘要"))[0] == "unknown"
    assert run(receipt(), period="2026-09-30")[0] == "unknown"
    assert run(receipt(), period="2023-12-31")[0] == "unknown"


def test_conflicting_two_documents_fail_closed():
    data = receipt()
    other = receipt(BANK.replace("12.50%", "12.51%"))
    data["documents"].append(other["documents"][0])
    assert run(data)[0] == "unknown"


def test_insurer_requires_own_current_core_total_and_solvency_rating():
    gate, _, metrics = run(receipt(INSURANCE), industry="保险")
    assert gate == "pass"
    assert metrics["core_solvency_ratio"] == 1.5
    assert metrics["comprehensive_solvency_ratio"] == 2.5


@pytest.mark.parametrize("text", [
    INSURANCE.replace("150.00%", "49.99%"),
    INSURANCE.replace("150.00%", "50.00%").replace("250.00%", "99.99%"),
    INSURANCE.replace("A类", "C类"), INSURANCE.replace("A类", "D类"),
])
def test_insurer_regulatory_breaches_veto(text):
    assert run(receipt(text), industry="保险")[0] == "veto"


@pytest.mark.parametrize("text", [
    INSURANCE.replace("本公司", "本公司子公司"),
    INSURANCE.replace("偿付能力风险综合评级为A类", "监管评级为2A"),
    INSURANCE.replace("偿付能力风险综合评级为A类", "信用评级为AAA"),
    INSURANCE.replace("，偿付能力风险综合评级为A类", ""),
    INSURANCE.replace("150.00%", "300.00%"),
    INSURANCE.replace("偿付能力风险综合评级", "2025年度偿付能力风险综合评级"),
    INSURANCE.replace("偿付能力风险综合评级", "第一季度偿付能力风险综合评级"),
])
def test_insurance_group_subsidiary_and_rating_type_are_not_interchangeable(text):
    assert run(receipt(text), industry="保险")[0] == "unknown"


def test_insurer_exact_regulatory_boundaries_pass():
    text = INSURANCE.replace("150.00%", "50.00%").replace("250.00%", "100.00%").replace("A类", "B类")
    assert run(receipt(text), industry="保险")[0] == "pass"


@pytest.mark.parametrize("industry", ["证券", "多元金融", "金融", "港口", ""])
def test_unsupported_sector_is_explicit_not_fake_pass(industry):
    assert run(industry=industry)[0:2] == (
        "unknown", ("SECTOR_POLICY_UNSUPPORTED_REQUIRE_SPECIALIST_EVIDENCE",),
    )


def test_evaluator_does_not_mutate_input():
    data = receipt()
    before = copy.deepcopy(data)
    run(data)
    assert data == before
