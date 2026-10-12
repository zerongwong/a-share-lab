"""Targeted, document-grounded basic risk screening, not full due diligence.

Only latest reports/audit and potentially material filings require PDF content.
Ordinary governance paperwork is not a separate approval queue. Ambiguous facts,
incomplete extraction and unsupported document families remain unknown. These
conservative recognisers are versioned research controls, not legal opinions.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_REPORT = re.compile(r"(?<!\d)(20\d{2})年?\s*(第一季度|一季度|半年度|中期|第三季度|三季度|年度|年报|度)\s*(?:报告|年报)?")
_MONTHS = {"第一季度": 3, "一季度": 3, "半年度": 6, "中期": 6, "第三季度": 9, "三季度": 9, "年度": 12, "年报": 12, "度": 12}
_HARD_TITLE = re.compile("立案|调查|处罚|违规|违约|财务造假|会计差错|非标|非标准|保留意见|否定意见|无法表示|持续经营|退市|风险警示|冻结|破产|清算|资金占用")
_REVIEW_TITLE = re.compile("诉讼|仲裁|担保|债务|减值|更正|业绩预|重大资产|重大事项|关联交易|停牌|质押|重组|监管措施|警示函|关注函|问询函|审计意见")
_FACT_PATTERNS = (
    ("OFFICIAL_DISCLOSURE_INVESTIGATION_OR_FRAUD", re.compile(r"(?P<event>收到.{0,25}立案告知|被.{0,20}立案调查|涉嫌.{0,15}(?:财务造假|信息披露违法))")),
    ("OFFICIAL_DISCLOSURE_DEBT_DEFAULT", re.compile(r"(?P<event>发生.{0,12}债务违约|债务.{0,12}已逾期|未能(?:按期)?.{0,10}(?:偿还|兑付))")),
    ("OFFICIAL_DISCLOSURE_DELISTING_OR_INSOLVENCY", re.compile(r"(?P<event>被实施.{0,10}退市风险警示|被终止上市|进入破产清算|资不抵债|被裁定.{0,10}破产)")),
    ("OFFICIAL_DISCLOSURE_MAJOR_ASSET_FREEZE", re.compile(r"(?:主要|核心|全部|大部分).{0,12}(?:资产|银行账户).{0,12}(?P<event>被司法冻结|已被冻结|被冻结|遭冻结)")),
    ("OFFICIAL_DISCLOSURE_ILLEGAL_FUNDS_OCCUPATION", re.compile(r"(?P<event>违规占用.{0,12}公司资金|非经营性占用.{0,12}公司资金)")),
    ("OFFICIAL_DISCLOSURE_NONSTANDARD_AUDIT", re.compile(r"(?P<event>(?:出具|发表).{0,18}(?:(?<!无)保留意见|否定意见|无法表示意见)|(?:审计意见为|审计报告类型为).{0,5}(?:(?<!无)保留意见|否定意见|无法表示意见))")),
)


def full_report_period(title: str) -> date | None:
    """Recognise full reports, not their summaries, forecasts or corrections."""
    # CNINFO also publishes filename-style issuer titles, for example the
    # authentic ``600789_2025年_年度报告``. Separators carry no report semantics.
    title = re.sub(r"_+", " ", title)
    if any(word in title for word in ("摘要", "预约", "更正", "修订", "补充", "提示", "英文", "业绩预")):
        return None
    match = _REPORT.search(title)
    if match is None or not any(word in match.group(0) for word in ("报告", "年报")):
        return None
    suffix = title[match.end():].strip(" ：:")
    if suffix not in {"", "（全文）", "(全文)", "全文", "正文", "及审计报告"}:
        return None
    month = _MONTHS[match.group(2)]
    return date(int(match.group(1)), month, {3: 31, 6: 30, 9: 30, 12: 31}[month])


def classify_document_title(title: str) -> str:
    """Selection priority, never a substitute for a substantive risk conclusion."""
    pure_policy = re.search(r"(?:制度|办法|公司章程|议事规则|工作细则)(?:修订|修订稿|（[^）]{0,20}修订[^）]{0,10}）|\([^)]{0,20}修订[^)]{0,10}\))?(?:的公告|公告)?$", title)
    if pure_policy and not re.search("收到|发生|涉及|违规|被|处罚|立案|调查|违约|冻结|终止上市|新增.*交易|出售|收购|提供.*担保", title):
        return "routine_governance"
    if _HARD_TITLE.search(title):
        return "material_risk"
    if full_report_period(title) is not None:
        return "financial_report"
    if "审计报告" in title:
        return "audit_report"
    if _REVIEW_TITLE.search(title):
        return "material_risk"
    return "routine_other"


def required_document_items(
    items: Sequence[Mapping[str, Any]], financial_period: str | None, knowledge_time: datetime,
) -> tuple[dict[str, Any], ...]:
    """Latest applicable full report/audit plus only substantive-risk candidates."""
    try:
        selected = date.fromisoformat(str(financial_period))
    except (TypeError, ValueError):
        selected = None
    known = knowledge_time.astimezone(_SHANGHAI).date()
    annual_floor = known.year - (1 if known >= date(known.year, 5, 1) else 2)
    if selected is not None and selected.month == 12:
        annual_floor = max(annual_floor, selected.year)
    reports = [(item, full_report_period(str(item.get("title") or ""))) for item in items]
    chosen: dict[str, dict[str, Any]] = {}
    current = [item for item, period in reports if period == selected and period is not None]
    annual = [(item, period) for item, period in reports if period is not None and period.month == 12 and annual_floor <= period.year < known.year]
    for group in (current, [item for item, period in annual if period == max((p for _, p in annual), default=None)]):
        if group:
            item = max(group, key=lambda row: (str(row.get("published_at")), str(row.get("announcement_id"))))
            chosen[str(item["announcement_id"])] = dict(item)
    for item in items:
        if classify_document_title(str(item.get("title") or "")) == "material_risk":
            chosen[str(item["announcement_id"])] = dict(item)
    return tuple(chosen.values())


# The first adapter prototype used this name; retain a harmless function alias.
targeted_document_items = required_document_items


def _aware(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("timestamp lacks timezone")
    return parsed


def validate_document(
    document: Mapping[str, Any], item: Mapping[str, Any], *, symbol: str, knowledge_time: datetime,
    retrieval_time: datetime | None = None,
) -> tuple[bool, str]:
    try:
        if document.get("error"):
            error = str(document["error"])
            return False, error if re.fullmatch(r"[A-Z_]{4,80}", error) else "TARGETED_DOCUMENT_EXTRACTION_INCOMPLETE"
        text = document["text"]
        fetched = _aware(document["retrieved_at"])
        published = _aware(document["published_at"])
        if (
            document.get("symbol") != symbol or document["announcement_id"] != item["announcement_id"]
            or document["url"] != item["url"] or published != _aware(item["published_at"])
            or published > knowledge_time or not published <= fetched <= (retrieval_time or knowledge_time)
        ):
            return False, "TARGETED_DOCUMENT_IDENTITY_OR_TIME_INVALID"
        if (
            document.get("error") or document.get("extraction_status") != "complete"
            or document.get("extraction_complete") is not True
            or not isinstance(document.get("page_count"), int) or isinstance(document["page_count"], bool)
            or document["page_count"] < 1 or not isinstance(text, str) or len(text.strip()) < 80
        ):
            return False, "TARGETED_DOCUMENT_EXTRACTION_INCOMPLETE"
        if (
            not re.fullmatch(r"[0-9a-f]{64}", str(document.get("sha256") or ""))
            or hashlib.sha256(text.encode()).hexdigest() != document.get("text_sha256")
        ):
            return False, "TARGETED_DOCUMENT_HASH_INVALID"
        return True, "TARGETED_DOCUMENT_PROVENANCE_CONFIRMED"
    except (ValueError, TypeError, KeyError, AttributeError):
        return False, "TARGETED_DOCUMENT_RECEIPT_INVALID"


def _fact_findings(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    # Paragraph boundaries prevent an unrelated disclaimer elsewhere in the PDF
    # from negating a specific finding. Broad risk-factor discussions aren't facts.
    reasons, uncertain = [], []
    # PDF soft line/page wraps cannot split a factual predicate. Chinese sentence
    # punctuation, not formatting newlines, bounds negation and hypothetical scope.
    for sentence in re.split(r"[。；;]", text):
        compact = re.sub(r"\s+", "", sentence)
        for reason, pattern in _FACT_PATTERNS:
            for match in pattern.finditer(compact):
                prefix = compact[max(0, match.start("event") - 65):match.start("event")]
                if not re.search(r"公司|发行人|本集团|股票|审计", prefix) and "AUDIT" not in reason:
                    continue
                # Only negation immediately attached to THIS event counts. An
                # earlier '不存在逾期担保，但...' cannot negate a later default.
                if re.search(r"(?:不存在|没有|尚未|未|不曾)[^，,。；;]{0,6}$", prefix):
                    continue
                if (
                    reason == "OFFICIAL_DISCLOSURE_ILLEGAL_FUNDS_OCCUPATION"
                    and re.search(r"(?:无|不存在)(?:(?:大股东|控股股东|实际控制人|其他关联方|关联方)(?:及|和|与|、)?)+$", prefix)
                    and re.match(r"(?:情形|情况)", compact[match.end():])
                ):
                    # '公司无大股东及关联方非经营性占用公司资金情形' is an
                    # explicit issuer denial, not a confirmed occupation fact.
                    continue
                if re.search(r"如果|若(?:本)?公司|倘若|假如|可能发生|一旦|防止|避免|防范", prefix):
                    continue
                # Standard exchange report templates print adverse headings even
                # when their applicability boxes are explicitly unchecked.
                tail = compact[match.end():match.end() + 100]
                if re.search(r"(?:报告期内)?公司股票$", prefix) and re.search(r"□适用[√✓✔]不适用", tail):
                    continue
                if re.search(r"已(?:全部)?(?:解除|结清|偿还|撤销|结案)|现已解决|不再存在|(?:公司|发行人).{0,8}曾|20\d{2}年.{0,20}(?:结案|解除|结清)", compact):
                    uncertain.append("HISTORICAL_RISK_RESOLUTION_NEEDS_CONTENT_REVIEW")
                    continue
                if (
                    reason == "OFFICIAL_DISCLOSURE_INVESTIGATION_OR_FRAUD"
                    and re.search(r"实际控制人|控股股东|董事|高级管理人员", prefix)
                    and not re.search(r"(?:本公司|上市公司|发行人)(?:自身|本身)?[^，,《》]{0,8}$", prefix)
                ):
                    # A person's case is material evidence to review, but cannot
                    # be relabelled as an investigation of the listed issuer.
                    uncertain.append("RELATED_PERSON_DISCLOSURE_REQUIRES_IMPACT_REVIEW")
                    continue
                reasons.append(reason)
        if (
            re.search(r"(?:上述事项表明|公司存在|我们提请关注).{0,45}持续经营.{0,40}重大不确定性", compact)
            and not re.search(r"如果|若我们|不存在|未发现", compact)
        ):
            uncertain.append("GOING_CONCERN_MATERIAL_UNCERTAINTY_REQUIRES_REVIEW")
    return tuple(dict.fromkeys(reasons)), tuple(dict.fromkeys(uncertain))


def _adverse_facts(text: str) -> tuple[str, ...]:
    return _fact_findings(text)[0]


def _routine_risk_document(title: str, text: str) -> bool:
    """Narrow positive evidence for common disclosures; ambiguity is not a pass."""
    compact = re.sub(r"\s+", "", text)
    # An ordinary-looking title cannot rescue a substantive allegation or a
    # disclosure outside these two well-defined, body-evidenced families.
    if _HARD_TITLE.search(title) or _adverse_facts(text):
        return False
    if "担保" in title:
        ratio = re.search(r"(?:担保总额|担保余额).{0,35}(?:占|超过).{0,30}净资产[^\d%]{0,15}(\d+(?:\.\d+)?)%", compact)
        return bool(
            re.search(r"(?:全资|控股)子公司", compact)
            and re.search(r"(?:不存在|无|未发生).{0,10}逾期担保", compact)
            and re.search(r"(?:已审议通过|授权|批准).{0,20}(?:担保额度|范围内)|(?:担保额度|范围内).{0,20}(?:已审议通过|授权|批准)", compact)
            and ratio and 0 <= float(ratio[1]) < 100
            and not re.search(r"重大.{0,10}(?:偿债|财务|违约)风险|超出.{0,10}承受能力", compact)
            and not re.search(r"(?:违规担保|超出.{0,10}授权|被担保人.{0,15}违约)", compact)
        )
    if "关联交易" in title:
        return bool(
            "日常关联交易" in compact and re.search(r"定价.{0,10}公允|市场价格|市场公允", compact)
            and re.search(r"不(?:会)?(?:损害|影响).{0,20}(?:利益|独立性)", compact)
            and re.search(r"(?:交易金额|交易额度).{0,20}\d[\d,.，]*(?:万元|亿元|人民币元)", compact)
            and re.search(r"(?:审议通过|批准|决策程序)", compact)
            and not re.search(r"(?:形成|存在).{0,10}(?:较大|重大|一定的?)依赖", compact)
            and not re.search(r"非经营性资金占用|违规.{0,8}资金|未履行.{0,10}审议", compact)
        )
    return False


def evaluate_targeted_material_review(
    manifest: Mapping[str, Any], *, symbol: str, financial_period: str | None,
    knowledge_time: datetime, retrieval_time: datetime | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Called only after the official manifest's identity and coverage validation."""
    try:
        selected = date.fromisoformat(str(financial_period))
    except (TypeError, ValueError):
        return "unknown", ("LATEST_FINANCIAL_REPORT_PERIOD_REQUIRED",)
    items = manifest["items"]
    targets = required_document_items(items, financial_period, knowledge_time)
    selected_items = [item for item in targets if full_report_period(item["title"]) == selected]
    if not selected_items:
        return "unknown", ("LATEST_FULL_OFFICIAL_FINANCIAL_REPORT_REQUIRED",)
    annual_items = [item for item in targets if (period := full_report_period(item["title"])) is not None and period.month == 12]
    if not annual_items:
        return "unknown", ("LATEST_FULL_OFFICIAL_ANNUAL_REPORT_REQUIRED",)
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        return "unknown", ("TARGETED_OFFICIAL_DOCUMENTS_REQUIRED",)
    indexed = {doc.get("announcement_id"): doc for doc in documents if isinstance(doc, Mapping)}
    if len(indexed) != len(documents):
        return "unknown", ("TARGETED_DOCUMENT_DUPLICATE_OR_INVALID",)
    missing, adverse, ambiguous = [], [], []
    verified = {}
    for item in targets:
        doc = indexed.get(item["announcement_id"])
        if doc is None:
            missing.append("TARGETED_OFFICIAL_DOCUMENT_MISSING")
            continue
        valid, reason = validate_document(doc, item, symbol=symbol, knowledge_time=knowledge_time, retrieval_time=retrieval_time)
        if not valid:
            missing.append(reason)
            continue
        verified[item["announcement_id"]] = doc
        facts, unresolved = _fact_findings(doc["text"])
        adverse.extend(facts)
        ambiguous.extend(unresolved)
    if adverse:
        return "veto", tuple(dict.fromkeys(adverse))
    if missing:
        return "unknown", tuple(dict.fromkeys(missing))
    if ambiguous:
        return "unknown", tuple(dict.fromkeys(ambiguous))
    for item in selected_items:
        text = re.sub(r"\s+", "", verified[item["announcement_id"]]["text"])
        if str(selected.year) not in text or not all(term in text for term in ("资产负债表", "利润表", "现金流量表")):
            return "unknown", ("LATEST_FINANCIAL_REPORT_CONTENT_UNCONFIRMED",)
    for item in annual_items:
        text = re.sub(r"\s+", "", verified[item["announcement_id"]]["text"])
        if re.search(r"(?:出具|发表).{0,20}带.{0,25}(?:强调事项|持续经营).{0,25}无保留意见", text):
            return "unknown", ("ANNUAL_AUDIT_EMPHASIS_OR_GOING_CONCERN_REQUIRES_REVIEW",)
        explicit = re.search(r"(?:出具|发表).{0,35}(?:标准的?无保留意见|标准无保留)|(?:审计意见|审计报告类型).{0,15}标准无保留", text)
        narrative = re.search(r"我们认为.{0,60}财务报表.{0,45}在所有重大方面.{0,55}按照.{0,35}企业会计准则.{0,80}公允反映", text)
        if not explicit and not narrative:
            return "unknown", ("LATEST_ANNUAL_AUDIT_OPINION_UNCONFIRMED",)
    for item in targets:
        if classify_document_title(item["title"]) == "material_risk" and not _routine_risk_document(item["title"], verified[item["announcement_id"]]["text"]):
            return "unknown", ("MATERIAL_FACT_CONTENT_NEEDS_REVIEW",)
    return "pass", ("TARGETED_FINANCIAL_AND_MATERIAL_RISK_CONTENT_CONFIRMED",)
