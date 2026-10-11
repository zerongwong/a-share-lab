"""Narrow, source-bound financial-sector risk checks, not return predictions.

Callers must first authenticate the official manifest/PDF bytes and check that
the latest disclosed report has been selected. This module repeats document
identity, text-hash and time checks and extracts only unambiguous *current,
issuer/group* disclosures. It does not turn ordinary positive profit into a
bank/insurer approval, borrow a subsidiary's ratios, or guess table columns.

Bank minimum capital ratios and the 2.5% conservation buffer come from NFRA's
2023 Capital Rules, articles 25-27 (effective 2024-01-01). An explicit issuer
statement of compliance is also required: firm-specific, countercyclical and
systemic surcharges cannot be inferred from the generic floors.
https://www.nfra.gov.cn/cn/view/pages/rulesDetail.html?docId=1134197&itemId=4214

Insurer floors are the 2021 Solvency Rules, article 8: core 50%, comprehensive
100%, risk grade B or better. This is the solvency risk grade, not the separate
2025 supervisory rating. Group and subsidiary evidence must not be mixed.
https://www.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId=962016&generaltype=0&itemId=925

Only regulatory floors are exclusion thresholds. Bank asset-quality ratios
are required and recorded, not assigned new, unvalidated alpha thresholds.
A pass is a limited risk screen, not regulatory certification, exhaustive
financial due diligence or evidence of superior investment performance.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any
from urllib.parse import urlparse

METHOD_VERSION = "sector-financial-risk-v1.0.0"

_NUMBER = r"[+-]?\d+(?:\.\d+)?"
_DATE = re.compile(r"(20\d{2})(?:年|-)(\d{1,2})(?:月|-)(\d{1,2})日?")
_LABELS = {
    "core_tier1_capital_ratio": r"核心一级资本充足率",
    "tier1_capital_ratio": r"(?<!核心)一级资本充足率",
    "capital_adequacy_ratio": r"(?<!一级)资本充足率",
    "nonperforming_loan_ratio": r"不良贷款(?:比)?率",
    "provision_coverage_ratio": r"拨备覆盖率",
    "core_solvency_ratio": r"核心偿付能力充足率",
    "comprehensive_solvency_ratio": r"综合偿付能力充足率",
}
_BANK_METRICS = tuple(_LABELS)[:5]
_INSURANCE_METRICS = tuple(_LABELS)[5:]
_BANK_FLOORS = {
    "core_tier1_capital_ratio": 0.075,
    "tier1_capital_ratio": 0.085,
    "capital_adequacy_ratio": 0.105,
}
_SUBJECT_CONFLICT = re.compile(r"子公司|附属公司|母公司|非并表|非合并|本行口径|母行")
_NEGATED_COMPLIANCE = re.compile(r"不满足|未满足|不符合|未符合|未达到|不达标|低于.{0,12}监管")


def _normalise(text: str) -> str:
    return re.sub(r"[^\S\n]+", "", unicodedata.normalize("NFKC", text))


def _report_period(title: str) -> date | None:
    title = _normalise(title)
    if re.search(r"摘要|预约|提示|英文|审计报告", title):
        return None
    match = re.search(r"(20\d{2})年?(第一季度|一季度|半年度|中期|第三季度|三季度|年度)报告", title)
    if not match:
        return None
    month, day = {
        "第一季度": (3, 31), "一季度": (3, 31), "半年度": (6, 30),
        "中期": (6, 30), "第三季度": (9, 30), "三季度": (9, 30), "年度": (12, 31),
    }[match[2]]
    return date(int(match[1]), month, day)


def _aware(value: Any) -> datetime:
    result = datetime.fromisoformat(str(value))
    if result.utcoffset() is None:
        raise ValueError("timezone required")
    return result


def _documents(payload: Mapping, symbol: str, period: date, knowledge_time: datetime):
    """Match report by manifest identity, never by an unverified latest flag."""
    items = payload.get("items", [])
    documents = payload.get("documents", [])
    if not isinstance(items, list) or not isinstance(documents, list):
        return [], "SECTOR_OFFICIAL_REPORT_SCHEMA_INVALID"
    selected = {
        str(item.get("announcement_id")): item for item in items
        if isinstance(item, Mapping) and _report_period(str(item.get("title", ""))) == period
    }
    if not selected:
        return [], "SECTOR_LATEST_OFFICIAL_REPORT_REQUIRED"
    texts = []
    for doc in documents:
        if not isinstance(doc, Mapping) or str(doc.get("announcement_id")) not in selected:
            continue
        item = selected[str(doc["announcement_id"])]
        try:
            text = doc["text"]
            url = urlparse(str(doc["url"]))
            if (
                doc["symbol"] != symbol
                or doc["url"] != item["url"]
                or url.scheme != "https"
                or url.hostname != "static.cninfo.com.cn"
                or not url.path.lower().endswith(".pdf")
                or not isinstance(text, str) or not text.strip()
                or doc.get("extraction_complete") is not True
                or doc.get("extraction_status") != "complete"
                or not re.fullmatch(r"[0-9a-f]{64}", str(doc.get("sha256", "")))
                or hashlib.sha256(text.encode("utf-8")).hexdigest() != doc["text_sha256"]
                or _aware(doc["published_at"]) != _aware(item["published_at"])
                or not _aware(doc["published_at"]) <= _aware(doc["retrieved_at"]) <= knowledge_time
                or _aware(doc["published_at"]).date() < period
            ):
                return [], "SECTOR_OFFICIAL_REPORT_PROVENANCE_INVALID"
            texts.append(_normalise(text))
        except (KeyError, TypeError, ValueError):
            return [], "SECTOR_OFFICIAL_REPORT_PROVENANCE_INVALID"
    return texts, None if texts else "SECTOR_OFFICIAL_REPORT_TEXT_REQUIRED"


def _dates(text: str) -> list[date]:
    found = []
    for match in _DATE.finditer(text):
        try:
            found.append(date(*(int(part) for part in match.groups())))
        except ValueError:
            return []
    return found


def _current_scope(text: str, period: date, sector: str) -> bool:
    if _SUBJECT_CONFLICT.search(text):
        return False
    if sector == "bank" and "本行" in text:
        return False
    if re.search(r"不得低于|最低要求|监管要求为|监管标准为|应不低于", text):
        return False
    if re.search(r"上年末|上年度|上期|上季度|去年|年初|历史数据", text):
        return False
    if any(int(year) != period.year for year in re.findall(r"(20\d{2})年", text)):
        return False
    for quarter, month in (("一", 3), ("二", 6), ("三", 9), ("四", 12)):
        if re.search(rf"第?{quarter}季度", text) and period.month != month:
            return False
    group_scope = r"本集团|合并口径|并表口径|(?:报告期末|日)[,，:]?集团"
    scope = group_scope if sector == "bank" else group_scope + r"|本公司"
    if not re.search(scope, text):
        return False
    dates = _dates(text)
    # A comparative clause with explicit multiple periods is not parsed as a
    # one-value narrative. The table parser below handles dated columns.
    return all(value == period for value in dates) and bool(dates or "报告期末" in text)


def _extract(texts: list[str], period: date, sector: str):
    values: dict[str, set[float]] = {key: set() for key in _LABELS}
    grades: set[str] = set()
    compliance = set()
    for text in texts:
        for raw_sentence in re.split(r"[。!?！？]", text):
            sentence = raw_sentence.replace("\n", "")
            if len(sentence) > 1400 or not _current_scope(sentence, period, sector):
                continue
            for key, label in _LABELS.items():
                pattern = rf"{label}(?:为|达到|达|是|:|约为)?({_NUMBER})%"
                for match in re.finditer(pattern, sentence):
                    value = float(match[1]) / 100
                    if math.isfinite(value):
                        values[key].add(value)
            # Common prose writes the three capital values in one parallel
            # phrase. Its order is fixed explicitly, never sorted by magnitude.
            parallel = re.search(
                rf"核心一级资本充足率[、,]一级资本充足率(?:[、,]|和|及)资本充足率"
                rf"分别为({_NUMBER})%[、,]({_NUMBER})%(?:[、,]|和|及)({_NUMBER})%", sentence,
            )
            if parallel:
                for key, raw in zip(_BANK_METRICS[:3], parallel.groups(), strict=True):
                    values[key].add(float(raw) / 100)
            for match in re.finditer(r"(?:偿付能力)?风险综合评级(?:结果)?(?:为|是|:)?([ABCD])(?:类|级)", sentence):
                grades.add(match[1])
            if sector == "bank" and re.search(r"资本(?:充足率|充足水平|指标)", sentence):
                if _NEGATED_COMPLIANCE.search(sentence):
                    compliance.add(False)
                elif (
                    not re.search(r"基本满足|基本符合|基本达到|计划|目标|将满足|拟满足", sentence)
                    and re.search(r"(?:均|全部|皆)?(?:满足|符合|达到)(?:适用的?|全部的?|各项的?)?监管要求", sentence)
                ):
                    compliance.add(True)
        # Deliberately support only a simple, explicitly scoped, dated table.
        # Multi-row group/parent headers and implicit percentage units stay
        # unknown. Never take the first/highest number from an arbitrary row.
        header_dates: list[date] = []
        rows_remaining = 0
        for line in text.splitlines():
            dates = _dates(line)
            if dates:
                if (
                    re.search(r"本集团|合并口径|并表口径", line)
                    and not _SUBJECT_CONFLICT.search(line)
                    and "本行" not in line
                    and dates.count(period) == 1
                ):
                    header_dates, rows_remaining = dates, 8
                else:
                    header_dates, rows_remaining = [], 0
                continue
            if rows_remaining <= 0:
                continue
            rows_remaining -= 1
            if _SUBJECT_CONFLICT.search(line) or "本行" in line:
                header_dates, rows_remaining = [], 0
                continue
            for key, label in _LABELS.items():
                match = re.fullmatch(rf"{label}(?:\(%\)|%)((?:{_NUMBER}%?(?:[,|])?)+)", line)
                if not match:
                    continue
                numbers = re.findall(_NUMBER, match[1])
                # Space-stripped bare-number rows are not recoverable; require
                # explicit delimiters or percent signs for every column.
                if len(numbers) != len(header_dates) or len(header_dates) < 2:
                    continue
                values[key].add(float(numbers[header_dates.index(period)]) / 100)
    return values, grades, compliance


def evaluate_sector_financial_risk(
    payload: Mapping[str, Any], *, industry: str, symbol: str,
    period: str | date, knowledge_time: datetime,
) -> tuple[str, tuple[str, ...], dict[str, float]]:
    """Return pass/veto/unknown from latest official sector-specific evidence.

This supplements (does not replace) the ordinary profit/ROE and major-risk
checks. ``payload`` is the authenticated announcement manifest with PDF
``documents``. All numerical output uses fractions, e.g. 12% == 0.12.
Unsupported industries and ambiguous/missing document values are explicit
unknowns; nothing is auto-approved merely because a deadline has elapsed.
"""
    industry = str(industry).lower()
    if "银行" in industry or "bank" in industry:
        sector = "bank"
    elif "保险" in industry or "insurance" in industry:
        sector = "insurance"
    else:
        return "unknown", ("SECTOR_POLICY_UNSUPPORTED_REQUIRE_SPECIALIST_EVIDENCE",), {}
    try:
        selected = date.fromisoformat(period) if isinstance(period, str) else period
        if not isinstance(selected, date) or knowledge_time.utcoffset() is None:
            raise ValueError("invalid period or decision time")
        if selected >= knowledge_time.date() or selected < date(2024, 1, 1):
            return "unknown", ("SECTOR_REPORT_PERIOD_OUTSIDE_SUPPORTED_POLICY",), {}
        if not isinstance(payload, Mapping):
            raise ValueError("invalid manifest")
        texts, error = _documents(payload, symbol, selected, knowledge_time)
    except (TypeError, ValueError, AttributeError):
        return "unknown", ("SECTOR_OFFICIAL_REPORT_SCHEMA_INVALID",), {}
    if error:
        return "unknown", (error,), {}
    values, grades, compliance = _extract(texts, selected, sector)
    required = _BANK_METRICS if sector == "bank" else _INSURANCE_METRICS
    metrics = {key: next(iter(values[key])) for key in required if len(values[key]) == 1}
    missing = [key.upper() for key in required if not values[key]]
    ambiguous = [key.upper() for key in required if len(values[key]) > 1]
    if ambiguous or len(grades) > 1 or len(compliance) > 1:
        return "unknown", ("SECTOR_CURRENT_GROUP_VALUES_CONFLICT",), metrics
    if missing:
        return "unknown", tuple(f"SECTOR_MISSING_CURRENT_GROUP_{key}" for key in missing), metrics
    if any(value < 0 or not math.isfinite(value) for value in metrics.values()):
        return "unknown", ("SECTOR_PERCENTAGE_UNIT_OR_VALUE_INVALID",), metrics
    if sector == "bank":
        if (
            metrics["nonperforming_loan_ratio"] > 1
            or not metrics["core_tier1_capital_ratio"] <= metrics["tier1_capital_ratio"] <= metrics["capital_adequacy_ratio"] <= 1
        ):
            return "unknown", ("SECTOR_BANK_RATIO_ORDER_OR_UNIT_INVALID",), metrics
        breaches = [key.upper() for key, floor in _BANK_FLOORS.items() if metrics[key] < floor]
        if breaches or compliance == {False}:
            return "veto", tuple(f"BANK_REGULATORY_CAPITAL_FLOOR_BREACH_{key}" for key in breaches) or ("BANK_REPORTED_CAPITAL_NONCOMPLIANCE",), metrics
        if compliance != {True}:
            return "unknown", ("BANK_APPLICABLE_CAPITAL_REQUIREMENTS_CONFIRMATION_REQUIRED",), metrics
        return "pass", ("BANK_OFFICIAL_CAPITAL_AND_ASSET_QUALITY_BASELINE_CONFIRMED",), metrics
    if metrics["core_solvency_ratio"] > metrics["comprehensive_solvency_ratio"]:
        return "unknown", ("SECTOR_INSURANCE_RATIO_ORDER_OR_UNIT_INVALID",), metrics
    if metrics["core_solvency_ratio"] < .50 or metrics["comprehensive_solvency_ratio"] < 1:
        return "veto", ("INSURANCE_REGULATORY_SOLVENCY_FLOOR_BREACH",), metrics
    if grades & {"C", "D"}:
        return "veto", ("INSURANCE_SOLVENCY_RISK_RATING_BELOW_B",), metrics
    if not grades:
        return "unknown", ("INSURANCE_CURRENT_ISSUER_SOLVENCY_RISK_RATING_REQUIRED",), metrics
    return "pass", ("INSURANCE_OFFICIAL_SOLVENCY_BASELINE_CONFIRMED",), metrics
