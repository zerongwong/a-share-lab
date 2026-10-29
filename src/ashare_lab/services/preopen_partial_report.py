"""Bounded local staging for the independent pre-open deadline sender.

Not a trading plan or a private holding journal. Only same-day public market
observations enter this cache. No prices, holdings, weights, arbitrary errors,
credentials or URLs cross the boundary. No network, analytics, or DB reads.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

SNAPSHOT_NAME = "preopen-partial-snapshot.json"
_SCHEMA = "preopen-observation-snapshot-v1"
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_MAX_BYTES = 16_384
_CYCLE_LABELS = frozenset({
    "中期上行｜短线增强", "中期上行｜短线回撤或分化",
    "中期过渡｜复苏尝试或证据混合", "中期下行｜短线修复反弹",
    "中期下行｜短线压力", "价格周期数据不可用",
})
_POSTURES = {
    "standard": "偏进攻", "tight": "均衡偏防守", "defensive": "防守",
    "exception_only": "强防守", "unavailable": "观望",
}
_FORMATIONS = frozenset({"确认突破", "回踩企稳", "回踩待企稳", "形态待核验"})
_MISSING = {
    "calendar_unverified": "当日是否开市未核实",
    "market_data_stale": "行情尚未追平前一交易日",
    "market_data_unavailable": "完整行情尚未取得",
    "screening_incomplete": "全市场筛选未完成",
    "fundamental_incomplete": "财务与重大风险核验未完成",
    "risk_review_incomplete": "组合风险与买入条件未完成确认",
    "holding_review_incomplete": "持仓复核未完成",
    "submission_unconfirmed": "完整计划推送尚未确认受理",
    "workflow_incomplete": "核验流程未完成",
}
_REASONS = {
    "fundamental": "基本面证据仍需核验", "execution": "可交易性仍需核验",
    "fundamental_pass": "基本面核验通过，完整买入条件未确认",
    "veto": "风险条件不合格，不新买",
    "risk": "风险与保护线仍需核验", "confirmation": "量价确认仍需核验",
    "portfolio": "组合与当日买价仍需核验", "unknown": "完整买入条件未确认",
}


def _date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        result = date.fromisoformat(value)
        return result if result.isoformat() == value else None
    except ValueError:
        return None


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value)
        return result.astimezone(_SHANGHAI) if result.tzinfo is not None else None
    except ValueError:
        return None


def _name(value: object) -> str | None:
    if not isinstance(value, str) or not 1 <= len(value) <= 24:
        return None
    if not re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9*·（）() -]+", value):
        return None
    if not re.search(r"[\u4e00-\u9fff]", value) or re.search(r"[A-Za-z]{5,}", value):
        return None
    return value


def _reason_code(row: Mapping) -> str:
    code = row.get("reason_code")
    if isinstance(code, str) and code in _REASONS:
        return code
    reason = row.get("reason")
    if not isinstance(reason, str):
        return "unknown"
    # Categories only: never disclose a free-form upstream reason.
    if any(term in reason for term in ("不合格", "超过8%", "排除")):
        return "veto"
    if "财务与重大风险核验通过" in reason:
        return "execution" if "可交易性未确认" in reason else "fundamental_pass"
    for terms, code in (
        (("财务", "基本面", "公告", "重大风险"), "fundamental"),
        (("可交易", "可成交"), "execution"),
        (("保护线", "结构风险", "风险计算", "初始风险"), "risk"),
        (("量价", "企稳", "突破确认"), "confirmation"),
        (("组合", "当日买价"), "portfolio"),
    ):
        if any(term in reason for term in terms):
            return code
    return "unknown"


def _candidates(value: object) -> list[dict]:
    if not isinstance(value, (list, tuple)):
        return []
    result, seen = [], set()
    for row in value[:20]:
        if not isinstance(row, Mapping):
            continue
        name = _name(row.get("name"))
        if name is None or name in seen:
            continue
        seen.add(name)
        formation = row.get("formation")
        result.append({
            "name": name,
            "formation": formation if isinstance(formation, str) and formation in _FORMATIONS else "形态待核验",
            "reason_code": _reason_code(row),
        })
        if len(result) == 5:
            break
    return result


def _read(path: Path) -> dict:
    """Bound bytes and file type so a malformed cache cannot stall the sender."""
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
            return {}
        raw = os.read(descriptor, _MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            return {}
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}
    finally:
        if descriptor is not None:
            os.close(descriptor)


def read_preopen_partial_snapshot(state_root: Path, *, now: datetime) -> dict | None:
    if now.tzinfo is None:
        return None
    now = now.astimezone(_SHANGHAI)
    raw = _read(Path(state_root) / SNAPSHOT_NAME)
    generated = _timestamp(raw.get("generated_at"))
    cutoff = _date(raw.get("data_cutoff"))
    if (
        raw.get("schema") != _SCHEMA
        or raw.get("scope") != "whole_market_observation_only"
        or raw.get("plan_for_date") != now.date().isoformat()
        or generated is None or generated.date() != now.date() or generated > now
        or (raw.get("data_cutoff") is not None and cutoff is None)
        or (cutoff is not None and cutoff >= now.date())
    ):
        return None
    codes = raw.get("missing_codes")
    if not isinstance(codes, list):
        codes = ["workflow_incomplete"]
    codes = list(dict.fromkeys(code if isinstance(code, str) and code in _MISSING
                               else "workflow_incomplete" for code in codes[:12]))
    cycle, strictness, method = (raw.get(key) for key in ("cycle_label", "entry_strictness", "method_version"))
    return {
        "schema": _SCHEMA, "scope": "whole_market_observation_only",
        "plan_for_date": now.date().isoformat(), "generated_at": generated.isoformat(),
        "data_cutoff": None if cutoff is None else cutoff.isoformat(),
        "method_version": method if isinstance(method, str) and len(method) <= 80 else None,
        "cycle_label": cycle if cutoff is not None and isinstance(cycle, str) and cycle in _CYCLE_LABELS else None,
        "entry_strictness": strictness if cutoff is not None and isinstance(strictness, str) and strictness in _POSTURES else None,
        "candidates": _candidates(raw.get("candidates")) if cutoff is not None else [],
        "missing_codes": codes,
    }


def write_preopen_partial_snapshot(
    state_root: Path, *, report_date: date, generated_at: datetime,
    data_cutoff: date | None = None, cycle_label: str | None = None,
    entry_strictness: str | None = None,
    screening_candidates: Sequence[Mapping] | None = None,
    missing_codes: Sequence[str] = (), stage: str | None = None,
    method_version: str | None = None,
) -> bool:
    """Atomically merge current-day progress, never the private archive.

    Stage is accepted for the progress hook but not persisted or displayed.
    A changed cutoff or method invalidates old evidence. Failure stays local.
    """
    del stage
    if (
        not isinstance(report_date, date) or isinstance(report_date, datetime)
        or not isinstance(generated_at, datetime) or generated_at.tzinfo is None
        or generated_at.astimezone(_SHANGHAI).date() != report_date
        or (data_cutoff is not None and (
            not isinstance(data_cutoff, date) or isinstance(data_cutoff, datetime)
            or data_cutoff >= report_date
        ))
    ):
        return False
    now, root = generated_at.astimezone(_SHANGHAI), Path(state_root)
    descriptor, temporary = None, None
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(root / "preopen-partial-snapshot.lock", os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = _read(root / SNAPSHOT_NAME)
        prior_time = _timestamp(raw.get("generated_at"))
        if prior_time is not None and prior_time > now:
            return False
        previous = read_preopen_partial_snapshot(root, now=now) or {}
        cutoff = data_cutoff.isoformat() if data_cutoff is not None else previous.get("data_cutoff")
        method = method_version if method_version is not None else previous.get("method_version")
        if cutoff != previous.get("data_cutoff") or method != previous.get("method_version"):
            previous = {}
        payload = {
            "schema": _SCHEMA, "scope": "whole_market_observation_only",
            "plan_for_date": report_date.isoformat(), "generated_at": now.isoformat(),
            "data_cutoff": cutoff, "method_version": method,
            "cycle_label": cycle_label if cycle_label is not None else previous.get("cycle_label"),
            "entry_strictness": entry_strictness if entry_strictness is not None else previous.get("entry_strictness"),
            "candidates": (_candidates(screening_candidates) if screening_candidates is not None
                           else previous.get("candidates", [])),
            "missing_codes": [code if isinstance(code, str) and code in _MISSING
                              else "workflow_incomplete" for code in list(missing_codes)[:12]],
        }
        cycle, strictness = payload["cycle_label"], payload["entry_strictness"]
        if not isinstance(cycle, str) or cycle not in _CYCLE_LABELS:
            payload["cycle_label"] = None
        if not isinstance(strictness, str) or strictness not in _POSTURES:
            payload["entry_strictness"] = None
        if not isinstance(method, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", method):
            payload["method_version"] = None
        if cutoff is None:
            payload.update(cycle_label=None, entry_strictness=None, candidates=[])
        temporary = root / f".{SNAPSHOT_NAME}.{uuid4().hex}.tmp"
        with temporary.open("x", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(root / SNAPSHOT_NAME)
        return True
    except (OSError, TypeError, ValueError):
        return False
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        if descriptor is not None:
            with suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def write_preopen_partial_digest(state_root: Path, digest, *, generated_at: datetime) -> bool:
    """Whitelist the independent market screen, not account-specific names."""
    plan, candidates = digest.continuous_plan, []
    if isinstance(plan, Mapping) and (
        plan.get("holding_based") is not True or isinstance(plan.get("market_model_portfolio"), Mapping)
    ):
        candidates = plan.get("screening_candidates", [])
    return write_preopen_partial_snapshot(
        state_root, report_date=digest.plan_for_date, generated_at=generated_at,
        data_cutoff=digest.common_cutoff, cycle_label=digest.cycle_label,
        entry_strictness=digest.entry_strictness, screening_candidates=candidates or [],
        missing_codes=("submission_unconfirmed", "risk_review_incomplete"),
        method_version=digest.method_version,
    )


def render_preopen_partial_message(state_root: Path, *, now: datetime, session: bool | None):
    """A dated partial observation report, never executable buy instructions."""
    from ashare_lab.ports.notifications import NotificationMessage

    snapshot = read_preopen_partial_snapshot(state_root, now=now)
    lines = [f"🗓 {now.astimezone(_SHANGHAI).date().isoformat()}｜盘前部分结果"]
    if snapshot is None:
        lines.extend([
            "数据截至：未知（未取得有效的当日摘要）", "市场周期：待核验",
            "候选：尚未取得，不等于没有合适股票。",
            "待完成：行情截止日、筛选结果与完整计划确认。",
        ])
    else:
        cutoff, cycle = snapshot["data_cutoff"] or "未知", snapshot["cycle_label"] or "待核验"
        posture = _POSTURES.get(snapshot["entry_strictness"])
        lines.extend([f"数据截至：{cutoff}", f"市场周期：{cycle}" + (f"｜{posture}" if posture else "")])
        candidates = snapshot["candidates"]
        if candidates:
            lines.append("🔎 已筛选候选｜仅观察，不代表可以买")
            lines.append("以下形态仅截至上述数据日，未确认今日买入条件。")
            for row in candidates:
                lines.append(f"{row['name']}｜{row['formation']}｜{_REASONS[row['reason_code']]}")
        else:
            lines.append("候选：暂无可展示的已核验摘要，不等于市场没有机会。")
        codes = list(snapshot["missing_codes"])
        if not snapshot["cycle_label"] or not snapshot["data_cutoff"]:
            codes.append("market_data_unavailable")
        missing = list(dict.fromkeys(_MISSING[code] for code in codes))
        lines.append("待完成：" + "；".join(missing or ["完整买入条件与计划确认"]) + "。")
    if session is None:
        lines.append("交易日核验尚未完成，不代表今日一定开市。")
    lines.append("本条不是荐股结论：未核验不等于通过，暂不据此新买；请勿沿用旧买入建议。")
    title = "A股盘前部分结果｜暂停新买"
    if session is None:
        title += "｜交易日待核验"
    return NotificationMessage(title=title, body="\n\n".join(lines), group="A股研究室·盘前摘要")
