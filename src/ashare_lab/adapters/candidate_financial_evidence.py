"""Bounded reads for public *screening candidates*, never account information.

BaoStock is a secondary structured financial source. CNINFO supplies a complete,
bounded announcement manifest, not a semantic "all risks clear" certificate.
This module is also the disposable worker entry point: the SDK's socket reads
and logout cannot hold the report process indefinitely.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import UTC, date, datetime, timedelta
from io import StringIO
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

METHOD_VERSION = "candidate-evidence-v1.0.0"
REPORTING_WINDOW_SOURCE = "https://www.sse.com.cn/lawandrules/sselawsrules2025/bond/convertible/listing/c/c_20260424_10817746.shtml"
MAX_CANDIDATES = 36
MAX_ANNOUNCEMENT_PAGES = 16
PAGE_SIZE = 30
_SYMBOL = re.compile(r"^(\d{6})\.(SH|SZ|BJ)$")
_MAX_WIRE_BYTES = 8_000_000
_SHANGHAI = ZoneInfo("Asia/Shanghai")


def content_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def required_report_period(on_date: date) -> date:
    """Latest mandatory quarter after statutory reporting windows have closed.

    SSE Listing Rules 5.2.2 (REPORTING_WINDOW_SOURCE) supply the ordinary
    four-month annual, two-month half-year, one-month quarterly windows.
    This is a freshness floor, not a claim that a report was published on its
    period end. Actual pubDate is checked separately against decision time.
    During April require the preceding Q3 until the annual/Q1 window closes;
    the collector additionally probes newer completed quarters.
    """
    if on_date >= date(on_date.year, 11, 1):
        return date(on_date.year, 9, 30)
    if on_date >= date(on_date.year, 9, 1):
        return date(on_date.year, 6, 30)
    if on_date >= date(on_date.year, 5, 1):
        return date(on_date.year, 3, 31)
    return date(on_date.year - 1, 9, 30)


def announcement_window_start(on_date: date) -> date:
    """Include the latest legally due annual report, also before April's deadline.

    In April 2027, the still-current 2025 annual may have been published in
    March 2026. A plain 365-day window would silently omit that mandatory audit.
    Start no later than January of its possible publication year; retain at
    least one year for subsequent risks. Pagination remains bounded/fail-closed.
    """
    required_annual_year = on_date.year - (1 if on_date >= date(on_date.year, 5, 1) else 2)
    return min(on_date - timedelta(days=365), date(required_annual_year + 1, 1, 1))


def _periods(on_date: date) -> list[date]:
    floor = required_report_period(on_date)
    return sorted(
        {
            date(year, month, day)
            for year in (on_date.year - 1, on_date.year)
            for month, day in ((3, 31), (6, 30), (9, 30), (12, 31))
            if floor <= date(year, month, day) < on_date
        },
        reverse=True,
    )


def _validate_request(symbols, cutoff, knowledge_time, timeout_seconds) -> tuple[str, ...]:
    if isinstance(symbols, (str, bytes)):
        raise ValueError("candidate symbols must be a sequence")
    requested = tuple(symbols)
    if len(requested) > MAX_CANDIDATES or len(set(requested)) != len(requested):
        raise ValueError("candidate symbols must be unique and bounded to 36")
    if any(not isinstance(s, str) or not _SYMBOL.fullmatch(s) for s in requested):
        raise ValueError("candidate symbols must use canonical six-digit exchange identities")
    if not isinstance(cutoff, date) or isinstance(cutoff, datetime):
        raise ValueError("cutoff must be a date")
    if not isinstance(knowledge_time, datetime) or knowledge_time.utcoffset() is None:
        raise ValueError("knowledge_time must be timezone-aware")
    if cutoff > knowledge_time.astimezone(_SHANGHAI).date():
        raise ValueError("price cutoff cannot be after knowledge time")
    if isinstance(timeout_seconds, bool) or not 0 < float(timeout_seconds) <= 120:
        raise ValueError("candidate evidence deadline must be in (0, 120]")
    return requested


def read_candidate_evidence(
    symbols, *, cutoff: date, knowledge_time: datetime, timeout_seconds=75, runner=subprocess.run,
    reuse_receipts=None,
) -> dict[str, dict[str, Any]]:
    """Return completed per-symbol receipts, retaining partial work on timeout.

    Only codes and public time boundaries cross the worker boundary. A timeout
    is provider/data failure, never a zero-eligible-market result. The parent
    independently verifies symbol identity and method version.
    """
    requested = _validate_request(symbols, cutoff, knowledge_time, timeout_seconds)
    if not requested:
        return {}
    request = {
        "symbols": requested,
        "cutoff": cutoff.isoformat(),
        "knowledge_time": knowledge_time.isoformat(),
    }
    reusable = {}
    for symbol, receipt in (reuse_receipts or {}).items():
        # The caller has already validated same-day freshness. Keep public
        # financial payloads in the parent; only a list of public codes is sent.
        if symbol in requested and isinstance(receipt, dict):
            reusable[symbol] = receipt
    if reusable:
        request["skip_financial_symbols"] = list(reusable)
    reason = "PROVIDER_RESULT_MISSING"
    try:
        completed = runner(
            [sys.executable, "-m", __name__],
            input=json.dumps(request), text=True, capture_output=True,
            timeout=float(timeout_seconds), check=False,
        )
        stdout = completed.stdout
        if completed.returncode:
            reason = "PROVIDER_WORKER_FAILED"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        reason = "PROVIDER_DEADLINE_EXCEEDED"
    except OSError:
        stdout = ""
        reason = "PROVIDER_WORKER_UNAVAILABLE"
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    result = {}
    if isinstance(stdout, str) and len(stdout.encode()) <= _MAX_WIRE_BYTES:
        for line in stdout.splitlines():
            try:
                item = json.loads(line)
                symbol = item["symbol"]
                if symbol in requested and item["method_version"] == METHOD_VERSION:
                    # A worker emits a partial financial receipt before a slow
                    # CNINFO request, then replaces it with the complete receipt.
                    result[symbol] = item
            except (ValueError, TypeError, KeyError):
                continue
    for symbol in requested:
        if symbol not in result:
            result[symbol] = {
                "symbol": symbol, "method_version": METHOD_VERSION,
                "error": reason, "financials": {}, "announcements": {"error": reason},
            }
        if symbol in reusable:
            original = reusable[symbol]
            result[symbol] = {
                **result[symbol], "financials": original["financials"],
                "execution": original["execution"],
                "financial_retrieved_at": original.get("financial_retrieved_at", original["retrieved_at"]),
                "retrieved_at": result[symbol].get("retrieved_at", original["retrieved_at"]),
            }
    return result


def _rows(result) -> list[dict[str, str]]:
    if str(getattr(result, "error_code", "")) != "0":
        raise ValueError("provider status")
    fields = result.fields
    if not isinstance(fields, list) or len(fields) != len(set(fields)):
        raise ValueError("provider fields")
    rows = []
    while result.next():
        values = result.get_row_data()
        if len(values) != len(fields) or len(rows) >= 10:
            raise ValueError("provider row shape")
        rows.append(dict(zip(fields, values, strict=True)))
    if str(getattr(result, "error_code", "")) != "0":
        raise ValueError("provider iteration")
    return rows


def collect_financials(module, symbol: str, knowledge_time: datetime) -> dict:
    """Collect complete corresponding periods, never silently use stale rows."""
    code, exchange = symbol.split(".")
    if exchange == "BJ":
        return {"error": "FINANCIAL_EXCHANGE_UNSUPPORTED"}
    external = f"{exchange.lower()}.{code}"
    last = None
    # Probe newer reports too: the reporting deadline is only a floor.
    known_date = knowledge_time.astimezone(_SHANGHAI).date()
    for period in _periods(known_date):
        rows = _rows(module.query_profit_data(
            code=external, year=period.year, quarter=(period.month + 2) // 3
        ))
        if not rows:
            continue
        valid = [r for r in rows if r.get("pubDate") and r["pubDate"] <= known_date.isoformat()]
        if not valid:
            continue
        if len(valid) != 1:
            return {"error": "FINANCIAL_DUPLICATE_PERIOD"}
        last = (period, valid[0])
        break
    if last is None:
        return {"error": "LATEST_REQUIRED_FINANCIAL_REPORT_UNAVAILABLE"}
    period, profit = last
    quarter = (period.month + 2) // 3
    balance = _rows(module.query_balance_data(code=external, year=period.year, quarter=quarter))
    cash = _rows(module.query_cash_flow_data(code=external, year=period.year, quarter=quarter))
    previous = _rows(module.query_profit_data(code=external, year=period.year - 1, quarter=quarter))
    return {
        "provider": "baostock", "source_url": "https://www.baostock.com/",
        "reporting_window_source_url": REPORTING_WINDOW_SOURCE,
        "required_period": required_report_period(known_date).isoformat(),
        "selected_period": period.isoformat(),
        "profit": profit, "balance": balance, "cash_flow": cash,
        "prior_year_profit": previous,
    }


def collect_execution_status(module, symbol: str, cutoff: date) -> dict:
    code, exchange = symbol.split(".")
    if exchange == "BJ":
        return {"error": "EXECUTION_EXCHANGE_UNSUPPORTED"}
    external = f"{exchange.lower()}.{code}"
    rows = _rows(module.query_history_k_data_plus(
        external, "date,code,tradestatus,isST", start_date=cutoff.isoformat(),
        end_date=cutoff.isoformat(), frequency="d", adjustflag="3",
    ))
    return {
        "provider": "baostock", "source_url": "https://www.baostock.com/",
        "cutoff": cutoff.isoformat(), "rows": rows, "content_hash": content_hash(rows),
    }


def _json_response(response, *, limit=8_000_000):
    response.raise_for_status()
    if len(response.content) > limit:
        raise ValueError("provider response too large")
    return response.json()


def collect_announcements(client, symbol: str, knowledge_time: datetime, identities: dict) -> dict:
    """Enumerate every item in a one-year public disclosure window, or fail closed."""
    code = symbol.split(".")[0]
    identity = identities.get(code)
    if identity is None:
        return {"error": "OFFICIAL_ISSUER_IDENTITY_UNAVAILABLE"}
    known_date = knowledge_time.astimezone(_SHANGHAI).date()
    start = announcement_window_start(known_date)
    items, observed_ids = [], set()
    expected_total = None
    complete = False
    for page in range(1, MAX_ANNOUNCEMENT_PAGES + 1):
        body = _json_response(client.post(
            "https://www.cninfo.com.cn/new/hisAnnouncement/query",
            data={
                "stock": f"{code},{identity}", "column": "szse", "tabName": "fulltext",
                "pageSize": str(PAGE_SIZE), "pageNum": str(page),
                "seDate": f"{start.isoformat()}~{known_date.isoformat()}",
                "searchkey": "", "plate": "", "category": "", "sortName": "time",
                "sortType": "desc", "isHLtitle": "true",
            },
        ))
        total = body.get("totalAnnouncement")
        raw = body.get("announcements") or []
        if isinstance(total, bool) or not isinstance(total, int) or total < 0 or not isinstance(raw, list):
            return {"error": "OFFICIAL_MANIFEST_SCHEMA_INVALID"}
        if expected_total is not None and expected_total != total:
            return {"error": "OFFICIAL_MANIFEST_CHANGED_DURING_READ"}
        expected_total = total
        for row in raw:
            item_id = str(row.get("announcementId", ""))
            if not item_id or item_id in observed_ids or row.get("secCode") != code:
                return {"error": "OFFICIAL_MANIFEST_IDENTITY_OR_PAGINATION_INVALID"}
            observed_ids.add(item_id)
            published = datetime.fromtimestamp(int(row["announcementTime"]) / 1000, UTC)
            if published > knowledge_time:
                # The manifest query is date-granular; never use later same-day announcements.
                continue
            if published.astimezone(_SHANGHAI).date() < start:
                return {"error": "OFFICIAL_MANIFEST_OUT_OF_WINDOW"}
            relative = str(row.get("adjunctUrl", ""))
            url = "https://static.cninfo.com.cn/" + relative.lstrip("/")
            if urlparse(url).hostname != "static.cninfo.com.cn" or not relative.endswith(".PDF") and not relative.endswith(".pdf"):
                return {"error": "OFFICIAL_DOCUMENT_URL_UNSUPPORTED"}
            items.append({
                "announcement_id": item_id,
                "title": str(row.get("announcementTitle", "")),
                "published_at": published.isoformat(), "url": url,
            })
        if body.get("hasMore") is False:
            complete = len(observed_ids) == total
            break
        if not raw or len(observed_ids) > total:
            break
    items.sort(key=lambda item: (item["published_at"], item["announcement_id"]))
    return {
        "provider": "cninfo", "source_url": "https://www.cninfo.com.cn/new/hisAnnouncement/query",
        "coverage_from": start.isoformat(), "coverage_through": knowledge_time.isoformat(),
        "complete": complete, "record_count": len(items), "total_provider_records": expected_total,
        "items": items, "content_hash": content_hash(items),
        "error": None if complete else "OFFICIAL_MANIFEST_TRUNCATED",
    }


def collect_worker_batch(
    symbols, *, cutoff, knowledge_time, module, client, skip_financial_symbols=(), emit=None,
):
    """Finish financial/execution receipts for all candidates before disclosures.

    The parent deadline is still authoritative. Started-phase receipts permit
    the parent to rotate retries past an actually attempted slow symbol, while
    successful financial snapshots can be reused independently of an outage in
    CNINFO. No announcement failure discards the other candidates' financials.
    """
    emit = emit or (lambda value: print(json.dumps(value, ensure_ascii=False), flush=True))
    results = {}
    for symbol in symbols:
        result = {
            "symbol": symbol, "method_version": METHOD_VERSION,
            "announcements": {"error": "OFFICIAL_MANIFEST_PENDING"},
            "retrieved_at": datetime.now(UTC).isoformat(),
            "financial_attempted": symbol not in skip_financial_symbols,
            "announcement_attempted": False,
        }
        if symbol not in skip_financial_symbols:
            result["financials"] = {"error": "FINANCIAL_PROVIDER_PENDING"}
            result["execution"] = {"error": "EXECUTION_PROVIDER_PENDING"}
            emit(result.copy())
            try:
                if module is None:
                    raise ValueError("financial provider unavailable")
                with redirect_stdout(StringIO()):
                    result["financials"] = collect_financials(module, symbol, knowledge_time)
            except Exception:
                result["financials"] = {"error": "FINANCIAL_PROVIDER_UNAVAILABLE"}
            try:
                if module is None:
                    raise ValueError("execution provider unavailable")
                with redirect_stdout(StringIO()):
                    result["execution"] = collect_execution_status(module, symbol, cutoff)
            except Exception:
                result["execution"] = {"error": "EXECUTION_PROVIDER_UNAVAILABLE"}
            result["retrieved_at"] = datetime.now(UTC).isoformat()
            result["financial_retrieved_at"] = result["retrieved_at"]
        results[symbol] = result
        emit(result.copy())
    try:
        master = _json_response(client.get("https://www.cninfo.com.cn/new/data/szse_stock.json"))
        identities = {str(row["code"]): str(row["orgId"]) for row in master["stockList"]}
    except Exception:
        identities = {}
    for symbol in symbols:
        result = results[symbol]
        result["announcement_attempted"] = True
        result["retrieved_at"] = datetime.now(UTC).isoformat()
        emit(result.copy())
        try:
            result["announcements"] = collect_announcements(client, symbol, knowledge_time, identities)
        except Exception:
            result["announcements"] = {"error": "OFFICIAL_MANIFEST_UNAVAILABLE"}
        result["retrieved_at"] = datetime.now(UTC).isoformat()
        emit(result.copy())


def _worker():
    request = json.loads(sys.stdin.read())
    symbols = _validate_request(
        request["symbols"], date.fromisoformat(request["cutoff"]),
        datetime.fromisoformat(request["knowledge_time"]), 75,
    )
    knowledge_time = datetime.fromisoformat(request["knowledge_time"])
    cutoff = date.fromisoformat(request["cutoff"])
    skip = tuple(request.get("skip_financial_symbols", ()))
    if not set(skip) <= set(symbols):
        raise ValueError("invalid public reuse symbols")
    import httpx

    try:
        import baostock
        with redirect_stdout(StringIO()):
            if len(skip) != len(symbols) and str(baostock.login().error_code) != "0":
                baostock = None
    except Exception:
        baostock = None
    with httpx.Client(timeout=4.0, follow_redirects=False, headers={
        "User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"
    }) as client:
        collect_worker_batch(symbols, cutoff=cutoff, knowledge_time=knowledge_time, module=baostock, client=client, skip_financial_symbols=skip)
    # No logout: process teardown closes the socket without an unbounded SDK call.
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker())
