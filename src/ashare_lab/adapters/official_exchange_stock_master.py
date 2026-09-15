"""Bounded, independently validated SSE/SZSE current A-share master."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import UTC, datetime

from ashare_lab.domain.errors import DataQualityError, DataUnavailableError

_SYMBOL = re.compile(r"^(?:6\d{5}\.SH|(?:00|30)\d{4}\.SZ)$")
_MAX_RESPONSE_BYTES = 1_000_000
_MIN_CURRENT_STOCK_MASTER_SIZE = 5_000


class BoundedOfficialExchangeStockMaster:
    """Read the two exchanges' current lists as corroborating evidence.

    The broad board-count checks protect the child protocol, but do not by
    themselves prove whole-market completeness.  The production orchestration
    must additionally compare this result with BaoStock or a recent verified
    master before publication.
    """

    provider = "sse_szse_official_via_akshare"

    def __init__(self, *, expected_symbols=None, clock=None, timeout=60.0, runner=subprocess.run):
        if not 0 < timeout <= 120:
            raise ValueError("official stock-master deadline must be in (0, 120]")
        self._expected = None if expected_symbols is None else self._normalize(expected_symbols)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._timeout = timeout
        self._runner = runner
        self.last_evidence = ""

    def fetch_cn_stock_symbols(self):
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise DataQualityError("official stock-master clock is not timezone aware")
        try:
            result = self._runner(
                [sys.executable, "-m", "ashare_lab.cli.exchange_stock_master_read"],
                input=json.dumps({"operation": "listed_a_share_symbols", "now": now.isoformat()}),
                text=True,
                capture_output=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise DataUnavailableError("上交所/深交所证券主表请求超过时限。") from None
        except OSError:
            raise DataUnavailableError("上交所/深交所证券主表隔离进程不可用。") from None
        if not isinstance(result.stdout, str):
            raise DataUnavailableError("上交所/深交所证券主表响应不可用。")
        if len(result.stdout.encode("utf-8")) > _MAX_RESPONSE_BYTES:
            raise DataQualityError("上交所/深交所证券主表响应异常过大。")
        try:
            payload = json.loads(result.stdout)
        except (TypeError, ValueError):
            raise DataUnavailableError("上交所/深交所证券主表响应不可用。") from None
        if not isinstance(payload, dict):
            raise DataUnavailableError("上交所/深交所证券主表响应不可用。")
        if payload.get("error") == "quality":
            raise DataQualityError("上交所/深交所证券主表质量校验失败。")
        if result.returncode or payload.get("error"):
            raise DataUnavailableError("上交所/深交所证券主表网络不可用。")
        document = payload.get("result")
        if not isinstance(document, dict):
            raise DataQualityError("上交所/深交所证券主表协议无效。")
        symbols = self._normalize(document.get("symbols"))
        counts = document.get("counts")
        try:
            retrieved_at = datetime.fromisoformat(document["retrieved_at"])
        except (KeyError, TypeError, ValueError):
            retrieved_at = None
        if (
            document.get("provider") != self.provider
            or document.get("method_version") != "exchange-listed-a-v1"
            or retrieved_at is None
            or retrieved_at.tzinfo is None
            or retrieved_at.utcoffset() is None
            or retrieved_at != now
            or not isinstance(counts, dict)
            or any(type(counts.get(key)) is not int for key in ("sse_main", "sse_star", "szse_a"))
            or sum(counts[key] for key in ("sse_main", "sse_star", "szse_a")) != len(symbols)
            or set(counts) != {"sse_main", "sse_star", "szse_a"}
            or not (1_000 <= counts["sse_main"] <= 2_500)
            or not (300 <= counts["sse_star"] <= 1_500)
            or not (2_000 <= counts["szse_a"] <= 4_000)
        ):
            raise DataQualityError("上交所/深交所证券主表回执不符合合同。")
        if self._expected is not None:
            drift = len(set(symbols).symmetric_difference(self._expected))
            allowed = max(50, math.ceil(len(self._expected) * 0.005))
            if drift > allowed:
                raise DataQualityError("上交所/深交所证券主表相对最近已验证名单漂移过大。")
        digest = hashlib.sha256("|".join(symbols).encode()).hexdigest()
        self.last_evidence = (
            f"{self.provider}:main={counts['sse_main']};star={counts['sse_star']};"
            f"sz={counts['szse_a']};sha256={digest};method=exchange-listed-a-v1;"
            f"retrieved_at={retrieved_at.astimezone(UTC).isoformat()};"
            "endpoints=sse_name_code+szse_a_list"
        )
        return symbols

    @staticmethod
    def _normalize(values):
        if not isinstance(values, (list, tuple)) or not values:
            raise DataQualityError("上交所/深交所证券主表为空。")
        symbols = tuple(values)
        if (
            any(not isinstance(value, str) or _SYMBOL.fullmatch(value) is None for value in symbols)
            or tuple(sorted(symbols)) != symbols
            or len(set(symbols)) != len(symbols)
            or not _MIN_CURRENT_STOCK_MASTER_SIZE <= len(symbols) < 6_000
        ):
            raise DataQualityError("上交所/深交所证券主表身份、排序或数量无效。")
        return symbols
