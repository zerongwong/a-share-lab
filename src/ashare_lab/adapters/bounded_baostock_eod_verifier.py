"""Bounded BaoStock cross-check for a canonical Tushare EOD batch.

This adapter is deliberately verification-only.  It applies the same
deterministic symbol sample, comparison fields, units and tolerances as the
default historical-sample path in :mod:`akshare_eod_verifier`, but performs
all BaoStock network I/O in a disposable child process with a hard deadline.
It never repairs or merges values into the canonical Tushare batch.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from ashare_lab.adapters.akshare_eod_verifier import (
    AKShareEodVerificationResult,
    AKShareEvidenceMode,
    AKShareVerificationStatus,
    AKShareVerificationTolerance,
    _comparison_result,
    _deterministic_sample,
    _validate_bar_frame,
    _validate_tushare_batch,
)
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError

_SYMBOL = re.compile(r"^[0-9]{6}$")
_METHOD_VERSION = "baostock-unadjusted-daily-sample-v1"
_MAX_SAMPLE_SIZE = 64
_MAX_RESPONSE_BYTES = 262_144
_RESULT_KEYS = {
    "provider",
    "method_version",
    "target_date",
    "retrieved_at",
    "rows",
}
_ROW_KEYS = {
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume_shares",
    "amount_cny",
}
_COMPARISON_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume_shares",
    "amount_cny",
)


class BoundedBaoStockEodVerifier:
    """Independently verify a deterministic stock sample through BaoStock."""

    provider = "baostock"
    method_version = _METHOD_VERSION

    def __init__(
        self,
        *,
        sample_size: int = 8,
        tolerance: AKShareVerificationTolerance | None = None,
        clock=None,
        timeout: float = 45.0,
        runner=subprocess.run,
    ) -> None:
        if not 1 <= sample_size <= _MAX_SAMPLE_SIZE:
            raise ValueError(f"sample_size must be in [1, {_MAX_SAMPLE_SIZE}]")
        if not 0 < timeout <= 120:
            raise ValueError("BaoStock verification deadline must be in (0, 120] seconds")
        self._sample_size = sample_size
        self._tolerance = tolerance or AKShareVerificationTolerance()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._timeout = timeout
        self._runner = runner

    def verify(
        self,
        tushare_batch: pd.DataFrame,
        target_date: date,
    ) -> AKShareEodVerificationResult:
        """Return the existing verification receipt shape for compatibility."""

        if not isinstance(target_date, date) or isinstance(target_date, datetime):
            raise TypeError("target_date must be a date")
        try:
            canonical = _validate_tushare_batch(tushare_batch, target_date)
        except (DataQualityError, DataUnavailableError, TypeError, ValueError):
            return _unavailable(target_date, _safe_length(tushare_batch), "TUSHARE_BATCH_UNUSABLE")

        symbols = _deterministic_sample(
            tuple(canonical["symbol"]),
            target_date,
            self._sample_size,
        )
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise DataQualityError("BaoStock核验时钟必须包含时区。")
        try:
            evidence, retrieved_at = self._read_evidence(
                symbols=symbols,
                target_date=target_date,
                now=now,
            )
        except DataUnavailableError:
            return _unavailable(
                target_date,
                len(canonical),
                "BAOSTOCK_HISTORICAL_SAMPLE_UNAVAILABLE",
            )

        return _comparison_result(
            canonical.loc[canonical["symbol"].isin(symbols)].copy(),
            evidence,
            target_date=target_date,
            mode=AKShareEvidenceMode.HISTORICAL_SAMPLE,
            tolerance=self._tolerance,
            evidence_retrieved_at=retrieved_at,
            compared_fields=_COMPARISON_FIELDS,
        )

    def verify_stock_frame(
        self,
        frame: pd.DataFrame,
        target_date: date,
        requested_symbols: Sequence[str],
    ) -> AKShareEodVerificationResult:
        """Strict daily-sync entrypoint; every non-verified result blocks use."""

        requested = tuple(requested_symbols)
        if not requested:
            raise DataUnavailableError("BaoStock核验没有收到目标股票代码。")
        if any(
            not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None for symbol in requested
        ):
            raise DataQualityError("BaoStock核验目标必须是精确的六位字符串股票代码。")
        if len(set(requested)) != len(requested):
            raise DataQualityError("BaoStock核验目标股票代码重复。")
        if not isinstance(frame, pd.DataFrame) or "symbol" not in frame.columns:
            raise DataUnavailableError("Tushare股票批次缺少可核验的证券身份。")
        frame_symbols = tuple(frame["symbol"].tolist())
        if set(frame_symbols) != set(requested) or len(frame_symbols) != len(requested):
            raise DataQualityError("Tushare股票批次与请求身份集合不一致。")

        result = self.verify(frame, target_date)
        if result.status is AKShareVerificationStatus.VERIFIED:
            return result
        if result.status is AKShareVerificationStatus.MISMATCH:
            fields = ", ".join(f"{item.symbol}:{item.field}" for item in result.mismatches[:8])
            raise DataQualityError(f"BaoStock交叉核验数值不一致：{fields or 'unknown'}")
        raise DataUnavailableError("BaoStock日线交叉核验证据不可用。")

    def _read_evidence(
        self,
        *,
        symbols: tuple[str, ...],
        target_date: date,
        now: datetime,
    ) -> tuple[pd.DataFrame, datetime]:
        request = {
            "operation": "verify_stock_sample",
            "target_date": target_date.isoformat(),
            "now": now.isoformat(),
            "symbols": symbols,
        }
        try:
            result = self._runner(
                [sys.executable, "-m", "ashare_lab.cli.baostock_eod_verify"],
                input=json.dumps(request),
                text=True,
                capture_output=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise DataUnavailableError("BaoStock日线核验请求超过时限，隔离进程已停止。") from None
        except OSError:
            raise DataUnavailableError("BaoStock日线核验隔离进程不可用。") from None

        stdout = getattr(result, "stdout", None)
        if not isinstance(stdout, str):
            raise DataUnavailableError("BaoStock日线核验隔离响应不可用。")
        if len(stdout.encode("utf-8")) > _MAX_RESPONSE_BYTES:
            raise DataQualityError("BaoStock日线核验响应异常过大。")
        try:
            payload = json.loads(stdout)
        except (TypeError, ValueError):
            raise DataUnavailableError("BaoStock日线核验隔离响应不可用。") from None
        if not isinstance(payload, dict) or set(payload) - {"result", "error"}:
            raise DataUnavailableError("BaoStock日线核验隔离响应结构无效。")
        if payload.get("error") == "quality":
            raise DataQualityError("BaoStock日线核验数据质量未通过，禁止切源掩盖。")
        if getattr(result, "returncode", 1) or payload.get("error"):
            raise DataUnavailableError("BaoStock日线核验数据请求不可用。")
        document = payload.get("result")
        return _normalize_document(
            document,
            symbols=symbols,
            target_date=target_date,
            now=now,
            tolerance=self._tolerance,
        )


class UnavailableOnlyEodVerifierFallback:
    """Use backup evidence only when the primary verifier is unavailable.

    A numeric mismatch, malformed evidence, or any other quality failure from
    the primary verifier is conclusive and must never be hidden by a second
    provider.
    """

    def __init__(self, *, primary: object, backup: object) -> None:
        for label, value in (("primary", primary), ("backup", backup)):
            if not callable(getattr(value, "verify_stock_frame", None)):
                raise TypeError(f"{label} verifier must expose verify_stock_frame")
        self._primary = primary
        self._backup = backup
        self.last_evidence = ""

    def verify_stock_frame(
        self,
        frame: pd.DataFrame,
        target_date: date,
        requested_symbols: Sequence[str],
    ) -> object:
        try:
            result = self._primary.verify_stock_frame(frame, target_date, requested_symbols)
            source = self._primary
        except DataUnavailableError:
            result = self._backup.verify_stock_frame(frame, target_date, requested_symbols)
            source = self._backup
        provider = getattr(source, "provider", type(source).__name__)
        method = getattr(source, "method_version", "historical-sample")
        mode = getattr(getattr(result, "mode", None), "value", "verified")
        self.last_evidence = (
            f"{provider}:{method}:{mode}:target={target_date.isoformat()}:"
            f"sample={len(getattr(result, 'compared_symbols', ()))}"
        )
        return result


def _normalize_document(
    document: object,
    *,
    symbols: tuple[str, ...],
    target_date: date,
    now: datetime,
    tolerance: AKShareVerificationTolerance,
) -> tuple[pd.DataFrame, datetime]:
    if not isinstance(document, dict) or set(document) != _RESULT_KEYS:
        raise DataQualityError("BaoStock日线核验回执不符合合同。")
    if (
        document.get("provider") != "baostock"
        or document.get("method_version") != _METHOD_VERSION
        or document.get("target_date") != target_date.isoformat()
    ):
        raise DataQualityError("BaoStock日线核验回执来源或日期不一致。")
    try:
        retrieved_at = datetime.fromisoformat(document["retrieved_at"])
    except (KeyError, TypeError, ValueError):
        raise DataQualityError("BaoStock日线核验回执抓取时间无效。") from None
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None or retrieved_at != now:
        raise DataQualityError("BaoStock日线核验回执抓取时间不一致。")

    rows = document.get("rows")
    if not isinstance(rows, list) or len(rows) != len(symbols):
        raise DataQualityError("BaoStock日线核验回执行数不一致。")
    normalized: list[dict[str, object]] = []
    for raw in rows:
        if not isinstance(raw, dict) or set(raw) != _ROW_KEYS:
            raise DataQualityError("BaoStock日线核验回执行结构无效。")
        symbol = raw.get("symbol")
        if not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None:
            raise DataQualityError("BaoStock日线核验回执证券身份无效。")
        if raw.get("trade_date") != target_date.isoformat():
            raise DataQualityError("BaoStock日线核验回执交易日期不一致。")
        row: dict[str, object] = {"symbol": symbol, "trade_date": target_date}
        for field in (
            "open",
            "high",
            "low",
            "close",
            "prev_close",
            "volume_shares",
            "amount_cny",
        ):
            value = raw.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DataQualityError("BaoStock日线核验回执包含非数值行情字段。")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise DataQualityError("BaoStock日线核验回执包含非有限行情字段。")
            row[field] = numeric
        normalized.append(row)

    evidence = pd.DataFrame(normalized)
    evidence_symbols = tuple(evidence["symbol"])
    if tuple(sorted(evidence_symbols)) != tuple(sorted(symbols)) or len(
        set(evidence_symbols)
    ) != len(evidence_symbols):
        raise DataQualityError("BaoStock日线核验回执证券身份集合不一致。")
    _validate_bar_frame(
        evidence,
        unit_band=tolerance.implied_price_band_relative,
        provider="BaoStock history",
    )
    return evidence.sort_values("symbol").reset_index(drop=True), retrieved_at


def _unavailable(
    target_date: date,
    row_count: int,
    reason: str,
) -> AKShareEodVerificationResult:
    return AKShareEodVerificationResult(
        status=AKShareVerificationStatus.UNAVAILABLE,
        target_date=target_date,
        mode=AKShareEvidenceMode.NONE,
        tushare_row_count=row_count,
        compared_symbols=(),
        unavailable_reasons=(reason,),
    )


def _safe_length(value: Any) -> int:
    return len(value) if isinstance(value, pd.DataFrame) else 0


__all__ = [
    "BoundedBaoStockEodVerifier",
    "UnavailableOnlyEodVerifierFallback",
]
