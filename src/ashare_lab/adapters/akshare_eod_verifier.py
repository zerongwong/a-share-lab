"""Fail-closed AKShare cross-checks for a canonical Tushare EOD batch.

AKShare's current full-market snapshot does not carry an authoritative trade
date.  Consequently this module accepts snapshot data only through an
explicit :class:`AKShareSnapshotEvidence` envelope.  Callers that cannot prove
the snapshot date should omit the snapshot fetcher; the verifier then uses a
small, deterministic set of per-symbol historical requests whose rows contain
their own dates.

The verifier is deliberately not a market-data source.  It never merges or
repairs the Tushare batch, and a failed cross-check cannot silently replace a
Tushare value with an AKShare value.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from types import ModuleType
from typing import Any

import pandas as pd

from ashare_lab.domain.errors import DataQualityError, DataUnavailableError

_SYMBOL = re.compile(r"^[0-9]{6}$")
_TUSHARE_REQUIRED_COLUMNS = (
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume_shares",
    "amount_cny",
    "source",
)
_SNAPSHOT_ALIASES: Mapping[str, tuple[str, ...]] = {
    "symbol": ("代码", "symbol"),
    "trade_date": ("交易日期", "日期", "trade_date"),
    "open": ("今开", "开盘", "open"),
    "high": ("最高", "high"),
    "low": ("最低", "low"),
    "close": ("最新价", "收盘", "close"),
    "prev_close": ("昨收", "前收盘", "prev_close"),
    "volume_lots": ("成交量", "volume_lots"),
    "amount_cny": ("成交额", "amount_cny"),
}
_HISTORY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "symbol": ("代码", "symbol"),
    "trade_date": ("日期", "交易日期", "trade_date"),
    "open": ("开盘", "open"),
    "high": ("最高", "high"),
    "low": ("最低", "low"),
    "close": ("收盘", "close"),
    "prev_close": ("昨收", "前收盘", "prev_close"),
    "volume_lots": ("成交量", "volume_lots"),
    "amount_cny": ("成交额", "amount_cny"),
}


class AKShareVerificationStatus(StrEnum):
    """Outcome of an independent AKShare evidence check."""

    VERIFIED = "VERIFIED"
    MISMATCH = "MISMATCH"
    UNAVAILABLE = "UNAVAILABLE"


class AKShareEvidenceMode(StrEnum):
    """Evidence path used for a verification result."""

    SNAPSHOT = "SNAPSHOT"
    HISTORICAL_SAMPLE = "HISTORICAL_SAMPLE"
    NONE = "NONE"


@dataclass(frozen=True, slots=True)
class AKShareVerificationTolerance:
    """Explicit comparison and unit-sanity tolerances.

    Price fields are quoted in CNY, volume in shares, and amount in CNY.  The
    absolute amount allowance accommodates Tushare's published daily amount
    precision after conversion from thousands of CNY.
    """

    price_absolute_cny: float = 0.011
    price_relative: float = 0.0005
    volume_absolute_shares: float = 100.0
    volume_relative: float = 0.002
    amount_absolute_cny: float = 1_000.0
    amount_relative: float = 0.01
    implied_price_band_relative: float = 0.01

    def __post_init__(self) -> None:
        for name, value in (
            ("price_absolute_cny", self.price_absolute_cny),
            ("price_relative", self.price_relative),
            ("volume_absolute_shares", self.volume_absolute_shares),
            ("volume_relative", self.volume_relative),
            ("amount_absolute_cny", self.amount_absolute_cny),
            ("amount_relative", self.amount_relative),
            ("implied_price_band_relative", self.implied_price_band_relative),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.price_relative > 0.02:
            raise ValueError("price_relative cannot exceed 2%")
        if self.volume_relative > 0.10 or self.amount_relative > 0.10:
            raise ValueError("volume and amount relative tolerances cannot exceed 10%")
        if self.implied_price_band_relative > 0.05:
            raise ValueError("implied_price_band_relative cannot exceed 5%")


@dataclass(frozen=True, slots=True)
class AKShareSnapshotEvidence:
    """A full-market snapshot plus an independently established trade date.

    ``stock_zh_a_spot_em`` itself does not return a reliable trade-date field.
    The component that creates this envelope must therefore establish
    ``trade_date`` separately and retain that evidence in its own audit trail.
    An aware retrieval timestamp is required so stale or future evidence is not
    accidentally presented as same-session verification.
    """

    frame: pd.DataFrame
    trade_date: date
    retrieved_at: datetime
    date_evidence: str
    endpoint: str = "stock_zh_a_spot_em"


@dataclass(frozen=True, slots=True)
class AKShareFieldMismatch:
    """One field outside the declared cross-provider tolerance."""

    symbol: str
    field: str
    tushare_value: float
    akshare_value: float
    absolute_difference: float
    allowed_difference: float


@dataclass(frozen=True, slots=True)
class AKShareEodVerificationResult:
    """Auditable result; only ``VERIFIED`` may pass a caller's quality gate."""

    status: AKShareVerificationStatus
    target_date: date
    mode: AKShareEvidenceMode
    tushare_row_count: int
    compared_symbols: tuple[str, ...]
    mismatches: tuple[AKShareFieldMismatch, ...] = ()
    unavailable_reasons: tuple[str, ...] = ()
    evidence_trade_date: date | None = None
    evidence_retrieved_at: datetime | None = None

    @property
    def is_verified(self) -> bool:
        return self.status is AKShareVerificationStatus.VERIFIED

    @property
    def compared_count(self) -> int:
        return len(self.compared_symbols)

    def require_verified(self) -> None:
        """Raise a typed gate error instead of letting a caller ignore status."""

        if self.status is AKShareVerificationStatus.VERIFIED:
            return
        if self.status is AKShareVerificationStatus.MISMATCH:
            fields = ", ".join(
                f"{item.symbol}:{item.field}" for item in self.mismatches[:8]
            )
            raise DataQualityError(f"AKShare交叉核验数值不一致：{fields or 'unknown'}")
        reason = "; ".join(self.unavailable_reasons) or "AKShare evidence unavailable"
        raise DataUnavailableError(reason)


SnapshotFetcher = Callable[[], AKShareSnapshotEvidence]
HistoryFetcher = Callable[[str, date, date], pd.DataFrame]


def _load_akshare() -> ModuleType:
    """Import AKShare only when historical evidence is actually requested."""

    try:
        return importlib.import_module("akshare")
    except ImportError as exc:
        raise DataUnavailableError(
            "未安装AKShare；无法进行独立收盘数据核验。"
        ) from exc


class AKShareEodVerifier:
    """Cross-check one canonical, unadjusted Tushare stock cross section.

    A dated full-market snapshot is preferred when supplied.  Otherwise a
    deterministic symbol sample is checked with AKShare's historical endpoint.
    The class is safe to construct when AKShare is not installed; its import is
    deferred until :meth:`verify` needs the default historical fetcher.
    """

    def __init__(
        self,
        *,
        sample_size: int = 8,
        tolerance: AKShareVerificationTolerance | None = None,
        snapshot_fetcher: SnapshotFetcher | None = None,
        history_fetcher: HistoryFetcher | None = None,
        module_loader: Callable[[], object] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if sample_size < 1:
            raise ValueError("sample_size must be at least one")
        self._sample_size = sample_size
        self._tolerance = tolerance or AKShareVerificationTolerance()
        self._snapshot_fetcher = snapshot_fetcher
        self._history_fetcher = history_fetcher
        self._module_loader = module_loader or _load_akshare
        self._clock = clock or (lambda: datetime.now(UTC))
        self._module: object | None = None

    def verify(
        self,
        tushare_batch: pd.DataFrame,
        target_date: date,
    ) -> AKShareEodVerificationResult:
        """Verify identity, date, OHLC, previous close, volume and amount.

        The method always returns an explicit result.  Provider failures,
        ambiguous identity, missing dates, and unknown units are
        ``UNAVAILABLE``; proven numeric disagreements are ``MISMATCH``.
        """

        if not isinstance(target_date, date) or isinstance(target_date, datetime):
            raise TypeError("target_date must be a date")
        try:
            canonical = _validate_tushare_batch(tushare_batch, target_date)
        except (DataQualityError, DataUnavailableError, TypeError, ValueError) as exc:
            return _unavailable_result(
                target_date,
                len(tushare_batch) if isinstance(tushare_batch, pd.DataFrame) else 0,
                f"TUSHARE_BATCH_UNUSABLE: {type(exc).__name__}: {exc}",
            )

        snapshot_rejection: str | None = None
        if self._snapshot_fetcher is not None:
            try:
                evidence = self._snapshot_fetcher()
                comparison = self._verify_snapshot(canonical, target_date, evidence)
                if comparison is not None:
                    return comparison
            except (DataQualityError, DataUnavailableError, TypeError, ValueError) as exc:
                snapshot_rejection = (
                    f"SNAPSHOT_UNUSABLE: {type(exc).__name__}: {str(exc)[:240]}"
                )
            except Exception as exc:  # noqa: BLE001 - normalize provider boundary
                snapshot_rejection = f"SNAPSHOT_FETCH_FAILED: {type(exc).__name__}"

        try:
            return self._verify_historical_sample(
                canonical,
                target_date,
                prior_reason=snapshot_rejection,
            )
        except (DataQualityError, DataUnavailableError, TypeError, ValueError) as exc:
            reasons = tuple(
                value
                for value in (
                    snapshot_rejection,
                    f"HISTORICAL_SAMPLE_UNUSABLE: {type(exc).__name__}: {str(exc)[:240]}",
                )
                if value
            )
            return _unavailable_result(
                target_date,
                len(canonical),
                *reasons,
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider boundary
            reasons = tuple(
                value
                for value in (
                    snapshot_rejection,
                    f"HISTORICAL_SAMPLE_FETCH_FAILED: {type(exc).__name__}",
                )
                if value
            )
            return _unavailable_result(target_date, len(canonical), *reasons)

    def verify_stock_frame(
        self,
        frame: pd.DataFrame,
        target_date: date,
        requested_symbols: Sequence[str],
    ) -> AKShareEodVerificationResult:
        """Strict integration entrypoint for the daily-sync quality gate.

        ``requested_symbols`` must be the exact canonical identities requested
        from Tushare.  Unlike :meth:`verify`, this method raises on every
        non-verified result so orchestration code cannot accidentally publish
        after forgetting to inspect a status field.
        """

        requested = tuple(requested_symbols)
        if not requested:
            raise DataUnavailableError("AKShare核验没有收到目标股票代码。")
        if any(not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None for symbol in requested):
            raise DataQualityError("AKShare核验目标必须是精确的六位字符串股票代码。")
        if len(set(requested)) != len(requested):
            raise DataQualityError("AKShare核验目标股票代码重复。")
        if not isinstance(frame, pd.DataFrame) or "symbol" not in frame.columns:
            raise DataUnavailableError("Tushare股票批次缺少可核验的证券身份。")
        frame_symbols = tuple(frame["symbol"].tolist())
        if set(frame_symbols) != set(requested) or len(frame_symbols) != len(requested):
            raise DataQualityError("Tushare股票批次与请求身份集合不一致。")

        result = self.verify(frame, target_date)
        result.require_verified()
        return result

    def _verify_snapshot(
        self,
        tushare: pd.DataFrame,
        target_date: date,
        evidence: AKShareSnapshotEvidence,
    ) -> AKShareEodVerificationResult | None:
        if not isinstance(evidence, AKShareSnapshotEvidence):
            raise DataUnavailableError("snapshot fetcher did not return dated evidence")
        if not isinstance(evidence.trade_date, date) or isinstance(
            evidence.trade_date, datetime
        ):
            raise DataUnavailableError("snapshot trade date is not an exact date")
        if evidence.trade_date != target_date:
            raise DataUnavailableError(
                "snapshot trade date does not equal the requested target date"
            )
        if evidence.retrieved_at.tzinfo is None or evidence.retrieved_at.utcoffset() is None:
            raise DataUnavailableError("snapshot retrieval time must be timezone-aware")
        if evidence.retrieved_at < datetime.combine(
            target_date,
            datetime.min.time(),
            tzinfo=UTC,
        ):
            raise DataUnavailableError("snapshot was retrieved before its claimed trade date")
        if not evidence.date_evidence.strip():
            raise DataUnavailableError("snapshot date evidence is missing")
        if evidence.endpoint != "stock_zh_a_spot_em":
            raise DataUnavailableError("snapshot endpoint identity is not recognized")

        akshare = _normalize_snapshot(
            evidence.frame,
            target_date=target_date,
            requested_symbols=tuple(tushare["symbol"]),
            unit_band=self._tolerance.implied_price_band_relative,
        )
        # A full-market snapshot is required to cover every target batch row.
        missing = sorted(set(tushare["symbol"]) - set(akshare["symbol"]))
        if missing:
            raise DataUnavailableError(
                "full-market snapshot is missing target identities: " + ",".join(missing[:8])
            )
        return _comparison_result(
            tushare,
            akshare,
            target_date=target_date,
            mode=AKShareEvidenceMode.SNAPSHOT,
            tolerance=self._tolerance,
            evidence_retrieved_at=evidence.retrieved_at,
        )

    def _verify_historical_sample(
        self,
        tushare: pd.DataFrame,
        target_date: date,
        *,
        prior_reason: str | None,
    ) -> AKShareEodVerificationResult:
        symbols = _deterministic_sample(
            tuple(tushare["symbol"]),
            target_date,
            self._sample_size,
        )
        rows: list[dict[str, Any]] = []
        start = target_date - timedelta(days=45)
        for symbol in symbols:
            raw = self._fetch_history(symbol, start, target_date)
            rows.append(
                _normalize_history_bar(
                    raw,
                    symbol=symbol,
                    target_date=target_date,
                    unit_band=self._tolerance.implied_price_band_relative,
                )
            )
        akshare = pd.DataFrame(rows)
        result = _comparison_result(
            tushare.loc[tushare["symbol"].isin(symbols)].copy(),
            akshare,
            target_date=target_date,
            mode=AKShareEvidenceMode.HISTORICAL_SAMPLE,
            tolerance=self._tolerance,
            evidence_retrieved_at=_aware_utc(self._clock()),
            # AKShare's historical endpoints do not publish the exchange's
            # official previous-close field.  Deriving it from the preceding
            # raw close is invalid on ex-right/ex-dividend sessions, when the
            # official pre-close is adjusted.  Keep deriving it for bar/unit
            # sanity, but do not present it as independent cross-source
            # evidence.  A dated full-market snapshot may still verify the
            # explicit previous-close field.
            compared_fields=(
                "open",
                "high",
                "low",
                "close",
                "volume_shares",
                "amount_cny",
            ),
        )
        # A rejected undated/stale snapshot is audit context, not a reason to
        # invalidate independently dated historical evidence.
        del prior_reason
        return result

    def _fetch_history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if self._history_fetcher is not None:
            return self._history_fetcher(symbol, start, end)
        if self._module is None:
            self._module = self._module_loader()
        eastmoney = getattr(self._module, "stock_zh_a_hist", None)
        if callable(eastmoney):
            try:
                return eastmoney(
                    symbol=symbol,
                    period="daily",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    adjust="",
                )
            except Exception:  # noqa: BLE001 - try an independent AKShare transport
                pass

        # Eastmoney occasionally closes the public connection before a
        # response.  AKShare's Sina adapter is an independent, still-free
        # transport.  Convert its documented shares/CNY fields into the same
        # internal history envelope; it remains verification-only and never
        # repairs Tushare values.
        sina = getattr(self._module, "stock_zh_a_daily", None)
        if not callable(sina):
            raise DataUnavailableError("当前AKShare版本没有可用的历史日线核验接口")
        provider_symbol = ("sh" if symbol.startswith("6") else "sz") + symbol
        try:
            raw = sina(
                symbol=provider_symbol,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="",
            )
        except Exception:
            raise DataUnavailableError("AKShare两个历史日线核验入口均不可用") from None
        return _convert_sina_history(raw, symbol=symbol)


def _convert_sina_history(raw: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise DataUnavailableError(f"AKShare Sina history is empty for {symbol}")
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    missing = required.difference(raw.columns)
    if missing:
        raise DataQualityError(
            "AKShare Sina history is missing fields: " + ",".join(sorted(missing))
        )
    result = pd.DataFrame(
        {
            "代码": symbol,
            "日期": raw["date"],
            "开盘": raw["open"],
            "最高": raw["high"],
            "最低": raw["low"],
            "收盘": raw["close"],
            # Sina exposes shares and CNY; the common AKShare history
            # normalizer expects lots and CNY.
            "成交量": pd.to_numeric(raw["volume"], errors="coerce") / 100.0,
            "成交额": raw["amount"],
        }
    )
    return result


def _validate_tushare_batch(frame: pd.DataFrame, target_date: date) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataUnavailableError("Tushare batch is empty")
    if isinstance(frame.columns, pd.MultiIndex):
        raise DataQualityError("Tushare batch has ambiguous multi-level columns")
    missing = [column for column in _TUSHARE_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataQualityError("Tushare batch is missing columns: " + ",".join(missing))

    result = frame.loc[:, list(_TUSHARE_REQUIRED_COLUMNS)].copy()
    if not result["symbol"].map(lambda value: isinstance(value, str)).all():
        raise DataQualityError("Tushare symbol identity must remain a six-character string")
    result["symbol"] = result["symbol"].str.strip()
    if not result["symbol"].str.fullmatch(_SYMBOL).all():
        raise DataQualityError("Tushare symbol identity is not an exact six-digit stock code")
    if result["symbol"].duplicated().any():
        raise DataQualityError("Tushare batch contains duplicate stock identities")
    if not result["source"].map(
        lambda value: isinstance(value, str) and value.strip().lower().startswith("tushare")
    ).all():
        raise DataQualityError("batch source is not explicitly identified as Tushare")

    parsed_dates = _exact_dates(result["trade_date"], "Tushare")
    if not all(value == target_date for value in parsed_dates):
        raise DataQualityError("Tushare batch date does not exactly equal target_date")
    result["trade_date"] = parsed_dates
    for field in (
        "open",
        "high",
        "low",
        "close",
        "prev_close",
        "volume_shares",
        "amount_cny",
    ):
        result[field] = _strict_numbers(result[field], f"Tushare {field}")
    _validate_bar_frame(result, unit_band=0.01, provider="Tushare")
    return result.sort_values("symbol").reset_index(drop=True)


def _normalize_snapshot(
    raw: pd.DataFrame,
    *,
    target_date: date,
    requested_symbols: tuple[str, ...],
    unit_band: float,
) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise DataUnavailableError("AKShare snapshot is empty")
    if isinstance(raw.columns, pd.MultiIndex):
        raise DataQualityError("AKShare snapshot has ambiguous multi-level columns")
    columns = {
        field: _find_column(raw, aliases, required=field != "trade_date")
        for field, aliases in _SNAPSHOT_ALIASES.items()
    }
    symbol_column = columns["symbol"]
    assert symbol_column is not None
    symbols = _strict_symbols(raw[symbol_column], "AKShare snapshot")
    if symbols.duplicated().any():
        raise DataQualityError("AKShare snapshot contains duplicate stock identities")
    selected = raw.loc[symbols.isin(requested_symbols)].copy()
    selected_symbols = symbols.loc[selected.index]
    if selected.empty:
        raise DataUnavailableError("AKShare snapshot shares no exact identity with target batch")

    date_column = columns["trade_date"]
    if date_column is not None:
        dates = _exact_dates(selected[date_column], "AKShare snapshot")
        if not all(value == target_date for value in dates):
            raise DataQualityError("AKShare snapshot row dates conflict with evidence envelope")

    result = pd.DataFrame({"symbol": selected_symbols.to_numpy()})
    for field in ("open", "high", "low", "close", "prev_close", "amount_cny"):
        column = columns[field]
        assert column is not None
        result[field] = _strict_numbers(selected[column], f"AKShare snapshot {field}").to_numpy()
    volume_column = columns["volume_lots"]
    assert volume_column is not None
    volume_lots = _strict_numbers(
        selected[volume_column], "AKShare snapshot volume_lots"
    )
    result["volume_shares"] = (volume_lots * 100.0).to_numpy()
    result["trade_date"] = target_date
    _validate_bar_frame(result, unit_band=unit_band, provider="AKShare snapshot")
    return result.sort_values("symbol").reset_index(drop=True)


def _normalize_history_bar(
    raw: pd.DataFrame,
    *,
    symbol: str,
    target_date: date,
    unit_band: float,
) -> dict[str, Any]:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise DataUnavailableError(f"AKShare history is empty for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raise DataQualityError(f"AKShare history has ambiguous columns for {symbol}")
    columns = {
        field: _find_column(
            raw,
            aliases,
            required=field not in {"symbol", "prev_close"},
        )
        for field, aliases in _HISTORY_ALIASES.items()
    }
    symbol_column = columns["symbol"]
    if symbol_column is not None:
        returned = _strict_symbols(raw[symbol_column], f"AKShare history {symbol}")
        if not (returned == symbol).all():
            raise DataQualityError(f"AKShare history identity conflicts for {symbol}")

    date_column = columns["trade_date"]
    assert date_column is not None
    dates = _exact_dates(raw[date_column], f"AKShare history {symbol}")
    if dates.duplicated().any():
        raise DataQualityError(f"AKShare history has duplicate dates for {symbol}")
    if any(value > target_date for value in dates):
        raise DataQualityError(f"AKShare history returned post-target rows for {symbol}")

    working = raw.copy()
    working["__date"] = dates
    working = working.sort_values("__date").reset_index(drop=True)
    target_indexes = working.index[working["__date"] == target_date].tolist()
    if len(target_indexes) != 1:
        raise DataUnavailableError(
            f"AKShare history has no unique target-date row for {symbol}"
        )
    target_index = target_indexes[0]
    if target_index == 0:
        raise DataUnavailableError(
            f"AKShare history cannot establish previous close for {symbol}"
        )

    target = working.loc[target_index]
    previous = working.loc[target_index - 1]
    result: dict[str, Any] = {"symbol": symbol, "trade_date": target_date}
    for field in ("open", "high", "low", "close", "amount_cny"):
        column = columns[field]
        assert column is not None
        result[field] = _strict_scalar(target[column], f"AKShare history {symbol} {field}")
    volume_column = columns["volume_lots"]
    assert volume_column is not None
    result["volume_shares"] = (
        _strict_scalar(
            target[volume_column], f"AKShare history {symbol} volume_lots"
        )
        * 100.0
    )
    close_column = columns["close"]
    assert close_column is not None
    derived_previous = _strict_scalar(
        previous[close_column], f"AKShare history {symbol} previous close"
    )
    explicit_previous_column = columns["prev_close"]
    if explicit_previous_column is not None:
        explicit_previous = _strict_scalar(
            target[explicit_previous_column],
            f"AKShare history {symbol} explicit previous close",
        )
        if not math.isclose(
            explicit_previous,
            derived_previous,
            rel_tol=0.0005,
            abs_tol=0.011,
        ):
            raise DataQualityError(
                f"AKShare history previous-close evidence conflicts for {symbol}"
            )
    result["prev_close"] = derived_previous
    normalized = pd.DataFrame([result])
    _validate_bar_frame(normalized, unit_band=unit_band, provider="AKShare history")
    return result


def _comparison_result(
    tushare: pd.DataFrame,
    akshare: pd.DataFrame,
    *,
    target_date: date,
    mode: AKShareEvidenceMode,
    tolerance: AKShareVerificationTolerance,
    evidence_retrieved_at: datetime,
    compared_fields: tuple[str, ...] | None = None,
) -> AKShareEodVerificationResult:
    left = tushare.set_index("symbol")
    right = akshare.set_index("symbol")
    if not left.index.is_unique or not right.index.is_unique:
        raise DataQualityError("cross-check evidence contains duplicate identities")
    symbols = tuple(sorted(set(left.index).intersection(right.index)))
    if not symbols:
        raise DataUnavailableError("AKShare evidence has no comparable exact identities")
    mismatches: list[AKShareFieldMismatch] = []
    field_tolerances = {
        "open": (tolerance.price_absolute_cny, tolerance.price_relative),
        "high": (tolerance.price_absolute_cny, tolerance.price_relative),
        "low": (tolerance.price_absolute_cny, tolerance.price_relative),
        "close": (tolerance.price_absolute_cny, tolerance.price_relative),
        "prev_close": (tolerance.price_absolute_cny, tolerance.price_relative),
        "volume_shares": (
            tolerance.volume_absolute_shares,
            tolerance.volume_relative,
        ),
        "amount_cny": (tolerance.amount_absolute_cny, tolerance.amount_relative),
    }
    fields = compared_fields or tuple(field_tolerances)
    unknown_fields = set(fields).difference(field_tolerances)
    if unknown_fields or len(fields) != len(set(fields)) or not fields:
        raise ValueError("comparison fields must be a unique non-empty supported tuple")
    for symbol in symbols:
        for field in fields:
            absolute, relative = field_tolerances[field]
            tushare_value = float(left.at[symbol, field])
            akshare_value = float(right.at[symbol, field])
            difference = abs(tushare_value - akshare_value)
            allowed = max(absolute, relative * max(abs(tushare_value), abs(akshare_value)))
            if difference > allowed:
                mismatches.append(
                    AKShareFieldMismatch(
                        symbol=symbol,
                        field=field,
                        tushare_value=tushare_value,
                        akshare_value=akshare_value,
                        absolute_difference=difference,
                        allowed_difference=allowed,
                    )
                )
    return AKShareEodVerificationResult(
        status=(
            AKShareVerificationStatus.MISMATCH
            if mismatches
            else AKShareVerificationStatus.VERIFIED
        ),
        target_date=target_date,
        mode=mode,
        tushare_row_count=len(tushare),
        compared_symbols=symbols,
        mismatches=tuple(mismatches),
        evidence_trade_date=target_date,
        evidence_retrieved_at=_aware_utc(evidence_retrieved_at),
    )


def _deterministic_sample(
    symbols: tuple[str, ...],
    target_date: date,
    sample_size: int,
) -> tuple[str, ...]:
    unique = tuple(sorted(set(symbols)))
    if not unique:
        raise DataUnavailableError("no symbols are available for historical sampling")
    ranked = sorted(
        unique,
        key=lambda symbol: (
            hashlib.sha256(f"{target_date.isoformat()}:{symbol}".encode()).digest(),
            symbol,
        ),
    )
    return tuple(sorted(ranked[: min(sample_size, len(ranked))]))


def _validate_bar_frame(frame: pd.DataFrame, *, unit_band: float, provider: str) -> None:
    prices = frame[["open", "high", "low", "close", "prev_close"]]
    if not prices.map(lambda value: math.isfinite(float(value))).all(axis=None):
        raise DataQualityError(f"{provider} has non-finite price values")
    if (prices <= 0).any(axis=None):
        raise DataQualityError(f"{provider} prices must be positive")
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any():
        raise DataQualityError(f"{provider} high is below another OHLC value")
    if (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any():
        raise DataQualityError(f"{provider} low is above another OHLC value")
    if (frame[["volume_shares", "amount_cny"]] < 0).any(axis=None):
        raise DataQualityError(f"{provider} volume or amount is negative")
    one_zero = frame["volume_shares"].eq(0) ^ frame["amount_cny"].eq(0)
    if one_zero.any():
        raise DataQualityError(f"{provider} volume and amount zero states conflict")
    traded = frame["volume_shares"] > 0
    implied = frame.loc[traded, "amount_cny"] / frame.loc[traded, "volume_shares"]
    lower = frame.loc[traded, "low"] * (1.0 - unit_band)
    upper = frame.loc[traded, "high"] * (1.0 + unit_band)
    if ((implied < lower) | (implied > upper)).any():
        raise DataQualityError(
            f"{provider} amount/volume implied price is outside the OHLC band; units are unverified"
        )


def _strict_numbers(values: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.map(lambda value: math.isfinite(float(value))).all():
        raise DataQualityError(f"{label} contains missing, non-numeric, or non-finite values")
    return numeric.astype(float)


def _strict_scalar(value: object, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DataQualityError(f"{label} is not numeric") from exc
    if not math.isfinite(numeric):
        raise DataQualityError(f"{label} is not finite")
    return numeric


def _strict_symbols(values: pd.Series, label: str) -> pd.Series:
    if not values.map(lambda value: isinstance(value, str)).all():
        raise DataQualityError(f"{label} identity is not an exact string")
    symbols = values.str.strip()
    if not symbols.str.fullmatch(_SYMBOL).all():
        raise DataQualityError(f"{label} identity is not an exact six-digit stock code")
    return symbols


def _exact_dates(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise DataQualityError(f"{label} contains invalid dates")
    if parsed.dt.tz is not None:
        raise DataQualityError(f"{label} date field unexpectedly contains timezone timestamps")
    if not (parsed == parsed.dt.normalize()).all():
        raise DataQualityError(f"{label} date field contains time-of-day ambiguity")
    return parsed.dt.date


def _find_column(
    frame: pd.DataFrame,
    aliases: tuple[str, ...],
    *,
    required: bool,
) -> str | None:
    matches = [name for name in aliases if name in frame.columns]
    if len(matches) > 1:
        raise DataQualityError("ambiguous aliases present: " + ",".join(matches))
    if not matches:
        if required:
            raise DataQualityError("required AKShare field is missing: " + "/".join(aliases))
        return None
    return matches[0]


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DataUnavailableError("verification retrieval time must be timezone-aware")
    return value.astimezone(UTC)


def _unavailable_result(
    target_date: date,
    row_count: int,
    *reasons: str,
) -> AKShareEodVerificationResult:
    return AKShareEodVerificationResult(
        status=AKShareVerificationStatus.UNAVAILABLE,
        target_date=target_date,
        mode=AKShareEvidenceMode.NONE,
        tushare_row_count=row_count,
        compared_symbols=(),
        unavailable_reasons=tuple(reason for reason in reasons if reason),
    )
