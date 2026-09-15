from __future__ import annotations

import io
import json
import subprocess
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from ashare_lab.adapters.akshare_eod_verifier import (
    AKShareEodVerifier,
    AKShareEvidenceMode,
    AKShareVerificationStatus,
)
from ashare_lab.adapters.bounded_baostock_eod_verifier import (
    BoundedBaoStockEodVerifier,
    UnavailableOnlyEodVerifierFallback,
)
from ashare_lab.cli import baostock_eod_verify as child
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError

TARGET = date(2026, 9, 14)
NOW = datetime(2026, 9, 14, 13, 30, tzinfo=UTC)
SYMBOLS = ("000001", "300001", "600000", "601398")


def _values(symbol: str) -> dict[str, float]:
    offset = SYMBOLS.index(symbol)
    previous = 10.0 + offset
    close = previous + 0.5
    volume = 100_000.0 + offset * 10_000.0
    return {
        "open": previous + 0.1,
        "high": previous + 1.0,
        "low": previous - 0.1,
        "close": close,
        "prev_close": previous,
        "volume_shares": volume,
        "amount_cny": close * volume,
    }


def _tushare_batch(symbols: tuple[str, ...] = SYMBOLS) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": TARGET,
                **_values(symbol),
                "source": "tushare:daily_unadjusted",
            }
            for symbol in symbols
        ]
    )


def _document(
    request: dict[str, object],
    *,
    close_offset: float = 0.0,
    previous_close_offset: float = 0.0,
) -> dict[str, object]:
    rows = []
    for symbol in request["symbols"]:
        assert isinstance(symbol, str)
        row = {
            "symbol": symbol,
            "trade_date": request["target_date"],
            **_values(symbol),
        }
        if symbol == request["symbols"][0]:
            row["close"] += close_offset
            row["prev_close"] += previous_close_offset
        rows.append(row)
    return {
        "provider": "baostock",
        "method_version": "baostock-unadjusted-daily-sample-v1",
        "target_date": request["target_date"],
        "retrieved_at": request["now"],
        "rows": rows,
    }


def _runner(
    *,
    close_offset: float = 0.0,
    previous_close_offset: float = 0.0,
    seen: list[dict[str, object]] | None = None,
):
    def run(*args, **kwargs):
        assert args[0][1:] == ["-m", "ashare_lab.cli.baostock_eod_verify"]
        assert kwargs["timeout"] == 45.0
        request = json.loads(kwargs["input"])
        if seen is not None:
            seen.append(request)
        payload = {
            "result": _document(
                request,
                close_offset=close_offset,
                previous_close_offset=previous_close_offset,
            )
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    return run


def _akshare_history(symbol: str, _start: date, _end: date) -> pd.DataFrame:
    values = _values(symbol)
    return pd.DataFrame(
        [
            {
                "代码": symbol,
                "日期": date(2026, 9, 11),
                "开盘": values["prev_close"],
                "最高": values["prev_close"] + 0.1,
                "最低": values["prev_close"] - 0.1,
                "收盘": values["prev_close"],
                "成交量": 1_000.0,
                "成交额": values["prev_close"] * 100_000.0,
            },
            {
                "代码": symbol,
                "日期": TARGET,
                "开盘": values["open"],
                "最高": values["high"],
                "最低": values["low"],
                "收盘": values["close"],
                "成交量": values["volume_shares"] / 100.0,
                "成交额": values["amount_cny"],
            },
        ]
    )


def test_verifies_with_same_deterministic_sample_as_akshare_history_path() -> None:
    seen: list[dict[str, object]] = []
    ak_calls: list[str] = []
    AKShareEodVerifier(
        sample_size=3,
        history_fetcher=lambda symbol, start, end: (
            ak_calls.append(symbol) or _akshare_history(symbol, start, end)
        ),
        clock=lambda: NOW,
    ).verify(_tushare_batch(), TARGET)
    verifier = BoundedBaoStockEodVerifier(
        sample_size=3,
        clock=lambda: NOW,
        runner=_runner(seen=seen),
    )

    result = verifier.verify_stock_frame(_tushare_batch(), TARGET, SYMBOLS)

    assert result.status is AKShareVerificationStatus.VERIFIED
    assert result.mode is AKShareEvidenceMode.HISTORICAL_SAMPLE
    assert result.compared_symbols == tuple(ak_calls)
    assert seen[0]["symbols"] == ak_calls
    assert result.evidence_retrieved_at == NOW


def test_baostock_volume_is_already_shares_and_amount_is_cny() -> None:
    verifier = BoundedBaoStockEodVerifier(
        sample_size=1,
        clock=lambda: NOW,
        runner=_runner(),
    )

    assert verifier.verify(_tushare_batch(("000001",)), TARGET).is_verified


def test_numeric_difference_is_mismatch_and_strict_entrypoint_is_quality_error() -> None:
    verifier = BoundedBaoStockEodVerifier(
        sample_size=1,
        clock=lambda: NOW,
        runner=_runner(close_offset=0.20),
    )

    result = verifier.verify(_tushare_batch(("000001",)), TARGET)

    assert result.status is AKShareVerificationStatus.MISMATCH
    assert result.mismatches[0].field == "close"
    with pytest.raises(DataQualityError, match="000001:close"):
        verifier.verify_stock_frame(_tushare_batch(("000001",)), TARGET, ("000001",))


def test_previous_close_difference_is_independently_rejected() -> None:
    verifier = BoundedBaoStockEodVerifier(
        sample_size=1,
        clock=lambda: NOW,
        runner=_runner(previous_close_offset=0.20),
    )

    result = verifier.verify(_tushare_batch(("000001",)), TARGET)

    assert result.status is AKShareVerificationStatus.MISMATCH
    assert result.mismatches[0].field == "prev_close"
    with pytest.raises(DataQualityError, match="000001:prev_close"):
        verifier.verify_stock_frame(_tushare_batch(("000001",)), TARGET, ("000001",))


def test_timeout_is_unavailable_and_never_leaks_child_diagnostics() -> None:
    secret = "upstream-sensitive-diagnostics"

    def runner(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("child", 1, output=secret, stderr=secret)

    verifier = BoundedBaoStockEodVerifier(
        sample_size=1,
        clock=lambda: NOW,
        timeout=1,
        runner=runner,
    )

    with pytest.raises(DataUnavailableError) as caught:
        verifier.verify_stock_frame(_tushare_batch(("000001",)), TARGET, ("000001",))

    assert secret not in str(caught.value)


def test_child_quality_failure_is_not_downgraded_to_unavailable() -> None:
    def runner(*_args, **_kwargs):
        return SimpleNamespace(returncode=2, stdout='{"error":"quality"}', stderr="")

    verifier = BoundedBaoStockEodVerifier(
        sample_size=1,
        clock=lambda: NOW,
        runner=runner,
    )

    with pytest.raises(DataQualityError, match="禁止切源"):
        verifier.verify_stock_frame(_tushare_batch(("000001",)), TARGET, ("000001",))


def test_malformed_or_oversized_child_output_is_fail_closed() -> None:
    malformed = lambda *_args, **_kwargs: SimpleNamespace(  # noqa: E731
        returncode=0, stdout="not-json", stderr=""
    )
    oversized = lambda *_args, **_kwargs: SimpleNamespace(  # noqa: E731
        returncode=0, stdout="x" * 262_145, stderr=""
    )
    for runner, error in (
        (malformed, DataUnavailableError),
        (oversized, DataQualityError),
    ):
        verifier = BoundedBaoStockEodVerifier(
            sample_size=1,
            clock=lambda: NOW,
            runner=runner,
        )
        with pytest.raises(error):
            verifier.verify_stock_frame(
                _tushare_batch(("000001",)),
                TARGET,
                ("000001",),
            )


def test_request_identity_mismatch_blocks_child_call() -> None:
    verifier = BoundedBaoStockEodVerifier(
        clock=lambda: NOW,
        runner=lambda *_args, **_kwargs: pytest.fail("child must not run"),
    )

    with pytest.raises(DataQualityError, match="身份集合不一致"):
        verifier.verify_stock_frame(_tushare_batch(("000001",)), TARGET, ("600000",))


class _Verifier:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls = 0

    def verify_stock_frame(self, *_args) -> object:
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_fallback_uses_backup_only_for_primary_data_unavailable() -> None:
    receipt = SimpleNamespace(status=AKShareVerificationStatus.VERIFIED)
    primary = _Verifier(DataUnavailableError("primary offline"))
    backup = _Verifier(receipt)
    fallback = UnavailableOnlyEodVerifierFallback(primary=primary, backup=backup)

    assert fallback.verify_stock_frame(pd.DataFrame(), TARGET, ()) is receipt
    assert primary.calls == 1
    assert backup.calls == 1
    assert fallback.last_evidence.startswith("_Verifier:historical-sample:verified:")
    assert fallback.last_evidence.endswith("sample=0")


@pytest.mark.parametrize(
    "failure",
    [DataQualityError("primary mismatch"), RuntimeError("unexpected defect")],
)
def test_fallback_never_masks_primary_quality_or_unexpected_failures(failure: Exception) -> None:
    primary = _Verifier(failure)
    backup = _Verifier(SimpleNamespace(status=AKShareVerificationStatus.VERIFIED))
    fallback = UnavailableOnlyEodVerifierFallback(primary=primary, backup=backup)

    with pytest.raises(type(failure)):
        fallback.verify_stock_frame(pd.DataFrame(), TARGET, ())
    assert backup.calls == 0


def test_fallback_never_masks_quality_failure_from_real_akshare_history_adapter() -> None:
    def malformed_history(symbol: str, start: date, end: date) -> pd.DataFrame:
        return _akshare_history(symbol, start, end).drop(columns=["成交额"])

    primary = AKShareEodVerifier(
        sample_size=1,
        history_fetcher=malformed_history,
        clock=lambda: NOW,
    )
    backup = _Verifier(SimpleNamespace(status=AKShareVerificationStatus.VERIFIED))
    fallback = UnavailableOnlyEodVerifierFallback(primary=primary, backup=backup)

    with pytest.raises(DataQualityError):
        fallback.verify_stock_frame(
            _tushare_batch(("000001",)),
            TARGET,
            ("000001",),
        )
    assert backup.calls == 0


class _Rows:
    fields = list(child._FIELDS)

    def __init__(self, rows: list[list[str]]) -> None:
        self.error_code = "0"
        self._rows = iter(rows)
        self._current: list[str] | None = None

    def next(self) -> bool:
        self._current = next(self._rows, None)
        return self._current is not None

    def get_row_data(self) -> list[str]:
        assert self._current is not None
        return self._current


class _FakeBaoStock:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.logout_calls = 0

    def login(self) -> SimpleNamespace:
        return SimpleNamespace(error_code="0")

    def logout(self) -> None:
        self.logout_calls += 1

    def query_history_k_data_plus(self, *args, **kwargs) -> _Rows:
        self.calls.append((*args, kwargs))
        provider_code = args[0]
        symbol = provider_code.split(".")[1]
        values = _values(symbol)
        return _Rows(
            [
                [
                    TARGET.isoformat(),
                    provider_code,
                    str(values["open"]),
                    str(values["high"]),
                    str(values["low"]),
                    str(values["close"]),
                    str(values["prev_close"]),
                    str(values["volume_shares"]),
                    str(values["amount_cny"]),
                ]
            ]
        )


def test_child_requests_unadjusted_daily_rows_and_preserves_units() -> None:
    module = _FakeBaoStock()

    document = child._read_sample(
        module,
        symbols=("000001", "600000"),
        target_date=TARGET,
        retrieved_at=NOW,
    )

    assert module.calls[0][0] == "sz.000001"
    assert module.calls[1][0] == "sh.600000"
    assert module.calls[0][-1] == {
        "start_date": TARGET.isoformat(),
        "end_date": TARGET.isoformat(),
        "frequency": "d",
        "adjustflag": "3",
    }
    assert document["rows"][0]["volume_shares"] == 100_000.0
    assert document["rows"][0]["amount_cny"] == 1_050_000.0
    assert module.logout_calls == 1


def test_child_main_emits_only_safe_error_classification(monkeypatch, capsys) -> None:
    secret = "provider-secret-in-exception"
    request = {
        "operation": "verify_stock_sample",
        "target_date": TARGET.isoformat(),
        "now": NOW.isoformat(),
        "symbols": ["000001"],
    }
    monkeypatch.setattr(child.sys, "stdin", io.StringIO(json.dumps(request)))
    monkeypatch.setattr(
        child,
        "_load_baostock",
        lambda: (_ for _ in ()).throw(RuntimeError(secret)),
    )

    assert child.main() == 2

    output = capsys.readouterr().out
    assert json.loads(output) == {"error": "unavailable"}
    assert secret not in output
