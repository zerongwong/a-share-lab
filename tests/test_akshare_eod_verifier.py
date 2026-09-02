from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from ashare_lab.adapters.akshare_eod_verifier import (
    AKShareEodVerifier,
    AKShareEvidenceMode,
    AKShareSnapshotEvidence,
    AKShareVerificationStatus,
)
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError

TARGET = date(2026, 9, 1)
NOW = datetime(2026, 9, 1, 8, 30, tzinfo=UTC)
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
    rows = []
    for symbol in symbols:
        rows.append(
            {
                "symbol": symbol,
                "trade_date": TARGET,
                **_values(symbol),
                "source": "tushare:daily_unadjusted",
            }
        )
    return pd.DataFrame(rows)


def _snapshot_frame(symbols: tuple[str, ...] = SYMBOLS) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        values = _values(symbol)
        rows.append(
            {
                "代码": symbol,
                "今开": values["open"],
                "最高": values["high"],
                "最低": values["low"],
                "最新价": values["close"],
                "昨收": values["prev_close"],
                "成交量": values["volume_shares"] / 100.0,
                "成交额": values["amount_cny"],
            }
        )
    return pd.DataFrame(rows)


def _snapshot_evidence(
    *,
    frame: pd.DataFrame | None = None,
    trade_date: date = TARGET,
) -> AKShareSnapshotEvidence:
    return AKShareSnapshotEvidence(
        frame=_snapshot_frame() if frame is None else frame,
        trade_date=trade_date,
        retrieved_at=NOW,
        date_evidence="independent-completed-session-2026-09-01",
    )


def _history(symbol: str, start: date, end: date) -> pd.DataFrame:
    del start, end
    values = _values(symbol)
    return pd.DataFrame(
        [
            {
                "代码": symbol,
                "日期": date(2026, 8, 31),
                "开盘": values["prev_close"] - 0.1,
                "最高": values["prev_close"] + 0.2,
                "最低": values["prev_close"] - 0.2,
                "收盘": values["prev_close"],
                "成交量": 900.0,
                "成交额": values["prev_close"] * 90_000.0,
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


def test_dated_full_market_snapshot_verifies_every_tushare_identity() -> None:
    verifier = AKShareEodVerifier(
        snapshot_fetcher=_snapshot_evidence,
        history_fetcher=lambda *_: pytest.fail("snapshot should avoid history calls"),
        clock=lambda: NOW,
    )

    result = verifier.verify(_tushare_batch(), TARGET)

    assert result.status is AKShareVerificationStatus.VERIFIED
    assert result.is_verified is True
    assert result.mode is AKShareEvidenceMode.SNAPSHOT
    assert result.compared_symbols == tuple(sorted(SYMBOLS))
    assert result.compared_count == len(SYMBOLS)
    assert result.evidence_trade_date == TARGET
    assert result.evidence_retrieved_at == NOW
    result.require_verified()


def test_strict_integration_entrypoint_returns_receipt_only_when_verified() -> None:
    verifier = AKShareEodVerifier(
        snapshot_fetcher=_snapshot_evidence,
        history_fetcher=lambda *_: pytest.fail("snapshot should avoid history calls"),
    )

    receipt = verifier.verify_stock_frame(_tushare_batch(), TARGET, SYMBOLS)

    assert receipt.status is AKShareVerificationStatus.VERIFIED


def test_strict_integration_entrypoint_rejects_requested_identity_mismatch() -> None:
    verifier = AKShareEodVerifier(snapshot_fetcher=_snapshot_evidence)

    with pytest.raises(DataQualityError, match="身份集合不一致"):
        verifier.verify_stock_frame(_tushare_batch(), TARGET, (*SYMBOLS[:-1], "600519"))


def test_snapshot_comparison_accepts_small_declared_rounding_differences() -> None:
    snapshot = _snapshot_frame()
    snapshot.loc[snapshot["代码"] == "000001", "最新价"] += 0.005
    snapshot.loc[snapshot["代码"] == "000001", "成交额"] += 500.0
    snapshot.loc[snapshot["代码"] == "000001", "成交量"] += 1.0
    verifier = AKShareEodVerifier(
        snapshot_fetcher=lambda: _snapshot_evidence(frame=snapshot),
        history_fetcher=lambda *_: pytest.fail("snapshot should be usable"),
    )

    assert verifier.verify(_tushare_batch(), TARGET).is_verified is True


def test_proven_numeric_difference_is_mismatch_not_unavailable() -> None:
    snapshot = _snapshot_frame()
    snapshot.loc[snapshot["代码"] == "600000", "最新价"] += 0.20
    verifier = AKShareEodVerifier(
        snapshot_fetcher=lambda: _snapshot_evidence(frame=snapshot),
        history_fetcher=lambda *_: pytest.fail("numeric mismatch is conclusive"),
    )

    result = verifier.verify(_tushare_batch(), TARGET)

    assert result.status is AKShareVerificationStatus.MISMATCH
    assert result.mode is AKShareEvidenceMode.SNAPSHOT
    assert any(item.symbol == "600000" and item.field == "close" for item in result.mismatches)
    with pytest.raises(DataQualityError, match="600000:close"):
        result.require_verified()


def test_wrong_snapshot_date_is_rejected_then_explicit_history_can_verify() -> None:
    calls: list[str] = []

    def history(symbol: str, start: date, end: date) -> pd.DataFrame:
        calls.append(symbol)
        return _history(symbol, start, end)

    verifier = AKShareEodVerifier(
        sample_size=2,
        snapshot_fetcher=lambda: _snapshot_evidence(trade_date=date(2026, 8, 31)),
        history_fetcher=history,
        clock=lambda: NOW,
    )

    result = verifier.verify(_tushare_batch(), TARGET)

    assert result.status is AKShareVerificationStatus.VERIFIED
    assert result.mode is AKShareEvidenceMode.HISTORICAL_SAMPLE
    assert result.compared_symbols == tuple(calls)
    assert len(calls) == 2


def test_historical_symbol_sample_is_deterministic_for_target_date() -> None:
    first_calls: list[str] = []
    second_calls: list[str] = []

    first = AKShareEodVerifier(
        sample_size=3,
        history_fetcher=lambda symbol, start, end: (
            first_calls.append(symbol) or _history(symbol, start, end)
        ),
        clock=lambda: NOW,
    )
    second = AKShareEodVerifier(
        sample_size=3,
        history_fetcher=lambda symbol, start, end: (
            second_calls.append(symbol) or _history(symbol, start, end)
        ),
        clock=lambda: NOW,
    )

    first_result = first.verify(_tushare_batch(), TARGET)
    second_result = second.verify(_tushare_batch(), TARGET)

    assert first_result.is_verified and second_result.is_verified
    assert first_calls == second_calls
    assert first_result.compared_symbols == tuple(first_calls)


def test_historical_sample_does_not_treat_derived_prior_close_as_official_evidence() -> None:
    """Raw prior close differs from official pre-close on ex-right sessions."""

    def ex_right_history(symbol: str, start: date, end: date) -> pd.DataFrame:
        frame = _history(symbol, start, end)
        target_index = frame.index[frame["日期"] == TARGET][0]
        prior_index = target_index - 1
        frame.loc[prior_index, "收盘"] = frame.loc[prior_index, "收盘"] + 0.15
        return frame

    result = AKShareEodVerifier(
        sample_size=1,
        history_fetcher=ex_right_history,
        clock=lambda: NOW,
    ).verify(_tushare_batch(("000001",)), TARGET)

    assert result.status is AKShareVerificationStatus.VERIFIED
    assert result.mismatches == ()


def test_snapshot_still_verifies_explicit_previous_close() -> None:
    snapshot = _snapshot_frame(("000001",))
    snapshot.loc[0, "昨收"] += 0.15
    result = AKShareEodVerifier(
        snapshot_fetcher=lambda: _snapshot_evidence(frame=snapshot),
        history_fetcher=lambda *_: pytest.fail("numeric mismatch is conclusive"),
    ).verify(_tushare_batch(("000001",)), TARGET)

    assert result.status is AKShareVerificationStatus.MISMATCH
    assert any(item.field == "prev_close" for item in result.mismatches)


def test_undated_snapshot_and_failed_history_return_explicit_unavailable() -> None:
    verifier = AKShareEodVerifier(
        snapshot_fetcher=lambda: _snapshot_frame(),  # type: ignore[arg-type,return-value]
        history_fetcher=lambda *_: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    result = verifier.verify(_tushare_batch(), TARGET)

    assert result.status is AKShareVerificationStatus.UNAVAILABLE
    assert result.mode is AKShareEvidenceMode.NONE
    assert result.compared_count == 0
    assert any("SNAPSHOT_UNUSABLE" in reason for reason in result.unavailable_reasons)
    assert any("HISTORICAL_SAMPLE_FETCH_FAILED" in reason for reason in result.unavailable_reasons)
    with pytest.raises(DataUnavailableError):
        result.require_verified()


def test_numeric_stock_identity_is_never_zero_filled_or_guessed() -> None:
    batch = _tushare_batch(("000001",))
    batch["symbol"] = pd.Series([1], dtype=object)
    fetch_calls = 0

    def history(*_args) -> pd.DataFrame:
        nonlocal fetch_calls
        fetch_calls += 1
        return pd.DataFrame()

    result = AKShareEodVerifier(history_fetcher=history).verify(batch, TARGET)

    assert result.status is AKShareVerificationStatus.UNAVAILABLE
    assert "identity" in result.unavailable_reasons[0]
    assert fetch_calls == 0


def test_intraday_timestamp_is_not_silently_coerced_to_target_date() -> None:
    batch = _tushare_batch(("600000",))
    batch.loc[0, "trade_date"] = datetime(2026, 9, 1, 15, 0)

    result = AKShareEodVerifier(history_fetcher=_history).verify(batch, TARGET)

    assert result.status is AKShareVerificationStatus.UNAVAILABLE
    assert "time-of-day" in result.unavailable_reasons[0]


def test_missing_history_target_date_is_unavailable_not_nearest_date() -> None:
    def missing_target(symbol: str, start: date, end: date) -> pd.DataFrame:
        return _history(symbol, start, end).iloc[:1].copy()

    result = AKShareEodVerifier(
        sample_size=1,
        history_fetcher=missing_target,
        clock=lambda: NOW,
    ).verify(_tushare_batch(("000001",)), TARGET)

    assert result.status is AKShareVerificationStatus.UNAVAILABLE
    assert any("no unique target-date row" in reason for reason in result.unavailable_reasons)


def test_unknown_history_volume_unit_fails_implied_price_sanity_check() -> None:
    def shares_mislabeled_as_lots(symbol: str, start: date, end: date) -> pd.DataFrame:
        frame = _history(symbol, start, end)
        frame.loc[frame["日期"] == TARGET, "成交量"] *= 100.0
        return frame

    result = AKShareEodVerifier(
        sample_size=1,
        history_fetcher=shares_mislabeled_as_lots,
        clock=lambda: NOW,
    ).verify(_tushare_batch(("000001",)), TARGET)

    assert result.status is AKShareVerificationStatus.UNAVAILABLE
    assert any("units are unverified" in reason for reason in result.unavailable_reasons)


def test_full_snapshot_missing_one_target_identity_cannot_claim_verification() -> None:
    partial = _snapshot_frame(SYMBOLS[:-1])
    result = AKShareEodVerifier(
        snapshot_fetcher=lambda: _snapshot_evidence(frame=partial),
        history_fetcher=lambda *_: (_ for _ in ()).throw(RuntimeError("offline")),
    ).verify(_tushare_batch(), TARGET)

    assert result.status is AKShareVerificationStatus.UNAVAILABLE
    assert any("missing target identities" in reason for reason in result.unavailable_reasons)


def test_default_akshare_module_is_loaded_lazily_and_uses_unadjusted_history() -> None:
    loader_calls = 0
    seen: list[dict[str, object]] = []

    class FakeAKShare:
        def stock_zh_a_hist(self, **kwargs) -> pd.DataFrame:
            seen.append(kwargs)
            return _history(kwargs["symbol"], date(2026, 8, 1), TARGET)

    def loader() -> FakeAKShare:
        nonlocal loader_calls
        loader_calls += 1
        return FakeAKShare()

    verifier = AKShareEodVerifier(sample_size=1, module_loader=loader, clock=lambda: NOW)
    assert loader_calls == 0

    result = verifier.verify(_tushare_batch(("600000",)), TARGET)

    assert result.is_verified is True
    assert loader_calls == 1
    assert seen == [
        {
            "symbol": "600000",
            "period": "daily",
            "start_date": "20260718",
            "end_date": "20260901",
            "adjust": "",
        }
    ]


def test_default_history_falls_back_to_unadjusted_sina_when_eastmoney_disconnects() -> None:
    seen: list[dict[str, object]] = []

    class FakeAKShare:
        def stock_zh_a_hist(self, **_kwargs) -> pd.DataFrame:
            raise ConnectionError("public endpoint disconnected")

        def stock_zh_a_daily(self, **kwargs) -> pd.DataFrame:
            seen.append(kwargs)
            values = _values("600000")
            return pd.DataFrame(
                [
                    {
                        "date": date(2026, 8, 31),
                        "open": values["prev_close"] - 0.1,
                        "high": values["prev_close"] + 0.2,
                        "low": values["prev_close"] - 0.2,
                        "close": values["prev_close"],
                        "volume": 90_000.0,
                        "amount": values["prev_close"] * 90_000.0,
                    },
                    {
                        "date": TARGET,
                        "open": values["open"],
                        "high": values["high"],
                        "low": values["low"],
                        "close": values["close"],
                        "volume": values["volume_shares"],
                        "amount": values["amount_cny"],
                    },
                ]
            )

    verifier = AKShareEodVerifier(
        sample_size=1,
        module_loader=FakeAKShare,
        clock=lambda: NOW,
    )

    result = verifier.verify(_tushare_batch(("600000",)), TARGET)

    assert result.is_verified is True
    assert seen == [
        {
            "symbol": "sh600000",
            "start_date": "20260718",
            "end_date": "20260901",
            "adjust": "",
        }
    ]


def test_snapshot_requires_aware_retrieval_time_and_nonempty_date_evidence() -> None:
    evidence = AKShareSnapshotEvidence(
        frame=_snapshot_frame(("000001",)),
        trade_date=TARGET,
        retrieved_at=datetime(2026, 9, 1, 16, 0),
        date_evidence="",
    )
    result = AKShareEodVerifier(
        snapshot_fetcher=lambda: evidence,
        history_fetcher=lambda *_: (_ for _ in ()).throw(RuntimeError("offline")),
    ).verify(_tushare_batch(("000001",)), TARGET)

    assert result.status is AKShareVerificationStatus.UNAVAILABLE
    assert any("timezone-aware" in reason for reason in result.unavailable_reasons)
