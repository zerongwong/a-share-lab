from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from ashare_lab.adapters.official_exchange_stock_master import (
    BoundedOfficialExchangeStockMaster,
)
from ashare_lab.cli import exchange_stock_master_read
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError

NOW = datetime(2026, 9, 14, 13, 30, tzinfo=UTC)


def official_symbols() -> tuple[str, ...]:
    sh_main = [f"{code:06d}.SH" for code in range(600_000, 601_701)]
    sh_star = [f"{code:06d}.SH" for code in range(688_000, 688_617)]
    sz_a = [f"{code:06d}.SZ" for code in range(0, 1_451)]
    sz_a.extend(f"{code:06d}.SZ" for code in range(300_000, 301_450))
    return tuple(sorted((*sh_main, *sh_star, *sz_a)))


def official_document(*, symbols=None, counts=None, **overrides):
    document = {
        "symbols": list(symbols if symbols is not None else official_symbols()),
        "counts": counts
        if counts is not None
        else {"sse_main": 1_701, "sse_star": 617, "szse_a": 2_901},
        "retrieved_at": NOW.isoformat(),
        "provider": "sse_szse_official_via_akshare",
        "method_version": "exchange-listed-a-v1",
    }
    document.update(overrides)
    return document


def runner_for(document, *, returncode=0):
    def runner(command, **kwargs):
        assert command == [sys.executable, "-m", "ashare_lab.cli.exchange_stock_master_read"]
        assert kwargs["text"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        assert kwargs["timeout"] == 3
        request = json.loads(kwargs["input"])
        assert request == {"operation": "listed_a_share_symbols", "now": NOW.isoformat()}
        return SimpleNamespace(
            returncode=returncode,
            stdout=json.dumps({"result": document}, ensure_ascii=False),
        )

    return runner


def test_official_stock_master_accepts_a_complete_sorted_contract_and_records_evidence():
    symbols = official_symbols()
    adapter = BoundedOfficialExchangeStockMaster(
        clock=lambda: NOW, timeout=3, runner=runner_for(official_document(symbols=symbols))
    )

    assert adapter.fetch_cn_stock_symbols() == symbols
    digest = hashlib.sha256("|".join(symbols).encode()).hexdigest()
    assert adapter.last_evidence == (
        "sse_szse_official_via_akshare:main=1701;star=617;sz=2901;"
        f"sha256={digest};method=exchange-listed-a-v1;"
        "retrieved_at=2026-09-14T13:30:00+00:00;"
        "endpoints=sse_name_code+szse_a_list"
    )


def test_official_stock_master_timeout_is_unavailable_and_does_not_leak_child_output():
    def runner(*_args, **kwargs):
        raise subprocess.TimeoutExpired(
            "private-command", kwargs["timeout"], output="sensitive provider diagnostics"
        )

    with pytest.raises(DataUnavailableError) as caught:
        BoundedOfficialExchangeStockMaster(
            clock=lambda: NOW, timeout=3, runner=runner
        ).fetch_cn_stock_symbols()
    assert "sensitive" not in str(caught.value)
    assert "private-command" not in str(caught.value)


@pytest.mark.parametrize("stdout", ["not-json", "null", "[]"])
def test_official_stock_master_rejects_an_invalid_child_envelope_as_unavailable(stdout):
    def runner(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=stdout)

    with pytest.raises(DataUnavailableError):
        BoundedOfficialExchangeStockMaster(
            clock=lambda: NOW, runner=runner
        ).fetch_cn_stock_symbols()


@pytest.mark.parametrize(
    ("returncode", "payload", "error"),
    [
        (2, {"error": "unavailable"}, DataUnavailableError),
        (2, {"error": "quality"}, DataQualityError),
        (0, {"result": []}, DataQualityError),
    ],
)
def test_official_stock_master_preserves_child_failure_class(returncode, payload, error):
    def runner(*_args, **_kwargs):
        return SimpleNamespace(returncode=returncode, stdout=json.dumps(payload))

    with pytest.raises(error):
        BoundedOfficialExchangeStockMaster(
            clock=lambda: NOW, runner=runner
        ).fetch_cn_stock_symbols()


@pytest.mark.parametrize(
    "document",
    [
        official_document(provider="lookalike-provider"),
        official_document(method_version="unknown-method"),
        official_document(counts={"sse_main": 1_701, "sse_star": 617, "szse_a": 2_900}),
        official_document(
            counts={"sse_main": 1_701, "sse_star": 617, "szse_a": 2_901, "extra": "bad"}
        ),
    ],
)
def test_official_stock_master_rejects_protocol_or_count_quality_violations(document):
    with pytest.raises(DataQualityError):
        BoundedOfficialExchangeStockMaster(
            clock=lambda: NOW, timeout=3, runner=runner_for(document)
        ).fetch_cn_stock_symbols()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda symbols: tuple(reversed(symbols)),
        lambda symbols: (*symbols[:-1], symbols[-2]),
        lambda symbols: (*symbols[:-1], "123456.SH"),
        lambda symbols: symbols[:3_999],
    ],
)
def test_official_stock_master_rejects_invalid_identity_order_uniqueness_or_size(mutate):
    symbols = mutate(official_symbols())
    counts = {"sse_main": 1_701, "sse_star": 617, "szse_a": len(symbols) - 2_318}
    with pytest.raises(DataQualityError):
        BoundedOfficialExchangeStockMaster(
            clock=lambda: NOW,
            timeout=3,
            runner=runner_for(official_document(symbols=symbols, counts=counts)),
        ).fetch_cn_stock_symbols()


def test_official_stock_master_rejects_a_plausible_but_currently_truncated_total():
    symbols = official_symbols()[:4_999]
    with pytest.raises(DataQualityError):
        BoundedOfficialExchangeStockMaster(
            clock=lambda: NOW,
            timeout=3,
            runner=runner_for(
                official_document(
                    symbols=symbols,
                    counts={"sse_main": 1_500, "sse_star": 500, "szse_a": 2_999},
                )
            ),
        ).fetch_cn_stock_symbols()


def symbols_with_replacements(count: int) -> tuple[str, ...]:
    symbols = set(official_symbols())
    for offset in range(count):
        symbols.remove(f"{offset:06d}.SZ")
        symbols.add(f"{2_000 + offset:06d}.SZ")
    return tuple(sorted(symbols))


def test_official_stock_master_allows_bounded_drift_but_rejects_more_than_fifty_changes():
    expected = official_symbols()
    allowed = symbols_with_replacements(25)  # 25 removals + 25 additions = 50 changes.
    rejected = symbols_with_replacements(26)

    assert (
        BoundedOfficialExchangeStockMaster(
            expected_symbols=expected,
            clock=lambda: NOW,
            timeout=3,
            runner=runner_for(official_document(symbols=allowed)),
        ).fetch_cn_stock_symbols()
        == allowed
    )
    with pytest.raises(DataQualityError, match="漂移过大"):
        BoundedOfficialExchangeStockMaster(
            expected_symbols=expected,
            clock=lambda: NOW,
            timeout=3,
            runner=runner_for(official_document(symbols=rejected)),
        ).fetch_cn_stock_symbols()


def frame(codes, *, code_column, date_column, listed="2020-01-01"):
    return pd.DataFrame({code_column: codes, date_column: [listed] * len(codes)})


def test_exchange_child_reads_and_normalizes_all_three_official_lists(monkeypatch):
    main_codes = [f"{code:06d}" for code in range(600_000, 601_500)]
    star_codes = [f"{code:06d}" for code in range(688_000, 688_600)]
    sz_codes = [f"{code:06d}" for code in range(0, 1_000)]
    sz_codes.extend(f"{code:06d}" for code in range(300_000, 301_000))
    akshare = SimpleNamespace(
        stock_info_sh_name_code=lambda symbol: frame(
            main_codes if symbol == "主板A股" else star_codes,
            code_column="证券代码",
            date_column="上市日期",
        ),
        stock_info_sz_name_code=lambda symbol: frame(
            sz_codes, code_column="A股代码", date_column="A股上市日期"
        ),
    )
    monkeypatch.setitem(sys.modules, "akshare", akshare)

    result = exchange_stock_master_read._read(NOW)

    assert result["counts"] == {"sse_main": 1_500, "sse_star": 600, "szse_a": 2_000}
    assert len(result["symbols"]) == 4_100
    assert result["symbols"] == tuple(sorted(result["symbols"]))
    assert result["retrieved_at"] == NOW.isoformat()
    assert result["provider"] == "sse_szse_official_via_akshare"
    assert result["method_version"] == "exchange-listed-a-v1"


@pytest.mark.parametrize(
    ("codes", "listed"),
    [(["600000", "600000"], "2020-01-01"), (["000001"], "2020-01-01"), (["600000"], "2027-01-01")],
)
def test_exchange_child_rejects_duplicate_wrong_exchange_or_future_listings(codes, listed):
    source = frame(codes, code_column="证券代码", date_column="上市日期", listed=listed)
    with pytest.raises(DataQualityError):
        exchange_stock_master_read._codes(
            source,
            code_column="证券代码",
            date_column="上市日期",
            pattern=exchange_stock_master_read._SH,
            suffix=".SH",
            now=NOW,
        )


def test_exchange_child_rejects_overlap_between_shanghai_boards(monkeypatch):
    same = frame(["600000"], code_column="证券代码", date_column="上市日期")
    sz = frame(["000001"], code_column="A股代码", date_column="A股上市日期")
    monkeypatch.setitem(
        sys.modules,
        "akshare",
        SimpleNamespace(
            stock_info_sh_name_code=lambda symbol: same,
            stock_info_sz_name_code=lambda symbol: sz,
        ),
    )
    with pytest.raises(DataQualityError, match="overlap"):
        exchange_stock_master_read._read(NOW)


def test_exchange_child_main_emits_one_clean_json_envelope(monkeypatch, capsys):
    expected = {"symbols": ["600000.SH"]}

    def noisy_read(_now):
        print("sdk stdout that must not escape")
        print("sdk stderr that must not escape", file=sys.stderr)
        return expected

    monkeypatch.setattr(exchange_stock_master_read, "_read", noisy_read)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"operation": "listed_a_share_symbols", "now": NOW.isoformat()})),
    )

    assert exchange_stock_master_read.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {"result": expected}


def test_exchange_child_main_classifies_quality_without_leaking_details(monkeypatch, capsys):
    def broken(_now):
        raise DataQualityError("sensitive exchange payload")

    monkeypatch.setattr(exchange_stock_master_read, "_read", broken)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"operation": "listed_a_share_symbols", "now": NOW.isoformat()})),
    )

    assert exchange_stock_master_read.main() == 2
    captured = capsys.readouterr()
    assert "sensitive" not in captured.out
    assert json.loads(captured.out) == {"error": "quality"}
