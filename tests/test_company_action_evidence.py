from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Event, Thread

import pytest

from ashare_lab import bootstrap
from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.cli.company_actions import main as company_actions_main
from ashare_lab.ports.company_actions import (
    CNINFO_COMPANY_ACTION_METHOD_VERSION,
    COMPANY_ACTION_COVERAGE_START,
    CompanyActionEvidence,
    CompanyActionEvidenceStatus,
    CompanyActionStreamReceipt,
)
from ashare_lab.services.company_action_evidence import (
    authorize_company_actions,
    is_company_action_authorized,
    refresh_and_load_company_action_clearances,
    revoke_company_actions,
)
from ashare_lab.services.holding_ledger import (
    HoldingPositionInput,
    clear_active_holdings,
    get_active_holding_portfolio,
    replace_active_holdings,
)

AS_OF = date(2026, 9, 15)
REVIEW_STARTED = datetime.now(UTC)
REQUIRED_STREAMS = ("identity", "dividend", "allotment", "share_change")


@pytest.fixture
def repository(tmp_path: Path) -> SQLiteRepository:
    repo = SQLiteRepository(
        tmp_path / "research.db",
        Path(__file__).resolve().parents[1] / "migrations",
    )
    repo.initialize()
    return repo


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "private" / "company-actions" / "config.json"
    authorize_company_actions(confirmed=True, config_path=path)
    return path


def _register(repository: SQLiteRepository, symbols: tuple[str, ...]) -> None:
    weight = 1.0 / len(symbols)
    replace_active_holdings(
        repository,
        tuple(
            HoldingPositionInput(
                symbol=symbol,
                name=f"PRIVATE-NAME-{index}",
                entry_date=date(2026, 9, 10),
                cost_price=10.0 + index,
                stock_sleeve_weight=weight,
                account_weight=weight / 2,
            )
            for index, symbol in enumerate(symbols)
        ),
        holding_weeks=4,
        effective_at=datetime(2026, 9, 10, 8, tzinfo=UTC),
        change_id=f"holding-{'-'.join(symbols)}",
    )


def _receipts(*, missing: str | None = None) -> tuple[CompanyActionStreamReceipt, ...]:
    return tuple(
        CompanyActionStreamReceipt(
            stream=stream,
            complete=True,
            record_count=0,
            response_hash=("a" * 64),
            reason_code="COMPLETE",
        )
        for stream in REQUIRED_STREAMS
        if stream != missing
    )


def _evidence(
    symbol: str,
    *,
    event_dates: tuple[date, ...] = (),
    receipts: tuple[CompanyActionStreamReceipt, ...] | None = None,
    status: CompanyActionEvidenceStatus | None = None,
    knowledge_time: datetime | None = None,
) -> CompanyActionEvidence:
    resolved_status = status or (
        CompanyActionEvidenceStatus.DETECTED if event_dates else CompanyActionEvidenceStatus.CLEAR
    )
    return CompanyActionEvidence(
        symbol=symbol,
        status=resolved_status,
        coverage_from=COMPANY_ACTION_COVERAGE_START,
        coverage_through=AS_OF,
        knowledge_time=knowledge_time or datetime.now(UTC),
        event_dates=event_dates,
        event_kinds=tuple("dividend" for _ in event_dates),
        stream_receipts=_receipts() if receipts is None else receipts,
        response_hash="f" * 64,
        reason_code=("EVENT_DETECTED" if event_dates else "COMPLETE_CLEAR"),
        method_version=CNINFO_COMPANY_ACTION_METHOD_VERSION,
    )


def _refresh(
    repository: SQLiteRepository,
    config_path: Path,
    fetcher,
    *,
    phase: str = "intraday",
):
    return refresh_and_load_company_action_clearances(
        repository,
        as_of=AS_OF,
        reviewed_at=REVIEW_STARTED,
        phase=phase,
        fetcher=fetcher,
        config_path=config_path,
    )


def test_authorization_is_narrow_private_idempotent_and_revocable(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    assert not is_company_action_authorized(config_path=path)
    with pytest.raises(ValueError, match="explicit_yes"):
        authorize_company_actions(confirmed=False, config_path=path)

    authorize_company_actions(confirmed=True, config_path=path)
    first = path.read_bytes()
    document = json.loads(first)
    assert document["enabled"] is True
    assert document["scope"] == "active_holding_symbols_only"
    assert document["authorized_fields"] == ["symbol"]
    assert document["orders_enabled"] is False
    assert datetime.fromisoformat(document["authorized_at"]).utcoffset() is not None
    assert path.stat().st_mode & 0o777 == 0o600
    assert is_company_action_authorized(config_path=path)

    authorize_company_actions(confirmed=True, config_path=path)
    assert path.read_bytes() == first

    revoke_company_actions(confirmed=True, config_path=path)
    assert not is_company_action_authorized(config_path=path)
    revoked = json.loads(path.read_bytes())
    assert revoked["orders_enabled"] is False
    assert revoked["authorized_at"] == document["authorized_at"]


def test_missing_or_broadened_authorization_never_calls_provider(
    repository: SQLiteRepository,
    tmp_path: Path,
) -> None:
    _register(repository, ("600919",))
    path = tmp_path / "config.json"
    calls = 0

    def fetcher(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return ()

    assert _refresh(repository, path, fetcher) == {}
    authorize_company_actions(confirmed=True, config_path=path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["authorized_fields"] = ["symbol", "cost_price"]
    path.write_text(json.dumps(document), encoding="utf-8")
    path.chmod(0o600)
    assert _refresh(repository, path, fetcher) == {}
    assert calls == 0


def test_only_each_six_digit_symbol_and_public_date_leave_device(
    repository: SQLiteRepository,
    config_path: Path,
) -> None:
    _register(repository, ("600919", "601919"))
    calls: list[tuple[tuple[str, ...], date, int]] = []

    def fetcher(symbols, as_of, *, timeout_seconds):
        calls.append((symbols, as_of, timeout_seconds))
        symbol = symbols[0]
        events = (date(2020, 1, 2),) if symbol == "600919" else (date(2026, 9, 14),)
        return (_evidence(symbol, event_dates=events),)

    clearances = _refresh(repository, config_path, fetcher)

    assert calls == [
        (("600919",), AS_OF, 8),
        (("601919",), AS_OF, 8),
    ]
    outbound = repr(calls)
    assert "PRIVATE" not in outbound and "10.0" not in outbound and "0.25" not in outbound
    assert clearances["600919"].clear is True
    assert clearances["601919"].clear is False
    assert clearances["600919"].knowledge_time is not None

    with repository.connection() as connection:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(company_action_evidence_attempts)")
        }
        rows = connection.execute(
            "SELECT symbol, status, coverage_from, events_json "
            "FROM company_action_evidence_attempts "
            "ORDER BY symbol"
        ).fetchall()
    prohibited_fragments = {
        "name",
        "cost",
        "quantity",
        "shares",
        "amount",
        "weight",
        "entry_date",
        "raw_response",
    }
    assert not {
        column for column in columns if any(fragment in column for fragment in prohibited_fragments)
    }
    assert dict(rows[0]) == {
        "symbol": "600919",
        "status": "clear",
        "coverage_from": "2026-09-10",
        "events_json": "[]",
    }


def test_same_position_day_phase_uses_append_only_cache(
    repository: SQLiteRepository,
    config_path: Path,
) -> None:
    _register(repository, ("600919",))
    calls = 0

    def fetcher(symbols, _as_of, *, timeout_seconds):
        nonlocal calls
        calls += 1
        assert timeout_seconds == 8
        return (_evidence(symbols[0]),)

    first = _refresh(repository, config_path, fetcher)
    second = _refresh(repository, config_path, fetcher)
    assert first == second
    assert calls == 1

    _refresh(repository, config_path, fetcher, phase="eod")
    assert calls == 2
    with repository.connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM company_action_evidence_attempts"
        ).fetchone()["count"]
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE company_action_evidence_attempts SET status='unknown'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM company_action_evidence_attempts")
    assert count == 2


def test_unknown_retries_after_backoff_and_then_caches_clearance(
    repository: SQLiteRepository,
    config_path: Path,
) -> None:
    _register(repository, ("600919",))
    calls = 0

    def fetcher(symbols, _as_of, *, timeout_seconds):
        nonlocal calls
        del timeout_seconds
        calls += 1
        if calls == 1:
            return (
                _evidence(
                    symbols[0],
                    status=CompanyActionEvidenceStatus.UNKNOWN,
                    receipts=(),
                    knowledge_time=datetime.now(UTC) - timedelta(minutes=16),
                ),
            )
        return (_evidence(symbols[0]),)

    assert _refresh(repository, config_path, fetcher) == {}
    assert set(_refresh(repository, config_path, fetcher)) == {"600919"}
    assert set(_refresh(repository, config_path, fetcher)) == {"600919"}
    assert calls == 2
    with repository.connection() as connection:
        attempts = [
            row["attempt"]
            for row in connection.execute(
                "SELECT attempt FROM company_action_evidence_attempts ORDER BY attempt"
            )
        ]
    assert attempts == [1, 2]


def test_unknown_retry_is_delayed_and_bounded_to_three_attempts(
    repository: SQLiteRepository,
    config_path: Path,
) -> None:
    _register(repository, ("600919",))
    calls = 0

    def recent_unknown(symbols, _as_of, *, timeout_seconds):
        nonlocal calls
        del timeout_seconds
        calls += 1
        return (
            _evidence(
                symbols[0],
                status=CompanyActionEvidenceStatus.UNKNOWN,
                receipts=(),
            ),
        )

    assert _refresh(repository, config_path, recent_unknown) == {}
    assert _refresh(repository, config_path, recent_unknown) == {}
    assert calls == 1

    # A separately archived old UNKNOWN may retry, but never beyond attempt 3.
    other = SQLiteRepository(
        repository.db_path.parent / "bounded.db",
        Path(__file__).resolve().parents[1] / "migrations",
    )
    other.initialize()
    _register(other, ("600919",))
    old_calls = 0

    def old_unknown(symbols, _as_of, *, timeout_seconds):
        nonlocal old_calls
        del timeout_seconds
        old_calls += 1
        return (
            _evidence(
                symbols[0],
                status=CompanyActionEvidenceStatus.UNKNOWN,
                receipts=(),
                knowledge_time=datetime.now(UTC) - timedelta(minutes=16),
            ),
        )

    for _ in range(4):
        assert _refresh(other, config_path, old_unknown) == {}
    assert old_calls == 3


def test_insert_or_replace_cannot_overwrite_archived_evidence(
    repository: SQLiteRepository,
    config_path: Path,
) -> None:
    _register(repository, ("600919",))
    _refresh(repository, config_path, lambda symbols, *_a, **_kw: (_evidence(symbols[0]),))

    with repository.connection() as connection:
        row = connection.execute("SELECT * FROM company_action_evidence_attempts").fetchone()
        columns = [
            item["name"]
            for item in connection.execute("PRAGMA table_info(company_action_evidence_attempts)")
        ]
        placeholders = ",".join("?" for _ in columns)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                f"INSERT OR REPLACE INTO company_action_evidence_attempts "
                f"({','.join(columns)}) VALUES ({placeholders})",
                tuple(row[column] for column in columns),
            )


def test_provider_method_versions_have_independent_attempt_namespaces(
    repository: SQLiteRepository,
) -> None:
    """A future provider parser can start at attempt one without collision."""

    common = (
        "portfolio",
        "position",
        1,
        "600919",
        AS_OF.isoformat(),
        "eod",
        1,
        "unknown",
        None,
        None,
        REVIEW_STARTED.isoformat(),
        "[]",
        None,
        "[]",
    )
    with repository.connection() as connection:
        for suffix, provider_version in (("a", "provider-v1"), ("b", "provider-v2")):
            connection.execute(
                """
                INSERT INTO company_action_evidence_attempts
                (evidence_id, portfolio_id, position_key, holding_version, symbol, as_of, phase,
                 attempt, status, coverage_from, coverage_through, knowledge_time, events_json,
                 provider_response_hash, stream_receipts_json, evidence_hash, reason_code,
                 provider_method_version, local_method_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"evidence-{suffix}",
                    *common,
                    suffix * 64,
                    "provider_unknown",
                    provider_version,
                    "local-v1",
                    REVIEW_STARTED.isoformat(),
                ),
            )
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM company_action_evidence_attempts"
        ).fetchone()["count"]
    assert count == 2


def test_concurrent_refresh_does_not_duplicate_external_disclosure(
    repository: SQLiteRepository,
    config_path: Path,
) -> None:
    _register(repository, ("600919",))
    started, release = Event(), Event()
    calls = []
    first_result = []

    def slow_fetcher(symbols, _as_of, *, timeout_seconds):
        calls.append(symbols)
        del timeout_seconds
        started.set()
        assert release.wait(timeout=3)
        return (_evidence(symbols[0]),)

    worker = Thread(
        target=lambda: first_result.append(_refresh(repository, config_path, slow_fetcher))
    )
    worker.start()
    assert started.wait(timeout=3)
    assert (
        _refresh(
            repository,
            config_path,
            lambda *_a, **_kw: pytest.fail("second refresh must not call provider"),
        )
        == {}
    )
    release.set()
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert calls == [("600919",)]
    assert set(first_result[0]) == {"600919"}


def test_canonical_holding_change_waits_for_inflight_provider_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A confirmed canonical mutation cannot return before an older read ends."""

    app_data = tmp_path / "app-data"
    monkeypatch.setattr(bootstrap, "application_data_dir", lambda: app_data)
    repository = SQLiteRepository(
        app_data / "research.db",
        Path(__file__).resolve().parents[1] / "migrations",
    )
    repository.initialize()
    authorize_company_actions(confirmed=True)
    _register(repository, ("600919",))

    fetch_started = Event()
    release_fetch = Event()
    mutation_started = Event()
    mutation_finished = Event()
    refresh_results: list[dict[str, object]] = []
    thread_errors: list[BaseException] = []

    def slow_fetcher(symbols, _as_of, *, timeout_seconds):
        del timeout_seconds
        fetch_started.set()
        assert release_fetch.wait(timeout=3)
        return (_evidence(symbols[0]),)

    def refresh_worker() -> None:
        try:
            refresh_results.append(
                refresh_and_load_company_action_clearances(
                    repository,
                    as_of=AS_OF,
                    reviewed_at=REVIEW_STARTED,
                    phase="intraday",
                    fetcher=slow_fetcher,
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            thread_errors.append(exc)

    def mutation_worker() -> None:
        try:
            mutation_started.set()
            clear_active_holdings(
                repository,
                effective_at=REVIEW_STARTED + timedelta(seconds=1),
                change_id="clear-after-provider-read",
            )
            mutation_finished.set()
        except BaseException as exc:  # pragma: no cover - surfaced below
            thread_errors.append(exc)

    refresh_thread = Thread(target=refresh_worker)
    refresh_thread.start()
    assert fetch_started.wait(timeout=3)

    mutation_thread = Thread(target=mutation_worker)
    mutation_thread.start()
    assert mutation_started.wait(timeout=3)
    assert not mutation_finished.wait(timeout=0.1)

    release_fetch.set()
    refresh_thread.join(timeout=3)
    mutation_thread.join(timeout=3)

    assert not refresh_thread.is_alive()
    assert not mutation_thread.is_alive()
    assert thread_errors == []
    assert set(refresh_results[0]) == {"600919"}
    current = get_active_holding_portfolio(repository)
    assert current is not None
    assert current.status == "cleared"
    assert current.positions == ()


def test_unknown_and_incomplete_pre_entry_event_never_become_false_or_clear(
    repository: SQLiteRepository,
    config_path: Path,
) -> None:
    _register(repository, ("600919", "601919"))

    def fetcher(symbols, _as_of, *, timeout_seconds):
        del timeout_seconds
        symbol = symbols[0]
        if symbol == "600919":
            return (
                _evidence(
                    symbol,
                    status=CompanyActionEvidenceStatus.UNKNOWN,
                    receipts=(),
                ),
            )
        return (
            _evidence(
                symbol,
                event_dates=(date(2020, 1, 2),),
                receipts=_receipts(missing="share_change"),
            ),
        )

    assert _refresh(repository, config_path, fetcher) == {}
    with repository.connection() as connection:
        statuses = [
            row["status"]
            for row in connection.execute(
                "SELECT status FROM company_action_evidence_attempts ORDER BY symbol"
            )
        ]
    assert statuses == ["unknown", "unknown"]


def test_one_symbol_exception_does_not_erase_other_symbol(
    repository: SQLiteRepository,
    config_path: Path,
) -> None:
    _register(repository, ("600919", "601919"))

    def fetcher(symbols, _as_of, *, timeout_seconds):
        del timeout_seconds
        if symbols == ("600919",):
            raise TimeoutError("private provider detail")
        return (_evidence(symbols[0]),)

    clearances = _refresh(repository, config_path, fetcher)
    assert set(clearances) == {"601919"}
    with repository.connection() as connection:
        statuses = {
            row["symbol"]: row["status"]
            for row in connection.execute(
                "SELECT symbol, status FROM company_action_evidence_attempts"
            )
        }
    assert statuses == {"600919": "unknown", "601919": "clear"}


def test_evidence_known_after_completed_load_time_remains_unknown_to_that_call(
    repository: SQLiteRepository,
    config_path: Path,
) -> None:
    _register(repository, ("600919",))
    future_knowledge = datetime(2099, 1, 1, tzinfo=UTC)

    clearances = _refresh(
        repository,
        config_path,
        lambda symbols, _as_of, *, timeout_seconds: (
            _evidence(
                symbols[0],
                knowledge_time=future_knowledge,
            ),
        ),
    )

    assert clearances == {}


def test_zero_and_more_than_eight_holdings_never_call_provider(
    repository: SQLiteRepository,
    config_path: Path,
) -> None:
    def forbidden(*_args, **_kwargs):
        pytest.fail("provider must not be called")

    assert _refresh(repository, config_path, forbidden) == {}
    _register(repository, tuple(f"6000{index:02}" for index in range(9)))
    assert _refresh(repository, config_path, forbidden) == {}
    with repository.connection() as connection:
        rows = connection.execute(
            "SELECT status, reason_code FROM company_action_evidence_attempts"
        ).fetchall()
    assert len(rows) == 9
    assert {(row["status"], row["reason_code"]) for row in rows} == {
        ("unknown", "holding_limit_exceeded")
    }


def test_exactly_eight_holdings_are_individually_verified(
    repository: SQLiteRepository,
    config_path: Path,
) -> None:
    symbols = tuple(f"6000{index:02}" for index in range(8))
    _register(repository, symbols)
    calls = []

    def fetcher(requested, _as_of, *, timeout_seconds):
        calls.append((requested, timeout_seconds))
        return (_evidence(requested[0]),)

    clearances = _refresh(repository, config_path, fetcher)

    assert set(clearances) == set(symbols)
    assert calls == [((symbol,), 8) for symbol in symbols]


def test_holding_or_authorization_change_during_fetch_discards_all_evidence(
    repository: SQLiteRepository,
    config_path: Path,
) -> None:
    _register(repository, ("600919", "601919"))
    calls = 0

    def changing_fetcher(symbols, _as_of, *, timeout_seconds):
        nonlocal calls
        del timeout_seconds
        calls += 1
        replace_active_holdings(
            repository,
            (
                HoldingPositionInput(
                    symbol="000001",
                    name="NEW-PRIVATE-NAME",
                    entry_date=date(2026, 9, 10),
                    stock_sleeve_weight=1.0,
                ),
            ),
            holding_weeks=4,
            effective_at=REVIEW_STARTED,
            change_id="changed-during-company-action-fetch",
        )
        return (_evidence(symbols[0]),)

    assert _refresh(repository, config_path, changing_fetcher) == {}
    assert calls == 1
    with repository.connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) AS count FROM company_action_evidence_attempts"
            ).fetchone()["count"]
            == 0
        )

    # A grant change during a separate run is also discarded.
    second_repo = SQLiteRepository(
        repository.db_path.parent / "second.db",
        Path(__file__).resolve().parents[1] / "migrations",
    )
    second_repo.initialize()
    _register(second_repo, ("600919",))
    authorize_company_actions(confirmed=True, config_path=config_path)

    def revoking_fetcher(symbols, _as_of, *, timeout_seconds):
        del timeout_seconds
        revoke_company_actions(confirmed=True, config_path=config_path)
        return (_evidence(symbols[0]),)

    assert _refresh(second_repo, config_path, revoking_fetcher) == {}
    with second_repo.connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) AS count FROM company_action_evidence_attempts"
            ).fetchone()["count"]
            == 0
        )


def test_global_authorization_does_not_enable_an_unrelated_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_data = tmp_path / "app-data"
    monkeypatch.setattr(bootstrap, "application_data_dir", lambda: app_data)
    authorize_company_actions(confirmed=True)
    unrelated = SQLiteRepository(
        tmp_path / "unrelated.db",
        Path(__file__).resolve().parents[1] / "migrations",
    )
    unrelated.initialize()
    _register(unrelated, ("600919",))
    calls = 0

    def fetcher(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return ()

    result = refresh_and_load_company_action_clearances(
        unrelated,
        as_of=AS_OF,
        reviewed_at=REVIEW_STARTED,
        phase="intraday",
        fetcher=fetcher,
    )
    assert result == {} and calls == 0


def test_cli_authorize_status_and_revoke_are_safe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "cli-config.json"
    assert company_actions_main(["authorize"], _config_path=path) == 2
    assert not path.exists()
    capsys.readouterr()

    assert company_actions_main(["authorize", "--yes"], _config_path=path) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["enabled"] is True
    assert payload["authorized_fields"] == ["symbol"]
    assert payload["orders_enabled"] is False

    assert company_actions_main(["status"], _config_path=path) == 0
    assert json.loads(capsys.readouterr().out)["enabled"] is True
    assert company_actions_main(["revoke", "--yes"], _config_path=path) == 0
    assert json.loads(capsys.readouterr().out)["enabled"] is False


def test_cli_refresh_with_no_holdings_is_quiet_and_does_not_call_provider(
    repository: SQLiteRepository,
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*_args, **_kwargs):
        pytest.fail("empty holdings must not call the provider")

    assert (
        company_actions_main(
            ["refresh", "--as-of", AS_OF.isoformat(), "--phase", "eod"],
            _repository=repository,
            _config_path=config_path,
            _fetcher=forbidden,
            _now=REVIEW_STARTED,
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["refresh_status"] == "no_active_holdings"
    assert payload["known_clearance_count"] == 0
    assert payload["authorized_fields"] == ["symbol"]
