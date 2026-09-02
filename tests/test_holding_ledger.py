from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.services.holding_ledger import (
    HOLDING_CHART_DELIVERY_CHANNELS_KEY,
    HOLDING_CHART_PUBLISHER_ID_KEY,
    HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY,
    HoldingPositionInput,
    clear_active_holdings,
    get_active_holding_portfolio,
    holding_chart_delivery_channels,
    holding_chart_publisher_id,
    holding_knowledge_context,
    holding_summary_delivery_channels,
    list_active_holdings,
    replace_active_holdings,
    resolve_current_holding_context,
)


@pytest.fixture
def repository(tmp_path: Path) -> SQLiteRepository:
    repo = SQLiteRepository(
        tmp_path / "research.db",
        Path(__file__).resolve().parents[1] / "migrations",
    )
    repo.initialize()
    return repo


def _positions() -> tuple[HoldingPositionInput, ...]:
    return (
        HoldingPositionInput(
            symbol="600919.SH",
            name="江苏银行",
            entry_date=date(2026, 8, 28),
            cost_price=None,
            stock_sleeve_weight=0.6,
            account_weight=None,
        ),
        HoldingPositionInput(
            symbol="601919",
            name="中远海控",
            entry_date=date(2026, 8, 28),
            cost_price=15.20,
            stock_sleeve_weight=0.4,
            account_weight=0.12,
        ),
    )


def test_explicit_snapshot_persists_until_another_explicit_change(
    repository: SQLiteRepository,
) -> None:
    first = replace_active_holdings(
        repository,
        _positions(),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 28, 21, 0, tzinfo=UTC),
        change_id="user-message-1",
    )

    assert first.version == 1
    assert first.holding_weeks == 4
    assert len(list_active_holdings(repository)) == 2
    unknown = next(item for item in first.positions if item.symbol == "600919")
    assert unknown.cost_price is None
    assert unknown.account_weight is None
    assert first.metadata[HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY] == []
    assert first.metadata[HOLDING_CHART_DELIVERY_CHANNELS_KEY] == []
    assert first.metadata[HOLDING_CHART_PUBLISHER_ID_KEY] is None
    assert holding_summary_delivery_channels(first) == frozenset()
    assert holding_chart_delivery_channels(first) == frozenset()
    assert holding_chart_publisher_id(first) is None

    # Reading and later reviews do not create a revision or alter the horizon.
    again = get_active_holding_portfolio(repository)
    assert again == first
    assert repository.next_holding_snapshot_version() == 2

    second = replace_active_holdings(
        repository,
        (
            HoldingPositionInput(
                symbol="600919",
                name="江苏银行",
                entry_date=date(2026, 8, 28),
                stock_sleeve_weight=1.0,
            ),
        ),
        holding_weeks=13,
        effective_at=datetime(2026, 8, 31, 21, 0, tzinfo=UTC),
        change_id="user-message-2",
    )
    assert second.version == 2
    assert second.holding_weeks == 13
    assert [item.symbol for item in list_active_holdings(repository)] == ["600919"]

    with repository.connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM holding_portfolio_revisions"
        ).fetchone()["count"]
    assert count == 2


def test_historical_snapshot_read_excludes_later_effective_revision(
    repository: SQLiteRepository,
) -> None:
    first = replace_active_holdings(
        repository,
        _positions(),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 28, 21, 0, tzinfo=UTC),
        change_id="historical-first",
    )
    second = replace_active_holdings(
        repository,
        (
            HoldingPositionInput(
                symbol="600919",
                name="江苏银行",
                entry_date=date(2026, 8, 28),
                stock_sleeve_weight=1.0,
            ),
        ),
        holding_weeks=13,
        effective_at=datetime(2026, 8, 31, 21, 0, tzinfo=UTC),
        change_id="historical-second",
    )

    replay = get_active_holding_portfolio(repository, as_of=date(2026, 8, 28))

    assert replay == first
    assert get_active_holding_portfolio(repository) == second


def test_frozen_holding_context_rejects_a_later_current_revision(
    repository: SQLiteRepository,
) -> None:
    first = replace_active_holdings(
        repository,
        _positions(),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 28, 21, 0, tzinfo=UTC),
        change_id="context-first",
    )
    context = holding_knowledge_context(
        first,
        known_at=datetime(2026, 8, 28, 22, 0, tzinfo=UTC),
    )

    assert resolve_current_holding_context(repository, context) == first

    replace_active_holdings(
        repository,
        (
            HoldingPositionInput(
                symbol="600919",
                name="江苏银行",
                entry_date=date(2026, 8, 28),
                stock_sleeve_weight=1.0,
            ),
        ),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 29, 21, 0, tzinfo=UTC),
        change_id="context-second",
    )

    with pytest.raises(ValueError, match="no longer current"):
        resolve_current_holding_context(repository, context)


def test_holding_knowledge_time_must_be_aware_and_not_predate_revision(
    repository: SQLiteRepository,
) -> None:
    portfolio = replace_active_holdings(
        repository,
        _positions(),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 28, 21, 0, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        holding_knowledge_context(
            portfolio,
            known_at=datetime(2026, 8, 28, 22, 0),
        )
    with pytest.raises(ValueError, match="newer than the live knowledge time"):
        holding_knowledge_context(
            portfolio,
            known_at=datetime(2026, 8, 28, 20, 59, tzinfo=UTC),
        )


def test_clear_requires_explicit_call_and_keeps_prior_revision(
    repository: SQLiteRepository,
) -> None:
    replace_active_holdings(
        repository,
        _positions(),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 28, 21, 0, tzinfo=UTC),
    )

    cleared = clear_active_holdings(
        repository,
        effective_at=datetime(2026, 8, 31, 21, 0, tzinfo=UTC),
    )

    assert cleared.status == "cleared"
    assert cleared.holding_weeks == 4
    assert cleared.positions == ()
    assert list_active_holdings(repository) == ()
    with repository.connection() as connection:
        old_positions = connection.execute(
            "SELECT COUNT(*) AS count FROM holding_positions"
        ).fetchone()["count"]
    assert old_positions == 2


def test_unknown_values_are_allowed_but_weights_are_never_guessed(
    repository: SQLiteRepository,
) -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        replace_active_holdings(
            repository,
            (
                HoldingPositionInput(
                    symbol="600919",
                    name="江苏银行",
                    entry_date=date(2026, 8, 28),
                    stock_sleeve_weight=0.8,
                ),
            ),
            holding_weeks=4,
            effective_at=datetime(2026, 8, 28, 21, 0, tzinfo=UTC),
        )

    assert get_active_holding_portfolio(repository) is None


def test_change_id_retry_is_idempotent_but_conflicting_payload_is_rejected(
    repository: SQLiteRepository,
) -> None:
    arguments = {
        "holding_weeks": 4,
        "effective_at": datetime(2026, 8, 28, 21, 0, tzinfo=UTC),
        "change_id": "same-user-message",
    }
    first = replace_active_holdings(repository, _positions(), **arguments)
    retry = replace_active_holdings(repository, _positions(), **arguments)

    assert retry == first
    assert repository.next_holding_snapshot_version() == 2

    with pytest.raises(ValueError, match="different holding"):
        replace_active_holdings(
            repository,
            (
                HoldingPositionInput(
                    symbol="600919",
                    name="江苏银行",
                    entry_date=date(2026, 8, 28),
                    stock_sleeve_weight=1.0,
                ),
            ),
            **arguments,
        )


def test_clear_change_id_retry_is_idempotent(repository: SQLiteRepository) -> None:
    replace_active_holdings(
        repository,
        _positions(),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 28, 21, 0, tzinfo=UTC),
    )
    arguments = {
        "effective_at": datetime(2026, 8, 31, 21, 0, tzinfo=UTC),
        "change_id": "same-clear-message",
        "metadata": {"reason": "user_explicit"},
    }

    first = clear_active_holdings(repository, **arguments)
    retry = clear_active_holdings(repository, **arguments)

    assert retry == first
    assert repository.next_holding_snapshot_version() == 3


def test_holding_summary_delivery_requires_explicit_supported_channel_list(
    repository: SQLiteRepository,
) -> None:
    portfolio = replace_active_holdings(
        repository,
        _positions(),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 28, 21, tzinfo=UTC),
        metadata={
            "external_delivery_consent": True,
            HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY: ["bark", "serverchan", "bark"],
        },
    )

    assert "external_delivery_consent" not in portfolio.metadata
    assert portfolio.metadata[HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY] == [
        "serverchan",
        "bark",
    ]
    assert holding_summary_delivery_channels(portfolio) == frozenset({"serverchan", "bark"})


def test_legacy_boolean_never_authorizes_a_delivery_channel(
    repository: SQLiteRepository,
) -> None:
    portfolio = replace_active_holdings(
        repository,
        _positions(),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 28, 21, tzinfo=UTC),
        metadata={"external_delivery_consent": True},
    )

    assert holding_summary_delivery_channels(portfolio) == frozenset()
    assert portfolio.metadata[HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY] == []


def test_chart_delivery_requires_independent_channels_and_publisher(
    repository: SQLiteRepository,
) -> None:
    portfolio = replace_active_holdings(
        repository,
        _positions(),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 28, 21, 0, tzinfo=UTC),
        metadata={
            HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY: ["serverchan", "bark"],
            HOLDING_CHART_DELIVERY_CHANNELS_KEY: ["serverchan"],
            HOLDING_CHART_PUBLISHER_ID_KEY: "cloudflare_r2",
        },
    )

    assert holding_summary_delivery_channels(portfolio) == frozenset({"serverchan", "bark"})
    assert holding_chart_delivery_channels(portfolio) == frozenset({"serverchan"})
    assert holding_chart_publisher_id(portfolio) == "cloudflare_r2"


@pytest.mark.parametrize(
    ("metadata_key", "value"),
    [
        (HOLDING_CHART_DELIVERY_CHANNELS_KEY, "serverchan"),
        (HOLDING_CHART_DELIVERY_CHANNELS_KEY, ["email"]),
        (HOLDING_CHART_DELIVERY_CHANNELS_KEY, [True]),
        (HOLDING_CHART_PUBLISHER_ID_KEY, "unknown-publisher"),
        (HOLDING_CHART_PUBLISHER_ID_KEY, True),
    ],
)
def test_invalid_chart_delivery_metadata_fails_closed(
    repository: SQLiteRepository,
    metadata_key: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="holding|Unsupported|provider"):
        replace_active_holdings(
            repository,
            _positions(),
            holding_weeks=4,
            effective_at=datetime(2026, 8, 28, 21, 0, tzinfo=UTC),
            metadata={metadata_key: value},
        )

    assert get_active_holding_portfolio(repository) is None


@pytest.mark.parametrize(
    "channels",
    ["serverchan", ["email"], [True], {"serverchan": True}],
)
def test_invalid_holding_summary_channel_metadata_fails_closed(
    repository: SQLiteRepository,
    channels: object,
) -> None:
    with pytest.raises(ValueError, match="holding|Unsupported"):
        replace_active_holdings(
            repository,
            _positions(),
            holding_weeks=4,
            effective_at=datetime(2026, 8, 28, 21, tzinfo=UTC),
            metadata={HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY: channels},
        )

    assert get_active_holding_portfolio(repository) is None


def test_holding_snapshot_cas_rejects_a_stale_displayed_revision(
    repository: SQLiteRepository,
) -> None:
    first = replace_active_holdings(
        repository,
        _positions(),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 28, 21, tzinfo=UTC),
        metadata={HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY: ["serverchan", "bark"]},
    )
    second = replace_active_holdings(
        repository,
        _positions(),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 29, 21, tzinfo=UTC),
        metadata={
            HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY: ["serverchan", "bark"],
            "unrelated_user_update": "preserve-me",
        },
        expected_current_revision_id=first.id,
        expected_current_version=first.version,
    )

    with pytest.raises(ValueError, match="changed; reload"):
        replace_active_holdings(
            repository,
            _positions(),
            holding_weeks=4,
            effective_at=datetime(2026, 8, 30, 21, tzinfo=UTC),
            metadata={HOLDING_CHART_DELIVERY_CHANNELS_KEY: ["serverchan"]},
            expected_current_revision_id=first.id,
            expected_current_version=first.version,
        )

    assert get_active_holding_portfolio(repository) == second
    assert repository.next_holding_snapshot_version() == 3


def test_holding_snapshot_cas_requires_id_and_version_together(
    repository: SQLiteRepository,
) -> None:
    with pytest.raises(ValueError, match="requires both"):
        replace_active_holdings(
            repository,
            _positions(),
            holding_weeks=4,
            effective_at=datetime(2026, 8, 28, 21, tzinfo=UTC),
            expected_current_revision_id="revision-only",
        )
