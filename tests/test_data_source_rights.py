from __future__ import annotations

import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ashare_lab.domain.data_sources import (
    AuthorizationBasis,
    DataAction,
    ProviderEnvelope,
    RightsPolicy,
    RightsViolationError,
    SourceDefinition,
    SourceId,
    SourceRegistry,
    SourceStatus,
)
from ashare_lab.domain.errors import DataQualityError


def _envelope(**changes: object) -> ProviderEnvelope[dict[str, float]]:
    base = datetime(2026, 8, 21, 10, 0, tzinfo=UTC)
    values: dict[str, object] = {
        "data": {"close": 38.26},
        "provider": SourceId.IFIND,
        "dataset": "daily_price",
        "request_id": "request-001",
        "entitlement": "licensed-market-data",
        "event_at": base,
        "published_at": base + timedelta(minutes=5),
        "provider_available_at": base + timedelta(minutes=8),
        "first_seen_at": base + timedelta(minutes=10),
        "retrieved_at": base + timedelta(minutes=11),
        "ingested_at": base + timedelta(minutes=12),
        "rights_scope": frozenset({DataAction.MARKET_DATA_READ}),
    }
    values.update(changes)
    return ProviderEnvelope(**values)  # type: ignore[arg-type]


def _authorized_registry(*actions: DataAction) -> SourceRegistry:
    return SourceRegistry(
        [
            SourceDefinition(
                source_id=SourceId.IFIND,
                display_name="同花顺 iFinD",
                status=SourceStatus.CONNECTED,
                purposes=("授权研究",),
                official_url="https://quantapi.10jqka.com.cn/",
                application_url="https://quantapi.10jqka.com.cn/",
                authorization_basis=AuthorizationBasis.ACCOUNT_ENTITLEMENT,
                authorization_reference="contract-local-reference",
                allowed_actions=frozenset(actions),
                configurable=True,
            )
        ]
    )


def test_default_registry_has_expected_fail_closed_states() -> None:
    registry = SourceRegistry.load_default()
    assert {source.source_id for source in registry.all()} == set(SourceId)
    assert registry.get(SourceId.IFIND).status is SourceStatus.NOT_CONNECTED
    assert registry.get(SourceId.CHOICE).status is SourceStatus.NOT_CONNECTED
    assert registry.get(SourceId.TUSHARE).status is SourceStatus.EXPERIMENTAL
    assert registry.get(SourceId.BAOSTOCK).status is SourceStatus.EXPERIMENTAL
    assert registry.get(SourceId.ZERO_BUDGET_EOD).status is SourceStatus.EXPERIMENTAL
    assert registry.get(SourceId.INFOWAY).status is SourceStatus.EXPERIMENTAL
    assert registry.get(SourceId.CLS).status is SourceStatus.BLOCKED_REQUIRE_WRITTEN_AUTHORIZATION
    assert registry.get(SourceId.STCN).status is SourceStatus.BLOCKED_REQUIRE_WRITTEN_AUTHORIZATION
    assert registry.get(SourceId.AKSHARE).status is SourceStatus.EXPERIMENTAL
    assert registry.get(SourceId.YAHOO).status is SourceStatus.PRICE_BACKUP


def test_not_connected_commercial_sources_are_denied() -> None:
    policy = RightsPolicy(SourceRegistry.load_default())
    for source_id in (SourceId.IFIND, SourceId.CHOICE):
        with pytest.raises(RightsViolationError):
            policy.require(source_id, DataAction.MARKET_DATA_READ)


@pytest.mark.parametrize("source_id", [SourceId.CLS, SourceId.STCN])
@pytest.mark.parametrize(
    "action",
    [
        DataAction.BODY_PERSIST,
        DataAction.BODY_TO_LLM,
        DataAction.BODY_DISPLAY_UI,
        DataAction.BODY_EXPORT,
    ],
)
def test_news_bodies_are_blocked_without_written_authorization(
    source_id: SourceId, action: DataAction
) -> None:
    policy = RightsPolicy(SourceRegistry.load_default())
    with pytest.raises(RightsViolationError):
        policy.require(source_id, action)


@pytest.mark.parametrize(
    "source_id",
    [
        SourceId.TUSHARE,
        SourceId.BAOSTOCK,
        SourceId.ZERO_BUDGET_EOD,
        SourceId.AKSHARE,
        SourceId.YAHOO,
    ],
)
def test_personal_research_price_sources_are_narrowly_scoped(source_id: SourceId) -> None:
    policy = RightsPolicy(SourceRegistry.load_default())
    assert policy.require(source_id, DataAction.MARKET_DATA_READ).source_id is source_id
    with pytest.raises(RightsViolationError):
        policy.require(source_id, DataAction.BODY_TO_LLM)


def test_unknown_source_and_action_fail_closed() -> None:
    policy = RightsPolicy(SourceRegistry.load_default())
    with pytest.raises(RightsViolationError):
        policy.require("unregistered", DataAction.MARKET_DATA_READ)
    with pytest.raises(RightsViolationError):
        policy.require(SourceId.AKSHARE, "unregistered_action")


def test_authorized_body_action_requires_connection_reference_and_explicit_scope() -> None:
    policy = RightsPolicy(_authorized_registry(DataAction.BODY_PERSIST, DataAction.BODY_TO_LLM))
    assert policy.require(SourceId.IFIND, DataAction.BODY_TO_LLM).status is SourceStatus.CONNECTED
    with pytest.raises(RightsViolationError):
        policy.require(SourceId.IFIND, DataAction.BODY_EXPORT)


def test_envelope_rights_can_only_narrow_registry_rights() -> None:
    policy = RightsPolicy(_authorized_registry(DataAction.MARKET_DATA_READ, DataAction.BODY_TO_LLM))
    envelope = _envelope(rights_scope=frozenset({DataAction.MARKET_DATA_READ}))
    assert policy.require_envelope(envelope, DataAction.MARKET_DATA_READ)
    with pytest.raises(RightsViolationError):
        policy.require_envelope(envelope, DataAction.BODY_TO_LLM)


def test_knowledge_at_and_no_lookahead_gate() -> None:
    envelope = _envelope()
    expected = datetime(2026, 8, 21, 10, 10, tzinfo=UTC)
    assert envelope.knowledge_at == expected
    assert not envelope.is_known_at(expected - timedelta(seconds=1))
    assert envelope.is_known_at(expected)
    with pytest.raises(DataQualityError):
        envelope.require_known_at(expected - timedelta(seconds=1))
    envelope.require_known_at(expected)


def test_timestamps_are_timezone_aware_and_in_causal_order() -> None:
    with pytest.raises(DataQualityError, match="timezone-aware"):
        _envelope(first_seen_at=datetime(2026, 8, 21, 10, 10))
    with pytest.raises(DataQualityError, match="provider_available_at"):
        _envelope(
            provider_available_at=datetime(2026, 8, 21, 10, 30, tzinfo=UTC),
        )


def test_non_point_in_time_payload_is_rejected_for_historical_use() -> None:
    envelope = _envelope(point_in_time=False)
    assert not envelope.is_known_at(envelope.ingested_at)
    with pytest.raises(DataQualityError, match="not point-in-time safe"):
        envelope.require_known_at(envelope.ingested_at)


def test_registry_and_page_contain_no_credential_inputs() -> None:
    config_path = SourceRegistry.default_config_path()
    with config_path.open("rb") as handle:
        document = tomllib.load(handle)
    forbidden_fragments = ("api_key", "token", "password", "secret")
    for definition in document["sources"].values():
        assert not any(
            fragment in field.lower() for field in definition for fragment in forbidden_fragments
        )

    page_path = Path(__file__).parents[1] / "src/ashare_lab/ui/pages/05_数据来源.py"
    page_source = page_path.read_text(encoding="utf-8")
    assert "text_input" not in page_source
    assert "text_area" not in page_source
    assert "file_uploader" not in page_source
    assert "查看申请／合作入口" in page_source
