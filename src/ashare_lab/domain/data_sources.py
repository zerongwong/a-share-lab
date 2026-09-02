"""Data-source provenance, point-in-time metadata and content-rights gates.

The registry deliberately contains no credentials.  Connection secrets belong in
an operating-system secret store or a provider SDK configuration, never in this
module, the TOML registry, a research record, or the Streamlit UI.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from ashare_lab.domain.errors import DataQualityError, FeatureDisabledError


class SourceId(StrEnum):
    """Stable identifiers used in provenance records."""

    IFIND = "ifind"
    CHOICE = "choice"
    TUSHARE = "tushare"
    BAOSTOCK = "baostock"
    ZERO_BUDGET_EOD = "zero_budget_eod"
    CSMAR = "csmar"
    INFOWAY = "infoway"
    CLS = "cls"
    STCN = "stcn"
    AKSHARE = "akshare"
    YAHOO = "yahoo"


# All live research consumers read one provider-isolated overlay chain.  The
# former Infoway chain remains on disk for audit/history, but new reports must
# not silently mix rows from two providers.
DEFAULT_MARKET_OVERLAY_SOURCE_ID = SourceId.ZERO_BUDGET_EOD.value


class SourceStatus(StrEnum):
    """Operational and legal state of a configured source."""

    NOT_CONNECTED = "not_connected"
    CONNECTED = "connected"
    BLOCKED_REQUIRE_WRITTEN_AUTHORIZATION = "blocked_require_written_authorization"
    EXPERIMENTAL = "experimental"
    PRICE_BACKUP = "price_backup"


class AuthorizationBasis(StrEnum):
    """Basis on which the application may process a source."""

    ACCOUNT_ENTITLEMENT = "account_entitlement"
    WRITTEN_AUTHORIZATION = "written_authorization"
    PERSONAL_RESEARCH_ONLY = "personal_research_only"


class DataAction(StrEnum):
    """Rights are granted per action; access to one action implies no others."""

    MARKET_DATA_READ = "market_data_read"
    MARKET_DATA_CACHE = "market_data_cache"
    FUNDAMENTAL_DATA_READ = "fundamental_data_read"
    METADATA_READ = "metadata_read"
    BODY_READ = "body_read"
    BODY_PERSIST = "body_persist"
    BODY_TO_LLM = "body_to_llm"
    BODY_DISPLAY_UI = "body_display_ui"
    BODY_EXPORT = "body_export"
    REDISTRIBUTE = "redistribute"


BODY_ACTIONS = frozenset(
    {
        DataAction.BODY_READ,
        DataAction.BODY_PERSIST,
        DataAction.BODY_TO_LLM,
        DataAction.BODY_DISPLAY_UI,
        DataAction.BODY_EXPORT,
        DataAction.REDISTRIBUTE,
    }
)


class RightsViolationError(FeatureDisabledError):
    """The requested operation is not covered by an explicit source right."""


def _require_https(value: str, field_name: str) -> None:
    if not value.startswith("https://"):
        raise ValueError(f"{field_name} must use https://")


@dataclass(frozen=True, slots=True)
class SourceDefinition:
    """Public, non-secret configuration for one provider."""

    source_id: SourceId
    display_name: str
    status: SourceStatus
    purposes: tuple[str, ...]
    official_url: str
    application_url: str
    authorization_basis: AuthorizationBasis
    allowed_actions: frozenset[DataAction]
    authorization_reference: str | None = None
    configurable: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.display_name.strip():
            raise ValueError("display_name cannot be blank")
        if not self.purposes or any(not purpose.strip() for purpose in self.purposes):
            raise ValueError("at least one non-empty purpose is required")
        _require_https(self.official_url, "official_url")
        _require_https(self.application_url, "application_url")

        reference = self.authorization_reference
        if reference is not None:
            normalized = reference.strip()
            object.__setattr__(self, "authorization_reference", normalized or None)


class SourceRegistry:
    """Read-only source registry loaded from a checked-in, credential-free TOML file."""

    _KNOWN_FIELDS = frozenset(
        {
            "display_name",
            "status",
            "purposes",
            "official_url",
            "application_url",
            "authorization_basis",
            "allowed_actions",
            "authorization_reference",
            "configurable",
            "notes",
        }
    )

    def __init__(self, sources: Iterable[SourceDefinition]) -> None:
        by_id: dict[SourceId, SourceDefinition] = {}
        for source in sources:
            if source.source_id in by_id:
                raise ValueError(f"duplicate source_id: {source.source_id}")
            by_id[source.source_id] = source
        if not by_id:
            raise ValueError("source registry cannot be empty")
        self._by_id = by_id

    @classmethod
    def default_config_path(cls) -> Path:
        return Path(__file__).resolve().parents[3] / "config" / "data_sources.toml"

    @classmethod
    def load_default(cls) -> SourceRegistry:
        return cls.from_toml(cls.default_config_path())

    @classmethod
    def from_toml(cls, path: str | Path) -> SourceRegistry:
        config_path = Path(path)
        with config_path.open("rb") as handle:
            document = tomllib.load(handle)

        if document.get("schema_version") != 1:
            raise ValueError("unsupported data-source registry schema")
        source_tables = document.get("sources")
        if not isinstance(source_tables, dict) or not source_tables:
            raise ValueError("data-source registry must contain [sources.*] tables")

        definitions: list[SourceDefinition] = []
        for raw_id, raw_definition in source_tables.items():
            if not isinstance(raw_definition, dict):
                raise ValueError(f"sources.{raw_id} must be a table")
            unknown_fields = set(raw_definition) - cls._KNOWN_FIELDS
            if unknown_fields:
                names = ", ".join(sorted(unknown_fields))
                raise ValueError(f"unsupported fields in sources.{raw_id}: {names}")
            try:
                definition = SourceDefinition(
                    source_id=SourceId(raw_id),
                    display_name=str(raw_definition["display_name"]),
                    status=SourceStatus(raw_definition["status"]),
                    purposes=tuple(str(value) for value in raw_definition["purposes"]),
                    official_url=str(raw_definition["official_url"]),
                    application_url=str(raw_definition["application_url"]),
                    authorization_basis=AuthorizationBasis(raw_definition["authorization_basis"]),
                    allowed_actions=frozenset(
                        DataAction(value) for value in raw_definition["allowed_actions"]
                    ),
                    authorization_reference=raw_definition.get("authorization_reference"),
                    configurable=bool(raw_definition.get("configurable", False)),
                    notes=str(raw_definition.get("notes", "")),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid data-source definition: sources.{raw_id}") from exc
            definitions.append(definition)
        return cls(definitions)

    def get(self, source_id: SourceId | str) -> SourceDefinition:
        try:
            normalized = source_id if isinstance(source_id, SourceId) else SourceId(source_id)
            return self._by_id[normalized]
        except (KeyError, ValueError) as exc:
            raise RightsViolationError(f"未知数据来源，已按默认拒绝处理：{source_id}") from exc

    def all(self) -> tuple[SourceDefinition, ...]:
        return tuple(self._by_id.values())


class RightsPolicy:
    """Fail-closed policy for source use, especially licensed article bodies."""

    def __init__(self, registry: SourceRegistry) -> None:
        self._registry = registry

    def require(self, source_id: SourceId | str, action: DataAction | str) -> SourceDefinition:
        source = self._registry.get(source_id)
        try:
            normalized_action = action if isinstance(action, DataAction) else DataAction(action)
        except ValueError as exc:
            raise RightsViolationError(f"未知数据动作，已按默认拒绝处理：{action}") from exc

        if source.status is SourceStatus.NOT_CONNECTED:
            raise RightsViolationError(
                f"{source.display_name} 尚未连接，不能执行 {normalized_action}"
            )
        if source.status is SourceStatus.BLOCKED_REQUIRE_WRITTEN_AUTHORIZATION:
            raise RightsViolationError(
                f"{source.display_name} 需要书面授权，当前不能执行 {normalized_action}"
            )
        if normalized_action not in source.allowed_actions:
            raise RightsViolationError(
                f"{source.display_name} 未明确授予动作 {normalized_action}，已拒绝"
            )

        if normalized_action in BODY_ACTIONS:
            authorized_basis = source.authorization_basis in {
                AuthorizationBasis.ACCOUNT_ENTITLEMENT,
                AuthorizationBasis.WRITTEN_AUTHORIZATION,
            }
            if (
                source.status is not SourceStatus.CONNECTED
                or not authorized_basis
                or not source.authorization_reference
            ):
                raise RightsViolationError(
                    f"{source.display_name} 正文没有可核验授权，禁止落盘、发送模型、展示或导出"
                )

        if (
            source.status is SourceStatus.CONNECTED
            and source.authorization_basis
            in {
                AuthorizationBasis.ACCOUNT_ENTITLEMENT,
                AuthorizationBasis.WRITTEN_AUTHORIZATION,
            }
            and not source.authorization_reference
        ):
            raise RightsViolationError(f"{source.display_name} 缺少可核验的授权依据")
        return source

    def require_envelope(
        self,
        envelope: ProviderEnvelope[object],
        action: DataAction | str,
    ) -> SourceDefinition:
        try:
            normalized_action = action if isinstance(action, DataAction) else DataAction(action)
        except ValueError as exc:
            raise RightsViolationError(f"未知数据动作，已按默认拒绝处理：{action}") from exc
        source = self.require(envelope.provider, normalized_action)
        if normalized_action not in envelope.rights_scope:
            raise RightsViolationError(
                f"本次 {envelope.dataset} 数据包不包含动作 {normalized_action} 的权利"
            )
        return source


def _require_aware(value: datetime | None, field_name: str) -> None:
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise DataQualityError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class ProviderEnvelope[T]:
    """Provider payload plus provenance, rights and point-in-time timestamps."""

    data: T
    provider: SourceId
    dataset: str
    request_id: str
    entitlement: str
    first_seen_at: datetime
    retrieved_at: datetime
    ingested_at: datetime
    rights_scope: frozenset[DataAction]
    event_at: datetime | None = None
    published_at: datetime | None = None
    provider_available_at: datetime | None = None
    source_updated_at: datetime | None = None
    quality: str = "unknown"
    cache_status: str = "live"
    schema_version: str = "1"
    checksum: str = ""
    point_in_time: bool = True

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise DataQualityError("dataset cannot be blank")
        if not self.request_id.strip():
            raise DataQualityError("request_id cannot be blank")
        for field_name in (
            "event_at",
            "published_at",
            "provider_available_at",
            "first_seen_at",
            "retrieved_at",
            "ingested_at",
            "source_updated_at",
        ):
            _require_aware(getattr(self, field_name), field_name)

        if self.first_seen_at > self.retrieved_at:
            raise DataQualityError("first_seen_at cannot be after retrieved_at")
        if self.retrieved_at > self.ingested_at:
            raise DataQualityError("retrieved_at cannot be after ingested_at")
        for field_name in (
            "event_at",
            "published_at",
            "provider_available_at",
            "source_updated_at",
        ):
            value = getattr(self, field_name)
            if value is not None and value > self.retrieved_at:
                raise DataQualityError(f"{field_name} cannot be after retrieved_at")

    @property
    def source_id(self) -> SourceId:
        """Compatibility alias for callers that name the provider as source_id."""

        return self.provider

    @property
    def knowledge_at(self) -> datetime:
        """Earliest defensible instant at which this exact observation was knowable."""

        candidates = [self.first_seen_at]
        candidates.extend(
            value
            for value in (self.event_at, self.published_at, self.provider_available_at)
            if value is not None
        )
        return max(candidates)

    def is_known_at(self, cutoff: datetime) -> bool:
        _require_aware(cutoff, "cutoff")
        return self.point_in_time and self.knowledge_at <= cutoff

    def require_known_at(self, cutoff: datetime, *, require_point_in_time: bool = True) -> None:
        _require_aware(cutoff, "cutoff")
        if require_point_in_time and not self.point_in_time:
            raise DataQualityError("provider payload is not point-in-time safe")
        if self.knowledge_at > cutoff:
            raise DataQualityError(
                f"payload was not knowable at cutoff: {self.knowledge_at.isoformat()}"
            )
