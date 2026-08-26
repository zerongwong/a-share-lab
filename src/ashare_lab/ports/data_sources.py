"""Ports for provider adapters, provenance registries and rights checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from ashare_lab.domain.data_sources import (
    DataAction,
    ProviderEnvelope,
    SourceDefinition,
    SourceId,
)


@runtime_checkable
class DataSourceProvider[T_co](Protocol):
    """A provider adapter that always returns provenance with its payload."""

    @property
    def source_id(self) -> SourceId: ...

    def fetch(self, dataset: str, request: Mapping[str, object]) -> ProviderEnvelope[T_co]: ...


@runtime_checkable
class SourceRegistryPort(Protocol):
    def get(self, source_id: SourceId | str) -> SourceDefinition: ...

    def all(self) -> tuple[SourceDefinition, ...]: ...


@runtime_checkable
class RightsPolicyPort(Protocol):
    def require(self, source_id: SourceId | str, action: DataAction | str) -> SourceDefinition: ...

    def require_envelope(
        self,
        envelope: ProviderEnvelope[object],
        action: DataAction | str,
    ) -> SourceDefinition: ...
