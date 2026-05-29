from __future__ import annotations

from dataclasses import dataclass


class CachePolicyError(ValueError):
    """Raised when a caller tries to cache unsafe evidence."""


@dataclass(frozen=True)
class Evidence:
    key: str
    text: str
    source_id: str
    policy_version: str
    contains_sensitive_data: bool = False


@dataclass(frozen=True)
class PermissionContext:
    actor_id: str
    tenant_id: str
    scopes: tuple[str, ...]

    def cache_prefix(self) -> str:
        if not self.actor_id or not self.tenant_id or not self.scopes:
            raise CachePolicyError("permission context must include actor, tenant, and scopes")
        scope_key = ",".join(sorted(self.scopes))
        return f"tenant={self.tenant_id}|actor={self.actor_id}|scopes={scope_key}"


class EvidenceCache:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], Evidence] = {}

    def put(self, evidence: Evidence) -> None:
        if evidence.contains_sensitive_data:
            raise CachePolicyError("sensitive evidence must not be cached")
        self._items[(evidence.key, evidence.policy_version)] = evidence

    def get(self, key: str, policy_version: str) -> Evidence | None:
        return self._items.get((key, policy_version))
