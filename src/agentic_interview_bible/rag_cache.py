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
    """Evidence cache keyed on ``(key, policy_version)``.

    Permission scoping is the caller's responsibility: this primitive does not
    enforce tenant or actor isolation. To make isolation a property of the
    cache rather than the wiring, include the tenant or permission prefix in
    the ``key`` (or wrap with a per-tenant instance) so one tenant cannot read
    another tenant's cached evidence.
    """

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], Evidence] = {}

    def put(self, evidence: Evidence) -> None:
        if evidence.contains_sensitive_data:
            raise CachePolicyError("sensitive evidence must not be cached")
        self._items[(evidence.key, evidence.policy_version)] = evidence

    def get(self, key: str, policy_version: str) -> Evidence | None:
        return self._items.get((key, policy_version))


class PerTenantEvidenceCache:
    """Evidence cache that scopes every entry by a PermissionContext, so
    isolation is a property of the cache rather than the caller. Prefer this
    over a bare ``EvidenceCache`` when serving more than one tenant or actor:
    two instances with different contexts cannot read each other's evidence.
    """

    def __init__(self, context: PermissionContext) -> None:
        self._context = context
        self._items: dict[tuple[str, str, str], Evidence] = {}

    def put(self, evidence: Evidence) -> None:
        if evidence.contains_sensitive_data:
            raise CachePolicyError("sensitive evidence must not be cached")
        self._items[
            (self._context.cache_prefix(), evidence.key, evidence.policy_version)
        ] = evidence

    def get(self, key: str, policy_version: str) -> Evidence | None:
        return self._items.get((self._context.cache_prefix(), key, policy_version))
