"""Embedding store with pre-filter authorization.

Public API mirrors the book listings exactly:

    Metadata, StoredDocument, SearchContext  -- data classes
    _passes_filter(doc, ctx, extra_filter)   -- predicate
    search(store, query_embedding, ctx, ...)  -- ranked retrieval

Backward-compat shim (used by existing __init__.py exports):
    CachePolicyError, Evidence, EvidenceCache,
    PermissionContext, PerTenantEvidenceCache
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable


# ---------------------------------------------------------------------------
# Book API — embedding store (problem-embedding-store.tex)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Metadata:
    tenant_id: str
    required_scopes: frozenset[str]
    class_: str
    last_modified: datetime


@dataclass(frozen=True)
class StoredDocument:
    id: str
    embedding: list[float]
    metadata: Metadata


@dataclass(frozen=True)
class SearchContext:
    tenant_id: str
    actor_scopes: frozenset[str]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _passes_filter(
    doc: StoredDocument,
    ctx: SearchContext,
    extra_filter: Callable | None,
) -> bool:
    if doc.metadata.tenant_id != ctx.tenant_id:
        return False
    if doc.metadata.required_scopes - ctx.actor_scopes:
        return False
    if extra_filter and not extra_filter(doc.metadata):
        return False
    return True


def search(
    store: list[StoredDocument],
    query_embedding: list[float],
    ctx: SearchContext,
    top_k: int = 5,
    extra_filter: Callable | None = None,
) -> list[StoredDocument]:
    """Pre-filter, then rank by cosine similarity.

    Documents that fail the actor scope or tenant check are never scored.
    """
    # Pre-filter: never compute similarity for documents the actor cannot see
    candidates = [d for d in store if _passes_filter(d, ctx, extra_filter)]
    if not candidates:
        return []
    scored = [
        (_cosine(query_embedding, d.embedding), d) for d in candidates
    ]
    scored.sort(key=lambda t: -t[0])
    return [d for _, d in scored[:top_k]]


# ---------------------------------------------------------------------------
# Backward-compat shim — kept so __init__.py and ch17 exercise/tests compile
# ---------------------------------------------------------------------------

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
            raise CachePolicyError(
                "permission context must include actor, tenant, and scopes"
            )
        scope_key = ",".join(sorted(self.scopes))
        return (
            f"tenant={self.tenant_id}"
            f"|actor={self.actor_id}"
            f"|scopes={scope_key}"
        )


class EvidenceCache:
    """Evidence cache keyed on ``(key, policy_version)``."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], Evidence] = {}

    def put(self, evidence: Evidence) -> None:
        if evidence.contains_sensitive_data:
            raise CachePolicyError("sensitive evidence must not be cached")
        self._items[(evidence.key, evidence.policy_version)] = evidence

    def get(self, key: str, policy_version: str) -> Evidence | None:
        return self._items.get((key, policy_version))


class PerTenantEvidenceCache:
    """Evidence cache scoped by a PermissionContext."""

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
        return self._items.get(
            (self._context.cache_prefix(), key, policy_version)
        )
