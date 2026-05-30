"""Chapter 17 solution — Embedding Store with Metadata Filters.

Exact implementation of the "strong attempt" shown in the book:
    problem-embedding-store.tex
"""
from __future__ import annotations

import math
from typing import Callable

from agentic_interview_bible.rag_cache import Metadata, StoredDocument, SearchContext


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
    # Pre-filter: never compute similarity for documents the actor cannot see
    candidates = [d for d in store if _passes_filter(d, ctx, extra_filter)]
    if not candidates:
        return []
    scored = [
        (_cosine(query_embedding, d.embedding), d) for d in candidates
    ]
    scored.sort(key=lambda t: -t[0])
    return [d for _, d in scored[:top_k]]
