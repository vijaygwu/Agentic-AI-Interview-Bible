"""Tests for chapter 17 — Embedding Store with Metadata Filters."""
from __future__ import annotations

import importlib.util
import math
from datetime import datetime
from pathlib import Path

import pytest

from agentic_interview_bible.rag_cache import Metadata, StoredDocument, SearchContext


def load_solution():
    path = Path(__file__).parents[1] / "solution.py"
    spec = importlib.util.spec_from_file_location("ch17_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_doc(
    doc_id: str,
    embedding: list[float],
    tenant_id: str = "tenant-a",
    required_scopes: frozenset[str] = frozenset(),
    class_: str = "kb_article",
) -> StoredDocument:
    return StoredDocument(
        id=doc_id,
        embedding=embedding,
        metadata=Metadata(
            tenant_id=tenant_id,
            required_scopes=required_scopes,
            class_=class_,
            last_modified=datetime(2025, 1, 1),
        ),
    )


def test_search_returns_top_k_by_similarity() -> None:
    sol = load_solution()
    # Three docs in tenant-a; query is identical to doc-1.
    doc1 = _make_doc("doc-1", [1.0, 0.0])
    doc2 = _make_doc("doc-2", [0.0, 1.0])
    doc3 = _make_doc("doc-3", [0.707, 0.707])
    store = [doc1, doc2, doc3]
    ctx = SearchContext(tenant_id="tenant-a", actor_scopes=frozenset())

    results = sol.search(store, [1.0, 0.0], ctx, top_k=2)

    assert results[0].id == "doc-1"
    assert len(results) == 2


def test_cross_tenant_isolation() -> None:
    """Tenant A's query must never return tenant B's documents."""
    sol = load_solution()
    # doc-b is semantically identical to the query but belongs to tenant-b.
    doc_a = _make_doc("doc-a", [1.0, 0.0], tenant_id="tenant-a")
    doc_b = _make_doc("doc-b", [1.0, 0.0], tenant_id="tenant-b")
    store = [doc_a, doc_b]
    ctx = SearchContext(tenant_id="tenant-a", actor_scopes=frozenset())

    results = sol.search(store, [1.0, 0.0], ctx, top_k=5)

    ids = [d.id for d in results]
    assert "doc-b" not in ids
    assert "doc-a" in ids


def test_scope_check_excludes_insufficient_actor() -> None:
    sol = load_solution()
    doc = _make_doc(
        "doc-restricted",
        [1.0, 0.0],
        required_scopes=frozenset({"admin"}),
    )
    store = [doc]
    # Actor only has "read", not "admin"
    ctx = SearchContext(
        tenant_id="tenant-a",
        actor_scopes=frozenset({"read"}),
    )

    results = sol.search(store, [1.0, 0.0], ctx)

    assert results == []


def test_scope_check_admits_sufficient_actor() -> None:
    sol = load_solution()
    doc = _make_doc(
        "doc-restricted",
        [1.0, 0.0],
        required_scopes=frozenset({"admin"}),
    )
    store = [doc]
    ctx = SearchContext(
        tenant_id="tenant-a",
        actor_scopes=frozenset({"read", "admin"}),
    )

    results = sol.search(store, [1.0, 0.0], ctx)

    assert len(results) == 1
    assert results[0].id == "doc-restricted"


def test_empty_store_returns_empty_list() -> None:
    sol = load_solution()
    ctx = SearchContext(tenant_id="tenant-a", actor_scopes=frozenset())
    results = sol.search([], [1.0, 0.0], ctx)
    assert results == []


def test_extra_filter_applied_before_similarity() -> None:
    sol = load_solution()
    doc_policy = _make_doc("doc-policy", [1.0, 0.0], class_="policy")
    doc_kb = _make_doc("doc-kb", [1.0, 0.0], class_="kb_article")
    store = [doc_policy, doc_kb]
    ctx = SearchContext(tenant_id="tenant-a", actor_scopes=frozenset())

    # Only return policy-class documents
    results = sol.search(
        store,
        [1.0, 0.0],
        ctx,
        extra_filter=lambda m: m.class_ == "policy",
    )

    ids = [d.id for d in results]
    assert "doc-policy" in ids
    assert "doc-kb" not in ids


def test_passes_filter_rejects_wrong_tenant() -> None:
    sol = load_solution()
    doc = _make_doc("d", [1.0], tenant_id="tenant-b")
    ctx = SearchContext(tenant_id="tenant-a", actor_scopes=frozenset())
    assert sol._passes_filter(doc, ctx, None) is False


def test_passes_filter_accepts_matching_tenant_no_scopes() -> None:
    sol = load_solution()
    doc = _make_doc("d", [1.0], tenant_id="tenant-a")
    ctx = SearchContext(tenant_id="tenant-a", actor_scopes=frozenset())
    assert sol._passes_filter(doc, ctx, None) is True
