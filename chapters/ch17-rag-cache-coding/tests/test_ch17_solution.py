import importlib.util
from pathlib import Path

import pytest

from agentic_interview_bible import (
    CachePolicyError,
    Evidence,
    EvidenceCache,
    PermissionContext,
)


def load_solution():
    path = Path(__file__).parents[1] / "solution.py"
    spec = importlib.util.spec_from_file_location("ch17_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_store_public_evidence() -> None:
    solution = load_solution()
    cache = EvidenceCache()
    evidence = Evidence("refund", "policy text", "policy-1", "v1")
    permission = PermissionContext(
        actor_id="user-1",
        tenant_id="tenant-1",
        scopes=("policy:read",),
    )

    solution.store_public_evidence(cache, evidence, permission)

    scoped_key = solution.scoped_cache_key(permission, evidence.key)
    assert cache.get(scoped_key, "v1").source_id == "policy-1"
    assert cache.get("refund", "v1") is None


def test_store_public_evidence_encodes_delimiter_bearing_key_parts() -> None:
    solution = load_solution()
    cache = EvidenceCache()
    evidence = Evidence("refund|key=x", "policy text", "policy-1", "v1")
    permission = PermissionContext(
        actor_id="user|actor=2",
        tenant_id="tenant|actor=user",
        scopes=("policy:read", "scope|key=refund"),
    )

    solution.store_public_evidence(cache, evidence, permission)

    unsafe_key = "tenant=tenant|actor=user|actor=user|actor=2|scopes=policy:read,scope|key=refund|key=refund|key=x"
    safe_key = solution.scoped_cache_key(permission, evidence.key)
    assert safe_key != unsafe_key
    assert "%7C" in safe_key
    assert cache.get(safe_key, "v1").source_id == "policy-1"
    assert cache.get(unsafe_key, "v1") is None


def test_store_public_evidence_rejects_sensitive_data() -> None:
    solution = load_solution()
    cache = EvidenceCache()
    evidence = Evidence(
        "customer-note",
        "private customer note",
        "crm-1",
        "v1",
        contains_sensitive_data=True,
    )
    permission = PermissionContext(
        actor_id="user-1",
        tenant_id="tenant-1",
        scopes=("crm:read",),
    )

    with pytest.raises(CachePolicyError):
        solution.store_public_evidence(cache, evidence, permission)


def test_store_public_evidence_requires_source_policy_and_permission_context() -> None:
    solution = load_solution()
    cache = EvidenceCache()
    permission = PermissionContext(
        actor_id="user-1",
        tenant_id="tenant-1",
        scopes=("policy:read",),
    )

    with pytest.raises(CachePolicyError):
        solution.store_public_evidence(
            cache,
            Evidence("refund", "policy text", "", "v1"),
            permission,
        )
    with pytest.raises(CachePolicyError):
        solution.store_public_evidence(
            cache,
            Evidence("refund", "policy text", "policy-1", ""),
            permission,
        )
    with pytest.raises(CachePolicyError):
        solution.store_public_evidence(
            cache,
            Evidence("refund", "policy text", "policy-1", "v1"),
            PermissionContext(actor_id="user-1", tenant_id="", scopes=("policy:read",)),
        )
