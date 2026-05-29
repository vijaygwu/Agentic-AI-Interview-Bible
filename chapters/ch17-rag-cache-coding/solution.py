from __future__ import annotations

from urllib.parse import quote

from agentic_interview_bible import CachePolicyError, Evidence


def encode_key_part(value: str) -> str:
    return quote(value, safe="")


def scoped_cache_key(permission_context, evidence_key: str) -> str:
    if not permission_context.actor_id or not permission_context.tenant_id or not permission_context.scopes:
        raise CachePolicyError("permission context must include actor, tenant, and scopes")
    scope_key = ",".join(encode_key_part(scope) for scope in sorted(permission_context.scopes))
    return (
        f"tenant={encode_key_part(permission_context.tenant_id)}"
        f"|actor={encode_key_part(permission_context.actor_id)}"
        f"|scopes={scope_key}"
        f"|key={encode_key_part(evidence_key)}"
    )


def store_public_evidence(cache, evidence, permission_context) -> None:
    if not evidence.source_id:
        raise CachePolicyError("source_id is required")
    if not evidence.policy_version:
        raise CachePolicyError("policy_version is required")
    if evidence.contains_sensitive_data:
        raise CachePolicyError("sensitive evidence must not be cached")

    scoped_evidence = Evidence(
        key=scoped_cache_key(permission_context, evidence.key),
        text=evidence.text,
        source_id=evidence.source_id,
        policy_version=evidence.policy_version,
        contains_sensitive_data=False,
    )
    cache.put(scoped_evidence)
