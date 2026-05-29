from __future__ import annotations


def store_public_evidence(cache, evidence, permission_context):
    """Store evidence only if it is safe to cache.

    Cache safety means source_id and policy_version are present, the cache key is
    scoped to the current permission context, and customer-private or
    tenant-private evidence is rejected instead of stored. The permission context
    must include tenant_id, actor_id, and delegated scopes.
    """
    raise NotImplementedError
