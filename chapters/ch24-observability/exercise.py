from __future__ import annotations


def record_refund_trace(sink, task_id: str) -> None:
    """Record safe spans for a refund task.

    Required metadata: trace_id, step_index, status, duration_ms, policy_version,
    tool name when applicable, idempotency key for action proposals, and no raw
    prompts, hidden reasoning, secrets, or API keys.
    """
    raise NotImplementedError
