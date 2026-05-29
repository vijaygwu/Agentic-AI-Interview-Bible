from __future__ import annotations

from uuid import uuid4

from agentic_interview_bible import TraceEvent


def record_refund_trace(sink, task_id: str, trace_id: str | None = None) -> None:
    trace_id = trace_id or f"trace-{uuid4().hex}"
    sink.record(
        TraceEvent(
            task_id=task_id,
            span="policy.check",
            safe_summary="checked refund eligibility",
            metadata={
                "trace_id": trace_id,
                "step_index": 0,
                "status": "ok",
                "duration_ms": 12,
                "policy_version": "refunds-2026-05",
            },
        )
    )
    sink.record(
        TraceEvent(
            task_id=task_id,
            span="tool.refund_proposal",
            safe_summary="prepared refund proposal",
            metadata={
                "trace_id": trace_id,
                "step_index": 1,
                "status": "queued_for_approval",
                "duration_ms": 21,
                "policy_version": "refunds-2026-05",
                "tool_name": "refund_proposal",
                "idempotency_key": f"{task_id}:1:refund_proposal",
            },
        )
    )
    sink.record(
        TraceEvent(
            task_id=task_id,
            span="tool.refund_execution.failure",
            safe_summary="refund execution blocked by circuit breaker",
            metadata={
                "trace_id": trace_id,
                "step_index": 2,
                "status": "blocked",
                "duration_ms": 4,
                "policy_version": "refunds-2026-05",
                "tool_name": "refund_execution",
                "error_type": "CircuitOpenError",
                "retry_count": 2,
                "budget_remaining": 0,
                "circuit_state": "open",
                "idempotency_key": f"{task_id}:2:refund_execution",
            },
        )
    )
