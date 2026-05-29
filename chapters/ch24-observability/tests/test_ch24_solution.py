import importlib.util
from pathlib import Path

from agentic_interview_bible import InMemoryTraceSink


def load_solution():
    path = Path(__file__).parents[1] / "solution.py"
    spec = importlib.util.spec_from_file_location("ch24_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_record_refund_trace() -> None:
    solution = load_solution()
    sink = InMemoryTraceSink()

    solution.record_refund_trace(sink, "refund-1")

    assert sink.spans_for_task("refund-1") == [
        "policy.check",
        "tool.refund_proposal",
        "tool.refund_execution.failure",
    ]
    first, second, third = sink.events
    assert first.safe_summary == "checked refund eligibility"
    trace_id = first.metadata["trace_id"]
    assert trace_id.startswith("trace-")
    assert trace_id != "trace-refund-1"
    assert second.metadata["trace_id"] == trace_id
    assert third.metadata["trace_id"] == trace_id
    assert first.metadata["status"] == "ok"
    assert first.metadata["duration_ms"] == 12
    assert second.metadata["policy_version"] == "refunds-2026-05"
    assert second.metadata["idempotency_key"] == "refund-1:1:refund_proposal"
    assert third.metadata["error_type"] == "CircuitOpenError"
    assert third.metadata["retry_count"] == 2
    assert third.metadata["budget_remaining"] == 0
    assert third.metadata["circuit_state"] == "open"
    assert "raw_prompt" not in first.metadata
    assert "chain_of_thought" not in second.metadata

    second_sink = InMemoryTraceSink()
    solution.record_refund_trace(second_sink, "refund-1")
    assert second_sink.events[0].metadata["trace_id"] != trace_id
