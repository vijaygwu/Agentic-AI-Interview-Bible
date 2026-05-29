import pytest

from agentic_interview_bible.circuit_breaker import (
    BreakerState,
    CircuitBreaker,
    CircuitOpenError,
)
from agentic_interview_bible.cost_budget import BudgetExceededError, TaskBudget
from agentic_interview_bible.eval_harness import EvalCase, run_eval_cases
from agentic_interview_bible.observability import InMemoryTraceSink, TraceEvent
from agentic_interview_bible.rag_cache import (
    CachePolicyError,
    Evidence,
    EvidenceCache,
    PermissionContext,
    PerTenantEvidenceCache,
)
from agentic_interview_bible.release_gate import decide_release
from agentic_interview_bible.retry_budget import RetryBudget, RetryExhaustedError
from agentic_interview_bible.structured_outputs import (
    StructuredOutputError,
    validate_refund_decision,
)


def test_structured_output_validation_accepts_refund_decision() -> None:
    payload = {
        "decision": "approve",
        "amount_cents": 1_200,
        "policy_version": "refunds-2026-05",
        "requires_human_approval": False,
        "evidence_ids": ["order-1", "policy-9"],
    }

    assert validate_refund_decision(payload)["decision"] == "approve"


def test_structured_output_validation_rejects_bad_type() -> None:
    with pytest.raises(StructuredOutputError):
        validate_refund_decision(
            {
                "decision": "approve",
                "amount_cents": "1200",
                "policy_version": "refunds-2026-05",
                "requires_human_approval": False,
                "evidence_ids": ["order-1"],
            }
        )


def test_structured_output_validation_rejects_invalid_decision() -> None:
    with pytest.raises(StructuredOutputError):
        validate_refund_decision(
            {
                "decision": "maybe",
                "amount_cents": 1_200,
                "policy_version": "refunds-2026-05",
                "requires_human_approval": False,
                "evidence_ids": ["order-1"],
            }
        )


def test_structured_output_validation_rejects_negative_amount_empty_evidence_and_bool_amount() -> None:
    base = {
        "decision": "approve",
        "amount_cents": 1_200,
        "policy_version": "refunds-2026-05",
        "requires_human_approval": False,
        "evidence_ids": ["order-1"],
    }

    with pytest.raises(StructuredOutputError):
        validate_refund_decision({**base, "amount_cents": -1})
    with pytest.raises(StructuredOutputError):
        validate_refund_decision({**base, "amount_cents": True})
    with pytest.raises(StructuredOutputError):
        validate_refund_decision({**base, "evidence_ids": []})


def test_retry_budget_retries_timeout_then_succeeds() -> None:
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("try again")
        return "ok"

    observed: list[tuple[int, float]] = []

    budget = RetryBudget(
        max_attempts=2,
        backoff=lambda attempt: attempt * 0.5,
        observer=lambda attempt, exc, delay: observed.append((attempt, delay)),
        sleep=lambda _d: None,
    )

    assert budget.run(flaky) == "ok"
    assert observed == [(1, 0.5)]


def test_retry_budget_exhaustion_is_typed() -> None:
    with pytest.raises(RetryExhaustedError):
        RetryBudget(max_attempts=2).run(lambda: (_ for _ in ()).throw(TimeoutError()))


def test_circuit_breaker_opens_and_recovers() -> None:
    breaker = CircuitBreaker(failure_threshold=1, recovery_after=10.0)

    with pytest.raises(TimeoutError):
        breaker.call(lambda: (_ for _ in ()).throw(TimeoutError("down")), now=0.0)

    assert breaker.state == BreakerState.OPEN
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "blocked", now=5.0)

    assert breaker.call(lambda: "recovered", now=11.0) == "recovered"
    assert breaker.state == BreakerState.CLOSED


def test_circuit_breaker_does_not_count_default_runtime_error() -> None:
    breaker = CircuitBreaker(failure_threshold=1, recovery_after=10.0)

    with pytest.raises(RuntimeError):
        breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("bug")), now=0.0)

    assert breaker.state == BreakerState.CLOSED
    assert breaker.failure_count == 0


def test_circuit_breaker_does_not_count_non_dependency_exception() -> None:
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_after=10.0,
        failure_exceptions=(TimeoutError,),
    )

    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("bad input")), now=0.0)

    assert breaker.state == BreakerState.CLOSED
    assert breaker.failure_count == 0


def test_eval_harness_and_release_gate() -> None:
    report = run_eval_cases(
        responder=lambda prompt: f"{prompt} approved with policy",
        cases=[
            EvalCase(
                name="refund_policy",
                prompt="refund",
                expected_substring="policy",
            )
        ],
    )

    assert report.pass_rate == 1.0
    assert decide_release(report, minimum_pass_rate=1.0).allowed is True


def test_eval_harness_supports_predicate_checks_and_critical_failures() -> None:
    report = run_eval_cases(
        responder=lambda prompt: f"{prompt} without escalation",
        cases=[
            EvalCase(
                name="must_escalate",
                prompt="refund exception",
                check=lambda output: "human approval" in output,
                category="safety",
                critical=True,
            )
        ],
    )

    decision = decide_release(report, minimum_pass_rate=0.0)

    assert report.pass_rate == 0.0
    assert report.critical_failures[0].name == "must_escalate"
    assert decision.allowed is False
    assert "critical eval failed" in decision.reason


def test_eval_harness_redacts_output_and_catches_responder_exceptions() -> None:
    report = run_eval_cases(
        responder=lambda prompt: (_ for _ in ()).throw(RuntimeError("provider down"))
        if "fail" in prompt
        else "contains api_key=secret",
        cases=[
            EvalCase("redact", "ok", expected_substring="api_key"),
            EvalCase("exception", "fail", expected_substring="never"),
        ],
    )

    assert report.results[0].passed is True
    assert report.results[0].output == "contains [redacted]"
    assert report.results[1].passed is False
    assert report.results[1].output == "[error:RuntimeError]"


def test_evidence_cache_keeps_policy_versions_separate() -> None:
    cache = EvidenceCache()
    cache.put(Evidence("refund", "old policy", "doc-1", "v1"))
    cache.put(Evidence("refund", "new policy", "doc-2", "v2"))

    assert cache.get("refund", "v1").source_id == "doc-1"
    assert cache.get("refund", "v2").source_id == "doc-2"


def test_per_tenant_evidence_cache_isolates_tenants() -> None:
    ctx_a = PermissionContext(actor_id="u1", tenant_id="t1", scopes=("read",))
    ctx_b = PermissionContext(actor_id="u1", tenant_id="t2", scopes=("read",))
    cache_a = PerTenantEvidenceCache(ctx_a)
    cache_b = PerTenantEvidenceCache(ctx_b)

    cache_a.put(Evidence("refund", "tenant-1 evidence", "doc-1", "v1"))

    # Same key + policy, different tenant context: must not read across.
    assert cache_a.get("refund", "v1").source_id == "doc-1"
    assert cache_b.get("refund", "v1") is None


def test_evidence_cache_rejects_sensitive_data() -> None:
    with pytest.raises(CachePolicyError):
        EvidenceCache().put(
            Evidence(
                key="customer",
                text="private account note",
                source_id="crm-1",
                policy_version="v1",
                contains_sensitive_data=True,
            )
        )


def test_task_budget_raises_on_over_budget() -> None:
    budget = TaskBudget(max_model_calls=1, max_tokens=10)
    budget.record_model_call(tokens=6)

    with pytest.raises(BudgetExceededError):
        budget.record_model_call(tokens=6)


def test_task_budget_rejects_invalid_values_before_mutating() -> None:
    with pytest.raises(ValueError):
        TaskBudget(max_model_calls=0, max_tokens=10)
    with pytest.raises(ValueError):
        TaskBudget(max_model_calls=1, max_tokens=0)

    budget = TaskBudget(max_model_calls=1, max_tokens=10)
    with pytest.raises(ValueError):
        budget.record_model_call(tokens=-1)

    assert budget.model_calls == 0

    # Type guard: a bool (int subclass) or a float must be rejected so the
    # running token count cannot be silently corrupted.
    with pytest.raises(ValueError):
        budget.record_model_call(tokens=True)
    with pytest.raises(ValueError):
        budget.record_model_call(tokens=1.5)
    assert budget.model_calls == 0
    assert budget.tokens == 0


def test_trace_sink_rejects_unsafe_metadata() -> None:
    sink = InMemoryTraceSink()
    sink.record(
        TraceEvent(
            task_id="task-1",
            span="tool.lookup",
            safe_summary="looked up order status",
            metadata={"tool_name": "order_lookup"},
        )
    )

    assert sink.spans_for_task("task-1") == ["tool.lookup"]
    with pytest.raises(ValueError):
        sink.record(
            TraceEvent(
                task_id="task-1",
                span="model.call",
                safe_summary="unsafe",
                metadata={"raw_prompt": "private content"},
            )
        )
