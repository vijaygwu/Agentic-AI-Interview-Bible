from __future__ import annotations

import time

from agentic_interview_bible.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    State,
)
from agentic_interview_bible.retry_budget import RetryBudget, RetryExhausted


def call_with_retry_budget(
    operation,
    max_attempts: int = 2,
    retryable: tuple = (TimeoutError,),
    observer=None,
    sleep=time.sleep,
    jitter=None,
):
    """Run an idempotent operation with an explicit retry policy.

    Delegates to RetryBudget.call so tests exercise the canonical book API.
    The jitter callable is injected so tests can pin delay sequences without
    seeding global random state.
    """
    import random

    rb = RetryBudget(
        max_attempts=max_attempts,
        retryable=tuple(retryable),
        sleep=sleep,
        jitter=jitter if jitter is not None else random.uniform,
        observer=observer,
    )
    return rb.call(operation)


def call_with_retry_budget_and_circuit(
    operation,
    breaker: CircuitBreaker,
    max_attempts: int = 2,
    retryable: tuple = (TimeoutError,),
    observer=None,
    sleep=time.sleep,
    jitter=None,
):
    """Run an idempotent dependency call behind retries and a circuit breaker.

    The circuit breaker uses its injected clock so half-open recovery is
    deterministic in tests.
    """
    def protected_operation():
        return breaker.call(operation)

    return call_with_retry_budget(
        protected_operation,
        max_attempts=max_attempts,
        retryable=retryable,
        observer=observer,
        sleep=sleep,
        jitter=jitter,
    )
