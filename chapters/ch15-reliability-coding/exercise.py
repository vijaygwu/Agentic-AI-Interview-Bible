from __future__ import annotations

from typing import Any, Callable

from agentic_interview_bible.circuit_breaker import CircuitBreaker
from agentic_interview_bible.retry_budget import RetryBudget, RetryExhausted


def call_with_retry_budget(
    operation: Callable[[], Any],
    max_attempts: int = 2,
    retryable: tuple = (TimeoutError,),
    observer: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], None] | None = None,
    jitter: Callable[[float, float], float] | None = None,
) -> Any:
    """Run an idempotent operation with an explicit retry policy.

    Requirements:
    - retry only configured retryable exceptions
    - stop after max_attempts
    - report each retry to observer as attempt number, exception, and delay
    - compute capped exponential backoff with jitter; make it deterministic
      under an injected jitter callable (not by removing jitter), and do not
      sleep in tests
    - propagate non-retryable exceptions immediately
    - raise RetryExhausted when retries are spent

    Hint: construct a RetryBudget and call rb.call(operation).
    """
    raise NotImplementedError


def call_with_retry_budget_and_circuit(
    operation: Callable[[], Any],
    breaker: CircuitBreaker,
    max_attempts: int = 2,
    retryable: tuple = (TimeoutError,),
    observer: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], None] | None = None,
    jitter: Callable[[float, float], float] | None = None,
) -> Any:
    """Run an idempotent dependency call behind retries and a circuit breaker.

    Requirements:
    - use the provided CircuitBreaker state instead of creating hidden state
    - support configurable failure_threshold and recovery_after_s on the breaker
    - raise CircuitOpenError immediately while the circuit is open
    - use the injected clock on the breaker so half-open recovery is deterministic
    - close the breaker after a successful half-open probe
    - avoid counting non-retryable/non-dependency exceptions as breaker failures

    Hint: wrap operation in a closure that calls breaker.call(operation), then
    pass that closure to call_with_retry_budget.
    """
    raise NotImplementedError
