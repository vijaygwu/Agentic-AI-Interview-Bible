from __future__ import annotations

from typing import Callable


def call_with_retry_budget(
    operation,
    max_attempts: int = 2,
    retryable: tuple[type[BaseException], ...] = (TimeoutError,),
    observer: Callable[[int, BaseException, float], None] | None = None,
):
    """Run an idempotent operation with an explicit retry policy.

    Requirements:
    - retry only configured retryable exceptions
    - stop after max_attempts
    - report each retry to observer as attempt number, exception, and delay
    - compute capped exponential backoff with jitter; make it deterministic
      under an injected rng (not by removing jitter), and do not sleep in tests
    - propagate non-retryable exceptions immediately
    - raise a typed exhaustion error when retries are spent
    """
    raise NotImplementedError


def call_with_retry_budget_and_circuit(
    operation,
    breaker,
    now: float,
    max_attempts: int = 2,
    retryable: tuple[type[BaseException], ...] = (TimeoutError,),
    observer: Callable[[int, BaseException, float], None] | None = None,
):
    """Run an idempotent dependency call behind retries and a circuit breaker.

    Requirements:
    - use the provided CircuitBreaker state instead of creating hidden state
    - support configurable failure_threshold and recovery_after on the breaker
    - raise CircuitOpenError immediately while the circuit is open
    - use the injected now value so half-open recovery is deterministic
    - close the breaker after a successful half-open probe
    - avoid counting non-retryable/non-dependency exceptions as breaker failures
    """
    raise NotImplementedError
