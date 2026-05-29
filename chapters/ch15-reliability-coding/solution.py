from __future__ import annotations

from agentic_interview_bible import RetryBudget


def call_with_retry_budget(
    operation,
    max_attempts: int = 2,
    retryable: tuple[type[BaseException], ...] = (TimeoutError,),
    observer=None,
):
    def capped_backoff(attempt: int) -> float:
        return min(0.25 * (2 ** (attempt - 1)), 2.0)

    return RetryBudget(
        max_attempts=max_attempts,
        retryable=retryable,
        backoff=capped_backoff,
        observer=observer,
    ).run(operation)


def call_with_retry_budget_and_circuit(
    operation,
    breaker,
    now: float,
    max_attempts: int = 2,
    retryable: tuple[type[BaseException], ...] = (TimeoutError,),
    observer=None,
):
    def protected_operation():
        return breaker.call(operation, now=now)

    return call_with_retry_budget(
        protected_operation,
        max_attempts=max_attempts,
        retryable=retryable,
        observer=observer,
    )
