from __future__ import annotations

import random
import time

from agentic_interview_bible import RetryBudget


def call_with_retry_budget(
    operation,
    max_attempts: int = 2,
    retryable: tuple[type[BaseException], ...] = (TimeoutError,),
    observer=None,
    sleep=time.sleep,
    rng=random.random,
):
    def capped_backoff(attempt: int) -> float:
        ceiling = min(0.25 * (2 ** (attempt - 1)), 2.0)
        # Full jitter: spread the retry across [0, ceiling] so that independent
        # clients backing off the same dependency do not resynchronize into a
        # retry storm. Inject `rng` in tests to make the delay deterministic.
        return rng() * ceiling

    return RetryBudget(
        max_attempts=max_attempts,
        retryable=retryable,
        backoff=capped_backoff,
        observer=observer,
        sleep=sleep,
    ).run(operation)


def call_with_retry_budget_and_circuit(
    operation,
    breaker,
    now: float,
    max_attempts: int = 2,
    retryable: tuple[type[BaseException], ...] = (TimeoutError,),
    observer=None,
    sleep=time.sleep,
    rng=random.random,
):
    def protected_operation():
        # A single fixed `now` is intentional here: retries within one
        # protected call are sub-second, so they share a clock reading.
        # Modeling retries that span the breaker's recovery window would
        # require advancing `now` between attempts.
        return breaker.call(operation, now=now)

    return call_with_retry_budget(
        protected_operation,
        max_attempts=max_attempts,
        retryable=retryable,
        observer=observer,
        sleep=sleep,
        rng=rng,
    )
