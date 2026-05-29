from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar


T = TypeVar("T")
BackoffStrategy = Callable[[int], float]
RetryObserver = Callable[[int, BaseException, float], None]


class RetryExhaustedError(RuntimeError):
    """Raised when a retry budget is exhausted."""


@dataclass(frozen=True)
class RetryBudget:
    max_attempts: int
    retryable: tuple[type[BaseException], ...] = (TimeoutError,)
    backoff: BackoffStrategy | None = None
    observer: RetryObserver | None = None
    # Injected so the production path actually spaces retries while tests stay
    # fast and deterministic by passing a no-op (e.g. ``sleep=lambda _d: None``).
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")

    def run(self, operation: Callable[[], T]) -> T:
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation()
            except self.retryable as exc:
                last_error = exc
                delay = self.backoff(attempt) if self.backoff is not None else 0.0
                if self.observer is not None:
                    self.observer(attempt, exc, delay)
                # Actually apply the backoff: a computed-but-unslept delay is the
                # canonical bug (retries fire back-to-back). Skip the wait after
                # the final attempt, since we are about to raise.
                if delay > 0 and attempt < self.max_attempts:
                    self.sleep(delay)
        raise RetryExhaustedError("retry budget exhausted") from last_error
