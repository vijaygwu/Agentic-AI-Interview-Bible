from __future__ import annotations

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
        raise RetryExhaustedError("retry budget exhausted") from last_error
