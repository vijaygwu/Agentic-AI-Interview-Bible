from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar


T = TypeVar("T")


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a protected dependency is not allowed to run."""


@dataclass
class CircuitBreaker:
    """Single-threaded circuit breaker: deterministic, with time passed in
    explicitly via ``now``. Half-open admits the next call by contract, not
    under a lock, so it is not safe for concurrent callers. For the concurrent
    version that leases a single half-open trial under a lock, see the strong
    attempt in ``problem-circuit-breaker.tex``.
    """

    failure_threshold: int
    recovery_after: float
    failure_exceptions: tuple[type[BaseException], ...] = (
        TimeoutError,
        ConnectionError,
    )
    state: BreakerState = BreakerState.CLOSED
    failure_count: int = 0
    opened_at: float | None = None

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        if self.recovery_after < 0:
            raise ValueError("recovery_after must be non-negative")

    def call(self, operation: Callable[[], T], now: float) -> T:
        if self.state == BreakerState.OPEN:
            assert self.opened_at is not None
            if now - self.opened_at < self.recovery_after:
                raise CircuitOpenError("circuit is open")
            self.state = BreakerState.HALF_OPEN

        try:
            result = operation()
        except Exception as exc:
            if isinstance(exc, self.failure_exceptions):
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.state = BreakerState.OPEN
                    self.opened_at = now
            raise

        self.failure_count = 0
        self.opened_at = None
        self.state = BreakerState.CLOSED
        return result
