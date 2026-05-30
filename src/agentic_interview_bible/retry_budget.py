from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class RetryableError(Exception):
    """Base class for errors that may be retried."""


class RetryExhausted(Exception):
    """Raised when a retry budget is exhausted after all attempts."""


@dataclass
class RetryBudget:
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 10.0
    retryable: tuple = (RetryableError,)
    sleep: Callable[[float], None] = field(default=time.sleep)
    # Inject the jitter function so tests can pin it. The default below is
    # decorrelated jitter; Full Jitter would be jitter(0, ceiling), which
    # does less total client work. Tests pass a deterministic jitter that
    # returns the midpoint and assert exact delay sequences.
    jitter: Callable[[float, float], float] = field(default=random.uniform)
    observer: Callable[[int, Exception, float], None] | None = None

    def call(self, fn: Callable[[], Any]) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                return fn()
            except self.retryable as exc:
                last_exc = exc
                if attempt + 1 >= self.max_attempts:
                    if self.observer:
                        self.observer(attempt, exc, 0.0)
                    break
                ceiling = min(
                    self.base_delay_s * (2 ** attempt),
                    self.max_delay_s,
                )
                delay = min(
                    self.jitter(self.base_delay_s, ceiling * 3),
                    self.max_delay_s,
                )
                if self.observer:
                    self.observer(attempt, exc, delay)
                self.sleep(delay)
        raise RetryExhausted(
            f"failed after {self.max_attempts} attempts: {last_exc}"
        ) from last_exc


# Backward-compatibility alias so __init__.py (which must not be edited) can
# still export RetryExhaustedError under its original name.
RetryExhaustedError = RetryExhausted
