from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable

from .retry_budget import RetryableError


class State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when the circuit breaker rejects a call because the circuit is open."""


class CircuitHalfOpenBusyError(Exception):
    """Raised when a trial call is already in flight during half-open state."""


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_after_s: float = 30.0
    dependency_errors: tuple = (RetryableError,)
    clock: Callable[[], float] = field(default=time.monotonic)
    state: State = State.CLOSED
    _failures: int = 0
    _opened_at: float | None = None
    _half_open_lease: bool = False  # true while a trial call is in flight
    _lock: Lock = field(default_factory=Lock)
    on_state_change: Callable[[State, State], None] | None = None

    def call(self, fn: Callable[[], Any]) -> Any:
        # The lease handshake guarantees half-open's one-trial semantics
        # under concurrent callers. Without it, every caller that
        # observes HALF_OPEN would race to execute a trial call.
        with self._lock:
            if self.state == State.OPEN:
                if self.clock() - self._opened_at >= self.recovery_after_s:
                    self._transition(State.HALF_OPEN)
                else:
                    raise CircuitOpenError("circuit open")
            if self.state == State.HALF_OPEN:
                if self._half_open_lease:
                    raise CircuitHalfOpenBusyError("trial in flight")
                self._half_open_lease = True
                is_trial = True
            else:
                is_trial = False

        try:
            result = fn()
        except self.dependency_errors:
            with self._lock:
                if is_trial:
                    self._transition(State.OPEN)
                    self._opened_at = self.clock()
                    self._half_open_lease = False
                else:
                    self._failures += 1
                    if self._failures >= self.failure_threshold:
                        self._transition(State.OPEN)
                        self._opened_at = self.clock()
            raise

        with self._lock:
            if is_trial:
                self._transition(State.CLOSED)
                self._half_open_lease = False
            self._failures = 0
        return result

    def _transition(self, new: State) -> None:
        if self.on_state_change:
            self.on_state_change(self.state, new)
        self.state = new


# Backward-compatibility alias so __init__.py (which must not be edited) can
# still export BreakerState under its original name.
BreakerState = State
