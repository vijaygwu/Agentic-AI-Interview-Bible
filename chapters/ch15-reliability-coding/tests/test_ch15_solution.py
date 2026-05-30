from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from agentic_interview_bible.circuit_breaker import (
    CircuitBreaker,
    CircuitHalfOpenBusyError,
    CircuitOpenError,
    State,
)
from agentic_interview_bible.retry_budget import RetryBudget, RetryExhausted, RetryableError


def load_solution():
    path = Path(__file__).parents[1] / "solution.py"
    spec = importlib.util.spec_from_file_location("ch15_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# RetryBudget direct tests (book API: rb.call(fn))
# ---------------------------------------------------------------------------

def test_retry_budget_call_succeeds_on_second_attempt() -> None:
    """RetryBudget.call retries a retryable error and returns on success."""
    calls = {"n": 0}
    observed: list[tuple[int, float]] = []

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RetryableError("transient")
        return "ok"

    rb = RetryBudget(
        max_attempts=3,
        base_delay_s=1.0,
        max_delay_s=10.0,
        retryable=(RetryableError,),
        sleep=lambda _d: None,
        jitter=lambda lo, hi: (lo + hi) / 2,
        observer=lambda attempt, exc, delay: observed.append((attempt, delay)),
    )
    assert rb.call(flaky) == "ok"
    assert calls["n"] == 2
    # attempt 0 succeeded on retry: observer called with attempt=0 and a delay
    assert len(observed) == 1
    assert observed[0][0] == 0


def test_retry_budget_call_exhausted_raises_retry_exhausted() -> None:
    """RetryBudget.call raises RetryExhausted after max_attempts failures."""
    def always_fails():
        raise RetryableError("always down")

    rb = RetryBudget(
        max_attempts=3,
        sleep=lambda _d: None,
        jitter=lambda lo, hi: lo,
    )
    with pytest.raises(RetryExhausted):
        rb.call(always_fails)


def test_retry_budget_call_does_not_retry_non_retryable() -> None:
    """RetryBudget.call propagates non-retryable exceptions immediately."""
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise ValueError("validation error")

    rb = RetryBudget(
        max_attempts=3,
        sleep=lambda _d: None,
        jitter=lambda lo, hi: lo,
    )
    with pytest.raises(ValueError):
        rb.call(bad_request)
    assert calls["n"] == 1


def test_retry_budget_call_delay_sequence_is_deterministic() -> None:
    """Observer receives exact delays produced by the injected jitter."""
    delays: list[float] = []

    def always_fails():
        raise RetryableError("down")

    # jitter returns midpoint of (base_delay_s, ceiling * 3)
    # attempt 0: ceiling = min(0.5 * 2^0, 10) = 0.5; midpoint(0.5, 1.5) = 1.0
    # attempt 1: ceiling = min(0.5 * 2^1, 10) = 1.0; midpoint(0.5, 3.0) = 1.75
    # attempt 2: final attempt — observer called with delay=0.0
    rb = RetryBudget(
        max_attempts=3,
        base_delay_s=0.5,
        max_delay_s=10.0,
        sleep=lambda _d: None,
        jitter=lambda lo, hi: (lo + hi) / 2,
        observer=lambda attempt, exc, delay: delays.append(delay),
    )
    with pytest.raises(RetryExhausted):
        rb.call(always_fails)

    assert len(delays) == 3
    assert delays[0] == pytest.approx(1.0)
    assert delays[1] == pytest.approx(1.75)
    assert delays[2] == 0.0  # final attempt: no sleep


# ---------------------------------------------------------------------------
# CircuitBreaker direct tests (book API: cb.call(fn) with injected clock)
# ---------------------------------------------------------------------------

def test_circuit_breaker_opens_after_threshold_failures() -> None:
    """CircuitBreaker opens after failure_threshold dependency errors."""
    now_val = [0.0]
    cb = CircuitBreaker(
        failure_threshold=2,
        recovery_after_s=10.0,
        dependency_errors=(RetryableError,),
        clock=lambda: now_val[0],
    )

    for _ in range(2):
        with pytest.raises(RetryableError):
            cb.call(lambda: (_ for _ in ()).throw(RetryableError("dep down")))

    assert cb.state == State.OPEN
    assert cb._opened_at == pytest.approx(0.0)


def test_circuit_breaker_rejects_calls_while_open() -> None:
    """CircuitBreaker raises CircuitOpenError while state is OPEN."""
    now_val = [0.0]
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_after_s=10.0,
        dependency_errors=(RetryableError,),
        clock=lambda: now_val[0],
    )
    with pytest.raises(RetryableError):
        cb.call(lambda: (_ for _ in ()).throw(RetryableError("dep down")))

    assert cb.state == State.OPEN

    with pytest.raises(CircuitOpenError):
        cb.call(lambda: "should not run")


def test_circuit_breaker_half_open_success_closes() -> None:
    """A successful half-open trial closes the circuit."""
    now_val = [0.0]
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_after_s=10.0,
        dependency_errors=(RetryableError,),
        clock=lambda: now_val[0],
    )
    with pytest.raises(RetryableError):
        cb.call(lambda: (_ for _ in ()).throw(RetryableError("dep down")))

    assert cb.state == State.OPEN

    # Advance clock past recovery window
    now_val[0] = 11.0
    result = cb.call(lambda: "recovered")
    assert result == "recovered"
    assert cb.state == State.CLOSED
    assert cb._failures == 0


def test_circuit_breaker_half_open_failure_reopens() -> None:
    """A failing half-open trial reopens the circuit."""
    now_val = [0.0]
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_after_s=10.0,
        dependency_errors=(RetryableError,),
        clock=lambda: now_val[0],
    )
    with pytest.raises(RetryableError):
        cb.call(lambda: (_ for _ in ()).throw(RetryableError("dep down")))

    now_val[0] = 11.0
    with pytest.raises(RetryableError):
        cb.call(lambda: (_ for _ in ()).throw(RetryableError("still down")))

    assert cb.state == State.OPEN


def test_circuit_breaker_does_not_count_non_dependency_errors() -> None:
    """Non-dependency exceptions do not increment the failure counter."""
    now_val = [0.0]
    cb = CircuitBreaker(
        failure_threshold=2,
        recovery_after_s=10.0,
        dependency_errors=(RetryableError,),
        clock=lambda: now_val[0],
    )
    with pytest.raises(ValueError):
        cb.call(lambda: (_ for _ in ()).throw(ValueError("validation")))

    assert cb.state == State.CLOSED
    assert cb._failures == 0


def test_circuit_breaker_state_change_observer_fires() -> None:
    """on_state_change observer is called on every state transition."""
    transitions: list[tuple[State, State]] = []
    now_val = [0.0]
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_after_s=10.0,
        dependency_errors=(RetryableError,),
        clock=lambda: now_val[0],
        on_state_change=lambda old, new: transitions.append((old, new)),
    )
    with pytest.raises(RetryableError):
        cb.call(lambda: (_ for _ in ()).throw(RetryableError("dep down")))

    assert (State.CLOSED, State.OPEN) in transitions

    now_val[0] = 11.0
    cb.call(lambda: "ok")

    assert (State.OPEN, State.HALF_OPEN) in transitions
    assert (State.HALF_OPEN, State.CLOSED) in transitions


# ---------------------------------------------------------------------------
# solution.py integration tests
# ---------------------------------------------------------------------------

def test_solution_retry_budget_wraps_flaky_operation() -> None:
    solution = load_solution()
    attempts = {"count": 0}
    observed: list[tuple[int, float]] = []

    def flaky():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("slow tool")
        return "ok"

    result = solution.call_with_retry_budget(
        flaky,
        retryable=(TimeoutError,),
        observer=lambda attempt, exc, delay: observed.append((attempt, delay)),
        sleep=lambda _d: None,
        jitter=lambda lo, hi: (lo + hi) / 2,
    )
    assert result == "ok"
    assert attempts["count"] == 2
    assert len(observed) == 1


def test_solution_retry_budget_exhausts_timeout() -> None:
    solution = load_solution()
    attempts = {"count": 0}

    def always_timeout():
        attempts["count"] += 1
        raise TimeoutError("still down")

    with pytest.raises(RetryExhausted):
        solution.call_with_retry_budget(
            always_timeout,
            max_attempts=3,
            retryable=(TimeoutError,),
            sleep=lambda _d: None,
        )

    assert attempts["count"] == 3


def test_solution_retry_budget_does_not_retry_non_retryable_exception() -> None:
    solution = load_solution()
    attempts = {"count": 0}

    def bad_request():
        attempts["count"] += 1
        raise ValueError("invalid arguments")

    with pytest.raises(ValueError):
        solution.call_with_retry_budget(bad_request, max_attempts=3)

    assert attempts["count"] == 1


def test_solution_circuit_opens_after_dependency_failures() -> None:
    solution = load_solution()
    now_val = [100.0]
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_after_s=10.0,
        dependency_errors=(TimeoutError,),
        clock=lambda: now_val[0],
    )

    with pytest.raises(RetryExhausted):
        solution.call_with_retry_budget_and_circuit(
            lambda: (_ for _ in ()).throw(TimeoutError("downstream slow")),
            cb,
            max_attempts=1,
            retryable=(TimeoutError,),
            sleep=lambda _d: None,
        )

    assert cb.state == State.OPEN
    assert cb._opened_at == pytest.approx(100.0)

    with pytest.raises(CircuitOpenError):
        solution.call_with_retry_budget_and_circuit(
            lambda: "should not run",
            cb,
            retryable=(TimeoutError,),
            sleep=lambda _d: None,
        )


def test_solution_circuit_half_open_success_closes_breaker() -> None:
    solution = load_solution()
    now_val = [100.0]
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_after_s=10.0,
        dependency_errors=(TimeoutError,),
        clock=lambda: now_val[0],
    )

    with pytest.raises(RetryExhausted):
        solution.call_with_retry_budget_and_circuit(
            lambda: (_ for _ in ()).throw(TimeoutError("downstream slow")),
            cb,
            max_attempts=1,
            retryable=(TimeoutError,),
            sleep=lambda _d: None,
        )

    # Advance past recovery window
    now_val[0] = 111.0
    result = solution.call_with_retry_budget_and_circuit(
        lambda: "ok",
        cb,
        retryable=(TimeoutError,),
        sleep=lambda _d: None,
    )
    assert result == "ok"
    assert cb.state == State.CLOSED
    assert cb._failures == 0


def test_solution_circuit_does_not_count_non_dependency_exception() -> None:
    solution = load_solution()
    now_val = [100.0]
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_after_s=10.0,
        dependency_errors=(TimeoutError,),
        clock=lambda: now_val[0],
    )

    with pytest.raises(ValueError):
        solution.call_with_retry_budget_and_circuit(
            lambda: (_ for _ in ()).throw(ValueError("bad request")),
            cb,
            retryable=(TimeoutError,),
            sleep=lambda _d: None,
        )

    assert cb.state == State.CLOSED
    assert cb._failures == 0
