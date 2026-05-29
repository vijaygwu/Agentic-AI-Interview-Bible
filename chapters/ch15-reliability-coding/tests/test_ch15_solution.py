import importlib.util
from pathlib import Path

import pytest

from agentic_interview_bible import (
    BreakerState,
    CircuitBreaker,
    CircuitOpenError,
    RetryExhaustedError,
)


def load_solution():
    path = Path(__file__).parents[1] / "solution.py"
    spec = importlib.util.spec_from_file_location("ch15_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_retry_budget_wraps_flaky_operation() -> None:
    solution = load_solution()
    attempts = {"count": 0}
    observed: list[tuple[int, float]] = []

    def flaky():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("slow tool")
        return "ok"

    assert solution.call_with_retry_budget(
        flaky,
        observer=lambda attempt, exc, delay: observed.append((attempt, delay)),
        sleep=lambda _d: None,
        rng=lambda: 1.0,
    ) == "ok"
    assert attempts["count"] == 2
    assert observed == [(1, 0.25)]


def test_retry_budget_exhausts_timeout() -> None:
    solution = load_solution()
    attempts = {"count": 0}

    def always_timeout():
        attempts["count"] += 1
        raise TimeoutError("still down")

    with pytest.raises(RetryExhaustedError):
        solution.call_with_retry_budget(
            always_timeout, max_attempts=3, sleep=lambda _d: None
        )

    assert attempts["count"] == 3


def test_retry_budget_does_not_retry_non_retryable_exception() -> None:
    solution = load_solution()
    attempts = {"count": 0}

    def bad_request():
        attempts["count"] += 1
        raise ValueError("invalid arguments")

    with pytest.raises(ValueError):
        solution.call_with_retry_budget(bad_request, max_attempts=3)

    assert attempts["count"] == 1


def test_retry_circuit_opens_after_dependency_failures() -> None:
    solution = load_solution()
    breaker = CircuitBreaker(failure_threshold=1, recovery_after=10.0)

    with pytest.raises(RetryExhaustedError):
        solution.call_with_retry_budget_and_circuit(
            lambda: (_ for _ in ()).throw(TimeoutError("downstream slow")),
            breaker,
            now=100.0,
            max_attempts=1,
        )

    assert breaker.state == BreakerState.OPEN
    assert breaker.opened_at == 100.0

    with pytest.raises(CircuitOpenError):
        solution.call_with_retry_budget_and_circuit(
            lambda: "should not run",
            breaker,
            now=101.0,
        )


def test_retry_circuit_half_open_success_closes_breaker() -> None:
    solution = load_solution()
    breaker = CircuitBreaker(failure_threshold=1, recovery_after=10.0)

    with pytest.raises(RetryExhaustedError):
        solution.call_with_retry_budget_and_circuit(
            lambda: (_ for _ in ()).throw(TimeoutError("downstream slow")),
            breaker,
            now=100.0,
            max_attempts=1,
        )

    assert solution.call_with_retry_budget_and_circuit(
        lambda: "ok",
        breaker,
        now=111.0,
    ) == "ok"
    assert breaker.state == BreakerState.CLOSED
    assert breaker.failure_count == 0


def test_retry_circuit_does_not_count_non_dependency_exception() -> None:
    solution = load_solution()
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_after=10.0,
        failure_exceptions=(TimeoutError,),
    )

    with pytest.raises(ValueError):
        solution.call_with_retry_budget_and_circuit(
            lambda: (_ for _ in ()).throw(ValueError("bad request")),
            breaker,
            now=100.0,
        )

    assert breaker.state == BreakerState.CLOSED
    assert breaker.failure_count == 0
