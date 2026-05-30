"""Tests for chapter 25 — Bounded Loop with Token Budget."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentic_interview_bible.cost_budget import BudgetExhausted, TokenBudget


def load_solution():
    path = Path(__file__).parents[1] / "solution.py"
    spec = importlib.util.spec_from_file_location("ch25_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _mock_model(estimate: int, actual: int):
    response = MagicMock()
    response.tokens_used = actual
    model = MagicMock()
    model.estimate_tokens.return_value = estimate
    model.call.return_value = response
    return model


# ---------------------------------------------------------------------------
# TokenBudget unit tests
# ---------------------------------------------------------------------------

def test_token_budget_remaining() -> None:
    b = TokenBudget(total=100, used=30)
    assert b.remaining() == 70


def test_token_budget_reserve_succeeds_within_budget() -> None:
    b = TokenBudget(total=100)
    b.reserve(50)  # should not raise


def test_token_budget_reserve_raises_when_over() -> None:
    b = TokenBudget(total=100, used=80)
    with pytest.raises(BudgetExhausted) as exc_info:
        b.reserve(30)
    assert exc_info.value.remaining == 20
    assert exc_info.value.requested == 30


def test_token_budget_reserve_does_not_mutate_used() -> None:
    b = TokenBudget(total=100)
    b.reserve(50)
    assert b.used == 0  # reserve is a check, not a deduction


def test_token_budget_consume_updates_used() -> None:
    b = TokenBudget(total=100)
    b.consume(40)
    assert b.used == 40
    b.consume(25)
    assert b.used == 65


# ---------------------------------------------------------------------------
# Pre-call check (reserve raises before the call)
# ---------------------------------------------------------------------------

def test_pre_call_check_rejects_over_budget_call() -> None:
    sol = load_solution()
    budget = TokenBudget(total=50, used=40)
    # estimate=20, but only 10 remaining -> should raise before model.call
    model = _mock_model(estimate=20, actual=20)

    with pytest.raises(BudgetExhausted):
        sol.call_with_budget(model, "prompt", budget)

    model.call.assert_not_called()


def test_pre_call_check_passes_within_budget() -> None:
    sol = load_solution()
    budget = TokenBudget(total=100)
    model = _mock_model(estimate=30, actual=25)

    response = sol.call_with_budget(model, "prompt", budget)

    model.call.assert_called_once_with("prompt")
    assert response is model.call.return_value


# ---------------------------------------------------------------------------
# Post-call accounting (consume uses actual, not estimate)
# ---------------------------------------------------------------------------

def test_post_call_accounting_uses_actual_tokens() -> None:
    sol = load_solution()
    budget = TokenBudget(total=100)
    # estimate says 30, actual is 25
    model = _mock_model(estimate=30, actual=25)

    sol.call_with_budget(model, "prompt", budget)

    assert budget.used == 25  # actual, not estimate


def test_post_call_accounting_accumulates_across_calls() -> None:
    sol = load_solution()
    budget = TokenBudget(total=200)
    model = _mock_model(estimate=30, actual=20)

    sol.call_with_budget(model, "p1", budget)
    sol.call_with_budget(model, "p2", budget)

    assert budget.used == 40


# ---------------------------------------------------------------------------
# BudgetExhausted exception attributes
# ---------------------------------------------------------------------------

def test_budget_exhausted_carries_remaining_and_requested() -> None:
    exc = BudgetExhausted(remaining=10, requested=50)
    assert exc.remaining == 10
    assert exc.requested == 50
    assert "10" in str(exc)
    assert "50" in str(exc)
