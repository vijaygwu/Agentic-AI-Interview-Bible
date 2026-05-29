import importlib.util
from pathlib import Path

import pytest

from agentic_interview_bible import BudgetExceededError, TaskBudget


def load_solution():
    path = Path(__file__).parents[1] / "solution.py"
    spec = importlib.util.spec_from_file_location("ch25_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_record_calls_updates_budget() -> None:
    solution = load_solution()
    budget = TaskBudget(max_model_calls=3, max_tokens=100)

    solution.record_calls(budget, [10, 20])

    assert budget.model_calls == 2
    assert budget.tokens == 30


def test_record_calls_blocks_token_budget_excess() -> None:
    solution = load_solution()
    budget = TaskBudget(max_model_calls=3, max_tokens=10)

    with pytest.raises(BudgetExceededError):
        solution.record_calls(budget, [7, 5])


def test_record_calls_blocks_model_call_excess() -> None:
    solution = load_solution()
    budget = TaskBudget(max_model_calls=1, max_tokens=100)

    with pytest.raises(BudgetExceededError):
        solution.record_calls(budget, [10, 10])


def test_record_calls_rejects_negative_token_count_without_mutating() -> None:
    solution = load_solution()
    budget = TaskBudget(max_model_calls=2, max_tokens=100)

    with pytest.raises(ValueError):
        solution.record_calls(budget, [-1])

    assert budget.model_calls == 0
    assert budget.tokens == 0


def test_record_calls_rejects_later_negative_without_partial_mutation() -> None:
    solution = load_solution()
    budget = TaskBudget(max_model_calls=2, max_tokens=100)

    with pytest.raises(ValueError):
        solution.record_calls(budget, [10, -1])

    assert budget.model_calls == 0
    assert budget.tokens == 0


def test_admit_task_returns_degradation_decisions() -> None:
    solution = load_solution()

    normal_budget = TaskBudget(max_model_calls=2, max_tokens=100)
    assert solution.admit_task(normal_budget, estimated_tokens=10).action == "run"
    assert normal_budget.model_calls == 0
    assert normal_budget.tokens == 0

    queue_budget = TaskBudget(max_model_calls=1, max_tokens=10)
    assert solution.admit_task(queue_budget, estimated_tokens=11).action == "queue"
    assert queue_budget.model_calls == 0
    assert queue_budget.tokens == 0

    partial_budget = TaskBudget(max_model_calls=1, max_tokens=10)
    assert solution.admit_task(
        partial_budget,
        estimated_tokens=11,
        read_only_fallback=True,
    ).action == "read_only"
    assert partial_budget.model_calls == 0
    assert partial_budget.tokens == 0

    risk_budget = TaskBudget(max_model_calls=1, max_tokens=10)
    assert solution.admit_task(
        risk_budget,
        estimated_tokens=11,
        high_risk=True,
    ).action == "escalate"
    assert risk_budget.model_calls == 0
    assert risk_budget.tokens == 0

    invalid_budget = TaskBudget(max_model_calls=1, max_tokens=10)
    assert solution.admit_task(invalid_budget, estimated_tokens=-1).action == "reject"

    invalid_calls_budget = TaskBudget(max_model_calls=1, max_tokens=10)
    assert solution.admit_task(
        invalid_calls_budget,
        estimated_tokens=1,
        estimated_model_calls=0,
    ).action == "reject"


def test_admit_task_accounts_for_multi_call_agents_without_mutating() -> None:
    solution = load_solution()
    budget = TaskBudget(max_model_calls=2, max_tokens=1_000)

    decision = solution.admit_task(
        budget,
        estimated_tokens=100,
        estimated_model_calls=3,
    )

    assert decision.action == "queue"
    assert budget.model_calls == 0
    assert budget.tokens == 0
