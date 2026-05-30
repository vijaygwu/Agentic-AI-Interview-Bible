"""Chapter 25 exercise — Bounded Loop with Token Budget.

Implement the ``TokenBudget`` reserve/consume pattern and
the ``call_with_budget`` orchestrator described in the book
(problem-bounded-loop-budget.tex).

Classes and functions to implement:

1. ``BudgetExhausted(remaining, requested)``
   Typed exception; attributes ``.remaining`` and ``.requested``.

2. ``TokenBudget(total, used=0)``
   Dataclass with:
     - ``.remaining() -> int``
     - ``.reserve(estimated) -> None``  (raises BudgetExhausted if over)
     - ``.consume(actual) -> None``

3. ``call_with_budget(model, prompt, budget)``
   Calls ``model.estimate_tokens(prompt)``, reserves, calls
   ``model.call(prompt)``, then consumes ``response.tokens_used``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class BudgetExhausted(Exception):
    def __init__(self, remaining: int, requested: int) -> None:
        super().__init__(
            f"budget exhausted: {remaining} remaining, {requested} requested"
        )
        self.remaining = remaining
        self.requested = requested


@dataclass
class TokenBudget:
    total: int
    used: int = 0

    def remaining(self) -> int:
        raise NotImplementedError

    def reserve(self, estimated: int) -> None:
        raise NotImplementedError

    def consume(self, actual: int) -> None:
        raise NotImplementedError


def call_with_budget(model: Any, prompt: Any, budget: TokenBudget) -> Any:
    raise NotImplementedError
