"""Token-budget primitives for bounded agent loops.

Public API mirrors the book listing exactly (problem-bounded-loop-budget.tex):

    BudgetExhausted(remaining, requested)   -- typed exception
    TokenBudget(total, used=0)              -- dataclass with
        .remaining() -> int
        .reserve(estimated) -> None
        .consume(actual)    -> None
    call_with_budget(model, prompt, budget) -- orchestrate reserve/consume

Backward-compat shim (used by existing __init__.py exports and ch25 tests):
    BudgetExceededError  -- alias for BudgetExhausted
    TaskBudget           -- call-count + token budget with .record_model_call
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Book API — bounded loop budget (problem-bounded-loop-budget.tex)
# ---------------------------------------------------------------------------

class BudgetExhausted(Exception):
    """Raised before a model call that would exceed the token budget."""

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
        return self.total - self.used

    def reserve(self, estimated: int) -> None:
        """Pre-call check: raise BudgetExhausted if estimated > remaining."""
        if estimated > self.remaining():
            raise BudgetExhausted(self.remaining(), estimated)

    def consume(self, actual: int) -> None:
        """Post-call accounting: subtract actual tokens consumed."""
        self.used += actual


def call_with_budget(model: Any, prompt: Any, budget: TokenBudget) -> Any:
    """Orchestrate reserve-then-consume for a single model call.

    1. Estimate the token cost via model.estimate_tokens(prompt).
    2. Reserve that amount (raises BudgetExhausted if insufficient).
    3. Execute the call via model.call(prompt).
    4. Consume the actual tokens from response.tokens_used.
    """
    estimated = model.estimate_tokens(prompt)
    budget.reserve(estimated)
    response = model.call(prompt)
    budget.consume(response.tokens_used)
    return response


# ---------------------------------------------------------------------------
# Backward-compat shim — kept so __init__.py and ch25 tests compile
# ---------------------------------------------------------------------------

# BudgetExceededError is the old name; alias it so imports keep working.
BudgetExceededError = BudgetExhausted


@dataclass
class TaskBudget:
    """Call-count + token budget used by the ch25 admission-control exercise."""

    max_model_calls: int
    max_tokens: int
    model_calls: int = 0
    tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_model_calls < 1:
            raise ValueError("max_model_calls must be positive")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.model_calls < 0 or self.tokens < 0:
            raise ValueError("initial budget usage must be non-negative")

    def record_model_call(self, tokens: int) -> None:
        # Guard the type, not just the sign: a bool is an int subclass and a
        # float silently corrupts the running count, so reject both.
        if not isinstance(tokens, int) or isinstance(tokens, bool):
            raise ValueError("tokens must be an int")
        if tokens < 0:
            raise ValueError("tokens must be non-negative")

        next_model_calls = self.model_calls + 1
        next_tokens = self.tokens + tokens
        if next_model_calls > self.max_model_calls:
            raise BudgetExceededError(
                self.max_tokens - self.tokens,
                tokens,
            )
        if next_tokens > self.max_tokens:
            raise BudgetExceededError(
                self.max_tokens - self.tokens,
                tokens,
            )

        self.model_calls = next_model_calls
        self.tokens = next_tokens
