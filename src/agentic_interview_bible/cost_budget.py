from __future__ import annotations

from dataclasses import dataclass


class BudgetExceededError(RuntimeError):
    """Raised when an agent exceeds the configured task budget."""


@dataclass
class TaskBudget:
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
        if tokens < 0:
            raise ValueError("tokens must be non-negative")

        next_model_calls = self.model_calls + 1
        next_tokens = self.tokens + tokens
        if next_model_calls > self.max_model_calls:
            raise BudgetExceededError("model-call budget exceeded")
        if next_tokens > self.max_tokens:
            raise BudgetExceededError("token budget exceeded")

        self.model_calls = next_model_calls
        self.tokens = next_tokens
