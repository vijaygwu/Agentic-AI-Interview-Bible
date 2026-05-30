"""Chapter 25 solution — Bounded Loop with Token Budget.

Exact implementation of the "strong attempt" shown in the book:
    problem-bounded-loop-budget.tex
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from agentic_interview_bible.cost_budget import BudgetExhausted, TokenBudget


def call_with_budget(model: Any, prompt: Any, budget: TokenBudget) -> Any:
    """Reserve then consume: pre-call estimation, post-call accounting."""
    estimated = model.estimate_tokens(prompt)
    budget.reserve(estimated)
    response = model.call(prompt)
    budget.consume(response.tokens_used)
    return response
