from __future__ import annotations

class AdmissionDecision:
    def __init__(self, action: str, reason: str) -> None:
        self.action = action
        self.reason = reason


def record_calls(budget, token_counts: list[int]) -> None:
    if any(tokens < 0 for tokens in token_counts):
        raise ValueError("tokens must be non-negative")
    for tokens in token_counts:
        budget.record_model_call(tokens)


def admit_task(
    budget,
    estimated_tokens: int,
    estimated_model_calls: int = 1,
    high_risk: bool = False,
    read_only_fallback: bool = False,
) -> AdmissionDecision:
    """Decide whether to admit a task under the budget.

    This is a check-only gate: it reads current budget usage but does not
    reserve it. Under concurrent callers that is a TOCTOU race (two tasks can
    both observe headroom and both be admitted, overcommitting the budget).
    Production admission control must reserve atomically, e.g. under a lock or
    a transactional compare-and-increment on the budget counters.
    """
    if estimated_tokens < 0 or estimated_model_calls < 1:
        return AdmissionDecision("reject", "invalid admission estimate")

    would_exceed_calls = budget.model_calls + estimated_model_calls > budget.max_model_calls
    would_exceed_tokens = budget.tokens + estimated_tokens > budget.max_tokens
    if would_exceed_calls or would_exceed_tokens:
        if high_risk:
            return AdmissionDecision("escalate", "budget exhausted for high-risk task")
        if read_only_fallback:
            return AdmissionDecision("read_only", "budget exhausted; use cached context only")
        return AdmissionDecision("queue", "budget exhausted; defer nonurgent work")
    return AdmissionDecision("run", "within budget")
