from __future__ import annotations


class AdmissionDecision:
    action: str
    reason: str


def record_calls(budget, token_counts: list[int]) -> None:
    """Implement budgeted admission control for one agent task.

    Requirements:
    - record each model call against a TaskBudget
    - reject negative token counts before mutating budget state
    - raise typed budget errors when model-call or token budgets are exhausted
    - expose a degradation decision for callers: run normally, queue/defer,
      switch to read-only partial answer, or escalate high-risk work
    - preserve counters so observability can report cost per successful task
    """
    raise NotImplementedError


def admit_task(
    budget,
    estimated_tokens: int,
    estimated_model_calls: int = 1,
    high_risk: bool = False,
    read_only_fallback: bool = False,
) -> AdmissionDecision:
    """Return the admission or degradation decision for one pending task.

    Outcomes:
    - run when the estimated task fits within budget
    - reject invalid token or model-call estimates without mutating counters
    - queue when nonurgent work exceeds budget
    - read_only when callers can answer from cached context without tool writes
    - escalate when high-risk work exceeds budget
    - account for multi-step agents by checking estimated_model_calls, not just
      one future model call
    """
    raise NotImplementedError
