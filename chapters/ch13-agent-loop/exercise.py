from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_interview_bible import AgentExecutor, TaskBudget


def build_executor(
    model,
    tools,
    max_steps: int = 4,
    allowed_tool_names: set[str] | None = None,
    budget: TaskBudget | None = None,
) -> AgentExecutor:
    """Return an interview-grade AgentExecutor.

    Requirements:
    - register tools with duplicate-name detection
    - authorize tool calls against allowed_tool_names, or all registered tools
      when allowed_tool_names is omitted
    - enforce max_steps
    - propagate task_id, step_index, deadline, trace_id, and idempotency key
    - pass the optional TaskBudget into the executor for model-call accounting
    - expose typed failures for unknown tools, denied tools, tool exceptions,
      and exhausted step budgets
    """
    raise NotImplementedError
