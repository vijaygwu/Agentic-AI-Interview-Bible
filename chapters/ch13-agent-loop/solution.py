from __future__ import annotations

from agentic_interview_bible import (
    AgentExecutor,
    AllowListedToolAuthorizer,
    TaskBudget,
    ToolRegistry,
)


def build_executor(
    model,
    tools,
    max_steps: int = 4,
    allowed_tool_names: set[str] | None = None,
    budget: TaskBudget | None = None,
) -> AgentExecutor:
    tool_list = list(tools)
    allowed = (
        {tool.name for tool in tool_list}
        if allowed_tool_names is None
        else allowed_tool_names
    )
    registry = ToolRegistry(
        tool_list,
        authorizer=AllowListedToolAuthorizer(allowed),
    )
    return AgentExecutor(
        model=model,
        tools=registry,
        max_steps=max_steps,
        budget=budget,
    )
