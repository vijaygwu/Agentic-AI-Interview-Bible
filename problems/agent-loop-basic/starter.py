from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


class AgentLoopError(Exception):
    """Base class for controlled agent-loop failures."""


class DuplicateToolError(AgentLoopError, ValueError):
    """Raised when two tools register the same name."""


class InvalidAgentStepError(AgentLoopError, ValueError):
    """Raised when model output does not match the AgentStep contract."""


class MaxStepsExceededError(AgentLoopError, RuntimeError):
    """Raised when the executor spends every allowed step."""


class ToolExecutionError(AgentLoopError, RuntimeError):
    """Raised when a tool dependency fails during execution."""


class UnknownToolError(AgentLoopError, ValueError):
    """Raised when the model asks for a tool that is not registered."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class AgentStep:
    thought_summary: str
    tool_call: ToolCall | None = None
    final_answer: str | None = None

    def __post_init__(self) -> None:
        has_tool_call = self.tool_call is not None
        has_final_answer = self.final_answer is not None
        if has_tool_call == has_final_answer:
            raise InvalidAgentStepError(
                "AgentStep must contain exactly one of tool_call or final_answer"
            )


@dataclass(frozen=True)
class ToolExecutionContext:
    task_id: str
    step_index: int
    trace_id: str
    idempotency_key: str


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    result: object
    context: ToolExecutionContext


HistoryEvent = AgentStep | ToolResult


class Model(Protocol):
    def next_step(self, prompt: str, history: list[HistoryEvent]) -> AgentStep:
        """Return the next step for the prompt and prior history."""


class Tool(Protocol):
    name: str

    def run(
        self,
        arguments: dict[str, object],
        context: ToolExecutionContext,
    ) -> object:
        """Run the tool with validated arguments and execution context."""


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        raise NotImplementedError

    def run(self, call: ToolCall, context: ToolExecutionContext) -> object:
        raise NotImplementedError


class AgentExecutor:
    def __init__(self, model: Model, tools: ToolRegistry, max_steps: int = 4) -> None:
        raise NotImplementedError

    def run(self, prompt: str, task_id: str | None = None) -> str:
        raise NotImplementedError


def build_executor(
    model: Model,
    tools: Iterable[Tool],
    max_steps: int = 4,
) -> AgentExecutor:
    # This factory is given; your task is to implement ToolRegistry and
    # AgentExecutor above (the methods that currently raise NotImplementedError).
    return AgentExecutor(
        model=model,
        tools=ToolRegistry(tools),
        max_steps=max_steps,
    )

