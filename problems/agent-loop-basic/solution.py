"""Standalone, deliberately reduced agent loop for this coding exercise.

This is a self-contained simplification scoped to the problem. The production
implementation, with logical-action idempotency, task budgets, sanitized
traces, and the circuit breaker, lives in the ``agentic_interview_bible``
package (see ``agent_loop.py``); this file intentionally reimplements a
minimal version so the exercise stays self-contained.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol
from uuid import uuid4


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
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise DuplicateToolError(f"duplicate tool: {tool.name}")
            self._tools[tool.name] = tool

    def run(self, call: ToolCall, context: ToolExecutionContext) -> object:
        try:
            tool = self._tools[call.name]
        except KeyError as exc:
            raise UnknownToolError(f"unknown tool: {call.name}") from exc

        try:
            return tool.run(call.arguments, context)
        except AgentLoopError:
            raise
        except Exception as exc:  # noqa: BLE001 - tools are user-provided code.
            raise ToolExecutionError(f"tool failed: {call.name}") from exc


class AgentExecutor:
    def __init__(self, model: Model, tools: ToolRegistry, max_steps: int = 4) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.model = model
        self.tools = tools
        self.max_steps = max_steps

    def run(self, prompt: str, task_id: str | None = None) -> str:
        task_id = task_id or f"task-{uuid4().hex}"
        trace_id = uuid4().hex
        history: list[HistoryEvent] = []

        for step_index in range(self.max_steps):
            step = self.model.next_step(prompt, list(history))
            if not isinstance(step, AgentStep):
                raise InvalidAgentStepError("model returned a non-AgentStep value")
            history.append(step)

            if step.final_answer is not None:
                return step.final_answer

            if step.tool_call is None:
                raise InvalidAgentStepError("model returned no tool call")

            context = ToolExecutionContext(
                task_id=task_id,
                step_index=step_index,
                trace_id=trace_id,
                idempotency_key=f"{task_id}:{step_index}:{step.tool_call.name}",
            )
            result = self.tools.run(step.tool_call, context)
            history.append(
                ToolResult(
                    tool_name=step.tool_call.name,
                    result=result,
                    context=context,
                )
            )

        raise MaxStepsExceededError("agent exceeded max_steps")


def build_executor(
    model: Model,
    tools: Iterable[Tool],
    max_steps: int = 4,
) -> AgentExecutor:
    return AgentExecutor(
        model=model,
        tools=ToolRegistry(tools),
        max_steps=max_steps,
    )

