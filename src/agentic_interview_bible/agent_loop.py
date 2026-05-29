from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from .cost_budget import TaskBudget
from .eval_harness import redact_output


class AgentLoopError(Exception):
    """Base class for controlled executor failures."""


class DuplicateToolError(AgentLoopError, ValueError):
    """Raised when a tool registry receives the same tool name twice."""


class InvalidAgentStepError(AgentLoopError, ValueError):
    """Raised when the model returns an invalid step."""


class MaxStepsExceededError(AgentLoopError, RuntimeError):
    """Raised when the executor exhausts its step budget."""


class ToolExecutionError(AgentLoopError, RuntimeError):
    """Raised when a tool fails during execution."""


class ToolAuthorizationError(AgentLoopError, PermissionError):
    """Raised when policy denies a tool call."""


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
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        has_tool_call = self.tool_call is not None
        has_final_answer = self.final_answer is not None
        if has_tool_call == has_final_answer:
            raise InvalidAgentStepError(
                "model step must contain exactly one of tool_call or final_answer"
            )
        for field_name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if value is not None and value < 0:
                raise InvalidAgentStepError(f"{field_name} must be non-negative")

    @property
    def usage_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


@dataclass(frozen=True)
class ToolExecutionContext:
    task_id: str
    step_index: int
    trace_id: str
    idempotency_key: str
    deadline_ms: int | None = None
    actor_id: str | None = None
    tenant_id: str | None = None
    delegated_scopes: tuple[str, ...] = ()
    request_source: str = "interactive"


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    result: object
    context: ToolExecutionContext
    data_classification: str = "model_safe"
    truncated: bool = False


HistoryEvent = AgentStep | ToolResult


class Model(Protocol):
    def next_step(self, prompt: str, history: list[HistoryEvent]) -> AgentStep:
        """Return the next agent step for a prompt and prior execution history."""


class Tool(Protocol):
    name: str

    def run(
        self,
        arguments: dict[str, object],
        context: ToolExecutionContext,
    ) -> object:
        """Execute the tool with validated arguments."""


class ToolAuthorizer(Protocol):
    def authorize(self, call: ToolCall, context: ToolExecutionContext) -> None:
        """Raise ToolAuthorizationError if the tool call is not allowed."""


class AllowAllToolAuthorizer:
    def authorize(self, call: ToolCall, context: ToolExecutionContext) -> None:
        del call, context


class AllowListedToolAuthorizer:
    def __init__(
        self,
        allowed_tool_names: set[str],
        required_scopes_by_tool: dict[str, set[str]] | None = None,
    ) -> None:
        self.allowed_tool_names = allowed_tool_names
        self.required_scopes_by_tool = required_scopes_by_tool or {}

    def authorize(self, call: ToolCall, context: ToolExecutionContext) -> None:
        if call.name not in self.allowed_tool_names:
            raise ToolAuthorizationError(f"tool not authorized: {call.name}")
        required_scopes = self.required_scopes_by_tool.get(call.name, set())
        delegated_scopes = set(context.delegated_scopes)
        missing_scopes = required_scopes - delegated_scopes
        if missing_scopes:
            missing = ", ".join(sorted(missing_scopes))
            raise ToolAuthorizationError(f"missing delegated scopes for {call.name}: {missing}")


class ToolRegistry:
    def __init__(
        self,
        tools: list[Tool],
        authorizer: ToolAuthorizer | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise DuplicateToolError(f"duplicate tool: {tool.name}")
            self._tools[tool.name] = tool
        self.authorizer = authorizer or AllowAllToolAuthorizer()

    def run(self, call: ToolCall, context: ToolExecutionContext) -> object:
        try:
            tool = self._tools[call.name]
        except KeyError as exc:
            raise UnknownToolError(f"unknown tool: {call.name}") from exc
        self.authorizer.authorize(call, context)
        try:
            return tool.run(call.arguments, context)
        except AgentLoopError:
            raise
        except Exception as exc:  # noqa: BLE001 - tools are user-provided code.
            raise ToolExecutionError(f"tool failed: {call.name}") from exc


class AgentExecutor:
    def __init__(
        self,
        model: Model,
        tools: ToolRegistry,
        max_steps: int = 8,
        budget: TaskBudget | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.model = model
        self.tools = tools
        self.max_steps = max_steps
        self.budget = budget

    def run(
        self,
        prompt: str,
        task_id: str | None = None,
        deadline_ms: int | None = None,
        actor_id: str | None = None,
        tenant_id: str | None = None,
        delegated_scopes: tuple[str, ...] = (),
        request_source: str = "interactive",
    ) -> str:
        task_id = task_id or f"task-{uuid4().hex}"
        trace_id = uuid4().hex
        history: list[HistoryEvent] = []
        for step_index in range(self.max_steps):
            step = self.model.next_step(prompt, history)
            if not isinstance(step, AgentStep):
                raise InvalidAgentStepError("model returned a non-AgentStep value")
            if self.budget is not None:
                self.budget.record_model_call(
                    tokens=step.usage_tokens
                    if step.usage_tokens is not None
                    else _estimate_prompt_tokens(prompt, history)
                )
            history.append(step)

            if step.final_answer is not None:
                return step.final_answer

            if step.tool_call is None:  # Defensive; AgentStep enforces this already.
                raise InvalidAgentStepError("model returned neither final_answer nor tool_call")

            context = ToolExecutionContext(
                task_id=task_id,
                step_index=step_index,
                trace_id=trace_id,
                idempotency_key=f"{task_id}:{step_index}:{step.tool_call.name}",
                deadline_ms=deadline_ms,
                actor_id=actor_id,
                tenant_id=tenant_id,
                delegated_scopes=delegated_scopes,
                request_source=request_source,
            )
            result = self.tools.run(step.tool_call, context)
            safe_result, truncated = _safe_tool_result(result)
            history.append(
                ToolResult(
                    tool_name=step.tool_call.name,
                    result=safe_result,
                    context=context,
                    truncated=truncated,
                )
            )

        raise MaxStepsExceededError("agent exceeded max_steps")


def _estimate_prompt_tokens(prompt: str, history: list[HistoryEvent]) -> int:
    history_tokens = sum(len(str(event).split()) for event in history)
    return max(1, len(prompt.split()) + history_tokens)


_MAX_TOOL_STRING_CHARS = 240
_MAX_TOOL_COLLECTION_ITEMS = 20
_SENSITIVE_KEY_TERMS = {
    "api_key",
    "authorization",
    "bearer",
    "chain_of_thought",
    "email",
    "key",
    "password",
    "phone",
    "private",
    "prompt",
    "raw_prompt",
    "secret",
    "ssn",
    "token",
}


def _safe_tool_result(value: object) -> tuple[object, bool]:
    return _sanitize_tool_result(value)


def _sanitize_tool_result(value: object) -> tuple[object, bool]:
    if value is None or isinstance(value, bool | int | float):
        return value, False

    if isinstance(value, str):
        redacted = redact_output(value)
        if len(redacted) > _MAX_TOOL_STRING_CHARS:
            return redacted[:_MAX_TOOL_STRING_CHARS] + "...[truncated]", True
        return redacted, redacted != value

    if isinstance(value, dict):
        safe: dict[str, object] = {}
        truncated = False
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_TOOL_COLLECTION_ITEMS:
                truncated = True
                break
            key_text = str(key)
            safe_key = key_text[:80]
            if _is_sensitive_result_key(key_text):
                safe[safe_key] = "[redacted]"
                truncated = True
                continue
            safe_item, item_truncated = _sanitize_tool_result(item)
            safe[safe_key] = safe_item
            truncated = truncated or item_truncated
        if truncated:
            safe["_truncated_or_redacted"] = True
        return safe, truncated

    if isinstance(value, list | tuple):
        safe_items: list[object] = []
        truncated = len(value) > _MAX_TOOL_COLLECTION_ITEMS
        for item in value[:_MAX_TOOL_COLLECTION_ITEMS]:
            safe_item, item_truncated = _sanitize_tool_result(item)
            safe_items.append(safe_item)
            truncated = truncated or item_truncated
        if truncated:
            safe_items.append("[truncated]")
        return safe_items, truncated

    return _sanitize_tool_result(str(value))


def _is_sensitive_result_key(key: str) -> bool:
    normalized = "".join(
        character.casefold() if character.isalnum() else "_"
        for character in key
    )
    parts = [part for part in normalized.split("_") if part]
    if not parts:
        return False
    if any(part in _SENSITIVE_KEY_TERMS for part in parts):
        return True
    collapsed = "".join(parts)
    return any(term.replace("_", "") in collapsed for term in _SENSITIVE_KEY_TERMS)
