import importlib.util
from pathlib import Path

import pytest

from agentic_interview_bible import (
    AgentStep,
    DuplicateToolError,
    MockLLM,
    ScriptedStep,
    ToolAuthorizationError,
    ToolCall,
    ToolExecutionError,
    UnknownToolError,
)


def load_solution():
    path = Path(__file__).parents[1] / "solution.py"
    spec = importlib.util.spec_from_file_location("ch13_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EchoTool:
    name = "echo"

    def run(self, arguments, context):
        del context
        return arguments["text"]


class RecordingTool:
    name = "record"

    def __init__(self) -> None:
        self.contexts = []

    def run(self, arguments, context):
        assert arguments == {"text": "ready"}
        assert context.task_id == "ctx-task"
        assert context.step_index == 0
        assert context.trace_id
        assert context.idempotency_key == "ctx-task:0:record"
        assert context.deadline_ms == 500
        assert context.actor_id == "user-1"
        assert context.tenant_id == "tenant-1"
        assert context.delegated_scopes == ("tool:run",)
        assert context.request_source == "test"
        self.contexts.append(context)
        return arguments["text"]


class FailingTool:
    name = "failing"

    def run(self, arguments, context):
        del arguments, context
        raise RuntimeError("dependency down")


def test_build_executor_runs_tool_flow() -> None:
    solution = load_solution()
    model = MockLLM(
        [
            ScriptedStep(
                AgentStep("need echo", tool_call=ToolCall("echo", {"text": "ready"}))
            ),
            ScriptedStep(AgentStep("done", final_answer="ready")),
        ]
    )

    assert solution.build_executor(model, [EchoTool()]).run("echo", task_id="ch13") == "ready"


def test_build_executor_passes_context_and_budget() -> None:
    solution = load_solution()
    tool = RecordingTool()
    model = MockLLM(
        [
            ScriptedStep(
                AgentStep(
                    "need record",
                    tool_call=ToolCall("record", {"text": "ready"}),
                    input_tokens=2,
                    output_tokens=3,
                )
            ),
            ScriptedStep(AgentStep("done", final_answer="ready", input_tokens=1)),
        ]
    )
    from agentic_interview_bible import TaskBudget

    budget = TaskBudget(max_model_calls=2, max_tokens=10)
    executor = solution.build_executor(model, [tool], budget=budget)

    assert (
        executor.run(
            "record",
            task_id="ctx-task",
            deadline_ms=500,
            actor_id="user-1",
            tenant_id="tenant-1",
            delegated_scopes=("tool:run",),
            request_source="test",
        )
        == "ready"
    )
    assert len(tool.contexts) == 1
    assert budget.tokens == 6


def test_build_executor_rejects_duplicate_tool_names() -> None:
    solution = load_solution()
    model = MockLLM([])

    with pytest.raises(DuplicateToolError):
        solution.build_executor(model, [EchoTool(), EchoTool()])


def test_build_executor_rejects_unknown_tool() -> None:
    solution = load_solution()
    model = MockLLM(
        [ScriptedStep(AgentStep("need missing", tool_call=ToolCall("missing", {})))]
    )

    with pytest.raises(UnknownToolError):
        solution.build_executor(model, [EchoTool()], max_steps=1).run("missing")


def test_build_executor_rejects_denied_tool() -> None:
    solution = load_solution()
    model = MockLLM(
        [ScriptedStep(AgentStep("need echo", tool_call=ToolCall("echo", {"text": "x"})))]
    )

    with pytest.raises(ToolAuthorizationError):
        solution.build_executor(
            model,
            [EchoTool()],
            allowed_tool_names={"other"},
            max_steps=1,
        ).run("denied")


def test_build_executor_honors_empty_allow_list() -> None:
    solution = load_solution()
    model = MockLLM(
        [ScriptedStep(AgentStep("need echo", tool_call=ToolCall("echo", {"text": "x"})))]
    )

    with pytest.raises(ToolAuthorizationError):
        solution.build_executor(
            model,
            [EchoTool()],
            allowed_tool_names=set(),
            max_steps=1,
        ).run("empty allow list")


def test_build_executor_wraps_tool_failure() -> None:
    solution = load_solution()
    model = MockLLM(
        [ScriptedStep(AgentStep("need failing", tool_call=ToolCall("failing", {})))]
    )

    with pytest.raises(ToolExecutionError):
        solution.build_executor(model, [FailingTool()], max_steps=1).run("failure")
