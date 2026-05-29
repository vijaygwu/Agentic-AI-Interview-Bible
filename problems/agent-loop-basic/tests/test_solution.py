import importlib.util
import sys
from pathlib import Path

import pytest


def load_solution():
    path = Path(__file__).parents[1] / "solution.py"
    spec = importlib.util.spec_from_file_location("agent_loop_basic_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ScriptedModel:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []

    def next_step(self, prompt, history):
        self.calls.append((prompt, history))
        return self.steps.pop(0)


class EchoTool:
    name = "echo"

    def run(self, arguments, context):
        del context
        return arguments["text"]


class RecordingTool:
    name = "lookup"

    def __init__(self):
        self.calls = []

    def run(self, arguments, context):
        self.calls.append((arguments, context))
        return {"value": arguments["key"].upper()}


class FailingTool:
    name = "failing"

    def run(self, arguments, context):
        del arguments, context
        raise RuntimeError("dependency down")


def test_returns_final_answer_without_calling_tools():
    solution = load_solution()

    class ExplodingTool:
        name = "explode"

        def run(self, arguments, context):
            del arguments, context
            raise AssertionError("tool should not run")

    model = ScriptedModel([solution.AgentStep("done", final_answer="ready")])
    executor = solution.build_executor(model, [ExplodingTool()])

    assert executor.run("answer directly", task_id="task-final") == "ready"
    assert len(model.calls) == 1


def test_runs_tool_flow_with_history_and_context():
    solution = load_solution()
    tool = RecordingTool()
    model = ScriptedModel(
        [
            solution.AgentStep(
                "need lookup",
                tool_call=solution.ToolCall("lookup", {"key": "alpha"}),
            ),
            solution.AgentStep("done", final_answer="ALPHA"),
        ]
    )

    executor = solution.build_executor(model, [tool], max_steps=2)

    assert executor.run("lookup alpha", task_id="task-123") == "ALPHA"
    assert len(tool.calls) == 1

    arguments, context = tool.calls[0]
    assert arguments == {"key": "alpha"}
    assert context.task_id == "task-123"
    assert context.step_index == 0
    assert context.trace_id
    assert context.idempotency_key == "task-123:0:lookup"

    assert len(model.calls) == 2
    second_history = model.calls[1][1]
    assert isinstance(second_history[0], solution.AgentStep)
    assert isinstance(second_history[1], solution.ToolResult)
    assert second_history[1].tool_name == "lookup"
    assert second_history[1].result == {"value": "ALPHA"}
    assert second_history[1].context == context


def test_rejects_duplicate_tool_names():
    solution = load_solution()

    with pytest.raises(solution.DuplicateToolError):
        solution.ToolRegistry([EchoTool(), EchoTool()])


def test_rejects_unknown_tool_names():
    solution = load_solution()
    model = ScriptedModel(
        [solution.AgentStep("need missing", tool_call=solution.ToolCall("missing", {}))]
    )
    executor = solution.build_executor(model, [EchoTool()], max_steps=1)

    with pytest.raises(solution.UnknownToolError):
        executor.run("use missing tool")


def test_wraps_tool_failures():
    solution = load_solution()
    model = ScriptedModel(
        [solution.AgentStep("need failing", tool_call=solution.ToolCall("failing", {}))]
    )
    executor = solution.build_executor(model, [FailingTool()], max_steps=1)

    with pytest.raises(solution.ToolExecutionError) as exc_info:
        executor.run("dependency failure")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_rejects_invalid_agent_steps():
    solution = load_solution()

    with pytest.raises(solution.InvalidAgentStepError):
        solution.AgentStep("no output")

    with pytest.raises(solution.InvalidAgentStepError):
        solution.AgentStep(
            "two outputs",
            tool_call=solution.ToolCall("echo", {"text": "x"}),
            final_answer="x",
        )

    model = ScriptedModel([{"tool_call": "not an AgentStep"}])
    executor = solution.build_executor(model, [EchoTool()], max_steps=1)

    with pytest.raises(solution.InvalidAgentStepError):
        executor.run("bad model output")


def test_enforces_max_steps_and_positive_step_count():
    solution = load_solution()

    with pytest.raises(ValueError):
        solution.build_executor(ScriptedModel([]), [EchoTool()], max_steps=0)

    model = ScriptedModel(
        [
            solution.AgentStep(
                "keep going",
                tool_call=solution.ToolCall("echo", {"text": "again"}),
            )
        ]
    )
    executor = solution.build_executor(model, [EchoTool()], max_steps=1)

    with pytest.raises(solution.MaxStepsExceededError):
        executor.run("never finishes")
