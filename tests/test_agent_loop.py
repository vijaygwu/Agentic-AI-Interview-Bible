import pytest

from agentic_interview_bible.agent_loop import (
    AgentExecutor,
    AgentStep,
    AllowListedToolAuthorizer,
    DuplicateToolError,
    HistoryEvent,
    InvalidAgentStepError,
    MaxStepsExceededError,
    ToolCall,
    ToolAuthorizationError,
    ToolExecutionContext,
    ToolExecutionError,
    ToolRegistry,
    ToolResult,
    UnknownToolError,
)
from agentic_interview_bible.cost_budget import BudgetExceededError, TaskBudget
from agentic_interview_bible.mock_llm import MockLLM, ScriptedStep


class PriceTool:
    name = "lookup_price"

    def run(
        self,
        arguments: dict[str, object],
        context: ToolExecutionContext,
    ) -> object:
        assert arguments == {"sku": "SKU-123"}
        assert context.step_index == 0
        assert context.idempotency_key == f"{context.task_id}:0:lookup_price"
        assert context.actor_id in {None, "user-1"}
        assert context.tenant_id in {None, "tenant-1"}
        return {"price": "$499.99"}


class FailingTool:
    name = "explode"

    def run(
        self,
        arguments: dict[str, object],
        context: ToolExecutionContext,
    ) -> object:
        del arguments, context
        raise RuntimeError("dependency unavailable")


class SensitiveTool:
    name = "customer_lookup"

    def run(
        self,
        arguments: dict[str, object],
        context: ToolExecutionContext,
    ) -> object:
        del arguments, context
        return {
            "access_token": "tok-secret-123",
            "customer_email": "customer@example.com",
            "customer_phone": "555-123-4567",
            "private_notes": "private CRM note",
            "x_api_key": "sk-123456789012345",
            "safe_summary": "eligible for refund",
            "note": "contact customer@example.com with api_key=secret",
            "nested": {
                "refreshToken": "refresh-secret-456",
                "status": "active",
            },
        }


class LargeTool:
    name = "large_payload"

    def run(
        self,
        arguments: dict[str, object],
        context: ToolExecutionContext,
    ) -> object:
        del arguments, context
        return {"text": "x" * 1_000, "items": list(range(30))}


def assert_price_result_seen(history: list[HistoryEvent]) -> None:
    assert any(
        isinstance(event, ToolResult)
        and event.tool_name == "lookup_price"
        and event.result == {"price": "$499.99"}
        for event in history
    )


def test_agent_uses_tool_result_in_final_answer() -> None:
    model = MockLLM(
        [
            ScriptedStep(
                AgentStep(
                    thought_summary="Need the authoritative price.",
                    tool_call=ToolCall("lookup_price", {"sku": "SKU-123"}),
                )
            ),
            ScriptedStep(
                AgentStep(
                    thought_summary="Use the tool result directly.",
                    final_answer="The price is $499.99.",
                ),
                require_history=assert_price_result_seen,
            ),
        ]
    )

    executor = AgentExecutor(
        model=model,
        tools=ToolRegistry([PriceTool()]),
        max_steps=4,
    )

    answer = executor.run("What is the price for SKU-123?", task_id="price-task")

    assert "$499.99" in answer
    assert "$574.99" not in answer


def test_agent_step_rejects_neither_tool_nor_final_answer() -> None:
    with pytest.raises(InvalidAgentStepError):
        AgentStep(thought_summary="ambiguous")


def test_agent_step_rejects_both_tool_and_final_answer() -> None:
    with pytest.raises(InvalidAgentStepError):
        AgentStep(
            thought_summary="ambiguous",
            tool_call=ToolCall("lookup_price", {"sku": "SKU-123"}),
            final_answer="done",
        )


def test_unknown_tool_is_rejected() -> None:
    model = MockLLM(
        [
            ScriptedStep(
                AgentStep(
                    thought_summary="Need a tool.",
                    tool_call=ToolCall("missing_tool", {}),
                )
            )
        ]
    )
    executor = AgentExecutor(model, ToolRegistry([PriceTool()]), max_steps=1)

    with pytest.raises(UnknownToolError):
        executor.run("Use an unknown tool.", task_id="unknown-tool-task")


def test_duplicate_tools_are_rejected() -> None:
    with pytest.raises(DuplicateToolError):
        ToolRegistry([PriceTool(), PriceTool()])


def test_max_steps_exhaustion_is_typed() -> None:
    model = MockLLM(
        [
            ScriptedStep(
                AgentStep(
                    thought_summary="Need the price.",
                    tool_call=ToolCall("lookup_price", {"sku": "SKU-123"}),
                )
            )
        ]
    )
    executor = AgentExecutor(model, ToolRegistry([PriceTool()]), max_steps=1)

    with pytest.raises(MaxStepsExceededError):
        executor.run("Never reaches a final answer.", task_id="max-step-task")


def test_tool_failure_is_wrapped() -> None:
    model = MockLLM(
        [
            ScriptedStep(
                AgentStep(
                    thought_summary="Try risky tool.",
                    tool_call=ToolCall("explode", {}),
                )
            )
        ]
    )
    executor = AgentExecutor(model, ToolRegistry([FailingTool()]), max_steps=1)

    with pytest.raises(ToolExecutionError):
        executor.run("Run risky tool.", task_id="tool-failure-task")


def test_tool_authorizer_denies_disallowed_registered_tool() -> None:
    model = MockLLM(
        [
            ScriptedStep(
                AgentStep(
                    thought_summary="Try risky tool.",
                    tool_call=ToolCall("lookup_price", {"sku": "SKU-123"}),
                )
            )
        ]
    )
    registry = ToolRegistry(
        [PriceTool()],
        authorizer=AllowListedToolAuthorizer({"different_tool"}),
    )
    executor = AgentExecutor(model, registry, max_steps=1)

    with pytest.raises(ToolAuthorizationError):
        executor.run("Use denied tool.", task_id="auth-task")


def test_tool_authorizer_requires_delegated_scope() -> None:
    model = MockLLM(
        [
            ScriptedStep(
                AgentStep(
                    thought_summary="Need scoped tool.",
                    tool_call=ToolCall("lookup_price", {"sku": "SKU-123"}),
                )
            )
        ]
    )
    registry = ToolRegistry(
        [PriceTool()],
        authorizer=AllowListedToolAuthorizer(
            {"lookup_price"},
            required_scopes_by_tool={"lookup_price": {"catalog:read"}},
        ),
    )
    executor = AgentExecutor(model, registry, max_steps=1)

    with pytest.raises(ToolAuthorizationError):
        executor.run("Use scoped tool.", task_id="scope-task")


def test_executor_records_model_call_budget() -> None:
    model = MockLLM(
        [
            ScriptedStep(
                AgentStep(
                    thought_summary="Need the price.",
                    tool_call=ToolCall("lookup_price", {"sku": "SKU-123"}),
                )
            ),
            ScriptedStep(AgentStep("done", final_answer="done")),
        ]
    )
    budget = TaskBudget(max_model_calls=1, max_tokens=100)
    executor = AgentExecutor(
        model,
        ToolRegistry([PriceTool()]),
        max_steps=2,
        budget=budget,
    )

    with pytest.raises(BudgetExceededError):
        executor.run("Spend one model call then exceed.", task_id="budget-task")


def test_executor_passes_identity_context_to_tools() -> None:
    model = MockLLM(
        [
            ScriptedStep(
                AgentStep(
                    thought_summary="Need the price.",
                    tool_call=ToolCall("lookup_price", {"sku": "SKU-123"}),
                    input_tokens=3,
                    output_tokens=4,
                )
            ),
            ScriptedStep(AgentStep("done", final_answer="done", input_tokens=2)),
        ]
    )
    budget = TaskBudget(max_model_calls=2, max_tokens=10)
    executor = AgentExecutor(
        model,
        ToolRegistry([PriceTool()]),
        max_steps=2,
        budget=budget,
    )

    assert (
        executor.run(
            "price",
            task_id="identity-task",
            actor_id="user-1",
            tenant_id="tenant-1",
            delegated_scopes=("refund:read",),
            request_source="support_console",
        )
        == "done"
    )
    assert budget.tokens == 9


def test_executor_redacts_sensitive_tool_results_before_model_history() -> None:
    def assert_safe_history(history: list[HistoryEvent]) -> None:
        tool_results = [event for event in history if isinstance(event, ToolResult)]
        assert len(tool_results) == 1
        result = tool_results[0].result
        assert isinstance(result, dict)
        assert result["access_token"] == "[redacted]"
        assert result["customer_email"] == "[redacted]"
        assert result["customer_phone"] == "[redacted]"
        assert result["private_notes"] == "[redacted]"
        assert result["x_api_key"] == "[redacted]"
        assert result["safe_summary"] == "eligible for refund"
        assert result["nested"]["refreshToken"] == "[redacted]"
        assert result["nested"]["status"] == "active"
        assert "tok-secret-123" not in str(result)
        assert "customer@example.com" not in str(result)
        assert "sk-123456789012345" not in str(result)
        assert "555-123-4567" not in str(result)
        assert "private CRM note" not in str(result)
        assert "refresh-secret-456" not in str(result)
        assert tool_results[0].data_classification == "model_safe"
        assert tool_results[0].truncated is True

    model = MockLLM(
        [
            ScriptedStep(
                AgentStep(
                    thought_summary="Need customer lookup.",
                    tool_call=ToolCall("customer_lookup", {}),
                )
            ),
            ScriptedStep(
                AgentStep("done", final_answer="safe"),
                require_history=assert_safe_history,
            ),
        ]
    )
    executor = AgentExecutor(model, ToolRegistry([SensitiveTool()]), max_steps=2)

    assert executor.run("lookup", task_id="sensitive-task") == "safe"


def test_executor_bounds_large_tool_results_before_model_history() -> None:
    def assert_bounded_history(history: list[HistoryEvent]) -> None:
        tool_results = [event for event in history if isinstance(event, ToolResult)]
        assert len(tool_results) == 1
        result = tool_results[0].result
        assert isinstance(result, dict)
        assert isinstance(result["text"], str)
        assert len(result["text"]) < 300
        assert result["text"].endswith("...[truncated]")
        assert result["items"][-1] == "[truncated]"
        assert len(result["items"]) == 21
        assert tool_results[0].truncated is True

    model = MockLLM(
        [
            ScriptedStep(
                AgentStep(
                    thought_summary="Need large payload.",
                    tool_call=ToolCall("large_payload", {}),
                )
            ),
            ScriptedStep(
                AgentStep("done", final_answer="safe"),
                require_history=assert_bounded_history,
            ),
        ]
    )
    executor = AgentExecutor(model, ToolRegistry([LargeTool()]), max_steps=2)

    assert executor.run("lookup", task_id="large-task") == "safe"
