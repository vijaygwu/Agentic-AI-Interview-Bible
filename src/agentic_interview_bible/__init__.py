"""Companion code for The AI Agent Engineer Interview Bible."""

from .agent_loop import (
    AgentExecutor,
    AgentStep,
    AllowListedToolAuthorizer,
    AllowAllToolAuthorizer,
    DuplicateToolError,
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
from .circuit_breaker import BreakerState, CircuitBreaker, CircuitOpenError
from .cost_budget import BudgetExceededError, TaskBudget
from .eval_harness import EvalCase, EvalReport, EvalResult, redact_output, run_eval_cases
from .mock_llm import MockLLM, ScriptedStep
from .observability import InMemoryTraceSink, TraceEvent
from .rag_cache import CachePolicyError, Evidence, EvidenceCache, PermissionContext
from .release_gate import ReleaseDecision, decide_release
from .retry_budget import RetryBudget, RetryExhaustedError
from .structured_outputs import (
    FieldSpec,
    StructuredOutputError,
    StructuredOutputValidator,
    validate_refund_decision,
)

__all__ = [
    "AgentExecutor",
    "AgentStep",
    "AllowAllToolAuthorizer",
    "AllowListedToolAuthorizer",
    "BreakerState",
    "BudgetExceededError",
    "CachePolicyError",
    "CircuitBreaker",
    "CircuitOpenError",
    "DuplicateToolError",
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "Evidence",
    "EvidenceCache",
    "FieldSpec",
    "InMemoryTraceSink",
    "InvalidAgentStepError",
    "MaxStepsExceededError",
    "MockLLM",
    "PermissionContext",
    "ReleaseDecision",
    "RetryBudget",
    "RetryExhaustedError",
    "ScriptedStep",
    "StructuredOutputError",
    "StructuredOutputValidator",
    "TaskBudget",
    "ToolAuthorizationError",
    "ToolCall",
    "ToolExecutionContext",
    "ToolExecutionError",
    "ToolRegistry",
    "ToolResult",
    "TraceEvent",
    "UnknownToolError",
    "__version__",
    "decide_release",
    "redact_output",
    "run_eval_cases",
    "validate_refund_decision",
]

__version__ = "0.1.0"
