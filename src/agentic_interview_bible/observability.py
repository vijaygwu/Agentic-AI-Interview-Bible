from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Book API: SpanType, SpanStatus, Span, Trace, SAFE_ATTRS
# (from problem-trace-schema.tex, Strong Attempt)
# ---------------------------------------------------------------------------

class SpanType(Enum):
    TASK = "task"
    MODEL_CALL = "model_call"
    RETRIEVAL = "retrieval"
    POLICY_CHECK = "policy_check"
    TOOL_CALL = "tool_call"
    FINAL_RESPONSE = "final_response"


class SpanStatus(Enum):
    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Span:
    span_id: str
    parent_span_id: str | None
    trace_id: str
    type: SpanType
    start_ms: int
    end_ms: int
    status: SpanStatus
    attributes: dict[str, Any]
    error_class: str | None = None


@dataclass(frozen=True)
class Trace:
    trace_id: str
    task_id: str
    tenant_id: str
    user_id_hash: str
    agent_id: str
    agent_version: str
    prompt_version: str
    model_alias: str
    spans: list[Span]
    outcome: str  # "answered", "escalated", "refunded", "denied", "failed"


# Recommended attribute names (redacted): raw model inputs/outputs are not
# stored as text; source IDs and hashes are.
SAFE_ATTRS: frozenset[str] = frozenset({
    "prompt_token_count",
    "completion_token_count",
    "source_ids",          # IDs only, not text
    "policy_decision_id",
    "tool_name",
    "args_hash",           # hash, not raw
    "auth_decision",       # "allowed" | "denied" | reason code
    "idempotency_key",
    "retry_attempt",
    "cost_cents",
})


# ---------------------------------------------------------------------------
# Sensitive-metadata protection (used by InMemoryTraceSink)
# ---------------------------------------------------------------------------

# Curated secret markers.  Matched as substrings of the normalized key so that
# variants (openai_api_key, user_password, bearer_token, secret_key) are
# caught, while benign infrastructure keys that merely contain "key"/"token"
# (e.g. idempotency_key, completion_tokens) are NOT false-positived.
_SENSITIVE_METADATA_MARKERS = (
    "raw_prompt",
    "chain_of_thought",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "access_token",
    "bearer",
    "private_key",
    "credential",
    "ssn",
)


def _is_sensitive_metadata_key(key: object) -> bool:
    normalized = "".join(
        character.casefold() if character.isalnum() else "_"
        for character in str(key)
    )
    return any(marker in normalized for marker in _SENSITIVE_METADATA_MARKERS)


# ---------------------------------------------------------------------------
# TraceEvent / InMemoryTraceSink (lightweight sink used by exercises)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TraceEvent:
    task_id: str
    span: str
    safe_summary: str
    metadata: dict[str, object]


class InMemoryTraceSink:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(self, event: TraceEvent) -> None:
        if any(_is_sensitive_metadata_key(key) for key in event.metadata):
            raise ValueError("trace metadata contains unsafe fields")
        self.events.append(event)

    def spans_for_task(self, task_id: str) -> list[str]:
        return [
            event.span for event in self.events if event.task_id == task_id
        ]
