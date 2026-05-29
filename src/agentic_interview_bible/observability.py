from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceEvent:
    task_id: str
    span: str
    safe_summary: str
    metadata: dict[str, object]


# Curated secret markers. Matched as substrings of the normalized key so that
# variants (openai_api_key, user_password, bearer_token, secret_key) are caught,
# while benign infrastructure keys that merely contain "key"/"token" (e.g.
# idempotency_key, completion_tokens) are NOT false-positived.
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
        character.casefold() if character.isalnum() else "_" for character in str(key)
    )
    return any(marker in normalized for marker in _SENSITIVE_METADATA_MARKERS)


class InMemoryTraceSink:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(self, event: TraceEvent) -> None:
        if any(_is_sensitive_metadata_key(key) for key in event.metadata):
            raise ValueError("trace metadata contains unsafe fields")
        self.events.append(event)

    def spans_for_task(self, task_id: str) -> list[str]:
        return [event.span for event in self.events if event.task_id == task_id]
