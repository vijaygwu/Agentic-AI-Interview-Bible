from __future__ import annotations

from dataclasses import dataclass


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
        blocked_keys = {"raw_prompt", "chain_of_thought", "secret", "api_key"}
        if blocked_keys.intersection(event.metadata):
            raise ValueError("trace metadata contains unsafe fields")
        self.events.append(event)

    def spans_for_task(self, task_id: str) -> list[str]:
        return [event.span for event in self.events if event.task_id == task_id]
