from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .agent_loop import AgentStep, HistoryEvent


HistoryCheck = Callable[[list[HistoryEvent]], None]


@dataclass(frozen=True)
class ScriptedStep:
    step: AgentStep
    require_history: HistoryCheck | None = None


class MockLLM:
    def __init__(self, steps: list[ScriptedStep]) -> None:
        self.steps = steps
        self.index = 0

    def next_step(self, prompt: str, history: list[HistoryEvent]) -> AgentStep:
        del prompt
        if self.index >= len(self.steps):
            raise AssertionError("script exhausted")
        scripted = self.steps[self.index]
        self.index += 1
        if scripted.require_history is not None:
            scripted.require_history(history)
        return scripted.step
