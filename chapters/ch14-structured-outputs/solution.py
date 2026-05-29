from __future__ import annotations

from agentic_interview_bible import validate_refund_decision


def parse_refund_decision(payload: dict[str, object]) -> dict[str, object]:
    return validate_refund_decision(payload)
