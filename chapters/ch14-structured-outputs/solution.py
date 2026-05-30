from __future__ import annotations

from agentic_interview_bible.structured_outputs import (
    RefundDecision,
    parse_strict,
)


def parse_refund_decision(raw: str) -> RefundDecision:
    """Parse and validate a raw JSON string against the RefundDecision schema."""
    return parse_strict(raw)
