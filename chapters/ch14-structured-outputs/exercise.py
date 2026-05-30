from __future__ import annotations

from agentic_interview_bible.structured_outputs import RefundDecision


def parse_refund_decision(raw: str) -> RefundDecision:
    """Parse and validate a raw JSON string against the RefundDecision schema.

    Requirements:
    - decision must be one of approve, deny, or escalate
    - amount_cents must be an int (or null/None), not a bool, and non-negative
    - reason_code must be a non-empty string
    - confidence must be a float in [0, 1], not a bool
    - Extra fields are rejected
    - Raises ParseError on malformed JSON
    - Raises SchemaError on valid JSON that violates the contract
    """
    raise NotImplementedError
