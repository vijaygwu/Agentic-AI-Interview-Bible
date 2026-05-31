from __future__ import annotations

import json

from agentic_interview_bible.structured_outputs import (
    ParseError,
    RefundDecision,
    SchemaError,
)

# The package ships ``parse_strict`` already; the interview point is to show
# the validation explicitly, so the schema contract is visible at the call
# site rather than hidden behind a helper. The logic below is exactly what a
# strong candidate would write at the whiteboard.

_VALID_DECISIONS = ("approve", "deny", "escalate")
_REQUIRED_FIELDS = ("decision", "amount_cents", "reason_code", "confidence")


def parse_refund_decision(raw: str) -> RefundDecision:
    """Parse and validate a raw JSON string against the RefundDecision schema.

    Raises ParseError on malformed JSON and SchemaError on valid JSON that
    violates the contract.
    """
    # 1. Parse. A decode failure is a ParseError, not a SchemaError: the two
    #    are distinct so callers can decide whether a repair retry is worth
    #    the budget.
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise SchemaError(f"expected JSON object, got {type(obj).__name__}")

    # 2. Shape: every required field present, no unexpected fields. Rejecting
    #    extras blocks a model from smuggling, say, an authorization flag.
    missing = [k for k in _REQUIRED_FIELDS if k not in obj]
    if missing:
        raise SchemaError(f"missing required fields: {missing}")
    extra = set(obj) - set(_REQUIRED_FIELDS)
    if extra:
        raise SchemaError(f"unexpected fields: {sorted(extra)}")

    # 3. Enum: decision is a closed set.
    decision = obj["decision"]
    if not isinstance(decision, str):
        raise SchemaError("decision must be a string")
    if decision not in _VALID_DECISIONS:
        raise SchemaError(
            f"decision must be one of {_VALID_DECISIONS}, got {decision!r}"
        )

    # 4. Money: int or null, never a bool (``True`` is an int in Python), and
    #    non-negative. The bool guard is the detail interviewers look for.
    amount = obj["amount_cents"]
    if amount is not None:
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise SchemaError("amount_cents must be int or null")
        if amount < 0:
            raise SchemaError("amount_cents must be non-negative")

    reason = obj["reason_code"]
    if not isinstance(reason, str) or not reason:
        raise SchemaError("reason_code must be a non-empty string")

    # 5. Confidence: a real number in [0, 1], again rejecting bool.
    confidence = obj["confidence"]
    if isinstance(confidence, bool):
        raise SchemaError("confidence must not be bool")
    if not isinstance(confidence, (int, float)):
        raise SchemaError("confidence must be a number")
    if not 0.0 <= float(confidence) <= 1.0:
        raise SchemaError(f"confidence must be in [0, 1], got {confidence}")

    return RefundDecision(
        decision=decision,
        amount_cents=amount,
        reason_code=reason,
        confidence=float(confidence),
    )
