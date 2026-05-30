from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Exception hierarchy  (book API)
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raw output is not valid JSON."""


class SchemaError(Exception):
    """Output is valid JSON but does not match the schema."""


class RepairExhausted(Exception):
    """Repair attempts exhausted; final state still invalid."""


# ---------------------------------------------------------------------------
# Schema  (book API)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefundDecision:
    decision: str          # one of: approve, deny, escalate
    amount_cents: int | None
    reason_code: str
    confidence: float


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = ("decision", "amount_cents", "reason_code", "confidence")
_VALID_DECISIONS = ("approve", "deny", "escalate")


# ---------------------------------------------------------------------------
# Primitives  (book API)
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    """Raise ParseError on malformed JSON; raise SchemaError if not an object."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise SchemaError(
            f"expected JSON object, got {type(obj).__name__}"
        )
    return obj


def _validate(obj: dict) -> RefundDecision:
    """Raise SchemaError on missing fields, extra fields, wrong types,
    invalid enum, out-of-range numeric, or bool-as-numeric."""
    missing = [k for k in _REQUIRED_FIELDS if k not in obj]
    if missing:
        raise SchemaError(f"missing required fields: {missing}")

    extra = set(obj) - set(_REQUIRED_FIELDS)
    if extra:
        raise SchemaError(f"unexpected fields: {sorted(extra)}")

    decision = obj["decision"]
    if not isinstance(decision, str):
        raise SchemaError("decision must be a string")
    if decision not in _VALID_DECISIONS:
        raise SchemaError(
            f"decision must be one of {_VALID_DECISIONS}, got {decision!r}"
        )

    amount = obj["amount_cents"]
    if amount is not None:
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise SchemaError("amount_cents must be int or null")
        if amount < 0:
            raise SchemaError("amount_cents must be non-negative")

    reason = obj["reason_code"]
    if not isinstance(reason, str):
        raise SchemaError("reason_code must be a string")

    confidence = obj["confidence"]
    if isinstance(confidence, bool):
        raise SchemaError("confidence must not be bool")
    if not isinstance(confidence, (int, float)):
        raise SchemaError("confidence must be a number")
    if not 0.0 <= float(confidence) <= 1.0:
        raise SchemaError(
            f"confidence must be in [0, 1], got {confidence}"
        )

    return RefundDecision(
        decision=decision,
        amount_cents=amount,
        reason_code=reason,
        confidence=float(confidence),
    )


# ---------------------------------------------------------------------------
# Public API  (book API)
# ---------------------------------------------------------------------------

def parse_strict(raw: str) -> RefundDecision:
    """Parse without repair. Raises ParseError or SchemaError on any failure."""
    return _validate(_parse_json(raw))


RepairFn = Callable[[str, Exception], str]


def parse_with_repair(
    raw: str,
    repair: RepairFn,
    *,
    max_repairs: int = 1,
) -> RefundDecision:
    """Parse with a bounded repair budget.

    The budget check happens *before* calling repair, so max_repairs=0 fails
    closed without invoking the repair function.  Raises RepairExhausted when
    the budget is exceeded.
    """
    if max_repairs < 0:
        raise ValueError("max_repairs must be non-negative")

    current = raw
    attempts_remaining = max_repairs

    while True:
        try:
            return parse_strict(current)
        except (ParseError, SchemaError) as exc:
            if attempts_remaining <= 0:
                raise RepairExhausted(
                    f"failed after {max_repairs} repair attempt(s): {exc}"
                ) from exc
            attempts_remaining -= 1
            current = repair(current, exc)


# ---------------------------------------------------------------------------
# Legacy API  (kept for __init__.py backward compatibility)
# ---------------------------------------------------------------------------

# StructuredOutputError is the legacy single-exception class; map it to
# SchemaError so existing callers still work via "from agentic_interview_bible
# import StructuredOutputError".
StructuredOutputError = SchemaError


@dataclass(frozen=True)
class FieldSpec:
    name: str
    expected_type: type
    required: bool = True
    validator: Callable[[Any], bool] | None = None


class StructuredOutputValidator:
    def __init__(self, fields: list[FieldSpec]) -> None:
        self.fields = fields

    def validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise StructuredOutputError("payload must be an object")

        validated: dict[str, Any] = {}
        for field in self.fields:
            if field.name not in payload:
                if field.required:
                    raise StructuredOutputError(
                        f"missing required field: {field.name}"
                    )
                continue
            value = payload[field.name]
            # Bool guard: isinstance(True, int) is True in Python.
            if isinstance(value, bool) and field.expected_type is not bool:
                raise StructuredOutputError(
                    f"field {field.name} expected "
                    f"{field.expected_type.__name__}, got bool"
                )
            if not isinstance(value, field.expected_type):
                expected = field.expected_type.__name__
                actual = type(value).__name__
                raise StructuredOutputError(
                    f"field {field.name} expected {expected}, got {actual}"
                )
            if field.validator is not None and not field.validator(value):
                raise StructuredOutputError(
                    f"field {field.name} failed validation"
                )
            validated[field.name] = value
        return validated


def validate_refund_decision(payload: dict[str, Any]) -> dict[str, Any]:
    """Legacy validator that accepts a dict and returns a validated dict.

    Used by the existing ch14 solution/tests that pre-date the book API.
    Schema: decision, amount_cents, policy_version, requires_human_approval,
    evidence_ids.
    """
    validator = StructuredOutputValidator(
        [
            FieldSpec("decision", str),
            FieldSpec("amount_cents", int, validator=lambda v: v >= 0),
            FieldSpec("policy_version", str),
            FieldSpec("requires_human_approval", bool),
            FieldSpec(
                "evidence_ids",
                list,
                validator=lambda v: bool(v)
                and all(type(item) is str for item in v),
            ),
        ]
    )
    validated = validator.validate(payload)
    if validated["decision"] not in {"approve", "deny", "escalate"}:
        raise StructuredOutputError(
            "decision must be approve, deny, or escalate"
        )
    return validated
