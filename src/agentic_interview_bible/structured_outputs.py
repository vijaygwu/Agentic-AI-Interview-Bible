from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class StructuredOutputError(ValueError):
    """Raised when model output does not match the expected schema."""


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
                    raise StructuredOutputError(f"missing required field: {field.name}")
                continue
            value = payload[field.name]
            if type(value) is not field.expected_type:
                expected = field.expected_type.__name__
                actual = type(value).__name__
                raise StructuredOutputError(
                    f"field {field.name} expected {expected}, got {actual}"
                )
            if field.validator is not None and not field.validator(value):
                raise StructuredOutputError(f"field {field.name} failed validation")
            validated[field.name] = value
        return validated


def validate_refund_decision(payload: dict[str, Any]) -> dict[str, Any]:
    validator = StructuredOutputValidator(
        [
            FieldSpec("decision", str),
            FieldSpec("amount_cents", int, validator=lambda value: value >= 0),
            FieldSpec("policy_version", str),
            FieldSpec("requires_human_approval", bool),
            FieldSpec(
                "evidence_ids",
                list,
                validator=lambda value: bool(value)
                and all(type(item) is str for item in value),
            ),
        ]
    )
    validated = validator.validate(payload)
    if validated["decision"] not in {"approve", "deny", "escalate"}:
        raise StructuredOutputError("decision must be approve, deny, or escalate")
    return validated
