import importlib.util
from pathlib import Path

import pytest

from agentic_interview_bible.structured_outputs import (
    ParseError,
    RefundDecision,
    RepairExhausted,
    SchemaError,
    parse_strict,
    parse_with_repair,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_JSON = (
    '{"decision": "approve", "amount_cents": 5000, '
    '"reason_code": "policy_match", "confidence": 0.92}'
)


def load_solution():
    path = Path(__file__).parents[1] / "solution.py"
    spec = importlib.util.spec_from_file_location("ch14_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Test 1: Happy path
# ---------------------------------------------------------------------------

def test_happy_path() -> None:
    solution = load_solution()
    result = solution.parse_refund_decision(VALID_JSON)
    assert isinstance(result, RefundDecision)
    assert result.decision == "approve"
    assert result.amount_cents == 5000
    assert result.reason_code == "policy_match"
    assert result.confidence == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# Test 2: Malformed JSON raises ParseError
# ---------------------------------------------------------------------------

def test_malformed_json_raises_parse_error() -> None:
    solution = load_solution()
    with pytest.raises(ParseError):
        solution.parse_refund_decision('{"decision": "approve"')


# ---------------------------------------------------------------------------
# Test 3: Missing required field raises SchemaError
# ---------------------------------------------------------------------------

def test_missing_field_raises_schema_error() -> None:
    solution = load_solution()
    with pytest.raises(SchemaError):
        solution.parse_refund_decision('{"decision": "approve"}')


# ---------------------------------------------------------------------------
# Test 4: Invalid enum value raises SchemaError
# ---------------------------------------------------------------------------

def test_invalid_decision_value_raises_schema_error() -> None:
    solution = load_solution()
    bad = (
        '{"decision": "maybe", "amount_cents": null, '
        '"reason_code": "x", "confidence": 0.5}'
    )
    with pytest.raises(SchemaError):
        solution.parse_refund_decision(bad)


# ---------------------------------------------------------------------------
# Test 5: Extra fields rejected
# ---------------------------------------------------------------------------

def test_extra_fields_rejected() -> None:
    solution = load_solution()
    bad = (
        '{"decision": "approve", "amount_cents": 5000, '
        '"reason_code": "x", "confidence": 0.5, '
        '"extra_authorization": "admin"}'
    )
    with pytest.raises(SchemaError):
        solution.parse_refund_decision(bad)


# ---------------------------------------------------------------------------
# Test 6: Bool not accepted as int
# ---------------------------------------------------------------------------

def test_bool_not_accepted_as_int() -> None:
    solution = load_solution()
    bad = (
        '{"decision": "approve", "amount_cents": true, '
        '"reason_code": "x", "confidence": 0.5}'
    )
    with pytest.raises(SchemaError):
        solution.parse_refund_decision(bad)


# ---------------------------------------------------------------------------
# Test 7: Numeric out of range raises SchemaError
# ---------------------------------------------------------------------------

def test_confidence_out_of_range() -> None:
    solution = load_solution()
    bad = (
        '{"decision": "approve", "amount_cents": 0, '
        '"reason_code": "x", "confidence": 1.5}'
    )
    with pytest.raises(SchemaError):
        solution.parse_refund_decision(bad)


# ---------------------------------------------------------------------------
# Test 8: Repair success
# ---------------------------------------------------------------------------

def test_repair_success() -> None:
    def repair(raw: str, err: Exception) -> str:
        return raw + "}"  # adds the missing closing brace

    bad = (
        '{"decision": "approve", "amount_cents": 5000, '
        '"reason_code": "x", "confidence": 0.5'
    )
    result = parse_with_repair(bad, repair, max_repairs=1)
    assert result.decision == "approve"


# ---------------------------------------------------------------------------
# Test 9: Repair failure raises RepairExhausted
# ---------------------------------------------------------------------------

def test_repair_failure_raises_repair_exhausted() -> None:
    def bad_repair(raw: str, err: Exception) -> str:
        return raw  # returns the same broken input

    with pytest.raises(RepairExhausted):
        parse_with_repair("not json", bad_repair, max_repairs=1)


# ---------------------------------------------------------------------------
# Test 10: max_repairs=0 fails closed without calling repair
# ---------------------------------------------------------------------------

def test_no_repair_budget_fails_closed() -> None:
    calls: list[int] = []

    def repair(raw: str, err: Exception) -> str:
        calls.append(1)
        return raw + "}"  # would have worked

    bad = (
        '{"decision": "approve", "amount_cents": 5000, '
        '"reason_code": "x", "confidence": 0.5'
    )
    with pytest.raises(RepairExhausted):
        parse_with_repair(bad, repair, max_repairs=0)

    assert calls == [], "repair must not be called when max_repairs=0"
