import importlib.util
from pathlib import Path

import pytest

from agentic_interview_bible import StructuredOutputError


def load_solution():
    path = Path(__file__).parents[1] / "solution.py"
    spec = importlib.util.spec_from_file_location("ch14_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_refund_decision() -> None:
    solution = load_solution()
    decision = solution.parse_refund_decision(
        {
            "decision": "deny",
            "amount_cents": 0,
            "policy_version": "refunds-2026-05",
            "requires_human_approval": False,
            "evidence_ids": ["policy-1"],
        }
    )

    assert decision["decision"] == "deny"


def test_parse_refund_decision_rejects_missing_required_field() -> None:
    solution = load_solution()

    with pytest.raises(StructuredOutputError):
        solution.parse_refund_decision(
            {
                "decision": "deny",
                "amount_cents": 0,
                "policy_version": "refunds-2026-05",
                "requires_human_approval": False,
            }
        )


def test_parse_refund_decision_rejects_bad_type() -> None:
    solution = load_solution()

    with pytest.raises(StructuredOutputError):
        solution.parse_refund_decision(
            {
                "decision": "approve",
                "amount_cents": "0",
                "policy_version": "refunds-2026-05",
                "requires_human_approval": False,
                "evidence_ids": ["policy-1"],
            }
        )


def test_parse_refund_decision_rejects_invalid_decision_value() -> None:
    solution = load_solution()

    with pytest.raises(StructuredOutputError):
        solution.parse_refund_decision(
            {
                "decision": "maybe",
                "amount_cents": 0,
                "policy_version": "refunds-2026-05",
                "requires_human_approval": False,
                "evidence_ids": ["policy-1"],
            }
        )


def test_parse_refund_decision_rejects_negative_amount_bool_amount_and_empty_evidence() -> None:
    solution = load_solution()
    base = {
        "decision": "approve",
        "amount_cents": 0,
        "policy_version": "refunds-2026-05",
        "requires_human_approval": False,
        "evidence_ids": ["policy-1"],
    }

    with pytest.raises(StructuredOutputError):
        solution.parse_refund_decision({**base, "amount_cents": -1})
    with pytest.raises(StructuredOutputError):
        solution.parse_refund_decision({**base, "amount_cents": True})
    with pytest.raises(StructuredOutputError):
        solution.parse_refund_decision({**base, "evidence_ids": []})
