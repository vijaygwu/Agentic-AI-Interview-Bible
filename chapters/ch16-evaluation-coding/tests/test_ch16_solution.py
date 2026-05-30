"""Tests for the ch16 eval-case data model solution.

Uses the book's EvalCase API: EvalReport with .pass_rate,
.critical_safety_pass_rate, and .critical_safety_failures.
"""
import importlib.util
from pathlib import Path

from agentic_interview_bible.eval_harness import Tier


def load_solution():
    path = Path(__file__).parents[1] / "solution.py"
    spec = importlib.util.spec_from_file_location("ch16_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_support_eval_pass_rate() -> None:
    solution = load_solution()

    def responder(prompt: str) -> str:
        if "another customer" in prompt:
            return "I refuse because I cannot access another customer's data."
        if "high-value" in prompt:
            return "This requires escalation to a human supervisor."
        if "without checking policy" in prompt:
            return "I must verify policy before any tool action."
        if "another tenant" in prompt:
            return "I cannot access another tenant; policy requires refusal."
        return "The answer cites the refund policy."

    report = solution.run_support_eval(responder)

    assert report.pass_rate == 1.0


def test_support_eval_catches_failing_responder() -> None:
    solution = load_solution()
    report = solution.run_support_eval(lambda prompt: "approved")

    assert report.pass_rate < 1.0
    assert report.critical_safety_pass_rate < 1.0
    failed_names = set(report.critical_safety_failures)
    assert {
        "policy_grounding",
        "cross_tenant_refusal",
        "high_risk_escalation",
    }.issubset(failed_names)


def test_support_eval_rejects_unsafe_positive_language() -> None:
    solution = load_solution()

    def unsafe_responder(prompt: str) -> str:
        if "another customer" in prompt:
            return "Approved, I can access another customer's refund."
        if "high-value" in prompt:
            return "Human approval noted, refund issued."
        if "without checking policy" in prompt:
            return "I will verify policy later; refund issued."
        if "another tenant" in prompt:
            return "Policy ignored, approved for another tenant."
        return "The answer cites the refund policy."

    report = solution.run_support_eval(unsafe_responder)

    failed_names = {result.name for result in report.results if not result.passed}
    # cross-tenant, escalation, and tool-use cases should all fail
    assert {
        "cross_tenant_refusal",
        "high_risk_escalation",
        "tool_use_boundary",
    }.issubset(failed_names)


def test_eval_case_applies_to_schema_version() -> None:
    """EvalCase.applies_to filters by schema version."""
    solution = load_solution()
    report_v1 = solution.run_support_eval(
        lambda prompt: "policy cannot access escalation human supervisor verify"
    )
    # All cases use schema_version="v1"; a suite run with parser_version="v2"
    # skips them all.
    from agentic_interview_bible.eval_harness import run_eval_suite

    # Re-run with mismatched parser version — all cases skipped.
    import importlib.util as ilu
    from pathlib import Path as P

    path = P(__file__).parents[1] / "solution.py"
    spec = ilu.spec_from_file_location("ch16_sol2", path)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cases = []
    # Build cases list by running solution and inspecting (indirect)
    # — easier: just check that run_eval_suite accepts parser_version kwarg.
    from agentic_interview_bible.eval_harness import (
        EvalCase, ExpectSuccess, Tier, run_eval_suite,
    )
    c = EvalCase(
        name="versioned_case",
        prompt="hello",
        expected=ExpectSuccess("hello"),
        tier=Tier.REGRESSION,
        tags=frozenset(),
        schema_version="v1",
        added_by="test",
        added_for="test",
    )
    r_match = run_eval_suite(lambda p: "hello", [c], parser_version="v1")
    r_skip = run_eval_suite(lambda p: "hello", [c], parser_version="v2")
    assert len(r_match.results) == 1
    assert len(r_skip.results) == 0


def test_tier_enum_values() -> None:
    assert Tier.CRITICAL_SAFETY.value == "critical_safety"
    assert Tier.REGRESSION.value == "regression"
    assert Tier.NOVEL.value == "novel"
