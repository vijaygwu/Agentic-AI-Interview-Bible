import importlib.util
from pathlib import Path


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
            return "This requires human approval."
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
    assert {result.name for result in report.critical_failures} >= {
        "policy_grounding",
        "cross_tenant_refusal",
        "high_risk_escalation",
        "regression_prompt_injection_refund",
    }


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
    assert {
        "cross_tenant_refusal",
        "high_risk_escalation",
        "tool_use_boundary",
        "regression_prompt_injection_refund",
    }.issubset(failed_names)
