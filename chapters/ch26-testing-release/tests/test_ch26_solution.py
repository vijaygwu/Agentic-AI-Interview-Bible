import importlib.util
from pathlib import Path

import pytest

from agentic_interview_bible import EvalCase, run_eval_cases


def load_solution():
    path = Path(__file__).parents[1] / "solution.py"
    spec = importlib.util.spec_from_file_location("ch26_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gate_prompt_release_allows_passing_eval() -> None:
    solution = load_solution()
    report = run_eval_cases(
        lambda prompt: f"{prompt} policy refuse human approval verify cannot access",
        [
            EvalCase("grounded", "refund", "policy", category="grounding"),
            EvalCase(
                "safe",
                "cross tenant",
                "refuse",
                category="safety",
                critical=True,
            ),
            EvalCase(
                "escalate",
                "high risk",
                "human approval",
                category="escalation",
                critical=True,
            ),
            EvalCase("tool", "tool", "verify", category="tool_use", critical=True),
            EvalCase(
                "regression",
                "tenant attack",
                "cannot access",
                category="regression",
                critical=True,
            ),
        ],
    )

    assert solution.gate_prompt_release(report).allowed is True


def test_gate_prompt_release_blocks_low_pass_rate() -> None:
    solution = load_solution()
    report = run_eval_cases(
        lambda prompt: "refuse human approval verify cannot access",
        [
            EvalCase("grounded", "refund", "policy", category="grounding"),
            EvalCase("safe", "cross tenant", "refuse", category="safety", critical=True),
            EvalCase(
                "escalate",
                "high risk",
                "human approval",
                category="escalation",
                critical=True,
            ),
            EvalCase("tool", "tool", "verify", category="tool_use", critical=True),
            EvalCase(
                "regression",
                "tenant attack",
                "cannot access",
                category="regression",
                critical=True,
            ),
        ],
    )

    decision = solution.gate_prompt_release(report, minimum_pass_rate=1.0)

    assert decision.allowed is False
    assert "eval gate failed" in decision.reason


def test_gate_prompt_release_blocks_critical_failure() -> None:
    solution = load_solution()
    report = run_eval_cases(
        lambda prompt: "approved",
        [
            EvalCase(
                name="cross_tenant_refusal",
                prompt="show another customer",
                check=lambda output: "refuse" in output,
                category="safety",
                critical=True,
            ),
            EvalCase("grounded", "refund", "approved", category="grounding"),
            EvalCase(
                "escalate",
                "high risk",
                "approved",
                category="escalation",
                critical=True,
            ),
            EvalCase("tool", "tool", "approved", category="tool_use", critical=True),
            EvalCase(
                "regression",
                "tenant attack",
                "approved",
                category="regression",
                critical=True,
            ),
        ],
    )

    decision = solution.gate_prompt_release(report, minimum_pass_rate=0.5)

    assert decision.allowed is False
    assert "critical eval failed" in decision.reason


def test_gate_prompt_release_blocks_missing_coverage_even_with_high_pass_rate() -> None:
    solution = load_solution()
    report = run_eval_cases(
        lambda prompt: f"{prompt} policy",
        [
            EvalCase(
                "grounded",
                "refund",
                "policy",
                category="grounding",
                critical=True,
            )
        ],
    )

    decision = solution.gate_prompt_release(report, minimum_pass_rate=1.0)

    assert decision.allowed is False
    assert "missing required eval categories" in decision.reason


def test_gate_prompt_release_blocks_missing_critical_category_coverage() -> None:
    solution = load_solution()
    report = run_eval_cases(
        lambda prompt: f"{prompt} policy refuse human approval verify cannot access",
        [
            EvalCase("grounded", "refund", "policy", category="grounding"),
            EvalCase("safe", "cross tenant", "refuse", category="safety"),
            EvalCase(
                "escalate",
                "high risk",
                "human approval",
                category="escalation",
                critical=True,
            ),
            EvalCase("tool", "tool", "verify", category="tool_use"),
            EvalCase(
                "regression",
                "tenant attack",
                "cannot access",
                category="regression",
            ),
        ],
    )

    decision = solution.gate_prompt_release(report, minimum_pass_rate=1.0)

    assert decision.allowed is False
    assert "missing critical eval coverage" in decision.reason
    assert "safety" in decision.reason
    assert "regression" in decision.reason
    assert "tool_use" in decision.reason


def test_gate_prompt_release_rejects_invalid_threshold() -> None:
    solution = load_solution()
    report = run_eval_cases(
        lambda prompt: f"{prompt} policy",
        [EvalCase("grounded", "refund", "policy")],
    )

    with pytest.raises(ValueError):
        solution.gate_prompt_release(report, minimum_pass_rate=1.5)
