from __future__ import annotations

from agentic_interview_bible import ReleaseDecision, decide_release


REQUIRED_CATEGORIES = {
    "grounding",
    "safety",
    "escalation",
    "tool_use",
    "regression",
}

REQUIRED_CRITICAL_CATEGORIES = {
    "safety",
    "escalation",
    "tool_use",
    "regression",
}


def gate_prompt_release(report, minimum_pass_rate: float = 1.0):
    if not 0 <= minimum_pass_rate <= 1:
        raise ValueError("minimum_pass_rate must be between 0 and 1")
    categories = {result.category for result in report.results}
    missing = sorted(REQUIRED_CATEGORIES - categories)
    if missing:
        return ReleaseDecision(False, f"missing required eval categories: {', '.join(missing)}")
    critical_categories = {result.category for result in report.results if result.critical}
    missing_critical = sorted(REQUIRED_CRITICAL_CATEGORIES - critical_categories)
    if missing_critical:
        return ReleaseDecision(
            False,
            f"missing critical eval coverage: {', '.join(missing_critical)}",
        )
    return decide_release(report, minimum_pass_rate)
