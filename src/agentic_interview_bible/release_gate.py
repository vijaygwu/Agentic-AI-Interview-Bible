from __future__ import annotations

from dataclasses import dataclass

from .eval_harness import EvalReport


@dataclass(frozen=True)
class ReleaseDecision:
    allowed: bool
    reason: str


def decide_release(report: EvalReport, minimum_pass_rate: float) -> ReleaseDecision:
    if not 0 <= minimum_pass_rate <= 1:
        raise ValueError("minimum_pass_rate must be between 0 and 1")
    if report.critical_failures:
        names = ", ".join(result.name for result in report.critical_failures)
        return ReleaseDecision(False, f"critical eval failed: {names}")
    if report.pass_rate >= minimum_pass_rate:
        return ReleaseDecision(True, "eval gate passed")
    return ReleaseDecision(False, "eval gate failed")
