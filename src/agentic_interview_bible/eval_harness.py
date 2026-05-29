from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


EvalCheck = Callable[[str], bool]
OutputRedactor = Callable[[str], str]


@dataclass(frozen=True)
class EvalCase:
    name: str
    prompt: str
    expected_substring: str | None = None
    check: EvalCheck | None = None
    category: str = "behavior"
    critical: bool = False
    redactor: OutputRedactor | None = None

    def evaluate(self, output: str) -> bool:
        if self.check is not None:
            return self.check(output)
        if self.expected_substring is None:
            raise ValueError("EvalCase needs expected_substring or check")
        return self.expected_substring.casefold() in output.casefold()


@dataclass(frozen=True)
class EvalResult:
    name: str
    passed: bool
    category: str
    output: str
    critical: bool = False


@dataclass(frozen=True)
class EvalReport:
    results: list[EvalResult]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        passed = sum(result.passed for result in self.results)
        return passed / len(self.results)

    @property
    def critical_failures(self) -> list[EvalResult]:
        return [result for result in self.results if result.critical and not result.passed]


def run_eval_cases(responder: Callable[[str], str], cases: list[EvalCase]) -> EvalReport:
    results: list[EvalResult] = []
    for case in cases:
        try:
            output = responder(case.prompt)
            passed = case.evaluate(output)
            safe_output = (case.redactor or redact_output)(output)
        except Exception as exc:  # noqa: BLE001 - eval responders are user code.
            passed = False
            safe_output = f"[error:{type(exc).__name__}]"
        results.append(
            EvalResult(
                name=case.name,
                passed=passed,
                category=case.category,
                output=safe_output,
                critical=case.critical,
            )
        )
    return EvalReport(results)


def redact_output(output: str) -> str:
    redacted = output[:500]
    patterns = [
        r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[A-Za-z0-9._/\-]+",
        r"sk-[A-Za-z0-9]{10,}",
        r"(?i)bearer\s+[A-Za-z0-9._/\-]+",
        r"[\w.+-]+@[\w-]+\.[\w.-]+",
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"(?i)raw_prompt\s*[:=].*",
        r"(?i)chain_of_thought\s*[:=].*",
    ]
    for pattern in patterns:
        redacted = re.sub(pattern, "[redacted]", redacted)
    return redacted
