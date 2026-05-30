"""Release gate with multi-artifact versioning.

Public API matches the book's ``problem-release-gate-versioning`` and
``problem-regression-gate`` listings exactly.  Names and signatures are
canonical; do not rename without updating the companion LaTeX source.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from .eval_harness import ReleaseScores


# ---------------------------------------------------------------------------
# Multi-artifact release structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReleaseArtifacts:
    code_version: str
    prompt_version: str
    model_alias: str
    tool_schema_version: str
    policy_version: str
    retrieval_corpus_version: str
    eval_suite_version: str


@dataclass(frozen=True)
class CanaryPlan:
    """Canary traffic rollout description stored alongside a release."""
    initial_traffic_pct: float = 1.0
    step_pct: float = 10.0
    bake_time_minutes: int = 30
    auto_promote: bool = False


@dataclass(frozen=True)
class RollbackPlan:
    """Rollback specification stored alongside a release record.

    *previous_artifacts* names the artifact set to restore; *side_effects*
    lists irreversible effects (e.g. emails sent, payments processed) that
    the version rollback cannot undo.
    """
    previous_artifacts: ReleaseArtifacts
    side_effects: list[str] = dataclasses.field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class Release:
    release_id: str
    artifacts: ReleaseArtifacts
    owner: str
    canary_plan: CanaryPlan
    rollback_plan: RollbackPlan


# ---------------------------------------------------------------------------
# diff_artifacts
# ---------------------------------------------------------------------------

def diff_artifacts(
    prev: ReleaseArtifacts, new: ReleaseArtifacts
) -> list[str]:
    """Return names of artifact fields that changed between *prev* and *new*."""
    changed = []
    for field in dataclasses.fields(ReleaseArtifacts):
        if getattr(prev, field.name) != getattr(new, field.name):
            changed.append(field.name)
    return changed


# ---------------------------------------------------------------------------
# ReleaseBudgets — kept here so both modules are self-contained; also
# re-exported from eval_harness for callers that import it there.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReleaseBudgets:
    max_cost_per_task_cents: float
    max_p95_latency_ms: float


# ---------------------------------------------------------------------------
# GateDecision — shared by both regression_gate and release_gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    blocked_by: list[str]   # specific failure reasons
    warnings: list[str]     # non-blocking concerns
    requires_override: bool = False
    changed_artifacts: list[str] = dataclasses.field(default_factory=list)
    summary: dict[str, Any] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# regression_gate  (problem-regression-gate listing)
# ---------------------------------------------------------------------------

def regression_gate(
    candidate: ReleaseScores,
    production: ReleaseScores,
    budgets: ReleaseBudgets,
) -> GateDecision:
    """Gate a candidate release against production scores and cost/latency budgets.

    Matches the book's ``problem-regression-gate`` strong-attempt listing.
    """
    blocked: list[str] = []
    warnings: list[str] = []

    # Critical safety: 100% required.
    if candidate.critical_safety_pass_rate < 1.0:
        failed = candidate.critical_safety_failures
        blocked.append(
            f"critical safety regressed: {len(failed)} failures "
            f"({failed[:3]}...)"
        )

    # Regression: per-category, not aggregate.
    for cat, prod_rate in production.regression_by_category.items():
        cand_rate = candidate.regression_by_category.get(cat, 0)
        if cand_rate < prod_rate:
            blocked.append(
                f"regression category {cat}: {cand_rate:.3f} < "
                f"production {prod_rate:.3f}"
            )

    # Cost: within budget.
    if candidate.cost_per_task_cents > budgets.max_cost_per_task_cents:
        blocked.append(
            f"cost per task {candidate.cost_per_task_cents}c > "
            f"budget {budgets.max_cost_per_task_cents}c"
        )

    # Latency: within budget.
    if candidate.p95_latency_ms > budgets.max_p95_latency_ms:
        blocked.append(
            f"p95 latency {candidate.p95_latency_ms}ms > "
            f"budget {budgets.max_p95_latency_ms}ms"
        )

    # Novel: warnings only.
    for cat, change in candidate.novel_score_changes.items():
        if change < -0.05:
            warnings.append(f"novel category {cat} declined {change:.3f}")

    return GateDecision(
        allowed=not blocked,
        blocked_by=blocked,
        warnings=warnings,
        summary=candidate.summary(),
    )


# ---------------------------------------------------------------------------
# release_gate  (problem-release-gate-versioning listing)
# ---------------------------------------------------------------------------

def release_gate(
    candidate: Release,
    production: Release,
    scores_candidate: ReleaseScores,
    scores_production: ReleaseScores,
    overrides: set[str],
) -> GateDecision:
    """Gate a multi-artifact release candidate.

    Matches the book's ``problem-release-gate-versioning`` strong-attempt
    listing.  Compares artifacts, enforces the two-artifact override rule,
    and delegates regression/safety scoring to the same checks used by
    :func:`regression_gate`.
    """
    changed = diff_artifacts(production.artifacts, candidate.artifacts)
    blocked: list[str] = []
    requires_override = False

    if len(changed) > 2 and candidate.release_id not in overrides:
        requires_override = True
        blocked.append(
            f"multi-artifact release changes {len(changed)} artifacts "
            f"({changed}); requires override"
        )

    # Standard regression checks.
    if scores_candidate.critical_safety_pass_rate < 1.0:
        blocked.append("critical safety regression")

    for cat, prod_rate in scores_production.regression_by_category.items():
        cand_rate = scores_candidate.regression_by_category.get(cat, 0)
        if cand_rate < prod_rate:
            blocked.append(f"regression on {cat}")

    return GateDecision(
        allowed=not blocked,
        blocked_by=blocked,
        warnings=[],
        requires_override=requires_override,
        changed_artifacts=changed,
        summary=scores_candidate.summary(),
    )


# ---------------------------------------------------------------------------
# Legacy shim kept so __init__.py can still export ReleaseDecision / decide_release
# without modification.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReleaseDecision:
    allowed: bool
    reason: str


def decide_release(report: Any, minimum_pass_rate: float) -> ReleaseDecision:
    """Legacy gate used by ch26 solution and tests.

    Accepts a *report* with ``.critical_failures`` and ``.pass_rate``
    attributes (as produced by :func:`run_eval_cases`).
    """
    if not 0 <= minimum_pass_rate <= 1:
        raise ValueError("minimum_pass_rate must be between 0 and 1")
    if report.critical_failures:
        names = ", ".join(result.name for result in report.critical_failures)
        return ReleaseDecision(False, f"critical eval failed: {names}")
    if report.pass_rate >= minimum_pass_rate:
        return ReleaseDecision(True, "eval gate passed")
    return ReleaseDecision(False, "eval gate failed")
