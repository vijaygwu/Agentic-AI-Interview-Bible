"""Tests for the ch26 release-gate solution.

Uses the book's release_gate API: ReleaseArtifacts, Release, CanaryPlan,
RollbackPlan, ReleaseScores, GateDecision, release_gate, diff_artifacts.
All tests are deterministic and offline — no live model calls.
"""
import importlib.util
from pathlib import Path

import pytest

from agentic_interview_bible.release_gate import (
    CanaryPlan,
    GateDecision,
    Release,
    ReleaseArtifacts,
    ReleaseScores,
    RollbackPlan,
    diff_artifacts,
)


def load_solution():
    path = Path(__file__).parents[1] / "solution.py"
    spec = importlib.util.spec_from_file_location("ch26_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _artifacts(**overrides) -> ReleaseArtifacts:
    base = dict(
        code_version="1.0.0",
        prompt_version="p1",
        model_alias="gpt-4o",
        tool_schema_version="ts1",
        policy_version="pol1",
        retrieval_corpus_version="rc1",
        eval_suite_version="es1",
    )
    base.update(overrides)
    return ReleaseArtifacts(**base)


def _scores(
    *,
    critical_safety_pass_rate: float = 1.0,
    critical_safety_failures: list[str] | None = None,
    regression_by_category: dict[str, float] | None = None,
) -> ReleaseScores:
    return ReleaseScores(
        critical_safety_pass_rate=critical_safety_pass_rate,
        critical_safety_failures=critical_safety_failures or [],
        regression_by_category=regression_by_category or {"refund": 1.0},
        novel_score_changes={},
        cost_per_task_cents=1.0,
        p95_latency_ms=200.0,
    )


def _release(release_id: str, artifacts: ReleaseArtifacts) -> Release:
    return Release(
        release_id=release_id,
        artifacts=artifacts,
        owner="ci-bot",
        canary_plan=CanaryPlan(),
        rollback_plan=RollbackPlan(previous_artifacts=artifacts),
    )


# ---------------------------------------------------------------------------
# diff_artifacts
# ---------------------------------------------------------------------------

def test_diff_artifacts_no_change() -> None:
    a = _artifacts()
    assert diff_artifacts(a, a) == []


def test_diff_artifacts_single_field() -> None:
    prev = _artifacts(code_version="1.0.0")
    new = _artifacts(code_version="1.1.0")
    assert diff_artifacts(prev, new) == ["code_version"]


def test_diff_artifacts_multiple_fields() -> None:
    prev = _artifacts(code_version="1.0.0", prompt_version="p1")
    new = _artifacts(code_version="1.1.0", prompt_version="p2")
    changed = diff_artifacts(prev, new)
    assert set(changed) == {"code_version", "prompt_version"}


# ---------------------------------------------------------------------------
# release_gate — single-artifact promotion
# ---------------------------------------------------------------------------

def test_single_artifact_promotion_allowed() -> None:
    solution = load_solution()
    prod_artifacts = _artifacts()
    cand_artifacts = _artifacts(code_version="1.1.0")  # one change
    prod = _release("prod-1", prod_artifacts)
    cand = _release("cand-1", cand_artifacts)

    decision = solution.gate_multi_artifact_release(
        cand, prod, _scores(), _scores()
    )

    assert decision.allowed is True
    assert decision.changed_artifacts == ["code_version"]
    assert decision.requires_override is False


# ---------------------------------------------------------------------------
# release_gate — multi-artifact requires override
# ---------------------------------------------------------------------------

def test_multi_artifact_blocked_without_override() -> None:
    solution = load_solution()
    prod_artifacts = _artifacts()
    cand_artifacts = _artifacts(
        code_version="1.1.0",
        prompt_version="p2",
        model_alias="gpt-4-turbo",
    )
    prod = _release("prod-1", prod_artifacts)
    cand = _release("cand-2", cand_artifacts)

    decision = solution.gate_multi_artifact_release(
        cand, prod, _scores(), _scores()
    )

    assert decision.allowed is False
    assert decision.requires_override is True
    assert len(decision.changed_artifacts) == 3
    assert any("multi-artifact" in reason for reason in decision.blocked_by)


def test_multi_artifact_allowed_with_override() -> None:
    solution = load_solution()
    prod_artifacts = _artifacts()
    cand_artifacts = _artifacts(
        code_version="1.1.0",
        prompt_version="p2",
        model_alias="gpt-4-turbo",
    )
    prod = _release("prod-1", prod_artifacts)
    cand = _release("cand-2", cand_artifacts)

    decision = solution.gate_multi_artifact_release(
        cand, prod, _scores(), _scores(), overrides={"cand-2"}
    )

    assert decision.allowed is True
    assert decision.requires_override is False


# ---------------------------------------------------------------------------
# release_gate — critical safety regression blocks
# ---------------------------------------------------------------------------

def test_critical_safety_regression_blocks() -> None:
    solution = load_solution()
    prod_artifacts = _artifacts()
    cand_artifacts = _artifacts(code_version="1.1.0")
    prod = _release("prod-1", prod_artifacts)
    cand = _release("cand-3", cand_artifacts)

    cand_scores = _scores(
        critical_safety_pass_rate=0.9,
        critical_safety_failures=["cross_tenant_refusal"],
    )

    decision = solution.gate_multi_artifact_release(
        cand, prod, cand_scores, _scores()
    )

    assert decision.allowed is False
    assert any("critical safety" in r for r in decision.blocked_by)


# ---------------------------------------------------------------------------
# release_gate — regression category blocks
# ---------------------------------------------------------------------------

def test_regression_category_blocks() -> None:
    solution = load_solution()
    prod_artifacts = _artifacts()
    cand_artifacts = _artifacts(prompt_version="p2")
    prod = _release("prod-1", prod_artifacts)
    cand = _release("cand-4", cand_artifacts)

    prod_scores = _scores(regression_by_category={"refund": 1.0, "auth": 0.95})
    cand_scores = _scores(regression_by_category={"refund": 0.8, "auth": 0.95})

    decision = solution.gate_multi_artifact_release(
        cand, prod, cand_scores, prod_scores
    )

    assert decision.allowed is False
    assert any("refund" in r for r in decision.blocked_by)


# ---------------------------------------------------------------------------
# Rollback: record carries previous artifacts
# ---------------------------------------------------------------------------

def test_rollback_plan_stores_previous_artifacts() -> None:
    prod_artifacts = _artifacts()
    cand_artifacts = _artifacts(code_version="1.1.0")
    rollback = RollbackPlan(previous_artifacts=prod_artifacts)
    cand = Release(
        release_id="cand-5",
        artifacts=cand_artifacts,
        owner="ci-bot",
        canary_plan=CanaryPlan(),
        rollback_plan=rollback,
    )

    # Rollback plan restores all seven artifact fields atomically.
    assert cand.rollback_plan.previous_artifacts == prod_artifacts
    assert cand.rollback_plan.previous_artifacts.code_version == "1.0.0"
