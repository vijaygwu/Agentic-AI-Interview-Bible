"""Ch26 solution: release gate with multi-artifact versioning.

Uses the book's release_gate API: ReleaseArtifacts, Release, diff_artifacts,
GateDecision, and release_gate from agentic_interview_bible.release_gate.
"""
from __future__ import annotations

from agentic_interview_bible.release_gate import (
    GateDecision,
    Release,
    ReleaseArtifacts,
    ReleaseScores,
    diff_artifacts,
    release_gate,
)

# Re-export ReleaseScores so exercise files can import it from here if needed.
__all__ = [
    "GateDecision",
    "Release",
    "ReleaseArtifacts",
    "ReleaseScores",
    "diff_artifacts",
    "gate_multi_artifact_release",
    "release_gate",
]


def gate_multi_artifact_release(
    candidate: Release,
    production: Release,
    scores_candidate: ReleaseScores,
    scores_production: ReleaseScores,
    overrides: set[str] | None = None,
) -> GateDecision:
    """Thin wrapper around release_gate with a default empty overrides set."""
    return release_gate(
        candidate,
        production,
        scores_candidate,
        scores_production,
        overrides=overrides or set(),
    )
