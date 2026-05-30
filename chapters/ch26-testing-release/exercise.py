from __future__ import annotations


def gate_multi_artifact_release(
    candidate,
    production,
    scores_candidate,
    scores_production,
    overrides=None,
):
    """Return a GateDecision for a multi-artifact release candidate.

    Use release_gate(candidate, production, scores_candidate,
    scores_production, overrides) from agentic_interview_bible.release_gate.
    The gate must reject when any artifact regressed, when critical safety
    pass rate is below 1.0, and when more than two artifacts changed without
    an explicit override.
    """
    raise NotImplementedError
