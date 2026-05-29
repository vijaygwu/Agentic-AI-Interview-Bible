from __future__ import annotations


def run_support_eval(responder):
    """Return an EvalReport for support-agent behavior.

    Include positive and negative cases across the canonical categories, using
    these exact category strings (the release gate keys on them):
    "grounding", "safety", "escalation", "tool_use", and "regression".
    Critical safety failures must be marked ``critical=True`` so release gates
    can block even when the aggregate pass rate is high.
    """
    raise NotImplementedError
