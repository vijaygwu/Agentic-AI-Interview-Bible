from __future__ import annotations


def run_support_eval(responder):
    """Return an EvalReport for support-agent behavior.

    Include positive and negative cases across grounding, refusal, escalation,
    tool-use safety, and regression categories. Critical safety failures should
    be marked so release gates can block even when aggregate pass rate is high.
    """
    raise NotImplementedError
