from __future__ import annotations


def run_support_eval(responder):
    """Return an EvalReport for support-agent behavior.

    Use EvalCase with typed Expected outcomes (ExpectSuccess, ExpectRefusal,
    ExpectEscalation) and the Tier enum (CRITICAL_SAFETY, REGRESSION, NOVEL).
    Include positive and negative cases across canonical categories; tag cases
    so the harness can run subsets.  Run via run_eval_suite.
    """
    raise NotImplementedError
