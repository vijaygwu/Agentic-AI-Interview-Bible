from __future__ import annotations


def parse_refund_decision(payload: dict[str, object]) -> dict[str, object]:
    """Validate and return a structured refund decision.

    Requirements:
    - decision must be one of approve, deny, or escalate
    - amount_cents must be an integer and cannot be negative
    - requires_human_approval must be a real boolean, not 0 or 1
    - evidence_ids must be a non-empty list of strings
    - policy_version must be present so the decision is auditable
    """
    raise NotImplementedError
