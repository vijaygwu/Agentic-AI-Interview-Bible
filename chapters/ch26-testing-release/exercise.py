from __future__ import annotations


def gate_prompt_release(report, minimum_pass_rate: float = 1.0):
    """Return a release decision from an eval report.

    The gate must block when aggregate pass rate is below the threshold or when
    any critical safety/reliability case fails, even if the pass rate is high.
    It must also block reports that omit required categories: grounding,
    safety, escalation, tool_use, and regression. Safety, escalation, tool_use,
    and regression cases must include critical eval coverage so the release gate
    cannot pass on optional safety checks.
    """
    raise NotImplementedError
