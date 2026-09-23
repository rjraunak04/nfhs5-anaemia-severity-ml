"""Governance policies enforced before an agent response is returned."""

from __future__ import annotations

from collections.abc import Mapping

from anaemia_ml.agents.schemas import AgentIntent


class AgentPolicyError(RuntimeError):
    """Raised when an agent action violates a project policy."""


PUBLIC_ALLOWED_INTENTS = frozenset(
    {
        AgentIntent.PROJECT_STATUS,
        AgentIntent.COMPARE_MODELS,
        AgentIntent.EXPLAIN_SELECTION,
        AgentIntent.CALIBRATION_STATUS,
        AgentIntent.EXPLAIN_FEATURES,
        AgentIntent.CHECK_FINAL_TEST_READINESS,
    }
)


def enforce_public_intent(intent: AgentIntent) -> None:
    """Reject any intent not explicitly approved for the public copilot."""
    if intent not in PUBLIC_ALLOWED_INTENTS:
        raise AgentPolicyError(f"Intent is not allowed in public mode: {intent}")


def enforce_disclosure_boundary(summary: Mapping[str, object]) -> None:
    """Require the public summary to retain the locked-test boundary."""
    disclosure = summary.get("disclosure")
    if not isinstance(disclosure, Mapping):
        raise AgentPolicyError("Public summary is missing disclosure metadata.")
    if disclosure.get("locked_test_evaluated") is not False:
        raise AgentPolicyError("Public copilot cannot consume evaluated locked-test results.")
    if disclosure.get("final_performance_claim_allowed") is not False:
        raise AgentPolicyError("Public copilot cannot make final-performance claims.")
