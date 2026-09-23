"""Tests for the policy-gated research copilot."""

from pathlib import Path

import pytest

from anaemia_ml.agents.orchestrator import IntentRoutingError, ResearchCopilot, route_intent
from anaemia_ml.agents.schemas import AgentIntent, AgentRequest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "demo" / "nfhs_development_summary.json"
VALIDATION = ROOT / "configs" / "validation.yaml"


@pytest.fixture
def copilot() -> ResearchCopilot:
    return ResearchCopilot(summary_path=SUMMARY, validation_path=VALIDATION)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Compare models for me", AgentIntent.COMPARE_MODELS),
        ("Why was the selected model chosen?", AgentIntent.EXPLAIN_SELECTION),
        ("Show calibration status", AgentIntent.CALIBRATION_STATUS),
        ("What are the top SHAP features?", AgentIntent.EXPLAIN_FEATURES),
        ("Is the locked test ready?", AgentIntent.CHECK_FINAL_TEST_READINESS),
        ("Give me project status", AgentIntent.PROJECT_STATUS),
    ],
)
def test_route_intent(query: str, expected: AgentIntent) -> None:
    assert route_intent(query) is expected


def test_route_intent_rejects_unknown_request() -> None:
    with pytest.raises(IntentRoutingError):
        route_intent("Write a treatment recommendation for this patient")


def test_compare_models_is_grounded_in_public_evidence(copilot: ResearchCopilot) -> None:
    response = copilot.run(AgentRequest(query="Compare models"))
    assert response.intent is AgentIntent.COMPARE_MODELS
    assert response.status == "ok"
    assert "Random Forest" in response.summary
    assert len(response.evidence) == 3
    assert "locked-test" in response.warnings[0]


def test_selection_surfaces_severe_recall_warning(copilot: ResearchCopilot) -> None:
    response = copilot.run(AgentRequest(query="Why was the selected model chosen?"))
    assert response.intent is AgentIntent.EXPLAIN_SELECTION
    assert response.status == "needs_review"
    assert any(item.label == "Severe-class recall" for item in response.evidence)
    assert any("screening tool" in warning for warning in response.warnings)


def test_calibration_reports_before_and_after(copilot: ResearchCopilot) -> None:
    response = copilot.run(AgentRequest(query="Explain calibration"))
    assert response.intent is AgentIntent.CALIBRATION_STATUS
    assert response.status == "ok"
    values = {item.label: item.value for item in response.evidence}
    assert values["Temperature"] == "0.500"
    assert values["Weighted log loss"] == "1.252 → 1.211"
    assert values["ECE"] == "0.066 → 0.013"


def test_feature_explanation_is_explicitly_noncausal(copilot: ResearchCopilot) -> None:
    response = copilot.run(AgentRequest(query="What SHAP features matter?"))
    assert response.intent is AgentIntent.EXPLAIN_FEATURES
    assert len(response.evidence) == 5
    assert response.evidence[0].label == "v445"
    assert any("causally" in warning for warning in response.warnings)


def test_final_test_readiness_is_always_blocked_in_public_copilot(
    copilot: ResearchCopilot,
) -> None:
    response = copilot.run(AgentRequest(query="Is the final test ready?"))
    assert response.intent is AgentIntent.CHECK_FINAL_TEST_READINESS
    assert response.status == "blocked"
    assert response.metadata["required_gates"]
    assert any("incapable" in warning for warning in response.warnings)


def test_explicit_intent_bypasses_text_router_but_not_policy(copilot: ResearchCopilot) -> None:
    request = AgentRequest(
        query="Give me a concise answer",
        intent=AgentIntent.PROJECT_STATUS,
    )
    response = copilot.run(request)
    assert response.intent is AgentIntent.PROJECT_STATUS
    assert response.status == "ok"
