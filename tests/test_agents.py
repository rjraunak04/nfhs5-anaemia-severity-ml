"""Tests for the policy-gated research copilot."""

from pathlib import Path

import pytest

from anaemia_ml.agents.orchestrator import IntentRoutingError, ResearchCopilot, route_intent
from anaemia_ml.agents.planner import CallablePlanner, HybridPlanner
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


def test_external_planner_is_constrained_to_approved_intents() -> None:
    planner = CallablePlanner(lambda _query: "compare_models", name="test_llm")
    copilot = ResearchCopilot(
        summary_path=SUMMARY,
        validation_path=VALIDATION,
        planner=planner,
    )
    response = copilot.run(AgentRequest(query="Use the external planner"))
    assert response.intent is AgentIntent.COMPARE_MODELS
    assert response.metadata["planner"] == "test_llm"
    assert response.metadata["evidence_only"] is True


def test_external_planner_cannot_invent_unapproved_action() -> None:
    planner = CallablePlanner(lambda _query: "diagnose_patient", name="test_llm")
    copilot = ResearchCopilot(
        summary_path=SUMMARY,
        validation_path=VALIDATION,
        planner=planner,
    )
    with pytest.raises(ValueError):
        copilot.run(AgentRequest(query="Do something unsafe"))



def test_release_readiness_audits_public_engineering_gates(copilot: ResearchCopilot) -> None:
    response = copilot.run(AgentRequest(query="Is the portfolio release ready?"))
    assert response.intent is AgentIntent.RELEASE_READINESS
    assert response.status == "ok"
    assert response.metadata["portfolio_release_ready"] is True
    assert response.metadata["confirmatory_release_ready"] is False
    assert all("PASS" in item.value for item in response.evidence)


def test_next_experiment_plan_preserves_locked_test_boundary(copilot: ResearchCopilot) -> None:
    response = copilot.run(AgentRequest(query="What experiment should I run next?"))
    assert response.intent is AgentIntent.NEXT_EXPERIMENT
    assert response.status == "needs_review"
    labels = [item.label for item in response.evidence]
    assert any("state-held-out validation" in label for label in labels)
    assert any("single-use locked-test evaluation" in label for label in labels)
    assert any("not authorization" in warning for warning in response.warnings)


def test_response_contains_auditable_execution_trace(copilot: ResearchCopilot) -> None:
    response = copilot.run(AgentRequest(query="Compare models"))
    assert [step.stage for step in response.trace] == [
        "plan",
        "policy",
        "policy",
        "tool",
        "response",
    ]
    assert response.metadata["planner"].startswith("hybrid:")
    assert response.metadata["tool"] == "compare_development_models"
    assert 0.0 <= response.metadata["planner_confidence"] <= 1.0


def test_hybrid_planner_skips_fallback_for_known_request() -> None:
    calls = {"count": 0}

    def fallback(_query: str) -> str:
        calls["count"] += 1
        return "project_status"

    planner = HybridPlanner(CallablePlanner(fallback, name="test_llm"))
    decision = planner.plan("Compare models")
    assert decision.intent is AgentIntent.COMPARE_MODELS
    assert decision.planner == "hybrid:rule_based"
    assert calls["count"] == 0


def test_hybrid_planner_uses_fallback_only_for_ambiguous_request() -> None:
    planner = HybridPlanner(
        CallablePlanner(lambda _query: "release_readiness", name="test_llm")
    )
    decision = planner.plan("Can this artifact ship?")
    assert decision.intent is AgentIntent.RELEASE_READINESS
    assert decision.planner == "hybrid:test_llm"
    assert "fallback" in decision.reason.casefold()


def test_hybrid_fallback_cannot_create_unapproved_intent() -> None:
    planner = HybridPlanner(
        CallablePlanner(lambda _query: "delete_repository", name="test_llm")
    )
    with pytest.raises(ValueError):
        planner.plan("Do something outside the supported vocabulary")
