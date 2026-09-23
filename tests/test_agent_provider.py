"""Tests for the optional external LLM intent-planner boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anaemia_ml.agents.orchestrator import ResearchCopilot
from anaemia_ml.agents.provider import (
    ExternalPlannerError,
    HttpIntentPlanner,
    HttpPlannerConfig,
    build_planner_from_env,
)
from anaemia_ml.agents.schemas import AgentIntent, AgentRequest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "demo" / "nfhs_development_summary.json"
VALIDATION = ROOT / "configs" / "validation.yaml"


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_external_planner_sends_only_query_model_and_allowlist(monkeypatch) -> None:
    captured: dict = {}

    def fake_urlopen(request, timeout: float):
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "intent": "release_readiness",
                "confidence": 0.91,
                "provider": "test-provider",
                "model": "router-small",
                "usage": {"prompt_tokens": 21, "completion_tokens": 4},
            }
        )

    monkeypatch.setattr("anaemia_ml.agents.provider.urlopen", fake_urlopen)
    planner = HttpIntentPlanner(
        HttpPlannerConfig(
            endpoint="https://planner.example.test/intent",
            model="router-small",
            api_key="unit-test-key",
            timeout_seconds=1.5,
        )
    )

    decision = planner.plan("Can this artifact ship?")

    assert set(captured["payload"]) == {"query", "model", "allowed_intents"}
    assert captured["payload"]["query"] == "Can this artifact ship?"
    assert captured["payload"]["model"] == "router-small"
    assert set(captured["payload"]["allowed_intents"]) == {
        intent.value for intent in AgentIntent
    }
    assert "summary" not in captured["payload"]
    assert "evidence" not in captured["payload"]
    assert captured["timeout"] == 1.5
    assert captured["headers"]["Authorization"] == "Bearer unit-test-key"
    assert decision.intent is AgentIntent.RELEASE_READINESS
    assert decision.telemetry is not None
    assert decision.telemetry.provider == "test-provider"
    assert decision.telemetry.prompt_tokens == 21
    assert decision.telemetry.completion_tokens == 4


def test_invalid_external_intent_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        "anaemia_ml.agents.provider.urlopen",
        lambda *_args, **_kwargs: FakeResponse({"intent": "delete_repository"}),
    )
    planner = HttpIntentPlanner(
        HttpPlannerConfig(endpoint="https://planner.example.test/intent")
    )

    with pytest.raises(ExternalPlannerError, match="failed safely"):
        planner.plan("Do something unusual")


def test_timeout_opens_circuit_after_repeated_failures(monkeypatch) -> None:
    calls = {"count": 0}

    def timeout_urlopen(*_args, **_kwargs):
        calls["count"] += 1
        raise TimeoutError("provider timeout")

    monkeypatch.setattr("anaemia_ml.agents.provider.urlopen", timeout_urlopen)
    planner = HttpIntentPlanner(
        HttpPlannerConfig(
            endpoint="https://planner.example.test/intent",
            failure_threshold=2,
            cooldown_seconds=60,
        )
    )

    with pytest.raises(ExternalPlannerError):
        planner.plan("Ambiguous request one")
    with pytest.raises(ExternalPlannerError):
        planner.plan("Ambiguous request two")

    assert planner.circuit_open is True

    with pytest.raises(ExternalPlannerError, match="circuit is open"):
        planner.plan("Ambiguous request three")
    assert calls["count"] == 2


def test_env_disabled_mode_has_no_external_fallback(monkeypatch) -> None:
    monkeypatch.delenv("ANAEMIA_AGENT_LLM_ENABLED", raising=False)
    planner = build_planner_from_env()

    assert planner.fallback is None


def test_env_enabled_requires_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("ANAEMIA_AGENT_LLM_ENABLED", "true")
    monkeypatch.delenv("ANAEMIA_AGENT_LLM_ENDPOINT", raising=False)

    with pytest.raises(ExternalPlannerError, match="ENDPOINT is missing"):
        build_planner_from_env()


def test_env_enabled_fallback_surfaces_safe_telemetry(monkeypatch) -> None:
    monkeypatch.setenv("ANAEMIA_AGENT_LLM_ENABLED", "true")
    monkeypatch.setenv(
        "ANAEMIA_AGENT_LLM_ENDPOINT",
        "https://planner.example.test/intent",
    )
    monkeypatch.setenv("ANAEMIA_AGENT_LLM_MODEL", "router-small")

    monkeypatch.setattr(
        "anaemia_ml.agents.provider.urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            {
                "intent": "release_readiness",
                "confidence": 0.88,
                "provider": "test-provider",
                "model": "router-small",
                "usage": {"prompt_tokens": 18, "completion_tokens": 3},
            }
        ),
    )

    copilot = ResearchCopilot(
        summary_path=SUMMARY,
        validation_path=VALIDATION,
        planner=build_planner_from_env(),
    )
    response = copilot.run(AgentRequest(query="Can this artifact ship?"))

    assert response.intent is AgentIntent.RELEASE_READINESS
    assert response.metadata["planner"] == "hybrid:external_http_llm"
    assert response.metadata["planner_provider"] == "test-provider"
    assert response.metadata["planner_model"] == "router-small"
    assert response.metadata["planner_prompt_tokens"] == 18
    assert response.metadata["planner_completion_tokens"] == 3
    assert response.metadata["evidence_only"] is True
