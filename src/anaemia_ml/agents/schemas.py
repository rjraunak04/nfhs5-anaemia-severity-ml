"""Typed contracts for agent planning, evidence, traces, and responses."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentIntent(StrEnum):
    """Supported research-copilot intents."""

    PROJECT_STATUS = "project_status"
    COMPARE_MODELS = "compare_models"
    EXPLAIN_SELECTION = "explain_selection"
    CALIBRATION_STATUS = "calibration_status"
    EXPLAIN_FEATURES = "explain_features"
    CHECK_FINAL_TEST_READINESS = "check_final_test_readiness"
    RELEASE_READINESS = "release_readiness"
    NEXT_EXPERIMENT = "next_experiment"


class AgentRequest(BaseModel):
    """Validated user request entering the copilot."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=500)
    intent: AgentIntent | None = None


class PlannerTelemetry(BaseModel):
    """Non-sensitive telemetry returned by an optional external planner."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str | None = None
    latency_ms: float = Field(ge=0.0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


class PlanDecision(BaseModel):
    """Auditable planner decision before any tool is executed."""

    model_config = ConfigDict(extra="forbid")

    intent: AgentIntent
    planner: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=300)
    telemetry: PlannerTelemetry | None = None


class EvidenceItem(BaseModel):
    """One auditable fact used by the copilot."""

    model_config = ConfigDict(extra="forbid")

    label: str
    value: str
    source: str


class TraceStep(BaseModel):
    """One safe, user-visible step in the agent execution trace."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal["plan", "policy", "tool", "response"]
    name: str
    status: Literal["ok", "blocked", "needs_review"]
    detail: str


class AgentResponse(BaseModel):
    """Structured response returned by the copilot."""

    model_config = ConfigDict(extra="forbid")

    intent: AgentIntent
    status: Literal["ok", "blocked", "needs_review"]
    title: str
    summary: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
