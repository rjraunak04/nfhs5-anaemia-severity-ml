"""Typed contracts for agent requests, evidence, and responses."""

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


class AgentRequest(BaseModel):
    """Validated user request entering the copilot."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=500)
    intent: AgentIntent | None = None


class EvidenceItem(BaseModel):
    """One auditable fact used by the copilot."""

    model_config = ConfigDict(extra="forbid")

    label: str
    value: str
    source: str


class AgentResponse(BaseModel):
    """Structured response returned by the copilot."""

    model_config = ConfigDict(extra="forbid")

    intent: AgentIntent
    status: Literal["ok", "blocked", "needs_review"]
    title: str
    summary: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
