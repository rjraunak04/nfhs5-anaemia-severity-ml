"""Planner abstraction for deterministic, hybrid, and optional LLM routing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from anaemia_ml.agents.schemas import AgentIntent, PlanDecision


class Planner(Protocol):
    """Minimal contract required by the research copilot."""

    name: str

    def plan(self, query: str) -> PlanDecision:
        """Choose one approved intent for a user query."""


class RuleBasedPlanner:
    """Dependency-free planner used by the public deployment."""

    name = "rule_based"

    def plan(self, query: str) -> PlanDecision:
        """Route with the auditable local intent matcher."""
        from anaemia_ml.agents.orchestrator import route_intent

        intent = route_intent(query)
        return PlanDecision(
            intent=intent,
            planner=self.name,
            confidence=0.98,
            reason="Matched an explicit approved research intent.",
        )


class CallablePlanner:
    """Adapter for an external LLM/tool-calling planner.

    The supplied callable may use any provider, but its output is validated
    against the closed AgentIntent enum before the copilot can execute a tool.
    """

    def __init__(
        self,
        planner: Callable[[str], str | AgentIntent | PlanDecision],
        *,
        name: str = "external_llm",
    ) -> None:
        self._planner = planner
        self.name = name

    def plan(self, query: str) -> PlanDecision:
        """Validate an external planner decision before execution."""
        result = self._planner(query)
        if isinstance(result, PlanDecision):
            return result.model_copy(update={"planner": self.name})
        return PlanDecision(
            intent=AgentIntent(result),
            planner=self.name,
            confidence=0.75,
            reason="External planner selected an approved intent.",
        )


class HybridPlanner:
    """Prefer deterministic routing and use a fallback planner only when needed."""

    name = "hybrid"

    def __init__(self, fallback: Planner | None = None) -> None:
        self.primary = RuleBasedPlanner()
        self.fallback = fallback

    def plan(self, query: str) -> PlanDecision:
        """Use rules for known requests and an optional LLM for ambiguity."""
        try:
            decision = self.primary.plan(query)
        except ValueError:
            if self.fallback is None:
                raise
            fallback = self.fallback.plan(query)
            return fallback.model_copy(
                update={
                    "planner": f"{self.name}:{fallback.planner}",
                    "reason": "Rule routing was ambiguous; fallback planner selected the intent.",
                }
            )
        return decision.model_copy(
            update={
                "planner": f"{self.name}:{decision.planner}",
                "reason": "High-confidence deterministic route; no LLM call was needed.",
            }
        )
