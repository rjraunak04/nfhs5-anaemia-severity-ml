"""Planner abstraction for deterministic and optional LLM-backed routing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from anaemia_ml.agents.schemas import AgentIntent


class Planner(Protocol):
    """Minimal contract required by the research copilot."""

    name: str

    def plan(self, query: str) -> AgentIntent:
        """Choose one approved intent for a user query."""


class RuleBasedPlanner:
    """Dependency-free planner used by the public deployment."""

    name = "rule_based"

    def plan(self, query: str) -> AgentIntent:
        """Route with the auditable local intent matcher."""
        from anaemia_ml.agents.orchestrator import route_intent

        return route_intent(query)


class CallablePlanner:
    """Adapter for an external LLM/tool-calling planner.

    The supplied callable may use any provider, but its output is validated
    against the closed AgentIntent enum before the copilot can execute a tool.
    """

    def __init__(
        self,
        planner: Callable[[str], str | AgentIntent],
        *,
        name: str = "external_llm",
    ) -> None:
        self._planner = planner
        self.name = name

    def plan(self, query: str) -> AgentIntent:
        """Validate an external planner decision before execution."""
        return AgentIntent(self._planner(query))
