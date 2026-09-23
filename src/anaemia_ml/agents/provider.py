"""Optional external intent-planner transport with fail-closed safeguards."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from time import monotonic, perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from anaemia_ml.agents.planner import HybridPlanner
from anaemia_ml.agents.schemas import AgentIntent, PlanDecision, PlannerTelemetry


class ExternalPlannerError(ValueError):
    """Raised when an external planner cannot produce a safe approved decision."""


@dataclass(frozen=True)
class HttpPlannerConfig:
    """Runtime configuration for a provider-agnostic intent gateway."""

    endpoint: str
    model: str = "intent-router"
    api_key: str | None = None
    timeout_seconds: float = 2.5
    failure_threshold: int = 2
    cooldown_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(("https://", "http://")):
            raise ValueError("External planner endpoint must be HTTP(S).")
        if self.timeout_seconds <= 0:
            raise ValueError("Planner timeout must be positive.")
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1.")
        if self.cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be positive.")


class HttpIntentPlanner:
    """Call a small external LLM gateway only for intent classification.

    Outbound payloads contain only the user's query, the model label, and the
    closed list of allowed intents. No project metrics, NFHS rows, model
    artifacts, validation configs, or evidence payloads are transmitted.
    """

    name = "external_http_llm"

    def __init__(self, config: HttpPlannerConfig) -> None:
        self.config = config
        self._consecutive_failures = 0
        self._open_until = 0.0

    @property
    def circuit_open(self) -> bool:
        return monotonic() < self._open_until

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.config.failure_threshold:
            self._open_until = monotonic() + self.config.cooldown_seconds

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._open_until = 0.0

    def plan(self, query: str) -> PlanDecision:
        """Request one validated intent from the configured gateway."""
        if self.circuit_open:
            raise ExternalPlannerError(
                "External planner circuit is open; request was not sent."
            )

        payload = {
            "query": query,
            "model": self.config.model,
            "allowed_intents": [intent.value for intent in AgentIntent],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        request = Request(
            self.config.endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        started = perf_counter()
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            intent = AgentIntent(parsed["intent"])
            confidence = float(parsed.get("confidence", 0.75))
            usage = parsed.get("usage") or {}
            telemetry = PlannerTelemetry(
                provider=str(parsed.get("provider", "external_http")),
                model=str(parsed.get("model", self.config.model)),
                latency_ms=round((perf_counter() - started) * 1000.0, 3),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
        except (
            HTTPError,
            URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            self._record_failure()
            raise ExternalPlannerError(
                "External planner failed safely; no project tool was executed."
            ) from exc

        self._record_success()
        return PlanDecision(
            intent=intent,
            planner=self.name,
            confidence=confidence,
            reason="External fallback mapped an ambiguous request to an approved intent.",
            telemetry=telemetry,
        )


def build_planner_from_env() -> HybridPlanner:
    """Build the live hybrid planner without ever hard-coding secrets."""
    enabled = os.getenv("ANAEMIA_AGENT_LLM_ENABLED", "").strip().casefold()
    if enabled not in {"1", "true", "yes", "on"}:
        return HybridPlanner()

    endpoint = os.getenv("ANAEMIA_AGENT_LLM_ENDPOINT", "").strip()
    if not endpoint:
        raise ExternalPlannerError(
            "External planner is enabled but ANAEMIA_AGENT_LLM_ENDPOINT is missing."
        )

    timeout = float(os.getenv("ANAEMIA_AGENT_LLM_TIMEOUT_SECONDS", "2.5"))
    threshold = int(os.getenv("ANAEMIA_AGENT_LLM_FAILURE_THRESHOLD", "2"))
    cooldown = float(os.getenv("ANAEMIA_AGENT_LLM_COOLDOWN_SECONDS", "30"))

    config = HttpPlannerConfig(
        endpoint=endpoint,
        model=os.getenv("ANAEMIA_AGENT_LLM_MODEL", "intent-router").strip()
        or "intent-router",
        api_key=os.getenv("ANAEMIA_AGENT_LLM_API_KEY") or None,
        timeout_seconds=timeout,
        failure_threshold=threshold,
        cooldown_seconds=cooldown,
    )
    return HybridPlanner(fallback=HttpIntentPlanner(config))


def external_planner_enabled() -> bool:
    """Return whether the live environment opted into external fallback."""
    return os.getenv("ANAEMIA_AGENT_LLM_ENABLED", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
