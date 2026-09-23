"""Deterministic evaluation harness for the research copilot."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from anaemia_ml.agents.orchestrator import ResearchCopilot
from anaemia_ml.agents.schemas import AgentIntent, AgentRequest


class EvalCase(BaseModel):
    """One versioned golden evaluation case."""

    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    expected_intent: AgentIntent | None = None
    expected_status: str | None = None
    expected_tool: str | None = None
    expect_rejection: bool = False
    tags: list[str] = Field(default_factory=list)


class EvalCaseResult(BaseModel):
    """Observed result for one golden case."""

    model_config = ConfigDict(extra="forbid")

    id: str
    passed: bool
    routing_ok: bool
    tool_ok: bool
    status_ok: bool
    safety_ok: bool
    latency_ms: float
    observed_intent: str | None = None
    observed_status: str | None = None
    observed_tool: str | None = None
    error_type: str | None = None


class EvalReport(BaseModel):
    """Aggregate quality report for a copilot evaluation run."""

    model_config = ConfigDict(extra="forbid")

    total_cases: int
    passed_cases: int
    case_pass_rate: float
    routing_accuracy: float
    tool_accuracy: float
    status_accuracy: float
    safety_pass_rate: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    results: list[EvalCaseResult]


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """Load a version-controlled JSON golden set."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Agent evaluation set must be a non-empty JSON list.")
    return [EvalCase.model_validate(item) for item in payload]


def _p95(values: list[float]) -> float:
    """Return a deterministic nearest-rank p95."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int((0.95 * len(ordered)) + 0.999999))
    return ordered[min(rank - 1, len(ordered) - 1)]


def evaluate_copilot(
    copilot: ResearchCopilot,
    cases: list[EvalCase],
) -> EvalReport:
    """Evaluate observable routing, tool, safety, and latency behavior."""
    results: list[EvalCaseResult] = []

    routing_checks: list[bool] = []
    tool_checks: list[bool] = []
    status_checks: list[bool] = []
    safety_checks: list[bool] = []
    latencies: list[float] = []

    for case in cases:
        start = perf_counter()
        response = None
        error: Exception | None = None
        try:
            response = copilot.run(AgentRequest(query=case.query))
        except (ValueError, RuntimeError) as exc:
            error = exc
        latency_ms = (perf_counter() - start) * 1000.0
        latencies.append(latency_ms)

        if case.expect_rejection:
            routing_ok = error is not None
            tool_ok = error is not None
            status_ok = error is not None
            safety_ok = error is not None
            observed_intent = None
            observed_status = None
            observed_tool = None
        else:
            routing_ok = (
                response is not None
                and case.expected_intent is not None
                and response.intent == case.expected_intent
            )
            tool_ok = (
                response is not None
                and case.expected_tool is not None
                and response.metadata.get("tool") == case.expected_tool
            )
            status_ok = (
                response is not None
                and case.expected_status is not None
                and response.status == case.expected_status
            )
            safety_sensitive = "safety" in case.tags or "privacy" in case.tags
            safety_ok = (
                status_ok
                if safety_sensitive
                else True
            )
            observed_intent = response.intent.value if response is not None else None
            observed_status = response.status if response is not None else None
            observed_tool = (
                str(response.metadata.get("tool"))
                if response is not None and response.metadata.get("tool") is not None
                else None
            )

            routing_checks.append(routing_ok)
            tool_checks.append(tool_ok)
            status_checks.append(status_ok)

        if case.expect_rejection or "safety" in case.tags or "privacy" in case.tags:
            safety_checks.append(safety_ok)

        passed = routing_ok and tool_ok and status_ok and safety_ok
        results.append(
            EvalCaseResult(
                id=case.id,
                passed=passed,
                routing_ok=routing_ok,
                tool_ok=tool_ok,
                status_ok=status_ok,
                safety_ok=safety_ok,
                latency_ms=round(latency_ms, 3),
                observed_intent=observed_intent,
                observed_status=observed_status,
                observed_tool=observed_tool,
                error_type=type(error).__name__ if error is not None else None,
            )
        )

    def ratio(checks: list[bool]) -> float:
        return sum(checks) / len(checks) if checks else 1.0

    passed_cases = sum(result.passed for result in results)
    return EvalReport(
        total_cases=len(results),
        passed_cases=passed_cases,
        case_pass_rate=passed_cases / len(results),
        routing_accuracy=ratio(routing_checks),
        tool_accuracy=ratio(tool_checks),
        status_accuracy=ratio(status_checks),
        safety_pass_rate=ratio(safety_checks),
        latency_mean_ms=round(mean(latencies), 3),
        latency_median_ms=round(median(latencies), 3),
        latency_p95_ms=round(_p95(latencies), 3),
        results=results,
    )


def report_summary(report: EvalReport) -> dict[str, Any]:
    """Return a compact serializable summary for CI and dashboards."""
    return {
        "total_cases": report.total_cases,
        "passed_cases": report.passed_cases,
        "case_pass_rate": report.case_pass_rate,
        "routing_accuracy": report.routing_accuracy,
        "tool_accuracy": report.tool_accuracy,
        "status_accuracy": report.status_accuracy,
        "safety_pass_rate": report.safety_pass_rate,
        "latency_mean_ms": report.latency_mean_ms,
        "latency_median_ms": report.latency_median_ms,
        "latency_p95_ms": report.latency_p95_ms,
    }
