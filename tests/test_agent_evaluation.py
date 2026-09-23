"""Tests for the versioned research-copilot evaluation harness."""

from pathlib import Path

from anaemia_ml.agents.evaluation import evaluate_copilot, load_eval_cases
from anaemia_ml.agents.orchestrator import ResearchCopilot

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "demo" / "nfhs_development_summary.json"
VALIDATION = ROOT / "configs" / "validation.yaml"
CASES = ROOT / "demo" / "agent_eval_cases.json"


def test_golden_agent_suite_passes() -> None:
    copilot = ResearchCopilot(
        summary_path=SUMMARY,
        validation_path=VALIDATION,
    )
    report = evaluate_copilot(copilot, load_eval_cases(CASES))

    assert report.total_cases == 20
    assert report.passed_cases == report.total_cases
    assert report.case_pass_rate == 1.0
    assert report.routing_accuracy == 1.0
    assert report.tool_accuracy == 1.0
    assert report.status_accuracy == 1.0
    assert report.safety_pass_rate == 1.0


def test_golden_suite_reports_latency_without_persisting_prompts() -> None:
    copilot = ResearchCopilot(
        summary_path=SUMMARY,
        validation_path=VALIDATION,
    )
    report = evaluate_copilot(copilot, load_eval_cases(CASES))

    assert report.latency_mean_ms >= 0.0
    assert report.latency_median_ms >= 0.0
    assert report.latency_p95_ms >= 0.0
    dumped = report.model_dump()
    assert "query" not in str(dumped)
