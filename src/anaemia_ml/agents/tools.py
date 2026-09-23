"""Deterministic tools used by the research copilot."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from anaemia_ml.dashboard import load_portfolio_summary
from anaemia_ml.evaluation.config import load_validation_config


class AgentToolError(ValueError):
    """Raised when a copilot tool receives incomplete aggregate evidence."""


def load_public_evidence(path: str | Path) -> dict[str, Any]:
    """Load disclosure-checked aggregate evidence."""
    return load_portfolio_summary(path)


def model_comparison(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return model rows ordered by the prespecified primary metric."""
    rows = summary.get("model_comparison")
    if not isinstance(rows, list) or not rows:
        raise AgentToolError("Model comparison evidence is unavailable.")
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: float(row["macro_f1_mean"]),
        reverse=True,
    )


def selected_model(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return the selected development model metadata."""
    selection = summary.get("selection")
    if not isinstance(selection, Mapping):
        raise AgentToolError("Model-selection evidence is unavailable.")
    return dict(selection)


def calibration_evidence(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return aggregate probability-calibration evidence."""
    calibration = summary.get("calibration")
    if not isinstance(calibration, Mapping):
        raise AgentToolError("Calibration evidence is unavailable.")
    return dict(calibration)


def explainability_evidence(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return aggregate explainability evidence only."""
    explainability = summary.get("explainability")
    if not isinstance(explainability, Mapping):
        raise AgentToolError("Explainability evidence is unavailable.")
    return dict(explainability)


def final_test_gate_names(validation_path: str | Path) -> list[str]:
    """Return protocol-defined gates required before final-test access."""
    config = load_validation_config(validation_path)
    return list(config["final_test"]["required_gates"])


def release_readiness(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Audit portfolio-release gates using public aggregate evidence only."""
    disclosure = summary.get("disclosure")
    workflow = summary.get("workflow")
    selection = summary.get("selection")
    calibration = summary.get("calibration")
    explainability = summary.get("explainability")

    if not all(
        isinstance(value, Mapping)
        for value in (disclosure, workflow, selection, calibration, explainability)
    ):
        raise AgentToolError("Release-readiness evidence is incomplete.")

    checks = [
        {
            "name": "group_disjoint_partitions",
            "passed": workflow.get("group_disjoint_partitions") is True,
            "detail": "Development, calibration and locked-test partitions are PSU-disjoint.",
        },
        {
            "name": "fold_local_preprocessing",
            "passed": workflow.get("fold_local_preprocessing") is True,
            "detail": "Preprocessing is fitted inside training folds.",
        },
        {
            "name": "model_selected",
            "passed": bool(selection.get("model_name")),
            "detail": "A development model is selected by the prespecified primary metric.",
        },
        {
            "name": "calibration_completed",
            "passed": calibration.get("accepted") is True,
            "detail": "Probability calibration completed on the calibration partition.",
        },
        {
            "name": "aggregate_explainability",
            "passed": (
                explainability.get("scope") == "aggregate_only"
                and explainability.get("row_level_values_persisted") is False
            ),
            "detail": "Explainability output is aggregate-only.",
        },
        {
            "name": "locked_test_untouched",
            "passed": disclosure.get("locked_test_evaluated") is False,
            "detail": "The final test remains untouched in the public development release.",
        },
        {
            "name": "no_final_claim",
            "passed": disclosure.get("final_performance_claim_allowed") is False,
            "detail": "Public artifacts do not claim confirmatory performance.",
        },
    ]

    portfolio_ready = all(bool(check["passed"]) for check in checks)
    return {
        "portfolio_release_ready": portfolio_ready,
        "confirmatory_release_ready": False,
        "checks": checks,
        "next_action": (
            "Keep the portfolio release frozen and continue confirmatory research separately."
            if portfolio_ready
            else "Fix failed public-release checks before publishing or redeploying."
        ),
    }


def next_experiment_plan(
    summary: Mapping[str, Any],
    validation_path: str | Path,
) -> list[dict[str, str]]:
    """Return the next protocol-safe research milestones in execution order."""
    disclosure = summary.get("disclosure")
    calibration = summary.get("calibration")
    if not isinstance(disclosure, Mapping) or not isinstance(calibration, Mapping):
        raise AgentToolError("Experiment-planning evidence is incomplete.")

    if disclosure.get("locked_test_evaluated") is not False:
        raise AgentToolError("Public planning cannot operate on evaluated locked-test results.")

    gates = final_test_gate_names(validation_path)
    gate_text = ", ".join(gates)

    plan = [
        {
            "step": "1",
            "action": "Complete state-held-out validation",
            "why": "Test geographic transportability before any final-test access.",
        },
        {
            "step": "2",
            "action": "Freeze model, calibration and evaluation plan",
            "why": "Prevent post-hoc tuning after confirmatory evidence is viewed.",
        },
        {
            "step": "3",
            "action": "Verify all final-test gates",
            "why": f"Required protocol gates: {gate_text}.",
        },
        {
            "step": "4",
            "action": "Run the single-use locked-test evaluation",
            "why": "Only after every gate is independently complete and explicitly authorized.",
        },
        {
            "step": "5",
            "action": "Quantify uncertainty and robustness",
            "why": "Run PSU-within-strata bootstrap intervals and subgroup/geographic checks.",
        },
        {
            "step": "6",
            "action": "Finalize manuscript and release evidence",
            "why": "Report confirmatory results without changing the frozen model afterward.",
        },
    ]

    if calibration.get("accepted") is not True:
        plan.insert(
            0,
            {
                "step": "0",
                "action": "Complete calibration selection",
                "why": "Calibration must be frozen before confirmatory evaluation.",
            },
        )
    return plan
