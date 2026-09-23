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
