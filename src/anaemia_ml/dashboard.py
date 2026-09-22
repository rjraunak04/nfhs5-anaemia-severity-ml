"""Validation and loading for aggregate-only recruiter dashboard data."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

FORBIDDEN_DASHBOARD_KEYS = frozenset(
    {
        "caseid",
        "group_ids",
        "groups",
        "indices",
        "predictions",
        "probabilities",
        "psu",
        "psu_id",
        "raw_rows",
        "sample_weight",
        "train_indices",
        "validation_indices",
        "y_proba",
        "y_true",
    }
)
MAX_DASHBOARD_BYTES = 1_000_000


class DashboardDataError(ValueError):
    """Raised when dashboard data are incomplete or contain row-level fields."""


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.casefold() in FORBIDDEN_DASHBOARD_KEYS:
                found.add(key)
            found.update(_forbidden_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_forbidden_keys(nested))
    return found


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def validate_portfolio_summary(value: Any) -> dict[str, Any]:
    """Validate the public aggregate schema and reject row-level content."""
    if not isinstance(value, Mapping):
        raise DashboardDataError("Dashboard JSON root must be an object.")
    document = dict(value)
    forbidden = sorted(_forbidden_keys(document))
    if forbidden:
        raise DashboardDataError(f"Dashboard JSON contains forbidden fields: {forbidden}.")
    required = {
        "schema_version",
        "project_title",
        "run_kind",
        "result_status",
        "disclosure",
        "workflow",
        "model_comparison",
        "selection",
        "calibration",
        "explainability",
        "responsible_use",
    }
    missing = sorted(required - set(document))
    if missing:
        raise DashboardDataError(f"Dashboard JSON is missing fields: {missing}.")
    disclosure = document["disclosure"]
    if not isinstance(disclosure, Mapping):
        raise DashboardDataError("disclosure must be an object.")
    if disclosure.get("locked_test_evaluated") is not False:
        raise DashboardDataError("Recruiter dashboard must not claim locked-test evaluation.")
    if disclosure.get("final_performance_claim_allowed") is not False:
        raise DashboardDataError("Final performance claims are not allowed before Day 3 gates.")
    models = document["model_comparison"]
    if not isinstance(models, list) or not models:
        raise DashboardDataError("model_comparison must contain at least one aggregate row.")
    for model in models:
        if not isinstance(model, Mapping):
            raise DashboardDataError("Every model comparison row must be an object.")
        for key in ("model_name", "display_name", "macro_f1_mean"):
            if key not in model:
                raise DashboardDataError(f"Model comparison row is missing {key}.")
    explainability = document["explainability"]
    if not isinstance(explainability, Mapping):
        raise DashboardDataError("explainability must be an object.")
    if explainability.get("scope") != "aggregate_only":
        raise DashboardDataError("Explainability scope must be aggregate_only.")
    if explainability.get("row_level_values_persisted") is not False:
        raise DashboardDataError("Row-level explainability values must not be persisted.")
    if not isinstance(explainability.get("global_features"), list):
        raise DashboardDataError("explainability.global_features must be a list.")
    return document


def portfolio_summary_from_bytes(payload: bytes) -> dict[str, Any]:
    """Decode a size-limited aggregate dashboard upload."""
    if len(payload) > MAX_DASHBOARD_BYTES:
        raise DashboardDataError("Dashboard JSON must be no larger than 1 MB.")
    try:
        document = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise DashboardDataError("Dashboard upload must be valid UTF-8 JSON.") from error
    return validate_portfolio_summary(document)


def load_portfolio_summary(path: str | Path) -> dict[str, Any]:
    """Load and validate a local aggregate dashboard summary."""
    summary_path = Path(path)
    try:
        payload = summary_path.read_bytes()
    except OSError as error:
        raise DashboardDataError(f"Could not read dashboard summary: {summary_path}") from error
    return portfolio_summary_from_bytes(payload)
