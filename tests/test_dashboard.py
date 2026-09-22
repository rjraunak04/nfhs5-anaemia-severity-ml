"""Tests for the aggregate-only Streamlit dashboard boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anaemia_ml.dashboard import (
    DashboardDataError,
    load_portfolio_summary,
    portfolio_summary_from_bytes,
    validate_portfolio_summary,
)

ROOT = Path(__file__).parents[1]
DEMO_PATH = ROOT / "demo" / "portfolio_summary.json"


def test_bundled_demo_passes_dashboard_schema() -> None:
    data = load_portfolio_summary(DEMO_PATH)

    assert data["result_status"] == "synthetic_engineering_demo"
    assert data["disclosure"]["locked_test_evaluated"] is False
    assert data["explainability"]["scope"] == "aggregate_only"


def test_dashboard_rejects_row_level_fields() -> None:
    data = load_portfolio_summary(DEMO_PATH)
    data["debug"] = {"predictions": [0, 1]}

    with pytest.raises(DashboardDataError, match="forbidden"):
        validate_portfolio_summary(data)


def test_dashboard_rejects_a_final_performance_claim() -> None:
    data = load_portfolio_summary(DEMO_PATH)
    data["disclosure"]["final_performance_claim_allowed"] = True

    with pytest.raises(DashboardDataError, match="claims"):
        validate_portfolio_summary(data)


def test_dashboard_upload_has_a_strict_size_limit() -> None:
    with pytest.raises(DashboardDataError, match="1 MB"):
        portfolio_summary_from_bytes(b" " * 1_000_001)


def test_dashboard_upload_rejects_non_finite_json_numbers() -> None:
    with pytest.raises(DashboardDataError, match="valid UTF-8 JSON"):
        portfolio_summary_from_bytes(b'{"metric": NaN}')


def test_demo_json_contains_no_known_row_level_key() -> None:
    serialized = json.dumps(load_portfolio_summary(DEMO_PATH)).casefold()

    for key in ("y_true", "y_proba", "predictions", "probabilities", "psu_id"):
        assert f'"{key}"' not in serialized
