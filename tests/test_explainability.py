"""Tests for privacy-preserving aggregate SHAP reporting."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from anaemia_ml.data.validate import load_contract
from anaemia_ml.evaluation.explainability import (
    ExplainabilityError,
    aggregate_shap_values,
    build_shap_report,
)
from anaemia_ml.features.schema import feature_columns, load_feature_schema
from anaemia_ml.preprocessing import build_model_pipeline
from anaemia_ml.training import make_synthetic_smoke_frame

ROOT = Path(__file__).parents[1]
CONTRACT_PATH = ROOT / "configs" / "data_contract.yaml"
FEATURE_PATH = ROOT / "configs" / "feature_schema.yaml"


def test_transformed_columns_are_aggregated_to_raw_features() -> None:
    values = np.array(
        [
            [[1.0, -1.0], [2.0, -2.0], [4.0, -4.0]],
            [[3.0, -3.0], [4.0, -4.0], [2.0, -2.0]],
        ]
    )

    report = aggregate_shap_values(
        values,
        ("age", "wealth", "wealth"),
        (0, 1),
        sample_count=2,
        top_n=2,
    )

    assert [row["feature"] for row in report["global_features"]] == ["wealth", "age"]
    assert report["global_features"][0]["mean_abs_shap"] == 3.0
    assert len(report["class_features"]) == 2


def test_shap_report_never_returns_row_level_values() -> None:
    contract = load_contract(CONTRACT_PATH)
    schema = load_feature_schema(FEATURE_PATH, contract_path=CONTRACT_PATH)
    frame = make_synthetic_smoke_frame(contract)
    predictors = frame[feature_columns(schema)]
    target = frame["v457"].map({1: 3, 2: 2, 3: 1, 4: 0}).to_numpy()
    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=500),
        schema,
        sparse_output=False,
    ).fit(predictors, target)

    def fake_factory(estimator, background):
        del estimator
        feature_count = background.shape[1]

        def explain(values):
            shape = (values.shape[0], feature_count, 4)
            return SimpleNamespace(values=np.ones(shape, dtype=float))

        return explain

    report = build_shap_report(
        pipeline,
        predictors.iloc[:16],
        schema,
        max_samples=12,
        background_size=6,
        top_n=10,
        explainer_factory=fake_factory,
    )

    assert report["scope"] == "aggregate_only"
    assert report["partition"] == "calibration"
    assert report["row_level_values_persisted"] is False
    assert report["locked_test_evaluated"] is False
    assert report["sample_count"] == 12
    assert len(report["global_features"]) == 10
    assert "values" not in report

    with pytest.raises(ExplainabilityError, match="locked test remains sealed"):
        build_shap_report(
            pipeline,
            predictors.iloc[:16],
            schema,
            partition_name="locked_test",
            explainer_factory=fake_factory,
        )
