"""End-to-end tests for the Day 2 aggregate workflow."""

from __future__ import annotations

import json
from pathlib import Path

from anaemia_ml.data.validate import load_contract
from anaemia_ml.day2 import make_day2_smoke_frame, run_day2_workflow
from anaemia_ml.evaluation.config import load_validation_config
from anaemia_ml.features.schema import feature_columns, load_feature_schema
from anaemia_ml.modeling.artifacts import load_model_artifact
from anaemia_ml.modeling.checkpoints import fingerprint_json
from anaemia_ml.modeling.search_space import load_search_space
from anaemia_ml.training import make_synthetic_smoke_frame

ROOT = Path(__file__).parents[1]
CONTRACT_PATH = ROOT / "configs" / "data_contract.yaml"
FEATURE_PATH = ROOT / "configs" / "feature_schema.yaml"
VALIDATION_PATH = ROOT / "configs" / "validation.yaml"
SEARCH_SPACE_PATH = ROOT / "configs" / "search_space.yaml"


def test_day2_demo_fixture_is_deterministic_and_breaks_direct_signal() -> None:
    contract = load_contract(CONTRACT_PATH)
    schema = load_feature_schema(FEATURE_PATH, contract_path=CONTRACT_PATH)
    original = make_synthetic_smoke_frame(contract)

    first = make_day2_smoke_frame(contract, schema)
    second = make_day2_smoke_frame(contract, schema)

    assert first.equals(second)
    assert first.columns.tolist() == contract["expected_column_order"]
    assert not first["v012"].equals(original["v012"])
    assert first["v457"].equals(original["v457"])
    assert first["v021"].equals(original["v021"])


def test_day2_workflow_writes_safe_loadable_outputs(tmp_path, monkeypatch) -> None:
    contract = load_contract(CONTRACT_PATH)
    schema = load_feature_schema(FEATURE_PATH, contract_path=CONTRACT_PATH)
    validation = load_validation_config(VALIDATION_PATH)
    validation["nested_cv"]["outer_folds"] = 2
    validation["nested_cv"]["inner_folds"] = 2
    validation["model_selection"]["optuna_trials_per_outer_fold"] = 1
    search_space = load_search_space(SEARCH_SPACE_PATH)
    frame = make_synthetic_smoke_frame(contract)

    def aggregate_only_report(*args, **kwargs):
        del args, kwargs
        return {
            "method": "SHAP",
            "scope": "aggregate_only",
            "partition": "calibration",
            "sample_count": 12,
            "background_count": 6,
            "variant": "india_policy",
            "interpretation": "associational_not_causal",
            "row_level_values_persisted": False,
            "locked_test_evaluated": False,
            "global_features": [
                {
                    "feature": "v012",
                    "mean_abs_shap": 0.2,
                    "mean_signed_shap": 0.0,
                    "rank": 1,
                }
            ],
            "class_features": [],
        }

    monkeypatch.setattr("anaemia_ml.day2.build_shap_report", aggregate_only_report)
    output = tmp_path / "day2"
    result = run_day2_workflow(
        frame,
        contract,
        schema,
        validation,
        search_space,
        output_directory=output,
        dataset_fingerprint=fingerprint_json({"synthetic": True}),
        code_version="test-commit",
        experiment_id="day2-test",
        model_names=("logistic_regression",),
        parameter_candidates={"logistic_regression": ({"C": 1.0},)},
        run_kind="synthetic_smoke",
        strict_profile=False,
    )

    summary = json.loads(result.portfolio_summary_path.read_text())
    loaded = load_model_artifact(output / "calibrated_model", trusted=True)

    assert result.summary()["locked_test_evaluated"] is False
    assert loaded.pipeline.predict(frame[feature_columns(schema)]).shape == (120,)
    assert summary["result_status"] == "synthetic_engineering_demo"
    assert summary["workflow"]["group_disjoint_partitions"] is True
    assert summary["disclosure"]["final_performance_claim_allowed"] is False
    assert summary["explainability"]["row_level_values_persisted"] is False
    serialized = json.dumps(summary)
    for forbidden in ("train_indices", "predictions", "probabilities", "y_true"):
        assert f'"{forbidden}"' not in serialized
