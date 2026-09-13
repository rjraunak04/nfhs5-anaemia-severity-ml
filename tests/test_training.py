"""Tests for the one-command development-only training workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anaemia_ml.data.validate import load_contract
from anaemia_ml.evaluation.config import load_validation_config
from anaemia_ml.features.schema import feature_columns, load_feature_schema
from anaemia_ml.modeling.artifacts import ArtifactError, load_model_artifact
from anaemia_ml.modeling.search_space import load_search_space
from anaemia_ml.training import (
    TrainingWorkflowError,
    main,
    make_synthetic_smoke_frame,
    prepare_development_data,
    run_development_training,
)

ROOT = Path(__file__).parents[1]
CONTRACT_PATH = ROOT / "configs" / "data_contract.yaml"
FEATURE_PATH = ROOT / "configs" / "feature_schema.yaml"
VALIDATION_PATH = ROOT / "configs" / "validation.yaml"
SEARCH_SPACE_PATH = ROOT / "configs" / "search_space.yaml"


@pytest.fixture
def configs() -> tuple[dict, dict, dict, dict]:
    contract = load_contract(CONTRACT_PATH)
    schema = load_feature_schema(FEATURE_PATH, contract_path=CONTRACT_PATH)
    validation = load_validation_config(VALIDATION_PATH)
    search_space = load_search_space(SEARCH_SPACE_PATH)
    validation["nested_cv"]["outer_folds"] = 2
    validation["nested_cv"]["inner_folds"] = 2
    validation["model_selection"]["optuna_trials_per_outer_fold"] = 1
    return contract, schema, validation, search_space


def test_synthetic_smoke_frame_matches_frozen_column_contract(configs) -> None:
    contract, _, _, _ = configs

    frame = make_synthetic_smoke_frame(contract)

    assert frame.shape == (120, 40)
    assert frame.columns.tolist() == contract["expected_column_order"]
    assert set(frame["v457"]) == {1, 2, 3, 4}
    assert not frame.isna().any().any()


def test_preparation_keeps_restricted_columns_out_of_predictors(configs) -> None:
    contract, schema, validation, _ = configs
    frame = make_synthetic_smoke_frame(contract)

    prepared = prepare_development_data(
        frame,
        contract,
        schema,
        validation,
        strict_profile=False,
    )

    assert prepared.predictors.columns.tolist() == feature_columns(schema)
    assert not set(contract["never_use_as_predictors"]) & set(
        prepared.predictors.columns
    )
    assert set(prepared.target) == {0, 1, 2, 3}
    assert sum(
        indices.size for indices in prepared.partitions.as_dict().values()
    ) == len(frame)
    group_sets = [
        set(prepared.groups[indices])
        for indices in prepared.partitions.as_dict().values()
    ]
    assert not group_sets[0] & group_sets[1]
    assert not group_sets[0] & group_sets[2]
    assert not group_sets[1] & group_sets[2]


def test_one_command_smoke_run_writes_report_and_loadable_model(
    tmp_path,
    capsys,
) -> None:
    output = tmp_path / "day-one-smoke"

    exit_code = main(
        [
            "--smoke",
            "--contract",
            str(CONTRACT_PATH),
            "--features",
            str(FEATURE_PATH),
            "--validation",
            str(VALIDATION_PATH),
            "--search-space",
            str(SEARCH_SPACE_PATH),
            "--output-directory",
            str(output),
            "--code-version",
            "test-commit",
        ]
    )

    command_summary = json.loads(capsys.readouterr().out)
    report = json.loads((output / "development_report.json").read_text())
    loaded = load_model_artifact(output / "model", trusted=True)
    contract = load_contract(CONTRACT_PATH)
    schema = load_feature_schema(FEATURE_PATH, contract_path=CONTRACT_PATH)
    smoke_frame = make_synthetic_smoke_frame(contract)

    assert exit_code == 0
    assert command_summary["run_kind"] == "synthetic_smoke"
    assert command_summary["locked_test_evaluated"] is False
    assert report["data_validation"]["passed"] is True
    assert [model["model_name"] for model in report["models"]] == [
        "logistic_regression",
        "random_forest",
        "lightgbm",
    ]
    assert report["selection"]["selected_model_name"] in {
        "logistic_regression",
        "random_forest",
        "lightgbm",
    }
    assert report["final_performance_claim_allowed"] is False
    assert loaded.manifest.experiment_fingerprint == report["experiment_fingerprint"]
    predictions = loaded.pipeline.predict(smoke_frame[feature_columns(schema)])
    assert predictions.shape == (120,)
    serialized = json.dumps(report)
    for forbidden in (
        "train_indices",
        "validation_indices",
        "predictions",
        "probabilities",
        '"pipeline"',
    ):
        assert forbidden not in serialized


def test_workflow_rejects_duplicate_models_before_training(configs, tmp_path) -> None:
    contract, schema, validation, search_space = configs

    with pytest.raises(TrainingWorkflowError, match="duplicates"):
        run_development_training(
            make_synthetic_smoke_frame(contract),
            contract,
            schema,
            validation,
            search_space,
            output_directory=tmp_path,
            dataset_fingerprint="synthetic-sha256",
            code_version="test-commit",
            experiment_id="test-run",
            model_names=("logistic_regression", "logistic_regression"),
            strict_profile=False,
        )


def test_workflow_refuses_to_replace_existing_output(configs, tmp_path) -> None:
    contract, schema, validation, search_space = configs
    output = tmp_path / "existing"
    output.mkdir()
    (output / "development_report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactError, match="overwrite"):
        run_development_training(
            make_synthetic_smoke_frame(contract),
            contract,
            schema,
            validation,
            search_space,
            output_directory=output,
            dataset_fingerprint="synthetic-sha256",
            code_version="test-commit",
            experiment_id="test-run",
            model_names=("logistic_regression",),
            parameter_candidates={"logistic_regression": ({"C": 1.0},)},
            run_kind="synthetic_smoke",
            strict_profile=False,
        )


def test_workflow_rejects_invalid_run_kind(configs, tmp_path) -> None:
    contract, schema, validation, search_space = configs

    with pytest.raises(TrainingWorkflowError, match="run_kind"):
        run_development_training(
            make_synthetic_smoke_frame(contract),
            contract,
            schema,
            validation,
            search_space,
            output_directory=tmp_path,
            dataset_fingerprint="synthetic-sha256",
            code_version="test-commit",
            experiment_id="test-run",
            model_names=("logistic_regression",),
            run_kind="invalid",
            strict_profile=False,
        )


def test_all_available_cores_are_an_allowed_execution_setting(
    configs,
    tmp_path,
) -> None:
    contract, schema, validation, search_space = configs
    result = run_development_training(
        make_synthetic_smoke_frame(contract),
        contract,
        schema,
        validation,
        search_space,
        output_directory=tmp_path,
        dataset_fingerprint="synthetic-sha256",
        code_version="test-commit",
        experiment_id="test-run",
        model_names=("logistic_regression",),
        parameter_candidates={"logistic_regression": ({"C": 1.0},)},
        run_kind="synthetic_smoke",
        strict_profile=False,
        n_jobs=-1,
    )

    assert result.selected_model_name == "logistic_regression"


def test_synthetic_generator_is_deterministic(configs) -> None:
    contract, _, _, _ = configs

    first = make_synthetic_smoke_frame(contract)
    second = make_synthetic_smoke_frame(contract)

    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
