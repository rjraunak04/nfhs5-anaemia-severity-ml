"""Tests for development-only grouped nested cross-validation."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from anaemia_ml.modeling import (
    NestedCVError,
    default_parameter_candidates,
    run_grouped_nested_cv,
)
from anaemia_ml.modeling.checkpoints import (
    build_experiment_identity,
    save_outer_fold_checkpoint,
)
from anaemia_ml.modeling.registry import ModelRegistryError
from anaemia_ml.preprocessing import predictor_groups


@pytest.fixture
def feature_schema() -> dict:
    return {
        "feature_groups": {
            "continuous_numeric": ["age", "bmi"],
            "count_numeric": ["children"],
            "nominal_categorical": ["region"],
            "indicator_categorical": ["rural"],
            "ordinal_categorical": ["wealth"],
        },
        "model_variants": {
            "india_policy": {
                "excluded_features": [],
                "expected_predictor_count": 6,
            }
        },
        "preprocessing_contract": {
            "fit_scope": "training_fold_only",
            "numeric": {
                "imputation": "median",
                "add_missing_indicator": True,
                "scaling": "model_dependent",
            },
            "categorical": {
                "imputation": "explicit_missing_category",
                "unknown_category_policy": "handle_without_failure",
                "linear_model_encoding": "one_hot",
                "tree_model_encoding": "model_appropriate",
                "verify_ordinal_order_from_official_labels": True,
            },
            "engineered_features": {
                "primary_analysis": [],
                "post_hoc_features_allowed": False,
            },
        },
        "leakage_guards": {
            "forbidden_predictors": ["haemoglobin"],
            "preprocessing_before_split_allowed": False,
        },
    }


@pytest.fixture
def validation_config() -> dict:
    return {
        "validation_version": "1.0.0",
        "random_seed": 42,
        "grouping": {
            "strategy": "composite_psu",
            "columns": ["v024", "v021"],
        },
        "partitions": {
            "development": 0.70,
            "calibration": 0.10,
            "locked_test": 0.20,
        },
        "nested_cv": {
            "splitter": "StratifiedGroupKFold",
            "outer_folds": 2,
            "inner_folds": 2,
            "shuffle": True,
            "scope": "development_only",
        },
        "model_selection": {
            "primary_metric": "macro_f1",
            "optuna_trials_per_outer_fold": 3,
            "calibration_during_nested_cv": False,
        },
        "preprocessing": {
            "fit_inside_training_fold_only": True,
            "primary_imbalance_strategy": "class_weight",
            "smotenc_sensitivity_ratio": 0.45,
        },
        "final_test": {
            "locked_by_default": True,
            "unlock_environment_variable": "ANAEMIA_UNLOCK_FINAL_TEST",
            "required_gates": [
                "feature_schema_frozen",
                "search_budget_frozen",
                "nested_cv_complete",
                "state_heldout_complete",
                "calibration_frozen",
                "model_frozen",
                "evaluation_plan_frozen",
            ],
        },
        "privacy": {
            "allow_raw_rows_in_repository": False,
            "allow_real_psu_ids_in_repository": False,
            "allow_split_indices_in_repository": False,
        },
    }


@pytest.fixture
def development_data(feature_schema: dict) -> tuple:
    target = np.tile(np.arange(4), 16)
    groups = np.repeat(np.arange(16), 4)
    row_number = np.arange(target.size)
    columns = predictor_groups(feature_schema)
    values = {}

    for position, column in enumerate(columns.numeric, start=1):
        values[column] = target * position + groups * 0.01 + row_number % 2
    for position, column in enumerate(columns.categorical, start=1):
        values[column] = ((target + position) % 4).astype(str)

    frame = pd.DataFrame(values)
    weights = 1.0 + (groups % 5) * 0.1
    return frame, target, groups, weights


def test_default_candidates_are_fresh_copies() -> None:
    first = default_parameter_candidates("logistic_regression")
    second = default_parameter_candidates("logistic_regression")

    first[0]["C"] = 999

    assert second[0]["C"] == 0.1


def test_nested_cv_produces_auditable_json_report(
    development_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    frame, target, groups, _ = development_data
    report = run_grouped_nested_cv(
        "logistic_regression",
        frame,
        target,
        groups,
        feature_schema,
        validation_config,
        parameter_candidates=({"C": 0.5}, {"C": 1.0}),
    )

    assert len(report.outer_fold_results) == 2
    assert report.total_model_fits == 10
    assert report.primary_metric == "macro_f1"
    assert len(report.aggregate_metrics) == 12
    for fold in report.outer_fold_results:
        assert len(fold.candidate_scores) == 2
        assert len(fold.candidate_scores[0].inner_macro_f1) == 2
        assert fold.selected_candidate_number in {1, 2}

    payload = report.summary()
    json.dumps(payload, allow_nan=False)
    rendered = json.dumps(payload)
    assert "train_indices" not in rendered
    assert "validation_indices" not in rendered
    assert "pipeline" not in rendered


def test_nested_cv_is_reproducible(
    development_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    frame, target, groups, _ = development_data
    arguments = (
        "logistic_regression",
        frame,
        target,
        groups,
        feature_schema,
        validation_config,
    )
    first = run_grouped_nested_cv(*arguments, parameter_candidates=({"C": 1.0},))
    second = run_grouped_nested_cv(*arguments, parameter_candidates=({"C": 1.0},))

    assert first.summary() == second.summary()


def test_nested_cv_resumes_completed_outer_folds(
    development_data: tuple,
    feature_schema: dict,
    validation_config: dict,
    tmp_path,
    monkeypatch,
) -> None:
    frame, target, groups, _ = development_data
    candidates = ({"C": 0.5}, {"C": 1.0})
    baseline = run_grouped_nested_cv(
        "logistic_regression",
        frame,
        target,
        groups,
        feature_schema,
        validation_config,
        parameter_candidates=candidates,
    )
    identity = build_experiment_identity(
        experiment_id="resume-test",
        dataset_fingerprint="dataset",
        validation_config=validation_config,
        feature_schema=feature_schema,
        search_space={"candidate_count": 2},
        code_version="test-commit",
    )
    checkpoint = tmp_path / "logistic-regression.json"
    save_outer_fold_checkpoint(
        checkpoint,
        identity,
        baseline.outer_fold_results[0].summary(),
    )

    import anaemia_ml.modeling.nested_cv as nested_cv_module

    original = nested_cv_module.fit_evaluate_model
    fit_calls = 0

    def counted_fit(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(nested_cv_module, "fit_evaluate_model", counted_fit)
    resumed = run_grouped_nested_cv(
        "logistic_regression",
        frame,
        target,
        groups,
        feature_schema,
        validation_config,
        parameter_candidates=candidates,
        checkpoint_path=checkpoint,
        checkpoint_identity=identity,
    )

    assert resumed.summary() == baseline.summary()
    assert fit_calls == 5


def test_checkpoint_arguments_must_be_provided_together(
    development_data: tuple,
    feature_schema: dict,
    validation_config: dict,
    tmp_path,
) -> None:
    frame, target, groups, _ = development_data

    with pytest.raises(NestedCVError, match="provided together"):
        run_grouped_nested_cv(
            "logistic_regression",
            frame,
            target,
            groups,
            feature_schema,
            validation_config,
            parameter_candidates=({"C": 1.0},),
            checkpoint_path=tmp_path / "orphan.json",
        )


def test_survey_weights_are_used_without_being_reported(
    development_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    frame, target, groups, weights = development_data
    report = run_grouped_nested_cv(
        "logistic_regression",
        frame,
        target,
        groups,
        feature_schema,
        validation_config,
        parameter_candidates=({"C": 1.0},),
        sample_weight=weights,
    )

    payload = report.summary()
    assert np.isfinite([metric["mean"] for metric in payload["aggregate_metrics"].values()]).all()
    assert "sample_weight" not in json.dumps(payload)


def test_search_budget_is_enforced_before_fitting(
    development_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    frame, target, groups, _ = development_data
    config = copy.deepcopy(validation_config)
    config["model_selection"]["optuna_trials_per_outer_fold"] = 1

    with pytest.raises(NestedCVError, match="search budget"):
        run_grouped_nested_cv(
            "logistic_regression",
            frame,
            target,
            groups,
            feature_schema,
            config,
            parameter_candidates=({"C": 0.5}, {"C": 1.0}),
        )


def test_duplicate_candidates_are_rejected(
    development_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    frame, target, groups, _ = development_data

    with pytest.raises(NestedCVError, match="duplicates"):
        run_grouped_nested_cv(
            "logistic_regression",
            frame,
            target,
            groups,
            feature_schema,
            validation_config,
            parameter_candidates=({"C": 1.0}, {"C": 1.0}),
        )


def test_reproducibility_parameters_cannot_be_tuned(
    development_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    frame, target, groups, _ = development_data

    with pytest.raises(ModelRegistryError, match="cannot be overridden"):
        run_grouped_nested_cv(
            "logistic_regression",
            frame,
            target,
            groups,
            feature_schema,
            validation_config,
            parameter_candidates=({"random_state": 99},),
        )


def test_mismatched_target_length_is_rejected(
    development_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    frame, target, groups, _ = development_data

    with pytest.raises(NestedCVError, match="64 values"):
        run_grouped_nested_cv(
            "logistic_regression",
            frame,
            target[:-1],
            groups,
            feature_schema,
            validation_config,
            parameter_candidates=({"C": 1.0},),
        )


def test_non_positive_survey_weight_is_rejected(
    development_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    frame, target, groups, weights = development_data
    weights[0] = 0

    with pytest.raises(NestedCVError, match="finite and positive"):
        run_grouped_nested_cv(
            "logistic_regression",
            frame,
            target,
            groups,
            feature_schema,
            validation_config,
            parameter_candidates=({"C": 1.0},),
            sample_weight=weights,
        )
