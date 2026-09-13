"""Integration tests for persistent bounded Optuna tuning."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from anaemia_ml.modeling.search_space import load_search_space
from anaemia_ml.modeling.tuning import TuningError, tune_parameters

ROOT = Path(__file__).parents[1]


@pytest.fixture
def search_space() -> dict:
    return load_search_space(ROOT / "configs" / "search_space.yaml")


def test_optuna_study_resumes_only_remaining_trials(tmp_path, search_space) -> None:
    pytest.importorskip("optuna")
    evaluations = []

    def evaluate(parameters) -> float:
        evaluations.append(dict(parameters))
        return max(0.0, 1.0 - abs(math.log10(parameters["C"])) / 10)

    arguments = {
        "model_name": "logistic_regression",
        "evaluate_parameters": evaluate,
        "search_space": search_space,
        "random_seed": 42,
        "storage_path": tmp_path / "study.sqlite3",
        "study_name": "outer-fold-1",
        "experiment_fingerprint": "experiment-sha256",
    }
    first = tune_parameters(**arguments, trial_budget=2)
    resumed = tune_parameters(**arguments, trial_budget=4)

    assert first.total_trials == 2
    assert resumed.total_trials == 4
    assert resumed.completed_trials == 4
    assert len(evaluations) == 4
    assert 0 <= resumed.best_value <= 1
    json.dumps(resumed.summary(), allow_nan=False)


def test_optuna_study_rejects_identity_mismatch(tmp_path, search_space) -> None:
    pytest.importorskip("optuna")
    arguments = {
        "model_name": "logistic_regression",
        "evaluate_parameters": lambda parameters: 0.5,
        "search_space": search_space,
        "trial_budget": 1,
        "random_seed": 42,
        "storage_path": tmp_path / "study.sqlite3",
        "study_name": "outer-fold-1",
    }
    tune_parameters(**arguments, experiment_fingerprint="first-experiment")

    with pytest.raises(TuningError, match="does not match"):
        tune_parameters(**arguments, experiment_fingerprint="second-experiment")


def test_unknown_model_is_rejected_before_storage_creation(
    tmp_path, search_space
) -> None:
    pytest.importorskip("optuna")
    storage_path = tmp_path / "study.sqlite3"

    with pytest.raises(TuningError, match="Unknown model"):
        tune_parameters(
            "unknown-model",
            lambda parameters: 0.5,
            search_space,
            trial_budget=1,
            random_seed=42,
            storage_path=storage_path,
            study_name="invalid-model",
            experiment_fingerprint="experiment-sha256",
        )

    assert not storage_path.exists()


@pytest.mark.parametrize("score", [float("nan"), -0.1, 1.1, "invalid"])
def test_invalid_objective_score_is_rejected(score, tmp_path, search_space) -> None:
    pytest.importorskip("optuna")

    with pytest.raises(TuningError, match="macro-F1"):
        tune_parameters(
            "logistic_regression",
            lambda parameters: score,
            search_space,
            trial_budget=1,
            random_seed=42,
            storage_path=tmp_path / "study.sqlite3",
            study_name="invalid-score",
            experiment_fingerprint="experiment-sha256",
        )
