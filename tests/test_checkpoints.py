"""Tests for privacy-safe atomic outer-fold checkpoints."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from anaemia_ml.modeling.checkpoints import (
    CheckpointError,
    CheckpointIdentityError,
    ExperimentIdentity,
    build_experiment_identity,
    fingerprint_json,
    load_checkpoint,
    save_outer_fold_checkpoint,
)


@pytest.fixture
def identity():
    return build_experiment_identity(
        experiment_id="anaemia-primary-v1",
        dataset_fingerprint="dataset-sha256",
        validation_config={"random_seed": 42},
        feature_schema={"schema_version": "1.0.0"},
        search_space={"search_space_version": "1.0.0"},
        code_version="abcdef123456",
    )


def _fold_summary(number: int, score: float) -> dict:
    return {
        "fold_number": number,
        "selected_parameters": {"C": 1.0},
        "metrics": {"macro_f1": score},
    }


def test_json_fingerprint_is_order_independent() -> None:
    assert fingerprint_json({"a": 1, "b": 2}) == fingerprint_json({"b": 2, "a": 1})


def test_direct_identity_construction_rejects_empty_fields() -> None:
    with pytest.raises(CheckpointError, match="non-empty strings"):
        ExperimentIdentity(
            experiment_id="",
            dataset_fingerprint="dataset-sha256",
            validation_fingerprint="validation-sha256",
            feature_schema_fingerprint="feature-sha256",
            search_space_fingerprint="search-sha256",
            code_version="abcdef123456",
        )


def test_completed_folds_can_be_saved_and_resumed(tmp_path, identity) -> None:
    path = tmp_path / "outer-folds.json"

    first = save_outer_fold_checkpoint(path, identity, _fold_summary(2, 0.62))
    second = save_outer_fold_checkpoint(path, identity, _fold_summary(1, 0.58))
    loaded = load_checkpoint(path, identity)

    assert first.completed_outer_folds == (2,)
    assert second.completed_outer_folds == (1, 2)
    assert loaded is not None
    assert loaded.completed_outer_folds == (1, 2)
    assert json.loads(path.read_text(encoding="utf-8")) == loaded.summary()


def test_saving_the_same_fold_is_idempotent(tmp_path, identity) -> None:
    path = tmp_path / "outer-folds.json"
    summary = _fold_summary(1, 0.58)

    save_outer_fold_checkpoint(path, identity, summary)
    resumed = save_outer_fold_checkpoint(path, identity, summary)

    assert resumed.completed_outer_folds == (1,)


def test_conflicting_completed_fold_is_rejected(tmp_path, identity) -> None:
    path = tmp_path / "outer-folds.json"
    save_outer_fold_checkpoint(path, identity, _fold_summary(1, 0.58))

    with pytest.raises(CheckpointError, match="already"):
        save_outer_fold_checkpoint(path, identity, _fold_summary(1, 0.99))


def test_stale_experiment_identity_is_rejected(tmp_path, identity) -> None:
    path = tmp_path / "outer-folds.json"
    save_outer_fold_checkpoint(path, identity, _fold_summary(1, 0.58))
    changed = replace(identity, dataset_fingerprint="different-dataset")

    with pytest.raises(
        CheckpointIdentityError, match="another experiment|does not match"
    ):
        load_checkpoint(path, changed)


@pytest.mark.parametrize(
    "forbidden_key",
    ["train_indices", "groups", "predictions", "sample_weight", "pipeline"],
)
def test_row_level_values_cannot_be_checkpointed(
    forbidden_key: str,
    tmp_path,
    identity,
) -> None:
    unsafe = _fold_summary(1, 0.58)
    unsafe["nested"] = {forbidden_key: [1, 2, 3]}

    with pytest.raises(CheckpointError, match="forbidden row-level"):
        save_outer_fold_checkpoint(tmp_path / "unsafe.json", identity, unsafe)


def test_non_finite_metric_is_rejected(tmp_path, identity) -> None:
    with pytest.raises(CheckpointError, match="JSON-safe finite"):
        save_outer_fold_checkpoint(
            tmp_path / "invalid.json",
            identity,
            _fold_summary(1, float("nan")),
        )


def test_corrupt_checkpoint_is_rejected(tmp_path, identity) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(CheckpointError, match="Could not read"):
        load_checkpoint(path, identity)
