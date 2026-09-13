"""Tests for integrity-checked fitted-pipeline artifacts."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from anaemia_ml.modeling.artifacts import (
    ArtifactError,
    ArtifactIdentityError,
    UntrustedArtifactError,
    load_model_artifact,
    load_model_manifest,
    save_model_artifact,
    write_json_artifact,
)


def _fitted_pipeline():
    features = np.arange(48, dtype=float).reshape(16, 3)
    target = np.repeat(np.arange(4), 4)
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500),
    ).fit(features, target)


def test_fitted_pipeline_round_trip_is_verified(tmp_path) -> None:
    pipeline = _fitted_pipeline()
    directory = tmp_path / "model"
    manifest = save_model_artifact(
        directory,
        pipeline,
        experiment_fingerprint="experiment-sha256",
        metadata={
            "model_name": "logistic_regression",
            "locked_test_evaluated": False,
        },
    )

    loaded = load_model_artifact(
        directory,
        expected_experiment_fingerprint="experiment-sha256",
        trusted=True,
    )

    assert loaded.manifest == manifest
    np.testing.assert_array_equal(
        loaded.pipeline.predict(np.arange(12, dtype=float).reshape(4, 3)),
        pipeline.predict(np.arange(12, dtype=float).reshape(4, 3)),
    )
    json.dumps(manifest.as_dict(), allow_nan=False)


def test_deserialization_requires_explicit_trust(tmp_path) -> None:
    with pytest.raises(UntrustedArtifactError, match="trusted"):
        load_model_artifact(tmp_path / "does-not-matter")


def test_unfitted_pipeline_cannot_be_saved(tmp_path) -> None:
    pipeline = make_pipeline(StandardScaler(), LogisticRegression())

    with pytest.raises(ArtifactError, match="fitted"):
        save_model_artifact(
            tmp_path / "model",
            pipeline,
            experiment_fingerprint="experiment-sha256",
            metadata={"model_name": "logistic_regression"},
        )


def test_tampered_model_file_is_rejected(tmp_path) -> None:
    directory = tmp_path / "model"
    manifest = save_model_artifact(
        directory,
        _fitted_pipeline(),
        experiment_fingerprint="experiment-sha256",
        metadata={"model_name": "logistic_regression"},
    )
    model_path = directory / manifest.model_file
    with model_path.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ArtifactError, match="checksum"):
        load_model_artifact(directory, trusted=True)


def test_experiment_identity_mismatch_is_rejected(tmp_path) -> None:
    directory = tmp_path / "model"
    save_model_artifact(
        directory,
        _fitted_pipeline(),
        experiment_fingerprint="first-experiment",
        metadata={"model_name": "logistic_regression"},
    )

    with pytest.raises(ArtifactIdentityError, match="another|does not match"):
        load_model_artifact(
            directory,
            expected_experiment_fingerprint="second-experiment",
            trusted=True,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"predictions": [0, 1]},
        {"nested": {"train_indices": [0, 1]}},
        {"metrics": {"macro_f1": float("nan")}},
    ],
)
def test_unsafe_metadata_is_rejected(tmp_path, metadata) -> None:
    with pytest.raises(ArtifactError, match="forbidden|JSON-safe finite"):
        save_model_artifact(
            tmp_path / "model",
            _fitted_pipeline(),
            experiment_fingerprint="experiment-sha256",
            metadata=metadata,
        )


def test_existing_artifact_is_not_overwritten_by_default(tmp_path) -> None:
    directory = tmp_path / "model"
    save_model_artifact(
        directory,
        _fitted_pipeline(),
        experiment_fingerprint="experiment-sha256",
        metadata={"model_name": "logistic_regression"},
    )

    with pytest.raises(ArtifactError, match="overwrite"):
        save_model_artifact(
            directory,
            _fitted_pipeline(),
            experiment_fingerprint="experiment-sha256",
            metadata={"model_name": "logistic_regression"},
        )

    assert load_model_manifest(directory).experiment_fingerprint == "experiment-sha256"


def test_json_artifact_is_atomic_and_aggregate_only(tmp_path) -> None:
    path = write_json_artifact(
        tmp_path / "report.json",
        {"metrics": {"macro_f1": 0.75}},
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "metrics": {"macro_f1": 0.75}
    }
    with pytest.raises(ArtifactError, match="overwrite"):
        write_json_artifact(path, {"metrics": {"macro_f1": 0.80}})
