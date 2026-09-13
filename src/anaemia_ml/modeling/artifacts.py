"""Integrity-checked persistence for trusted fitted model pipelines."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

ARTIFACT_VERSION = "1.0.0"
MANIFEST_FILENAME = "manifest.json"
FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {
        "group_ids",
        "groups",
        "pipeline",
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


class ArtifactError(ValueError):
    """Raised when a model artifact is invalid, unsafe, or corrupt."""


class ArtifactIdentityError(ArtifactError):
    """Raised when an artifact belongs to another experiment."""


class UntrustedArtifactError(ArtifactError):
    """Raised before deserializing an artifact not marked as trusted."""


@dataclass(frozen=True)
class ModelArtifactManifest:
    """JSON-safe identity and integrity metadata for one fitted pipeline."""

    artifact_version: str
    model_file: str
    model_sha256: str
    experiment_fingerprint: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical manifest document."""
        return {
            "artifact_version": self.artifact_version,
            "model_file": self.model_file,
            "model_sha256": self.model_sha256,
            "experiment_fingerprint": self.experiment_fingerprint,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LoadedModelArtifact:
    """A verified manifest and its trusted deserialized pipeline."""

    manifest: ModelArtifactManifest
    pipeline: Pipeline


def _non_empty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"{name} must be a non-empty string.")
    return value


def _json_round_trip(value: Any, *, name: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"{name} must contain JSON-safe finite values.") from error


def _forbidden_keys(value: Any) -> set[str]:
    found = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.casefold() in FORBIDDEN_ARTIFACT_KEYS:
                found.add(key)
            found.update(_forbidden_keys(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found.update(_forbidden_keys(nested))
    return found


def _validated_payload(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{name} must be a mapping.")
    forbidden = sorted(_forbidden_keys(value))
    if forbidden:
        raise ArtifactError(f"{name} contains forbidden row-level keys: {forbidden}.")
    normalized = _json_round_trip(dict(value), name=name)
    if not isinstance(normalized, dict):
        raise ArtifactError(f"{name} must be a mapping.")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ArtifactError(f"Could not read artifact file: {path}") from error
    return digest.hexdigest()


def _check_fitted_pipeline(pipeline: Any) -> Pipeline:
    if not isinstance(pipeline, Pipeline):
        raise ArtifactError("pipeline must be a sklearn Pipeline.")
    try:
        check_is_fitted(pipeline)
    except (NotFittedError, TypeError) as error:
        raise ArtifactError("pipeline must be fitted before it is saved.") from error
    return pipeline


def _atomic_json_write(
    path: Path,
    document: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise ArtifactError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(document, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ArtifactError(f"Could not write JSON artifact: {path}") from error


def write_json_artifact(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write an aggregate-only JSON artifact."""
    artifact_path = Path(path)
    normalized = _validated_payload(payload, name="payload")
    _atomic_json_write(artifact_path, normalized, overwrite=overwrite)
    return artifact_path


def _write_pipeline_file(directory: Path, pipeline: Pipeline) -> tuple[str, str]:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=directory,
            prefix=".model.",
            suffix=".joblib.tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        joblib.dump(pipeline, temporary_path, compress=3, protocol=5)
        with temporary_path.open("rb+") as stream:
            os.fsync(stream.fileno())
        digest = _sha256_file(temporary_path)
        filename = f"model-{digest[:16]}.joblib"
        final_path = directory / filename
        if final_path.exists() and _sha256_file(final_path) != digest:
            raise ArtifactError(f"Artifact filename collision: {final_path}")
        os.replace(temporary_path, final_path)
        return filename, digest
    except ArtifactError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    except Exception as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ArtifactError("Could not serialize the fitted pipeline.") from error


def save_model_artifact(
    directory: str | Path,
    pipeline: Pipeline,
    *,
    experiment_fingerprint: str,
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> ModelArtifactManifest:
    """Persist a fitted pipeline and atomically publish its manifest."""
    checked_pipeline = _check_fitted_pipeline(pipeline)
    checked_fingerprint = _non_empty_text(
        experiment_fingerprint,
        name="experiment_fingerprint",
    )
    checked_metadata = _validated_payload(metadata, name="metadata")
    artifact_directory = Path(directory)
    if artifact_directory.exists() and not artifact_directory.is_dir():
        raise ArtifactError(
            f"Artifact directory is not a directory: {artifact_directory}"
        )
    artifact_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_directory / MANIFEST_FILENAME
    if manifest_path.exists() and not overwrite:
        raise ArtifactError(f"Refusing to overwrite existing artifact: {manifest_path}")

    model_file, digest = _write_pipeline_file(artifact_directory, checked_pipeline)
    manifest = ModelArtifactManifest(
        artifact_version=ARTIFACT_VERSION,
        model_file=model_file,
        model_sha256=digest,
        experiment_fingerprint=checked_fingerprint,
        metadata=checked_metadata,
    )
    _atomic_json_write(manifest_path, manifest.as_dict(), overwrite=overwrite)
    return manifest


def load_model_manifest(directory: str | Path) -> ModelArtifactManifest:
    """Load and structurally validate an artifact manifest without deserializing."""
    manifest_path = Path(directory) / MANIFEST_FILENAME
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(
            f"Could not read artifact manifest: {manifest_path}"
        ) from error
    if not isinstance(document, dict):
        raise ArtifactError("Artifact manifest root must be a mapping.")
    expected_keys = {
        "artifact_version",
        "model_file",
        "model_sha256",
        "experiment_fingerprint",
        "metadata",
    }
    if set(document) != expected_keys:
        raise ArtifactError("Artifact manifest fields are incomplete or unknown.")
    if document["artifact_version"] != ARTIFACT_VERSION:
        raise ArtifactError("Unsupported artifact version.")
    model_file = _non_empty_text(document["model_file"], name="model_file")
    if Path(model_file).name != model_file or not model_file.endswith(".joblib"):
        raise ArtifactError("model_file must be a safe joblib filename.")
    model_sha256 = _non_empty_text(document["model_sha256"], name="model_sha256")
    if len(model_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in model_sha256
    ):
        raise ArtifactError("model_sha256 must be a lowercase SHA-256 digest.")
    return ModelArtifactManifest(
        artifact_version=ARTIFACT_VERSION,
        model_file=model_file,
        model_sha256=model_sha256,
        experiment_fingerprint=_non_empty_text(
            document["experiment_fingerprint"],
            name="experiment_fingerprint",
        ),
        metadata=_validated_payload(document["metadata"], name="metadata"),
    )


def load_model_artifact(
    directory: str | Path,
    *,
    trusted: bool = False,
    expected_experiment_fingerprint: str | None = None,
) -> LoadedModelArtifact:
    """Verify and deserialize a locally trusted fitted pipeline.

    Joblib uses pickle internally, so callers must explicitly confirm that the
    artifact directory comes from a trusted source. Integrity checks detect
    corruption; they do not make untrusted pickle content safe.
    """
    if trusted is not True:
        raise UntrustedArtifactError(
            "Refusing to deserialize joblib without trusted=True."
        )
    artifact_directory = Path(directory)
    manifest = load_model_manifest(artifact_directory)
    if (
        expected_experiment_fingerprint is not None
        and manifest.experiment_fingerprint != expected_experiment_fingerprint
    ):
        raise ArtifactIdentityError(
            "Artifact experiment fingerprint does not match the expected experiment."
        )
    model_path = artifact_directory / manifest.model_file
    if _sha256_file(model_path) != manifest.model_sha256:
        raise ArtifactError("Artifact checksum does not match the manifest.")
    try:
        pipeline = joblib.load(model_path)
    except Exception as error:
        raise ArtifactError(
            "Could not deserialize the trusted model artifact."
        ) from error
    return LoadedModelArtifact(
        manifest=manifest,
        pipeline=_check_fitted_pipeline(pipeline),
    )
