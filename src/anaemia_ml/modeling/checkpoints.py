"""Atomic, identity-checked checkpoints for completed outer CV folds."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_VERSION = "1.0.0"
FORBIDDEN_CHECKPOINT_KEYS = frozenset(
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
    }
)


class CheckpointError(ValueError):
    """Raised when a checkpoint is malformed or privacy-unsafe."""


class CheckpointIdentityError(CheckpointError):
    """Raised when a checkpoint belongs to another experiment."""


@dataclass(frozen=True)
class ExperimentIdentity:
    """Inputs that must match before saved work may be reused."""

    experiment_id: str
    dataset_fingerprint: str
    validation_fingerprint: str
    feature_schema_fingerprint: str
    search_space_fingerprint: str
    code_version: str

    def __post_init__(self) -> None:
        invalid = sorted(
            field_name
            for field_name, value in self.as_dict().items()
            if not isinstance(value, str) or not value.strip()
        )
        if invalid:
            raise CheckpointError(
                f"Identity fields must be non-empty strings: {invalid}."
            )

    def as_dict(self) -> dict[str, str]:
        """Return identity fields in stable dataclass order."""
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        """Return a canonical SHA-256 identity fingerprint."""
        return fingerprint_json(self.as_dict())


@dataclass(frozen=True)
class CheckpointSnapshot:
    """Validated summaries recovered from one checkpoint file."""

    identity: ExperimentIdentity
    outer_fold_summaries: tuple[dict[str, Any], ...]

    @property
    def completed_outer_folds(self) -> tuple[int, ...]:
        """Return completed fold numbers in ascending order."""
        return tuple(summary["fold_number"] for summary in self.outer_fold_summaries)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe checkpoint document."""
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "identity": self.identity.as_dict(),
            "identity_fingerprint": self.identity.fingerprint,
            "outer_folds": [dict(summary) for summary in self.outer_fold_summaries],
        }


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
        raise CheckpointError(
            f"{name} must contain JSON-safe finite values."
        ) from error


def fingerprint_json(value: Any) -> str:
    """Return a canonical SHA-256 fingerprint for JSON-safe content."""
    normalized = _json_round_trip(value, name="Fingerprint input")
    canonical = json.dumps(
        normalized,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_experiment_identity(
    *,
    experiment_id: str,
    dataset_fingerprint: str,
    validation_config: Mapping[str, Any],
    feature_schema: Mapping[str, Any],
    search_space: Mapping[str, Any],
    code_version: str,
) -> ExperimentIdentity:
    """Build a complete identity without retaining restricted inputs."""
    if not isinstance(validation_config, Mapping):
        raise CheckpointError("validation_config must be a mapping.")
    if not isinstance(feature_schema, Mapping):
        raise CheckpointError("feature_schema must be a mapping.")
    if not isinstance(search_space, Mapping):
        raise CheckpointError("search_space must be a mapping.")
    return ExperimentIdentity(
        experiment_id=experiment_id,
        dataset_fingerprint=dataset_fingerprint,
        validation_fingerprint=fingerprint_json(dict(validation_config)),
        feature_schema_fingerprint=fingerprint_json(dict(feature_schema)),
        search_space_fingerprint=fingerprint_json(dict(search_space)),
        code_version=code_version,
    )


def _forbidden_keys(value: Any) -> set[str]:
    found = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.casefold() in FORBIDDEN_CHECKPOINT_KEYS:
                found.add(key)
            found.update(_forbidden_keys(nested))
    elif isinstance(value, list | tuple):
        for nested in value:
            found.update(_forbidden_keys(nested))
    return found


def _validated_fold_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        raise CheckpointError("outer_fold_summary must be a mapping.")
    forbidden = sorted(_forbidden_keys(summary))
    if forbidden:
        raise CheckpointError(
            f"Checkpoint contains forbidden row-level keys: {forbidden}."
        )
    normalized = _json_round_trip(dict(summary), name="outer_fold_summary")
    fold_number = normalized.get("fold_number")
    if (
        isinstance(fold_number, bool)
        or not isinstance(fold_number, int)
        or fold_number < 1
    ):
        raise CheckpointError(
            "outer_fold_summary.fold_number must be a positive integer."
        )
    return normalized


def _identity_from_mapping(value: Any) -> ExperimentIdentity:
    identity = _json_round_trip(value, name="checkpoint identity")
    if not isinstance(identity, dict):
        raise CheckpointError("checkpoint identity must be a mapping.")
    expected = {
        "experiment_id",
        "dataset_fingerprint",
        "validation_fingerprint",
        "feature_schema_fingerprint",
        "search_space_fingerprint",
        "code_version",
    }
    if set(identity) != expected:
        raise CheckpointError("checkpoint identity fields are incomplete or unknown.")
    if any(not isinstance(value, str) or not value for value in identity.values()):
        raise CheckpointError("checkpoint identity values must be non-empty strings.")
    return ExperimentIdentity(**identity)


def empty_checkpoint(identity: ExperimentIdentity) -> CheckpointSnapshot:
    """Create an in-memory checkpoint with no completed folds."""
    if not isinstance(identity, ExperimentIdentity):
        raise CheckpointError("identity must be an ExperimentIdentity.")
    return CheckpointSnapshot(identity=identity, outer_fold_summaries=())


def load_checkpoint(
    path: str | Path,
    expected_identity: ExperimentIdentity,
) -> CheckpointSnapshot | None:
    """Load a checkpoint, rejecting stale or incompatible experiments."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None
    if not checkpoint_path.is_file():
        raise CheckpointError(f"Checkpoint path is not a file: {checkpoint_path}")
    try:
        document = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(
            f"Could not read checkpoint: {checkpoint_path}"
        ) from error
    if not isinstance(document, dict):
        raise CheckpointError("Checkpoint root must be a mapping.")
    if document.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise CheckpointError("Unsupported checkpoint version.")

    identity = _identity_from_mapping(document.get("identity"))
    stored_fingerprint = document.get("identity_fingerprint")
    if stored_fingerprint != identity.fingerprint:
        raise CheckpointError("Checkpoint identity fingerprint is corrupt.")
    if identity != expected_identity:
        raise CheckpointIdentityError(
            "Checkpoint identity does not match the current experiment inputs."
        )

    raw_folds = document.get("outer_folds")
    if not isinstance(raw_folds, list):
        raise CheckpointError("Checkpoint outer_folds must be a list.")
    folds = tuple(_validated_fold_summary(summary) for summary in raw_folds)
    numbers = [summary["fold_number"] for summary in folds]
    if len(numbers) != len(set(numbers)):
        raise CheckpointError("Checkpoint contains duplicate outer fold numbers.")
    ordered = tuple(sorted(folds, key=lambda summary: summary["fold_number"]))
    return CheckpointSnapshot(identity=identity, outer_fold_summaries=ordered)


def _atomic_write(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
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
        raise CheckpointError(f"Could not write checkpoint: {path}") from error


def save_outer_fold_checkpoint(
    path: str | Path,
    identity: ExperimentIdentity,
    outer_fold_summary: Mapping[str, Any],
) -> CheckpointSnapshot:
    """Atomically append one completed outer-fold summary."""
    checkpoint_path = Path(path)
    existing = load_checkpoint(checkpoint_path, identity)
    snapshot = empty_checkpoint(identity) if existing is None else existing
    validated = _validated_fold_summary(outer_fold_summary)
    fold_number = validated["fold_number"]
    summaries = {
        summary["fold_number"]: summary for summary in snapshot.outer_fold_summaries
    }
    if fold_number in summaries:
        if summaries[fold_number] != validated:
            raise CheckpointError(
                f"Outer fold {fold_number} already has a different checkpoint."
            )
        return snapshot
    summaries[fold_number] = validated
    updated = CheckpointSnapshot(
        identity=identity,
        outer_fold_summaries=tuple(summaries[number] for number in sorted(summaries)),
    )
    _atomic_write(checkpoint_path, updated.summary())
    return updated
