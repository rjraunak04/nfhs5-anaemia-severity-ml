"""One-command, development-only training workflow for anaemia models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from anaemia_ml.data.validate import (
    ValidationReport,
    load_contract,
    read_dataset,
    validate_contract_definition,
    validate_dataframe,
)
from anaemia_ml.evaluation.config import (
    load_validation_config,
    validate_validation_config,
)
from anaemia_ml.evaluation.partitions import GroupedPartitions, make_grouped_partitions
from anaemia_ml.features.schema import (
    feature_columns,
    load_feature_schema,
    validate_feature_schema,
)
from anaemia_ml.modeling.artifacts import (
    MANIFEST_FILENAME,
    ArtifactError,
    ModelArtifactManifest,
    save_model_artifact,
    write_json_artifact,
)
from anaemia_ml.modeling.checkpoints import (
    ExperimentIdentity,
    build_experiment_identity,
    fingerprint_json,
)
from anaemia_ml.modeling.imbalance import build_primary_imbalance_classifier
from anaemia_ml.modeling.nested_cv import NestedCVReport, run_grouped_nested_cv
from anaemia_ml.modeling.registry import build_estimator, model_spec, registered_models
from anaemia_ml.modeling.search_space import load_search_space, validate_search_space
from anaemia_ml.preprocessing import build_model_pipeline

WORKFLOW_VERSION = "1.0.0"
REPORT_FILENAME = "development_report.json"
DEFAULT_TRAINING_MODELS = tuple(spec.name for spec in registered_models())
RunKind = Literal["research_development", "synthetic_smoke"]


class TrainingWorkflowError(ValueError):
    """Raised when the one-command workflow would be unsafe or incomplete."""


@dataclass(frozen=True)
class PreparedDevelopmentData:
    """Validated in-memory data; never serialize this row-level container."""

    predictors: pd.DataFrame
    target: np.ndarray
    groups: np.ndarray
    sample_weight: np.ndarray
    partitions: GroupedPartitions
    validation_report: ValidationReport


@dataclass(frozen=True)
class DevelopmentTrainingResult:
    """Aggregate result and artifact identity from a development-only run."""

    run_kind: RunKind
    identity: ExperimentIdentity
    selected_model_name: str
    selected_parameters: dict[str, Any]
    model_reports: tuple[NestedCVReport, ...]
    artifact_manifest: ModelArtifactManifest
    report_path: Path

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-safe command result."""
        return {
            "workflow_version": WORKFLOW_VERSION,
            "run_kind": self.run_kind,
            "scope": "development_only",
            "experiment_fingerprint": self.identity.fingerprint,
            "selected_model_name": self.selected_model_name,
            "selected_parameters": dict(self.selected_parameters),
            "model_sha256": self.artifact_manifest.model_sha256,
            "report_file": self.report_path.name,
            "locked_test_evaluated": False,
        }


def _non_empty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingWorkflowError(f"{name} must be a non-empty string.")
    return value


def _positive_integer(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise TrainingWorkflowError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _validated_n_jobs(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) == 0
        or int(value) < -1
    ):
        raise TrainingWorkflowError("n_jobs must be -1 or a positive integer.")
    return int(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise TrainingWorkflowError(f"Could not fingerprint dataset: {path}") from error
    return digest.hexdigest()


def _validated_models(model_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(model_names, (str, bytes)):
        raise TrainingWorkflowError("model_names must be a sequence of model names.")
    names = tuple(model_names)
    if not names:
        raise TrainingWorkflowError("model_names must not be empty.")
    resolved = tuple(model_spec(name).name for name in names)
    if len(resolved) != len(set(resolved)):
        raise TrainingWorkflowError("model_names must not contain duplicates.")
    return resolved


def _target_values(frame: pd.DataFrame, contract: Mapping[str, Any]) -> np.ndarray:
    target_settings = contract["target"]
    raw_column = target_settings["raw_column"]
    mapping = target_settings["raw_to_model_mapping"]
    mapped = frame[raw_column].map(mapping)
    if mapped.isna().any():
        raise TrainingWorkflowError("Target mapping produced missing model labels.")
    target = mapped.to_numpy(dtype=int)
    expected = set(target_settings["ordered_classes"])
    if set(target.tolist()) != expected:
        raise TrainingWorkflowError("Training data must contain every prespecified target class.")
    return target


def _composite_groups(
    frame: pd.DataFrame,
    validation_config: Mapping[str, Any],
) -> np.ndarray:
    columns = validation_config["grouping"]["columns"]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise TrainingWorkflowError(f"Grouping columns are missing: {missing}.")
    group_index = pd.MultiIndex.from_frame(frame.loc[:, columns])
    codes, _ = pd.factorize(group_index, sort=True)
    if (codes < 0).any():
        raise TrainingWorkflowError("Composite PSU groups must not contain missing values.")
    return codes.astype(np.int64, copy=False)


def _survey_weights(frame: pd.DataFrame, contract: Mapping[str, Any]) -> np.ndarray:
    survey = contract["survey_design"]
    column = survey["weight_column"]
    divisor = survey["weight_divisor"]
    if isinstance(divisor, bool) or not isinstance(divisor, int | float) or divisor <= 0:
        raise TrainingWorkflowError("survey_design.weight_divisor must be positive.")
    try:
        weights = frame[column].to_numpy(dtype=float) / float(divisor)
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingWorkflowError("Survey weights must be numeric.") from error
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise TrainingWorkflowError("Survey weights must be finite and positive.")
    return weights


def prepare_development_data(
    frame: pd.DataFrame,
    data_contract: Mapping[str, Any],
    feature_schema: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    *,
    variant: str = "india_policy",
    strict_profile: bool = True,
) -> PreparedDevelopmentData:
    """Validate raw input and construct group-disjoint research partitions."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise TrainingWorkflowError("frame must be a non-empty pandas DataFrame.")
    if not isinstance(data_contract, Mapping):
        raise TrainingWorkflowError("data_contract must be a mapping.")
    if not isinstance(feature_schema, Mapping):
        raise TrainingWorkflowError("feature_schema must be a mapping.")
    if not isinstance(validation_config, Mapping):
        raise TrainingWorkflowError("validation_config must be a mapping.")

    contract = dict(data_contract)
    schema = dict(feature_schema)
    config = dict(validation_config)
    validate_contract_definition(contract)
    validate_feature_schema(schema, contract)
    validate_validation_config(config)
    report = validate_dataframe(frame, contract, strict_profile=strict_profile)
    report.raise_for_errors()

    target = _target_values(frame, contract)
    groups = _composite_groups(frame, config)
    weights = _survey_weights(frame, contract)
    columns = feature_columns(schema, variant=variant)
    predictors = frame.loc[:, columns].copy()
    fractions = config["partitions"]
    partitions = make_grouped_partitions(
        target,
        groups,
        development_fraction=float(fractions["development"]),
        calibration_fraction=float(fractions["calibration"]),
        locked_test_fraction=float(fractions["locked_test"]),
        random_state=int(config["random_seed"]),
    )
    return PreparedDevelopmentData(
        predictors=predictors,
        target=target,
        groups=groups,
        sample_weight=weights,
        partitions=partitions,
        validation_report=report,
    )


def _macro_f1(report: NestedCVReport) -> float:
    for aggregate in report.aggregate_metrics:
        if aggregate.name == "macro_f1":
            return aggregate.mean
    raise TrainingWorkflowError("Nested CV report is missing macro_f1.")


def _consensus_parameters(report: NestedCVReport) -> dict[str, Any]:
    candidates: dict[int, tuple[dict[str, Any], list[float]]] = {}
    for fold in report.outer_fold_results:
        for score in fold.candidate_scores:
            existing = candidates.get(score.candidate_number)
            if existing is None:
                candidates[score.candidate_number] = (
                    dict(score.parameters),
                    [score.mean_macro_f1],
                )
            else:
                parameters, values = existing
                if parameters != score.parameters:
                    raise TrainingWorkflowError("Candidate parameters changed between outer folds.")
                values.append(score.mean_macro_f1)
    if not candidates:
        raise TrainingWorkflowError("Nested CV report contains no candidates.")
    selected_number = max(
        candidates,
        key=lambda number: (
            float(np.mean(candidates[number][1])),
            -number,
        ),
    )
    return dict(candidates[selected_number][0])


def _fit_development_pipeline(
    model_name: str,
    parameters: Mapping[str, Any],
    prepared: PreparedDevelopmentData,
    feature_schema: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    *,
    variant: str,
    n_jobs: int,
) -> Pipeline:
    development = prepared.partitions.development
    spec = model_spec(model_name)
    estimator = build_estimator(
        spec.name,
        random_seed=int(validation_config["random_seed"]),
        n_jobs=n_jobs,
        parameters=parameters,
    )
    classifier = build_primary_imbalance_classifier(estimator, validation_config)
    pipeline = build_model_pipeline(
        classifier,
        dict(feature_schema),
        variant=variant,
        model_family=spec.family,
        sparse_output=spec.sparse_output,
    )
    pipeline.fit(
        prepared.predictors.iloc[development],
        prepared.target[development],
        model__sample_weight=prepared.sample_weight[development],
    )
    return pipeline


def _validated_candidate_map(
    model_names: Sequence[str],
    parameter_candidates: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> dict[str, Sequence[Mapping[str, Any]] | None]:
    if parameter_candidates is None:
        return {name: None for name in model_names}
    if not isinstance(parameter_candidates, Mapping):
        raise TrainingWorkflowError("parameter_candidates must be a mapping.")
    unknown = sorted(set(parameter_candidates) - set(model_names))
    if unknown:
        raise TrainingWorkflowError(f"parameter_candidates contains unrequested models: {unknown}.")
    return {name: parameter_candidates.get(name) for name in model_names}


def _model_checkpoint_identity(
    identity: ExperimentIdentity,
    *,
    model_name: str,
    variant: str,
) -> ExperimentIdentity:
    """Bind an experiment identity to one model and feature variant."""
    return ExperimentIdentity(
        experiment_id=(f"{identity.experiment_id}:model={model_name}:variant={variant}"),
        dataset_fingerprint=identity.dataset_fingerprint,
        validation_fingerprint=identity.validation_fingerprint,
        feature_schema_fingerprint=identity.feature_schema_fingerprint,
        search_space_fingerprint=identity.search_space_fingerprint,
        code_version=identity.code_version,
    )


def _preflight_output(output_directory: Path, *, overwrite: bool) -> None:
    if output_directory.exists() and not output_directory.is_dir():
        raise ArtifactError(f"output_directory is not a directory: {output_directory}")
    targets = (
        output_directory / REPORT_FILENAME,
        output_directory / "model" / MANIFEST_FILENAME,
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        raise ArtifactError(f"Refusing to overwrite existing workflow output: {existing}.")


def run_development_training(
    frame: pd.DataFrame,
    data_contract: Mapping[str, Any],
    feature_schema: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    search_space: Mapping[str, Any],
    *,
    output_directory: str | Path,
    dataset_fingerprint: str,
    code_version: str,
    experiment_id: str,
    model_names: Sequence[str] = DEFAULT_TRAINING_MODELS,
    parameter_candidates: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    variant: str = "india_policy",
    run_kind: RunKind = "research_development",
    strict_profile: bool = True,
    n_jobs: int = 1,
    overwrite: bool = False,
) -> DevelopmentTrainingResult:
    """Run nested development evaluation, refit its winner, and save it."""
    if run_kind not in {"research_development", "synthetic_smoke"}:
        raise TrainingWorkflowError(f"Unsupported run_kind: {run_kind!r}.")
    if not isinstance(strict_profile, bool):
        raise TrainingWorkflowError("strict_profile must be boolean.")
    if not isinstance(overwrite, bool):
        raise TrainingWorkflowError("overwrite must be boolean.")
    checked_fingerprint = _non_empty_text(
        dataset_fingerprint,
        name="dataset_fingerprint",
    )
    checked_code_version = _non_empty_text(code_version, name="code_version")
    checked_experiment_id = _non_empty_text(experiment_id, name="experiment_id")
    checked_jobs = _validated_n_jobs(n_jobs)
    names = _validated_models(model_names)
    candidates = _validated_candidate_map(names, parameter_candidates)
    if not isinstance(search_space, Mapping):
        raise TrainingWorkflowError("search_space must be a mapping.")
    validate_search_space(search_space)
    output_path = Path(output_directory)
    _preflight_output(output_path, overwrite=overwrite)
    identity = build_experiment_identity(
        experiment_id=checked_experiment_id,
        dataset_fingerprint=checked_fingerprint,
        validation_config=validation_config,
        feature_schema=feature_schema,
        search_space=search_space,
        code_version=checked_code_version,
    )

    prepared = prepare_development_data(
        frame,
        data_contract,
        feature_schema,
        validation_config,
        variant=variant,
        strict_profile=strict_profile,
    )
    development = prepared.partitions.development
    reports = []
    for name in names:
        reports.append(
            run_grouped_nested_cv(
                name,
                prepared.predictors.iloc[development],
                prepared.target[development],
                prepared.groups[development],
                feature_schema,
                validation_config,
                parameter_candidates=candidates[name],
                sample_weight=prepared.sample_weight[development],
                variant=variant,
                n_jobs=checked_jobs,
                checkpoint_path=output_path / "checkpoints" / f"{name}.json",
                checkpoint_identity=_model_checkpoint_identity(
                    identity,
                    model_name=name,
                    variant=variant,
                ),
            )
        )
    reports = tuple(reports)
    selected_index = max(
        range(len(reports)),
        key=lambda index: (_macro_f1(reports[index]), -index),
    )
    selected_report = reports[selected_index]
    selected_parameters = _consensus_parameters(selected_report)
    pipeline = _fit_development_pipeline(
        selected_report.model_name,
        selected_parameters,
        prepared,
        feature_schema,
        validation_config,
        variant=variant,
        n_jobs=checked_jobs,
    )
    artifact_metadata = {
        "workflow_version": WORKFLOW_VERSION,
        "artifact_role": "development_candidate",
        "run_kind": run_kind,
        "scope": "development_only",
        "model_name": selected_report.model_name,
        "display_name": selected_report.display_name,
        "variant": variant,
        "selected_parameters": selected_parameters,
        "primary_metric": selected_report.primary_metric,
        "development_training_rows": int(development.size),
        "calibration_evaluated": False,
        "locked_test_evaluated": False,
        "clinical_use_allowed": False,
    }
    artifact_manifest = save_model_artifact(
        output_path / "model",
        pipeline,
        experiment_fingerprint=identity.fingerprint,
        metadata=artifact_metadata,
        overwrite=overwrite,
    )
    partition_sizes = {
        name: int(indices.size) for name, indices in prepared.partitions.as_dict().items()
    }
    report_payload = {
        "workflow_version": WORKFLOW_VERSION,
        "run_kind": run_kind,
        "scope": "development_only",
        "experiment_id": identity.experiment_id,
        "experiment_fingerprint": identity.fingerprint,
        "dataset_fingerprint": checked_fingerprint,
        "variant": variant,
        "partition_rows": partition_sizes,
        "data_validation": prepared.validation_report.to_dict(),
        "models": [report.summary() for report in reports],
        "selection": {
            "primary_metric": "macro_f1",
            "selected_model_name": selected_report.model_name,
            "selected_parameters": selected_parameters,
            "selected_development_macro_f1": _macro_f1(selected_report),
        },
        "model_artifact": artifact_manifest.as_dict(),
        "resumability": {
            "granularity": "completed_outer_fold",
            "atomic_checkpoint_writes": True,
            "row_level_values_persisted": False,
        },
        "calibration_evaluated": False,
        "locked_test_evaluated": False,
        "final_performance_claim_allowed": False,
    }
    report_path = write_json_artifact(
        output_path / REPORT_FILENAME,
        report_payload,
        overwrite=overwrite,
    )
    return DevelopmentTrainingResult(
        run_kind=run_kind,
        identity=identity,
        selected_model_name=selected_report.model_name,
        selected_parameters=selected_parameters,
        model_reports=reports,
        artifact_manifest=artifact_manifest,
        report_path=report_path,
    )


def make_synthetic_smoke_frame(
    data_contract: Mapping[str, Any],
    *,
    group_count: int = 30,
) -> pd.DataFrame:
    """Create deterministic contract-shaped data for engineering smoke tests."""
    groups = _positive_integer(group_count, name="group_count", minimum=20)
    row_count = groups * 4
    row = np.arange(row_count)
    group = row // 4
    model_target = row % 4
    values: dict[str, Any] = {
        "v002": group + 1,
        "v003": model_target + 1,
        "v005": 900_000 + (group % 7) * 20_000,
        "v021": 10_000 + group,
        "v022": 1 + group % 8,
        "v024": 1 + group % 6,
        "sdist": 1 + group % 12,
        "v012": 20 + model_target * 6 + group % 3,
        "v025": 1 + row % 2,
        "v501": 1 + model_target % 3,
        "v130": 1 + group % 4,
        "s116": 1 + group % 4,
        "v190": 1 + group % 5,
        "v133": 4 + model_target * 3 + group % 2,
        "v201": model_target + group % 3,
        "v208": row % 3,
        "v212": 17 + model_target + group % 4,
        "v213": row % 2,
        "v228": (row + 1) % 2,
        "v312": 1 + row % 4,
        "v404": row % 2,
        "v405": (row + group) % 2,
        "v445": 1_800 + model_target * 180 + group % 40,
        "v467b": 1 + row % 3,
        "v467c": 1 + (row + 1) % 3,
        "v467d": 1 + (row + 2) % 3,
        "v467f": 1 + group % 3,
        "v113": 1 + group % 5,
        "v116": 1 + group % 4,
        "v161": 1 + group % 3,
        "v481": row % 2,
        "s728a": (row + 1) % 2,
        "s731b": 1 + row % 4,
        "s731c": 1 + (row + 1) % 4,
        "s731d": 1 + (row + 2) % 4,
        "s731e": 1 + (row + 3) % 4,
        "s731f": 1 + group % 4,
        "s731g": 1 + (group + 1) % 4,
        "v456": 125 - model_target * 18 + group % 3,
        "v457": 4 - model_target,
    }
    expected = list(data_contract["expected_column_order"])
    missing = sorted(set(expected) - set(values))
    unknown = sorted(set(values) - set(expected))
    if missing or unknown:
        raise TrainingWorkflowError(
            "Synthetic generator differs from the data contract: "
            f"missing={missing}, unknown={unknown}."
        )
    return pd.DataFrame({column: values[column] for column in expected})


def _smoke_validation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(config))
    copied["nested_cv"]["outer_folds"] = 2
    copied["nested_cv"]["inner_folds"] = 2
    copied["model_selection"]["optuna_trials_per_outer_fold"] = 1
    validate_validation_config(copied)
    return copied


def _smoke_candidates(
    model_names: Sequence[str],
) -> dict[str, tuple[dict[str, Any], ...]]:
    candidates = {
        "logistic_regression": ({"C": 1.0},),
        "random_forest": ({"n_estimators": 20, "max_depth": 8, "min_samples_leaf": 1},),
        "lightgbm": (
            {
                "n_estimators": 30,
                "num_leaves": 15,
                "min_child_samples": 5,
            },
        ),
    }
    return {name: candidates[name] for name in model_names}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Restricted CSV or Parquet extract.")
    source.add_argument(
        "--smoke",
        action="store_true",
        help="Run on deterministic synthetic data; produces no research result.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/data_contract.yaml"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("configs/feature_schema.yaml"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("configs/validation.yaml"),
    )
    parser.add_argument(
        "--search-space",
        type=Path,
        default=Path("configs/search_space.yaml"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("runs/development"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_TRAINING_MODELS),
        help="Registered model names in comparison order.",
    )
    parser.add_argument(
        "--variant",
        choices=("india_policy", "transportable"),
        default="india_policy",
    )
    parser.add_argument("--experiment-id", default="anaemia-development-v1")
    parser.add_argument(
        "--code-version",
        default=os.environ.get("ANAEMIA_CODE_VERSION", "working-tree"),
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--synthetic-groups", type=int, default=30)
    parser.add_argument(
        "--no-strict-profile",
        action="store_true",
        help="Allow a non-reference research extract size; never weakens column checks.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the command-line development workflow."""
    args = _build_parser().parse_args(argv)
    contract = load_contract(args.contract)
    schema = load_feature_schema(args.features, contract_path=args.contract)
    config = load_validation_config(args.validation)
    search_space = load_search_space(args.search_space)
    names = _validated_models(args.models)

    if args.smoke:
        frame = make_synthetic_smoke_frame(
            contract,
            group_count=args.synthetic_groups,
        )
        config = _smoke_validation_config(config)
        candidates = _smoke_candidates(names)
        dataset_fingerprint = fingerprint_json(
            {
                "generator": "anaemia-synthetic-smoke-v1",
                "groups": args.synthetic_groups,
            }
        )
        run_kind: RunKind = "synthetic_smoke"
        strict_profile = False
    else:
        frame = read_dataset(args.data)
        candidates = None
        dataset_fingerprint = _file_sha256(args.data)
        run_kind = "research_development"
        strict_profile = not args.no_strict_profile

    result = run_development_training(
        frame,
        contract,
        schema,
        config,
        search_space,
        output_directory=args.output_directory,
        dataset_fingerprint=dataset_fingerprint,
        code_version=args.code_version,
        experiment_id=args.experiment_id,
        model_names=names,
        parameter_candidates=candidates,
        variant=args.variant,
        run_kind=run_kind,
        strict_profile=strict_profile,
        n_jobs=args.n_jobs,
        overwrite=args.overwrite,
    )
    print(json.dumps(result.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
