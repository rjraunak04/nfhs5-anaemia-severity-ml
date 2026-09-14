"""One-command Day 2 calibration, SHAP, and recruiter-demo workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from anaemia_ml.data.validate import load_contract, read_dataset
from anaemia_ml.evaluation.calibration import (
    CalibrationSelection,
    TemperatureScaledClassifier,
    select_temperature_scaling,
)
from anaemia_ml.evaluation.config import (
    load_validation_config,
    validate_validation_config,
)
from anaemia_ml.evaluation.explainability import build_shap_report
from anaemia_ml.evaluation.metrics import DEFAULT_CLASSES, align_probability_columns
from anaemia_ml.features.schema import feature_columns, load_feature_schema
from anaemia_ml.modeling.artifacts import (
    ModelArtifactManifest,
    load_model_artifact,
    save_model_artifact,
    write_json_artifact,
)
from anaemia_ml.modeling.checkpoints import fingerprint_json
from anaemia_ml.modeling.search_space import load_search_space
from anaemia_ml.training import (
    DEFAULT_TRAINING_MODELS,
    DevelopmentTrainingResult,
    make_synthetic_smoke_frame,
    prepare_development_data,
    run_development_training,
)

DAY2_WORKFLOW_VERSION = "1.0.0"
PORTFOLIO_SUMMARY_FILENAME = "portfolio_summary.json"


class Day2WorkflowError(ValueError):
    """Raised when the Day 2 workflow would be incomplete or unsafe."""


@dataclass(frozen=True)
class Day2Result:
    """Paths and aggregate identities from one completed Day 2 run."""

    run_kind: str
    development: DevelopmentTrainingResult
    calibration: CalibrationSelection
    calibrated_manifest: ModelArtifactManifest
    calibration_report_path: Path
    explainability_report_path: Path
    portfolio_summary_path: Path

    def summary(self) -> dict[str, Any]:
        """Return a concise, JSON-safe CLI result."""
        return {
            "workflow_version": DAY2_WORKFLOW_VERSION,
            "run_kind": self.run_kind,
            "selected_model_name": self.development.selected_model_name,
            "calibration_method": self.calibration.method,
            "calibrated_model_sha256": self.calibrated_manifest.model_sha256,
            "portfolio_summary_file": self.portfolio_summary_path.name,
            "shap_scope": "aggregate_only",
            "locked_test_evaluated": False,
            "final_performance_claim_allowed": False,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise Day2WorkflowError(f"Could not fingerprint dataset: {path}") from error
    return digest.hexdigest()


def _preflight_output(output: Path, *, overwrite: bool) -> None:
    if output.exists() and not output.is_dir():
        raise Day2WorkflowError(f"output_directory is not a directory: {output}")
    targets = (
        output / "calibration_report.json",
        output / "explainability_report.json",
        output / PORTFOLIO_SUMMARY_FILENAME,
        output / "calibrated_model" / "manifest.json",
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        raise Day2WorkflowError(f"Refusing to overwrite Day 2 outputs: {existing}.")


def _model_comparison(result: DevelopmentTrainingResult) -> list[dict[str, Any]]:
    rows = []
    for report in result.model_reports:
        metrics = {aggregate.name: aggregate for aggregate in report.aggregate_metrics}
        rows.append(
            {
                "model_name": report.model_name,
                "display_name": report.display_name,
                "macro_f1_mean": metrics["macro_f1"].mean,
                "macro_f1_standard_deviation": metrics["macro_f1"].standard_deviation,
                "balanced_accuracy_mean": metrics["balanced_accuracy"].mean,
                "severe_recall_mean": metrics["severe_recall"].mean,
                "total_model_fits": report.total_model_fits,
            }
        )
    return rows


def _calibrated_pipeline(
    fitted_pipeline: Pipeline,
    selection: CalibrationSelection,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "calibrated_model",
                TemperatureScaledClassifier(
                    fitted_pipeline,
                    temperature=selection.temperature,
                    classes=DEFAULT_CLASSES,
                ),
            )
        ]
    )


def _portfolio_payload(
    *,
    run_kind: str,
    development: DevelopmentTrainingResult,
    calibration: CalibrationSelection,
    explainability: Mapping[str, Any],
    partition_rows: Mapping[str, int],
) -> dict[str, Any]:
    synthetic = run_kind == "synthetic_smoke"
    return {
        "schema_version": "1.0.0",
        "project_title": "NFHS-5 Anaemia Severity ML",
        "run_kind": run_kind,
        "result_status": (
            "synthetic_engineering_demo" if synthetic else "development_calibration_only"
        ),
        "disclosure": {
            "headline": (
                "Synthetic engineering demonstration — not an NFHS research result"
                if synthetic
                else "Restricted NFHS development/calibration analysis — final test locked"
            ),
            "data_source": (
                "Deterministic contract-shaped synthetic data"
                if synthetic
                else "Restricted NFHS-5 analytic extract"
            ),
            "final_performance_claim_allowed": False,
            "clinical_use_allowed": False,
            "locked_test_evaluated": False,
        },
        "workflow": {
            "version": DAY2_WORKFLOW_VERSION,
            "experiment_fingerprint": development.identity.fingerprint,
            "variant": development.model_reports[0].variant,
            "partition_rows": dict(partition_rows),
            "group_disjoint_partitions": True,
            "fold_local_preprocessing": True,
            "survey_weighted_metrics": True,
        },
        "model_comparison": _model_comparison(development),
        "selection": {
            "model_name": development.selected_model_name,
            "parameters": dict(development.selected_parameters),
            "primary_metric": "macro_f1",
            "selection_scope": "nested_grouped_cv_on_development_only",
        },
        "calibration": calibration.as_dict(),
        "explainability": dict(explainability),
        "responsible_use": [
            "Research and engineering demonstration only",
            "No individual diagnosis or treatment decisions",
            "SHAP associations are not causal effects",
            "Final claims require the protocol-gated locked-test evaluation",
        ],
    }


def run_day2_workflow(
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
    run_kind: str = "research_development",
    strict_profile: bool = True,
    n_jobs: int = 1,
    shap_max_samples: int = 200,
    shap_background_size: int = 50,
    overwrite: bool = False,
) -> Day2Result:
    """Run development selection, calibration, SHAP and safe report export."""
    if run_kind not in {"research_development", "synthetic_smoke"}:
        raise Day2WorkflowError(f"Unsupported run_kind: {run_kind!r}.")
    output = Path(output_directory)
    _preflight_output(output, overwrite=overwrite)
    development = run_development_training(
        frame,
        data_contract,
        feature_schema,
        validation_config,
        search_space,
        output_directory=output / "development",
        dataset_fingerprint=dataset_fingerprint,
        code_version=code_version,
        experiment_id=experiment_id,
        model_names=model_names,
        parameter_candidates=parameter_candidates,
        variant=variant,
        run_kind=run_kind,
        strict_profile=strict_profile,
        n_jobs=n_jobs,
        overwrite=overwrite,
    )
    prepared = prepare_development_data(
        frame,
        data_contract,
        feature_schema,
        validation_config,
        variant=variant,
        strict_profile=strict_profile,
    )
    calibration_index = prepared.partitions.calibration
    loaded = load_model_artifact(
        output / "development" / "model",
        trusted=True,
        expected_experiment_fingerprint=development.identity.fingerprint,
    )
    calibration_frame = prepared.predictors.iloc[calibration_index]
    raw_scores = loaded.pipeline.predict_proba(calibration_frame)
    model_classes = getattr(loaded.pipeline, "classes_", DEFAULT_CLASSES)
    aligned_scores = align_probability_columns(
        raw_scores,
        model_classes,
        expected_classes=DEFAULT_CLASSES,
    )
    calibration = select_temperature_scaling(
        prepared.target[calibration_index],
        aligned_scores,
        prepared.groups[calibration_index],
        model_classes=DEFAULT_CLASSES,
        expected_classes=DEFAULT_CLASSES,
        sample_weight=prepared.sample_weight[calibration_index],
        n_splits=5,
        random_state=int(validation_config["random_seed"]),
    )
    calibrated_pipeline = _calibrated_pipeline(loaded.pipeline, calibration)
    calibrated_manifest = save_model_artifact(
        output / "calibrated_model",
        calibrated_pipeline,
        experiment_fingerprint=development.identity.fingerprint,
        metadata={
            "workflow_version": DAY2_WORKFLOW_VERSION,
            "artifact_role": "calibrated_development_candidate",
            "run_kind": run_kind,
            "model_name": development.selected_model_name,
            "variant": variant,
            "calibration_method": calibration.method,
            "temperature": calibration.temperature,
            "calibration_partition_rows": int(calibration_index.size),
            "locked_test_evaluated": False,
            "final_performance_claim_allowed": False,
            "clinical_use_allowed": False,
        },
        overwrite=overwrite,
    )
    calibration_payload = {
        "workflow_version": DAY2_WORKFLOW_VERSION,
        "run_kind": run_kind,
        "experiment_fingerprint": development.identity.fingerprint,
        "model_name": development.selected_model_name,
        "calibration": calibration.as_dict(),
        "calibrated_model": calibrated_manifest.as_dict(),
        "final_performance_claim_allowed": False,
    }
    calibration_report_path = write_json_artifact(
        output / "calibration_report.json",
        calibration_payload,
        overwrite=overwrite,
    )
    explainability = build_shap_report(
        loaded.pipeline,
        calibration_frame,
        feature_schema,
        variant=variant,
        partition_name="calibration",
        max_samples=shap_max_samples,
        background_size=shap_background_size,
        top_n=min(20, calibration_frame.shape[1]),
        random_state=int(validation_config["random_seed"]),
    )
    explainability_report_path = write_json_artifact(
        output / "explainability_report.json",
        {
            "workflow_version": DAY2_WORKFLOW_VERSION,
            "run_kind": run_kind,
            "experiment_fingerprint": development.identity.fingerprint,
            "model_name": development.selected_model_name,
            "explainability": explainability,
            "final_performance_claim_allowed": False,
        },
        overwrite=overwrite,
    )
    partition_rows = {
        name: int(indices.size) for name, indices in prepared.partitions.as_dict().items()
    }
    portfolio_summary_path = write_json_artifact(
        output / PORTFOLIO_SUMMARY_FILENAME,
        _portfolio_payload(
            run_kind=run_kind,
            development=development,
            calibration=calibration,
            explainability=explainability,
            partition_rows=partition_rows,
        ),
        overwrite=overwrite,
    )
    return Day2Result(
        run_kind=run_kind,
        development=development,
        calibration=calibration,
        calibrated_manifest=calibrated_manifest,
        calibration_report_path=calibration_report_path,
        explainability_report_path=explainability_report_path,
        portfolio_summary_path=portfolio_summary_path,
    )


def _smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(config))
    copied["nested_cv"]["outer_folds"] = 2
    copied["nested_cv"]["inner_folds"] = 2
    copied["model_selection"]["optuna_trials_per_outer_fold"] = 1
    validate_validation_config(copied)
    return copied


def make_day2_smoke_frame(
    data_contract: Mapping[str, Any],
    feature_schema: Mapping[str, Any],
    *,
    group_count: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a valid but deliberately non-predictive recruiter demo fixture."""
    frame = make_synthetic_smoke_frame(data_contract, group_count=group_count)
    grouping_columns = set(data_contract["survey_design"]["composite_psu_components"])
    rng = np.random.default_rng(int(random_state))
    for column in feature_columns(dict(feature_schema)):
        if column not in grouping_columns:
            frame.loc[:, column] = rng.permutation(frame[column].to_numpy())
    return frame


def _smoke_candidates(
    model_names: Sequence[str],
) -> dict[str, tuple[dict[str, Any], ...]]:
    candidates = {
        "logistic_regression": ({"C": 1.0},),
        "random_forest": ({"n_estimators": 20, "max_depth": 8, "min_samples_leaf": 1},),
        "lightgbm": ({"n_estimators": 30, "num_leaves": 15, "min_child_samples": 5},),
    }
    try:
        return {name: candidates[name] for name in model_names}
    except KeyError as error:
        raise Day2WorkflowError(f"Unsupported smoke model: {error.args[0]!r}.") from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Restricted 40-column CSV or Parquet.")
    source.add_argument(
        "--smoke",
        action="store_true",
        help="Use deterministic synthetic data; creates no research result.",
    )
    parser.add_argument("--contract", type=Path, default=Path("configs/data_contract.yaml"))
    parser.add_argument("--features", type=Path, default=Path("configs/feature_schema.yaml"))
    parser.add_argument("--validation", type=Path, default=Path("configs/validation.yaml"))
    parser.add_argument("--search-space", type=Path, default=Path("configs/search_space.yaml"))
    parser.add_argument("--output-directory", type=Path, default=Path("runs/day2"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_TRAINING_MODELS))
    parser.add_argument(
        "--variant",
        choices=("india_policy", "transportable"),
        default="india_policy",
    )
    parser.add_argument("--experiment-id", default="anaemia-day2-v1")
    parser.add_argument(
        "--code-version",
        default=os.environ.get("ANAEMIA_CODE_VERSION", "working-tree"),
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--synthetic-groups", type=int, default=30)
    parser.add_argument("--shap-max-samples", type=int, default=200)
    parser.add_argument("--shap-background-size", type=int, default=50)
    parser.add_argument("--no-strict-profile", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the complete Day 2 workflow."""
    args = _build_parser().parse_args(argv)
    contract = load_contract(args.contract)
    schema = load_feature_schema(args.features, contract_path=args.contract)
    validation = load_validation_config(args.validation)
    search_space = load_search_space(args.search_space)
    model_names = tuple(args.models)
    if args.smoke:
        frame = make_day2_smoke_frame(
            contract,
            schema,
            group_count=args.synthetic_groups,
            random_state=int(validation["random_seed"]),
        )
        validation = _smoke_config(validation)
        candidates = _smoke_candidates(model_names)
        dataset_fingerprint = fingerprint_json(
            {
                "generator": "anaemia-day2-synthetic-smoke-v2",
                "groups": args.synthetic_groups,
            }
        )
        run_kind = "synthetic_smoke"
        strict_profile = False
    else:
        frame = read_dataset(args.data)
        candidates = None
        dataset_fingerprint = _sha256_file(args.data)
        run_kind = "research_development"
        strict_profile = not args.no_strict_profile

    result = run_day2_workflow(
        frame,
        contract,
        schema,
        validation,
        search_space,
        output_directory=args.output_directory,
        dataset_fingerprint=dataset_fingerprint,
        code_version=args.code_version,
        experiment_id=args.experiment_id,
        model_names=model_names,
        parameter_candidates=candidates,
        variant=args.variant,
        run_kind=run_kind,
        strict_profile=strict_profile,
        n_jobs=args.n_jobs,
        shap_max_samples=args.shap_max_samples,
        shap_background_size=args.shap_background_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(result.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
