"""Load and enforce the confirmatory validation configuration."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

FINAL_TEST_UNLOCK_TOKEN = "I_UNDERSTAND_FINAL_TEST_IS_SINGLE_USE"


class ValidationConfigError(ValueError):
    """Raised when the validation configuration is incomplete or unsafe."""


class FinalTestLockedError(RuntimeError):
    """Raised when final-test access is attempted prematurely."""


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationConfigError(f"{name} must be a mapping.")
    return value


def _positive_integer(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationConfigError(f"{name} must be an integer >= {minimum}.")
    return value


def validate_validation_config(config: dict[str, Any]) -> None:
    """Reject incomplete, inconsistent, or unsafe validation settings."""
    required_sections = {
        "validation_version",
        "random_seed",
        "grouping",
        "partitions",
        "nested_cv",
        "model_selection",
        "preprocessing",
        "final_test",
        "privacy",
    }
    missing = sorted(required_sections - config.keys())
    if missing:
        raise ValidationConfigError(f"Missing validation sections: {missing}")

    if not isinstance(config["validation_version"], str):
        raise ValidationConfigError("validation_version must be a string.")

    _positive_integer(config["random_seed"], name="random_seed", minimum=0)

    grouping = _mapping(config["grouping"], name="grouping")
    if grouping.get("strategy") != "composite_psu":
        raise ValidationConfigError("grouping.strategy must be composite_psu.")

    columns = grouping.get("columns")
    if not isinstance(columns, list) or len(columns) < 2:
        raise ValidationConfigError(
            "grouping.columns must contain composite PSU columns."
        )
    if len(columns) != len(set(columns)):
        raise ValidationConfigError(
            "grouping.columns must not contain duplicates."
        )

    partitions = _mapping(config["partitions"], name="partitions")
    expected_partitions = {"development", "calibration", "locked_test"}
    if set(partitions) != expected_partitions:
        raise ValidationConfigError(
            "partitions must define development, calibration, and locked_test."
        )

    for name, value in partitions.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValidationConfigError(f"partitions.{name} must be numeric.")
        if not 0 < float(value) < 1:
            raise ValidationConfigError(
                f"partitions.{name} must lie between 0 and 1."
            )

    if not math.isclose(sum(partitions.values()), 1.0, abs_tol=1e-9):
        raise ValidationConfigError("Partition fractions must sum to 1.0.")

    nested = _mapping(config["nested_cv"], name="nested_cv")
    if nested.get("splitter") != "StratifiedGroupKFold":
        raise ValidationConfigError("nested_cv.splitter is not supported.")

    _positive_integer(
        nested.get("outer_folds"),
        name="nested_cv.outer_folds",
        minimum=2,
    )
    _positive_integer(
        nested.get("inner_folds"),
        name="nested_cv.inner_folds",
        minimum=2,
    )

    if nested.get("shuffle") is not True:
        raise ValidationConfigError("nested_cv.shuffle must be true.")
    if nested.get("scope") != "development_only":
        raise ValidationConfigError(
            "nested_cv.scope must be development_only."
        )

    selection = _mapping(config["model_selection"], name="model_selection")
    if selection.get("primary_metric") != "macro_f1":
        raise ValidationConfigError(
            "model_selection.primary_metric must be macro_f1."
        )

    _positive_integer(
        selection.get("optuna_trials_per_outer_fold"),
        name="model_selection.optuna_trials_per_outer_fold",
    )

    if selection.get("calibration_during_nested_cv") is not False:
        raise ValidationConfigError(
            "Calibration must remain separate from nested CV."
        )

    preprocessing = _mapping(config["preprocessing"], name="preprocessing")
    if preprocessing.get("fit_inside_training_fold_only") is not True:
        raise ValidationConfigError(
            "Preprocessing must be fitted inside training folds."
        )

    if preprocessing.get("primary_imbalance_strategy") != "class_weight":
        raise ValidationConfigError(
            "Primary imbalance strategy must be class_weight."
        )

    smotenc_ratio = preprocessing.get("smotenc_sensitivity_ratio")
    if isinstance(smotenc_ratio, bool) or not isinstance(
        smotenc_ratio, int | float
    ):
        raise ValidationConfigError(
            "smotenc_sensitivity_ratio must be numeric."
        )
    if not 0 < float(smotenc_ratio) <= 1:
        raise ValidationConfigError(
            "smotenc_sensitivity_ratio must lie in (0, 1]."
        )

    final_test = _mapping(config["final_test"], name="final_test")
    if final_test.get("locked_by_default") is not True:
        raise ValidationConfigError(
            "The final test must be locked by default."
        )

    environment_variable = final_test.get("unlock_environment_variable")
    if not isinstance(environment_variable, str) or not environment_variable.strip():
        raise ValidationConfigError(
            "A final-test unlock environment variable is required."
        )

    required_gates = final_test.get("required_gates")
    if not isinstance(required_gates, list) or not required_gates:
        raise ValidationConfigError(
            "final_test.required_gates must be a non-empty list."
        )

    if not all(
        isinstance(gate, str) and gate.strip()
        for gate in required_gates
    ):
        raise ValidationConfigError(
            "Every final-test gate must be a non-empty string."
        )

    if len(required_gates) != len(set(required_gates)):
        raise ValidationConfigError(
            "final_test.required_gates contains duplicates."
        )

    privacy = _mapping(config["privacy"], name="privacy")
    unsafe = sorted(
        name for name, allowed in privacy.items() if allowed is not False
    )
    if unsafe:
        raise ValidationConfigError(
            f"Unsafe repository privacy settings: {unsafe}"
        )


def load_validation_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML validation configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Validation config not found: {config_path}"
        )

    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValidationConfigError(
            "The validation YAML root must be a mapping."
        )

    validate_validation_config(config)
    return config


def require_final_test_unlock(
    config: dict[str, Any],
    gate_status: Mapping[str, bool],
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Permit final-test access only after every gate and explicit unlock."""
    validate_validation_config(config)

    final_test = config["final_test"]
    missing_gates = [
        gate
        for gate in final_test["required_gates"]
        if gate_status.get(gate) is not True
    ]

    if missing_gates:
        raise FinalTestLockedError(
            f"Incomplete final-test gates: {missing_gates}"
        )

    environment = os.environ if environ is None else environ
    variable = final_test["unlock_environment_variable"]

    if environment.get(variable) != FINAL_TEST_UNLOCK_TOKEN:
        raise FinalTestLockedError(
            f"Final test is locked. Set {variable} to the exact single-use "
            "acknowledgement only after the analysis is frozen."
        )