"""Load, validate, and fingerprint the frozen model-feature schema."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from anaemia_ml.data.validate import load_contract, predictor_columns

FEATURE_GROUP_KEYS = (
    "continuous_numeric",
    "count_numeric",
    "nominal_categorical",
    "indicator_categorical",
    "ordinal_categorical",
)

MODEL_VARIANTS = (
    "india_policy",
    "transportable",
)


class FeatureSchemaError(ValueError):
    """Raised when the frozen feature schema is inconsistent or unsafe."""


def _mapping(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FeatureSchemaError(
            f"{name} must be a mapping."
        )
    return value


def _string_list(
    value: Any,
    *,
    name: str,
    allow_empty: bool = True,
) -> list[str]:
    if not isinstance(value, list):
        raise FeatureSchemaError(
            f"{name} must be a list."
        )

    if not allow_empty and not value:
        raise FeatureSchemaError(
            f"{name} must not be empty."
        )

    if any(
        not isinstance(item, str) or not item.strip()
        for item in value
    ):
        raise FeatureSchemaError(
            f"{name} must contain non-empty strings only."
        )

    return value


def _grouped_columns(
    container: dict[str, Any],
    *,
    section: str,
    required_groups: tuple[str, ...] | None = None,
) -> list[str]:
    groups = _mapping(
        container.get(section),
        name=section,
    )

    if required_groups is not None:
        missing = sorted(
            set(required_groups) - groups.keys()
        )
        extra = sorted(
            groups.keys() - set(required_groups)
        )

        if missing or extra:
            raise FeatureSchemaError(
                f"{section} groups differ: "
                f"missing={missing}, extra={extra}."
            )

        ordered_groups = required_groups
    else:
        ordered_groups = tuple(groups)

    columns: list[str] = []

    for group in ordered_groups:
        columns.extend(
            _string_list(
                groups[group],
                name=f"{section}.{group}",
                allow_empty=False,
            )
        )

    return columns


def _duplicates(
    values: list[str],
) -> list[str]:
    counts = Counter(values)

    return sorted(
        value
        for value, count in counts.items()
        if count > 1
    )


def feature_columns(
    schema: dict[str, Any],
    *,
    variant: str = "india_policy",
) -> list[str]:
    """Return predictors for one prespecified model variant."""

    features = _grouped_columns(
        schema,
        section="feature_groups",
        required_groups=FEATURE_GROUP_KEYS,
    )

    variants = _mapping(
        schema.get("model_variants"),
        name="model_variants",
    )

    if variant not in variants:
        raise FeatureSchemaError(
            f"Unknown model variant: {variant}"
        )

    settings = _mapping(
        variants[variant],
        name=f"model_variants.{variant}",
    )

    excluded = set(
        _string_list(
            settings.get("excluded_features"),
            name=(
                f"model_variants.{variant}"
                ".excluded_features"
            ),
        )
    )

    unknown = sorted(
        excluded - set(features)
    )

    if unknown:
        raise FeatureSchemaError(
            f"Model variant {variant} excludes "
            f"unknown features: {unknown}"
        )

    return [
        feature
        for feature in features
        if feature not in excluded
    ]


def validate_feature_schema(
    schema: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    """Validate the schema against the data contract."""

    required_sections = {
        "schema_version",
        "status",
        "source_contract",
        "target",
        "feature_groups",
        "non_predictor_roles",
        "model_variants",
        "preprocessing_contract",
        "leakage_guards",
        "change_control",
    }

    missing_sections = sorted(
        required_sections - schema.keys()
    )

    if missing_sections:
        raise FeatureSchemaError(
            "Missing feature-schema sections: "
            f"{missing_sections}"
        )

    if (
        not isinstance(schema["schema_version"], str)
        or not schema["schema_version"].strip()
    ):
        raise FeatureSchemaError(
            "schema_version must be a non-empty string."
        )

    if (
        schema["status"]
        != "frozen_before_model_training"
    ):
        raise FeatureSchemaError(
            "Feature schema must be frozen "
            "before model training."
        )

    source = _mapping(
        schema["source_contract"],
        name="source_contract",
    )

    if (
        source.get("contract_version")
        != contract.get("contract_version")
    ):
        raise FeatureSchemaError(
            "Feature and data-contract "
            "versions do not match."
        )

    if (
        not isinstance(source.get("path"), str)
        or not source["path"].strip()
    ):
        raise FeatureSchemaError(
            "source_contract.path must be "
            "a non-empty string."
        )

    target = _mapping(
        schema["target"],
        name="target",
    )

    contract_target = _mapping(
        contract.get("target"),
        name="contract.target",
    )

    for key in (
        "raw_column",
        "model_column",
        "ordered_classes",
    ):
        if target.get(key) != contract_target.get(key):
            raise FeatureSchemaError(
                f"Feature-schema target.{key} "
                "differs from contract."
            )

    features = _grouped_columns(
        schema,
        section="feature_groups",
        required_groups=FEATURE_GROUP_KEYS,
    )

    duplicate_features = _duplicates(features)

    if duplicate_features:
        raise FeatureSchemaError(
            "Features appear in multiple groups: "
            f"{duplicate_features}"
        )

    contract_features = predictor_columns(contract)

    missing_features = sorted(
        set(contract_features) - set(features)
    )

    extra_features = sorted(
        set(features) - set(contract_features)
    )

    if missing_features or extra_features:
        raise FeatureSchemaError(
            "Feature schema differs from data contract: "
            f"missing={missing_features}, "
            f"extra={extra_features}."
        )

    non_predictors = _grouped_columns(
        schema,
        section="non_predictor_roles",
    )

    duplicate_non_predictors = _duplicates(
        non_predictors
    )

    if duplicate_non_predictors:
        raise FeatureSchemaError(
            "Non-predictors appear in multiple roles: "
            f"{duplicate_non_predictors}"
        )

    contract_forbidden = set(
        contract["never_use_as_predictors"]
    )

    if set(non_predictors) != contract_forbidden:
        raise FeatureSchemaError(
            "Non-predictor roles differ "
            "from the data contract."
        )

    guards = _mapping(
        schema["leakage_guards"],
        name="leakage_guards",
    )

    forbidden = _string_list(
        guards.get("forbidden_predictors"),
        name="leakage_guards.forbidden_predictors",
        allow_empty=False,
    )

    duplicate_forbidden = _duplicates(forbidden)

    if duplicate_forbidden:
        raise FeatureSchemaError(
            "Forbidden predictors contain duplicates: "
            f"{duplicate_forbidden}"
        )

    if set(forbidden) != contract_forbidden:
        raise FeatureSchemaError(
            "Forbidden predictors differ "
            "from the data contract."
        )

    overlap = sorted(
        set(features) & set(forbidden)
    )

    if overlap:
        raise FeatureSchemaError(
            "Features overlap forbidden predictors: "
            f"{overlap}"
        )

    false_guards = (
        "outcome_derived_features_allowed",
        "preprocessing_before_split_allowed",
        "feature_selection_outside_training_fold_allowed",
        "test_informed_feature_changes_allowed",
    )

    unsafe_guards = [
        name
        for name in false_guards
        if guards.get(name) is not False
    ]

    if unsafe_guards:
        raise FeatureSchemaError(
            f"Unsafe leakage guards: {unsafe_guards}"
        )

    variants = _mapping(
        schema["model_variants"],
        name="model_variants",
    )

    if set(variants) != set(MODEL_VARIANTS):
        raise FeatureSchemaError(
            "Exactly two prespecified model "
            "variants are required."
        )

    geographic_features = set(
        contract["column_roles"].get(
            "primary_geographic_predictor",
            [],
        )
    )

    expected_exclusions = {
        "india_policy": set(),
        "transportable": geographic_features,
    }

    for variant, required_exclusions in (
        expected_exclusions.items()
    ):
        settings = _mapping(
            variants[variant],
            name=f"model_variants.{variant}",
        )

        description = settings.get("description")

        if (
            not isinstance(description, str)
            or not description.strip()
        ):
            raise FeatureSchemaError(
                f"model_variants.{variant}.description "
                "must be a non-empty string."
            )

        excluded = set(
            _string_list(
                settings.get("excluded_features"),
                name=(
                    f"model_variants.{variant}"
                    ".excluded_features"
                ),
            )
        )

        if excluded != required_exclusions:
            raise FeatureSchemaError(
                "Unexpected exclusions for model "
                f"variant {variant}: {sorted(excluded)}"
            )

        expected_count = settings.get(
            "expected_predictor_count"
        )

        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
        ):
            raise FeatureSchemaError(
                f"model_variants.{variant}."
                "expected_predictor_count "
                "must be an integer."
            )

        if expected_count != len(
            feature_columns(
                schema,
                variant=variant,
            )
        ):
            raise FeatureSchemaError(
                "Predictor count is incorrect for "
                f"model variant {variant}."
            )

    preprocessing = _mapping(
        schema["preprocessing_contract"],
        name="preprocessing_contract",
    )

    if (
        preprocessing.get("fit_scope")
        != "training_fold_only"
    ):
        raise FeatureSchemaError(
            "Preprocessing must be fitted "
            "inside training folds."
        )

    numeric = _mapping(
        preprocessing.get("numeric"),
        name="preprocessing_contract.numeric",
    )

    if numeric.get("imputation") != "median":
        raise FeatureSchemaError(
            "Numeric imputation must be median."
        )

    if (
        numeric.get("add_missing_indicator")
        is not True
    ):
        raise FeatureSchemaError(
            "Numeric missing indicators "
            "must be enabled."
        )

    if (
        numeric.get("scaling")
        != "model_dependent"
    ):
        raise FeatureSchemaError(
            "Numeric scaling must remain "
            "model-dependent."
        )

    categorical = _mapping(
        preprocessing.get("categorical"),
        name="preprocessing_contract.categorical",
    )

    expected_categorical = {
        "imputation": "explicit_missing_category",
        "unknown_category_policy": (
            "handle_without_failure"
        ),
        "linear_model_encoding": "one_hot",
        "tree_model_encoding": "model_appropriate",
    }

    for key, expected in expected_categorical.items():
        if categorical.get(key) != expected:
            raise FeatureSchemaError(
                "Unexpected categorical "
                f"preprocessing rule for {key}."
            )

    if (
        categorical.get(
            "verify_ordinal_order_from_official_labels"
        )
        is not True
    ):
        raise FeatureSchemaError(
            "Ordinal order must use verified "
            "official labels."
        )

    engineered = _mapping(
        preprocessing.get("engineered_features"),
        name=(
            "preprocessing_contract."
            "engineered_features"
        ),
    )

    if engineered.get("primary_analysis") != []:
        raise FeatureSchemaError(
            "Primary engineered features "
            "must remain empty."
        )

    if (
        engineered.get("post_hoc_features_allowed")
        is not False
    ):
        raise FeatureSchemaError(
            "Post-hoc engineered features "
            "must be disabled."
        )

    change_control = _mapping(
        schema["change_control"],
        name="change_control",
    )

    if (
        change_control.get(
            "version_increment_required"
        )
        is not True
    ):
        raise FeatureSchemaError(
            "Schema changes must increment "
            "the version."
        )

    if (
        change_control.get(
            "documented_reason_required"
        )
        is not True
    ):
        raise FeatureSchemaError(
            "Schema changes must document "
            "a reason."
        )

    if (
        change_control.get(
            "update_after_final_test_allowed"
        )
        is not False
    ):
        raise FeatureSchemaError(
            "Feature changes after final-test "
            "access are forbidden."
        )


def _contract_path_from_schema(
    schema_path: Path,
    schema: dict[str, Any],
) -> Path:
    source = _mapping(
        schema.get("source_contract"),
        name="source_contract",
    )

    configured_value = source.get("path")

    if (
        not isinstance(configured_value, str)
        or not configured_value.strip()
    ):
        raise FeatureSchemaError(
            "source_contract.path must be "
            "a non-empty string."
        )

    configured = Path(configured_value)

    if configured.is_absolute() or configured.is_file():
        return configured

    return (
        schema_path.resolve().parents[1]
        / configured
    )


def load_feature_schema(
    path: str | Path,
    *,
    contract_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the feature schema."""

    schema_path = Path(path)

    if not schema_path.is_file():
        raise FileNotFoundError(
            f"Feature schema not found: {schema_path}"
        )

    with schema_path.open(
        encoding="utf-8"
    ) as stream:
        schema = yaml.safe_load(stream)

    if not isinstance(schema, dict):
        raise FeatureSchemaError(
            "The feature-schema YAML root "
            "must be a mapping."
        )

    resolved_contract_path = (
        Path(contract_path)
        if contract_path is not None
        else _contract_path_from_schema(
            schema_path,
            schema,
        )
    )

    contract = load_contract(
        resolved_contract_path
    )

    validate_feature_schema(
        schema,
        contract,
    )

    return schema


def feature_schema_fingerprint(
    schema: dict[str, Any],
) -> str:
    """Return a stable SHA-256 schema fingerprint."""

    canonical = json.dumps(
        schema,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()