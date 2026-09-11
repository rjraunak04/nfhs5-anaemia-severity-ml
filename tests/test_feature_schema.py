"""Tests for the frozen model-feature schema."""

from copy import deepcopy
from pathlib import Path

import pytest

from anaemia_ml.data.validate import load_contract
from anaemia_ml.features.schema import (
    FeatureSchemaError,
    feature_columns,
    feature_schema_fingerprint,
    load_feature_schema,
    validate_feature_schema,
)

ROOT = Path(__file__).parents[1]
SCHEMA_PATH = ROOT / "configs" / "feature_schema.yaml"
CONTRACT_PATH = ROOT / "configs" / "data_contract.yaml"


@pytest.fixture
def schema() -> dict:
    """Load the repository feature schema."""
    return load_feature_schema(SCHEMA_PATH)


@pytest.fixture
def contract() -> dict:
    """Load the repository data contract."""
    return load_contract(CONTRACT_PATH)


def test_frozen_schema_loads(schema: dict) -> None:
    assert schema["schema_version"] == "1.0.0"
    assert schema["status"] == "frozen_before_model_training"
    assert schema["target"]["ordered_classes"] == [0, 1, 2, 3]


def test_model_variant_feature_counts(schema: dict) -> None:
    india_policy = feature_columns(
        schema,
        variant="india_policy",
    )
    transportable = feature_columns(
        schema,
        variant="transportable",
    )

    assert len(india_policy) == 32
    assert len(transportable) == 31
    assert "v024" in india_policy
    assert "v024" not in transportable


def test_predictors_are_disjoint_from_forbidden_columns(
    schema: dict,
) -> None:
    predictors = set(
        feature_columns(
            schema,
            variant="india_policy",
        )
    )
    forbidden = set(
        schema["leakage_guards"]["forbidden_predictors"]
    )

    assert forbidden == {
        "v002",
        "v003",
        "v005",
        "v021",
        "v022",
        "sdist",
        "v456",
        "v457",
    }
    assert predictors.isdisjoint(forbidden)


def test_duplicate_feature_is_rejected(
    schema: dict,
    contract: dict,
) -> None:
    invalid = deepcopy(schema)
    invalid["feature_groups"]["count_numeric"].append(
        "v012"
    )

    with pytest.raises(
        FeatureSchemaError,
        match="multiple groups",
    ):
        validate_feature_schema(invalid, contract)


def test_missing_contract_feature_is_rejected(
    schema: dict,
    contract: dict,
) -> None:
    invalid = deepcopy(schema)
    invalid["feature_groups"]["continuous_numeric"].remove(
        "v012"
    )

    with pytest.raises(
        FeatureSchemaError,
        match="differs from data contract",
    ):
        validate_feature_schema(invalid, contract)


def test_contract_version_mismatch_is_rejected(
    schema: dict,
    contract: dict,
) -> None:
    invalid = deepcopy(schema)
    invalid["source_contract"]["contract_version"] = "9.9.9"

    with pytest.raises(
        FeatureSchemaError,
        match="versions do not match",
    ):
        validate_feature_schema(invalid, contract)


def test_unexpected_transportable_exclusion_is_rejected(
    schema: dict,
    contract: dict,
) -> None:
    invalid = deepcopy(schema)
    invalid["model_variants"]["transportable"][
        "excluded_features"
    ].append("v025")

    with pytest.raises(
        FeatureSchemaError,
        match="Unexpected exclusions",
    ):
        validate_feature_schema(invalid, contract)


def test_unsafe_leakage_guard_is_rejected(
    schema: dict,
    contract: dict,
) -> None:
    invalid = deepcopy(schema)
    invalid["leakage_guards"][
        "preprocessing_before_split_allowed"
    ] = True

    with pytest.raises(
        FeatureSchemaError,
        match="Unsafe leakage guards",
    ):
        validate_feature_schema(invalid, contract)


def test_global_preprocessing_is_rejected(
    schema: dict,
    contract: dict,
) -> None:
    invalid = deepcopy(schema)
    invalid["preprocessing_contract"][
        "fit_scope"
    ] = "full_dataset"

    with pytest.raises(
        FeatureSchemaError,
        match="training folds",
    ):
        validate_feature_schema(invalid, contract)


def test_unknown_model_variant_is_rejected(
    schema: dict,
) -> None:
    with pytest.raises(
        FeatureSchemaError,
        match="Unknown model variant",
    ):
        feature_columns(
            schema,
            variant="unknown_model",
        )


def test_schema_fingerprint_is_stable_and_sensitive(
    schema: dict,
) -> None:
    first = feature_schema_fingerprint(schema)
    repeated = feature_schema_fingerprint(
        deepcopy(schema)
    )

    changed = deepcopy(schema)
    changed["schema_version"] = "1.0.1"
    changed_fingerprint = feature_schema_fingerprint(
        changed
    )

    assert first == repeated
    assert len(first) == 64
    assert changed_fingerprint != first