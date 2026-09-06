"""Tests for the NFHS-5 data contract validator."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from anaemia_ml.data.validate import (
    DataContractError,
    load_contract,
    predictor_columns,
    validate_contract_definition,
    validate_dataframe,
)

CONTRACT_PATH = Path(__file__).parents[1] / "configs" / "data_contract.yaml"


@pytest.fixture
def contract() -> dict:
    return load_contract(CONTRACT_PATH)


@pytest.fixture
def valid_frame(contract: dict) -> pd.DataFrame:
    rows = 4
    frame = pd.DataFrame(
        {column: [0] * rows for column in contract["expected_column_order"]}
    )
    frame["v002"] = [1, 2, 3, 4]
    frame["v003"] = [1, 1, 1, 1]
    frame["v005"] = [1_000_000, 900_000, 1_100_000, 800_000]
    frame["v021"] = [101, 102, 103, 104]
    frame["v022"] = [11, 11, 12, 12]
    frame["v024"] = [1, 1, 2, 2]
    frame["sdist"] = [101, 102, 201, 202]
    frame["v012"] = [18, 25, 35, 49]
    frame["v025"] = [1, 2, 1, 2]
    frame["v190"] = [1, 2, 4, 5]
    frame["v457"] = [1, 2, 3, 4]
    return frame


def issue_codes(report) -> set[str]:
    return {issue.code for issue in report.errors}


def test_contract_is_internally_consistent(contract: dict) -> None:
    validate_contract_definition(contract)
    assert len(contract["expected_column_order"]) == 40


def test_predictor_sets_block_leakage(contract: dict) -> None:
    primary = predictor_columns(contract)
    transportable = predictor_columns(contract, transportable=True)
    excluded = set(contract["never_use_as_predictors"])

    assert len(primary) == 32
    assert excluded.isdisjoint(primary)
    assert "v024" in primary
    assert "v024" not in transportable
    assert len(transportable) == 31


def test_valid_small_frame_passes_schema_rules(
    contract: dict, valid_frame: pd.DataFrame
) -> None:
    report = validate_dataframe(valid_frame, contract, strict_profile=False)
    assert report.passed
    assert report.statistics["rows"] == 4
    assert report.statistics["duplicate_respondent_rows"] == 0
    assert report.statistics["target_counts_raw"] == {"1": 1, "2": 1, "3": 1, "4": 1}


def test_missing_column_is_rejected(contract: dict, valid_frame: pd.DataFrame) -> None:
    report = validate_dataframe(
        valid_frame.drop(columns="v457"), contract, strict_profile=False
    )
    assert "missing_columns" in issue_codes(report)


def test_duplicate_respondent_key_is_rejected(
    contract: dict, valid_frame: pd.DataFrame
) -> None:
    duplicated = pd.concat([valid_frame, valid_frame.iloc[[0]]], ignore_index=True)
    report = validate_dataframe(duplicated, contract, strict_profile=False)
    assert "duplicate_respondent_key" in issue_codes(report)


@pytest.mark.parametrize(
    ("column", "value", "expected_code"),
    [
        ("v012", 55, "invalid_age"),
        ("v025", 7, "invalid_allowed_values"),
        ("v190", 9, "invalid_allowed_values"),
        ("v457", 9, "invalid_allowed_values"),
        ("v005", 0, "nonpositive_weight"),
    ],
)
def test_invalid_core_values_are_rejected(
    contract: dict,
    valid_frame: pd.DataFrame,
    column: str,
    value: int,
    expected_code: str,
) -> None:
    invalid = valid_frame.copy()
    invalid.loc[0, column] = value
    report = validate_dataframe(invalid, contract, strict_profile=False)
    assert expected_code in issue_codes(report)


def test_contract_rejects_predictor_leakage(contract: dict) -> None:
    invalid_contract = deepcopy(contract)
    invalid_contract["column_roles"]["sociodemographic_predictors"].append("v456")
    with pytest.raises(DataContractError, match="overlap leakage exclusions"):
        validate_contract_definition(invalid_contract)
