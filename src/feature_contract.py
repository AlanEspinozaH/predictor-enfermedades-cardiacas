"""Canonical feature contract for the deployed NHANES model.

The contract in this module is the source of truth used by validation,
training, inference, and tests.  ``models/model_config.json`` mirrors these
values as artifact metadata and is checked by the test suite.
"""

from collections.abc import Mapping, Sequence
from enum import IntEnum
from typing import Any

FEATURE_SCHEMA_VERSION = "1.0.0"


class SexCode(IntEnum):
    """NHANES RIAGENDR codes retained by the current trained model."""

    MALE = 1
    FEMALE = 2


MODEL_NUMERIC_FEATURES: tuple[str, ...] = (
    "Age",
    "IncomeRatio",
    "SystolicBP",
    "BMI",
    "WaistCircumference",
    "Height",
    "TotalCholesterol",
    "Triglycerides",
    "LDL",
    "HDL",
    "HbA1c",
    "Glucose",
    "Creatinine",
    "UricAcid",
    "ALT_Enzyme",
    "Albumin",
    "Potassium",
    "Sodium",
    "GGT_Enzyme",
    "AST_Enzyme",
)

MODEL_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Sex",
    "Race",
    "Education",
    "Smoking",
    "PhysicalActivity",
    "HealthInsurance",
    "Alcohol",
)

MODEL_INPUT_FEATURES: tuple[str, ...] = (
    *MODEL_NUMERIC_FEATURES,
    *MODEL_CATEGORICAL_FEATURES,
)

MODEL_IGNORED_FEATURES: tuple[str, ...] = ("SEQN",)
TARGET_COLUMN = "HeartDisease"
MINIMUM_ELIGIBLE_AGE = 20


def feature_names_from_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the ordered input features declared in model metadata.

    New configurations should provide ``input_features`` explicitly.  The
    numeric/categorical fallback keeps older metadata readable while still
    validating it against the canonical contract.
    """

    explicit = config.get("input_features")
    if explicit is not None:
        return tuple(str(name) for name in explicit)

    numeric = config.get("numeric_features", ())
    categorical = config.get("categorical_features", ())
    return tuple(str(name) for name in (*numeric, *categorical))


def validate_feature_names(
    feature_names: Sequence[str],
    *,
    expected: Sequence[str] = MODEL_INPUT_FEATURES,
) -> None:
    """Raise ``ValueError`` when a feature collection violates the contract."""

    actual = tuple(str(name) for name in feature_names)
    expected_tuple = tuple(str(name) for name in expected)

    missing = [name for name in expected_tuple if name not in actual]
    unexpected = [name for name in actual if name not in expected_tuple]

    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing features: {missing}")
        if unexpected:
            details.append(f"unexpected features: {unexpected}")
        raise ValueError("Feature contract mismatch (" + "; ".join(details) + ")")

    if actual != expected_tuple:
        raise ValueError(
            "Feature contract mismatch: columns are valid but not in the "
            "canonical order."
        )
