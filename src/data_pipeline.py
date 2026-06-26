"""Scientific data preparation helpers for the NHANES heart-attack prototype.

This module deliberately separates cohort construction from model fitting.
Predictor values may remain missing here; imputation must be learned inside the
training pipeline after the independent test set has been separated.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.feature_contract import (
        MINIMUM_ELIGIBLE_AGE,
        MODEL_INPUT_FEATURES,
        MODEL_NUMERIC_FEATURES,
        TARGET_COLUMN,
    )
except ModuleNotFoundError:
    from feature_contract import (
        MINIMUM_ELIGIBLE_AGE,
        MODEL_INPUT_FEATURES,
        MODEL_NUMERIC_FEATURES,
        TARGET_COLUMN,
    )


TARGET_SOURCE_COLUMN = "MCQ160E"
IDENTIFIER_COLUMN = "SEQN"
SOURCE_CYCLE_COLUMN = "NHANESCycle"

BINARY_FEATURES: tuple[str, ...] = (
    "Smoking",
    "PhysicalActivity",
    "HealthInsurance",
    "Alcohol",
)

CANONICAL_RENAME_MAP: Mapping[str, str] = {
    # Legacy Spanish names.
    "TARGET": TARGET_COLUMN,
    "Edad": "Age",
    "Sexo": "Sex",
    "Raza": "Race",
    "Educacion": "Education",
    "Ingresos_Ratio": "IncomeRatio",
    "Presion_Sistolica": "SystolicBP",
    "Cintura": "WaistCircumference",
    "Altura": "Height",
    "Colesterol_Total": "TotalCholesterol",
    "Trigliceridos": "Triglycerides",
    "Glucosa": "Glucose",
    "Creatinina": "Creatinine",
    "Acido_Urico": "UricAcid",
    "Enzima_ALT": "ALT_Enzyme",
    "Albumina": "Albumin",
    "Potasio": "Potassium",
    "Sodio": "Sodium",
    "Enzima_GGT": "GGT_Enzyme",
    "Enzima_AST": "AST_Enzyme",
    "Fumador": "Smoking",
    "Actividad_Fisica": "PhysicalActivity",
    "Seguro_Medico": "HealthInsurance",
    # Raw NHANES names.
    "RIAGENDR": "Sex",
    "RIDAGEYR": "Age",
    "RIDRETH1": "Race",
    "DMDEDUC2": "Education",
    "INDFMPIR": "IncomeRatio",
    "BMXBMI": "BMI",
    "BMXWAIST": "WaistCircumference",
    "BMXHT": "Height",
    "LBXTC": "TotalCholesterol",
    "LBXTR": "Triglycerides",
    "LBDLDL": "LDL",
    "LBDHDD": "HDL",
    "LBXGH": "HbA1c",
    "LBXSGL": "Glucose",
    "LBXSCR": "Creatinine",
    "LBXSUA": "UricAcid",
    "LBXSATSI": "ALT_Enzyme",
    "LBXSAL": "Albumin",
    "LBXSKSI": "Potassium",
    "LBXSNASI": "Sodium",
    "LBXSGTSI": "GGT_Enzyme",
    "LBXSASSI": "AST_Enzyme",
    "SMQ020": "Smoking",
    "PAQ650": "PhysicalActivity",
    "HIQ011": "HealthInsurance",
}


@dataclass(frozen=True)
class DataSplit:
    """Development/test partition and reproducibility metadata."""

    development: pd.DataFrame
    independent_test: pd.DataFrame
    metadata: Mapping[str, Any]


def provenance_path_for(data_path: str | Path) -> Path:
    """Return the sidecar provenance path for a dataset artifact."""

    return Path(data_path).expanduser().resolve().with_suffix(".provenance.json")


def write_dataset_provenance(
    data_path: str | Path,
    metadata: Mapping[str, Any],
) -> Path:
    """Write a deterministic JSON sidecar next to a dataset."""

    output = provenance_path_for(data_path)
    output.write_text(json.dumps(dict(metadata), indent=4), encoding="utf-8")
    return output


def load_training_provenance(data_path: str | Path) -> Mapping[str, Any]:
    """Load and validate the minimum provenance required for training."""

    path = provenance_path_for(data_path)
    if not path.is_file():
        raise ValueError(
            f"Training provenance sidecar not found: {path}. Rebuild the "
            "dataset with data/02_processed/carga.py."
        )
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid provenance JSON: {path}") from exc

    errors: list[str] = []
    if metadata.get("target_source") != TARGET_SOURCE_COLUMN:
        errors.append(f"target_source must be {TARGET_SOURCE_COLUMN}")
    if metadata.get("target_provenance_verified") is not True:
        errors.append("target_provenance_verified must be true")
    try:
        minimum_age = int(metadata.get("minimum_age", -1))
    except (TypeError, ValueError):
        minimum_age = -1
    if minimum_age < MINIMUM_ELIGIBLE_AGE:
        errors.append(f"minimum_age must be at least {MINIMUM_ELIGIBLE_AGE}")
    if metadata.get("statistical_imputation_applied") is not False:
        errors.append("statistical_imputation_applied must be false")
    if errors:
        raise ValueError("Training provenance rejected: " + "; ".join(errors))
    return metadata


def read_tabular_data(source: str | Path) -> pd.DataFrame:
    """Read CSV, Parquet, or SAS transport data from a path or URL.

    ``pandas.read_sas`` returns a single DataFrame.  Keeping that behavior here
    prevents the previous erroneous tuple unpacking in ``data_ingestion.py``.
    """

    source_text = str(source)
    parsed_path = urlparse(source_text).path if "://" in source_text else source_text
    suffix = Path(parsed_path).suffix.lower()

    if suffix == ".xpt":
        return pd.read_sas(source_text, format="xport", encoding="latin1")
    if suffix == ".parquet":
        return pd.read_parquet(source_text)
    if suffix == ".csv":
        return pd.read_csv(source_text)
    raise ValueError(f"Unsupported data format: {source_text}")


def normalize_feature_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with known legacy/raw columns renamed canonically."""

    frame = data.copy()
    effective_map = {
        source: destination
        for source, destination in CANONICAL_RENAME_MAP.items()
        if source in frame.columns and destination not in frame.columns
    }
    return frame.rename(columns=effective_map)


def encode_heart_attack_target(values: pd.Series) -> pd.Series:
    """Encode NHANES MCQ160E: 1=Yes, 2=No; all other codes are missing."""

    numeric = pd.to_numeric(values, errors="coerce")
    encoded = numeric.map({1.0: 1, 2.0: 0})
    return encoded.astype("Int8")


def _validate_existing_binary_target(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    encoded = numeric.where(numeric.isin([0, 1]))
    return encoded.astype("Int8")


def normalize_binary_feature(values: pd.Series) -> pd.Series:
    """Normalize canonical 0/1 or raw NHANES 1/2 binary coding.

    Refused/unknown codes become missing. Other values, including fractional
    values produced by a previous whole-dataset imputer, are rejected.
    """

    numeric = pd.to_numeric(values, errors="coerce")
    allowed = {0, 1, 2, 7, 9}
    invalid = sorted(set(numeric.dropna().unique()) - allowed)
    if invalid:
        raise ValueError(f"Binary feature contains invalid values: {invalid}")

    normalized = pd.Series(pd.NA, index=values.index, dtype="Int8")
    normalized.loc[numeric == 1] = 1
    normalized.loc[numeric.isin([0, 2])] = 0
    return normalized


def encode_alcohol_past_year(data: pd.DataFrame) -> pd.Series | None:
    """Harmonize any alcohol use in the past 12 months across cycles.

    NHANES 2011-2016 uses ALQ120Q (quantity/frequency, with zero meaning
    never).  The 2017-March 2020 file uses ALQ121 (categorical frequency,
    again with zero meaning never).  Refused, unknown, and missing responses
    remain missing.
    """

    if "ALQ121" in data.columns:
        values = pd.to_numeric(data["ALQ121"], errors="coerce")
        result = pd.Series(pd.NA, index=data.index, dtype="Int8")
        result.loc[values == 0] = 0
        result.loc[values.between(1, 10, inclusive="both")] = 1
        return result

    if "ALQ120Q" in data.columns:
        values = pd.to_numeric(data["ALQ120Q"], errors="coerce")
        result = pd.Series(pd.NA, index=data.index, dtype="Int8")
        result.loc[values == 0] = 0
        result.loc[values.between(1, 366, inclusive="both")] = 1
        return result

    return None


def derive_ldl_with_friedewald(data: pd.DataFrame) -> pd.DataFrame:
    """Fill eligible missing LDL values without fitting a statistical imputer.

    The calculation is restricted to non-negative triglycerides below
    400 mg/dL and to derived values between zero and total cholesterol.
    Existing measured LDL values are never overwritten.
    """

    frame = data.copy()
    required = {"LDL", "TotalCholesterol", "HDL", "Triglycerides"}
    if not required.issubset(frame.columns):
        return frame

    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    calculated = frame["TotalCholesterol"] - frame["HDL"] - frame["Triglycerides"] / 5.0
    eligible = (
        frame["LDL"].isna()
        & frame["TotalCholesterol"].notna()
        & frame["HDL"].notna()
        & frame["Triglycerides"].between(0, 400, inclusive="left")
        & calculated.ge(0)
        & calculated.le(frame["TotalCholesterol"])
    )
    frame.loc[eligible, "LDL"] = calculated.loc[eligible]
    return frame


def prepare_modeling_cohort(data: pd.DataFrame) -> pd.DataFrame:
    """Construct the eligible, unimputed modeling cohort.

    The authoritative target is MCQ160E whenever that raw column is present.
    Participants younger than 20 and target responses other than explicit
    Yes/No are excluded.  Missing predictors are retained for training-time
    imputation.
    """

    frame = normalize_feature_columns(data)

    if TARGET_SOURCE_COLUMN in frame.columns:
        frame[TARGET_COLUMN] = encode_heart_attack_target(frame[TARGET_SOURCE_COLUMN])
    elif TARGET_COLUMN in frame.columns:
        frame[TARGET_COLUMN] = _validate_existing_binary_target(frame[TARGET_COLUMN])
    else:
        raise ValueError(
            f"Target source '{TARGET_SOURCE_COLUMN}' or canonical target "
            f"'{TARGET_COLUMN}' is required."
        )

    if "Age" not in frame.columns:
        raise ValueError("Age/RIDAGEYR is required to define the eligible cohort.")

    frame["Age"] = pd.to_numeric(frame["Age"], errors="coerce")
    frame = frame.loc[
        frame["Age"].ge(MINIMUM_ELIGIBLE_AGE) & frame[TARGET_COLUMN].notna()
    ].copy()

    for column in MODEL_NUMERIC_FEATURES:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    harmonized_alcohol = encode_alcohol_past_year(frame)
    if harmonized_alcohol is not None:
        frame["Alcohol"] = harmonized_alcohol

    for column in BINARY_FEATURES:
        if column in frame.columns:
            frame[column] = normalize_binary_feature(frame[column])

    categorical_domains = {
        "Sex": {1, 2},
        "Race": {1, 2, 3, 4, 5},
        "Education": {1, 2, 3, 4, 5},
    }
    for column, valid_codes in categorical_domains.items():
        if column in frame.columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            known_missing_codes = {7, 9}
            invalid = sorted(
                set(numeric.dropna().unique()) - valid_codes - known_missing_codes
            )
            if invalid:
                raise ValueError(
                    f"Categorical feature '{column}' contains invalid values: {invalid}"
                )
            frame[column] = numeric.where(numeric.isin(valid_codes)).astype("Int8")

    frame = derive_ldl_with_friedewald(frame)

    required_columns = [*MODEL_INPUT_FEATURES, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in frame]
    if missing_columns:
        raise ValueError(
            "Dataset does not contain the canonical modeling columns. "
            f"Missing: {missing_columns}"
        )

    leading_columns = [
        column
        for column in (IDENTIFIER_COLUMN, SOURCE_CYCLE_COLUMN)
        if column in frame.columns
    ]
    result = frame.loc[:, [*leading_columns, *required_columns]].copy()
    result[TARGET_COLUMN] = result[TARGET_COLUMN].astype("Int8")
    return result.reset_index(drop=True)


def validate_modeling_cohort(
    data: pd.DataFrame,
    *,
    require_both_classes: bool = True,
) -> None:
    """Fail fast when a cohort violates the scientific data contract."""

    required = [*MODEL_INPUT_FEATURES, TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing canonical modeling columns: {missing}")

    age = pd.to_numeric(data["Age"], errors="coerce")
    if age.isna().any() or age.lt(MINIMUM_ELIGIBLE_AGE).any():
        raise ValueError(
            f"Modeling cohort must contain only participants aged "
            f"{MINIMUM_ELIGIBLE_AGE} or older."
        )

    target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only explicit binary values 0 and 1.")
    if require_both_classes and set(target.unique()) != {0, 1}:
        raise ValueError("Modeling cohort must contain both target classes.")

    domains = {
        "Sex": {1, 2},
        "Race": {1, 2, 3, 4, 5},
        "Education": {1, 2, 3, 4, 5},
        **{column: {0, 1} for column in BINARY_FEATURES},
    }
    for column, allowed in domains.items():
        non_missing = pd.to_numeric(data[column], errors="coerce").dropna()
        invalid = sorted(set(non_missing.unique()) - allowed)
        if invalid:
            raise ValueError(
                f"Categorical feature '{column}' contains invalid values: {invalid}"
            )

    if IDENTIFIER_COLUMN in data.columns:
        if SOURCE_CYCLE_COLUMN in data.columns:
            duplicated = data.duplicated([SOURCE_CYCLE_COLUMN, IDENTIFIER_COLUMN])
        else:
            duplicated = data.duplicated(IDENTIFIER_COLUMN)
        if duplicated.any():
            raise ValueError("Duplicate participant identifiers found in the cohort.")


def _cycle_sort_key(value: Any) -> tuple[int, ...]:
    numbers = tuple(int(number) for number in re.findall(r"\d{4}", str(value)))
    return numbers or (-1,)


def split_development_and_test(
    data: pd.DataFrame,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> DataSplit:
    """Create a test set before any imputation, tuning, or model selection.

    When multiple NHANES cycles are available, the latest cycle is held out
    temporally.  Otherwise a reproducible stratified random split is used.
    """

    validate_modeling_cohort(data)
    frame = data.reset_index(drop=True).copy()

    strategy = "stratified_random"
    held_out_cycles: list[str] = []

    if SOURCE_CYCLE_COLUMN in frame.columns:
        cycles = [value for value in frame[SOURCE_CYCLE_COLUMN].dropna().unique()]
        if len(cycles) >= 2:
            latest_cycle = max(cycles, key=_cycle_sort_key)
            temporal_test = frame.loc[frame[SOURCE_CYCLE_COLUMN] == latest_cycle].copy()
            temporal_development = frame.loc[
                frame[SOURCE_CYCLE_COLUMN] != latest_cycle
            ].copy()
            if (
                not temporal_test.empty
                and not temporal_development.empty
                and set(temporal_test[TARGET_COLUMN].unique()) == {0, 1}
                and set(temporal_development[TARGET_COLUMN].unique()) == {0, 1}
            ):
                development = temporal_development
                independent_test = temporal_test
                strategy = "temporal_latest_nhanes_cycle"
                held_out_cycles = [str(latest_cycle)]
            else:
                development, independent_test = train_test_split(
                    frame,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=frame[TARGET_COLUMN],
                )
        else:
            development, independent_test = train_test_split(
                frame,
                test_size=test_size,
                random_state=random_state,
                stratify=frame[TARGET_COLUMN],
            )
    else:
        development, independent_test = train_test_split(
            frame,
            test_size=test_size,
            random_state=random_state,
            stratify=frame[TARGET_COLUMN],
        )

    development = development.reset_index(drop=True)
    independent_test = independent_test.reset_index(drop=True)

    if IDENTIFIER_COLUMN in frame.columns:
        development_ids = set(development[IDENTIFIER_COLUMN].dropna())
        test_ids = set(independent_test[IDENTIFIER_COLUMN].dropna())
        if development_ids.intersection(test_ids):
            raise ValueError("Participant leakage detected across data partitions.")

    metadata = {
        "strategy": strategy,
        "random_state": random_state,
        "requested_test_size": test_size,
        "development_rows": int(len(development)),
        "independent_test_rows": int(len(independent_test)),
        "development_positive_count": int(development[TARGET_COLUMN].sum()),
        "independent_test_positive_count": int(independent_test[TARGET_COLUMN].sum()),
        "held_out_cycles": held_out_cycles,
        "imputation_fitted_before_split": False,
    }
    return DataSplit(
        development=development,
        independent_test=independent_test,
        metadata=metadata,
    )
