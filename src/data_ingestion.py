"""Prepare an eligible, unimputed NHANES modeling cohort.

This script intentionally does not fit an imputer.  Missing predictors are
preserved so that imputation can be learned only from development data inside
the training pipeline after the independent test set has been separated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from src.artifact_registry import PROJECT_ROOT
    from src.data_pipeline import (
        TARGET_SOURCE_COLUMN,
        prepare_modeling_cohort,
        provenance_path_for,
        read_tabular_data,
        validate_modeling_cohort,
        write_dataset_provenance,
    )
except ModuleNotFoundError:
    from artifact_registry import PROJECT_ROOT
    from data_pipeline import (
        TARGET_SOURCE_COLUMN,
        prepare_modeling_cohort,
        provenance_path_for,
        read_tabular_data,
        validate_modeling_cohort,
        write_dataset_provenance,
    )


DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "02_processed" / "nhanes_heart_attack_modeling_raw.parquet"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "02_intermediate" / "nhanes_heart_attack_modeling.parquet"
)


def load_and_process_data(
    filepath: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Read, validate, and save the scientific modeling cohort."""

    selected_input = Path(filepath).expanduser().resolve()
    selected_output = Path(output_path).expanduser().resolve()
    if not selected_input.is_file():
        raise FileNotFoundError(f"Input data file not found: {selected_input}")

    print(f"Loading data from {selected_input}...")
    source_data = read_tabular_data(selected_input)
    print(f"Initial shape: {source_data.shape}")

    cohort = prepare_modeling_cohort(source_data)
    validate_modeling_cohort(cohort)

    selected_output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(selected_output, index=False)
    inherited_provenance: dict[str, object] = {}
    inherited_path = provenance_path_for(selected_input)
    if inherited_path.is_file():
        try:
            loaded = json.loads(inherited_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid input provenance JSON: {inherited_path}"
            ) from exc
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Input provenance must be a JSON object: {inherited_path}"
            )
        inherited_provenance = loaded

    raw_target_present = TARGET_SOURCE_COLUMN in source_data.columns
    target_verified = raw_target_present or (
        inherited_provenance.get("target_source") == TARGET_SOURCE_COLUMN
        and inherited_provenance.get("target_provenance_verified") is True
    )
    provenance_path = write_dataset_provenance(
        selected_output,
        {
            "source": str(selected_input),
            "source_provenance": (
                str(inherited_path) if inherited_path.is_file() else None
            ),
            "target_source": (
                TARGET_SOURCE_COLUMN
                if target_verified
                else "unverified_canonical_target"
            ),
            "target_provenance_verified": target_verified,
            "minimum_age": 20,
            "statistical_imputation_applied": False,
            "rows": int(len(cohort)),
            "positive_count": int(cohort["HeartDisease"].sum()),
        },
    )
    print(f"Eligible, unimputed cohort shape: {cohort.shape}")
    print(f"Provenance saved to {provenance_path}")
    print(f"Saved to {selected_output}")
    print(
        "No statistical imputer was fitted. Missing predictors are retained "
        "for training-time preprocessing after the test split."
    )
    return selected_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the eligible NHANES modeling cohort without leakage."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Input CSV, Parquet, or XPT file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output Parquet file.",
    )
    args = parser.parse_args()
    load_and_process_data(args.input, args.output)


if __name__ == "__main__":
    main()
