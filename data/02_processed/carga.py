"""Download and harmonize NHANES cycles for heart-attack classification.

The target is exclusively MCQ160E (ever told you had a heart attack).  MCQ160F
is a stroke question and is never used as a fallback.  The output contains only
participants aged 20 or older with an explicit Yes/No target response.  Missing
predictors are preserved; no statistical imputation is performed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import (  # noqa: E402
    SOURCE_CYCLE_COLUMN,
    TARGET_SOURCE_COLUMN,
    prepare_modeling_cohort,
    read_tabular_data,
    validate_modeling_cohort,
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "02_processed" / "nhanes_heart_attack_modeling_raw.parquet"
)

CYCLES_CONFIG: tuple[dict[str, Any], ...] = (
    {
        "year": "2017-2020",
        "base": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/",
        "files": {
            "DEMO": "P_DEMO.XPT",
            "MCQ": "P_MCQ.XPT",
            "BMX": "P_BMX.XPT",
            "BP": "P_BPXO.XPT",
            "TCHOL": "P_TCHOL.XPT",
            "TRIGLY": "P_TRIGLY.XPT",
            "HDL": "P_HDL.XPT",
            "GHB": "P_GHB.XPT",
            "BIOPRO": "P_BIOPRO.XPT",
            "SMQ": "P_SMQ.XPT",
            "ALQ": "P_ALQ.XPT",
            "PAQ": "P_PAQ.XPT",
            "HIQ": "P_HIQ.XPT",
        },
        "systolic_columns": ("BPXOSY1", "BPXOSY2", "BPXOSY3"),
        "alcohol_column": "ALQ121",
    },
    {
        "year": "2015-2016",
        "base": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/",
        "files": {
            "DEMO": "DEMO_I.XPT",
            "MCQ": "MCQ_I.XPT",
            "BMX": "BMX_I.XPT",
            "BP": "BPX_I.XPT",
            "TCHOL": "TCHOL_I.XPT",
            "TRIGLY": "TRIGLY_I.XPT",
            "HDL": "HDL_I.XPT",
            "GHB": "GHB_I.XPT",
            "BIOPRO": "BIOPRO_I.XPT",
            "SMQ": "SMQ_I.XPT",
            "ALQ": "ALQ_I.XPT",
            "PAQ": "PAQ_I.XPT",
            "HIQ": "HIQ_I.XPT",
        },
        "systolic_columns": ("BPXSY1", "BPXSY2", "BPXSY3"),
        "alcohol_column": "ALQ120Q",
    },
    {
        "year": "2013-2014",
        "base": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/",
        "files": {
            "DEMO": "DEMO_H.XPT",
            "MCQ": "MCQ_H.XPT",
            "BMX": "BMX_H.XPT",
            "BP": "BPX_H.XPT",
            "TCHOL": "TCHOL_H.XPT",
            "TRIGLY": "TRIGLY_H.XPT",
            "HDL": "HDL_H.XPT",
            "GHB": "GHB_H.XPT",
            "BIOPRO": "BIOPRO_H.XPT",
            "SMQ": "SMQ_H.XPT",
            "ALQ": "ALQ_H.XPT",
            "PAQ": "PAQ_H.XPT",
            "HIQ": "HIQ_H.XPT",
        },
        "systolic_columns": ("BPXSY1", "BPXSY2", "BPXSY3"),
        "alcohol_column": "ALQ120Q",
    },
    {
        "year": "2011-2012",
        "base": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/",
        "files": {
            "DEMO": "DEMO_G.XPT",
            "MCQ": "MCQ_G.XPT",
            "BMX": "BMX_G.XPT",
            "BP": "BPX_G.XPT",
            "TCHOL": "TCHOL_G.XPT",
            "TRIGLY": "TRIGLY_G.XPT",
            "HDL": "HDL_G.XPT",
            "GHB": "GHB_G.XPT",
            "BIOPRO": "BIOPRO_G.XPT",
            "SMQ": "SMQ_G.XPT",
            "ALQ": "ALQ_G.XPT",
            "PAQ": "PAQ_G.XPT",
            "HIQ": "HIQ_G.XPT",
        },
        "systolic_columns": ("BPXSY1", "BPXSY2", "BPXSY3"),
        "alcohol_column": "ALQ120Q",
    },
)

COMPONENT_COLUMNS: dict[str, tuple[str, ...]] = {
    "DEMO": ("RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2", "INDFMPIR"),
    "MCQ": (TARGET_SOURCE_COLUMN,),
    "BMX": ("BMXBMI", "BMXWAIST", "BMXHT"),
    "TCHOL": ("LBXTC",),
    "TRIGLY": ("LBXTR", "LBDLDL"),
    "HDL": ("LBDHDD",),
    "GHB": ("LBXGH",),
    "BIOPRO": (
        "LBXSGL",
        "LBXSCR",
        "LBXSUA",
        "LBXSATSI",
        "LBXSAL",
        "LBXSKSI",
        "LBXSNASI",
        "LBXSGTSI",
        "LBXSASSI",
    ),
    "SMQ": ("SMQ020",),
    "PAQ": ("PAQ650",),
    "HIQ": ("HIQ011",),
}


def _select_component_columns(
    data: pd.DataFrame,
    requested: tuple[str, ...],
    *,
    component: str,
    cycle: str,
) -> pd.DataFrame:
    if "SEQN" not in data.columns:
        raise ValueError(f"{component} {cycle} does not contain SEQN")
    present = [column for column in requested if column in data.columns]
    missing = [column for column in requested if column not in data.columns]
    if missing:
        print(f"  WARNING {component} {cycle}: missing columns {missing}")
    return data.loc[:, ["SEQN", *present]].copy()


def _load_component(
    cycle: dict[str, Any],
    component: str,
    *,
    required: bool = False,
) -> pd.DataFrame:
    url = cycle["base"] + cycle["files"][component]
    try:
        data = read_tabular_data(url)
        requested = (cycle["alcohol_column"],) if component == "ALQ" else COMPONENT_COLUMNS[component]
        return _select_component_columns(
            data,
            requested,
            component=component,
            cycle=cycle["year"],
        )
    except Exception as exc:
        if required:
            raise RuntimeError(
                f"Required component {component} failed for {cycle['year']}: {exc}"
            ) from exc
        print(f"  WARNING could not load {component} for {cycle['year']}: {exc}")
        return pd.DataFrame(columns=["SEQN"])


def _load_blood_pressure(cycle: dict[str, Any]) -> pd.DataFrame:
    url = cycle["base"] + cycle["files"]["BP"]
    try:
        data = read_tabular_data(url)
    except Exception as exc:
        print(f"  WARNING could not load BP for {cycle['year']}: {exc}")
        return pd.DataFrame(columns=["SEQN", "SystolicBP"])

    if "SEQN" not in data.columns:
        return pd.DataFrame(columns=["SEQN", "SystolicBP"])
    available = [
        column for column in cycle["systolic_columns"] if column in data.columns
    ]
    if not available:
        print(f"  WARNING BP {cycle['year']}: no systolic measurements found")
        return pd.DataFrame(columns=["SEQN", "SystolicBP"])

    result = data.loc[:, ["SEQN", *available]].copy()
    result["SystolicBP"] = result[available].mean(axis=1, skipna=True)
    return result.loc[:, ["SEQN", "SystolicBP"]]


def build_cycle_cohort(cycle: dict[str, Any]) -> pd.DataFrame:
    """Download and construct one eligible cycle without imputing predictors."""

    print(f"\nProcessing NHANES cycle {cycle['year']}...")
    demo = _load_component(cycle, "DEMO", required=True)
    mcq = _load_component(cycle, "MCQ", required=True)
    if TARGET_SOURCE_COLUMN not in mcq.columns:
        raise RuntimeError(
            f"{TARGET_SOURCE_COLUMN} is absent from MCQ for {cycle['year']}. "
            "The cycle is not eligible for the heart-attack target."
        )

    merged = demo.merge(mcq, on="SEQN", how="inner", validate="one_to_one")
    optional_components = (
        _load_blood_pressure(cycle),
        _load_component(cycle, "BMX"),
        _load_component(cycle, "TCHOL"),
        _load_component(cycle, "TRIGLY"),
        _load_component(cycle, "HDL"),
        _load_component(cycle, "GHB"),
        _load_component(cycle, "BIOPRO"),
        _load_component(cycle, "SMQ"),
        _load_component(cycle, "PAQ"),
        _load_component(cycle, "HIQ"),
        _load_component(cycle, "ALQ"),
    )
    for component in optional_components:
        merged = merged.merge(component, on="SEQN", how="left", validate="one_to_one")

    merged[SOURCE_CYCLE_COLUMN] = cycle["year"]
    cohort = prepare_modeling_cohort(merged)
    print(
        f"  Eligible rows: {len(cohort)}; "
        f"positive target: {int(cohort['HeartDisease'].sum())}"
    )
    return cohort


def build_all_cycles(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    cohorts: list[pd.DataFrame] = []
    cycle_summaries: list[dict[str, Any]] = []

    for cycle in CYCLES_CONFIG:
        cohort = build_cycle_cohort(cycle)
        cohorts.append(cohort)
        cycle_summaries.append(
            {
                "cycle": cycle["year"],
                "rows": int(len(cohort)),
                "positive_count": int(cohort["HeartDisease"].sum()),
            }
        )

    combined = pd.concat(cohorts, ignore_index=True)
    validate_modeling_cohort(combined)

    selected_output = Path(output_path).expanduser().resolve()
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(selected_output, index=False)

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "CDC NHANES public XPT files",
        "cycles": cycle_summaries,
        "target_source": TARGET_SOURCE_COLUMN,
        "target_provenance_verified": True,
        "target_definition": {
            "1": "Yes, participant reports being told they had a heart attack",
            "0": "No, participant reports not being told they had a heart attack",
            "excluded": ["Refused", "Don't know", "missing", "not eligible"],
        },
        "minimum_age": 20,
        "alcohol_definition": (
            "Any alcohol use in the past 12 months, harmonized from "
            "ALQ120Q (2011-2016) and ALQ121 (2017-2020)."
        ),
        "statistical_imputation_applied": False,
        "output": selected_output.name,
        "rows": int(len(combined)),
        "positive_count": int(combined["HeartDisease"].sum()),
    }
    provenance_path = selected_output.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=4), encoding="utf-8")

    print(f"\nSaved eligible, unimputed cohort to {selected_output}")
    print(f"Saved provenance metadata to {provenance_path}")
    return selected_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and harmonize NHANES heart-attack data."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output Parquet path.",
    )
    args = parser.parse_args()
    build_all_cycles(args.output)


if __name__ == "__main__":
    main()
