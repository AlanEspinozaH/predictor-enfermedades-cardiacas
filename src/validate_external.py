"""Evaluate the deployed model or an explicit candidate on an external dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from src.adapters import PyCaretAdapter
    from src.artifact_registry import (
        ArtifactManifestError,
        load_deployed_artifacts,
        pycaret_model_stem,
    )
    from src.data_pipeline import read_tabular_data
    from src.evaluation import binary_classification_metrics, validate_threshold
    from src.feature_contract import MODEL_INPUT_FEATURES
except ModuleNotFoundError:
    from adapters import PyCaretAdapter
    from artifact_registry import (
        ArtifactManifestError,
        load_deployed_artifacts,
        pycaret_model_stem,
    )
    from data_pipeline import read_tabular_data
    from evaluation import binary_classification_metrics, validate_threshold
    from feature_contract import MODEL_INPUT_FEATURES


RENAME_MAP = {
    "TARGET": "HeartDisease",
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
}


def _resolve_model_and_threshold(
    model_path: str | None,
    threshold: float | None,
) -> tuple[Path, str, float]:
    if model_path is None:
        artifacts = load_deployed_artifacts(verify_hashes=True)
        selected_threshold = (
            artifacts.decision_threshold if threshold is None else threshold
        )
        return (
            artifacts.model_path,
            artifacts.model_id,
            validate_threshold(selected_threshold),
        )

    candidate = Path(model_path).expanduser().resolve()
    if candidate.suffix.lower() != ".pkl":
        candidate = candidate.with_suffix(".pkl")
    if not candidate.is_file():
        raise FileNotFoundError(f"Model file not found: {candidate}")
    selected_threshold = 0.5 if threshold is None else threshold
    return (
        candidate,
        f"explicit-candidate:{candidate.name}",
        validate_threshold(selected_threshold),
    )


def validate_external(
    data_path: str,
    output_path: str = "external_predictions.csv",
    model_path: str | None = None,
    threshold: float | None = None,
) -> Path:
    """Save predictions and print metrics when a canonical target is available."""

    selected_model_path, model_id, selected_threshold = _resolve_model_and_threshold(
        model_path,
        threshold,
    )

    from pycaret.classification import load_model

    pipeline = load_model(pycaret_model_stem(selected_model_path))
    model = PyCaretAdapter(pipeline)

    data = read_tabular_data(data_path).rename(columns=RENAME_MAP)
    missing = [
        feature for feature in MODEL_INPUT_FEATURES if feature not in data.columns
    ]
    if missing:
        raise ValueError(f"External dataset is missing canonical features: {missing}")

    features = data.loc[:, MODEL_INPUT_FEATURES].copy()
    scores = model.predict_proba(features)
    labels = (scores >= selected_threshold).astype(int)

    results = data.copy()
    results["prediction_score_1"] = scores
    results["prediction_label"] = labels
    results["decision_threshold"] = selected_threshold
    results["model_id"] = model_id

    if "HeartDisease" in data.columns:
        target = pd.to_numeric(data["HeartDisease"], errors="coerce")
        valid = target.isin([0, 1])
        if not valid.all():
            raise ValueError(
                "HeartDisease must contain only explicit binary labels 0 and 1."
            )
        metrics = binary_classification_metrics(
            target.astype(int),
            scores,
            selected_threshold,
        )
        print(json.dumps(metrics, indent=2))
    else:
        print("No canonical HeartDisease target was supplied; metrics were skipped.")

    selected_output = Path(output_path).expanduser().resolve()
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(selected_output, index=False)
    print(f"Predictions saved to: {selected_output}")
    return selected_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", default="external_predictions.csv")
    parser.add_argument("--model-path")
    parser.add_argument("--threshold", type=float)
    arguments = parser.parse_args()

    try:
        validate_external(
            data_path=arguments.data_path,
            output_path=arguments.output_path,
            model_path=arguments.model_path,
            threshold=arguments.threshold,
        )
    except (
        ArtifactManifestError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise SystemExit(f"External validation failed: {exc}") from exc
