"""Exploratory subgroup audit for the explicitly deployed legacy artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from src.adapters import PyCaretAdapter
    from src.artifact_registry import (
        PROJECT_ROOT,
        ArtifactManifestError,
        load_deployed_artifacts,
        pycaret_model_stem,
    )
    from src.evaluation import binary_classification_metrics
    from src.feature_contract import MODEL_INPUT_FEATURES, SexCode
except ModuleNotFoundError:
    from adapters import PyCaretAdapter
    from artifact_registry import (
        PROJECT_ROOT,
        ArtifactManifestError,
        load_deployed_artifacts,
        pycaret_model_stem,
    )
    from evaluation import binary_classification_metrics
    from feature_contract import MODEL_INPUT_FEATURES, SexCode


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


def _group_metrics(
    data: pd.DataFrame,
    scores: pd.Series,
    threshold: float,
) -> dict[str, object]:
    metrics = binary_classification_metrics(
        data["HeartDisease"],
        scores.loc[data.index],
        threshold,
    )
    return {
        "count": metrics["row_count"],
        "positive_count": metrics["positive_count"],
        "recall": metrics["recall"],
        "recall_95_ci": metrics["recall_95_ci"],
        "false_negative_rate": 1.0 - float(metrics["recall"]),
        "confusion_matrix": metrics["confusion_matrix"],
    }


def audit_fairness(report_path: Path | None = None) -> Path:
    """Generate a diagnostic report without claiming independent fairness evidence."""

    print("=== Auditoría exploratoria por sexo ===")
    print(
        "ADVERTENCIA: se usa el dataset heredado asociado al modelo, no una "
        "cohorte independiente."
    )

    artifacts = load_deployed_artifacts(
        verify_hashes=True,
        require_training_data=True,
    )
    if artifacts.training_data_path is None:
        raise ArtifactManifestError("The manifest does not declare training data.")

    from pycaret.classification import load_model

    pipeline = load_model(pycaret_model_stem(artifacts.model_path))
    model = PyCaretAdapter(pipeline)
    data = pd.read_parquet(artifacts.training_data_path).rename(columns=RENAME_MAP)

    required = [*MODEL_INPUT_FEATURES, "HeartDisease"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Legacy audit dataset is missing columns: {missing}")

    audit_data = data.loc[:, required].copy()
    target = pd.to_numeric(audit_data["HeartDisease"], errors="coerce")
    valid_target = target.isin([0, 1])
    audit_data = audit_data.loc[valid_target].copy()
    audit_data["HeartDisease"] = target.loc[valid_target].astype(int)

    scores = pd.Series(
        model.predict_proba(audit_data.loc[:, MODEL_INPUT_FEATURES]),
        index=audit_data.index,
        name="score_class_1",
    )
    threshold = artifacts.decision_threshold

    men = audit_data.loc[audit_data["Sex"] == int(SexCode.MALE)]
    women = audit_data.loc[audit_data["Sex"] == int(SexCode.FEMALE)]
    if men.empty or women.empty:
        raise ValueError("Both NHANES sex groups are required (1=male, 2=female).")

    men_metrics = _group_metrics(men, scores, threshold)
    women_metrics = _group_metrics(women, scores, threshold)

    recall_m = float(men_metrics["recall"])
    recall_f = float(women_metrics["recall"])
    positive_m = int(men_metrics["positive_count"])
    positive_f = int(women_metrics["positive_count"])
    equal_opportunity_ratio = (
        recall_f / recall_m
        if recall_m > 0 and positive_m > 0 and positive_f > 0
        else None
    )

    report = {
        "methodology_status": "exploratory_not_independent_validation",
        "generated_from": {
            "model_id": artifacts.model_id,
            "model_path": artifacts.manifest["model"]["path"],
            "model_sha256": artifacts.model_sha256,
            "data_path": artifacts.manifest["training_data"]["path"],
            "threshold": threshold,
        },
        "warning": (
            "These metrics reuse the legacy dataset associated with the model. "
            "They are diagnostic only and cannot establish clinical fairness or "
            "generalization."
        ),
        "groups": {
            "men_sex_1": men_metrics,
            "women_sex_2": women_metrics,
        },
        "fairness_metrics": {
            "equal_opportunity_ratio_recall_female_over_male": (equal_opportunity_ratio)
        },
        "conclusion": "NOT_VALIDATED",
        "required_next_step": (
            "Evaluate the retrained candidate once on a protected independent test "
            "set and report uncertainty for every subgroup."
        ),
    }

    selected_report = (
        report_path or PROJECT_ROOT / "docs" / "fairness_audit_report.json"
    )
    selected_report.parent.mkdir(parents=True, exist_ok=True)
    selected_report.write_text(json.dumps(report, indent=4), encoding="utf-8")
    print(f"Reporte exploratorio guardado en: {selected_report}")
    return selected_report


if __name__ == "__main__":
    try:
        audit_fairness()
    except (ArtifactManifestError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"Fairness audit failed: {exc}") from exc
