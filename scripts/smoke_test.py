"""Minimal end-to-end smoke test for the deployed academic classifier."""

# ruff: noqa: E402 -- Direct execution requires the project-root bootstrap below.

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.exceptions import InconsistentVersionWarning

from src.adapters import PyCaretAdapter, UserInputAdapter
from src.artifact_registry import (
    deployed_estimator,
    load_deployed_artifacts,
    load_validated_pycaret_pipeline,
)
from src.feature_contract import MODEL_INPUT_FEATURES


class SmokeTestError(RuntimeError):
    """Raised when a smoke-test invariant is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestError(message)


def _synthetic_input() -> dict[str, Any]:
    """Return the synthetic row already exercised by the integration test."""

    return {
        "Age": 45,
        "IncomeRatio": 2.5,
        "SystolicBP": 120.0,
        "BMI": 24.5,
        "WaistCircumference": 90.0,
        "Height": 175.0,
        "TotalCholesterol": 200.0,
        "Triglycerides": 150.0,
        "LDL": 100.0,
        "HDL": 50.0,
        "HbA1c": 5.5,
        "Glucose": 90.0,
        "Creatinine": 0.9,
        "UricAcid": 5.0,
        "ALT_Enzyme": 25.0,
        "Albumin": 4.5,
        "Potassium": 4.0,
        "Sodium": 140.0,
        "GGT_Enzyme": 30.0,
        "AST_Enzyme": 22.0,
        "Sex": 1,
        "Race": 3,
        "Education": 4,
        "Smoking": 0,
        "PhysicalActivity": 1,
        "HealthInsurance": 1,
        "Alcohol": 0,
    }


def run_smoke_test() -> None:
    artifacts = load_deployed_artifacts(verify_hashes=True)
    print("[OK] Integridad de artefactos verificada")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        pipeline = load_validated_pycaret_pipeline(
            artifacts.model_path,
            expected_features=MODEL_INPUT_FEATURES,
        )

    estimator = deployed_estimator(pipeline)
    model_name = f"{type(estimator).__module__}.{type(estimator).__name__}"
    print(f"[OK] Modelo: {model_name}")

    input_frame = UserInputAdapter(MODEL_INPUT_FEATURES).transform(_synthetic_input())
    _require(len(input_frame.columns) == 27, "La entrada externa no tiene 27 variables")
    _require(
        tuple(input_frame.columns) == MODEL_INPUT_FEATURES,
        "El orden de entrada no coincide con MODEL_INPUT_FEATURES",
    )
    print(f"[OK] Entrada externa: {len(input_frame.columns)} variables")

    transformed_features = getattr(estimator, "feature_names_in_", None)
    _require(
        transformed_features is not None and len(transformed_features) == 31,
        "El estimador final no espera 31 características",
    )
    print("[OK] Entrada transformada esperada: 31 características")

    classes = getattr(estimator, "classes_", None)
    _require(classes is not None, "El estimador final no declara sus clases")
    _require(
        sum(int(value == 1) for value in classes) == 1,
        "La clase positiva 1 no está declarada exactamente una vez",
    )

    adapter = PyCaretAdapter(pipeline, expected_features=MODEL_INPUT_FEATURES)
    scores = adapter.predict_proba(input_frame)
    _require(scores.shape == (1,), "El adaptador no devolvió un único score")
    score = float(scores[0])
    _require(math.isfinite(score), "El score no es finito")
    _require(0.0 <= score <= 1.0, "El score está fuera del intervalo [0, 1]")

    classification = int(score >= artifacts.decision_threshold)
    _require(classification in (0, 1), "La clasificación no es binaria")

    print("[OK] Clase positiva: 1")
    print(f"[OK] Score: {score:.6f}")
    print(f"[OK] Umbral operativo: {artifacts.decision_threshold:.2f}")
    print(f"[OK] Clasificación académica: {classification}")


def main() -> int:
    try:
        run_smoke_test()
    except Exception as exc:
        print(f"[ERROR] Smoke test fallido: {exc}", file=sys.stderr)
        return 1

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
