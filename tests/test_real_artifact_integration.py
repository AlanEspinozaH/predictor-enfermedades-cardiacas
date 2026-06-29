"""Integration tests for the actual deployed PyCaret artifact.

These tests intentionally load ``models/best_pipeline.pkl``. They are skipped
only when the optional runtime dependencies are not installed; in the declared
Python 3.10 project environment they must execute, not use mocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pycaret", reason="PyCaret runtime is required")

from src.adapters import PyCaretAdapter
from src.artifact_registry import (
    PROJECT_ROOT,
    deployed_estimator,
    load_deployed_artifacts,
    load_validated_pycaret_pipeline,
    pipeline_metadata_features,
    validate_deployed_pipeline_contract,
)
from src.feature_contract import MODEL_INPUT_FEATURES, TARGET_COLUMN

pytestmark = pytest.mark.integration


def _canonical_input_frame() -> pd.DataFrame:
    row = {
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
    frame = pd.DataFrame([row], columns=MODEL_INPUT_FEATURES)
    assert tuple(frame.columns) == MODEL_INPUT_FEATURES
    assert TARGET_COLUMN not in frame.columns
    return frame


def _element_by_key(elements, key: str):
    matches = [element for element in elements if element.key == key]
    assert len(matches) == 1, f"Expected exactly one element with key {key!r}"
    return matches[0]


def _rendered_text(container) -> str:
    values: list[str] = []
    for element_type in (
        "title",
        "header",
        "subheader",
        "markdown",
        "caption",
        "info",
        "warning",
        "error",
        "metric",
    ):
        for element in getattr(container, element_type):
            label = getattr(element, "label", "")
            value = getattr(element, "value", "")
            if label:
                values.append(str(label))
            if value:
                values.append(str(value))
    return "\n".join(values)


@pytest.fixture(scope="module")
def real_deployed_pipeline():
    artifacts = load_deployed_artifacts(verify_hashes=True)
    assert artifacts.model_path == PROJECT_ROOT / "models" / "best_pipeline.pkl"

    pipeline = load_validated_pycaret_pipeline(
        artifacts.model_path,
        expected_features=MODEL_INPUT_FEATURES,
    )
    return pipeline


def test_real_artifact_metadata_and_estimator_contract(real_deployed_pipeline):
    metadata = pipeline_metadata_features(real_deployed_pipeline)
    assert metadata is not None
    assert TARGET_COLUMN in metadata

    normalized = validate_deployed_pipeline_contract(real_deployed_pipeline)
    assert normalized == MODEL_INPUT_FEATURES

    estimator = deployed_estimator(real_deployed_pipeline)
    estimator_features = tuple(str(name) for name in estimator.feature_names_in_)
    assert TARGET_COLUMN not in estimator_features
    assert len(estimator_features) == 31


def test_real_artifact_predict_and_predict_proba(real_deployed_pipeline):
    frame = _canonical_input_frame()

    predictions = np.asarray(real_deployed_pipeline.predict(frame))
    probabilities = np.asarray(real_deployed_pipeline.predict_proba(frame))

    assert predictions.shape == (1,)
    assert probabilities.ndim == 2
    assert probabilities.shape[0] == 1
    assert probabilities.shape[1] == 2
    assert np.isfinite(probabilities).all()

    adapter = PyCaretAdapter(real_deployed_pipeline)
    positive_probability = adapter.predict_proba(frame)
    assert positive_probability.shape == (1,)
    assert 0.0 <= positive_probability[0] <= 1.0


def test_streamlit_app_loads_real_artifact_without_contract_error():
    pytest.importorskip("streamlit", reason="Streamlit runtime is required")
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file(
        str(PROJECT_ROOT / "src" / "app.py"),
        default_timeout=120,
    ).run()

    error_messages = [str(element.value) for element in app_test.error]
    forbidden = "Feature contract mismatch (unexpected features: ['HeartDisease'])"

    assert forbidden not in "\n".join(error_messages)
    assert not app_test.exception
    assert not error_messages


def test_streamlit_complete_form_submission_uses_stable_feature_keys():
    pytest.importorskip("streamlit", reason="Streamlit runtime is required")
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file(
        str(PROJECT_ROOT / "src" / "app.py"),
        default_timeout=120,
    ).run()

    assert not app_test.exception
    assert not app_test.error
    assert any(title.value == "CardioHistory ML" for title in app_test.title)

    initial_text = _rendered_text(app_test)
    sidebar_text = _rendered_text(app_test.sidebar)
    assert "Prototipo académico" in initial_text
    assert "Ficha técnica del despliegue" in sidebar_text
    assert "Modelo:" in sidebar_text
    assert "XGBClassifier" in sidebar_text
    assert "Integridad verificada mediante manifiesto y SHA-256" in sidebar_text

    assert [tab.label for tab in app_test.tabs] == [
        "A. Perfil y contexto",
        "B. Mediciones antropométricas y bioquímicas",
        "C. Hábitos",
    ]
    expander_labels = {expander.label for expander in app_test.expander}
    assert expander_labels == {"Cómo funciona el modelo", "Alcance y limitaciones"}

    input_widgets = [
        *app_test.number_input,
        *app_test.slider,
        *app_test.selectbox,
        *app_test.radio,
        *app_test.checkbox,
    ]
    input_keys = [
        widget.key
        for widget in input_widgets
        if isinstance(widget.key, str) and widget.key.startswith("input_")
    ]
    expected_keys = {f"input_{feature}" for feature in MODEL_INPUT_FEATURES}
    assert len(input_keys) == 27
    assert len(set(input_keys)) == 27
    assert set(input_keys) == expected_keys
    assert "input_DiastolicBP" not in input_keys
    assert "input_HDL" in input_keys

    glucose_widget = _element_by_key(app_test.number_input, "input_Glucose")
    assert (
        "Glucosa sérica del perfil bioquímico NHANES (LBXSGL)"
        in glucose_widget.label
    )

    submit_button = _element_by_key(
        app_test.button,
        "submit_academic_classification",
    )
    assert submit_button.label == "Ejecutar clasificación académica"

    row = _canonical_input_frame().iloc[0]
    for feature in (
        "Age",
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
    ):
        value = int(row[feature]) if feature == "Age" else float(row[feature])
        _element_by_key(app_test.number_input, f"input_{feature}").set_value(value)

    for feature in ("IncomeRatio", "SystolicBP"):
        _element_by_key(app_test.slider, f"input_{feature}").set_value(
            float(row[feature])
        )

    _element_by_key(app_test.radio, "input_Sex").set_value("1 — Hombre")
    _element_by_key(app_test.selectbox, "input_Race").set_value(
        "3 — Blanco no hispano"
    )
    _element_by_key(app_test.selectbox, "input_Education").set_value(
        "4 — Estudios superiores incompletos"
    )

    for feature in ("Smoking", "PhysicalActivity", "HealthInsurance", "Alcohol"):
        _element_by_key(app_test.checkbox, f"input_{feature}").set_value(
            bool(row[feature])
        )

    app_test = submit_button.click().run()

    assert not app_test.exception
    assert not app_test.error
    result_text = _rendered_text(app_test)
    assert "Salida del prototipo: clase" in result_text
    assert "Score de la clase positiva" in result_text
    assert "Umbral operativo" in result_text
    assert "0.20" in result_text
    assert "no es diagnóstico" in result_text
    assert "Esta salida no confirma ni descarta una condición médica" in result_text

    normalized_text = result_text.casefold()
    for forbidden in (
        "riesgo futuro",
        "probabilidad de sufrir un infarto",
        "traceback",
        "error de contrato",
        "feature contract mismatch",
    ):
        assert forbidden not in normalized_text
