"""Integration tests for the actual deployed PyCaret artifact.

These tests intentionally load ``models/best_pipeline.pkl``. They are skipped
only when the optional runtime dependencies are not installed; in the declared
Python 3.10 project environment they must execute, not use mocks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pycaret", reason="PyCaret runtime is required")

import src.artifact_registry as artifact_registry_module
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


def _valid_session_snapshot(artifacts) -> dict:
    score = 0.5
    threshold = artifacts.decision_threshold
    return {
        "score": score,
        "deployed_threshold": threshold,
        "deployed_class": int(score >= threshold),
        "submitted_input": _canonical_input_frame().to_dict(orient="records")[0],
        "model_id": artifacts.model_id,
        "model_sha256": artifacts.model_sha256,
    }


def _element_by_key(elements, key: str):
    matches = [element for element in elements if element.key == key]
    assert len(matches) == 1, f"Expected exactly one element with key {key!r}"
    return matches[0]


def _metric_value(container, label: str) -> str:
    matches = [metric for metric in container.metric if metric.label == label]
    assert len(matches) == 1, f"Expected exactly one metric labeled {label!r}"
    return str(matches[0].value)


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
    assert "Estado: prototipo académico desplegado" in sidebar_text.replace("*", "")
    assert "Entrada:" in sidebar_text
    assert "27" in sidebar_text
    assert "Transformación:" in sidebar_text
    assert "31" in sidebar_text
    assert "Clase positiva:" in sidebar_text
    assert "Umbral operativo:" in sidebar_text
    assert "0.20" in sidebar_text
    assert "Integridad verificada mediante manifiesto y SHA-256" in sidebar_text

    initial_rendered_text = "\n".join((initial_text, sidebar_text)).casefold()
    for forbidden in (
        "prototipo heredado",
        "prototipo académico heredado",
        "umbral heredado",
        "nhanes-heart-disease-pycaret-legacy-v1",
    ):
        assert forbidden not in initial_rendered_text

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
        "Glucosa sérica del perfil bioquímico NHANES (LBXSGL)" in glucose_widget.label
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
    _element_by_key(app_test.selectbox, "input_Race").set_value("3 — Blanco no hispano")
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
    assert "Resultado oficial del artefacto desplegado: clase" in result_text
    assert "Score oficial de la clase positiva" in result_text
    assert "Umbral oficial" in result_text
    assert "0.20" in result_text
    assert "no es diagnóstico" in result_text
    assert "Esta salida no confirma ni descarta una condición médica" in result_text

    assert {expander.label for expander in app_test.expander} == {
        "Cómo funciona el modelo",
        "Alcance y limitaciones",
        "Explorador didáctico de la regla de decisión",
        "Auditoría de la decisión desplegada",
    }

    normalized_text = "\n".join(
        (result_text, _rendered_text(app_test.sidebar))
    ).casefold()
    for forbidden in (
        "prototipo heredado",
        "prototipo académico heredado",
        "umbral heredado",
        "riesgo futuro",
        "probabilidad de sufrir un infarto",
        "traceback",
        "error de contrato",
        "feature contract mismatch",
    ):
        assert forbidden not in normalized_text


def test_streamlit_explorer_preserves_official_result_without_reinference(monkeypatch):
    pytest.importorskip("streamlit", reason="Streamlit runtime is required")
    from streamlit.testing.v1 import AppTest

    original_predict_proba = PyCaretAdapter.predict_proba
    predict_proba_calls = 0

    def counting_predict_proba(self, data):
        nonlocal predict_proba_calls
        predict_proba_calls += 1
        return original_predict_proba(self, data)

    monkeypatch.setattr(PyCaretAdapter, "predict_proba", counting_predict_proba)

    app_test = AppTest.from_file(
        str(PROJECT_ROOT / "src" / "app.py"),
        default_timeout=120,
    ).run()

    initial_text = _rendered_text(app_test)
    assert "Resultado oficial del artefacto desplegado" not in initial_text
    assert "Explorador didáctico de la regla de decisión" not in {
        expander.label for expander in app_test.expander
    }
    assert predict_proba_calls == 0

    submit_button = _element_by_key(
        app_test.button,
        "submit_academic_classification",
    )
    app_test = submit_button.click().run()

    assert not app_test.exception
    assert not app_test.error
    assert predict_proba_calls == 1
    official_score = _metric_value(app_test, "Score oficial de la clase positiva")
    official_threshold = _metric_value(app_test, "Umbral oficial")
    official_class = _metric_value(app_test, "Clase oficial")
    assert official_threshold == "0.20"

    explorer_slider = _element_by_key(app_test.slider, "explorer_threshold")
    app_test = explorer_slider.set_value(0.0).run()

    assert not app_test.exception
    assert not app_test.error
    assert predict_proba_calls == 1
    assert _metric_value(app_test, "Score oficial de la clase positiva") == official_score
    assert _metric_value(app_test, "Umbral oficial") == official_threshold
    assert _metric_value(app_test, "Clase oficial") == official_class
    assert _metric_value(app_test, "Umbral simulado") == "0.00"
    assert _metric_value(app_test, "Clase simulada") == "1"
    explorer_text = _rendered_text(app_test)
    assert "Explorador didáctico de la regla de decisión" in {
        expander.label for expander in app_test.expander
    }
    assert "Esta simulación no modifica XGBoost" in explorer_text

    submit_button = _element_by_key(
        app_test.button,
        "submit_academic_classification",
    )
    app_test = submit_button.click().run()

    assert not app_test.exception
    assert not app_test.error
    assert predict_proba_calls == 2
    assert _element_by_key(app_test.slider, "explorer_threshold").value == pytest.approx(
        0.20
    )
    assert app_test.session_state["last_inference"]["model_id"]
    assert app_test.session_state["last_inference"]["model_sha256"]


def test_streamlit_discards_inference_from_different_deployment():
    pytest.importorskip("streamlit", reason="Streamlit runtime is required")
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file(
        str(PROJECT_ROOT / "src" / "app.py"),
        default_timeout=120,
    ).run()
    app_test.session_state["last_inference"] = {
        "score": 0.5,
        "deployed_threshold": 0.2,
        "deployed_class": 1,
        "submitted_input": {},
        "model_id": "different-deployment",
        "model_sha256": "0" * 64,
    }
    app_test.session_state["explorer_threshold"] = 0.9

    app_test = app_test.run()

    assert not app_test.exception
    assert "last_inference" not in app_test.session_state
    assert "explorer_threshold" not in app_test.session_state
    warning_text = "\n".join(str(item.value) for item in app_test.warning)
    assert "resultado anterior fue descartado" in warning_text
    assert "Resultado oficial del artefacto desplegado" not in _rendered_text(app_test)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_score",
        "class_outside_domain",
        "boolean_class",
        "non_finite_score",
        "incomplete_input",
    ],
)
def test_streamlit_discards_partially_corrupt_inference(corruption):
    pytest.importorskip("streamlit", reason="Streamlit runtime is required")
    from streamlit.testing.v1 import AppTest

    artifacts = load_deployed_artifacts(verify_hashes=True)
    app_test = AppTest.from_file(
        str(PROJECT_ROOT / "src" / "app.py"),
        default_timeout=120,
    ).run()
    snapshot = _valid_session_snapshot(artifacts)

    if corruption == "missing_score":
        del snapshot["score"]
    elif corruption == "class_outside_domain":
        snapshot["deployed_class"] = 2
    elif corruption == "boolean_class":
        snapshot["deployed_class"] = True
    elif corruption == "non_finite_score":
        snapshot["score"] = math.nan
    elif corruption == "incomplete_input":
        snapshot["submitted_input"].pop(MODEL_INPUT_FEATURES[-1])

    app_test.session_state["last_inference"] = snapshot
    app_test.session_state["explorer_threshold"] = 0.9
    app_test = app_test.run()

    assert not app_test.exception
    assert "last_inference" not in app_test.session_state
    assert "explorer_threshold" not in app_test.session_state
    assert "Resultado oficial del artefacto desplegado" not in _rendered_text(app_test)
    warning_text = "\n".join(str(item.value) for item in app_test.warning)
    assert "resultado anterior fue descartado" in warning_text


@pytest.mark.parametrize("invalid_threshold", [2.0, math.nan, True, "0.50"])
def test_streamlit_resets_corrupt_explorer_threshold(invalid_threshold):
    pytest.importorskip("streamlit", reason="Streamlit runtime is required")
    from streamlit.testing.v1 import AppTest

    artifacts = load_deployed_artifacts(verify_hashes=True)
    app_test = AppTest.from_file(
        str(PROJECT_ROOT / "src" / "app.py"),
        default_timeout=120,
    ).run()
    app_test.session_state["last_inference"] = _valid_session_snapshot(artifacts)
    app_test.session_state["explorer_threshold"] = invalid_threshold

    app_test = app_test.run()

    assert not app_test.exception
    assert not app_test.error
    assert "last_inference" in app_test.session_state
    assert _element_by_key(app_test.slider, "explorer_threshold").value == pytest.approx(
        artifacts.decision_threshold
    )
    assert "Resultado oficial del artefacto desplegado" in _rendered_text(app_test)


def test_streamlit_reloads_artifact_metadata_on_rerun(monkeypatch):
    pytest.importorskip("streamlit", reason="Streamlit runtime is required")
    from streamlit.testing.v1 import AppTest

    real_artifacts = load_deployed_artifacts(verify_hashes=True)
    artifact_load_calls = 0

    def changing_artifacts(*, verify_hashes=True):
        nonlocal artifact_load_calls
        assert verify_hashes is True
        artifact_load_calls += 1
        if artifact_load_calls == 1:
            return real_artifacts
        return replace(real_artifacts, decision_threshold=0.3)

    monkeypatch.setattr(
        artifact_registry_module,
        "load_deployed_artifacts",
        changing_artifacts,
    )

    app_test = AppTest.from_file(
        str(PROJECT_ROOT / "src" / "app.py"),
        default_timeout=120,
    ).run()
    assert artifact_load_calls == 1
    app_test.session_state["last_inference"] = _valid_session_snapshot(real_artifacts)
    app_test.session_state["explorer_threshold"] = 0.9

    app_test = app_test.run()

    assert not app_test.exception
    assert artifact_load_calls == 2
    assert "last_inference" not in app_test.session_state
    assert "explorer_threshold" not in app_test.session_state
    assert "0.30" in _rendered_text(app_test.sidebar)
    warning_text = "\n".join(str(item.value) for item in app_test.warning)
    assert "resultado anterior fue descartado" in warning_text


def test_streamlit_pipeline_cache_key_includes_model_hash(
    monkeypatch,
    real_deployed_pipeline,
):
    streamlit = pytest.importorskip("streamlit", reason="Streamlit runtime is required")
    from streamlit.testing.v1 import AppTest

    real_artifacts = load_deployed_artifacts(verify_hashes=True)
    changed_hash = "1" * 64
    artifact_load_calls = 0
    pipeline_load_calls = 0

    def changing_artifacts(*, verify_hashes=True):
        nonlocal artifact_load_calls
        assert verify_hashes is True
        artifact_load_calls += 1
        selected_hash = real_artifacts.model_sha256
        if artifact_load_calls >= 2:
            selected_hash = changed_hash
        return replace(real_artifacts, model_sha256=selected_hash)

    def counting_pipeline_loader(model_path, *, expected_features):
        nonlocal pipeline_load_calls
        assert model_path == real_artifacts.model_path
        assert expected_features == MODEL_INPUT_FEATURES
        pipeline_load_calls += 1
        return real_deployed_pipeline

    monkeypatch.setattr(
        artifact_registry_module,
        "load_deployed_artifacts",
        changing_artifacts,
    )
    monkeypatch.setattr(
        artifact_registry_module,
        "load_validated_pycaret_pipeline",
        counting_pipeline_loader,
    )

    streamlit.cache_resource.clear()
    try:
        app_test = AppTest.from_file(
            str(PROJECT_ROOT / "src" / "app.py"),
            default_timeout=120,
        ).run()
        assert pipeline_load_calls == 1

        app_test = app_test.run()
        assert pipeline_load_calls == 2

        app_test = app_test.run()
        assert artifact_load_calls == 3
        assert pipeline_load_calls == 2
        assert not app_test.exception
    finally:
        streamlit.cache_resource.clear()


def test_streamlit_failed_new_inference_removes_previous_result(monkeypatch, caplog):
    pytest.importorskip("streamlit", reason="Streamlit runtime is required")
    from streamlit.testing.v1 import AppTest

    original_predict_proba = PyCaretAdapter.predict_proba
    predict_proba_calls = 0

    def fail_on_second_prediction(self, data):
        nonlocal predict_proba_calls
        predict_proba_calls += 1
        if predict_proba_calls == 2:
            raise ValueError("sensitive internal detail")
        return original_predict_proba(self, data)

    monkeypatch.setattr(PyCaretAdapter, "predict_proba", fail_on_second_prediction)

    app_test = AppTest.from_file(
        str(PROJECT_ROOT / "src" / "app.py"),
        default_timeout=120,
    ).run()
    submit_button = _element_by_key(
        app_test.button,
        "submit_academic_classification",
    )
    app_test = submit_button.click().run()
    assert "last_inference" in app_test.session_state

    submit_button = _element_by_key(
        app_test.button,
        "submit_academic_classification",
    )
    with caplog.at_level(logging.ERROR):
        app_test = submit_button.click().run()

    assert not app_test.exception
    assert predict_proba_calls == 2
    assert "last_inference" not in app_test.session_state
    assert "explorer_threshold" not in app_test.session_state
    assert "Resultado oficial del artefacto desplegado" not in _rendered_text(app_test)
    error_text = "\n".join(str(item.value) for item in app_test.error)
    assert "La inferencia no pudo completarse" in error_text
    assert "sensitive internal detail" not in error_text
    assert "Model inference failed for a validated input" in caplog.text
