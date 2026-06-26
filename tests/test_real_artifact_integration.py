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
