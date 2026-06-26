"""Tests for serialized-model inference through the canonical adapter."""

import numpy as np
import pandas as pd
import pytest

from src.adapters import PyCaretAdapter
from src.feature_contract import MODEL_INPUT_FEATURES


class FakeProbabilityModel:
    feature_names_in_ = np.array(MODEL_INPUT_FEATURES)
    classes_ = np.array([0, 1])

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        assert tuple(data.columns) == MODEL_INPUT_FEATURES
        return np.array([[0.80, 0.20], [0.25, 0.75]])


class ReversedClassModel(FakeProbabilityModel):
    classes_ = np.array([1, 0])

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        assert tuple(data.columns) == MODEL_INPUT_FEATURES
        return np.array([[0.20, 0.80], [0.75, 0.25]])


class MismatchedFeatureModel(FakeProbabilityModel):
    feature_names_in_ = np.array([*MODEL_INPUT_FEATURES[:-1], "UnexpectedFeature"])


def _input_frame(rows: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        np.zeros((rows, len(MODEL_INPUT_FEATURES))),
        columns=MODEL_INPUT_FEATURES,
    )


def test_predict_proba_returns_positive_class_probability():
    adapter = PyCaretAdapter(FakeProbabilityModel())

    probabilities = adapter.predict_proba(_input_frame())

    np.testing.assert_allclose(probabilities, np.array([0.20, 0.75]))


def test_predict_proba_uses_class_label_not_fixed_column_position():
    adapter = PyCaretAdapter(ReversedClassModel())

    probabilities = adapter.predict_proba(_input_frame())

    np.testing.assert_allclose(probabilities, np.array([0.20, 0.75]))


def test_predict_applies_custom_threshold():
    adapter = PyCaretAdapter(FakeProbabilityModel())

    predictions = adapter.predict(_input_frame(), threshold=0.70)

    np.testing.assert_array_equal(predictions, np.array([0, 1]))


def test_dataframe_column_order_mismatch_is_rejected():
    adapter = PyCaretAdapter(FakeProbabilityModel())
    invalid = _input_frame().loc[:, list(reversed(MODEL_INPUT_FEATURES))]

    with pytest.raises(ValueError, match="canonical order"):
        adapter.predict_proba(invalid)


def test_loaded_model_feature_mismatch_is_rejected_at_construction():
    with pytest.raises(ValueError, match="Feature contract mismatch"):
        PyCaretAdapter(MismatchedFeatureModel())


def test_fit_is_not_supported_for_serialized_models():
    adapter = PyCaretAdapter(FakeProbabilityModel())

    with pytest.raises(NotImplementedError):
        adapter.fit(np.empty((0, len(MODEL_INPUT_FEATURES))), np.empty((0,)))


class TargetMetadataPipeline(FakeProbabilityModel):
    from types import SimpleNamespace

    from src.feature_contract import TARGET_COLUMN

    feature_names_in_ = np.array([*MODEL_INPUT_FEATURES, TARGET_COLUMN])
    named_steps = {
        "actual_estimator": SimpleNamespace(
            feature_names_in_=np.array(
                [*MODEL_INPUT_FEATURES, "Race_1.0", "Race_2.0", "Race_3.0", "Race_4.0"]
            )
        )
    }


def test_adapter_accepts_pycaret_target_only_in_external_metadata():
    adapter = PyCaretAdapter(TargetMetadataPipeline())

    probabilities = adapter.predict_proba(_input_frame())

    np.testing.assert_allclose(probabilities, np.array([0.20, 0.75]))
