"""Adapters for validated user input and serialized model inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.artifact_registry import validate_deployed_pipeline_contract
from src.feature_contract import MODEL_INPUT_FEATURES, validate_feature_names
from src.interfaces import HeartDiseaseModel, InputData


def _model_classes(model: Any) -> np.ndarray | None:
    """Return estimator classes when exposed by a pipeline or final estimator."""

    classes = getattr(model, "classes_", None)
    if classes is not None:
        return np.asarray(classes)

    steps = getattr(model, "steps", None)
    if steps:
        classes = getattr(steps[-1][1], "classes_", None)
        if classes is not None:
            return np.asarray(classes)
    return None


class PyCaretAdapter(HeartDiseaseModel):
    """Expose a loaded sklearn/PyCaret pipeline through a small stable interface."""

    def __init__(
        self,
        model: Any,
        expected_features: Sequence[str] = MODEL_INPUT_FEATURES,
    ) -> None:
        if not hasattr(model, "predict_proba"):
            raise TypeError("The loaded model does not implement predict_proba().")

        self.model = model
        self.expected_features = tuple(str(name) for name in expected_features)
        validate_feature_names(self.expected_features)

        validate_deployed_pipeline_contract(
            model,
            expected_features=self.expected_features,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError(
            "PyCaretAdapter is intended for inference with pre-trained models."
        )

    def _as_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            validate_feature_names(
                tuple(str(column) for column in X.columns),
                expected=self.expected_features,
            )
            return X.loc[:, self.expected_features].copy()

        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError(
                "Model input must be a two-dimensional array or DataFrame."
            )
        if array.shape[1] != len(self.expected_features):
            raise ValueError(
                "Model input column count does not match the canonical feature "
                f"contract: {array.shape[1]} != {len(self.expected_features)}."
            )
        return pd.DataFrame(array, columns=self.expected_features)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return probabilities for class 1 without suppressing model errors."""

        data = self._as_dataframe(X)
        probabilities = np.asarray(self.model.predict_proba(data))
        if probabilities.ndim != 2 or probabilities.shape[0] != len(data):
            raise ValueError("The loaded model returned an invalid probability matrix.")

        classes = _model_classes(self.model)
        if classes is not None:
            positive_positions = np.flatnonzero(classes == 1)
            if len(positive_positions) != 1:
                raise ValueError(
                    "The loaded model does not expose exactly one positive class labeled 1."
                )
            positive_index = int(positive_positions[0])
        elif probabilities.shape[1] == 2:
            positive_index = 1
        else:
            raise ValueError(
                "Positive-class probability cannot be identified from model output."
            )

        positive_probability = probabilities[:, positive_index].astype(float)
        if not np.isfinite(positive_probability).all():
            raise ValueError("The model returned non-finite probability values.")
        if ((positive_probability < 0) | (positive_probability > 1)).any():
            raise ValueError("The model returned probabilities outside [0, 1].")
        return positive_probability

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Decision threshold must be between 0 and 1.")
        return (self.predict_proba(X) >= threshold).astype(int)


class UserInputAdapter:
    """Validate UI data and order it according to the canonical feature contract."""

    def __init__(self, expected_features: Sequence[str] = MODEL_INPUT_FEATURES) -> None:
        self.expected_features = tuple(str(name) for name in expected_features)
        validate_feature_names(self.expected_features)

    def transform(self, user_input: Mapping[str, Any]) -> pd.DataFrame:
        """Return one validated row in the exact model feature order."""

        try:
            validated_data = InputData(**dict(user_input))
        except ValidationError as exc:
            raise ValueError(f"Invalid input data: {exc}") from exc

        data_dict = validated_data.model_dump()
        validate_feature_names(tuple(data_dict), expected=self.expected_features)
        ordered_row = {name: data_dict[name] for name in self.expected_features}
        return pd.DataFrame([ordered_row], columns=list(self.expected_features))
