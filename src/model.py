"""Educational binary gradient-boosted trees implementation."""

from __future__ import annotations

import numpy as np

from src.interfaces import HeartDiseaseModel
from src.tree.decision_tree import DecisionTree
from src.tree.loss_functions import LogLoss


class XGBoostScratch(HeartDiseaseModel):
    """Small XGBoost-like classifier for mathematical demonstration only."""

    def __init__(
        self,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        lambda_: float = 1.0,
        gamma: float = 0.0,
        scale_pos_weight: float = 1.0,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")
        if lambda_ < 0 or gamma < 0 or scale_pos_weight <= 0:
            raise ValueError("Regularization and class-weight parameters are invalid.")

        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)
        self.scale_pos_weight = float(scale_pos_weight)
        self.trees: list[DecisionTree] = []
        self.loss_func = LogLoss(scale_pos_weight=self.scale_pos_weight)
        self.base_score = 0.5
        self.base_margin = 0.0
        self.n_features_in_: int | None = None

    @staticmethod
    def _validate_training_data(
        X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=int)
        if features.ndim != 2 or target.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D.")
        if len(features) != len(target) or len(target) == 0:
            raise ValueError("X and y must be non-empty and have the same row count.")
        if not np.isfinite(features).all():
            raise ValueError(
                "The scratch implementation does not support missing values."
            )
        if set(np.unique(target)) != {0, 1}:
            raise ValueError("y must contain both binary classes 0 and 1.")
        return features, target

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        features, target = self._validate_training_data(X, y)

        self.trees = []
        self.n_features_in_ = features.shape[1]
        positive_rate = float(np.clip(target.mean(), 1e-6, 1 - 1e-6))
        self.base_score = positive_rate
        self.base_margin = float(np.log(positive_rate / (1 - positive_rate)))
        raw_prediction = np.full(target.shape, self.base_margin, dtype=float)

        for _ in range(self.n_estimators):
            gradient = self.loss_func.gradient(target, raw_prediction)
            hessian = self.loss_func.hessian(target, raw_prediction)
            tree = DecisionTree(
                max_depth=self.max_depth,
                lambda_=self.lambda_,
                gamma=self.gamma,
            )
            tree.fit(features, gradient, hessian)
            self.trees.append(tree)
            raw_prediction += self.learning_rate * tree.predict(features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees or self.n_features_in_ is None:
            raise RuntimeError("The model must be fitted before prediction.")
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features do not match the fitted model.")
        if not np.isfinite(features).all():
            raise ValueError(
                "The scratch implementation does not support missing values."
            )

        raw_prediction = np.full(features.shape[0], self.base_margin, dtype=float)
        for tree in self.trees:
            raw_prediction += self.learning_rate * tree.predict(features)
        return self.loss_func.sigmoid(raw_prediction)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Decision threshold must be between 0 and 1.")
        return (self.predict_proba(X) >= threshold).astype(int)
