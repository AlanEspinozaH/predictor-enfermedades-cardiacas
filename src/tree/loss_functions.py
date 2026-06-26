"""Second-order binary log-loss used by the educational tree booster."""

from __future__ import annotations

import numpy as np


class LogLoss:
    def __init__(self, scale_pos_weight: float = 1.0) -> None:
        if scale_pos_weight <= 0:
            raise ValueError("scale_pos_weight must be positive.")
        self.scale_pos_weight = float(scale_pos_weight)

    @staticmethod
    def sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(values, dtype=float), -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _weights(self, y_true: np.ndarray) -> np.ndarray:
        return np.where(y_true == 1, self.scale_pos_weight, 1.0)

    def gradient(self, y_true: np.ndarray, raw_score: np.ndarray) -> np.ndarray:
        truth = np.asarray(y_true, dtype=float)
        probability = self.sigmoid(raw_score)
        return (probability - truth) * self._weights(truth)

    def hessian(self, y_true: np.ndarray, raw_score: np.ndarray) -> np.ndarray:
        truth = np.asarray(y_true, dtype=float)
        probability = self.sigmoid(raw_score)
        return probability * (1.0 - probability) * self._weights(truth)

    def loss(self, y_true: np.ndarray, raw_score: np.ndarray) -> float:
        truth = np.asarray(y_true, dtype=float)
        probability = np.clip(self.sigmoid(raw_score), 1e-15, 1 - 1e-15)
        losses = -(
            truth * np.log(probability) + (1.0 - truth) * np.log(1.0 - probability)
        )
        return float(np.mean(losses * self._weights(truth)))
