import numpy as np
import pytest

from src.evaluation import (
    binary_classification_metrics,
    validate_threshold,
    wilson_interval,
)


def test_binary_metrics_use_requested_threshold():
    metrics = binary_classification_metrics(
        y_true=np.array([0, 0, 1, 1]),
        y_score=np.array([0.1, 0.7, 0.6, 0.9]),
        threshold=0.65,
    )

    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["threshold"] == pytest.approx(0.65)


def test_wilson_interval_is_bounded():
    interval = wilson_interval(successes=8, trials=10)

    assert interval is not None
    assert 0.0 <= interval[0] <= interval[1] <= 1.0
    assert interval[0] < 0.8 < interval[1]


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        validate_threshold(1.01)
