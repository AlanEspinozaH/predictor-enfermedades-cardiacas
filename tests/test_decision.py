"""Unit tests for the pure decision rule."""

import math

import pytest

from src.decision import classify_score


def test_score_below_threshold_returns_class_zero():
    assert classify_score(0.19, 0.20) == 0


def test_score_equal_to_threshold_returns_class_one():
    assert classify_score(0.20, 0.20) == 1


def test_score_above_threshold_returns_class_one():
    assert classify_score(0.21, 0.20) == 1


@pytest.mark.parametrize(
    ("score", "threshold", "expected_class"),
    [
        (0.0, 0.0, 1),
        (0.0, 1.0, 0),
        (1.0, 0.0, 1),
        (1.0, 1.0, 1),
    ],
)
def test_closed_unit_interval_boundaries_are_valid(
    score,
    threshold,
    expected_class,
):
    assert classify_score(score, threshold) == expected_class


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_score_outside_unit_interval_is_rejected(score):
    with pytest.raises(ValueError, match="score must be between 0 and 1"):
        classify_score(score, 0.20)


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_threshold_outside_unit_interval_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        classify_score(0.20, threshold)


@pytest.mark.parametrize(
    ("score", "threshold"),
    [
        (math.nan, 0.20),
        (math.inf, 0.20),
        (-math.inf, 0.20),
        (0.20, math.nan),
        (0.20, math.inf),
        (0.20, -math.inf),
    ],
)
def test_non_finite_values_are_rejected(score, threshold):
    with pytest.raises(ValueError, match="must be finite"):
        classify_score(score, threshold)


@pytest.mark.parametrize(
    ("score", "threshold"),
    [
        (True, 0.20),
        (False, 0.20),
        (0.20, True),
        (0.20, False),
    ],
)
def test_boolean_values_are_rejected(score, threshold):
    with pytest.raises(ValueError, match="non-boolean numeric value"):
        classify_score(score, threshold)


@pytest.mark.parametrize("value", ["0.20", "abc", b"0.20"])
def test_textual_scores_are_rejected(value):
    with pytest.raises(ValueError, match="non-boolean numeric value"):
        classify_score(value, 0.20)


@pytest.mark.parametrize("value", ["0.20", "abc", b"0.20"])
def test_textual_thresholds_are_rejected(value):
    with pytest.raises(ValueError, match="non-boolean numeric value"):
        classify_score(0.20, value)
