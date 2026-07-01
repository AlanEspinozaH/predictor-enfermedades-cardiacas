"""Pure decision-rule utilities for model scores."""

from __future__ import annotations

import math
from numbers import Real


def _unit_interval_value(value: Real, *, name: str) -> float:
    """Return a finite numeric value in the closed unit interval."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a non-boolean numeric value")
    try:
        numeric_value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return numeric_value


def classify_score(score: Real, threshold: Real) -> int:
    """Apply a binary threshold to a finite score in the unit interval.

    Both values must be non-boolean real numbers in ``[0, 1]``. Textual values,
    including numeric strings and bytes, are rejected. The function returns
    class ``1`` at or above the threshold and class ``0`` below it.
    """

    validated_score = _unit_interval_value(score, name="score")
    validated_threshold = _unit_interval_value(threshold, name="threshold")
    return int(validated_score >= validated_threshold)
