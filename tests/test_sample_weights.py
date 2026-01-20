"""Tests for sample-weight helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.ml import sample_weights


def test_compute_recency_sample_weight_weeks() -> None:
    """Week-based recency weights decay by half-life."""

    df = pd.DataFrame({"season": [2023, 2023, 2023], "week": [1, 2, 3]})
    weights = sample_weights.compute_recency_sample_weight(df, half_life_weeks=1.0)

    assert weights is not None
    assert weights[2] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5)
    assert weights[0] == pytest.approx(0.25)


def test_compute_recency_sample_weight_seasons() -> None:
    """Season-based recency weights decay by half-life."""

    df = pd.DataFrame({"season": [2021, 2022, 2023], "week": [1, 1, 1]})
    weights = sample_weights.compute_recency_sample_weight(df, half_life_seasons=1.0)

    assert weights is not None
    assert weights[2] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5)
    assert weights[0] == pytest.approx(0.25)


def test_combine_sample_weights_multiplies() -> None:
    """Combined weights multiply and ignore None values."""

    w1 = np.array([1.0, 0.5, 0.25])
    w2 = np.array([1.0, 0.5, 1.0])
    combined = sample_weights.combine_sample_weights(w1, None, w2)

    assert combined is not None
    assert combined.tolist() == [1.0, 0.25, 0.25]


def test_compute_recency_sample_weight_rejects_multiple_halflives() -> None:
    """Both half-life parameters cannot be set at once."""

    df = pd.DataFrame({"season": [2023], "week": [1]})
    with pytest.raises(ValueError):
        sample_weights.compute_recency_sample_weight(df, half_life_weeks=1.0, half_life_seasons=1.0)
