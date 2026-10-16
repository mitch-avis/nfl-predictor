"""Tests for sample-weight helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.ml import sample_weights


def test_compute_recency_sample_weight_seasons() -> None:
    """Season-based recency weights decay by half-life."""
    # Arrange
    df = pd.DataFrame({"season": [2021, 2022, 2023], "week": [1, 1, 1]})

    # Act
    weights = sample_weights.compute_recency_sample_weight(df, half_life_seasons=1.0)

    # Assert
    assert weights is not None
    assert weights[2] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5)
    assert weights[0] == pytest.approx(0.25)


def test_combine_sample_weights_multiplies() -> None:
    """Combined weights multiply and ignore None values."""
    # Arrange
    w1 = np.array([1.0, 0.5, 0.25])
    w2 = np.array([1.0, 0.5, 1.0])

    # Act
    combined = sample_weights.combine_sample_weights(w1, None, w2)

    # Assert
    assert combined is not None
    assert combined.tolist() == [1.0, 0.25, 0.25]


def test_compute_postseason_sample_weight_rejects_non_positive_weight() -> None:
    """Postseason weight must be strictly positive."""
    # Arrange
    df = pd.DataFrame({"game_type": ["REG"]})

    # Act & Assert
    with pytest.raises(ValueError, match="postseason_weight must be positive"):
        sample_weights.compute_postseason_sample_weight(
            df,
            include_postseason=True,
            postseason_weight=0.0,
        )


@pytest.mark.parametrize(
    ("df", "include_postseason"),
    [
        (pd.DataFrame({"game_type": ["WC"]}), False),
        (pd.DataFrame({"season": [2024]}), True),
    ],
)
def test_compute_postseason_sample_weight_returns_none_when_not_applicable(
    df: pd.DataFrame,
    *,
    include_postseason: bool,
) -> None:
    """Postseason weights are skipped when disabled or when game_type is unavailable."""
    # Act
    weights = sample_weights.compute_postseason_sample_weight(
        df,
        include_postseason=include_postseason,
        postseason_weight=2.0,
    )

    # Assert
    assert weights is None


def test_compute_postseason_sample_weight_returns_none_for_all_regular_games() -> None:
    """All-regular slates do not need a dedicated sample-weight vector."""
    # Arrange
    df = pd.DataFrame({"game_type": ["REG", "reg"]})

    # Act
    weights = sample_weights.compute_postseason_sample_weight(
        df,
        include_postseason=True,
        postseason_weight=2.0,
    )

    # Assert
    assert weights is None


def test_compute_postseason_sample_weight_returns_none_when_unit_weight_changes_nothing() -> None:
    """A unit postseason weight should collapse back to no sample weights."""
    # Arrange
    df = pd.DataFrame({"game_type": ["REG", "WC"]})

    # Act
    weights = sample_weights.compute_postseason_sample_weight(
        df,
        include_postseason=True,
        postseason_weight=1.0,
    )

    # Assert
    assert weights is None


def test_compute_postseason_sample_weight_marks_postseason_rows() -> None:
    """Postseason rows receive the configured upweight while regular rows stay at one."""
    # Arrange
    df = pd.DataFrame({"game_type": ["REG", "WC", "div"]})

    # Act
    weights = sample_weights.compute_postseason_sample_weight(
        df,
        include_postseason=True,
        postseason_weight=2.5,
    )

    # Assert
    assert weights is not None
    assert weights.tolist() == [1.0, 2.5, 2.5]


def test_compute_recency_sample_weight_returns_none_without_half_life_or_for_empty_frames() -> None:
    """No half-life means there is no recency weighting to apply."""
    # Act
    weights = sample_weights.compute_recency_sample_weight(pd.DataFrame())

    # Assert
    assert weights is None


def test_compute_recency_sample_weight_returns_none_for_an_empty_frame() -> None:
    """No rows means there is no recency weighting to apply."""
    # Arrange
    df = pd.DataFrame({"season": [], "week": []})

    # Act
    weights = sample_weights.compute_recency_sample_weight(df, half_life_seasons=2.0)

    # Assert
    assert weights is None


@pytest.mark.parametrize("half_life", [0.0, -1.0])
def test_compute_recency_sample_weight_rejects_non_positive_half_lives(half_life: float) -> None:
    """A recency half-life must be strictly positive when supplied."""
    # Arrange
    df = pd.DataFrame({"season": [2024], "week": [1]})

    # Act & Assert
    with pytest.raises(ValueError, match="half_life_seasons must be positive"):
        sample_weights.compute_recency_sample_weight(df, half_life_seasons=half_life)


@pytest.mark.parametrize(
    ("df", "match"),
    [
        pytest.param(pd.DataFrame({"week": [1]}), "season column required", id="no_season"),
        pytest.param(
            pd.DataFrame({"season": ["bad"], "week": [1]}),
            "season column contains non-numeric",
            id="non_numeric_season",
        ),
    ],
)
def test_compute_recency_sample_weight_requires_numeric_season_values(
    df: pd.DataFrame, match: str
) -> None:
    """Season-based recency weighting requires a numeric season column."""
    # Act & Assert
    with pytest.raises(ValueError, match=match):
        sample_weights.compute_recency_sample_weight(df, half_life_seasons=1.0)


def test_compute_recency_sample_weight_returns_none_when_all_seasons_are_current() -> None:
    """A single-season slate yields all-one season weights and collapses to None."""
    # Arrange
    df = pd.DataFrame({"season": [2024, 2024], "week": [1, 2]})

    # Act
    weights = sample_weights.compute_recency_sample_weight(df, half_life_seasons=3.0)

    # Assert
    assert weights is None


@pytest.mark.parametrize(
    "weights",
    [
        pytest.param((None, None), id="no_weights"),
        pytest.param((np.ones(2), np.ones(2)), id="all_one_weights"),
    ],
)
def test_combine_sample_weights_returns_none_without_active_weights_or_net_change(
    weights: tuple[np.ndarray | None, ...],
) -> None:
    """Combining no weights or all-one weights should collapse to None."""
    # Act
    combined = sample_weights.combine_sample_weights(*weights)

    # Assert
    assert combined is None
