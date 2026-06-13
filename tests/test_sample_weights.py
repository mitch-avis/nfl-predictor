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


def test_compute_postseason_sample_weight_rejects_non_positive_weight() -> None:
    """Postseason weight must be strictly positive."""
    df = pd.DataFrame({"game_type": ["REG"]})

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
    include_postseason: bool,
) -> None:
    """Postseason weights are skipped when disabled or when game_type is unavailable."""
    assert (
        sample_weights.compute_postseason_sample_weight(
            df,
            include_postseason=include_postseason,
            postseason_weight=2.0,
        )
        is None
    )


def test_compute_postseason_sample_weight_returns_none_for_all_regular_games() -> None:
    """All-regular slates do not need a dedicated sample-weight vector."""
    df = pd.DataFrame({"game_type": ["REG", "reg"]})

    assert (
        sample_weights.compute_postseason_sample_weight(
            df,
            include_postseason=True,
            postseason_weight=2.0,
        )
        is None
    )


def test_compute_postseason_sample_weight_returns_none_when_unit_weight_changes_nothing() -> None:
    """A unit postseason weight should collapse back to no sample weights."""
    df = pd.DataFrame({"game_type": ["REG", "WC"]})

    assert (
        sample_weights.compute_postseason_sample_weight(
            df,
            include_postseason=True,
            postseason_weight=1.0,
        )
        is None
    )


def test_compute_postseason_sample_weight_marks_postseason_rows() -> None:
    """Postseason rows receive the configured upweight while regular rows stay at one."""
    df = pd.DataFrame({"game_type": ["REG", "WC", "div"]})

    weights = sample_weights.compute_postseason_sample_weight(
        df,
        include_postseason=True,
        postseason_weight=2.5,
    )

    assert weights is not None
    assert weights.tolist() == [1.0, 2.5, 2.5]


def test_compute_recency_sample_weight_returns_none_without_half_life_or_for_empty_frames() -> None:
    """No half-life or no rows means there is no recency weighting to apply."""
    assert sample_weights.compute_recency_sample_weight(pd.DataFrame()) is None
    assert (
        sample_weights.compute_recency_sample_weight(
            pd.DataFrame({"season": [], "week": []}),
            half_life_weeks=2.0,
        )
        is None
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"half_life_weeks": 0.0}, "half_life_weeks must be positive"),
        ({"half_life_seasons": 0.0}, "half_life_seasons must be positive"),
    ],
)
def test_compute_recency_sample_weight_rejects_non_positive_half_lives(
    kwargs: dict[str, float],
    message: str,
) -> None:
    """Recency half-lives must be strictly positive when supplied."""
    df = pd.DataFrame({"season": [2024], "week": [1]})

    with pytest.raises(ValueError, match=message):
        sample_weights.compute_recency_sample_weight(df, **kwargs)


def test_compute_recency_sample_weight_requires_numeric_season_values() -> None:
    """Season-based recency weighting requires a numeric season column."""
    with pytest.raises(ValueError, match="season column required"):
        sample_weights.compute_recency_sample_weight(
            pd.DataFrame({"week": [1]}),
            half_life_weeks=1.0,
        )

    with pytest.raises(ValueError, match="season column contains non-numeric"):
        sample_weights.compute_recency_sample_weight(
            pd.DataFrame({"season": ["bad"], "week": [1]}),
            half_life_weeks=1.0,
        )


def test_compute_recency_sample_weight_requires_numeric_week_values_for_week_mode() -> None:
    """Week-based recency weighting requires a numeric week column."""
    with pytest.raises(ValueError, match="week column required"):
        sample_weights.compute_recency_sample_weight(
            pd.DataFrame({"season": [2024]}),
            half_life_weeks=1.0,
        )

    with pytest.raises(ValueError, match="week column contains non-numeric"):
        sample_weights.compute_recency_sample_weight(
            pd.DataFrame({"season": [2024], "week": ["bad"]}),
            half_life_weeks=1.0,
        )


def test_compute_recency_sample_weight_returns_none_when_all_seasons_are_current() -> None:
    """A single-season slate yields all-one season weights and collapses to None."""
    df = pd.DataFrame({"season": [2024, 2024], "week": [1, 2]})

    assert sample_weights.compute_recency_sample_weight(df, half_life_seasons=3.0) is None


def test_compute_recency_sample_weight_returns_none_when_all_week_indices_match() -> None:
    """Identical week indices produce all-one week-based recency weights."""
    df = pd.DataFrame({"season": [2024], "week": [5]})

    assert sample_weights.compute_recency_sample_weight(df, half_life_weeks=3.0) is None


def test_combine_sample_weights_returns_none_without_active_weights_or_net_change() -> None:
    """Combining no weights or all-one weights should collapse to None."""
    assert sample_weights.combine_sample_weights(None, None) is None
    assert sample_weights.combine_sample_weights(np.ones(2), np.ones(2)) is None
