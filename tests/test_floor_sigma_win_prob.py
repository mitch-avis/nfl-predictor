"""``floor_sigma.home_win_prob``: the floor ``Phi(margin / sigma)`` takes only a usable sigma.

A sigma that is not finite and positive has no meaning as the spread of the margin errors: an
infinite one would turn every game into a coin flip, NaN would make every probability NaN, and
zero or a negative one would divide by zero or flip the favorite. The betting report refuses
the same values. A sigma is one number per call: a vector with an unusable element is refused as
unusable, and a vector of usable ones as more than one number, both as ``ValueError``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.ml import floor_sigma

MARGINS = np.array([-7.0, 0.0, 3.5, 14.0])


def _as_given(value: object) -> Any:
    """Pass ``value`` on untyped, as a caller holding a vector of sigmas would."""
    return value


@pytest.mark.parametrize(
    "sigma",
    [math.inf, -math.inf, math.nan, 0.0, -13.5],
    ids=["inf", "minus-inf", "nan", "zero", "negative"],
)
def test_a_sigma_that_is_not_finite_and_positive_is_refused(sigma: float) -> None:
    # Act & Assert
    with pytest.raises(ValueError, match="finite and positive"):
        floor_sigma.home_win_prob(MARGINS, sigma)


def test_a_finite_positive_sigma_gives_the_normal_curve_at_that_spread() -> None:
    # Act
    probabilities = floor_sigma.home_win_prob(MARGINS, 13.5)

    # Assert
    assert probabilities.tolist() == pytest.approx(
        [0.30204827147857694, 0.5, 0.6022823943305156, 0.8501407016943955], rel=1e-12
    )


@pytest.mark.parametrize(
    "sigma",
    [
        pd.Series([13.5, math.nan, 13.5, 13.5]),
        np.array([13.5, 13.5, 0.0, 13.5]),
        np.array([13.5, math.inf]),
    ],
    ids=["series-with-nan", "array-with-zero", "array-with-inf"],
)
def test_a_vector_sigma_with_an_unusable_element_is_refused_as_a_value_error(
    sigma: pd.Series | np.ndarray,
) -> None:
    # Act & Assert
    with pytest.raises(ValueError, match="finite and positive"):
        floor_sigma.home_win_prob(MARGINS, _as_given(sigma))


def test_a_vector_of_usable_sigmas_is_refused_as_one_sigma_per_call() -> None:
    # Arrange
    sigma = np.array([13.5, 13.5, 13.5, 13.5])

    # Act & Assert
    with pytest.raises(ValueError, match="one number"):
        floor_sigma.home_win_prob(MARGINS, _as_given(sigma))
