"""``floor_sigma.home_win_prob``: the floor ``Phi(margin / sigma)`` takes only a usable sigma.

A sigma that is not finite and positive has no meaning as the spread of the margin errors: an
infinite one would turn every game into a coin flip, NaN would make every probability NaN, and
zero or a negative one would divide by zero or flip the favorite. The betting report refuses
the same values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nfl_predictor.ml import floor_sigma

MARGINS = np.array([-7.0, 0.0, 3.5, 14.0])


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
