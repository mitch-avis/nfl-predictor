"""The production floor-sigma pool defaults to the adopted-settings reference walk-forward."""

from __future__ import annotations

from nfl_predictor import constants


def test_production_pool_defaults_to_both_seeds_of_the_adopted_settings_reference() -> None:
    """The weekly run and ``train`` read the pick-time-line reference runs by default."""
    # Arrange
    expected = (
        "models/step4_reference/adopted_seed42",
        "models/step4_reference/adopted_seed7",
    )

    # Act
    runs = constants.FLOOR_SIGMA_REFERENCE_RUNS

    # Assert
    assert runs == expected
