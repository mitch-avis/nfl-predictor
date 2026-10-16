"""The production floor-sigma pool defaults to the reviewed reference of production's features.

Production trains on the current ``qb_elos.csv`` with production's feature groups left out,
market anchored; the two seeds of the walk-forward run with exactly those settings are the pool
the weekly run and ``nfl-predictor train`` read by default.
"""

from __future__ import annotations

from pathlib import Path

from nfl_predictor import constants
from nfl_predictor.weekly_run import config


def test_production_pool_defaults_to_both_seeds_of_the_new_file_reference() -> None:
    """The weekly run and ``train`` read the new-file production-settings runs by default."""
    # Arrange
    expected = (
        "models/qb_elos_measure/new_file_seed42",
        "models/qb_elos_measure/new_file_seed7",
    )

    # Act
    runs = constants.FLOOR_SIGMA_REFERENCE_RUNS

    # Assert
    assert runs == expected


def test_the_shipped_config_reads_the_constants_reference_runs() -> None:
    """``config/weekly_run.yaml`` names the same runs, in the same order, as the constant."""
    # Act
    args = config.parse_args([])

    # Assert
    assert args.floor_sigma_reference_runs == [
        Path(path) for path in constants.FLOOR_SIGMA_REFERENCE_RUNS
    ]
