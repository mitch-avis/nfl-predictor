"""The production floor-sigma pool defaults to the reviewed reference of production's features.

Production trains on the current ``qb_elos.csv`` with production's feature groups left out,
market anchored; the two seeds of the walk-forward run with exactly those settings are the pool
the weekly run and ``nfl-predictor train`` read by default. A run without game lines reads the two
seeds of the walk-forward with its own settings instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nfl_predictor import constants
from nfl_predictor.cli import options
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


def test_runs_without_game_lines_default_to_both_seeds_of_their_own_reference() -> None:
    """A run without game lines pools the errors of the reviewed runs with its own settings."""
    # Arrange
    expected = (
        "models/qb_adj_no_lines/no_lines_seed42",
        "models/qb_adj_no_lines/no_lines_seed7",
    )

    # Act
    runs = constants.NO_GAME_LINES_FLOOR_SIGMA_REFERENCE_RUNS

    # Assert
    assert runs == expected


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("hybrid", constants.FLOOR_SIGMA_REFERENCE_RUNS),
        ("features", constants.FLOOR_SIGMA_REFERENCE_RUNS),
        ("anchor", constants.FLOOR_SIGMA_REFERENCE_RUNS),
        ("none", constants.NO_GAME_LINES_FLOOR_SIGMA_REFERENCE_RUNS),
    ],
)
def test_the_weekly_run_reads_the_reference_of_its_market_mode(
    mode: str, expected: tuple[str, ...]
) -> None:
    """Through the shipped config, the mode without game lines reads its own reference runs."""
    # Act
    args = config.parse_args(["--wf-market-mode", mode])

    # Assert
    assert args.floor_sigma_reference_runs == [Path(path) for path in expected]


def test_listed_reference_runs_win_in_the_mode_without_game_lines(tmp_path: Path) -> None:
    """A config that names reference runs replaces the default of every mode."""
    # Arrange
    config_path = tmp_path / "weekly.json"
    config_path.write_text(
        '{"wf_market_mode": "none", "floor_sigma_reference_runs": ["a/b"]}', encoding="utf-8"
    )

    # Act
    args = config.parse_args(["--config", str(config_path)])

    # Assert
    assert args.floor_sigma_reference_runs == [Path("a/b")]


@pytest.mark.parametrize(
    ("explicit", "without_game_lines", "expected"),
    [
        (None, False, constants.FLOOR_SIGMA_REFERENCE_RUNS),
        (None, True, constants.NO_GAME_LINES_FLOOR_SIGMA_REFERENCE_RUNS),
        ([Path("a/b")], True, ("a/b",)),
        ([], True, ()),
    ],
    ids=["production", "without-game-lines", "listed", "empty"],
)
def test_an_explicit_list_wins_and_none_takes_the_default_of_the_mode(
    explicit: list[Path] | None, *, without_game_lines: bool, expected: tuple[str, ...]
) -> None:
    """``resolve_floor_sigma_reference_runs`` mirrors how the feature-group default resolves."""
    # Act
    runs = options.resolve_floor_sigma_reference_runs(
        explicit, without_game_lines=without_game_lines
    )

    # Assert
    assert runs == [Path(path) for path in expected]
