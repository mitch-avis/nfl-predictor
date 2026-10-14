"""Leaving out the strength snapshot needs the ported ratings in the data.

Production leaves out the strength snapshot (``strength``) because the ported team ratings
(``ratings``) replace it. A dataset built before the port has no ratings column, so a run
leaving out ``strength`` there would train on neither. ``nfl-predictor train`` and the weekly run
(stage 1 and the final fit) stop; ``nfl-predictor backtest``, which runs ablations on purpose,
warns and carries on (``tests/test_cli_backtest.py``). Both name the two remedies: rebuild the
dataset with ``nfl-predictor data``, or pass a group list that keeps ``strength``.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import pytest
from tests.feature_group_fixture import (
    GROUP_COLUMNS,
    RATINGS_COLUMNS,
    build_group_fixture,
    write_group_weekly_config,
)

from nfl_predictor.cli import train
from nfl_predictor.ml import feature_spec
from nfl_predictor.weekly_run import pipeline

if TYPE_CHECKING:
    from pathlib import Path

# The groups production leaves out: the strength snapshot and both presets of the own Elo.
PRODUCTION_GROUPS = ("strength", "own_elo", "own_elo_wt")
# The columns of a dataset built before the port: the strength snapshot, no ported ratings.
COLUMNS_WITHOUT_RATINGS = ("season", "week", *GROUP_COLUMNS["strength"], *GROUP_COLUMNS["elo"])
COLUMNS_WITH_RATINGS = (*COLUMNS_WITHOUT_RATINGS, *RATINGS_COLUMNS)
# Words every message names: both remedies.
REMEDIES = ("nfl-predictor data", "--disable-feature-groups", "disable_feature_groups")


def _train_argv(data_path: Path, run_dir: Path, *extra: str) -> list[str]:
    """Return ``nfl-predictor train``'s arguments for a CPU fit of ``data_path``."""
    return [
        "nfl-predictor train",
        "--data-path",
        str(data_path),
        "--holdout-seasons",
        "0",
        "--floor-sigma-reference-runs",
        "--xgb-device",
        "cpu",
        "--xgb-n-jobs",
        "1",
        "--run-dir",
        str(run_dir),
        *extra,
    ]


def test_the_check_stops_when_strength_is_left_out_and_the_data_has_no_ratings() -> None:
    # Act & Assert
    with pytest.raises(ValueError, match="no column of the ported ratings") as stopped:
        feature_spec.check_ratings_replace_strength(
            COLUMNS_WITHOUT_RATINGS, PRODUCTION_GROUPS, stop=True
        )
    assert all(remedy in str(stopped.value) for remedy in REMEDIES)


def test_the_check_warns_with_the_same_remedies_when_asked_to_carry_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    caplog.set_level(logging.WARNING)

    # Act
    feature_spec.check_ratings_replace_strength(
        COLUMNS_WITHOUT_RATINGS, PRODUCTION_GROUPS, stop=False
    )

    # Assert
    (message,) = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "no column of the ported ratings" in message
    assert all(remedy in message for remedy in REMEDIES)


@pytest.mark.parametrize(
    ("columns", "groups"),
    [
        (COLUMNS_WITH_RATINGS, PRODUCTION_GROUPS),
        (COLUMNS_WITHOUT_RATINGS, ()),
        (COLUMNS_WITHOUT_RATINGS, ("own_elo", "own_elo_wt")),
    ],
    ids=["ratings present", "no group left out", "strength kept"],
)
@pytest.mark.parametrize("stop", [True, False], ids=["stop", "warn"])
def test_the_check_says_nothing_when_the_ratings_are_there_or_strength_is_kept(
    columns: tuple[str, ...],
    groups: tuple[str, ...],
    caplog: pytest.LogCaptureFixture,
    *,
    stop: bool,
) -> None:
    # Arrange
    caplog.set_level(logging.WARNING)

    # Act
    feature_spec.check_ratings_replace_strength(columns, groups, stop=stop)

    # Assert
    assert caplog.records == []


def test_train_stops_on_data_without_the_ported_ratings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", _train_argv(paths["completed"], run_dir))

    # Act & Assert
    with pytest.raises(ValueError, match="nfl-predictor data"):
        train.main()
    assert not (run_dir / "metadata.json").exists()


@pytest.mark.parametrize(
    ("ratings", "extra"),
    [(True, ()), (False, ("--disable-feature-groups", ""))],
    ids=["ratings present", "strength kept"],
)
def test_train_fits_without_a_message_when_the_ratings_are_there_or_strength_is_kept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    extra: tuple[str, ...],
    *,
    ratings: bool,
) -> None:
    # Arrange
    caplog.set_level(logging.WARNING)
    paths = build_group_fixture(tmp_path, ratings=ratings)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", _train_argv(paths["completed"], run_dir, *extra))

    # Act
    train.main()

    # Assert
    assert (run_dir / "metadata.json").exists()
    assert not [r for r in caplog.records if "ported ratings" in r.getMessage()]


def test_the_weekly_run_stops_in_stage_1_on_data_without_the_ported_ratings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path)
    config_path = write_group_weekly_config(tmp_path, paths)
    monkeypatch.setattr(sys, "argv", ["nfl-predictor weekly", "--config", str(config_path)])

    # Act & Assert
    with pytest.raises(ValueError, match="nfl-predictor data"):
        pipeline.main()
    assert not list((tmp_path / "run").glob("wf_compare/wf_folds/*/fold_*.joblib"))


def test_the_weekly_runs_final_fit_stops_on_data_without_the_ported_ratings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage 1 reused from an earlier run does not let the final fit train on neither."""
    # Arrange
    paths = build_group_fixture(tmp_path)
    config_path = write_group_weekly_config(tmp_path, paths)
    monkeypatch.setattr(sys, "argv", ["nfl-predictor weekly", "--config", str(config_path)])

    def reused_stage1(*_args: Any) -> dict[str, Any]:
        """Stand in for a finished stage 1."""
        return {}

    monkeypatch.setattr(pipeline, "_run_stage1", reused_stage1)

    # Act & Assert
    with pytest.raises(ValueError, match="nfl-predictor data"):
        pipeline.main()
    assert not (tmp_path / "run" / "metadata.json").exists()
