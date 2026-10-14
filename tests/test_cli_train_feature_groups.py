"""``nfl-predictor train --disable-feature-groups``: the fit leaves the named groups out.

The option takes what ``nfl-predictor backtest --disable-feature-groups`` takes, a
comma-separated list of the groups in ``constants.FEATURE_GROUP_COLUMN_MARKERS``. It is checked
when the arguments are parsed, so an unknown group stops the command before any data loads.
Its default is the weekly run's: the strength snapshot and both own-Elo presets are left out, and
an empty value leaves no group out.
The run's ``metadata.json`` records the option in its config and, beside the feature list, the
groups the model was trained without and the columns they removed. A saved model predicts with
the columns it was trained on, so ``nfl-predictor predict`` says the option does not apply.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING

import pytest
from tests.feature_group_fixture import GROUP_COLUMNS, RATINGS_COLUMNS, build_group_fixture

from nfl_predictor.cli import train
from nfl_predictor.weekly_run import config as weekly_config

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path

# The groups production leaves out: the strength snapshot and both presets of the own Elo.
PRODUCTION_GROUPS = ("strength", "own_elo", "own_elo_wt")


def _parse(argv: Sequence[str], monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    """Parse ``argv`` as ``nfl-predictor train`` does."""
    monkeypatch.setattr(sys, "argv", ["nfl-predictor train", *argv])
    return train._parse_args()


def test_without_the_option_the_weekly_runs_groups_are_left_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Act
    args = _parse([], monkeypatch)

    # Assert
    assert args.disable_feature_groups == PRODUCTION_GROUPS


def test_the_default_matches_the_weekly_runs_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    weekly = weekly_config.parse_args([])

    # Act
    args = _parse([], monkeypatch)

    # Assert
    assert args.disable_feature_groups == weekly.disable_feature_groups


def test_an_empty_value_leaves_no_group_out(monkeypatch: pytest.MonkeyPatch) -> None:
    # Act
    args = _parse(["--disable-feature-groups", ""], monkeypatch)

    # Assert
    assert args.disable_feature_groups == ()


def test_the_option_takes_the_backtests_comma_separated_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Act
    args = _parse(["--disable-feature-groups", " elo , own_elo_wt ,"], monkeypatch)

    # Assert
    assert args.disable_feature_groups == ("elo", "own_elo_wt")


def test_an_unknown_group_stops_the_command_and_names_the_group(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Act & Assert
    with pytest.raises(SystemExit) as stopped:
        _parse(["--disable-feature-groups", "elo,not_a_group"], monkeypatch)
    assert stopped.value.code == 2
    assert "Unknown feature group(s): ['not_a_group']" in capsys.readouterr().err


def test_the_fit_leaves_the_groups_out_and_the_run_records_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path)
    run_dir = tmp_path / "run"
    dropped = sorted([*GROUP_COLUMNS["strength"], *GROUP_COLUMNS["own_elo"]])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nfl-predictor train",
            "--data-path",
            str(paths["completed"]),
            "--holdout-seasons",
            "0",
            "--floor-sigma-reference-runs",
            "--xgb-device",
            "cpu",
            "--xgb-n-jobs",
            "1",
            "--run-dir",
            str(run_dir),
            "--disable-feature-groups",
            "strength,own_elo",
        ],
    )

    # Act
    train.main()

    # Assert
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["config"]["disable_feature_groups"] == ["strength", "own_elo"]
    assert metadata["disabled_feature_groups"] == ["strength", "own_elo"]
    assert metadata["dropped_feature_group_columns"] == dropped
    assert not set(metadata["feature_list"]) & set(dropped)
    assert set(metadata["feature_list"]) >= {*GROUP_COLUMNS["elo"], *GROUP_COLUMNS["own_elo_wt"]}


def test_a_default_fit_records_the_weekly_runs_groups_and_keeps_the_ratings_and_the_elo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path, ratings=True)
    run_dir = tmp_path / "run"
    left_out = sorted(column for group in PRODUCTION_GROUPS for column in GROUP_COLUMNS[group])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nfl-predictor train",
            "--data-path",
            str(paths["completed"]),
            "--holdout-seasons",
            "0",
            "--floor-sigma-reference-runs",
            "--xgb-device",
            "cpu",
            "--xgb-n-jobs",
            "1",
            "--run-dir",
            str(run_dir),
        ],
    )

    # Act
    train.main()

    # Assert
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["disabled_feature_groups"] == list(PRODUCTION_GROUPS)
    assert metadata["dropped_feature_group_columns"] == left_out
    assert not set(metadata["feature_list"]) & set(left_out)
    assert set(metadata["feature_list"]) >= {*GROUP_COLUMNS["elo"], *RATINGS_COLUMNS}


def test_predicting_with_a_saved_model_says_the_option_does_not_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(train, "load_model_checkpoint", lambda *_args: "model")
    monkeypatch.setattr(train.artifacts, "sha256_file", lambda _path: "hash")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nfl-predictor predict",
            "--model-in",
            str(tmp_path / "model.joblib"),
            "--disable-feature-groups",
            "elo",
        ],
    )

    # Act
    train.main()

    # Assert
    assert any("--disable-feature-groups" in message for message in caplog.messages)


def test_predicting_without_the_option_says_nothing_about_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(train, "load_model_checkpoint", lambda *_args: "model")
    monkeypatch.setattr(train.artifacts, "sha256_file", lambda _path: "hash")
    monkeypatch.setattr(
        sys, "argv", ["nfl-predictor predict", "--model-in", str(tmp_path / "model.joblib")]
    )

    # Act
    train.main()

    # Assert
    assert not any("--disable-feature-groups" in message for message in caplog.messages)
