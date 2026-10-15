"""The weekly run's ``disable_feature_groups`` setting and its ``--disable-feature-groups`` option.

``config/weekly_run.yaml`` lists the feature groups (``constants.FEATURE_GROUP_COLUMN_MARKERS``)
that stage 1's walk-forward and the final fit both leave out: by default the strength snapshot
(``strength``) and both presets of this repository's own Elo (``own_elo``, ``own_elo_wt``), so
production trains on the ported team ratings and the ``qb_elos.csv`` Elo. An explicit empty list
leaves no group out, a config without the key leaves out production's groups, and the key with
no value is refused, so commenting out every item cannot bring the groups back. The command line
takes the comma-separated form ``nfl-predictor backtest`` takes and overrides the file. An
unknown group stops the run when its options are parsed, before the data refresh.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import joblib
import pandas as pd
import pytest

from nfl_predictor.ml import artifacts, walk_forward
from nfl_predictor.ml.floor_sigma import ErrorPool
from nfl_predictor.weekly_run import config, pipeline, stage1
from tests.feature_group_fixture import (
    GROUP_COLUMNS,
    RATINGS_COLUMNS,
    build_group_fixture,
    write_group_weekly_config,
)
from tests.weekly_fixture import CURRENT_SEASON

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# The groups production leaves out: the strength snapshot and both presets of the own Elo.
PRODUCTION_GROUPS = ("strength", "own_elo", "own_elo_wt")
TEST_BOOTSTRAP_SAMPLES = 50


def _write_config(tmp_path: Path, settings: dict[str, Any]) -> Path:
    """Write a weekly config file holding only ``settings``; return its path."""
    path = tmp_path / "weekly_run.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return path


def _parse(argv: Sequence[str]) -> Any:
    """Parse ``argv`` as the weekly run does."""
    return config.parse_args(argv)


def test_the_shipped_config_leaves_out_the_strength_snapshot_and_both_own_elo_presets() -> None:
    # Act
    args = _parse([])

    # Assert
    assert args.disable_feature_groups == PRODUCTION_GROUPS


def test_the_code_default_leaves_out_the_same_groups_as_the_shipped_config() -> None:
    # Act
    args = config._build_parser().parse_args([])

    # Assert
    assert args.disable_feature_groups == PRODUCTION_GROUPS


def test_an_explicit_empty_config_list_leaves_no_group_out(tmp_path: Path) -> None:
    # Arrange
    path = _write_config(tmp_path, {"disable_feature_groups": []})

    # Act
    args = _parse(["--config", str(path)])

    # Assert
    assert args.disable_feature_groups == ()


def test_an_empty_command_line_value_leaves_no_group_out() -> None:
    # Act
    args = _parse(["--disable-feature-groups", ""])

    # Assert
    assert args.disable_feature_groups == ()


def test_a_config_list_names_the_groups(tmp_path: Path) -> None:
    # Arrange
    path = _write_config(tmp_path, {"disable_feature_groups": ["elo", "own_elo_wt"]})

    # Act
    args = _parse(["--config", str(path)])

    # Assert
    assert args.disable_feature_groups == ("elo", "own_elo_wt")


def test_the_command_line_takes_the_backtests_comma_separated_names() -> None:
    # Act
    args = _parse(["--disable-feature-groups", " elo , own_elo_wt ,"])

    # Assert
    assert args.disable_feature_groups == ("elo", "own_elo_wt")


@pytest.mark.parametrize(
    ("value", "expected"), [("strength", ("strength",)), ("", ())], ids=["other", "none"]
)
def test_the_command_line_overrides_the_config_list(
    tmp_path: Path, value: str, expected: tuple[str, ...]
) -> None:
    # Arrange
    path = _write_config(tmp_path, {"disable_feature_groups": ["elo"]})

    # Act
    args = _parse(["--config", str(path), "--disable-feature-groups", value])

    # Assert
    assert args.disable_feature_groups == expected


def test_a_config_key_with_no_value_is_refused_with_both_spellings(tmp_path: Path) -> None:
    # Arrange
    path = _write_config(tmp_path, {"disable_feature_groups": None})

    # Act & Assert
    with pytest.raises(ValueError, match="disable_feature_groups has no value") as refused:
        _parse(["--config", str(path)])
    assert "[]" in str(refused.value)
    assert "remove the key" in str(refused.value)


def test_a_config_without_the_key_leaves_out_productions_groups(tmp_path: Path) -> None:
    # Arrange
    path = _write_config(tmp_path, {})

    # Act
    args = _parse(["--config", str(path)])

    # Assert
    assert args.disable_feature_groups == PRODUCTION_GROUPS


def test_a_config_string_instead_of_a_list_is_refused(tmp_path: Path) -> None:
    # Arrange
    path = _write_config(tmp_path, {"disable_feature_groups": "elo"})

    # Act & Assert
    with pytest.raises(ValueError, match="disable_feature_groups must be a list"):
        _parse(["--config", str(path)])


def test_an_unknown_group_stops_the_run_before_the_data_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    refreshes: list[object] = []
    monkeypatch.setattr(pipeline, "_refresh_data", refreshes.append)
    path = _write_config(tmp_path, {"disable_feature_groups": ["elo", "not_a_group"]})
    monkeypatch.setattr(sys, "argv", ["nfl-predictor weekly", "--config", str(path)])

    # Act & Assert
    with pytest.raises(SystemExit) as stopped:
        pipeline.main()
    assert stopped.value.code == 2
    assert "Unknown feature group(s): ['not_a_group']" in capsys.readouterr().err
    assert refreshes == []


def test_stage1_walks_forward_without_the_groups(tmp_path: Path) -> None:
    # Arrange
    path = _write_config(tmp_path, {"disable_feature_groups": ["elo", "own_elo_wt"]})
    args = _parse(["--config", str(path), "--xgb-device", "cpu"])

    # Act
    walk_forward_config = stage1.production_walk_forward_config(config.stage1_options(args))

    # Assert
    assert walk_forward_config.disabled_feature_groups == ("elo", "own_elo_wt")


def test_stage1s_run_fingerprint_covers_the_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(artifacts, "git_commit_hash", lambda: "commit")
    path = _write_config(tmp_path, {"disable_feature_groups": ["elo"]})
    runs = [_parse(["--xgb-device", "cpu"]), _parse(["--config", str(path), "--xgb-device", "cpu"])]
    contexts = [
        pipeline._RunContext(
            args=args,
            run_id="run",
            run_dir=tmp_path,
            output_dir=tmp_path,
            created_at="now",
            dataset_hash="hash",
            dataset_fingerprint={"sha256": "dataset"},
            config_payload={},
        )
        for args in runs
    ]

    # Act
    fingerprints = {
        pipeline._wf_run_fingerprint(context, config.stage1_options(context.args), ErrorPool())
        for context in contexts
    }

    # Assert
    assert len(fingerprints) == 2


def test_the_final_fit_settings_record_the_groups(tmp_path: Path) -> None:
    # Arrange
    path = _write_config(tmp_path, {"disable_feature_groups": ["strength"]})
    args = _parse(["--config", str(path), "--xgb-device", "cpu"])
    frame = pd.DataFrame({"season": [2025], "home_spread": [-3.0], "total_line": [44.5]})

    # Act
    _, train_config = config.final_fit_options(args, frame, run_dir=None)

    # Assert
    assert train_config["disabled_feature_groups"] == ["strength"]


def _metadata(run_dir: Path) -> dict[str, Any]:
    """Return a run's ``metadata.json``."""
    return json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))


def _first_stage1_fold_columns(run_dir: Path) -> list[str]:
    """Return the feature columns stage 1's first week trained on, from its checkpoint."""
    (path,) = sorted(run_dir.glob(f"wf_compare/wf_folds/*/fold_{CURRENT_SEASON}_w01.joblib"))
    return list(joblib.load(path)["feature_columns"])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_default_weekly_run_trains_on_the_ratings_and_the_elo_without_the_left_out_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path, ratings=True)
    config_path = write_group_weekly_config(tmp_path, paths)
    monkeypatch.setattr(walk_forward, "BOOTSTRAP_SAMPLES", TEST_BOOTSTRAP_SAMPLES)
    monkeypatch.setattr(sys, "argv", ["nfl-predictor weekly", "--config", str(config_path)])
    left_out = {column for group in PRODUCTION_GROUPS for column in GROUP_COLUMNS[group]}
    kept = {*GROUP_COLUMNS["elo"], *RATINGS_COLUMNS}

    # Act
    status = pipeline.main()

    # Assert
    assert status == 0
    final_fit = _metadata(tmp_path / "run")
    assert final_fit["disabled_feature_groups"] == list(PRODUCTION_GROUPS)
    assert final_fit["dropped_feature_group_columns"] == sorted(left_out)
    assert not set(final_fit["feature_list"]) & left_out
    assert set(final_fit["feature_list"]) >= kept
    stage1_columns = set(_first_stage1_fold_columns(tmp_path / "run"))
    assert not stage1_columns & left_out
    assert stage1_columns >= kept
