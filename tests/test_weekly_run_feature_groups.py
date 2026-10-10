"""The weekly run's ``disable_feature_groups`` setting and its ``--disable-feature-groups`` option.

``config/weekly_run.yaml`` lists the feature groups (``constants.FEATURE_GROUP_COLUMN_MARKERS``)
that stage 1's walk-forward and the final fit both leave out, empty by default. The command line
takes the comma-separated form ``nfl-predictor backtest`` takes and overrides the file. An
unknown group stops the run when its options are parsed, before the data refresh.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from nfl_predictor.ml import artifacts
from nfl_predictor.ml.floor_sigma import ErrorPool
from nfl_predictor.weekly_run import config, pipeline, stage1

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def _write_config(tmp_path: Path, settings: dict[str, Any]) -> Path:
    """Write a weekly config file holding only ``settings``; return its path."""
    path = tmp_path / "weekly_run.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return path


def _parse(argv: Sequence[str]) -> Any:
    """Parse ``argv`` as the weekly run does."""
    return config.parse_args(argv)


def test_the_shipped_config_leaves_no_group_out() -> None:
    # Act
    args = _parse([])

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


def test_a_config_key_with_no_value_leaves_no_group_out(tmp_path: Path) -> None:
    # Arrange
    path = _write_config(tmp_path, {"disable_feature_groups": None})

    # Act
    args = _parse(["--config", str(path)])

    # Assert
    assert args.disable_feature_groups == ()


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
