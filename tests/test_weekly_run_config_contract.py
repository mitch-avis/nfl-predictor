"""The weekly run's config file: how it loads, normalizes, and yields to the command line.

``config/weekly_run.yaml`` documents the contract: the weekly run reads that file by default,
``--config <path>`` reads another one instead, and command-line options override either. A
relative single path in the file is taken from the repository root; path lists keep their paths
as written for the code that reads them (``ml/floor_sigma.py`` resolves the reference runs against
the root). A file read with ``--config`` sets its own value or none, so a key it leaves out takes
the code default rather than the shipped value. Every config-file problem stops the run with a
``ValueError`` that names the file, before any data refresh.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from nfl_predictor import constants
from nfl_predictor.ml import ml_model_xgb_utils
from nfl_predictor.weekly_run import config

if TYPE_CHECKING:
    import argparse

# The single-path keys a config may set.
SCALAR_PATH_KEYS = (
    "data_path",
    "predict_path",
    "run_dir",
    "output_dir",
    "power_rankings_data_ml",
    "power_rankings_data_schedule",
    "power_rankings_out_dir",
    "power_rankings_strength_snapshots",
    "power_rankings_ratings_dir",
)
# The path keys whose code default is no path at all.
NO_PATH_DEFAULT_KEYS = ("predict_path", "run_dir", "output_dir", "power_rankings_out_dir")
PATH_LIST_KEYS = ("floor_sigma_reference_runs", "power_rankings_model_predictions")


def _write(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` and return the path."""
    path.write_text(text, encoding="utf-8")
    return path


def _parse_with(path: Path, *argv: str) -> argparse.Namespace:
    """Parse ``--config path`` and ``argv`` as the weekly run does."""
    return config.parse_args(["--config", str(path), *argv])


def _code_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Return every option's code default: the parse with no config file to read."""
    with monkeypatch.context() as patch:
        patch.setattr(config, "DEFAULT_CONFIG_PATH", tmp_path / "no_shipped_config.yaml")
        return vars(config.parse_args([]))


# Config files


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("weekly.json", '{"wf_eval_last_n_seasons": 4, "tune": true}'),
        ("weekly.yaml", "wf_eval_last_n_seasons: 4\ntune: true\n"),
        ("weekly.yml", "wf_eval_last_n_seasons: 4\ntune: true\n"),
    ],
    ids=["json", "yaml", "yml"],
)
def test_a_mapping_file_sets_its_options(tmp_path: Path, name: str, text: str) -> None:
    # Arrange
    path = _write(tmp_path / name, text)

    # Act
    args = _parse_with(path)

    # Assert
    assert (args.wf_eval_last_n_seasons, args.tune) == (4, True)


@pytest.mark.parametrize("text", ["", "# nothing set\n"], ids=["empty", "comments-only"])
def test_an_empty_yaml_config_leaves_every_option_at_its_code_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    # Arrange
    path = _write(tmp_path / "weekly.yaml", text)
    code_defaults = _code_defaults(tmp_path, monkeypatch)

    # Act
    args = _parse_with(path)

    # Assert
    assert {key: value for key, value in vars(args).items() if key != "config"} == {
        key: value for key, value in code_defaults.items() if key != "config"
    }


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("weekly.yaml", "- wf_eval_last_n_seasons\n- 4\n"),
        ("weekly.yaml", "just a string\n"),
        ("weekly.yml", "4\n"),
        ("weekly.json", "[1, 2]"),
        ("weekly.json", '"wf_eval_last_n_seasons"'),
        ("weekly.json", "null"),
    ],
    ids=["yaml-list", "yaml-string", "yml-number", "json-list", "json-string", "json-null"],
)
def test_a_file_whose_top_level_is_not_a_mapping_is_rejected(
    tmp_path: Path, name: str, text: str
) -> None:
    # Arrange
    path = _write(tmp_path / name, text)

    # Act & Assert
    with pytest.raises(ValueError, match=re.escape(str(path))) as rejected:
        _parse_with(path)
    assert "mapping" in str(rejected.value)


@pytest.mark.parametrize("name", ["weekly.toml", "weekly.txt", "weekly"])
def test_an_unsupported_extension_is_rejected_naming_the_supported_formats(
    tmp_path: Path, name: str
) -> None:
    # Arrange
    path = _write(tmp_path / name, "wf_eval_last_n_seasons = 4\n")

    # Act & Assert
    with pytest.raises(ValueError, match=re.escape(str(path))) as rejected:
        _parse_with(path)
    message = str(rejected.value).lower()
    assert "json" in message
    assert "yaml" in message


def test_a_missing_config_file_is_rejected_naming_the_path(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "no_such_dir" / "weekly.yaml"

    # Act & Assert
    with pytest.raises(ValueError, match=re.escape(str(path))):
        _parse_with(path)


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("weekly.json", '{"wf_eval_last_n_seasons": 4,'),
        ("weekly.yaml", "wf_eval_last_n_seasons: [4\n"),
        ("weekly.yml", "a: b: c\n"),
    ],
    ids=["json", "yaml", "yml"],
)
def test_a_malformed_file_is_rejected_naming_the_file(tmp_path: Path, name: str, text: str) -> None:
    # Arrange
    path = _write(tmp_path / name, text)

    # Act & Assert
    with pytest.raises(ValueError, match=re.escape(str(path))):
        _parse_with(path)


# Normalization


@pytest.mark.parametrize("key", SCALAR_PATH_KEYS)
def test_a_relative_path_resolves_against_the_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    # Arrange
    path = _write(tmp_path / "weekly.json", json.dumps({key: "some/dir/file.csv"}))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    # Act
    args = _parse_with(path)

    # Assert
    assert getattr(args, key) == Path(constants.ROOT_DIR) / "some" / "dir" / "file.csv"


@pytest.mark.parametrize("key", SCALAR_PATH_KEYS)
def test_an_absolute_path_stays_as_given(tmp_path: Path, key: str) -> None:
    # Arrange
    target = tmp_path / "abs" / "file.csv"
    path = _write(tmp_path / "weekly.json", json.dumps({key: str(target)}))

    # Act
    args = _parse_with(path)

    # Assert
    assert getattr(args, key) == target


def test_a_relative_config_path_is_read_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    _write(tmp_path / "weekly.yaml", "wf_eval_last_n_seasons: 4\n")
    monkeypatch.chdir(tmp_path)

    # Act
    args = config.parse_args(["--config", "weekly.yaml"])

    # Assert
    assert (args.config, args.wf_eval_last_n_seasons) == (Path("weekly.yaml"), 4)


@pytest.mark.parametrize("key", PATH_LIST_KEYS)
def test_a_path_list_keeps_its_paths_in_order(tmp_path: Path, key: str) -> None:
    # Arrange
    absolute = tmp_path / "abs_run"
    path = _write(tmp_path / "weekly.json", json.dumps({key: ["runs/a", str(absolute)]}))

    # Act
    args = _parse_with(path)

    # Assert
    assert list(getattr(args, key)) == [Path("runs/a"), absolute]


@pytest.mark.parametrize("key", PATH_LIST_KEYS)
def test_a_path_list_given_as_one_string_is_refused(tmp_path: Path, key: str) -> None:
    # Arrange
    path = _write(tmp_path / "weekly.yaml", f"{key}: runs/a\n")

    # Act & Assert
    with pytest.raises(ValueError, match=rf"{key} must be a list of paths"):
        _parse_with(path)


@pytest.mark.parametrize("key", NO_PATH_DEFAULT_KEYS)
def test_a_null_path_leaves_the_option_with_no_path(tmp_path: Path, key: str) -> None:
    # Arrange
    path = _write(tmp_path / "weekly.yaml", f"{key}: null\n")

    # Act
    args = _parse_with(path)

    # Assert
    assert getattr(args, key) is None


def test_a_yaml_group_list_names_the_groups(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path / "weekly.yaml", "disable_feature_groups:\n  - elo\n  - own_elo_wt\n")

    # Act
    args = _parse_with(path)

    # Assert
    assert args.disable_feature_groups == ("elo", "own_elo_wt")


def test_an_empty_yaml_group_list_leaves_no_group_out(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path / "weekly.yaml", "disable_feature_groups: []\n")

    # Act
    args = _parse_with(path)

    # Assert
    assert args.disable_feature_groups == ()


def test_an_empty_group_string_in_a_config_is_refused_as_not_a_list(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path / "weekly.yaml", "disable_feature_groups: ''\n")

    # Act & Assert
    with pytest.raises(ValueError, match="disable_feature_groups must be a list"):
        _parse_with(path)


# Precedence


def test_the_command_line_overrides_a_config_value(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path / "weekly.json", '{"wf_eval_last_n_seasons": 5}')

    # Act
    args = _parse_with(path, "--wf-eval-last-n-seasons", "2")

    # Assert
    assert args.wf_eval_last_n_seasons == 2


def test_the_command_line_overrides_the_shipped_config() -> None:
    # Act
    args = config.parse_args(["--postseason-weight", "2.0"])

    # Assert
    assert args.postseason_weight == 2.0


def test_without_config_or_a_shipped_file_every_option_takes_its_code_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", tmp_path / "missing.yaml")

    # Act
    args = config.parse_args([])

    # Assert
    assert args.config is None
    assert args.postseason_weight == 1.0
    assert args.data_collection_args is None


def test_without_config_the_shipped_file_supplies_its_values() -> None:
    # Act
    args = config.parse_args([])

    # Assert
    assert args.config == config.DEFAULT_CONFIG_PATH
    assert args.postseason_weight == 1.3
    assert args.data_collection_args == "--incremental"


def test_a_key_a_config_file_leaves_out_takes_the_code_default_not_the_shipped_value(
    tmp_path: Path,
) -> None:
    # Arrange
    path = _write(tmp_path / "weekly.yaml", "tune: false\n")

    # Act
    args = _parse_with(path)

    # Assert
    assert args.postseason_weight == 1.0
    assert args.data_collection_args is None


def test_an_old_key_name_lands_on_the_renamed_option(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path / "weekly.yaml", "train_early_stopping_rounds: 17\n")

    # Act
    args = _parse_with(path)

    # Assert
    assert args.tune_early_stopping_rounds == 17


def test_a_removed_key_fails_with_its_retirement_reason(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path / "weekly.yaml", "wf_calibration_weeks: 2\n")

    # Act & Assert
    with pytest.raises(ValueError, match=r"wf_calibration_weeks.*no held-out calibration weeks"):
        _parse_with(path)


def test_an_unknown_key_is_rejected(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path / "weekly.yaml", "not_an_option: 1\n")

    # Act & Assert
    with pytest.raises(ValueError, match="not_an_option"):
        _parse_with(path)


# apply_run_defaults


def test_the_final_fit_half_life_follows_stage_1_when_unset() -> None:
    # Arrange
    args = config.parse_args(["--wf-recency-half-life-seasons", "8", "--xgb-device", "cpu"])

    # Act
    config.apply_run_defaults(args)

    # Assert
    assert args.train_recency_half_life_seasons == 8


def test_an_explicit_final_fit_half_life_is_kept() -> None:
    # Arrange
    args = config.parse_args(
        [
            "--wf-recency-half-life-seasons",
            "8",
            "--train-recency-half-life-seasons",
            "4",
            "--xgb-device",
            "cpu",
        ]
    )

    # Act
    config.apply_run_defaults(args)

    # Assert
    assert args.train_recency_half_life_seasons == 4


def test_no_half_life_anywhere_leaves_both_unweighted() -> None:
    # Arrange
    args = config.parse_args(["--xgb-device", "cpu"])

    # Act
    config.apply_run_defaults(args)

    # Assert
    assert args.wf_recency_half_life_seasons is None
    assert args.train_recency_half_life_seasons is None


@pytest.mark.parametrize(("usable", "expected"), [(True, "cuda"), (False, "cpu")])
def test_auto_resolves_to_a_concrete_device(
    monkeypatch: pytest.MonkeyPatch, *, usable: bool, expected: str
) -> None:
    # Arrange
    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: usable)
    args = config.parse_args(["--xgb-device", "auto"])

    # Act
    config.apply_run_defaults(args)

    # Assert
    assert args.xgb_device == expected


def test_an_explicit_device_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: True)
    args = config.parse_args(["--xgb-device", "cpu"])

    # Act
    config.apply_run_defaults(args)

    # Assert
    assert args.xgb_device == "cpu"


@pytest.mark.parametrize("usable", [True, False])
def test_stage_1_names_the_resolved_device(
    monkeypatch: pytest.MonkeyPatch, *, usable: bool
) -> None:
    # Arrange
    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: usable)
    args = config.parse_args([])
    config.apply_run_defaults(args)

    # Act
    options = config.stage1_options(args)

    # Assert
    assert options.xgb_params_overrides["device"] == ("cuda" if usable else "cpu")


@pytest.mark.parametrize("usable", [True, False])
def test_the_final_fit_names_the_resolved_device(
    monkeypatch: pytest.MonkeyPatch, *, usable: bool
) -> None:
    # Arrange
    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: usable)
    args = config.parse_args([])
    config.apply_run_defaults(args)

    # Act
    optuna_config, _ = config.final_fit_options(args, pd.DataFrame(), run_dir=None)

    # Assert
    assert optuna_config.device == ("cuda" if usable else "cpu")
