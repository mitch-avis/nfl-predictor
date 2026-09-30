"""Pin the repository-root paths the package resolves at import time.

Data, run directories, checkpoints and the weekly-run config all live beside the checkout, not
beside the package, so these paths must name the repository root wherever the package sits.
"""

from __future__ import annotations

from pathlib import Path

from nfl_predictor import constants
from nfl_predictor.ml import walk_forward
from nfl_predictor.weekly_run import config as weekly_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_root_dir_is_the_repository_root() -> None:
    """ROOT_DIR names the checkout that holds pyproject.toml and these tests."""
    root = Path(constants.ROOT_DIR)

    assert root.is_absolute()
    assert root.resolve() == REPO_ROOT
    assert (root / "pyproject.toml").is_file()


def test_data_paths_sit_under_the_repository_root() -> None:
    """The data directory and the nflreadpy cache resolve under the repository root."""
    assert Path(constants.DATA_PATH) == Path(constants.ROOT_DIR) / "data"
    assert Path(constants.DATA_PATH).resolve() == REPO_ROOT / "data"
    assert Path(constants.NFLREADPY_CACHE_DIR).resolve() == REPO_ROOT / "data" / "cache" / (
        "nflreadpy"
    )


def test_models_and_config_defaults_sit_under_the_repository_root() -> None:
    """Walk-forward checkpoints and the weekly-run config default to the repository root."""
    assert walk_forward.DEFAULT_CHECKPOINT_DIR.resolve() == REPO_ROOT / "models" / (
        "wf_checkpoints"
    )
    assert weekly_config.DEFAULT_CONFIG_PATH.resolve() == REPO_ROOT / "config" / "weekly_run.yaml"
    assert weekly_config.DEFAULT_CONFIG_PATH.is_file()
