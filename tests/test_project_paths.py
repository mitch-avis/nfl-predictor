"""Pin the repository-root paths the package resolves at import time.

Data, run directories, checkpoints and the weekly-run config all live beside the checkout, not
beside the package, so these paths must name the repository root wherever the package sits.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nfl_predictor import constants
from nfl_predictor.ml import walk_forward
from nfl_predictor.weekly_run import config as weekly_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_root_dir_is_the_repository_root() -> None:
    """ROOT_DIR names the checkout that holds pyproject.toml and these tests."""
    # Act
    root = Path(constants.ROOT_DIR)

    # Assert
    assert root.is_absolute()
    assert root.resolve() == REPO_ROOT
    assert (root / "pyproject.toml").is_file()


@pytest.mark.usefixtures("packaged_data_paths")
def test_data_paths_sit_under_the_repository_root() -> None:
    """The data directory and the nflreadpy cache resolve under the repository root."""
    # Act
    data_path = Path(constants.DATA_PATH)
    cache_dir = Path(constants.NFLREADPY_CACHE_DIR)

    # Assert
    assert data_path == Path(constants.ROOT_DIR) / "data"
    assert data_path.resolve() == REPO_ROOT / "data"
    assert cache_dir.resolve() == REPO_ROOT / "data" / "cache" / "nflreadpy"


def test_models_and_config_defaults_sit_under_the_repository_root() -> None:
    """Walk-forward checkpoints and the weekly-run config default to the repository root."""
    # Act
    checkpoint_dir = walk_forward.DEFAULT_CHECKPOINT_DIR.resolve()
    config_path = weekly_config.DEFAULT_CONFIG_PATH

    # Assert
    assert checkpoint_dir == REPO_ROOT / "models" / "wf_checkpoints"
    assert config_path.resolve() == REPO_ROOT / "config" / "weekly_run.yaml"
    assert config_path.is_file()
