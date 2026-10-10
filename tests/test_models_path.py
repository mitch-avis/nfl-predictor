"""Pin where run directories, walk-forward checkpoints and the web app's run index default to.

Each of them is the checkout's models directory or a directory inside it. The tests that write a
run point the package's roots at a temporary directory first, so nothing lands in the checkout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from nfl_predictor import constants
from nfl_predictor.api.settings import Settings
from nfl_predictor.cli import backtest, checkpoints
from nfl_predictor.ml import artifacts, walk_forward
from nfl_predictor.weekly_run import pipeline

if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKOUT_MODELS = REPO_ROOT / "models"


def _models_directory_at(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    """Point the package's repository root, and so its models directory, at ``root``."""
    monkeypatch.setattr(constants, "ROOT_DIR", root)
    monkeypatch.setattr(constants, "MODELS_PATH", root / "models", raising=False)
    return root / "models"


def test_run_paths_default_to_the_checkouts_models_directory() -> None:
    # Act
    paths = artifacts.resolve_run_paths("weekly_run")

    # Assert
    assert paths.run_dir.resolve() == CHECKOUT_MODELS / "weekly_run"


def test_the_checkpoint_listing_reads_the_checkouts_models_directory() -> None:
    # Act
    args = checkpoints._parse_args([])

    # Assert
    assert args.checkpoint_dir.resolve() == CHECKOUT_MODELS / "wf_checkpoints"
    assert args.models_dir.resolve() == CHECKOUT_MODELS


def test_a_backtest_keeps_its_checkpoints_in_the_checkouts_models_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py"])

    # Act
    args = backtest._parse_args()

    # Assert
    assert args.checkpoint_dir.resolve() == CHECKOUT_MODELS / "wf_checkpoints"


def test_the_web_app_indexes_the_runs_in_the_checkouts_models_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    for variable in ("NFLP_ROOT_DIR", "NFLP_MODELS_DIR"):
        monkeypatch.delenv(variable, raising=False)

    # Act
    settings = Settings()

    # Assert
    assert settings.models_path == CHECKOUT_MODELS


def test_a_backtest_writes_its_report_in_a_run_directory_under_the_models_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    models = _models_directory_at(monkeypatch, tmp_path)
    monkeypatch.setattr(walk_forward, "load_games", lambda _path: pd.DataFrame({"season": [2023]}))
    monkeypatch.setattr(walk_forward, "dataset_fingerprint", lambda _path: "hash")
    monkeypatch.setattr(
        walk_forward,
        "run_walk_forward_backtest",
        lambda _df, _config, **_kwargs: {"checkpoint": {}, "per_week": []},
    )
    monkeypatch.setattr(
        walk_forward, "build_metrics_report", lambda _r, _c, payload, _res: {"config": payload}
    )
    monkeypatch.setattr(walk_forward, "build_metadata", lambda _c, _h, payload: {"config": payload})
    monkeypatch.setattr(
        sys,
        "argv",
        ["walk_forward_backtest.py", "--checkpoint-dir", str(tmp_path / "checkpoints")],
    )

    # Act
    backtest.main()

    # Assert
    reports = sorted(models.glob("*/metrics_report.json"))
    assert len(reports) == 1
    assert (reports[0].parent / "metadata.json").is_file()


def test_a_weekly_run_directory_defaults_to_one_under_the_models_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    models = _models_directory_at(monkeypatch, tmp_path)
    dataset = tmp_path / "games.csv"
    dataset.write_text("season,week\n2025,1\n", encoding="utf-8")
    args = argparse.Namespace(data_path=dataset, run_id="pinned_run", run_dir=None, output_dir=None)

    # Act
    context = pipeline._start_run(args)

    # Assert
    assert context.run_dir == models / "pinned_run"
    assert context.run_dir.is_dir()
    assert context.output_dir == context.run_dir
