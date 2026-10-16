"""Pin where run directories, walk-forward checkpoints and the web app's run index default to.

Each of them is the models directory, ``constants.MODELS_PATH``, or a directory inside it. The
tests that compute a path restore the checkout's models directory (``packaged_models_path``);
the tests that write a run use the test's own models directory (``tests/conftest.py``), so
nothing lands in the checkout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from nfl_predictor.api.settings import Settings
from nfl_predictor.cli import backtest, checkpoints
from nfl_predictor.ml import artifacts, walk_forward
from nfl_predictor.weekly_run import pipeline
from tests.fakes import returning

if TYPE_CHECKING:
    from tests.conftest import DataIsolation

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKOUT_MODELS = REPO_ROOT / "models"


@pytest.mark.usefixtures("packaged_models_path")
def test_run_paths_default_to_the_checkouts_models_directory() -> None:
    # Act
    paths = artifacts.resolve_run_paths("weekly_run")

    # Assert
    assert paths.run_dir.resolve() == CHECKOUT_MODELS / "weekly_run"


@pytest.mark.usefixtures("packaged_models_path")
def test_the_checkpoint_listing_reads_the_checkouts_models_directory() -> None:
    # Act
    args = checkpoints._parse_args([])

    # Assert
    assert args.checkpoint_dir.resolve() == CHECKOUT_MODELS / "wf_checkpoints"
    assert args.models_dir.resolve() == CHECKOUT_MODELS


@pytest.mark.usefixtures("packaged_models_path")
def test_a_backtest_keeps_its_checkpoints_in_the_checkouts_models_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py"])

    # Act
    args = backtest._parse_args()

    # Assert
    assert args.checkpoint_dir.resolve() == CHECKOUT_MODELS / "wf_checkpoints"


@pytest.mark.usefixtures("packaged_models_path")
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


def test_the_checkpoint_root_follows_the_models_directory(isolated_data: DataIsolation) -> None:
    # Act
    root = walk_forward.default_checkpoint_dir()

    # Assert
    assert root == isolated_data.models_dir / "wf_checkpoints"


def test_a_backtest_defaults_its_checkpoints_to_the_models_directory(
    monkeypatch: pytest.MonkeyPatch, isolated_data: DataIsolation
) -> None:
    # Arrange
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py"])

    # Act
    args = backtest._parse_args()

    # Assert
    assert args.checkpoint_dir == isolated_data.models_dir / "wf_checkpoints"


def test_the_checkpoint_listing_defaults_to_the_models_directory(
    isolated_data: DataIsolation,
) -> None:
    # Act
    args = checkpoints._parse_args([])

    # Assert
    assert args.checkpoint_dir == isolated_data.models_dir / "wf_checkpoints"
    assert args.models_dir == isolated_data.models_dir


def test_a_backtest_writes_its_report_in_a_run_directory_under_the_models_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_data: DataIsolation
) -> None:
    # Arrange
    models = isolated_data.models_dir

    def one_season(_path: object) -> pd.DataFrame:
        return pd.DataFrame({"season": [2023]})

    def empty_backtest(_df: object, _config: object, **_kwargs: object) -> dict[str, object]:
        return {"checkpoint": {}, "per_week": []}

    def report_config(_r: object, _c: object, payload: object, _res: object) -> dict[str, object]:
        return {"config": payload}

    def metadata_config(_c: object, _h: object, payload: object) -> dict[str, object]:
        return {"config": payload}

    monkeypatch.setattr(walk_forward, "load_games", one_season)
    monkeypatch.setattr(walk_forward, "dataset_fingerprint", returning("hash"))
    monkeypatch.setattr(walk_forward, "run_walk_forward_backtest", empty_backtest)
    monkeypatch.setattr(walk_forward, "build_metrics_report", report_config)
    monkeypatch.setattr(walk_forward, "build_metadata", metadata_config)
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
    tmp_path: Path, isolated_data: DataIsolation
) -> None:
    # Arrange
    models = isolated_data.models_dir
    dataset = tmp_path / "games.csv"
    dataset.write_text("season,week\n2025,1\n", encoding="utf-8")
    args = argparse.Namespace(data_path=dataset, run_id="pinned_run", run_dir=None, output_dir=None)

    # Act
    context = pipeline._start_run(args)

    # Assert
    assert context.run_dir == models / "pinned_run"
    assert context.run_dir.is_dir()
    assert context.output_dir == context.run_dir
