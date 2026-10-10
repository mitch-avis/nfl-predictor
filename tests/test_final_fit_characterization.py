"""Characterization tests: the final fit's features and predictions with no group left out.

The model the week's picks come from is trained two ways: by ``nfl-predictor train`` and by the
weekly run's final fit (its stage 2, after stage 1's walk-forward). Both run here on the weekly
fixture with columns from four feature groups added (``tests/feature_group_fixture.py``), with
no feature group left out, and the feature list each records in ``metadata.json``, the
predictions each writes, and stage 1's summary row are compared with the snapshots under
``tests/fixtures/final_fit_characterization/``. So the default must keep every group column
and train exactly as before. Runs use the CPU and one thread for determinism, and the fixture's
reference run for the floor sigma; fields that change on every run (timestamps, durations, and
the fingerprint that hashes the git commit) are dropped first. Rewrite the snapshots with
``NFLP_UPDATE_SNAPSHOTS=1``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from tests import snapshots
from tests.feature_group_fixture import (
    GROUP_COLUMNS,
    build_group_fixture,
    write_group_weekly_config,
)
from tests.weekly_fixture import CURRENT_SEASON, PREDICT_WEEK

from nfl_predictor.cli import train
from nfl_predictor.ml import walk_forward
from nfl_predictor.weekly_run import pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence

SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "final_fit_characterization"
TEST_BOOTSTRAP_SAMPLES = 200
VOLATILE_FIELDS = frozenset({"completed_at", "duration_seconds", "wf_run_fingerprint"})
EVERY_GROUP_COLUMN = frozenset(column for group in GROUP_COLUMNS.values() for column in group)


def _run_train(argv: Sequence[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ``nfl-predictor train`` with ``argv``."""
    monkeypatch.setattr(sys, "argv", ["nfl-predictor train", *argv])
    train.main()


def _run_weekly(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ``nfl-predictor weekly`` on ``config_path`` with a small paired bootstrap."""
    monkeypatch.setattr(walk_forward, "BOOTSTRAP_SAMPLES", TEST_BOOTSTRAP_SAMPLES)
    monkeypatch.setattr(sys, "argv", ["nfl-predictor weekly", "--config", str(config_path)])
    assert pipeline.main() == 0


def _feature_list(run_dir: Path) -> list[str]:
    """Return the feature list a run's ``metadata.json`` records."""
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    return list(metadata["feature_list"])


def _stable_frame(path: Path) -> pd.DataFrame:
    """Return a CSV output without the columns that change on every run."""
    frame = pd.read_csv(path)
    return frame.drop(columns=[column for column in frame.columns if column in VOLATILE_FIELDS])


def test_train_keeps_every_group_column_and_predicts_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path)
    run_dir = tmp_path / "train_run"
    predictions_path = tmp_path / "train_predictions.csv"
    argv = [
        "--data-path",
        str(paths["completed"]),
        "--holdout-seasons",
        "0",
        "--floor-sigma-reference-runs",
        str(paths["reference"]),
        "--xgb-device",
        "cpu",
        "--xgb-n-jobs",
        "1",
        "--run-dir",
        str(run_dir),
        "--run-id",
        "characterization",
        "--predict-path",
        str(paths["predict"]),
        "--output-path",
        str(predictions_path),
        "--no-pretty-output",
    ]

    # Act
    _run_train(argv, monkeypatch)

    # Assert
    features = _feature_list(run_dir)
    assert set(features) >= EVERY_GROUP_COLUMN
    snapshots.check_json(
        "train_feature_list.json", features, SNAPSHOT_DIR / "train_feature_list.json"
    )
    snapshots.check_csv(
        "train_predictions.csv",
        _stable_frame(predictions_path),
        SNAPSHOT_DIR / "train_predictions.csv",
    )
    if snapshots.updating():
        pytest.skip(f"snapshots rewritten in {SNAPSHOT_DIR}")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_weekly_final_fit_keeps_every_group_column_and_predicts_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path)
    config_path = write_group_weekly_config(tmp_path, paths)
    suffix = f"season_{CURRENT_SEASON}_week_{PREDICT_WEEK:02d}"

    # Act
    _run_weekly(config_path, monkeypatch)

    # Assert
    features = _feature_list(tmp_path / "run")
    assert set(features) >= EVERY_GROUP_COLUMN
    snapshots.check_json(
        "weekly_feature_list.json", features, SNAPSHOT_DIR / "weekly_feature_list.json"
    )
    snapshots.check_csv(
        "weekly_predictions.csv",
        _stable_frame(tmp_path / "out" / f"{suffix}_predictions.csv"),
        SNAPSHOT_DIR / "weekly_predictions.csv",
    )
    snapshots.check_csv(
        "weekly_wf_compare.csv",
        _stable_frame(tmp_path / "run" / "wf_compare.csv"),
        SNAPSHOT_DIR / "weekly_wf_compare.csv",
    )
    if snapshots.updating():
        pytest.skip(f"snapshots rewritten in {SNAPSHOT_DIR}")
