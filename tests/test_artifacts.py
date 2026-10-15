"""Tests for run artifact helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import joblib
import numpy as np

from nfl_predictor.ml import artifacts

if TYPE_CHECKING:
    from pathlib import Path


def _metadata(run_id: str) -> dict[str, object]:
    """Build the metadata payload the round-trip tests write."""
    return artifacts.build_metadata(
        created_at="2025-01-01T00:00:00+00:00",
        run_id=run_id,
        dataset_hash="abc123",
        config={"foo": "bar"},
        details=artifacts.TrainedModelDetails(
            feature_list=["feat1", "feat2"],
            splits={"train_seasons": [2023], "holdout_seasons": [2024]},
            params={"eval_metric": "mae"},
            tuned_params=None,
            early_stopping={"best_iteration": 7},
        ),
    )


def test_artifacts_write_and_load_roundtrip(tmp_path: Path) -> None:
    """Artifacts write/read should be a lossless round-trip."""
    # Arrange
    run_id = "unit_test_run"
    paths = artifacts.resolve_run_paths(run_id, run_dir=tmp_path / run_id)
    model_obj = {"hello": "world", "n": 1}

    # Act
    artifacts.save_model(paths.model_path, model_obj)

    # Assert
    loaded = joblib.load(paths.model_path)
    assert loaded == model_obj


def test_build_metadata_has_required_keys() -> None:
    """The metadata carries every key the artifact contract requires."""
    # Act
    metadata = _metadata("unit_test_run")

    # Assert
    assert "created_at" in metadata
    assert "run_id" in metadata
    assert "git_commit_hash" in metadata
    assert "dataset_hash" in metadata
    assert "library_versions" in metadata
    assert "config" in metadata
    assert "feature_list" in metadata
    assert "splits" in metadata
    assert "params" in metadata
    assert "tuned_params" in metadata
    assert "early_stopping" in metadata


def test_write_json_writes_the_metadata(tmp_path: Path) -> None:
    """The metadata payload is written next to the model."""
    # Arrange
    run_id = "unit_test_run"
    paths = artifacts.resolve_run_paths(run_id, run_dir=tmp_path / run_id)
    metadata = _metadata(run_id)

    # Act
    artifacts.write_json(paths.metadata_path, metadata)

    # Assert
    assert paths.metadata_path.exists()


def test_write_json_writes_the_metrics_report(tmp_path: Path) -> None:
    """The metrics report is written into the run directory."""
    # Arrange
    run_id = "unit_test_run"
    paths = artifacts.resolve_run_paths(run_id, run_dir=tmp_path / run_id)
    report = {"run_id": run_id, "metrics": {"overall": {"margin_mae": 3.0}}}

    # Act
    artifacts.write_json(paths.metrics_path, report)

    # Assert
    assert paths.metrics_path.exists()


def test_write_json_handles_numpy_scalars(tmp_path: Path) -> None:
    """write_json should tolerate numpy scalar types like numpy.int64."""
    # Arrange
    path = tmp_path / "payload.json"
    payload = {
        "a": np.int64(7),
        "b": [np.int64(1), np.float64(2.5)],
        "c": {"nested": np.int64(3)},
    }

    # Act
    artifacts.write_json(path, payload)

    # Assert
    text = path.read_text(encoding="utf-8")
    assert '"a"' in text
