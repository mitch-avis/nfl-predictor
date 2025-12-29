"""Tests for run artifact helpers."""

from __future__ import annotations

from pathlib import Path

import joblib

from nfl_predictor.ml import artifacts


def test_artifacts_write_and_load_roundtrip(tmp_path: Path) -> None:
    """Artifacts write/read should be a lossless round-trip."""
    run_id = "unit_test_run"
    paths = artifacts.resolve_run_paths(run_id, run_dir=tmp_path / run_id)

    model_obj = {"hello": "world", "n": 1}
    artifacts.save_model(paths.model_path, model_obj)

    metadata = artifacts.build_metadata(
        created_at="2025-01-01T00:00:00+00:00",
        run_id=run_id,
        dataset_hash="abc123",
        config={"foo": "bar"},
        feature_list=["feat1", "feat2"],
        splits={"train_seasons": [2023], "holdout_seasons": [2024]},
        params={"eval_metric": "mae"},
        tuned_params=None,
        early_stopping={"best_iteration": 7},
    )
    artifacts.write_json(paths.metadata_path, metadata)

    report = {"run_id": run_id, "metrics": {"overall": {"margin_mae": 3.0}}}
    artifacts.write_json(paths.metrics_path, report)

    loaded = joblib.load(paths.model_path)
    assert loaded == model_obj

    # Required keys present.
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
