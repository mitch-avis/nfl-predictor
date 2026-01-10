"""Tests for the betting pipeline script.

These tests are intentionally lightweight and only validate that the script can be imported
and that --dry-run exits successfully.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from scripts import betting_pipeline


def test_betting_pipeline_dry_run_exits_successfully() -> None:
    """The betting pipeline script should support a dry run without heavy work."""

    old_argv = sys.argv
    try:
        sys.argv = [
            "betting_pipeline.py",
            "--dry-run",
            "--run-id",
            "test_betting_pipeline",
        ]
        exit_code = betting_pipeline.main()
    finally:
        sys.argv = old_argv
    assert isinstance(exit_code, int)
    assert exit_code == 0


def test_betting_pipeline_stage2_passes_calibration_for_blend(tmp_path, monkeypatch) -> None:
    """Stage 2 should always provide a calibration window for blended training.

    This is a lightweight regression test that avoids expensive model fitting.
    """

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Minimal dataset (only needs to exist + be readable for stage-1 reuse).
    data_path = tmp_path / "tiny.csv"
    pd.DataFrame(
        [
            {"season": 2024, "week": 1, "away_score": 10, "home_score": 20},
            {"season": 2024, "week": 2, "away_score": 17, "home_score": 14},
        ]
    ).to_csv(data_path, index=False)

    # Force stage 1 reuse so we don't run walk-forward.
    (run_dir / "wf_compare.csv").write_text("label,brier,log_loss\n", encoding="utf-8")
    (run_dir / "wf_best.json").write_text(
        (
            '{"label":"elo_base","calibration":"elo",'
            '"market_prob_weight":0.0,"market_prob_clamp":0.0}'
        ),
        encoding="utf-8",
    )

    # Skip stage 3 by pre-creating the expected artifacts.
    (run_dir / "final_model.joblib").write_text("", encoding="utf-8")
    (run_dir / "predictions.csv").write_text("", encoding="utf-8")

    def fake_train(**kwargs):
        assert kwargs.get("calibration_seasons", 0) > 0 or kwargs.get("calibration_weeks", 0) > 0
        assert kwargs.get("market_anchor") is False
        return betting_pipeline.TrainingResult(
            model={"dummy": True},
            metrics_report={},
            splits={},
            params={},
            tuned_params=None,
            feature_list=[],
            early_stopping={},
        )

    monkeypatch.setattr(
        betting_pipeline,
        "train_blended_margin_total_model_with_report",
        fake_train,
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "betting_pipeline.py",
            "--data-path",
            str(Path(data_path)),
            "--predict-path",
            str(tmp_path / "predict.csv"),
            "--run-id",
            "test_betting_pipeline_stage2",
            "--run-dir",
            str(Path(run_dir)),
        ]
        exit_code = betting_pipeline.main()
    finally:
        sys.argv = old_argv

    assert exit_code == 0
