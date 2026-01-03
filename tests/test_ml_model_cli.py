"""Tests for CLI argument handling and orchestration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

from nfl_predictor.ml import ml_model_cli
from nfl_predictor.ml.ml_model_core import OptunaConfig, TrainingResult


def test_main_model_in_no_predict(monkeypatch, tmp_path: Path) -> None:
    """Loading model without prediction works and skips prediction steps."""
    model_path = tmp_path / "model.joblib"
    calls: dict[str, Any] = {}

    def fake_load_model_checkpoint(path: Path, kind: str) -> str:
        calls["model"] = (path, kind)
        return "model"

    def fake_with_market_prob_config(model: str, config: Any) -> str:
        calls["config"] = config
        return model

    monkeypatch.setattr(ml_model_cli, "_load_model_checkpoint", fake_load_model_checkpoint)
    monkeypatch.setattr(ml_model_cli, "_with_market_prob_config", fake_with_market_prob_config)
    monkeypatch.setattr(ml_model_cli, "get_current_nfl_week", lambda: (2024, 1))
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda _: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")

    def fail_predict(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("prediction should not run without --predict-path")

    monkeypatch.setattr(ml_model_cli, "predict_week", fail_predict)
    monkeypatch.setattr(ml_model_cli, "predict_week_margin_total", fail_predict)
    monkeypatch.setattr(ml_model_cli, "predict_week_blended", fail_predict)

    monkeypatch.setattr(sys, "argv", ["prog", "--model-in", str(model_path)])
    ml_model_cli.main()

    assert calls["model"] == (model_path, "margin_total")
    assert calls["config"] is None


def test_main_model_in_predict_defaults_output(monkeypatch, tmp_path: Path) -> None:
    """Loading model with prediction runs prediction with expected args and output path."""
    model_path = tmp_path / "model.joblib"
    predict_path = tmp_path / "week.csv"
    calls: dict[str, Any] = {}

    def fake_load_games(path: Path) -> pd.DataFrame:
        calls["load_games"] = path
        return pd.DataFrame()

    def fake_load_model_checkpoint(path: Path, kind: str) -> str:
        calls["model"] = (path, kind)
        return "model"

    def fake_with_market_prob_config(model: str, config: Any) -> str:
        calls["config"] = config
        return model

    def fake_predict(
        model: str,
        games_path: Path,
        output_path: Path,
        *,
        pretty_output: bool,
        score_rounding: str,
    ) -> pd.DataFrame:
        calls["predict_args"] = (model, games_path, output_path, pretty_output, score_rounding)
        return pd.DataFrame()

    monkeypatch.setattr(ml_model_cli, "_load_games", fake_load_games)
    monkeypatch.setattr(ml_model_cli, "_load_model_checkpoint", fake_load_model_checkpoint)
    monkeypatch.setattr(ml_model_cli, "_with_market_prob_config", fake_with_market_prob_config)
    monkeypatch.setattr(ml_model_cli, "predict_week_margin_total", fake_predict)
    monkeypatch.setattr(ml_model_cli, "get_current_nfl_week", lambda: (2024, 1))
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda _: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model-in",
            str(model_path),
            "--predict-path",
            str(predict_path),
            "--model-kind",
            "margin_total",
        ],
    )
    ml_model_cli.main()

    expected_output = predict_path.with_name("week_predictions.csv")
    assert calls["predict_args"][2] == expected_output
    assert calls["model"] == (model_path, "margin_total")


def test_main_training_writes_artifacts_and_defaults_study(monkeypatch, tmp_path: Path) -> None:
    """Training run writes artifacts and defaults Optuna study name and storage."""
    run_dir = tmp_path / "run_001"
    result = TrainingResult(
        model={"model": "stub"},
        metrics_report={"kind": "train"},
        splits={"train_seasons": [2020], "holdout_seasons": [2021]},
        params={"n_estimators": 1},
        tuned_params=None,
        feature_list=["feat1"],
        early_stopping={"best_iteration": 1},
    )
    calls: dict[str, Any] = {}

    def fake_train_margin_total_model_with_report(**kwargs: object) -> TrainingResult:
        calls["optuna_config"] = kwargs["optuna_config"]
        return result

    def fake_save_model(path: Path, model: object) -> None:
        calls["model_path"] = path
        calls["model"] = model

    def fake_write_json(path: Path, payload: dict[str, object]) -> None:
        if path.name == "metrics_report.json":
            calls["metrics_payload"] = payload
        elif path.name == "metadata.json":
            calls["metadata_payload"] = payload

    monkeypatch.setattr(
        ml_model_cli,
        "train_margin_total_model_with_report",
        fake_train_margin_total_model_with_report,
    )
    monkeypatch.setattr(ml_model_cli.artifacts, "save_model", fake_save_model)
    monkeypatch.setattr(ml_model_cli.artifacts, "write_json", fake_write_json)
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda _: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")
    monkeypatch.setattr(ml_model_cli, "get_current_nfl_week", lambda: (2024, 1))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model-kind",
            "margin_total",
            "--tune",
            "--tune-storage",
            "sqlite:///optuna.db",
            "--run-dir",
            str(run_dir),
        ],
    )
    ml_model_cli.main()

    optuna_config = cast(OptunaConfig, calls["optuna_config"])
    assert optuna_config.storage == "sqlite:///optuna.db"
    assert optuna_config.study_name == "nfl_predictor_margin_total_combined_mae"
    model_path = cast(Path, calls["model_path"])
    assert model_path.parent == run_dir
    metrics_payload = cast(dict[str, Any], calls["metrics_payload"])
    metadata_payload = cast(dict[str, Any], calls["metadata_payload"])
    assert metrics_payload["run_id"] == run_dir.name
    assert metadata_payload["run_id"] == run_dir.name
