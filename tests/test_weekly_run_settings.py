"""The settings the weekly run hands to stage 1 and the final fit, and their resume hashes.

These pin what ``nfl-predictor weekly`` builds from its options, so moving that construction
into shared helpers changes neither the settings nor the stage-marker hashes an existing run
directory resumes from.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from nfl_predictor.ml import artifacts, floor_sigma, ml_model_core, ml_model_xgb_utils, walk_forward
from nfl_predictor.utils import fingerprints
from nfl_predictor.weekly_run import pipeline, stage1

if TYPE_CHECKING:
    from pathlib import Path

    from nfl_predictor.ml.ml_model_training import TrainingOptions


class _StopAtFinalFitError(Exception):
    """Stop the weekly run once the final fit's settings have been captured."""


# The floor sigma's pools, stubbed: their contents are tested in test_weekly_run_floor_sigma.py.
REFERENCE_POOL = floor_sigma.ErrorPool()
PRODUCTION_POOL = floor_sigma.ErrorPool(sources=("stage1",))


def _capture_weekly_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra_argv: list[str]
) -> dict[str, Any]:
    """Run the weekly entrypoint up to the final fit and return what each stage received."""
    captured: dict[str, Any] = {"hashes": []}
    data_path = tmp_path / "completed_games_ml.csv"
    data_path.write_text("season,week,home_spread,total_line\n2025,1,-3.0,44.5\n", encoding="utf-8")

    def fake_reuse(_marker: Path, _dataset_hash: str, config_hash: str, _paths: object) -> bool:
        captured["hashes"].append(config_hash)
        return False

    def fake_stage1(
        _df: pd.DataFrame, options: stage1.ProductionOptions, run: stage1.Stage1Run
    ) -> dict[str, Any]:
        captured["stage1"] = {
            "resume": run.resume,
            "checkpoint_per_fold": run.checkpoint_per_fold,
            "floor_sigma_history": run.floor_sigma_history,
            **vars(options),
        }
        return {"market_mode": options.market_mode}

    def fake_train(options: TrainingOptions) -> None:
        captured["train"] = {
            key: value for key, value in vars(options).items() if key != "data_path"
        }
        raise _StopAtFinalFitError

    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: False)
    monkeypatch.setattr(walk_forward, "load_games", lambda _path: pd.DataFrame())
    monkeypatch.setattr(artifacts, "sha256_file", lambda _path: "hash")
    monkeypatch.setattr(artifacts, "git_commit_hash", lambda: "commit")
    monkeypatch.setattr(fingerprints, "dataset_fingerprint", lambda _path: {"sha256": "fp"})
    monkeypatch.setattr(pipeline, "_stage_can_reuse", fake_reuse)
    monkeypatch.setattr(stage1, "evaluate_production", fake_stage1)
    monkeypatch.setattr(pipeline, "train_margin_total_model_with_report", fake_train)
    monkeypatch.setattr(floor_sigma, "load_reference_pool", lambda _paths: REFERENCE_POOL)
    monkeypatch.setattr(
        pipeline,
        "_production_floor_sigma",
        lambda _args, _run_dir, _reference: (PRODUCTION_POOL, (2025, 2)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_run.py",
            "--skip-data-refresh",
            "--data-path",
            str(data_path),
            "--run-dir",
            str(tmp_path / "run"),
            "--output-dir",
            str(tmp_path / "out"),
            "--xgb-n-jobs",
            "2",
            *extra_argv,
        ],
    )
    with pytest.raises(_StopAtFinalFitError):
        pipeline.main()
    assert captured["stage1"].pop("floor_sigma_history") is REFERENCE_POOL
    assert captured["train"].pop("floor_sigma_pool") is PRODUCTION_POOL
    assert captured["train"].pop("floor_sigma_week") == (2025, 2)
    return captured


def test_default_weekly_settings_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shipped configuration's stage-1 options, final-fit options and resume hashes."""
    captured = _capture_weekly_settings(tmp_path, monkeypatch, [])

    assert captured["stage1"] == {
        "resume": True,
        "checkpoint_per_fold": False,
        "eval_last_n_seasons": 3,
        "wf_start_week": 1,
        "include_postseason": False,
        "exclude_incomplete_seasons": False,
        "recency_half_life_seasons": None,
        "market_mode": "hybrid",
        "xgb_params_overrides": {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.0165,
            "n_jobs": 2,
            "verbosity": 0,
            "device": "cpu",
        },
        "include_quantiles": False,
        "market_transform": None,
        "max_cardinality_ratio": 0.5,
    }
    optuna_config = captured["train"].pop("optuna_config")
    assert optuna_config == ml_model_core.OptunaConfig(
        enabled=False,
        timeout_seconds=600,
        n_trials=None,
        cv_splits=3,
        objective="brier",
        early_stopping_rounds=50,
        tree_method=None,
        device="cpu",
        storage=None,
        study_name=None,
        best_params_out=None,
        xgb_n_jobs=2,
    )
    assert captured["train"] == {
        "holdout_seasons": 0,
        "include_market": True,
        "max_cardinality_ratio": 0.5,
        "market_transform": True,
        "market_anchor": True,
        "include_postseason": False,
        "postseason_weight": 1.3,
        "recency_half_life_seasons": None,
        "min_season": None,
        "max_season": None,
        "feature_start": ml_model_core.DEFAULT_FEATURE_START_COLUMN,
        "feature_end": ml_model_core.DEFAULT_FEATURE_END_COLUMN,
        "xgb_params_overrides": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.0165},
    }
    # Both markers cover the floor sigma's inputs (the reference errors and the final fit's
    # pool and week), so a run directory from before them retrains instead of resuming. The
    # stage-1 hash also covers the start week (now 1).
    assert captured["hashes"] == ["ca6a81c8", "082f6915"]


def test_overridden_weekly_settings_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overrides reach stage 1 and the final fit; the training half-life follows stage 1's."""
    captured = _capture_weekly_settings(
        tmp_path,
        monkeypatch,
        [
            "--xgb-tree-method",
            "hist",
            "--wf-recency-half-life-seasons",
            "8",
            "--wf-include-postseason",
            "--wf-market-mode",
            "anchor",
            "--no-market-transform",
            "--wf-n-estimators",
            "300",
            "--tune",
        ],
    )

    assert captured["stage1"]["xgb_params_overrides"] == {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.0165,
        "n_jobs": 2,
        "verbosity": 0,
        "device": "cpu",
        "tree_method": "hist",
    }
    assert captured["stage1"]["recency_half_life_seasons"] == 8.0
    assert captured["stage1"]["include_postseason"] is True
    assert captured["stage1"]["market_mode"] == "anchor"
    assert captured["stage1"]["market_transform"] is False
    optuna_config = captured["train"].pop("optuna_config")
    assert optuna_config.enabled is True
    assert optuna_config.tree_method == "hist"
    assert optuna_config.storage == f"sqlite:///{(tmp_path / 'run' / 'optuna.db').resolve()}"
    assert captured["train"]["recency_half_life_seasons"] == 8.0
    assert (
        captured["train"]["include_market"],
        captured["train"]["market_transform"],
        captured["train"]["market_anchor"],
    ) == (False, False, True)
    # The final fit's hash covers the Optuna storage path, which names the temporary directory.
    assert captured["hashes"][0] == "8ae118ca"


def test_config_tree_settings_reach_stage1_and_the_final_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The config file's tree budget, depth and learning rate train both stages."""
    config_path = tmp_path / "weekly_run.json"
    config_path.write_text(
        json.dumps({"wf_n_estimators": 300, "wf_max_depth": 3, "wf_learning_rate": 0.05}),
        encoding="utf-8",
    )

    captured = _capture_weekly_settings(tmp_path, monkeypatch, ["--config", str(config_path)])

    expected = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}
    stage1_overrides = captured["stage1"]["xgb_params_overrides"]
    assert {name: stage1_overrides[name] for name in expected} == expected
    assert captured["train"]["xgb_params_overrides"] == expected
