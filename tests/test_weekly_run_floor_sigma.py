"""The weekly run's floor sigma: reference runs, stage 1's own folds, and the final fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import xgboost as xgb

from nfl_predictor import constants
from nfl_predictor.ml import artifacts, floor_sigma, ml_model_core, walk_forward
from nfl_predictor.utils import fingerprints
from nfl_predictor.weekly_run import config as run_config
from nfl_predictor.weekly_run import pipeline, stage1


def _errors(rows: list[tuple[str, int, int, float]]) -> pd.DataFrame:
    """Return an error pool from ``(game_id, season, week, squared_error)`` rows."""
    return pd.DataFrame(rows, columns=list(floor_sigma.ERROR_COLUMNS))


def test_the_reference_runs_default_to_the_gpu_reference_in_code_and_config() -> None:
    """Both seeds of the reference walk-forward are the default pool, in code and shipped config."""
    expected = [Path(path) for path in constants.FLOOR_SIGMA_REFERENCE_RUNS]

    assert run_config._build_parser().parse_args([]).floor_sigma_reference_runs == expected
    assert run_config._parse_args([]).floor_sigma_reference_runs == expected


def test_a_config_can_name_other_reference_runs_or_none(tmp_path: Path) -> None:
    """A config lists run directories; an empty list means no reference runs."""
    config_path = tmp_path / "weekly.json"
    config_path.write_text('{"floor_sigma_reference_runs": ["a/b", "/c"]}', encoding="utf-8")
    empty_path = tmp_path / "empty.json"
    empty_path.write_text('{"floor_sigma_reference_runs": []}', encoding="utf-8")

    named = run_config._parse_args(["--config", str(config_path)])
    empty = run_config._parse_args(["--config", str(empty_path)])

    assert named.floor_sigma_reference_runs == [Path("a/b"), Path("/c")]
    assert empty.floor_sigma_reference_runs == []


def test_stage1_walks_forward_with_the_history_and_saves_its_own_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 1 pools the reference errors and writes its folds' errors for the final fit."""
    seen: dict[str, object] = {}
    predictions = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2024, 2024],
            "week": [3, 3],
            "actual_margin": [7.0, -3.0],
            "predicted_margin": [4.0, 1.0],
        }
    )

    def _fake_run(_df: pd.DataFrame, _config: object, **kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"overall": {}, "reliability": [], "predictions": predictions}

    monkeypatch.setattr(walk_forward, "run_walk_forward_backtest", _fake_run)
    history = floor_sigma.ErrorPool(_errors([("r1", 2020, 1, 4.0)]), ("ref",))

    stage1.evaluate_production(
        pd.DataFrame(),
        run_dir=tmp_path,
        resume=True,
        dataset_fingerprint={"sha256": "fp"},
        wf_run_fingerprint="wf",
        checkpoint_per_fold=False,
        eval_last_n_seasons=3,
        wf_start_week=3,
        include_postseason=False,
        exclude_incomplete_seasons=False,
        recency_half_life_seasons=None,
        market_mode="hybrid",
        xgb_params_overrides={},
        include_quantiles=False,
        floor_sigma_history=history,
    )

    assert seen["floor_sigma_history"] is history
    saved = stage1.read_margin_errors(tmp_path)
    assert saved["squared_error"].tolist() == [9.0, 16.0]
    assert stage1.margin_errors_path(tmp_path).exists()


def test_reading_missing_stage1_errors_is_an_error(tmp_path: Path) -> None:
    """The final fit never silently drops the current season's errors."""
    with pytest.raises(FileNotFoundError, match="wf_margin_errors"):
        stage1.read_margin_errors(tmp_path)


def _args(tmp_path: Path, predict_rows: pd.DataFrame) -> argparse.Namespace:
    """Return the options the production pool reads, with a written week to predict."""
    predict_path = tmp_path / "week_05_games_to_predict.csv"
    predict_rows.to_csv(predict_path, index=False)
    data_path = tmp_path / "completed.csv"
    pd.DataFrame({"season": [2024], "week": [4]}).to_csv(data_path, index=False)
    return argparse.Namespace(predict_path=predict_path, data_path=data_path)


def _stage1_errors(run_dir: Path) -> None:
    """Write stage 1's errors: 2023 week 3, and 2024 weeks 3-6."""
    frame = _errors(
        [
            ("s2023_3", 2023, 3, 1.0),
            ("s2024_3", 2024, 3, 9.0),
            ("s2024_4", 2024, 4, 16.0),
            ("s2024_5", 2024, 5, 25.0),
            ("s2024_6", 2024, 6, 36.0),
        ]
    )
    stage1.write_margin_errors(run_dir, frame)


def test_the_reference_keeps_its_weeks_and_stage1_adds_the_rest_before_the_week(
    tmp_path: Path,
) -> None:
    """Stage 1's rows join only at weeks the reference lacks, and only before the predicted week."""
    _stage1_errors(tmp_path)
    reference = floor_sigma.ErrorPool(
        _errors([("r2023_3", 2023, 3, 100.0), ("r2024_3", 2024, 3, 100.0)]), ("ref",)
    )
    args = _args(tmp_path, pd.DataFrame({"season": [2024, 2024], "week": [5, 5]}))

    pool, week = pipeline._production_floor_sigma(args, tmp_path, reference)

    assert week == (2024, 5)
    assert sorted(pool.errors["game_id"]) == ["r2023_3", "r2024_3", "s2024_4"]
    assert pool.sources == ("ref", str(stage1.margin_errors_path(tmp_path)))


def test_stage1_fills_every_season_after_the_reference(tmp_path: Path) -> None:
    """A reference ending in 2025: stage 1 adds all of 2026 and 2027 weeks 1-4, not 2025 again."""
    reference = floor_sigma.ErrorPool(
        _errors(
            [
                (f"r{season}_{week}", season, week, 100.0)
                for season in range(2007, 2026)
                for week in (1, 2, 3, 4, 5, 6)
            ]
        ),
        ("ref",),
    )
    stage1.write_margin_errors(
        tmp_path,
        _errors(
            [
                (f"s{season}_{week}", season, week, 1.0)
                for season in (2025, 2026, 2027)
                for week in (1, 2, 3, 4, 5, 6)
            ]
        ),
    )
    args = _args(tmp_path, pd.DataFrame({"season": [2027], "week": [5]}))

    pool, week = pipeline._production_floor_sigma(args, tmp_path, reference)

    assert week == (2027, 5)
    errors = pool.errors
    assert sorted(errors.loc[errors["game_id"].str.startswith("r"), "season"].unique()) == list(
        range(2007, 2026)
    )
    stage1_rows = errors[errors["game_id"].str.startswith("s")]
    assert sorted(zip(stage1_rows["season"], stage1_rows["week"], strict=True)) == [
        *((2026, week) for week in range(1, 7)),
        *((2027, week) for week in range(1, 5)),
    ]
    reference_weeks = set(zip(reference.errors["season"], reference.errors["week"], strict=True))
    assert not reference_weeks & set(zip(stage1_rows["season"], stage1_rows["week"], strict=True))


def test_a_multi_week_prediction_file_estimates_for_the_week_after_the_data(
    tmp_path: Path,
) -> None:
    """Without one predicted week, the sigma is for the week after the newest completed game."""
    _stage1_errors(tmp_path)
    args = _args(tmp_path, pd.DataFrame({"season": [2024, 2024], "week": [5, 6]}))

    _pool, week = pipeline._production_floor_sigma(args, tmp_path, floor_sigma.ErrorPool())

    assert week == (2024, 5)


def _main_argv(tmp_path: Path, *extra: str) -> list[str]:
    """Return a weekly-run command line on a stub dataset."""
    data_path = tmp_path / "completed_games_ml.csv"
    data_path.write_text("season,week\n2025,1\n", encoding="utf-8")
    return [
        "weekly_run.py",
        "--skip-data-refresh",
        "--data-path",
        str(data_path),
        "--run-dir",
        str(tmp_path / "run"),
        "--output-dir",
        str(tmp_path / "out"),
        *extra,
    ]


def _stub_run_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the dataset reads and every stage marker."""
    monkeypatch.setattr(walk_forward, "load_games", lambda _path: pd.DataFrame())
    monkeypatch.setattr(artifacts, "sha256_file", lambda _path: "hash")
    monkeypatch.setattr(fingerprints, "dataset_fingerprint", lambda _path: {"sha256": "fp"})
    monkeypatch.setattr(pipeline, "_stage_can_reuse", lambda *_args, **_kwargs: False)


def test_a_missing_reference_run_stops_the_weekly_run_before_stage1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probability path never changes silently: a missing run is an error, not a fallback."""
    walked: list[object] = []
    _stub_run_inputs(monkeypatch)
    monkeypatch.setattr(stage1, "evaluate_production", lambda *a, **k: walked.append(a))
    monkeypatch.setattr(
        sys,
        "argv",
        _main_argv(tmp_path, "--floor-sigma-reference-runs", str(tmp_path / "gone")),
    )

    with pytest.raises(FileNotFoundError, match="gone"):
        pipeline.main()
    assert walked == []


def test_stage1_and_the_final_fit_receive_the_floor_sigma_pools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 1 gets the reference as its history; the final fit gets the production pool."""

    class _StopAtFinalFitError(Exception):
        """Stop weekly_run once the final fit's options have been captured."""

    captured: dict[str, object] = {}
    reference = floor_sigma.ErrorPool(_errors([("r", 2020, 1, 4.0)]), ("ref",))
    production = floor_sigma.ErrorPool(_errors([("p", 2020, 1, 9.0)]), ("ref", "stage1"))

    def _fake_stage1(_df: pd.DataFrame, **kwargs: object) -> dict[str, object]:
        captured["stage1"] = kwargs["floor_sigma_history"]
        return {"market_mode": "hybrid"}

    def _fake_train(**kwargs: object) -> None:
        captured["pool"] = kwargs["floor_sigma_pool"]
        captured["week"] = kwargs["floor_sigma_week"]
        raise _StopAtFinalFitError()

    _stub_run_inputs(monkeypatch)
    monkeypatch.setattr(floor_sigma, "load_reference_pool", lambda _paths: reference)
    monkeypatch.setattr(stage1, "evaluate_production", _fake_stage1)
    monkeypatch.setattr(
        pipeline, "_production_floor_sigma", lambda _a, _d, ref: (production, (2025, 2))
    )
    monkeypatch.setattr(pipeline, "train_margin_total_model_with_report", _fake_train)
    monkeypatch.setattr(sys, "argv", _main_argv(tmp_path))

    with pytest.raises(_StopAtFinalFitError):
        pipeline.main()

    assert captured == {"stage1": reference, "pool": production, "week": (2025, 2)}


def test_training_artifacts_record_the_models_floor_sigma(tmp_path: Path) -> None:
    """The saved model's metadata carries its sigma record, so it predicts without the pool."""
    record = floor_sigma.FloorSigma(13.2, False, 2025, 2, 900, (2020, 2021, 2022), ("ref",))
    result = ml_model_core.TrainingResult(
        model=SimpleNamespace(margin_model=xgb.XGBRegressor(device="cpu"), floor_sigma=record),
        metrics_report={},
        splits={},
        params={},
        tuned_params=None,
        feature_list=["feat1"],
        early_stopping={},
    )

    paths = pipeline._write_training_artifacts(
        result,
        run_id="weekly_test",
        run_dir=tmp_path,
        created_at="2026-09-28T00:00:00+00:00",
        dataset_hash="hash",
        config_payload={},
    )

    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    assert floor_sigma.FloorSigma.from_dict(metadata["floor_sigma"]) == record
