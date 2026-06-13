"""Tests for CLI argument handling and orchestration."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from nfl_predictor.ml.ml_model_core import OptunaConfig, TrainingResult


def _import_ml_model_cli(monkeypatch):
    stub = types.SimpleNamespace(
        train_score_model=lambda **_kwargs: None,
        train_margin_total_model=lambda **_kwargs: None,
        train_blended_margin_total_model=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "nfl_predictor.ml_model", stub)
    monkeypatch.delitem(sys.modules, "nfl_predictor.ml.ml_model_cli", raising=False)
    monkeypatch.delitem(sys.modules, "nfl_predictor.ml.ml_model_training", raising=False)
    return importlib.import_module("nfl_predictor.ml.ml_model_cli")


def _default_main_args(ml_model_cli: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return the parsed default CLI namespace for targeted main() tests."""
    monkeypatch.setattr(sys, "argv", ["prog"])
    return ml_model_cli._parse_args()


def test_main_model_in_no_predict(monkeypatch, tmp_path: Path) -> None:
    """Loading model without prediction works and skips prediction steps."""
    ml_model_cli = _import_ml_model_cli(monkeypatch)
    model_path = tmp_path / "model.joblib"
    calls: dict[str, Any] = {}

    def fake_load_model_checkpoint(path: Path, kind: str) -> str:
        """Record path/kind and return model."""
        calls["model"] = (path, kind)
        return "model"

    def fake_with_market_prob_config(model: str, config: Any) -> str:
        """Record config and return model."""
        calls["config"] = config
        return model

    monkeypatch.setattr(ml_model_cli, "_load_model_checkpoint", fake_load_model_checkpoint)
    monkeypatch.setattr(ml_model_cli, "_with_market_prob_config", fake_with_market_prob_config)
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda _: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")

    def fail_predict(*_args: object, **_kwargs: object) -> None:
        """Fail if prediction is attempted."""
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
    ml_model_cli = _import_ml_model_cli(monkeypatch)
    model_path = tmp_path / "model.joblib"
    predict_path = tmp_path / "week.csv"
    calls: dict[str, Any] = {}

    def fake_load_model_checkpoint(path: Path, kind: str) -> str:
        """Record path/kind and return model."""
        calls["model"] = (path, kind)
        return "model"

    def fake_with_market_prob_config(model: str, config: Any) -> str:
        """Record config and return model."""
        calls["config"] = config
        return model

    def fake_predict(
        model: str,
        games_path: Path,
        output_path: Path,
        *,
        pretty_output: bool,
        score_rounding: str,
        win_prob_use_uncertainty: bool = False,
    ) -> pd.DataFrame:
        """Record args and return empty DataFrame."""
        calls["predict_args"] = (
            model,
            games_path,
            output_path,
            pretty_output,
            score_rounding,
        )
        return pd.DataFrame()

    monkeypatch.setattr(ml_model_cli, "_load_model_checkpoint", fake_load_model_checkpoint)
    monkeypatch.setattr(ml_model_cli, "_with_market_prob_config", fake_with_market_prob_config)
    monkeypatch.setattr(ml_model_cli, "predict_week_margin_total", fake_predict)
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
    ml_model_cli = _import_ml_model_cli(monkeypatch)
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
        """Record Optuna config and return training result."""
        calls["optuna_config"] = kwargs["optuna_config"]
        return result

    def fake_save_model(path: Path, model: object) -> None:
        """Record model path and model."""
        calls["model_path"] = path
        calls["model"] = model

    def fake_write_json(path: Path, payload: dict[str, object]) -> None:
        """Record metrics and metadata payloads."""
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


def test_main_training_logistic_alias(monkeypatch, tmp_path: Path) -> None:
    """Logistic alias should map to platt for training."""
    ml_model_cli = _import_ml_model_cli(monkeypatch)
    run_dir = tmp_path / "run_alias"
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
        """Record win-prob calibration argument."""
        calls["win_prob_calibration"] = kwargs["win_prob_calibration"]
        return result

    monkeypatch.setattr(
        ml_model_cli,
        "train_margin_total_model_with_report",
        fake_train_margin_total_model_with_report,
    )
    monkeypatch.setattr(ml_model_cli.artifacts, "save_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ml_model_cli.artifacts, "write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda *_: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model-kind",
            "margin_total",
            "--win-prob-calibration",
            "logistic",
            "--run-dir",
            str(run_dir),
        ],
    )
    ml_model_cli.main()

    assert calls["win_prob_calibration"] == "platt"


def test_main_model_in_with_tune_logs(monkeypatch, tmp_path: Path) -> None:
    """Loading model with --tune logs appropriate messages."""
    ml_model_cli = _import_ml_model_cli(monkeypatch)
    model_path = tmp_path / "model.joblib"
    messages: list[str] = []

    def fake_log(fmt: str, *args: object) -> None:
        """Capture logged messages."""
        messages.append(fmt % args if args else fmt)

    monkeypatch.setattr(ml_model_cli.log, "info", fake_log)
    monkeypatch.setattr(ml_model_cli, "_load_model_checkpoint", lambda *_args, **_kwargs: "model")
    monkeypatch.setattr(ml_model_cli, "_with_market_prob_config", lambda model, _cfg: model)
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda _: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")

    monkeypatch.setattr(sys, "argv", ["prog", "--model-in", str(model_path), "--tune"])
    ml_model_cli.main()

    assert any("Model checkpoint provided; ignoring training" in msg for msg in messages)
    assert any("No --predict-path provided" in msg for msg in messages)


def test_main_model_in_predict_score(monkeypatch, tmp_path: Path) -> None:
    """Loading score model with prediction runs prediction with expected args."""
    ml_model_cli = _import_ml_model_cli(monkeypatch)
    model_path = tmp_path / "model.joblib"
    predict_path = tmp_path / "week.csv"
    calls: dict[str, Any] = {}

    monkeypatch.setattr(ml_model_cli, "_load_model_checkpoint", lambda *_args, **_kwargs: "model")
    monkeypatch.setattr(ml_model_cli, "_with_market_prob_config", lambda model, _cfg: model)
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda _: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")

    def fake_predict(*args: object, **kwargs: object) -> pd.DataFrame:
        """Record args and return empty DataFrame."""
        calls["args"] = args
        calls["kwargs"] = kwargs
        return pd.DataFrame()

    monkeypatch.setattr(ml_model_cli, "predict_week", fake_predict)

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
            "score",
        ],
    )
    ml_model_cli.main()

    assert calls["args"][1] == predict_path


def test_main_score_training_predicts_and_writes(monkeypatch, tmp_path: Path) -> None:
    """Training score model runs prediction and writes artifacts."""
    ml_model_cli = _import_ml_model_cli(monkeypatch)
    run_dir = tmp_path / "run_002"
    predict_path = tmp_path / "games.csv"
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

    monkeypatch.setattr(ml_model_cli, "train_score_model_with_report", lambda **_kwargs: result)
    monkeypatch.setattr(
        ml_model_cli.artifacts,
        "save_model",
        lambda path, _model: calls.setdefault("model_path", path),
    )
    monkeypatch.setattr(ml_model_cli.artifacts, "write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda _: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")

    def fake_predict(*_args: object, **_kwargs: object) -> pd.DataFrame:
        """Record that prediction was called and return empty DataFrame."""
        calls["predicted"] = True
        return pd.DataFrame()

    monkeypatch.setattr(ml_model_cli, "predict_week", fake_predict)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model-kind",
            "score",
            "--run-dir",
            str(run_dir),
            "--predict-path",
            str(predict_path),
        ],
    )
    ml_model_cli.main()

    assert Path(calls["model_path"]).parent == run_dir
    assert calls["predicted"] is True


def test_main_blend_training_predict(monkeypatch, tmp_path: Path) -> None:
    """Training blended model runs prediction."""
    ml_model_cli = _import_ml_model_cli(monkeypatch)
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

    monkeypatch.setattr(
        ml_model_cli,
        "train_blended_margin_total_model_with_report",
        lambda **_kwargs: result,
    )
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda _: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")

    def fake_predict(*_args: object, **_kwargs: object) -> pd.DataFrame:
        """Record that prediction was called and return empty DataFrame."""
        calls["predicted"] = True
        return pd.DataFrame()

    monkeypatch.setattr(ml_model_cli, "predict_week_blended", fake_predict)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model-kind",
            "blend",
            "--predict-path",
            str(tmp_path / "games.csv"),
        ],
    )
    ml_model_cli.main()

    assert calls["predicted"] is True


def test_main_model_outside_run_dir_raises(monkeypatch, tmp_path: Path) -> None:
    """Training with model output outside run dir raises error."""
    ml_model_cli = _import_ml_model_cli(monkeypatch)
    run_dir = tmp_path / "run_003"
    model_out = tmp_path / "model.joblib"
    result = TrainingResult(
        model={"model": "stub"},
        metrics_report={"kind": "train"},
        splits={"train_seasons": [2020], "holdout_seasons": [2021]},
        params={"n_estimators": 1},
        tuned_params=None,
        feature_list=["feat1"],
        early_stopping={"best_iteration": 1},
    )

    monkeypatch.setattr(ml_model_cli, "train_score_model_with_report", lambda **_kwargs: result)
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda _: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model-kind",
            "score",
            "--run-dir",
            str(run_dir),
            "--model-out",
            str(model_out),
        ],
    )

    with pytest.raises(ValueError):
        ml_model_cli.main()


def test_main_rejects_both_recency_half_life_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Main should reject simultaneous week-based and season-based recency flags."""
    ml_model_cli = _import_ml_model_cli(monkeypatch)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--recency-half-life-weeks",
            "2",
            "--recency-half-life-seasons",
            "1",
        ],
    )

    with pytest.raises(ValueError, match="Specify only one of --recency-half-life-weeks"):
        ml_model_cli.main()


def test_main_model_in_blend_predict_uses_market_prob_config_and_disables_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Blend checkpoint loads should keep explicit market config and warn on uncertainty flags."""
    ml_model_cli = _import_ml_model_cli(monkeypatch)
    model_path = tmp_path / "blend.joblib"
    predict_path = tmp_path / "week.csv"
    calls: dict[str, Any] = {}
    warnings: list[str] = []

    monkeypatch.setattr(ml_model_cli, "_load_model_checkpoint", lambda *_args, **_kwargs: "model")
    monkeypatch.setattr(
        ml_model_cli,
        "_with_market_prob_config",
        lambda model, config: calls.setdefault("config", config) or model,
    )
    monkeypatch.setattr(
        ml_model_cli,
        "predict_week_blended",
        lambda *args, **kwargs: calls.setdefault("predict", (args, kwargs)) or pd.DataFrame(),
    )
    monkeypatch.setattr(
        ml_model_cli.log, "warning", lambda message, *_args: warnings.append(message)
    )
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda *_args, **_kwargs: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model-in",
            str(model_path),
            "--model-kind",
            "blend",
            "--predict-path",
            str(predict_path),
            "--market-prob-weight",
            "0.25",
            "--win-prob-uncertainty",
        ],
    )

    ml_model_cli.main()

    config = calls["config"]
    assert config is not None
    assert config.blend_weight == 0.25
    assert config.clamp_delta == 0.0
    assert any("Uncertainty-aware win prob is only supported" in warning for warning in warnings)
    predict_args, predict_kwargs = calls["predict"]
    assert predict_args[1] == predict_path
    assert predict_args[2] == predict_path.with_name("week_predictions.csv")
    assert predict_kwargs["score_rounding"] == "none"


def test_main_score_training_without_artifacts_or_prediction_skips_optional_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Score training should not write artifacts or predict unless those outputs are requested."""
    ml_model_cli = _import_ml_model_cli(monkeypatch)
    result = TrainingResult(
        model={"model": "stub"},
        metrics_report={"kind": "train"},
        splits={"train_seasons": [2020], "holdout_seasons": [2021]},
        params={"n_estimators": 1},
        tuned_params=None,
        feature_list=["feat1"],
        early_stopping={"best_iteration": 1},
    )

    monkeypatch.setattr(ml_model_cli, "train_score_model_with_report", lambda **_kwargs: result)
    monkeypatch.setattr(
        ml_model_cli.artifacts,
        "save_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("save_model not expected")),
    )
    monkeypatch.setattr(
        ml_model_cli.artifacts,
        "write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("write_json not expected")),
    )
    monkeypatch.setattr(
        ml_model_cli,
        "predict_week",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("predict not expected")),
    )
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda *_args, **_kwargs: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")

    monkeypatch.setattr(sys, "argv", ["prog", "--model-kind", "score"])
    ml_model_cli.main()


def test_main_margin_total_training_predicts_without_writing_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Margin-total training should predict even when no model output path is requested."""
    ml_model_cli = _import_ml_model_cli(monkeypatch)
    predict_path = tmp_path / "games.csv"
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

    monkeypatch.setattr(
        ml_model_cli,
        "train_margin_total_model_with_report",
        lambda **_kwargs: result,
    )
    monkeypatch.setattr(
        ml_model_cli.artifacts,
        "save_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("save_model not expected")),
    )
    monkeypatch.setattr(
        ml_model_cli.artifacts,
        "write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("write_json not expected")),
    )
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda *_args, **_kwargs: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")

    def fake_predict(*args: object, **kwargs: object) -> pd.DataFrame:
        """Capture prediction arguments for the margin-total path."""
        calls["predict"] = (args, kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(ml_model_cli, "predict_week_margin_total", fake_predict)

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--model-kind", "margin_total", "--predict-path", str(predict_path)],
    )
    ml_model_cli.main()

    predict_args, predict_kwargs = calls["predict"]
    assert predict_args[1] == predict_path
    assert predict_args[2] == predict_path.with_name("games_predictions.csv")
    assert predict_kwargs["win_prob_use_uncertainty"] is False


def test_main_blend_training_writes_feature_importance_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Blend training should write the optional feature-importance artifact when available."""
    ml_model_cli = _import_ml_model_cli(monkeypatch)
    run_dir = tmp_path / "run_blend"
    result = TrainingResult(
        model={"model": "stub"},
        metrics_report={"kind": "train"},
        splits={"train_seasons": [2020], "holdout_seasons": [2021]},
        params={"n_estimators": 1},
        tuned_params=None,
        feature_list=["feat1"],
        early_stopping={"best_iteration": 1},
        feature_importance={"team": {"feat1": 0.7}},
    )
    calls: dict[str, dict[str, object]] = {}

    monkeypatch.setattr(
        ml_model_cli,
        "train_blended_margin_total_model_with_report",
        lambda **_kwargs: result,
    )
    monkeypatch.setattr(ml_model_cli.artifacts, "save_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda *_args, **_kwargs: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")
    monkeypatch.setattr(
        ml_model_cli.artifacts,
        "write_json",
        lambda path, payload: calls.setdefault(path.name, cast(dict[str, object], payload)),
    )
    monkeypatch.setattr(
        ml_model_cli,
        "predict_week_blended",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("predict not expected")),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--model-kind", "blend", "--run-dir", str(run_dir)],
    )
    ml_model_cli.main()

    assert "feature_importance.json" in calls
    assert calls["feature_importance.json"]["team"] == {"feat1": 0.7}


def test_main_unknown_model_kind_raises_for_loaded_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The model-loading path should still reject unknown model kinds defensively."""
    ml_model_cli = _import_ml_model_cli(monkeypatch)
    args = _default_main_args(ml_model_cli, monkeypatch)
    args.model_kind = "unknown"
    args.model_in = tmp_path / "model.joblib"
    args.predict_path = tmp_path / "week.csv"

    monkeypatch.setattr(ml_model_cli, "_parse_args", lambda: args)
    monkeypatch.setattr(ml_model_cli, "_load_model_checkpoint", lambda *_args, **_kwargs: "model")
    monkeypatch.setattr(ml_model_cli, "_with_market_prob_config", lambda model, _cfg: model)
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda *_args, **_kwargs: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")

    with pytest.raises(ValueError, match="Unknown model kind: unknown"):
        ml_model_cli.main()


def test_main_unknown_model_kind_raises_for_training_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The training path should still reject unknown model kinds defensively."""
    ml_model_cli = _import_ml_model_cli(monkeypatch)
    args = _default_main_args(ml_model_cli, monkeypatch)
    args.model_kind = "unknown"
    args.model_in = None

    monkeypatch.setattr(ml_model_cli, "_parse_args", lambda: args)
    monkeypatch.setattr(ml_model_cli.artifacts, "sha256_file", lambda *_args, **_kwargs: "hash")
    monkeypatch.setattr(ml_model_cli.artifacts, "now_utc_iso", lambda: "time")

    with pytest.raises(ValueError, match="Unknown model kind: unknown"):
        ml_model_cli.main()
