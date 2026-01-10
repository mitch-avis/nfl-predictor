"""Tests for objective model comparison helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pytest import MonkeyPatch

from nfl_predictor.ml import walk_forward
from nfl_predictor.ml.ml_model_core import (
    BlendedMarginTotalModel,
    BlendLayer,
    FeatureSpec,
    MarginTotalModel,
    MarketProbConfig,
    WinProbCalibrator,
)
from nfl_predictor.ml.model_compare import (
    CompareConfig,
    ModelRecipe,
    bootstrap_overall_metrics,
    load_model,
    recipe_from_model,
    run_objective_compare,
    write_compare_outputs,
)


def _dummy_feature_spec(*, include_market: bool) -> FeatureSpec:
    """Build a minimal FeatureSpec suitable for unit tests."""

    cols = ["away_rest", "home_rest"]
    market_cols = ["home_moneyline", "away_moneyline"] if include_market else []
    return FeatureSpec(
        feature_columns=cols + market_cols,
        categorical_columns=[],
        numeric_columns=cols + market_cols,
        dropped_columns=[],
        id_columns=[],
        constant_columns=[],
        high_cardinality_columns=[],
        feature_start="away_rest",
        feature_end="home_moneyline",
        metadata_columns=[],
        post_feature_columns=[],
        market_columns=market_cols,
    )


def test_recipe_from_margin_total_model() -> None:
    """Extracts recipe fields from a MarginTotalModel."""

    model = MarginTotalModel(
        preprocessor=None,  # type: ignore[arg-type]
        feature_spec=_dummy_feature_spec(include_market=True),
        margin_model=None,  # type: ignore[arg-type]
        total_model=None,  # type: ignore[arg-type]
        target_columns=("away_score", "home_score"),
        calibrator=WinProbCalibrator(method="elo", model=None),
        market_anchor=True,
        market_prob_config=MarketProbConfig(blend_weight=0.2, clamp_delta=0.1),
        xgb_params={"n_estimators": 123},
    )

    recipe = recipe_from_model(model, label="A")
    assert recipe.kind == "margin_total"
    assert recipe.calibration_method == "elo"
    assert recipe.market_anchor is True
    assert recipe.include_market_features is True
    assert recipe.market_prob_config.blend_weight == 0.2
    assert recipe.xgb_params["n_estimators"] == 123


def test_recipe_from_blended_model() -> None:
    """Extracts team XGB params from a BlendedMarginTotalModel."""

    team_model = MarginTotalModel(
        preprocessor=None,  # type: ignore[arg-type]
        feature_spec=_dummy_feature_spec(include_market=False),
        margin_model=None,  # type: ignore[arg-type]
        total_model=None,  # type: ignore[arg-type]
        target_columns=("away_score", "home_score"),
        calibrator=None,
        market_anchor=False,
        market_prob_config=MarketProbConfig(blend_weight=0.2, clamp_delta=0.1),
        xgb_params={"n_estimators": 10},
    )

    model = BlendedMarginTotalModel(
        team_model=team_model,
        market_model=None,
        blend_layer=BlendLayer(margin_model=None, total_model=None),  # type: ignore[arg-type]
        calibrator=WinProbCalibrator(method="elo", model=None),
        target_columns=("away_score", "home_score"),
        market_prob_config=MarketProbConfig(blend_weight=0.2, clamp_delta=0.1),
        xgb_params={"team": {"n_estimators": 77}},
    )

    recipe = recipe_from_model(model, label="B")
    assert recipe.kind == "blend"
    assert recipe.calibration_method == "elo"
    assert recipe.include_market_features is True
    assert recipe.market_anchor is False
    assert recipe.xgb_params["n_estimators"] == 77


def test_bootstrap_overall_metrics_empty_inputs() -> None:
    """Returns empty outputs for empty predictions or non-positive samples."""

    empty = pd.DataFrame()
    assert bootstrap_overall_metrics(empty, n_samples=10, seed=1) == {}
    df = pd.DataFrame(
        {
            "away_score": [10],
            "home_score": [7],
            "predicted_margin": [-3.0],
            "predicted_total": [17.0],
            "home_win_prob": [0.25],
        }
    )
    assert bootstrap_overall_metrics(df, n_samples=0, seed=1) == {}


def test_bootstrap_overall_metrics_produces_percentiles() -> None:
    """Bootstraps basic metrics and produces p05/p50/p95 bands."""

    predictions = pd.DataFrame(
        {
            "away_score": [10, 14, 21, 17],
            "home_score": [13, 10, 24, 20],
            "predicted_margin": [2.5, -3.2, 1.1, 0.0],
            "predicted_total": [24.0, 23.0, 45.0, 37.0],
            "home_win_prob": [0.63, 0.31, 0.58, 0.50],
        }
    )

    ci = bootstrap_overall_metrics(predictions, n_samples=50, seed=123)
    assert ci["n_samples"] == 50
    for key in ("brier", "log_loss", "margin_mae", "total_mae"):
        assert set(ci[key].keys()) == {"p05", "p50", "p95"}
        assert ci[key]["p05"] <= ci[key]["p50"] <= ci[key]["p95"]


def test_load_model_from_dir_or_file(tmp_path: Path) -> None:
    """Loads a joblib model from either a direct file path or a run directory."""

    import joblib

    obj = {"hello": "world", "n": 1}
    model_file = tmp_path / "model.joblib"
    joblib.dump(obj, model_file)

    assert load_model(model_file) == obj
    assert load_model(tmp_path) == obj


def test_run_objective_compare_aligns_by_game_key(monkeypatch: MonkeyPatch) -> None:
    """Aligns predictions by common game keys and records skipped blended folds."""

    from nfl_predictor import ml_model
    from nfl_predictor.ml import model_compare

    base = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [3, 4],
            "game_id": [1, 2],
            "away_abbr": ["NE", "DAL"],
            "home_abbr": ["NYJ", "PHI"],
            "away_pts": [10.0, 14.0],
            "home_pts": [13.0, 7.0],
        }
    )

    monkeypatch.setattr(ml_model, "get_target_columns", lambda df: ("away_pts", "home_pts"))

    fold_1 = walk_forward.WalkForwardFold(
        season=2024,
        week=3,
        train_df=base.iloc[[0]].copy(),
        eval_df=base.iloc[[0]].copy(),
    )
    fold_2 = walk_forward.WalkForwardFold(
        season=2024,
        week=4,
        train_df=base.copy(),
        eval_df=base.iloc[[1]].copy(),
    )

    monkeypatch.setattr(model_compare.walk_forward, "filter_regular_season", lambda df: df)
    monkeypatch.setattr(model_compare.walk_forward, "resolve_eval_seasons", lambda *_: [2024])
    monkeypatch.setattr(
        model_compare.walk_forward,
        "build_walk_forward_folds",
        lambda *_: [fold_1, fold_2],
    )

    def _fake_fit_margin_total_fold(
        fold: walk_forward.WalkForwardFold,
        *,
        recipe: ModelRecipe,
        cfg: CompareConfig,
        target_columns: tuple[str, str],
    ) -> pd.DataFrame:
        df = fold.eval_df.copy()
        df["predicted_margin"] = np.array([1.0])
        df["predicted_total"] = np.array([30.0])
        df["predicted_home_score"] = np.array([15.5])
        df["predicted_away_score"] = np.array([14.5])
        df["home_win_prob"] = np.array([0.6])
        df["away_win_prob"] = 1 - df["home_win_prob"]
        df["calibration_method"] = "none"
        return df

    def _fake_fit_blended_fold(
        fold: walk_forward.WalkForwardFold,
        *,
        recipe: ModelRecipe,
        cfg: CompareConfig,
        target_columns: tuple[str, str],
    ) -> pd.DataFrame:
        if fold.week == 3:
            raise ValueError("no calibration")
        df = fold.eval_df.copy()
        df["predicted_margin"] = np.array([2.0])
        df["predicted_total"] = np.array([31.0])
        df["predicted_home_score"] = np.array([16.5])
        df["predicted_away_score"] = np.array([14.5])
        df["home_win_prob"] = np.array([0.65])
        df["away_win_prob"] = 1 - df["home_win_prob"]
        df["calibration_method"] = "platt"
        df["market_baseline_margin"] = np.array([1.5])
        df["market_baseline_total"] = np.array([30.5])
        return df

    monkeypatch.setattr(model_compare, "_fit_margin_total_fold", _fake_fit_margin_total_fold)
    monkeypatch.setattr(model_compare, "_fit_blended_fold", _fake_fit_blended_fold)

    recipe_a = ModelRecipe(
        label="A",
        kind="margin_total",
        xgb_params={},
        calibration_method="none",
        market_prob_config=MarketProbConfig(blend_weight=0.0, clamp_delta=0.0),
        include_market_features=False,
        market_transform=None,
        market_anchor=False,
    )
    recipe_b = ModelRecipe(
        label="B",
        kind="blend",
        xgb_params={},
        calibration_method="platt",
        market_prob_config=MarketProbConfig(blend_weight=0.0, clamp_delta=0.0),
        include_market_features=True,
        market_transform=True,
        market_anchor=False,
    )

    cfg = CompareConfig(
        eval_seasons=[2024],
        eval_last_n_seasons=1,
        wf_start_week=3,
        include_quantiles=False,
    )
    results = run_objective_compare(
        base,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        cfg=cfg,
        bootstrap_samples=10,
    )

    assert results["skipped_blend_folds"] == 1
    pred_a = results["predictions"]["A"]
    pred_b = results["predictions"]["B"]

    assert list(pred_a["game_id"]) == [2]
    assert list(pred_b["game_id"]) == [2]
    assert "away_score" in pred_a.columns
    assert "home_score" in pred_a.columns
    assert "bootstrap" in results
    assert set(results["bootstrap"].keys()) == {"A", "B"}


def test_write_compare_outputs_writes_all_files(tmp_path: Path) -> None:
    """Writes summary, predictions, and meta json to output directory."""

    pred = pd.DataFrame(
        {
            "season": [2024],
            "week": [3],
            "game_id": [1],
            "away_score": [10.0],
            "home_score": [13.0],
            "predicted_margin": [1.0],
            "predicted_total": [30.0],
            "home_win_prob": [0.6],
        }
    )

    results = {
        "overall": {"A": {"games": 1, "margin_mae": 1.0}, "B": {"games": 1, "margin_mae": 2.0}},
        "predictions": {"A": pred, "B": pred},
        "resolved_eval_seasons": [2024],
        "skipped_blend_folds": 0,
        "bootstrap": {},
    }

    out_dir = tmp_path / "out"
    write_compare_outputs(out_dir, results)

    assert (out_dir / "objective_compare_summary.csv").exists()
    assert (out_dir / "objective_compare_predictions_A.csv").exists()
    assert (out_dir / "objective_compare_predictions_B.csv").exists()
    assert (out_dir / "objective_compare_meta.json").exists()


def test_fit_margin_total_fold_smoke_with_stubs(monkeypatch: MonkeyPatch) -> None:
    """Runs the margin/total fold path with stubbed training and calibration."""

    from nfl_predictor import ml_model
    from nfl_predictor.ml import model_compare

    class _DummyPreprocessor:
        def fit_transform(self, x: pd.DataFrame) -> np.ndarray:  # noqa: ANN001
            return np.ones((len(x), 2), dtype=float)

        def transform(self, x: pd.DataFrame) -> np.ndarray:  # noqa: ANN001
            return np.ones((len(x), 2), dtype=float)

    class _DummyModel:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(
        model_compare,
        "_resolve_market_settings_for_recipe",
        lambda *_: (True, False, True),
    )
    monkeypatch.setattr(ml_model, "_build_feature_spec", lambda *_, **__: object())
    monkeypatch.setattr(ml_model, "_build_preprocessor", lambda *_, **__: _DummyPreprocessor())
    monkeypatch.setattr(ml_model, "apply_feature_spec", lambda df, spec: df)

    def _prep_targets(df: pd.DataFrame, *_args: object, **_kwargs: object):
        n = len(df)
        y_margin = np.zeros(n, dtype=float)
        y_total = np.zeros(n, dtype=float)
        baseline_margin = np.ones(n, dtype=float)
        baseline_total = np.ones(n, dtype=float) * 40.0
        return y_margin, y_total, baseline_margin, baseline_total

    monkeypatch.setattr(ml_model, "_prepare_margin_total_targets_with_anchor", _prep_targets)
    monkeypatch.setattr(
        model_compare.walk_forward,
        "select_calibration_data",
        lambda *_: pd.DataFrame({"away_pts": [10.0], "home_pts": [13.0]}),
    )
    monkeypatch.setattr(
        ml_model,
        "_fit_margin_total_models",
        lambda *_args, **_kwargs: (_DummyModel("margin"), _DummyModel("total")),
    )

    def _predict_xgb(model: _DummyModel, x: np.ndarray) -> np.ndarray:  # noqa: ANN001
        if model.name == "margin":
            return np.full(len(x), 3.0)
        return np.full(len(x), 44.0)

    monkeypatch.setattr(ml_model, "_predict_xgb", _predict_xgb)
    monkeypatch.setattr(
        ml_model,
        "get_market_baseline",
        lambda df: (np.zeros(len(df)), np.zeros(len(df))),
    )
    monkeypatch.setattr(
        ml_model,
        "derive_scores_from_margin_total",
        lambda margin, total: ((total - margin) / 2.0, (total + margin) / 2.0),
    )
    monkeypatch.setattr(ml_model, "predict_home_win_prob", lambda *_: np.array([0.7]))
    monkeypatch.setattr(ml_model, "adjust_home_win_prob", lambda _df, probs, _cfg: probs)
    monkeypatch.setattr(model_compare.walk_forward, "_fit_calibrator", lambda *_: object())

    fold = walk_forward.WalkForwardFold(
        season=2024,
        week=4,
        train_df=pd.DataFrame({"away_pts": [10.0], "home_pts": [13.0]}),
        eval_df=pd.DataFrame({"away_pts": [14.0], "home_pts": [17.0]}),
    )
    recipe = ModelRecipe(
        label="X",
        kind="margin_total",
        xgb_params={"n_estimators": 5},
        calibration_method="platt",
        market_prob_config=MarketProbConfig(blend_weight=0.0, clamp_delta=0.0),
        include_market_features=True,
        market_transform=None,
        market_anchor=True,
    )
    cfg = CompareConfig(include_quantiles=False)

    out = model_compare._fit_margin_total_fold(
        fold,
        recipe=recipe,
        cfg=cfg,
        target_columns=("away_pts", "home_pts"),
    )

    assert out["calibration_method"].iloc[0] in {"platt", "none"}
    assert "predicted_margin" in out.columns
    assert "predicted_total" in out.columns


def test_fit_blended_fold_smoke_with_stubs(monkeypatch: MonkeyPatch) -> None:
    """Runs the blended fold path with stubbed team models and blenders."""

    from nfl_predictor.ml import ml_model_core as core
    from nfl_predictor.ml import model_compare

    class _DummyPreprocessor:
        def fit_transform(self, x: pd.DataFrame) -> np.ndarray:  # noqa: ANN001
            return np.ones((len(x), 2), dtype=float)

        def transform(self, x: pd.DataFrame) -> np.ndarray:  # noqa: ANN001
            return np.ones((len(x), 2), dtype=float)

    class _DummyModel:
        def __init__(self, name: str) -> None:
            self.name = name

    class _DummyBlender:
        def predict(self, x: np.ndarray) -> np.ndarray:  # noqa: ANN001
            return x.mean(axis=1)

    monkeypatch.setattr(
        model_compare,
        "_resolve_market_settings_for_recipe",
        lambda *_: (True, True, False),
    )
    monkeypatch.setattr(
        model_compare.walk_forward,
        "select_calibration_data",
        lambda *_: pd.DataFrame({"away_pts": [10.0], "home_pts": [13.0]}),
    )
    monkeypatch.setattr(core, "_build_feature_spec", lambda *_, **__: object())
    monkeypatch.setattr(core, "_build_preprocessor", lambda *_, **__: _DummyPreprocessor())
    monkeypatch.setattr(core, "_apply_feature_spec", lambda df, spec: df)
    monkeypatch.setattr(
        core,
        "_prepare_margin_total_targets",
        lambda df, *_: (np.zeros(len(df)), np.zeros(len(df))),
    )
    monkeypatch.setattr(
        core,
        "_fit_margin_total_models",
        lambda *_args, **_kwargs: (_DummyModel("margin"), _DummyModel("total")),
    )

    def _predict_xgb(model: _DummyModel, x: np.ndarray) -> np.ndarray:  # noqa: ANN001
        if model.name == "margin":
            return np.full(len(x), 4.0)
        return np.full(len(x), 46.0)

    monkeypatch.setattr(core, "_predict_xgb", _predict_xgb)
    monkeypatch.setattr(
        core,
        "get_market_baseline",
        lambda df: (np.ones(len(df)), np.ones(len(df)) * 41.0),
    )
    monkeypatch.setattr(
        core,
        "_fit_blend_ridge_constrained",
        lambda *_args, **_kwargs: _DummyBlender(),
    )
    monkeypatch.setattr(
        core,
        "derive_scores_from_margin_total",
        lambda m, t: ((t - m) / 2.0, (t + m) / 2.0),
    )
    monkeypatch.setattr(core, "_predict_home_win_prob", lambda *_: np.array([0.66]))
    monkeypatch.setattr(core, "_adjust_home_win_prob", lambda _df, probs, _cfg: probs)
    monkeypatch.setattr(core, "_fit_win_prob_calibrator", lambda *_: object())

    fold = walk_forward.WalkForwardFold(
        season=2024,
        week=4,
        train_df=pd.DataFrame({"away_pts": [10.0], "home_pts": [13.0]}),
        eval_df=pd.DataFrame({"away_pts": [14.0], "home_pts": [17.0]}),
    )
    recipe = ModelRecipe(
        label="Y",
        kind="blend",
        xgb_params={"n_estimators": 5},
        calibration_method="platt",
        market_prob_config=MarketProbConfig(blend_weight=0.0, clamp_delta=0.0),
        include_market_features=True,
        market_transform=True,
        market_anchor=False,
    )
    cfg = CompareConfig(include_quantiles=False)

    out = model_compare._fit_blended_fold(
        fold,
        recipe=recipe,
        cfg=cfg,
        target_columns=("away_pts", "home_pts"),
    )

    assert "market_baseline_margin" in out.columns
    assert "market_baseline_total" in out.columns
    assert out["calibration_method"].iloc[0] in {"platt", "none"}
