"""Tests for feature importance reporting utilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.compose import ColumnTransformer

from nfl_predictor.ml import feature_importance, ml_model_core


def _as_preprocessor(value: object) -> ColumnTransformer:
    """Cast a test double to the preprocessor type expected by helper signatures."""
    return cast(ColumnTransformer, value)


def _as_regressor(value: object) -> xgb.XGBRegressor:
    """Cast a test double to the regressor type expected by helper signatures."""
    return cast(xgb.XGBRegressor, value)


def test_coerce_score_value_sums_sequence_inputs() -> None:
    """Sequence-based score values are summed before conversion to float."""
    assert feature_importance._coerce_score_value([1.25, 2.75]) == 4.0


def test_feature_importance_report_margin_total() -> None:
    """Feature importance report returns aligned gain/weight arrays."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=40),
            "f2": rng.normal(size=40),
        }
    )
    spec = ml_model_core.FeatureSpec(
        feature_columns=["f1", "f2"],
        categorical_columns=[],
        numeric_columns=["f1", "f2"],
        dropped_columns=[],
        id_columns=[],
        constant_columns=[],
        high_cardinality_columns=[],
        feature_start="f1",
        feature_end="f2",
        metadata_columns=[],
        post_feature_columns=[],
        market_columns=[],
    )
    preprocessor = ml_model_core._build_preprocessor(spec, for_tree=True)
    x_matrix = ml_model_core._fit_transform_matrix(preprocessor, df)
    y_margin = rng.normal(size=40)
    y_total = rng.normal(size=40)
    params = ml_model_core._resolve_xgb_params(
        ml_model_core.DEFAULT_XGB_PARAMS,
        overrides={
            "n_estimators": 15,
            "max_depth": 2,
            "learning_rate": 0.1,
            "verbosity": 0,
            "n_jobs": 1,
        },
    )
    margin_model, total_model = ml_model_core._fit_margin_total_models(
        x_matrix,
        y_margin,
        y_total,
        params,
    )
    model = ml_model_core.MarginTotalModel(
        preprocessor=preprocessor,
        feature_spec=spec,
        margin_model=margin_model,
        total_model=total_model,
        target_columns=("away_score", "home_score"),
        calibrator=None,
    )

    report = feature_importance.build_feature_importance_report(model)
    assert report is not None
    assert report["model_kind"] == "margin_total"
    assert "feature_names" in report
    assert "models" in report
    assert set(report["models"].keys()) == {"margin", "total"}
    feature_names = report["feature_names"]
    for key in ("margin", "total"):
        assert len(report["models"][key]["gain"]) == len(feature_names)
        assert len(report["models"][key]["weight"]) == len(feature_names)


def test_resolve_feature_names_falls_back_on_errors_and_mismatches() -> None:
    """Feature-name resolution should fall back cleanly when names are unavailable or mismatched."""

    class _FailingPreprocessor:
        """Preprocessor whose feature-name resolution fails."""

        def get_feature_names_out(self) -> list[str]:
            """Raise to exercise the fallback path."""
            raise RuntimeError("boom")

    class _Model:
        """Minimal model exposing only n_features_in_."""

        def __init__(self, n_features_in_: int | None) -> None:
            """Store the synthetic feature count."""
            self.n_features_in_ = n_features_in_

    assert feature_importance.resolve_feature_names(
        _as_preprocessor(_FailingPreprocessor()),
        _as_regressor(_Model(3)),
    ) == [
        "f0",
        "f1",
        "f2",
    ]

    mismatch_preprocessor = SimpleNamespace(get_feature_names_out=lambda: ["only_one"])
    assert feature_importance.resolve_feature_names(
        _as_preprocessor(mismatch_preprocessor),
        _as_regressor(_Model(2)),
    ) == [
        "f0",
        "f1",
    ]

    no_feature_count_preprocessor = SimpleNamespace(get_feature_names_out=lambda: ["a", "b"])
    assert feature_importance.resolve_feature_names(
        _as_preprocessor(no_feature_count_preprocessor),
        _as_regressor(_Model(None)),
    ) == [
        "a",
        "b",
    ]

    assert (
        feature_importance.resolve_feature_names(
            _as_preprocessor(SimpleNamespace()),
            _as_regressor(_Model(None)),
        )
        == []
    )


def test_build_feature_importance_report_dispatches_supported_model_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level report building should dispatch across supported model families."""

    class DummyBlend:
        """Synthetic blended model class for dispatch testing."""

        def __init__(self, team_model: object) -> None:
            """Store the team model for the test."""
            self.team_model = team_model

    class DummyMargin:
        """Synthetic margin/total model class for dispatch testing."""

        pass

    monkeypatch.setattr(feature_importance, "BlendedMarginTotalModel", DummyBlend)
    monkeypatch.setattr(feature_importance, "MarginTotalModel", DummyMargin)

    def fake_margin_report(model: object, _shap_rows: object = None) -> dict[str, object] | None:
        """Return a synthetic report keyed off the input object."""
        if model == "team":
            return {"team": True}
        return {"margin": True}

    monkeypatch.setattr(feature_importance, "_build_margin_total_report", fake_margin_report)

    blend_report = feature_importance.build_feature_importance_report(DummyBlend("team"))
    assert blend_report == {"model_kind": "blend", "components": {"team": {"team": True}}}

    margin_report = feature_importance.build_feature_importance_report(DummyMargin())
    assert margin_report == {"margin": True, "model_kind": "margin_total"}

    assert feature_importance.build_feature_importance_report(object()) is None


def test_build_feature_importance_report_handles_attribute_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attribute errors from report building should be swallowed into a skipped report."""

    class DummyMargin:
        """Synthetic margin/total model class for error-path testing."""

        pass

    monkeypatch.setattr(feature_importance, "MarginTotalModel", DummyMargin)
    monkeypatch.setattr(feature_importance, "BlendedMarginTotalModel", tuple)

    def _raise_attribute_error(
        _model: object, _shap_rows: object = None
    ) -> dict[str, object] | None:
        """Raise an AttributeError to exercise the guarded path."""
        raise AttributeError("missing booster")

    monkeypatch.setattr(feature_importance, "_build_margin_total_report", _raise_attribute_error)
    monkeypatch.setattr(feature_importance.log, "debug", lambda *_args, **_kwargs: None)

    assert feature_importance.build_feature_importance_report(DummyMargin()) is None


def test_build_report_from_models_handles_empty_unsupported_and_no_base_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report building should handle empty models, unsupported models, and missing base features."""
    monkeypatch.setattr(feature_importance.log, "debug", lambda *_args, **_kwargs: None)
    assert (
        feature_importance._build_report_from_models(_as_preprocessor(SimpleNamespace()), {})
        is None
    )
    assert (
        feature_importance._build_report_from_models(
            _as_preprocessor(SimpleNamespace()),
            cast(dict[str, xgb.XGBRegressor], {"margin": object()}),
        )
        is None
    )

    booster = SimpleNamespace(get_score=lambda importance_type: {"feat": 1.0})
    model = SimpleNamespace(get_booster=lambda: booster, n_features_in_=1)
    preprocessor = SimpleNamespace(get_feature_names_out=lambda: ["feat"])
    monkeypatch.setattr(feature_importance, "_build_base_features", lambda *_args: None)

    report = feature_importance._build_report_from_models(
        _as_preprocessor(preprocessor),
        cast(dict[str, xgb.XGBRegressor], {"margin": model}),
    )
    assert report == {
        "schema_version": feature_importance.SCHEMA_VERSION,
        "measures": feature_importance.MEASURES,
        "feature_names": ["feat"],
        "models": {"margin": {"gain": [1.0], "total_gain": [1.0], "weight": [1.0]}},
    }


def test_score_dict_to_list_and_model_importance_cover_fallback_keys() -> None:
    """Importance helpers should align named and XGBoost fallback keys into ordered lists."""
    scores = {"known": 1.5, "f1": [2.0, 3.0], "f9": 99.0}
    assert feature_importance._score_dict_to_list(scores, ["known", "other"]) == [1.5, 5.0]
    assert feature_importance._coerce_score_value(2.5) == 2.5
    assert feature_importance._models_support_importance(
        [SimpleNamespace(get_booster=lambda: None)]
    )
    assert not feature_importance._models_support_importance([object()])

    scores_by_type = {"gain": {"known": 1.0}, "total_gain": {"known": 6.0}, "weight": {"f0": 2.0}}
    booster = SimpleNamespace(get_score=lambda importance_type: scores_by_type[importance_type])
    model = SimpleNamespace(get_booster=lambda: booster)

    assert feature_importance._build_model_importance(_as_regressor(model), ["known"]) == {
        "gain": [1.0],
        "total_gain": [6.0],
        "weight": [2.0],
    }


def test_build_base_features_and_aggregate_without_combined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Base-feature aggregation should work for single-model reports without a combined section."""
    monkeypatch.setattr(
        feature_importance,
        "_build_base_feature_map",
        lambda *_args: {"num__yards": "yards", "cat__team_A": "team"},
    )

    base_features = feature_importance._build_base_features(
        _as_preprocessor(SimpleNamespace()),
        ["num__yards", "cat__team_A"],
        {"margin": {"gain": [1.5, 2.0], "total_gain": [4.5, 8.0], "weight": [3.0, 4.0]}},
    )

    assert base_features == {
        "feature_names": ["team", "yards"],
        "margin": {"total_gain": [8.0, 4.5], "weight": [4.0, 3.0]},
    }


def test_build_base_feature_map_handles_guard_paths_and_success() -> None:
    """Base-feature mapping should return None for unsupported preprocessors and map valid ones."""
    assert (
        feature_importance._build_base_feature_map(
            _as_preprocessor(SimpleNamespace()),
            ["feat"],
        )
        is None
    )
    assert (
        feature_importance._build_base_feature_map(
            _as_preprocessor(SimpleNamespace(transformers_=[])),
            ["feat"],
        )
        is None
    )

    failing_preprocessor = SimpleNamespace(
        transformers_=[],
        get_feature_names_out=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert (
        feature_importance._build_base_feature_map(
            _as_preprocessor(failing_preprocessor),
            ["feat"],
        )
        is None
    )

    mismatch_preprocessor = SimpleNamespace(
        transformers_=[],
        get_feature_names_out=lambda: ["other"],
    )
    assert (
        feature_importance._build_base_feature_map(
            _as_preprocessor(mismatch_preprocessor),
            ["feat"],
        )
        is None
    )

    onehot = SimpleNamespace(categories_=[["A", "B"]])
    valid_preprocessor = SimpleNamespace(
        feature_names_in_=np.array(["yards", "team", "week", "unused"], dtype=object),
        transformers_=[
            ("remainder", "drop", []),
            ("dropper", "drop", ["unused"]),
            ("num", "passthrough", ["yards"]),
            ("cat", SimpleNamespace(named_steps={"onehot": onehot}), ["team"]),
            ("scalar", SimpleNamespace(named_steps={}), "week"),
        ],
        get_feature_names_out=lambda: ["num__yards", "cat__team_A", "cat__team_B", "scalar__week"],
    )

    assert feature_importance._build_base_feature_map(
        _as_preprocessor(valid_preprocessor),
        ["num__yards", "cat__team_A", "cat__team_B", "scalar__week"],
    ) == {
        "num__yards": "yards",
        "cat__team_A": "team",
        "cat__team_B": "team",
        "scalar__week": "week",
    }


def test_normalize_cols_handles_slice_list_and_scalar_inputs() -> None:
    """Column selector normalization should support slices, sequences, and scalar values."""
    preprocessor = SimpleNamespace(feature_names_in_=np.array(["a", "b", "c"], dtype=object))

    assert feature_importance._normalize_cols(slice(0, 2), _as_preprocessor(preprocessor)) == [
        "a",
        "b",
    ]
    assert feature_importance._normalize_cols(
        ["x", "y"],
        _as_preprocessor(preprocessor),
    ) == ["x", "y"]
    assert feature_importance._normalize_cols(
        np.array(["x", "y"]),
        _as_preprocessor(preprocessor),
    ) == ["x", "y"]
    assert feature_importance._normalize_cols("week", _as_preprocessor(preprocessor)) == ["week"]
    assert (
        feature_importance._normalize_cols(
            slice(0, 1),
            _as_preprocessor(SimpleNamespace()),
        )
        == []
    )


def test_base_features_rank_by_total_gain_not_summed_average_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A many-category feature split rarely ranks below a numeric feature the trees use often.

    Summing each one-hot column's average gain would put the categorical feature first
    (3 x 5.0 = 15.0 against 4.0); total gain (average gain times splits) does not.
    """
    columns = ["num__rest", "cat__opp_A", "cat__opp_B", "cat__opp_C"]
    monkeypatch.setattr(
        feature_importance,
        "_build_base_feature_map",
        lambda *_args: {
            "num__rest": "rest",
            "cat__opp_A": "opp",
            "cat__opp_B": "opp",
            "cat__opp_C": "opp",
        },
    )
    head = {
        "gain": [4.0, 5.0, 5.0, 5.0],
        "weight": [30.0, 1.0, 1.0, 1.0],
        "total_gain": [120.0, 5.0, 5.0, 5.0],
    }

    base = feature_importance._build_base_features(
        _as_preprocessor(SimpleNamespace()),
        columns,
        {"margin": head, "total": head},
    )

    assert base is not None
    assert base["feature_names"] == ["opp", "rest"]
    assert base["margin"] == {"total_gain": [15.0, 120.0], "weight": [3.0, 30.0]}
    assert base["combined"] == {"total_gain": [30.0, 240.0], "weight": [6.0, 60.0]}
    assert "gain" not in base["combined"]


def _fit_rest_opp_model(rows: int = 120) -> tuple[ml_model_core.MarginTotalModel, pd.DataFrame]:
    """Fit a small margin/total model on a numeric feature and a four-category one-hot feature."""
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {
            "rest": rng.normal(size=rows),
            "opp": rng.choice(["A", "B", "C", "D"], size=rows),
        }
    )
    spec = ml_model_core.FeatureSpec(
        feature_columns=["rest", "opp"],
        categorical_columns=["opp"],
        numeric_columns=["rest"],
        dropped_columns=[],
        id_columns=[],
        constant_columns=[],
        high_cardinality_columns=[],
        feature_start="rest",
        feature_end="opp",
        metadata_columns=[],
        post_feature_columns=[],
        market_columns=[],
    )
    preprocessor = ml_model_core._build_preprocessor(spec, for_tree=True)
    x_matrix = ml_model_core._fit_transform_matrix(preprocessor, df)
    opp_effect = df["opp"].map({"A": 2.0, "B": -1.0, "C": 0.5, "D": -1.5}).to_numpy()
    y_margin = 3.0 * df["rest"].to_numpy() + opp_effect + rng.normal(scale=0.1, size=rows)
    y_total = df["rest"].to_numpy() - opp_effect + rng.normal(scale=0.1, size=rows)
    params = ml_model_core._resolve_xgb_params(
        ml_model_core.DEFAULT_XGB_PARAMS,
        overrides={
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.3,
            "verbosity": 0,
            "n_jobs": 1,
        },
    )
    margin_model, total_model = ml_model_core._fit_margin_total_models(
        x_matrix, y_margin, y_total, params
    )
    model = ml_model_core.MarginTotalModel(
        preprocessor=preprocessor,
        feature_spec=spec,
        margin_model=margin_model,
        total_model=total_model,
        target_columns=("away_score", "home_score"),
        calibrator=None,
    )
    return model, df


def test_fitted_report_sums_total_gain_over_one_hot_columns_and_heads() -> None:
    """Base-feature total gain is the booster's per-column total gain summed per base feature."""
    model, _ = _fit_rest_opp_model()
    margin_model, total_model = model.margin_model, model.total_model

    report = feature_importance.build_feature_importance_report(model)

    assert report is not None
    assert report["schema_version"] == feature_importance.SCHEMA_VERSION
    assert set(report["measures"]) == {"gain", "total_gain", "weight"}
    assert "shap" not in report
    assert "mean_abs_shap" not in report["base_features"]["combined"]
    names = report["feature_names"]
    base = report["base_features"]
    assert base["feature_names"] == ["opp", "rest"]
    assert sum("opp" in name for name in names) == 4
    expected: dict[str, list[float]] = {}
    for head, regressor in (("margin", margin_model), ("total", total_model)):
        scores = regressor.get_booster().get_score(importance_type="total_gain")
        per_column: list[float] = []
        for i, name in enumerate(names):
            value = scores.get(name, scores.get(f"f{i}", 0.0))
            assert isinstance(value, float)
            per_column.append(value)
        assert report["models"][head]["total_gain"] == pytest.approx(per_column)
        opp_total = sum(v for n, v in zip(names, per_column, strict=True) if "opp" in n)
        rest_total = sum(v for n, v in zip(names, per_column, strict=True) if "rest" in n)
        expected[head] = [opp_total, rest_total]
        assert opp_total > 0.0
        assert base[head]["total_gain"] == pytest.approx(expected[head])
    assert base["combined"]["total_gain"] == pytest.approx(
        [expected["margin"][i] + expected["total"][i] for i in range(2)]
    )


def _preprocessed(model: ml_model_core.MarginTotalModel, df: pd.DataFrame) -> Any:
    """Return ``df`` encoded by the model's own feature spec and preprocessor."""
    return ml_model_core._transform_matrix(
        model.preprocessor, ml_model_core._apply_feature_spec(df, model.feature_spec)
    )


def _early_stopped_head(x_matrix: Any, rows: int) -> xgb.XGBRegressor:
    """Fit a head on noise with early stopping, so it keeps fewer trees than it grew."""
    rng = np.random.default_rng(5)
    target = rng.normal(size=rows)
    head = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.3,
        early_stopping_rounds=5,
        n_jobs=1,
        verbosity=0,
    )
    head.fit(x_matrix[:80], target[:80], eval_set=[(x_matrix[80:], target[80:])], verbose=False)
    assert head.best_iteration < 199
    return head


@pytest.mark.parametrize("early_stopped", [False, True])
def test_shap_values_add_up_to_each_rows_prediction(early_stopped: bool) -> None:
    """Each row's SHAP values plus the base value reproduce the head's prediction.

    An early-stopped head is explained over the trees it predicts with, up to its best
    iteration, not over every tree it grew.
    """
    model, df = _fit_rest_opp_model()
    x_matrix = _preprocessed(model, df)
    heads = (
        (_early_stopped_head(x_matrix, len(df)),)
        if early_stopped
        else (model.margin_model, model.total_model)
    )

    for head in heads:
        values, base_value = feature_importance.shap_values(head, x_matrix)

        assert values.shape == (len(df), 5)
        np.testing.assert_allclose(
            values.sum(axis=1) + base_value,
            ml_model_core._predict_xgb(head, x_matrix),
            rtol=1e-5,
            atol=1e-5,
        )


def test_mean_abs_shap_sums_a_base_features_columns_before_the_absolute_value() -> None:
    """One-hot columns' signed contributions cancel within a row before the absolute value.

    Row 1 gives ``opp`` +1 and -1 (net 0), row 2 gives +2: the mean is 1, where averaging each
    column's absolute value and summing would report 2.
    """
    values = np.array([[1.0, -1.0, 0.5], [2.0, 0.0, -1.5]])

    result = feature_importance.mean_abs_shap_by_base(
        values,
        ["cat__opp_A", "cat__opp_B", "num__rest"],
        {"cat__opp_A": "opp", "cat__opp_B": "opp", "num__rest": "rest"},
    )

    assert result == {"opp": pytest.approx(1.0), "rest": pytest.approx(1.0)}


def test_fitted_report_records_mean_abs_shap_per_head_and_combined() -> None:
    """With rows to explain, each head and the combined block carry mean |SHAP| per base feature."""
    model, df = _fit_rest_opp_model()
    x_matrix = _preprocessed(model, df)

    report = feature_importance.build_feature_importance_report(model, shap_rows=df)

    assert report is not None
    assert report["schema_version"] == 3
    assert "mean_abs_shap" in report["measures"]
    assert report["shap"]["row_count"] == len(df)
    names = report["feature_names"]
    base = report["base_features"]
    assert base["feature_names"] == ["opp", "rest"]
    expected: dict[str, list[float]] = {}
    for head, regressor in (("margin", model.margin_model), ("total", model.total_model)):
        contribs = regressor.get_booster().predict(xgb.DMatrix(x_matrix), pred_contribs=True)
        opp = contribs[:, [i for i, n in enumerate(names) if "opp" in n]].sum(axis=1)
        rest = contribs[:, [i for i, n in enumerate(names) if "rest" in n]].sum(axis=1)
        expected[head] = [float(np.abs(opp).mean()), float(np.abs(rest).mean())]
        assert base[head]["mean_abs_shap"] == pytest.approx(expected[head], rel=1e-5)
        assert report["models"][head]["mean_abs_shap"] == pytest.approx(
            np.abs(contribs[:, :-1]).mean(axis=0).tolist(), rel=1e-5
        )
    assert base["combined"]["mean_abs_shap"] == pytest.approx(
        [expected["margin"][i] + expected["total"][i] for i in range(2)], rel=1e-5
    )


def test_report_records_the_explained_rows_seasons() -> None:
    """The SHAP block names the seasons of the rows it explains when they carry a season."""
    model, df = _fit_rest_opp_model()
    rows = df.assign(season=[2020] * 60 + [2022] * 60)

    report = feature_importance.build_feature_importance_report(model, shap_rows=rows)

    assert report is not None
    assert report["shap"]["seasons"] == [2020, 2022]


def test_empty_shap_rows_leave_the_report_without_shap() -> None:
    """No rows to explain means no SHAP section, and the gain measures still load."""
    model, df = _fit_rest_opp_model()

    report = feature_importance.build_feature_importance_report(model, shap_rows=df.iloc[:0])

    assert report is not None
    assert "shap" not in report
    assert "mean_abs_shap" not in report["base_features"]["combined"]
    assert "total_gain" in report["base_features"]["combined"]
