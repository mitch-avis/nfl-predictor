"""Tests for feature importance reporting utilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from nfl_predictor.ml import feature_importance, ml_model_core

if TYPE_CHECKING:
    from sklearn.compose import ColumnTransformer


def _as_preprocessor(value: object) -> ColumnTransformer:
    """Cast a test double to the preprocessor type expected by helper signatures."""
    return cast("ColumnTransformer", value)


def _as_regressor(value: object) -> xgb.XGBRegressor:
    """Cast a test double to the regressor type expected by helper signatures."""
    return cast("xgb.XGBRegressor", value)


def test_coerce_score_value_sums_sequence_inputs() -> None:
    """Sequence-based score values are summed before conversion to float."""
    # Act
    value = feature_importance._coerce_score_value([1.25, 2.75])

    # Assert
    assert value == 4.0


def test_feature_importance_report_margin_total() -> None:
    """Feature importance report returns aligned gain/weight arrays."""
    # Arrange
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
    preprocessor = ml_model_core.build_preprocessor(spec, for_tree=True)
    x_matrix = ml_model_core.fit_transform_matrix(preprocessor, df)
    y_margin = rng.normal(size=40)
    y_total = rng.normal(size=40)
    params = ml_model_core.resolve_xgb_params(
        ml_model_core.DEFAULT_XGB_PARAMS,
        overrides={
            "n_estimators": 15,
            "max_depth": 2,
            "learning_rate": 0.1,
            "verbosity": 0,
            "n_jobs": 1,
        },
    )
    margin_model, total_model = ml_model_core.fit_margin_total_models(
        ml_model_core.FitData(x_matrix, y_margin, y_total), params
    )
    model = ml_model_core.MarginTotalModel(
        preprocessor=preprocessor,
        feature_spec=spec,
        margin_model=margin_model,
        total_model=total_model,
        target_columns=("away_score", "home_score"),
    )

    # Act
    report = feature_importance.build_feature_importance_report(model)

    # Assert
    assert report is not None
    assert report["model_kind"] == "margin_total"
    assert "feature_names" in report
    assert "models" in report
    assert set(report["models"].keys()) == {"margin", "total"}
    feature_names = report["feature_names"]
    for key in ("margin", "total"):
        assert len(report["models"][key]["gain"]) == len(feature_names)
        assert len(report["models"][key]["weight"]) == len(feature_names)


class _FailingPreprocessor:
    """Preprocessor whose feature-name resolution fails."""

    def get_feature_names_out(self) -> list[str]:
        """Raise to exercise the fallback path."""
        msg = "boom"
        raise AttributeError(msg)


class _Model:
    """Minimal model exposing only n_features_in_."""

    def __init__(self, n_features_in_: int | None) -> None:
        """Store the synthetic feature count."""
        self.n_features_in_ = n_features_in_


@pytest.mark.parametrize(
    ("preprocessor", "n_features_in", "expected"),
    [
        pytest.param(_FailingPreprocessor(), 3, ["f0", "f1", "f2"], id="names_fail"),
        pytest.param(
            SimpleNamespace(get_feature_names_out=lambda: ["only_one"]),
            2,
            ["f0", "f1"],
            id="names_mismatch",
        ),
        pytest.param(
            SimpleNamespace(get_feature_names_out=lambda: ["a", "b"]),
            None,
            ["a", "b"],
            id="no_feature_count",
        ),
        pytest.param(SimpleNamespace(), None, [], id="nothing_available"),
    ],
)
def test_resolve_feature_names_falls_back_on_errors_and_mismatches(
    preprocessor: object, n_features_in: int | None, expected: list[str]
) -> None:
    """Feature-name resolution should fall back cleanly when names are unavailable or mismatched."""
    # Act
    names = feature_importance.resolve_feature_names(
        _as_preprocessor(preprocessor),
        _as_regressor(_Model(n_features_in)),
    )

    # Assert
    assert names == expected


class _DispatchMargin:
    """Synthetic margin/total model class for dispatch testing."""


def _patch_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap in a synthetic margin/total class and a report builder keyed off its input."""
    monkeypatch.setattr(feature_importance, "MarginTotalModel", _DispatchMargin)

    def fake_margin_report(model: object, _shap_rows: object = None) -> dict[str, object] | None:
        """Return a synthetic report keyed off the input object."""
        if model == "team":
            return {"team": True}
        return {"margin": True}

    monkeypatch.setattr(feature_importance, "_build_margin_total_report", fake_margin_report)


def test_build_feature_importance_report_dispatches_supported_model_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only margin/total models get a report; a saved blend model of the retired kind gets none."""
    # Arrange
    _patch_dispatch(monkeypatch)
    blend = ml_model_core.BlendedMarginTotalModel(
        team_model=cast("ml_model_core.MarginTotalModel", "team"),
        blend_layer=cast("ml_model_core.BlendLayer", None),
        calibrator=None,
        target_columns=("away_score", "home_score"),
    )

    # Act
    report = feature_importance.build_feature_importance_report(blend)

    # Assert
    assert report is None


def test_build_feature_importance_report_reports_a_margin_total_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A margin/total model gets its report tagged with the model kind."""
    # Arrange
    _patch_dispatch(monkeypatch)

    # Act
    margin_report = feature_importance.build_feature_importance_report(_DispatchMargin())

    # Assert
    assert margin_report == {"margin": True, "model_kind": "margin_total"}


def test_build_feature_importance_report_skips_an_unsupported_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An object that is not a model gets no report."""
    # Arrange
    _patch_dispatch(monkeypatch)

    # Act
    report = feature_importance.build_feature_importance_report(object())

    # Assert
    assert report is None


def test_build_feature_importance_report_handles_attribute_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attribute errors from report building should be swallowed into a skipped report."""

    # Arrange
    class DummyMargin:
        """Synthetic margin/total model class for error-path testing."""

    monkeypatch.setattr(feature_importance, "MarginTotalModel", DummyMargin)

    def _raise_attribute_error(
        _model: object, _shap_rows: object = None
    ) -> dict[str, object] | None:
        """Raise an AttributeError to exercise the guarded path."""
        msg = "missing booster"
        raise AttributeError(msg)

    monkeypatch.setattr(feature_importance, "_build_margin_total_report", _raise_attribute_error)
    monkeypatch.setattr(feature_importance.log, "debug", lambda *_args, **_kwargs: None)

    # Act
    report = feature_importance.build_feature_importance_report(DummyMargin())

    # Assert
    assert report is None


def _silence_debug_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the module's debug log calls."""
    monkeypatch.setattr(feature_importance.log, "debug", lambda *_args, **_kwargs: None)


@pytest.mark.parametrize(
    "models",
    [
        pytest.param({}, id="empty"),
        pytest.param({"margin": object()}, id="unsupported"),
    ],
)
def test_build_report_from_models_handles_empty_and_unsupported(
    monkeypatch: pytest.MonkeyPatch, models: dict[str, object]
) -> None:
    """Report building should return no report for empty or unsupported models."""
    # Arrange
    _silence_debug_log(monkeypatch)

    # Act
    report = feature_importance._build_report_from_models(
        _as_preprocessor(SimpleNamespace()),
        cast("dict[str, xgb.XGBRegressor]", models),
    )

    # Assert
    assert report is None


def test_build_report_from_models_handles_no_base_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report building should still report the models when base features are missing."""
    # Arrange
    _silence_debug_log(monkeypatch)
    booster = SimpleNamespace(get_score=lambda importance_type: {"feat": 1.0})
    model = SimpleNamespace(get_booster=lambda: booster, n_features_in_=1)
    preprocessor = SimpleNamespace(get_feature_names_out=lambda: ["feat"])
    monkeypatch.setattr(feature_importance, "_build_base_features", lambda *_args: None)

    # Act
    report = feature_importance._build_report_from_models(
        _as_preprocessor(preprocessor),
        cast("dict[str, xgb.XGBRegressor]", {"margin": model}),
    )

    # Assert
    assert report == {
        "schema_version": feature_importance.SCHEMA_VERSION,
        "measures": feature_importance.MEASURES,
        "feature_names": ["feat"],
        "models": {"margin": {"gain": [1.0], "total_gain": [1.0], "weight": [1.0]}},
    }


def test_score_dict_to_list_covers_fallback_keys() -> None:
    """Importance helpers should align named and XGBoost fallback keys into ordered lists."""
    # Arrange
    scores = {"known": 1.5, "f1": [2.0, 3.0], "f9": 99.0}

    # Act
    values = feature_importance._score_dict_to_list(scores, ["known", "other"])

    # Assert
    assert values == [1.5, 5.0]


def test_coerce_score_value_keeps_a_scalar() -> None:
    """A scalar score value converts to the same float."""
    # Act
    value = feature_importance._coerce_score_value(2.5)

    # Assert
    assert value == 2.5


def test_models_support_importance_with_a_booster() -> None:
    """A model exposing ``get_booster`` supports importance."""
    # Act
    supported = feature_importance._models_support_importance(
        [SimpleNamespace(get_booster=lambda: None)]
    )

    # Assert
    assert supported


def test_models_support_importance_without_a_booster() -> None:
    """A model without ``get_booster`` does not support importance."""
    # Act
    supported = feature_importance._models_support_importance([object()])

    # Assert
    assert not supported


def test_build_model_importance_covers_fallback_keys() -> None:
    """Model importance aligns named and XGBoost fallback keys per importance type."""
    # Arrange
    scores_by_type = {"gain": {"known": 1.0}, "total_gain": {"known": 6.0}, "weight": {"f0": 2.0}}
    booster = SimpleNamespace(get_score=lambda importance_type: scores_by_type[importance_type])
    model = SimpleNamespace(get_booster=lambda: booster)

    # Act
    importance = feature_importance._build_model_importance(_as_regressor(model), ["known"])

    # Assert
    assert importance == {
        "gain": [1.0],
        "total_gain": [6.0],
        "weight": [2.0],
    }


def test_build_base_features_and_aggregate_without_combined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Base-feature aggregation should work for single-model reports without a combined section."""
    # Arrange
    monkeypatch.setattr(
        feature_importance,
        "_build_base_feature_map",
        lambda *_args: {"num__yards": "yards", "cat__team_A": "team"},
    )

    # Act
    base_features = feature_importance._build_base_features(
        _as_preprocessor(SimpleNamespace()),
        ["num__yards", "cat__team_A"],
        {"margin": {"gain": [1.5, 2.0], "total_gain": [4.5, 8.0], "weight": [3.0, 4.0]}},
    )

    # Assert
    assert base_features == {
        "feature_names": ["team", "yards"],
        "margin": {"total_gain": [8.0, 4.5], "weight": [4.0, 3.0]},
    }


@pytest.mark.parametrize(
    "preprocessor",
    [
        pytest.param(SimpleNamespace(), id="no_transformers"),
        pytest.param(SimpleNamespace(transformers_=[]), id="no_feature_names"),
        pytest.param(
            SimpleNamespace(
                transformers_=[],
                get_feature_names_out=lambda: (_ for _ in ()).throw(AttributeError("boom")),
            ),
            id="feature_names_fail",
        ),
        pytest.param(
            SimpleNamespace(
                transformers_=[],
                get_feature_names_out=lambda: ["other"],
            ),
            id="feature_names_mismatch",
        ),
    ],
)
def test_build_base_feature_map_handles_guard_paths(preprocessor: object) -> None:
    """Base-feature mapping should return None for unsupported preprocessors."""
    # Act
    mapping = feature_importance._build_base_feature_map(
        _as_preprocessor(preprocessor),
        ["feat"],
    )

    # Assert
    assert mapping is None


def test_build_base_feature_map_maps_a_valid_preprocessor() -> None:
    """Base-feature mapping should map every output column of a valid preprocessor."""
    # Arrange
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

    # Act
    mapping = feature_importance._build_base_feature_map(
        _as_preprocessor(valid_preprocessor),
        ["num__yards", "cat__team_A", "cat__team_B", "scalar__week"],
    )

    # Assert
    assert mapping == {
        "num__yards": "yards",
        "cat__team_A": "team",
        "cat__team_B": "team",
        "scalar__week": "week",
    }


_ABC_PREPROCESSOR = SimpleNamespace(feature_names_in_=np.array(["a", "b", "c"], dtype=object))


@pytest.mark.parametrize(
    ("columns", "preprocessor", "expected"),
    [
        pytest.param(slice(0, 2), _ABC_PREPROCESSOR, ["a", "b"], id="slice"),
        pytest.param(["x", "y"], _ABC_PREPROCESSOR, ["x", "y"], id="list"),
        pytest.param(np.array(["x", "y"]), _ABC_PREPROCESSOR, ["x", "y"], id="array"),
        pytest.param("week", _ABC_PREPROCESSOR, ["week"], id="scalar"),
        pytest.param(slice(0, 1), SimpleNamespace(), [], id="slice_without_names"),
    ],
)
def test_normalize_cols_handles_slice_list_and_scalar_inputs(
    columns: object, preprocessor: object, expected: list[str]
) -> None:
    """Column selector normalization should support slices, sequences, and scalar values."""
    # Act
    normalized = feature_importance._normalize_cols(columns, _as_preprocessor(preprocessor))

    # Assert
    assert normalized == expected


def test_base_features_rank_by_total_gain_not_summed_average_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A many-category feature split rarely ranks below a numeric feature the trees use often.

    Summing each one-hot column's average gain would put the categorical feature first
    (3 x 5.0 = 15.0 against 4.0); total gain (average gain times splits) does not.
    """
    # Arrange
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

    # Act
    base = feature_importance._build_base_features(
        _as_preprocessor(SimpleNamespace()),
        columns,
        {"margin": head, "total": head},
    )

    # Assert
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
    preprocessor = ml_model_core.build_preprocessor(spec, for_tree=True)
    x_matrix = ml_model_core.fit_transform_matrix(preprocessor, df)
    opp_effect = df["opp"].map({"A": 2.0, "B": -1.0, "C": 0.5, "D": -1.5}).to_numpy()
    y_margin = 3.0 * df["rest"].to_numpy() + opp_effect + rng.normal(scale=0.1, size=rows)
    y_total = df["rest"].to_numpy() - opp_effect + rng.normal(scale=0.1, size=rows)
    params = ml_model_core.resolve_xgb_params(
        ml_model_core.DEFAULT_XGB_PARAMS,
        overrides={
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.3,
            "verbosity": 0,
            "n_jobs": 1,
        },
    )
    margin_model, total_model = ml_model_core.fit_margin_total_models(
        ml_model_core.FitData(x_matrix, y_margin, y_total), params
    )
    model = ml_model_core.MarginTotalModel(
        preprocessor=preprocessor,
        feature_spec=spec,
        margin_model=margin_model,
        total_model=total_model,
        target_columns=("away_score", "home_score"),
    )
    return model, df


def test_fitted_report_sums_total_gain_over_one_hot_columns_and_heads() -> None:
    """Base-feature total gain is the booster's per-column total gain summed per base feature."""
    # Arrange
    model, _ = _fit_rest_opp_model()
    margin_model, total_model = model.margin_model, model.total_model

    # Act
    report = feature_importance.build_feature_importance_report(model)

    # Assert
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
    return ml_model_core.transform_matrix(
        model.preprocessor, ml_model_core.apply_feature_spec(df, model.feature_spec)
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


@pytest.mark.parametrize("head_name", ["margin", "total", "early_stopped"])
def test_shap_values_add_up_to_each_rows_prediction(head_name: str) -> None:
    """Each row's SHAP values plus the base value reproduce the head's prediction.

    An early-stopped head is explained over the trees it predicts with, up to its best
    iteration, not over every tree it grew.
    """
    # Arrange
    model, df = _fit_rest_opp_model()
    x_matrix = _preprocessed(model, df)
    heads = {
        "margin": lambda: model.margin_model,
        "total": lambda: model.total_model,
        "early_stopped": lambda: _early_stopped_head(x_matrix, len(df)),
    }
    head = heads[head_name]()

    # Act
    values, base_value = feature_importance.shap_values(head, x_matrix)

    # Assert
    assert values.shape == (len(df), 5)
    np.testing.assert_allclose(
        values.sum(axis=1) + base_value,
        ml_model_core.predict_xgb(head, x_matrix),
        rtol=1e-5,
        atol=1e-5,
    )


def test_mean_abs_shap_sums_a_base_features_columns_before_the_absolute_value() -> None:
    """One-hot columns' signed contributions cancel within a row before the absolute value.

    Row 1 gives ``opp`` +1 and -1 (net 0), row 2 gives +2: the mean is 1, where averaging each
    column's absolute value and summing would report 2.
    """
    # Arrange
    values = np.array([[1.0, -1.0, 0.5], [2.0, 0.0, -1.5]])

    # Act
    result = feature_importance.mean_abs_shap_by_base(
        values,
        ["cat__opp_A", "cat__opp_B", "num__rest"],
        {"cat__opp_A": "opp", "cat__opp_B": "opp", "num__rest": "rest"},
    )

    # Assert
    assert result == {"opp": pytest.approx(1.0), "rest": pytest.approx(1.0)}


def test_fitted_report_records_mean_abs_shap_per_head_and_combined() -> None:
    """With rows to explain, each head and the combined block carry mean |SHAP| per base feature."""
    # Arrange
    model, df = _fit_rest_opp_model()
    x_matrix = _preprocessed(model, df)

    # Act
    report = feature_importance.build_feature_importance_report(model, shap_rows=df)

    # Assert
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
    # Arrange
    model, df = _fit_rest_opp_model()
    rows = df.assign(season=[2020] * 60 + [2022] * 60)

    # Act
    report = feature_importance.build_feature_importance_report(model, shap_rows=rows)

    # Assert
    assert report is not None
    assert report["shap"]["seasons"] == [2020, 2022]


def test_empty_shap_rows_leave_the_report_without_shap() -> None:
    """No rows to explain means no SHAP section, and the gain measures still load."""
    # Arrange
    model, df = _fit_rest_opp_model()
    rows = df.iloc[:0]

    # Act
    report = feature_importance.build_feature_importance_report(model, shap_rows=rows)

    # Assert
    assert report is not None
    assert "shap" not in report
    assert "mean_abs_shap" not in report["base_features"]["combined"]
    assert "total_gain" in report["base_features"]["combined"]


def _broken_preprocessor() -> SimpleNamespace:
    """Return a preprocessor whose feature-name method fails with an unexpected error."""
    return SimpleNamespace(
        transformers_=[],
        get_feature_names_out=lambda: (_ for _ in ()).throw(RuntimeError("broken")),
    )


def test_feature_name_resolution_surfaces_unexpected_errors() -> None:
    """Only a missing or unfitted feature-name method falls back; other errors surface."""
    # Arrange
    broken = _broken_preprocessor()

    # Act & Assert
    with pytest.raises(RuntimeError, match="broken"):
        feature_importance.resolve_feature_names(
            _as_preprocessor(broken), _as_regressor(SimpleNamespace(n_features_in_=1))
        )


def test_base_feature_map_surfaces_unexpected_errors() -> None:
    """The base-feature map lets an unexpected feature-name error surface."""
    # Arrange
    broken = _broken_preprocessor()

    # Act & Assert
    with pytest.raises(RuntimeError, match="broken"):
        feature_importance._build_base_feature_map(_as_preprocessor(broken), ["feat"])
