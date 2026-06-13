"""Tests for feature importance reporting utilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

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

        def __init__(self, team_model: object, market_model: object | None) -> None:
            """Store component models for the test."""
            self.team_model = team_model
            self.market_model = market_model

    class DummyMargin:
        """Synthetic margin/total model class for dispatch testing."""

        pass

    class DummyScore:
        """Synthetic score model class for dispatch testing."""

        pass

    monkeypatch.setattr(feature_importance, "BlendedMarginTotalModel", DummyBlend)
    monkeypatch.setattr(feature_importance, "MarginTotalModel", DummyMargin)
    monkeypatch.setattr(feature_importance, "ScoreModel", DummyScore)

    def fake_margin_report(model: object) -> dict[str, object] | None:
        """Return a synthetic report keyed off the input object."""
        if model == "team":
            return {"team": True}
        if model == "market":
            return {"market": True}
        return {"margin": True}

    monkeypatch.setattr(feature_importance, "_build_margin_total_report", fake_margin_report)
    monkeypatch.setattr(feature_importance, "_build_score_report", lambda _model: {"score": True})

    blend_report = feature_importance.build_feature_importance_report(DummyBlend("team", "market"))
    assert blend_report == {
        "model_kind": "blend",
        "components": {"team": {"team": True}, "market": {"market": True}},
    }

    margin_report = feature_importance.build_feature_importance_report(DummyMargin())
    assert margin_report == {"margin": True, "model_kind": "margin_total"}

    score_report = feature_importance.build_feature_importance_report(DummyScore())
    assert score_report == {"score": True, "model_kind": "score"}

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
    monkeypatch.setattr(feature_importance, "ScoreModel", tuple)

    def _raise_attribute_error(_model: object) -> dict[str, object] | None:
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
        "feature_names": ["feat"],
        "models": {"margin": {"gain": [1.0], "weight": [1.0]}},
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

    booster = SimpleNamespace(
        get_score=lambda importance_type: (
            {"known": 1.0} if importance_type == "gain" else {"f0": 2.0}
        )
    )
    model = SimpleNamespace(get_booster=lambda: booster)

    assert feature_importance._build_model_importance(_as_regressor(model), ["known"]) == {
        "gain": [1.0],
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
        {"margin": {"gain": [1.5, 2.0], "weight": [3.0, 4.0]}},
    )

    assert base_features == {
        "feature_names": ["team", "yards"],
        "margin": {"gain": [2.0, 1.5], "weight": [4.0, 3.0]},
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
