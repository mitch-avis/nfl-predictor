"""Tests for the SHAP analysis command."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, cast

import joblib
import pytest
from tests.test_feature_importance import _fit_rest_opp_model

from nfl_predictor.cli import explain
from nfl_predictor.ml import ml_model_core

if TYPE_CHECKING:
    from pathlib import Path


def test_explain_runs_without_the_shap_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SHAP values come from XGBoost itself, so the command needs no ``shap`` package."""
    model, df = _fit_rest_opp_model()
    model_path = tmp_path / "model.joblib"
    data_path = tmp_path / "data.csv"
    out_path = tmp_path / "shap_report.json"
    joblib.dump(model, model_path)
    df.to_csv(data_path, index=False)
    monkeypatch.setitem(sys.modules, "shap", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "explain",
            "--model-path",
            str(model_path),
            "--data-path",
            str(data_path),
            "--output-path",
            str(out_path),
        ],
    )

    assert explain.main() == 0

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["sample_size"] == len(df)
    assert {row["feature"] for row in report["rows"]} == set(
        model.preprocessor.get_feature_names_out()
    )


def test_explain_refuses_a_saved_blend_model() -> None:
    """The blend model kind was retired, so a saved blend model is refused with the reason."""
    blend = ml_model_core.BlendedMarginTotalModel(
        team_model=cast("ml_model_core.MarginTotalModel", None),
        blend_layer=cast("ml_model_core.BlendLayer", None),
        calibrator=None,
        target_columns=("away_score", "home_score"),
    )

    with pytest.raises(TypeError, match="blend model kind was retired"):
        explain._select_model_component(blend)


def test_explain_refuses_an_object_that_is_not_a_model() -> None:
    """A pickle that holds something other than a margin/total model is the wrong type."""
    with pytest.raises(TypeError, match="Unsupported model type: dict"):
        explain._select_model_component({"not": "a model"})


def test_explain_accepts_only_the_margin_and_total_heads() -> None:
    """The target names one of the two heads."""
    model, _ = _fit_rest_opp_model()
    assert explain._resolve_head(model, "margin") == (model.margin_model, "margin")
    assert explain._resolve_head(model, "total") == (model.total_model, "total")
    with pytest.raises(ValueError, match=r"--target margin\|total"):
        explain._resolve_head(model, "spread")
