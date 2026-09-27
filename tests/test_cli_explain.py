"""Tests for the SHAP analysis command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pytest

from nfl_predictor.cli import explain
from tests.test_feature_importance import _fit_rest_opp_model


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
