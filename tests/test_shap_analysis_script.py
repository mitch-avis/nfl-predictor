"""Tests for the optional SHAP analysis script."""

from __future__ import annotations

import sys
from pathlib import Path


def test_shap_analysis_missing_shap(monkeypatch, tmp_path: Path) -> None:
    """Script should return non-zero when shap is unavailable."""
    from scripts import shap_analysis

    monkeypatch.setattr(shap_analysis, "_import_shap", lambda: None)
    model_path = tmp_path / "model.joblib"
    data_path = tmp_path / "data.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shap_analysis.py",
            "--model-path",
            str(model_path),
            "--data-path",
            str(data_path),
        ],
    )
    assert shap_analysis.main() == 2
