"""Tests for the optional SHAP analysis script."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


def test_import_shap_returns_module_when_available(monkeypatch) -> None:
    """SHAP import helper returns the imported module when available."""
    from nfl_predictor.cli import explain

    sentinel = SimpleNamespace(name="shap")
    monkeypatch.setattr(
        importlib, "import_module", lambda name: sentinel if name == "shap" else None
    )

    assert explain._import_shap() is sentinel


def test_import_shap_returns_none_on_import_error(monkeypatch) -> None:
    """SHAP import helper returns None when the optional dependency is missing."""
    from nfl_predictor.cli import explain

    def raise_import_error(name: str) -> object:
        if name == "shap":
            raise ImportError("missing shap")
        raise AssertionError(f"unexpected import request: {name}")

    monkeypatch.setattr(importlib, "import_module", raise_import_error)

    assert explain._import_shap() is None


def test_shap_analysis_missing_shap(monkeypatch, tmp_path: Path) -> None:
    """Script should return non-zero when shap is unavailable."""
    from nfl_predictor.cli import explain

    monkeypatch.setattr(explain, "_import_shap", lambda: None)
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
    assert explain.main() == 2
