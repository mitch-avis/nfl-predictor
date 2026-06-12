"""Tests for the legacy nfl_predictor.ml_model facade module."""

from __future__ import annotations

import types

from pytest import MonkeyPatch

from nfl_predictor import ml_model


def test_ml_model_main_delegates_to_cli(monkeypatch: MonkeyPatch) -> None:
    """Calls into nfl_predictor.ml.ml_model_cli.main via importlib."""
    called = {"ok": False}

    dummy = types.SimpleNamespace(main=lambda: called.__setitem__("ok", True))
    monkeypatch.setattr(ml_model.importlib, "import_module", lambda _name: dummy)

    ml_model.main()
    assert called["ok"] is True


def test_ml_model_getattr_and_dir_forwarding() -> None:
    """Exposes symbols from split implementation modules via __getattr__/__dir__."""
    assert callable(ml_model.get_target_columns)
    names = dir(ml_model)
    assert "get_target_columns" in names
