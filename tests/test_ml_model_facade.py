"""Tests for the legacy nfl_predictor.ml_model facade module."""

from __future__ import annotations

import importlib
import types
from typing import TYPE_CHECKING

from nfl_predictor import ml_model
from tests.fakes import returning

if TYPE_CHECKING:
    import pytest


def test_ml_model_main_delegates_to_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calls into nfl_predictor.cli.train.main via importlib."""
    # Arrange
    called = {"ok": False}

    dummy = types.SimpleNamespace(main=lambda: called.update(ok=True))
    monkeypatch.setattr(importlib, "import_module", returning(dummy))

    # Act
    ml_model.main()

    # Assert
    assert called["ok"] is True


def test_ml_model_getattr_and_dir_forwarding() -> None:
    """Exposes symbols from split implementation modules via __getattr__."""
    # Act
    attribute = ml_model.get_target_columns

    # Assert
    assert callable(attribute)


def test_ml_model_dir_lists_forwarded_symbols() -> None:
    """Lists symbols from split implementation modules via __dir__."""
    # Act
    names = dir(ml_model)

    # Assert
    assert "get_target_columns" in names
