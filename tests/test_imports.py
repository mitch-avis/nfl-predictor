"""Import-sanity tests for the public import surface.

These tests are intentionally lightweight. They exist to ensure that the package's compatibility
facades remain importable and that basic import paths used by scripts/tests do not regress.
"""

from __future__ import annotations

import importlib


def test_import_nfl_predictor_package() -> None:
    """The top-level package imports cleanly."""

    importlib.import_module("nfl_predictor")


def test_import_legacy_facades() -> None:
    """Legacy facade modules import cleanly."""

    importlib.import_module("nfl_predictor.ml_model")
    importlib.import_module("nfl_predictor.utils.polars_utils")
