"""Tests for the column-source reader under ``.agents/sos_port/``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.sos_port_scripts import load_sos_port_script

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture(scope="module")
def sos_sources() -> ModuleType:
    """Return the column-source reader, imported from its directory."""
    return load_sos_port_script("sos_sources")


def test_string_triples_returns_each_three_string_item_as_a_tuple(sos_sources: ModuleType) -> None:
    # Arrange
    value = [["completions", "attempts", "completion_pct"], ("yards", "plays", "yards_per_play")]

    # Act
    triples = sos_sources._string_triples(value, "RATES")

    # Assert
    assert triples == [
        ("completions", "attempts", "completion_pct"),
        ("yards", "plays", "yards_per_play"),
    ]


def test_string_triples_with_a_two_string_item_raises(sos_sources: ModuleType) -> None:
    # Arrange
    value = [["completions", "attempts", "completion_pct"], ["yards", "plays"]]

    # Act & Assert
    with pytest.raises(TypeError, match="RATES is not a sequence of string triples"):
        sos_sources._string_triples(value, "RATES")
