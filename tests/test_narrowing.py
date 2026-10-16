"""Tests for ``nfl_predictor.utils.narrowing``, which types parsed JSON and unpickled values."""

from __future__ import annotations

import pytest

from nfl_predictor.utils import narrowing


def test_as_str_dict_dict_returns_the_same_dict() -> None:
    # Arrange
    value = {"version": 1}

    # Act
    result = narrowing.as_str_dict(value)

    # Assert
    assert result is value


@pytest.mark.parametrize("value", [None, [1, 2], "text", 3, ("a", 1)])
def test_as_str_dict_non_dict_returns_none(value: object) -> None:
    # Act
    result = narrowing.as_str_dict(value)

    # Assert
    assert result is None


def test_as_list_list_returns_the_same_list() -> None:
    # Arrange
    value = [1, "two"]

    # Act
    result = narrowing.as_list(value)

    # Assert
    assert result is value


@pytest.mark.parametrize("value", [None, (1, 2), "text", {"a": 1}, 3])
def test_as_list_non_list_returns_none(value: object) -> None:
    # Act
    result = narrowing.as_list(value)

    # Assert
    assert result is None
