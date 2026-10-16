"""Tests for the column registry and projection."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from nfl_predictor.api.registry import REGISTRY, group_map, project
from nfl_predictor.api.registry.betting import BETTING_COLUMNS
from nfl_predictor.api.registry.columns import ColumnMeta, jsonable, register
from nfl_predictor.api.registry.model import WF_COMPARE_COLUMNS
from nfl_predictor.api.registry.power import POWER_COLUMNS, STANDINGS_COLUMNS
from nfl_predictor.api.registry.predictions import PICK_COLUMNS, PREDICTION_COLUMNS


def test_every_column_is_documented() -> None:
    """Each registered column has a label, a description, and a group."""
    # Act
    entries = list(REGISTRY.items())

    # Assert
    assert len(entries) > 100
    for key, meta in entries:
        assert meta.key == key
        assert meta.label, key
        assert meta.description, key
        assert meta.group, key


@pytest.mark.parametrize(
    "columns",
    [
        PREDICTION_COLUMNS,
        PICK_COLUMNS,
        BETTING_COLUMNS,
        POWER_COLUMNS,
        STANDINGS_COLUMNS,
        WF_COMPARE_COLUMNS,
    ],
)
def test_display_lists_only_reference_registered_columns(columns: list[str]) -> None:
    """Display orderings never mention an unregistered key."""
    # Act
    missing = [key for key in columns if key not in REGISTRY]

    # Assert
    assert missing == []


def test_total_betting_columns_are_not_actionable() -> None:
    """Every total/over-under betting column is flagged informational."""
    # Act
    totals = [key for key in BETTING_COLUMNS if key.startswith("total_") and key != "total_line"]

    # Assert
    assert totals
    assert all(not REGISTRY[key].actionable for key in totals)
    assert REGISTRY["moneyline_action"].actionable


def test_register_rejects_duplicates() -> None:
    """Registering the same key twice is an error."""
    # Arrange
    duplicate = ColumnMeta("game_id", "x", "y", "z")

    # Act & Assert
    with pytest.raises(ValueError, match="already registered"):
        register(duplicate)


def test_project_drops_unknown_and_jsonifies() -> None:
    """Projection keeps registered, present columns in order and makes cells JSON-safe."""
    # Arrange
    df = pl.DataFrame(
        {
            "home_win_prob": [0.6, float("nan")],
            "game_id": ["a", "b"],
            "date": [dt.date(2026, 9, 13), None],
            "unregistered": [1, 2],
        }
    )

    # Act
    payload = project(df, ["game_id", "home_win_prob", "date", "missing", "unregistered"])

    # Assert
    assert payload.visible_columns == ["game_id", "home_win_prob", "date"]
    assert payload.rows == [
        {"game_id": "a", "home_win_prob": 0.6, "date": "2026-09-13"},
        {"game_id": "b", "home_win_prob": None, "date": None},
    ]
    assert payload.column_groups == {"Game": ["game_id", "date"], "Model": ["home_win_prob"]}
    assert payload.column_metadata["home_win_prob"].kind == "prob"


def test_jsonable_turns_infinity_into_none() -> None:
    """Non-finite floats become ``None``."""
    # Act
    result = jsonable(float("inf"))

    # Assert
    assert result is None


def test_jsonable_turns_datetimes_into_iso_strings() -> None:
    """Datetimes become ISO strings."""
    # Act
    result = jsonable(dt.datetime(2026, 9, 13, 13, 0))

    # Assert
    assert result == "2026-09-13T13:00:00"


def test_group_map_of_unknown_keys_is_empty() -> None:
    """Unregistered keys belong to no group."""
    # Act
    groups = group_map(["nope"])

    # Assert
    assert groups == {}
