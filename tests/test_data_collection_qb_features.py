"""Tests for attaching the quarterback family during data collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from nfl_predictor import constants, data_collection
from nfl_predictor.utils import polars_utils


def _dropbacks(season: int, week: int, team: str, passer: str, count: int) -> list[dict[str, Any]]:
    """Return ``count`` completed-pass dropbacks by ``passer`` for ``team``."""
    return [
        {
            "season": season,
            "week": week,
            "season_type": "REG",
            "posteam": team,
            "defteam": "OPP",
            "qb_dropback": 1,
            "sack": 0,
            "complete_pass": 1,
            "yards_gained": 5.0,
            "qb_epa": 0.2,
            "passer_player_id": passer,
            "passer_player_name": None,
        }
        for _ in range(count)
    ]


def _identity_file(tmp_path: Path) -> Path:
    """Write a two-quarterback identity file and return its path."""
    path = tmp_path / "qb_meta_data.csv"
    pl.DataFrame({"name_id": ["Tom Brady", "Josh Allen"], "gsis_id": ["TB", "JA"]}).write_csv(path)
    return path


def test_attach_qb_features_loads_missing_history_and_joins_both_sides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Seasons missing from memory are loaded so career rates span the whole history."""
    in_memory = pl.DataFrame(
        _dropbacks(2001, 1, "NE", "TB", 20) + _dropbacks(2001, 1, "BUF", "JA", 12)
    )
    history = pl.DataFrame(_dropbacks(2000, 5, "NE", "TB", 30))
    requested: list[list[int]] = []
    refreshed: list[bool] = []

    def fake_load_pbp(seasons: list[int], **kwargs: Any) -> pl.DataFrame:
        """Record the requested seasons and refresh flag, return the earlier history."""
        requested.append(list(seasons))
        refreshed.append(bool(kwargs.get("force_refresh")))
        return history

    monkeypatch.setattr(polars_utils, "load_pbp", fake_load_pbp)
    games = pl.DataFrame(
        {
            "game_id": ["g1"],
            "season": [2001],
            "week": [2],
            "away_qb": ["Tom Brady"],
            "home_qb": ["Josh Allen"],
        }
    )

    out = data_collection._attach_qb_features(
        games,
        in_memory,
        max_season=2001,
        current_season=2026,
        identity_path=_identity_file(tmp_path),
    )

    assert requested == [list(range(constants.NFLREADPY_MIN_SEASON, 2001))]
    assert refreshed == [False], "history seasons always come from the cache"
    row = out.row(0, named=True)
    assert row["away_qb_history_dropbacks"] == 50
    assert row["home_qb_history_dropbacks"] == 12
    assert row["qb_history_dropbacks_diff"] == 38
    assert out["game_id"].to_list() == ["g1"]


def test_attach_qb_features_leaves_rows_without_quarterbacks_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rows without quarterback columns come back unchanged and trigger no loading."""

    def fail_load_pbp(*_args: Any, **_kwargs: Any) -> pl.DataFrame:
        """Fail if play-by-play is requested."""
        raise AssertionError("load_pbp should not be called")

    monkeypatch.setattr(polars_utils, "load_pbp", fail_load_pbp)
    games = pl.DataFrame({"game_id": ["g1"], "season": [2001], "week": [1]})

    out = data_collection._attach_qb_features(
        games,
        pl.DataFrame(),
        max_season=2001,
        current_season=2026,
        identity_path=tmp_path / "missing.csv",
    )

    assert out.equals(games)
