"""Tests for the play-by-play cache's column list: what it keeps, types and refreshes."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils.polars import loaders

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

# The columns every play-by-play cache file written with the earlier, shorter list holds, in
# that list's order.
_EARLIER_PBP_COLUMNS = (
    "game_id",
    "season",
    "season_type",
    "week",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "posteam_type",
    "play_type",
    "qb_dropback",
    "qb_kneel",
    "qb_spike",
    "rush",
    "pass",
    "rush_attempt",
    "pass_attempt",
    "down",
    "ydstogo",
    "yardline_100",
    "epa",
    "qb_epa",
    "success",
    "yards_gained",
    "air_yards",
    "cpoe",
    "xpass",
    "pass_oe",
    "third_down_converted",
    "third_down_failed",
    "fourth_down_converted",
    "fourth_down_failed",
    "two_point_attempt",
    "two_point_conv_result",
    "td_team",
    "sack",
    "interception",
    "fumble",
    "fumble_lost",
    "complete_pass",
    "pass_touchdown",
    "rush_touchdown",
    "first_down",
    "first_down_pass",
    "first_down_rush",
    "penalty",
    "penalty_team",
    "penalty_yards",
    "special",
    "special_teams_play",
    "fixed_drive",
    "fixed_drive_result",
    "drive_start_yard_line",
    "passer_player_id",
    "passer_player_name",
)
# Every column the ratings port reads that the earlier list lacked, with a value as nflverse
# publishes it: text columns hold text, every other column a number. Team codes are canonical,
# so normalization leaves them as they are.
_ADDED_COLUMNS: dict[str, str | float] = {
    "play_id": 1045.0,
    "qtr": 4.0,
    "half_seconds_remaining": 95.0,
    "score_differential": -3.0,
    "score_differential_post": 4.0,
    "posteam_score": 17.0,
    "posteam_score_post": 24.0,
    "wp": 0.31,
    "wpa": 0.12,
    "goal_to_go": 1.0,
    "yrdln": "KC 8",
    "side_of_field": "KC",
    "end_yard_line": "KC 1",
    "qb_scramble": 1.0,
    "aborted_play": 0.0,
    "shotgun": 1.0,
    "no_huddle": 0.0,
    "pass_length": "short",
    "incomplete_pass": 0.0,
    "passing_yards": 8.0,
    "rushing_yards": 7.0,
    "return_yards": 0.0,
    "air_epa": 0.4,
    "yac_epa": 0.2,
    "yards_after_catch": 3.0,
    "xyac_mean_yardage": 2.5,
    "touchdown": 1.0,
    "first_down_penalty": 0.0,
    "qb_hit": 0.0,
    "tackled_for_loss": 0.0,
    "fumble_forced": 0.0,
    "defensive_two_point_conv": 0.0,
    "penalty_type": "Defensive Holding",
    "punt_attempt": 0.0,
    "field_goal_attempt": 0.0,
    "kickoff_attempt": 0.0,
    "return_team": "KC",
    "timeout_team": "BUF",
    "drive_play_count": 9.0,
    "drive_first_downs": 4.0,
    "drive_inside20": 1.0,
    "drive_time_of_possession": "4:51",
    "drive_yards_penalized": 5.0,
    "ydsnet": 75.0,
    "drive_end_yard_line": "KC 8",
    "series": 12.0,
    "series_success": 1.0,
    "series_result": "Touchdown",
    "rusher_player_id": "00-0034857",
    "rusher_player_name": "J.Allen",
    "receiver_player_id": "00-0036410",
    "receiver_player_name": "K.Coleman",
    "td_player_id": "00-0034857",
    "penalty_player_id": "00-0035000",
    "pass_defense_1_player_id": "00-0036001",
    "fumbled_1_team": "BUF",
    "fumbled_1_player_id": "00-0034857",
    "fumbled_2_team": "KC",
    "fumbled_2_player_id": "00-0036002",
    "fumble_recovery_1_team": "BUF",
    "fumble_recovery_1_player_id": "00-0036003",
    "fumble_recovery_2_team": "KC",
    "fumble_recovery_2_player_id": "00-0036004",
    "forced_fumble_player_1_team": "KC",
    "forced_fumble_player_1_player_id": "00-0036005",
    "forced_fumble_player_2_team": "KC",
    "forced_fumble_player_2_player_id": "00-0036006",
    "solo_tackle_1_team": "KC",
    "solo_tackle_1_player_id": "00-0036007",
    "solo_tackle_2_team": "KC",
    "solo_tackle_2_player_id": "00-0036008",
    "assist_tackle_1_team": "KC",
    "assist_tackle_1_player_id": "00-0036009",
    "assist_tackle_2_team": "KC",
    "assist_tackle_2_player_id": "00-0036010",
    "assist_tackle_3_team": "KC",
    "assist_tackle_3_player_id": "00-0036011",
    "assist_tackle_4_team": "KC",
    "assist_tackle_4_player_id": "00-0036012",
    "tackle_with_assist_1_team": "KC",
    "tackle_with_assist_1_player_id": "00-0036013",
    "tackle_with_assist_2_team": "KC",
    "tackle_with_assist_2_player_id": "00-0036014",
    "desc": "(1:35) (Shotgun) 17-J.Allen scrambles up the middle to KC 1 for 7 yards.",
}
# Columns that hold a team code, as nflverse names them, each with a code normalization rewrites.
# nflverse play-by-play writes only LA and WAS of these (LV, LAC and LA in every season); the
# older codes (OAK, SD, STL) exercise the rest of the alias table.
_TEAM_CODE_ALIASES: dict[str, str] = {
    "fumbled_1_team": "OAK",
    "fumbled_2_team": "SD",
    "fumble_recovery_1_team": "STL",
    "fumble_recovery_2_team": "LA",
    "forced_fumble_player_1_team": "WAS",
    "forced_fumble_player_2_team": "OAK",
    "solo_tackle_1_team": "SD",
    "solo_tackle_2_team": "STL",
    "assist_tackle_1_team": "LA",
    "assist_tackle_2_team": "WAS",
    "assist_tackle_3_team": "OAK",
    "assist_tackle_4_team": "SD",
    "tackle_with_assist_1_team": "STL",
    "tackle_with_assist_2_team": "LA",
    "return_team": "WAS",
    "timeout_team": "OAK",
    "side_of_field": "LA",
}
_YARD_LINE_TEXT: dict[str, str] = {
    "yrdln": "LA 25",
    "drive_start_yard_line": "OAK 30",
    "drive_end_yard_line": "SD 10",
    "end_yard_line": "STL 5",
}


def _download(columns: Mapping[str, object] | None = None, *, season: int = 2020) -> pl.DataFrame:
    """Return a one-play raw download carrying the identity columns and ``columns``."""
    identity: dict[str, object] = {
        "game_id": f"{season}_01_BUF_KC",
        "season": season,
        "season_type": "REG",
        "week": 1,
        "posteam": "BUF",
        "defteam": "KC",
        "home_team": "KC",
        "away_team": "BUF",
    }
    return pl.DataFrame([{**identity, **(columns or {})}])


def _fail_download(*_args: object, **_kwargs: object) -> pl.DataFrame:
    """Fail if nflreadpy is called."""
    msg = "nflreadpy play-by-play load should not be called"
    raise AssertionError(msg)


def test_pbp_columns_keep_the_earlier_list_and_add_every_ported_read() -> None:
    """The cache requests every earlier column, in its order, and every column the port reads."""
    # Arrange
    requested = list(constants.PBP_COLUMNS)

    # Act
    earlier_in_order = [column for column in requested if column in _EARLIER_PBP_COLUMNS]

    # Assert
    assert earlier_in_order == list(_EARLIER_PBP_COLUMNS)
    assert set(_ADDED_COLUMNS) <= set(requested)
    assert len(requested) == len(set(requested))


def test_pbp_team_columns_name_every_cached_team_code_column() -> None:
    """Every requested column that holds a team code is normalized at load time."""
    # Arrange
    team_code_columns = {
        column
        for column in constants.PBP_COLUMNS
        if column.endswith("team") or column == "side_of_field"
    }

    # Act
    normalized = set(constants.PBP_TEAM_COLUMNS)

    # Assert
    assert normalized == team_code_columns


@pytest.mark.parametrize("recorded", [True, False], ids=["recorded", "unrecorded"])
def test_load_pbp_earlier_cache_file_is_downloaded_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, recorded: bool
) -> None:
    """A cache file written with the earlier list is stale, with or without its column record."""
    # Arrange
    stale = pl.DataFrame(
        [{**dict.fromkeys(_EARLIER_PBP_COLUMNS), "season": 2020, "week": 1, "posteam": "OLD"}],
        schema={column: loaders._PBP_COLUMN_DTYPES[column] for column in _EARLIER_PBP_COLUMNS},
    )
    metadata = (
        {loaders._REQUESTED_COLUMNS_METADATA_KEY: json.dumps(list(_EARLIER_PBP_COLUMNS))}
        if recorded
        else None
    )
    stale.write_parquet(tmp_path / "pbp_2020_reg.parquet", metadata=metadata)
    fresh = _download(_ADDED_COLUMNS)
    monkeypatch.setattr(loaders.nfl, "load_pbp", lambda seasons: fresh)

    # Act
    loaded = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)

    # Assert
    assert loaded.get_column("posteam").to_list() == ["BUF"]
    assert set(_ADDED_COLUMNS) <= set(pl.read_parquet_schema(tmp_path / "pbp_2020_reg.parquet"))


def _unavailable(seasons: list[int]) -> pl.DataFrame:
    """Fail the way an nflreadpy outage does."""
    msg = f"play-by-play for {seasons} is unavailable"
    raise ConnectionError(msg)


def _gap_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the warnings that name requested columns a cache file lacks."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "lacks the requested columns" in record.getMessage()
    ]


def test_load_pbp_fallback_to_an_earlier_cache_file_names_the_columns_it_lacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed current-season refresh that falls back to an earlier file warns of its gaps."""
    # Arrange
    stale = pl.DataFrame(
        [{**dict.fromkeys(_EARLIER_PBP_COLUMNS), "season": 2026, "week": 1, "posteam": "OLD"}],
        schema={column: loaders._PBP_COLUMN_DTYPES[column] for column in _EARLIER_PBP_COLUMNS},
    )
    stale.write_parquet(tmp_path / "pbp_2026_reg.parquet")
    monkeypatch.setattr(loaders.nfl, "load_pbp", _unavailable)
    caplog.set_level(logging.WARNING)
    lacking = [column for column in constants.PBP_COLUMNS if column not in _EARLIER_PBP_COLUMNS]

    # Act
    loaded = loaders.load_pbp([2026], cache_dir=tmp_path, current_season=2026)

    # Assert
    assert loaded.get_column("posteam").to_list() == ["OLD"]
    warnings = _gap_warnings(caplog)
    assert len(warnings) == 1
    assert "drive_inside20" in lacking
    assert warnings[0].endswith(": " + ", ".join(lacking))


def test_load_pbp_fallback_to_a_complete_cache_file_names_no_columns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A fallback file that holds every requested column draws no warning about columns."""
    # Arrange
    complete = pl.DataFrame(
        [{**dict.fromkeys(constants.PBP_COLUMNS), "season": 2026, "week": 1, "posteam": "NEW"}],
        schema=loaders._PBP_COLUMN_DTYPES,
    )
    complete.write_parquet(tmp_path / "pbp_2026_reg.parquet")
    monkeypatch.setattr(loaders.nfl, "load_pbp", _unavailable)
    caplog.set_level(logging.WARNING)

    # Act
    loaded = loaders.load_pbp([2026], cache_dir=tmp_path, current_season=2026)

    # Assert
    assert loaded.get_column("posteam").to_list() == ["NEW"]
    assert _gap_warnings(caplog) == []


def test_load_pbp_keeps_every_added_column_as_published(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each added column reaches the cache and the caller with its published value."""
    # Arrange
    monkeypatch.setattr(loaders.nfl, "load_pbp", lambda seasons: _download(_ADDED_COLUMNS))

    # Act
    loaded = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)

    # Assert
    assert {column: loaded.get_column(column)[0] for column in _ADDED_COLUMNS} == _ADDED_COLUMNS


def test_load_pbp_reads_the_added_columns_back_from_the_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A season cached with the added columns is served from the cache, columns included."""
    # Arrange
    monkeypatch.setattr(loaders.nfl, "load_pbp", lambda seasons: _download(_ADDED_COLUMNS))
    downloaded = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)
    monkeypatch.setattr(loaders.nfl, "load_pbp", _fail_download)

    # Act
    cached = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)

    # Assert
    assert cached.equals(downloaded)
    assert set(_ADDED_COLUMNS) <= set(cached.columns)


def test_empty_pbp_frame_types_the_added_columns_as_published() -> None:
    """With no season loaded, each added column has the type its published values have."""
    # Act
    empty = loaders.load_pbp([])

    # Assert
    assert {column: empty.schema[column] for column in _ADDED_COLUMNS} == {
        column: pl.String() if isinstance(value, str) else pl.Float64()
        for column, value in _ADDED_COLUMNS.items()
    }


def test_load_pbp_normalizes_the_added_team_code_columns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Historical and alias team codes in the added team columns become canonical codes."""
    # Arrange
    monkeypatch.setattr(loaders.nfl, "load_pbp", lambda seasons: _download(_TEAM_CODE_ALIASES))

    # Act
    loaded = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)

    # Assert
    assert {column: loaded.get_column(column)[0] for column in _TEAM_CODE_ALIASES} == {
        column: constants.ALIAS_TO_CANONICAL[code] for column, code in _TEAM_CODE_ALIASES.items()
    }


def test_load_pbp_keeps_yard_line_text_as_published(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Yard lines written as a team code and a number keep the code nflverse wrote."""
    # Arrange
    monkeypatch.setattr(loaders.nfl, "load_pbp", lambda seasons: _download(_YARD_LINE_TEXT))

    # Act
    loaded = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)

    # Assert
    assert {column: loaded.get_column(column)[0] for column in _YARD_LINE_TEXT} == _YARD_LINE_TEXT


def test_load_pbp_stacks_seasons_that_type_goal_to_go_differently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A flag nflverse types as an integer in one season and a float in another loads as floats."""
    # Arrange
    downloads = {
        2003: _download(season=2003).with_columns(pl.lit(1, dtype=pl.Int32).alias("goal_to_go")),
        2020: _download(season=2020).with_columns(
            pl.lit(0.0, dtype=pl.Float64).alias("goal_to_go")
        ),
    }
    monkeypatch.setattr(loaders.nfl, "load_pbp", lambda seasons: downloads[seasons[0]])

    # Act
    loaded = loaders.load_pbp([2003, 2020], cache_dir=tmp_path, current_season=2024)

    # Assert
    assert loaded.schema["goal_to_go"] == pl.Float64()
    assert loaded.get_column("goal_to_go").to_list() == [1.0, 0.0]
