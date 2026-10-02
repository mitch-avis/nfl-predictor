"""Tests for Polars loader helpers."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.polars import loaders, teamrankings

if TYPE_CHECKING:
    from pathlib import Path


def test_is_numeric_dtype() -> None:
    """Check numeric dtype detection works as expected."""
    assert loaders._is_numeric_dtype(pl.Int32())
    assert loaders._is_numeric_dtype(pl.Float64())
    assert not loaders._is_numeric_dtype(pl.Utf8())


def test_polars_utils_facade() -> None:
    """Polars utils facade exposes expected functions."""
    assert hasattr(polars_utils, "load_schedule")
    assert "load_schedule" in dir(polars_utils)


def test_add_stadium_location() -> None:
    """Stadium location data is added correctly based on stadium_id."""
    stadium_id = next(iter(constants.STADIUMS.keys()))
    df = pl.DataFrame({"stadium_id": [stadium_id]})

    out = loaders._add_stadium_location(df)

    assert "stadium_name" in out.columns
    assert "stadium_city" in out.columns
    assert "stadium_state" in out.columns
    assert out["stadium_name"][0] == constants.STADIUMS[stadium_id]["name"]
    assert out["stadium_city"][0] == constants.STADIUMS[stadium_id]["city"]


def test_add_stadium_features() -> None:
    """Stadium features include type/altitude with safe defaults."""
    df = pl.DataFrame(
        {
            "stadium_id": ["DEN00", "XXX00"],
            "stadium_roof": ["Outdoors", "Dome"],
            "stadium_surface": ["Grass", "FieldTurf"],
        }
    )

    out = loaders._add_stadium_features(df)

    assert out["stadium_type"].to_list() == ["open", "dome"]
    assert out["stadium_elevation"][0] == 5280.0
    assert out["stadium_elevation"][1] == 0.0
    assert out["stadium_surface"].to_list() == ["grass", "fieldturf"]
    assert out.select(pl.col("stadium_city").is_null().all()).item() is True
    assert out.select(pl.col("stadium_state").is_null().all()).item() is True


def test_load_schedule_transforms(monkeypatch, tmp_path: Path) -> None:
    """Schedule loading applies expected transformations."""

    def fake_load_schedules(seasons):
        """Fake schedule loader for testing."""
        _ = seasons
        return pl.DataFrame(
            {
                "game_id": ["game1"],
                "season": [2023],
                "game_type": ["REG"],
                "week": [1],
                "gameday": ["2023-09-10"],
                "gametime": ["20:20"],
                "away_team": ["AAA"],
                "home_team": ["BBB"],
                "location": ["Neutral"],
                "spread_line": [-3.5],
            }
        )

    monkeypatch.setattr(loaders.nfl, "load_schedules", fake_load_schedules)
    monkeypatch.setattr(loaders, "normalize_team_column", lambda df, _col: df)

    df = loaders.load_schedule([2023], cache_dir=tmp_path, current_season=2024)

    assert "away_abbr" in df.columns
    assert "home_abbr" in df.columns
    assert df["neutral"][0] == 1
    assert df["home_spread"][0] == 3.5
    assert df["date"].dtype == pl.Date
    assert df["gametime"][0] == "20:20"
    assert "game_datetime" in df.columns


_SCHEDULE_ALTERNATE_TIME_COLUMNS = ("game_time", "kickoff_time", "start_time")

_RAW_SCHEDULE_VALUES: dict[str, object] = {
    "season": 2020,
    "game_type": "REG",
    "week": 1,
    "gameday": "2020-09-13",
    "gametime": "13:00",
    "stadium_id": "DEN00",
    "roof": "outdoors",
    "surface": "grass",
    "away_team": "AAA",
    "home_team": "BBB",
    "location": "Home",
    "spread_line": -3.0,
}


def _raw_schedule(*, game_id: str, season: int = 2020) -> pl.DataFrame:
    """Build one raw nflreadpy schedule row carrying every column the loader selects.

    The alternate kickoff-time columns are left out, as nflverse publishes only `gametime`.

    Returns:
        Raw schedule frame for a single game

    """
    columns = [
        column
        for column in constants.NFLREADPY_SCHEDULE_COLUMNS
        if column not in _SCHEDULE_ALTERNATE_TIME_COLUMNS
    ]
    values = {**dict.fromkeys(columns, 1), **_RAW_SCHEDULE_VALUES}
    row = {column: [values[column]] for column in columns}
    return pl.DataFrame({**row, "game_id": [game_id], "season": [season]})


def _fail_load_schedules(*_args: object, **_kwargs: object) -> pl.DataFrame:
    """Fail if nflreadpy is called."""
    msg = "nflreadpy schedule load should not be called"
    raise AssertionError(msg)


def test_load_schedule_uses_cache_for_historical_seasons(monkeypatch, tmp_path: Path) -> None:
    """A cache file with every column the loader produces is reused, even an older one."""
    cached = loaders._prepare_schedule(_raw_schedule(game_id="cached_game"))
    cached.write_parquet(tmp_path / "schedule_2020.parquet")
    monkeypatch.setattr(loaders.nfl, "load_schedules", _fail_load_schedules)

    df = loaders.load_schedule([2020], cache_dir=tmp_path, current_season=2024)

    assert df["game_id"][0] == "cached_game"


def test_load_schedule_refreshes_current_season(monkeypatch, tmp_path: Path) -> None:
    """Current season schedules are refreshed even when cached."""
    cached = pl.DataFrame(
        {
            "game_id": ["old_game"],
            "season": [2024],
            "game_type": ["REG"],
            "week": [1],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
        }
    )
    cache_path = tmp_path / "schedule_2024.parquet"
    cached.write_parquet(cache_path)

    def fake_load_schedules(seasons):
        """Return a fresh schedule payload."""
        _ = seasons
        return pl.DataFrame(
            {
                "game_id": ["fresh_game"],
                "season": [2024],
                "game_type": ["REG"],
                "week": [1],
                "gameday": ["2024-09-10"],
                "away_team": ["AAA"],
                "home_team": ["BBB"],
                "location": ["Home"],
                "spread_line": [-2.5],
            }
        )

    monkeypatch.setattr(loaders.nfl, "load_schedules", fake_load_schedules)
    monkeypatch.setattr(loaders, "normalize_team_column", lambda df, _col: df)

    df = loaders.load_schedule([2024], cache_dir=tmp_path, current_season=2024)

    assert df["game_id"][0] == "fresh_game"
    refreshed = pl.read_parquet(cache_path)
    assert refreshed["game_id"][0] == "fresh_game"


def test_schedule_requested_columns_follow_the_preparation() -> None:
    """The requested schedule columns are the prepared names, not the raw nflreadpy names."""
    requested = loaders._schedule_requested_columns()

    prepared = loaders._prepare_schedule(_raw_schedule(game_id="g"))
    assert requested == tuple(prepared.columns)
    renamed = constants.NFLREADPY_SCHEDULE_RENAME
    assert set(renamed.values()) <= set(requested)
    assert not set(renamed) & set(requested)
    assert not set(_SCHEDULE_ALTERNATE_TIME_COLUMNS) & set(requested)
    assert {"gametime", "home_spread", "game_datetime", "stadium_elevation"} <= set(requested)


def test_load_schedule_refetches_a_cache_that_lacks_a_requested_column(
    monkeypatch, tmp_path: Path
) -> None:
    """A historical cache missing a column the loader now produces is a cache miss."""
    stale = loaders._prepare_schedule(_raw_schedule(game_id="stale_game")).drop("home_spread")
    stale.write_parquet(tmp_path / "schedule_2020.parquet")
    monkeypatch.setattr(
        loaders.nfl, "load_schedules", lambda seasons: _raw_schedule(game_id="fresh_game")
    )

    df = loaders.load_schedule([2020], cache_dir=tmp_path, current_season=2024)

    assert df["game_id"].to_list() == ["fresh_game"]
    assert df["home_spread"].to_list() == [3.0]
    assert pl.read_parquet(tmp_path / "schedule_2020.parquet").equals(df)


def test_load_schedule_records_the_requested_columns(monkeypatch, tmp_path: Path) -> None:
    """A schedule cache file records the columns the loader requested when writing it."""
    monkeypatch.setattr(loaders.nfl, "load_schedules", lambda seasons: _raw_schedule(game_id="g"))

    loaders.load_schedule([2020], cache_dir=tmp_path, current_season=2024)

    recorded = loaders._recorded_requested_columns(tmp_path / "schedule_2020.parquet")
    assert recorded == frozenset(loaders._schedule_requested_columns())


def test_load_schedule_refetches_when_the_selected_columns_grow(
    monkeypatch, tmp_path: Path
) -> None:
    """Selecting another nflreadpy schedule column invalidates older cache files."""
    monkeypatch.setattr(
        loaders.nfl, "load_schedules", lambda seasons: _raw_schedule(game_id="old_game")
    )
    loaders.load_schedule([2020], cache_dir=tmp_path, current_season=2024)

    grown = [*constants.NFLREADPY_SCHEDULE_COLUMNS, "referee"]
    monkeypatch.setattr(constants, "NFLREADPY_SCHEDULE_COLUMNS", grown)
    monkeypatch.setattr(
        loaders.nfl,
        "load_schedules",
        lambda seasons: _raw_schedule(game_id="new_game").with_columns(referee=pl.lit("Ref")),
    )

    df = loaders.load_schedule([2020], cache_dir=tmp_path, current_season=2024)

    assert df["game_id"].to_list() == ["new_game"]
    assert df["referee"].to_list() == ["Ref"]


def test_load_schedule_refetches_when_a_rename_changes(monkeypatch, tmp_path: Path) -> None:
    """Renaming a selected schedule column invalidates older cache files."""
    monkeypatch.setattr(
        loaders.nfl, "load_schedules", lambda seasons: _raw_schedule(game_id="old_game")
    )
    loaders.load_schedule([2020], cache_dir=tmp_path, current_season=2024)

    renamed = {**constants.NFLREADPY_SCHEDULE_RENAME, "home_coach": "home_head_coach"}
    monkeypatch.setattr(constants, "NFLREADPY_SCHEDULE_RENAME", renamed)
    monkeypatch.setattr(
        loaders.nfl, "load_schedules", lambda seasons: _raw_schedule(game_id="new_game")
    )

    df = loaders.load_schedule([2020], cache_dir=tmp_path, current_season=2024)

    assert df["game_id"].to_list() == ["new_game"]
    assert "home_head_coach" in df.columns


def test_load_schedule_reuses_a_cache_whose_source_lacked_columns(
    monkeypatch, tmp_path: Path
) -> None:
    """A column the source never published does not force a refetch on every run."""
    sparse = _raw_schedule(game_id="sparse_game").drop("spread_line", "away_coach")
    monkeypatch.setattr(loaders.nfl, "load_schedules", lambda seasons: sparse)
    first = loaders.load_schedule([2020], cache_dir=tmp_path, current_season=2024)
    monkeypatch.setattr(loaders.nfl, "load_schedules", _fail_load_schedules)

    second = loaders.load_schedule([2020], cache_dir=tmp_path, current_season=2024)

    assert second.equals(first)
    assert "home_spread" not in second.columns


def test_load_team_stats_combines(monkeypatch, tmp_path: Path) -> None:
    """Team stats loading combines related columns correctly."""

    def fake_load_team_stats(seasons):
        """Fake team stats loader for testing."""
        _ = seasons
        return pl.DataFrame(
            {
                "season": [2023],
                "week": [1],
                "season_type": ["REG"],
                "team": ["AAA"],
                "opponent_team": ["BBB"],
                "sack_fumbles": [1],
                "rushing_fumbles": [0],
                "receiving_fumbles": [1],
            }
        )

    monkeypatch.setattr(loaders.nfl, "load_team_stats", fake_load_team_stats)
    monkeypatch.setattr(loaders, "normalize_team_column", lambda df, _col: df)

    df = loaders.load_team_stats(
        [2023], regular_season_only=True, cache_dir=tmp_path, current_season=2024
    )

    assert "team_abbr" in df.columns
    assert "opponent_abbr" in df.columns
    assert "fumbles" in df.columns
    assert "sack_fumbles" not in df.columns


def _raw_team_stats(*, season: int, team: str) -> pl.DataFrame:
    """Build one raw nflreadpy team-stat row carrying every column the loader maps or reads.

    Returns:
        Raw team-stat frame for a single regular-season team-game

    """
    mapped = list(constants.NFLREADPY_TEAM_STATS_MAPPING)
    targets = set(constants.NFLREADPY_TEAM_STATS_MAPPING.values())
    summed = [part for _, group in loaders._SUMMED_STATS for part in group]
    parts = [part for part in dict.fromkeys([*summed, "def_interceptions"]) if part not in targets]
    return pl.DataFrame(
        {
            "season": [season],
            "week": [1],
            "season_type": ["REG"],
            "team": [team],
            "opponent_team": ["BBB"],
            **{column: [1] for column in [*mapped, *parts]},
        }
    )


def test_load_team_stats_uses_cache_for_historical_seasons(monkeypatch, tmp_path: Path) -> None:
    """A cache file with every column the loader produces is reused, even an older one."""
    cached = loaders._prepare_team_stats(
        _raw_team_stats(season=2021, team="AAA"), regular_season_only=True
    )
    cache_path = tmp_path / "team_stats_2021_reg.parquet"
    cached.write_parquet(cache_path)

    def fail_load_team_stats(*_args, **_kwargs):
        """Fail if nflreadpy is called."""
        msg = "nflreadpy team stats load should not be called"
        raise AssertionError(msg)

    monkeypatch.setattr(loaders.nfl, "load_team_stats", fail_load_team_stats)

    df = loaders.load_team_stats(
        [2021], regular_season_only=True, cache_dir=tmp_path, current_season=2024
    )

    assert df["team_abbr"][0] == "AAA"


def test_load_team_stats_refreshes_current_season(monkeypatch, tmp_path: Path) -> None:
    """Current season team stats are refreshed even when cached."""
    cached = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "team_abbr": ["OLD"],
            "opponent_abbr": ["BBB"],
            "fumbles": [0],
        }
    )
    cache_path = tmp_path / "team_stats_2024_reg.parquet"
    cached.write_parquet(cache_path)

    def fake_load_team_stats(seasons):
        """Return fresh team stats payload."""
        _ = seasons
        return pl.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "season_type": ["REG"],
                "team": ["NEW"],
                "opponent_team": ["BBB"],
                "sack_fumbles": [1],
                "rushing_fumbles": [0],
                "receiving_fumbles": [0],
            }
        )

    monkeypatch.setattr(loaders.nfl, "load_team_stats", fake_load_team_stats)
    monkeypatch.setattr(loaders, "normalize_team_column", lambda df, _col: df)

    df = loaders.load_team_stats(
        [2024], regular_season_only=True, cache_dir=tmp_path, current_season=2024
    )

    assert df["team_abbr"][0] == "NEW"


def test_load_team_stats_handles_unavailable_current_season(monkeypatch, tmp_path: Path) -> None:
    """Current-season team stats should degrade gracefully when nflreadpy has no file yet."""

    def fail_load_team_stats(*_args, **_kwargs):
        """Simulate nflreadpy not publishing the current season stats parquet yet."""
        msg = "404 Client Error: stats_team_week_2026.parquet"
        raise ConnectionError(msg)

    monkeypatch.setattr(loaders.nfl, "load_team_stats", fail_load_team_stats)

    df = loaders.load_team_stats(
        [2026], regular_season_only=True, cache_dir=tmp_path, current_season=2026
    )

    assert df.is_empty()


def test_load_team_stats_uses_cached_fallback_when_current_refresh_fails(
    monkeypatch, tmp_path: Path
) -> None:
    """Current-season team stats should reuse cache when refresh fails after a prior run."""
    cached = pl.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "team_abbr": ["AAA"],
            "opponent_abbr": ["BBB"],
            "fumbles": [1],
        }
    )
    cache_path = tmp_path / "team_stats_2026_reg.parquet"
    cached.write_parquet(cache_path)

    def fail_load_team_stats(*_args, **_kwargs):
        """Simulate nflreadpy not publishing the current season stats parquet yet."""
        msg = "404 Client Error: stats_team_week_2026.parquet"
        raise ConnectionError(msg)

    monkeypatch.setattr(loaders.nfl, "load_team_stats", fail_load_team_stats)

    df = loaders.load_team_stats(
        [2026], regular_season_only=True, cache_dir=tmp_path, current_season=2026
    )

    assert df["team_abbr"][0] == "AAA"


def test_load_team_stats_reraises_historical_download_failures(monkeypatch, tmp_path: Path) -> None:
    """Historical-season download failures should still surface instead of being hidden."""

    def fail_load_team_stats(*_args, **_kwargs):
        """Simulate a broken historical load."""
        msg = "historical load failed"
        raise ConnectionError(msg)

    monkeypatch.setattr(loaders.nfl, "load_team_stats", fail_load_team_stats)

    with pytest.raises(ConnectionError, match="historical load failed"):
        loaders.load_team_stats(
            [2024], regular_season_only=True, cache_dir=tmp_path, current_season=2026
        )


def _fail_load_team_stats(*_args: object, **_kwargs: object) -> pl.DataFrame:
    """Fail if nflreadpy is called."""
    msg = "nflreadpy team stats load should not be called"
    raise AssertionError(msg)


def test_load_team_stats_refetches_a_cache_that_lacks_a_produced_column(
    monkeypatch, tmp_path: Path
) -> None:
    """A historical cache missing a column the loader now produces is a cache miss."""
    stale = pl.DataFrame(
        {"season": [2021], "week": [1], "team_abbr": ["OLD"], "opponent_abbr": ["BBB"]}
    )
    stale.write_parquet(tmp_path / "team_stats_2021_reg.parquet")
    monkeypatch.setattr(
        loaders.nfl,
        "load_team_stats",
        lambda seasons: _raw_team_stats(season=seasons[0], team="NEW"),
    )

    df = loaders.load_team_stats(
        [2021], regular_season_only=True, cache_dir=tmp_path, current_season=2024
    )

    assert df["team_abbr"].to_list() == ["NEW"]
    assert "pass_yards" in df.columns
    assert pl.read_parquet(tmp_path / "team_stats_2021_reg.parquet").equals(df)


@pytest.mark.parametrize("derived", ["total_yards", "turnover_margin"])
def test_load_team_stats_refetches_a_cache_that_lacks_a_derived_column(
    monkeypatch, tmp_path: Path, derived: str
) -> None:
    """A historical cache missing a column the stat combination derives is a cache miss."""
    stale = loaders._prepare_team_stats(
        _raw_team_stats(season=2021, team="OLD"), regular_season_only=True
    ).drop(derived)
    stale.write_parquet(tmp_path / "team_stats_2021_reg.parquet")
    monkeypatch.setattr(
        loaders.nfl,
        "load_team_stats",
        lambda seasons: _raw_team_stats(season=seasons[0], team="NEW"),
    )

    df = loaders.load_team_stats(
        [2021], regular_season_only=True, cache_dir=tmp_path, current_season=2024
    )

    assert df["team_abbr"].to_list() == ["NEW"]
    assert derived in df.columns


def test_load_team_stats_refetches_when_the_mapping_grows(monkeypatch, tmp_path: Path) -> None:
    """Adding a renamed column to the team-stat mapping invalidates older cache files."""
    monkeypatch.setattr(
        loaders.nfl, "load_team_stats", lambda seasons: _raw_team_stats(season=seasons[0], team="A")
    )
    loaders.load_team_stats(
        [2021], regular_season_only=True, cache_dir=tmp_path, current_season=2024
    )

    grown = {**constants.NFLREADPY_TEAM_STATS_MAPPING, "penalty_yards": "penalty_yards_lost"}
    monkeypatch.setattr(constants, "NFLREADPY_TEAM_STATS_MAPPING", grown)
    monkeypatch.setattr(
        loaders.nfl,
        "load_team_stats",
        lambda seasons: _raw_team_stats(season=seasons[0], team="B").with_columns(
            penalty_yards=pl.lit(40)
        ),
    )

    df = loaders.load_team_stats(
        [2021], regular_season_only=True, cache_dir=tmp_path, current_season=2024
    )

    assert df["team_abbr"].to_list() == ["B"]
    assert df["penalty_yards_lost"].to_list() == [40]


def test_load_team_stats_reuses_a_cache_whose_source_lacked_columns(
    monkeypatch, tmp_path: Path
) -> None:
    """A column the source never published does not force a refetch on every run."""
    sparse = pl.DataFrame(
        {
            "season": [2021],
            "week": [1],
            "season_type": ["REG"],
            "team": ["AAA"],
            "opponent_team": ["BBB"],
            "passing_yards": [250],
        }
    )
    monkeypatch.setattr(loaders.nfl, "load_team_stats", lambda **_kwargs: sparse)
    first = loaders.load_team_stats(
        [2021], regular_season_only=True, cache_dir=tmp_path, current_season=2024
    )
    monkeypatch.setattr(loaders.nfl, "load_team_stats", _fail_load_team_stats)

    second = loaders.load_team_stats(
        [2021], regular_season_only=True, cache_dir=tmp_path, current_season=2024
    )

    assert second.equals(first)


def test_add_scoring_data_to_team_stats() -> None:
    """Schedule scores are joined into per-team stats."""
    team_stats = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "team_abbr": ["AAA", "BBB"],
        }
    )
    schedule = pl.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
            "away_score": [17],
            "home_score": [10],
        }
    )

    out = loaders.add_scoring_data_to_team_stats(team_stats, schedule)

    assert "points_scored" in out.columns
    assert "points_allowed" in out.columns


def test_combine_stats_turnovers_and_yards() -> None:
    """Combining stats computes turnovers and total yards correctly."""
    df = pl.DataFrame(
        {
            "sack_fumbles": [1],
            "rushing_fumbles": [2],
            "receiving_fumbles": [0],
            "sack_fumbles_lost": [1],
            "rushing_fumbles_lost": [1],
            "receiving_fumbles_lost": [0],
            "passing_first_downs": [5],
            "rushing_first_downs": [3],
            "passing_2pt_conversions": [1],
            "rushing_2pt_conversions": [0],
            "receiving_2pt_conversions": [1],
            "fumble_recovery_own": [2],
            "fumble_recovery_opp": [1],
            "def_interceptions": [1],
            "interceptions_thrown": [2],
            "pass_yards": [250],
            "rush_yards": [100],
            "yards_lost_from_sacks": [10],
        }
    )

    out = loaders.combine_stats(df)

    assert "fumbles" in out.columns
    assert "fumbles_lost" in out.columns
    assert "first_downs" in out.columns
    assert "2pt_conversions" in out.columns
    assert "fumble_recoveries" in out.columns
    assert "turnover_margin" in out.columns
    assert "total_yards" in out.columns


@pytest.mark.parametrize(
    ("sack_yards", "expected"),
    [({}, 300), ({"yards_lost_from_sacks": [15]}, 285)],
    ids=["without-sack-yards", "with-sack-yards"],
)
def test_combine_stats_total_yards_subtracts_sack_yards_only_when_present(
    sack_yards: dict[str, list[int]], expected: int
) -> None:
    """Total yards need only passing and rushing yards; sack yards are subtracted if given."""
    df = pl.DataFrame({"pass_yards": [200], "rush_yards": [100], **sack_yards})

    out = loaders.combine_stats(df)

    assert out["total_yards"].to_list() == [expected]


def test_add_per_game_opponent_stats() -> None:
    """Per-game opponent stats are added correctly."""
    df = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "team_abbr": ["AAA", "BBB"],
            "opponent_abbr": ["BBB", "AAA"],
            "pass_yards": [200, 150],
        }
    )

    out = loaders.add_per_game_opponent_stats(df)

    assert "opponent_pass_yards" in out.columns
    assert out.filter(pl.col("team_abbr") == "AAA")["opponent_pass_yards"][0] == 150


def test_add_per_game_opponent_stats_keeps_the_times_sacked_intermediate() -> None:
    """The opponent's times sacked is mirrored for the plays-allowed denominator only.

    `opponent_points_per_play` divides `points_allowed` by the opponent's pass attempts,
    rush attempts and times sacked, so that one excluded mirror must still be built per game;
    `opponent_def_sacks` stays out because nothing downstream reads it.
    """
    df = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "team_abbr": ["AAA", "BBB"],
            "opponent_abbr": ["BBB", "AAA"],
            "times_sacked": [2, 5],
            "def_sacks": [5, 2],
            "points_allowed": [10, 20],
        }
    )

    out = loaders.add_per_game_opponent_stats(df)

    assert "opponent_times_sacked" in out.columns
    assert "opponent_def_sacks" not in out.columns
    assert "opponent_points_allowed" not in out.columns
    assert out.filter(pl.col("team_abbr") == "AAA")["opponent_times_sacked"][0] == 5


def test_load_elo_ratings_and_latest(tmp_path: Path, monkeypatch) -> None:
    """Elo ratings loading and latest extraction work as expected."""
    qb_path = tmp_path / "qb_elos.csv"
    qb_path.write_text(
        "season,week,team1,team2,elo1_pre,elo2_pre,qb1,qb2,qb1_value_pre,"
        "qb2_value_pre,qbelo1_pre,qbelo2_pre\n"
        "2023,1.0,AAA,BBB,1500,1450,QB1,QB2,1.5,1.0,1400,1350\n"
        "2023,2.0,AAA,BBB,1510,1440,QB1,QB2,1.6,0.9,1405,1345\n"
    )
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)

    elo = loaders.load_elo_ratings([2023])

    assert "away_abbr" in elo.columns
    assert "home_abbr" in elo.columns
    assert elo["week"].dtype in (pl.Int64, pl.Int32)

    raw = loaders.load_raw_elo_data()
    assert "team1" in raw.columns

    latest = loaders.get_latest_elo_by_team(elo, season=2023)
    assert latest.height == 2


_QB_ELO_HEADER = (
    "date,season,neutral,team1,team2,elo1_pre,elo2_pre,qbelo1_pre,qbelo2_pre,qb1,qb2,"
    "qb1_value_pre,qb2_value_pre,qb1_adj,game_id,week"
)


def _qb_elo_line(date: str, teams: tuple[str, str], **fields: str) -> str:
    """Write one qb_elos.csv row; columns not given are blank, the file's missing value."""
    row = {
        "date": date,
        "season": date[:4],
        "neutral": "0",
        "team1": teams[0],
        "team2": teams[1],
        "elo1_pre": "1500.0",
        "elo2_pre": "1450.0",
    } | fields
    if row.get("week"):
        row.setdefault("game_id", f"{row['season']}_{row['week']}_{teams[1]}_{teams[0]}")
    return ",".join(row.get(name, "") for name in _QB_ELO_HEADER.split(","))


def _write_qb_elos_with_blank_history(tmp_path: Path, *, blank_rows: int = 150) -> Path:
    """Write a qb_elos.csv whose early rows leave the QB, week and Elo columns blank.

    The real file starts in 1920, decades before it carries quarterback ratings or weeks,
    so a reader that guesses types from its first rows sees only blanks there.
    """
    lines = [_QB_ELO_HEADER]
    lines.extend(
        _qb_elo_line("1990-09-09", ("AAA", "BBB"), elo1_pre="", elo2_pre="")
        for _ in range(blank_rows)
    )
    lines.append(
        _qb_elo_line(
            "2023-09-10",
            ("AAA", "BBB"),
            week="1.0",
            elo1_pre="1510.5",
            elo2_pre="1440.25",
            qb1="QB One",
            qb2="QB Two",
            qbelo1_pre="1400.5",
            qbelo2_pre="1350.25",
            qb1_value_pre="1.5",
            qb2_value_pre="-0.75",
            qb1_adj="1.5",
        )
    )
    lines.append(
        _qb_elo_line(
            "2023-09-17",
            ("BBB", "AAA"),
            week="2.0",
            elo1_pre="1445.0",
            elo2_pre="1505.0",
            qb1="QB Two",
            qb2="QB One",
            qbelo1_pre="1352.0",
            qbelo2_pre="1401.0",
            qb1_value_pre="-0.5",
            qb2_value_pre="1.25",
            qb1_adj="-0.5",
        )
    )
    path = tmp_path / "qb_elos.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_load_elo_ratings_types_columns_blank_for_the_first_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Numeric columns stay numeric when the first hundred-plus rows leave them blank."""
    _write_qb_elos_with_blank_history(tmp_path)
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)

    elo = loaders.load_elo_ratings([2023])

    assert elo.schema == pl.Schema(
        {
            "season": pl.Int64,
            "week": pl.Int64,
            "home_abbr": pl.String,
            "away_abbr": pl.String,
            "home_elo_pre": pl.Float64,
            "away_elo_pre": pl.Float64,
            "home_qb": pl.String,
            "away_qb": pl.String,
            "home_qb_value_pre": pl.Float64,
            "away_qb_value_pre": pl.Float64,
            "home_qb_elo_pre": pl.Float64,
            "away_qb_elo_pre": pl.Float64,
        }
    )
    assert elo["week"].to_list() == [1, 2]
    assert elo["home_elo_pre"].to_list() == [1510.5, 1445.0]
    assert elo["away_qb_value_pre"].to_list() == [-0.75, 1.25]


def test_load_raw_elo_data_types_columns_blank_for_the_first_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The raw rows carry numeric weeks and ratings even when early rows are blank."""
    _write_qb_elos_with_blank_history(tmp_path)
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)

    raw = loaders.load_raw_elo_data()

    expected = {
        "date": pl.String,
        "season": pl.Int64,
        "team1": pl.String,
        "team2": pl.String,
        "elo1_pre": pl.Float64,
        "elo2_pre": pl.Float64,
        "qbelo1_pre": pl.Float64,
        "qbelo2_pre": pl.Float64,
        "qb1": pl.String,
        "qb2": pl.String,
        "qb1_value_pre": pl.Float64,
        "qb2_value_pre": pl.Float64,
        "week": pl.Float64,
    }
    assert {name: raw.schema[name] for name in expected} == expected
    dated = raw.filter(pl.col("week").is_not_null())
    assert dated["week"].to_list() == [1.0, 2.0]
    assert dated["qbelo2_pre"].to_list() == [1350.25, 1401.0]


def test_load_elo_ratings_drops_rows_with_a_blank_week(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows without a week are dropped whether or not the first rows carry weeks."""
    lines = [
        _QB_ELO_HEADER,
        _qb_elo_line("2023-09-10", ("AAA", "BBB"), week="1.0"),
        _qb_elo_line("2023-09-12", ("CCC", "DDD")),
        _qb_elo_line("2023-09-17", ("BBB", "AAA"), week="2.0"),
    ]
    (tmp_path / "qb_elos.csv").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)

    elo = loaders.load_elo_ratings([2023])

    assert elo.select("week", "home_abbr").rows() == [(1, "AAA"), (2, "BBB")]


def test_load_raw_elo_data_rejects_text_in_a_numeric_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A text token in a numeric column fails the read and names the column."""
    lines = [
        _QB_ELO_HEADER,
        _qb_elo_line("2023-09-10", ("AAA", "BBB"), week="1.0", qb1_value_pre="NA"),
    ]
    (tmp_path / "qb_elos.csv").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)

    with pytest.raises(pl.exceptions.ComputeError, match="qb1_value_pre"):
        loaders.load_raw_elo_data()


def test_load_raw_elo_data_reads_a_blank_numeric_value_as_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blank numeric value reads as null rather than failing the read."""
    lines = [
        _QB_ELO_HEADER,
        _qb_elo_line("2023-09-10", ("AAA", "BBB"), week="1.0", qb2_value_pre="0.5"),
    ]
    (tmp_path / "qb_elos.csv").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)

    raw = loaders.load_raw_elo_data()

    assert raw.select("qb1_value_pre", "qb2_value_pre").rows() == [(None, 0.5)]


def _pbp_payload(
    *,
    season: int = 2020,
    posteam: str = "AAA",
    season_types: list[str] | None = None,
) -> pl.DataFrame:
    """Build a small raw play-by-play payload for loader tests."""
    types = season_types if season_types is not None else ["REG"]
    rows = len(types)
    return pl.DataFrame(
        {
            "game_id": [f"{season}_{i:02d}" for i in range(rows)],
            "season": [season] * rows,
            "season_type": types,
            "week": list(range(1, rows + 1)),
            "posteam": [posteam] * rows,
            "defteam": ["BBB"] * rows,
            "home_team": [posteam] * rows,
            "away_team": ["BBB"] * rows,
            "play_type": ["pass"] * rows,
            "epa": [0.5] * rows,
            "extra_unused_column": [1] * rows,
        }
    )


def _complete_pbp_cache(*, season: int, posteam: str) -> pl.DataFrame:
    """Build a one-play cached frame carrying every requested play-by-play column.

    Returns:
        Play-by-play frame shaped like a cache file written before the requested
        columns were recorded

    """
    row = {**dict.fromkeys(constants.PBP_COLUMNS), "season": season, "week": 1, "posteam": posteam}
    return pl.DataFrame([row], schema=loaders._PBP_COLUMN_DTYPES)


def test_load_pbp_uses_cache_for_historical_seasons(monkeypatch, tmp_path: Path) -> None:
    """A cache file with every requested column is reused, even an older one."""
    _complete_pbp_cache(season=2020, posteam="CACHED").write_parquet(
        tmp_path / "pbp_2020_reg.parquet"
    )

    def fail_load_pbp(*_args, **_kwargs) -> pl.DataFrame:
        """Fail if nflreadpy is called."""
        msg = "nflreadpy play-by-play load should not be called"
        raise AssertionError(msg)

    monkeypatch.setattr(loaders.nfl, "load_pbp", fail_load_pbp)

    df = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)

    assert df["posteam"].to_list() == ["CACHED"]


def test_load_pbp_writes_cache_on_miss(monkeypatch, tmp_path: Path) -> None:
    """A historical cache miss downloads play-by-play data and writes the parquet cache."""

    def fake_load_pbp(seasons: list[int]) -> pl.DataFrame:
        """Return a raw play-by-play payload for the requested season."""
        assert seasons == [2020]
        return _pbp_payload()

    monkeypatch.setattr(loaders.nfl, "load_pbp", fake_load_pbp)

    df = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)

    cache_path = tmp_path / "pbp_2020_reg.parquet"
    assert cache_path.exists()
    assert "extra_unused_column" not in df.columns
    assert pl.read_parquet(cache_path).equals(df)


def test_load_pbp_force_refresh_ignores_cache(monkeypatch, tmp_path: Path) -> None:
    """force_refresh re-downloads play-by-play data even when a cache exists."""
    stale = pl.DataFrame({"season": [2020], "week": [1], "posteam": ["STALE"]})
    stale.write_parquet(tmp_path / "pbp_2020_reg.parquet")

    def fake_load_pbp(seasons: list[int]) -> pl.DataFrame:
        """Return refreshed play-by-play data."""
        _ = seasons
        return _pbp_payload(posteam="FRESH")

    monkeypatch.setattr(loaders.nfl, "load_pbp", fake_load_pbp)

    df = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024, force_refresh=True)

    assert df["posteam"].to_list() == ["FRESH"]


def _fail_load_pbp(*_args: object, **_kwargs: object) -> pl.DataFrame:
    """Fail if nflreadpy is called."""
    msg = "nflreadpy play-by-play load should not be called"
    raise AssertionError(msg)


def test_load_pbp_refetches_a_cache_that_lacks_a_requested_column(
    monkeypatch, tmp_path: Path
) -> None:
    """A historical cache missing a requested column is a cache miss and is rewritten."""
    stale = pl.DataFrame({"season": [2020], "week": [1], "posteam": ["STALE"]})
    stale.write_parquet(tmp_path / "pbp_2020_reg.parquet")
    monkeypatch.setattr(loaders.nfl, "load_pbp", lambda **_kwargs: _pbp_payload(posteam="FRESH"))

    df = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)

    assert df["posteam"].to_list() == ["FRESH"]
    assert pl.read_parquet(tmp_path / "pbp_2020_reg.parquet").equals(df)


def test_load_pbp_refetches_when_the_requested_columns_grow(monkeypatch, tmp_path: Path) -> None:
    """Growing the requested play-by-play columns invalidates older cache files."""
    monkeypatch.setattr(loaders.nfl, "load_pbp", lambda **_kwargs: _pbp_payload(posteam="OLD"))
    loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)

    monkeypatch.setattr(constants, "PBP_COLUMNS", [*constants.PBP_COLUMNS, "extra_unused_column"])
    monkeypatch.setattr(loaders.nfl, "load_pbp", lambda **_kwargs: _pbp_payload(posteam="NEW"))

    df = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)

    assert df["posteam"].to_list() == ["NEW"]
    assert df["extra_unused_column"].to_list() == [1]


def test_load_pbp_reuses_a_cache_whose_source_lacked_columns(monkeypatch, tmp_path: Path) -> None:
    """A column the source never published does not force a refetch on every run."""
    monkeypatch.setattr(loaders.nfl, "load_pbp", lambda **_kwargs: _pbp_payload(posteam="ONCE"))
    first = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)
    monkeypatch.setattr(loaders.nfl, "load_pbp", _fail_load_pbp)

    second = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)

    assert second.equals(first)


def test_load_pbp_drops_cached_columns_no_longer_requested(monkeypatch, tmp_path: Path) -> None:
    """A cache hit returns only the requested columns, as a fresh download would."""
    grown = [*constants.PBP_COLUMNS, "extra_unused_column"]
    with monkeypatch.context() as patch:
        patch.setattr(constants, "PBP_COLUMNS", grown)
        patch.setattr(loaders.nfl, "load_pbp", lambda **_kwargs: _pbp_payload())
        loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)
    monkeypatch.setattr(loaders.nfl, "load_pbp", _fail_load_pbp)

    df = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)

    assert "extra_unused_column" not in df.columns
    assert df["posteam"].to_list() == ["AAA"]


@pytest.mark.parametrize(
    "record",
    [
        "not json",
        json.dumps(dict.fromkeys(constants.PBP_COLUMNS, 1)),
        json.dumps([*constants.PBP_COLUMNS, 1]),
    ],
    ids=["not-json", "object", "non-string-entry"],
)
def test_load_pbp_refetches_a_cache_with_an_unreadable_column_record(
    monkeypatch, tmp_path: Path, record: str
) -> None:
    """A recorded request that is not a JSON list of column names makes the file unreadable."""
    stale = pl.DataFrame({"season": [2020], "week": [1], "posteam": ["STALE"]})
    stale.write_parquet(
        tmp_path / "pbp_2020_reg.parquet",
        metadata={loaders._REQUESTED_COLUMNS_METADATA_KEY: record},
    )
    monkeypatch.setattr(loaders.nfl, "load_pbp", lambda **_kwargs: _pbp_payload(posteam="FRESH"))

    df = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)

    assert df["posteam"].to_list() == ["FRESH"]


def test_load_pbp_current_season_failure_uses_cache(monkeypatch, tmp_path: Path) -> None:
    """A current-season connection failure falls back to cached play-by-play data."""
    cached = pl.DataFrame({"season": [2024], "week": [1], "posteam": ["CACHED"]})
    cached.write_parquet(tmp_path / "pbp_2024_reg.parquet")

    def failing_load_pbp(seasons: list[int]) -> pl.DataFrame:
        """Simulate an nflreadpy outage."""
        _ = seasons
        msg = "boom"
        raise ConnectionError(msg)

    monkeypatch.setattr(loaders.nfl, "load_pbp", failing_load_pbp)

    df = loaders.load_pbp([2024], cache_dir=tmp_path, current_season=2024)

    assert df["posteam"].to_list() == ["CACHED"]


def test_load_pbp_current_season_failure_without_cache(monkeypatch, tmp_path: Path) -> None:
    """A current-season failure without a cache returns a typed empty frame."""

    def failing_load_pbp(seasons: list[int]) -> pl.DataFrame:
        """Simulate an nflreadpy outage."""
        _ = seasons
        msg = "boom"
        raise ConnectionError(msg)

    monkeypatch.setattr(loaders.nfl, "load_pbp", failing_load_pbp)

    df = loaders.load_pbp([2024], cache_dir=tmp_path, current_season=2024)

    assert df.height == 0
    assert df.columns == list(constants.PBP_COLUMNS)
    assert df.schema["game_id"] == pl.Utf8
    assert df.schema["season"] == pl.Int64
    assert df.schema["week"] == pl.Int64
    assert df.schema["epa"] == pl.Float64


def test_load_pbp_historical_failure_raises(monkeypatch, tmp_path: Path) -> None:
    """A historical connection failure is treated as a bug and re-raised."""

    def failing_load_pbp(seasons: list[int]) -> pl.DataFrame:
        """Simulate an nflreadpy outage."""
        _ = seasons
        msg = "boom"
        raise ConnectionError(msg)

    monkeypatch.setattr(loaders.nfl, "load_pbp", failing_load_pbp)

    with pytest.raises(ConnectionError):
        loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)


def test_load_pbp_tolerates_missing_optional_columns(monkeypatch, tmp_path: Path) -> None:
    """Seasons missing optional play-by-play columns load with only the present ones."""

    def fake_load_pbp(seasons: list[int]) -> pl.DataFrame:
        """Return a sparse play-by-play payload."""
        _ = seasons
        return pl.DataFrame(
            {
                "game_id": ["2020_01"],
                "season": [2020],
                "week": [1],
                "posteam": ["AAA"],
                "epa": [0.25],
            }
        )

    monkeypatch.setattr(loaders.nfl, "load_pbp", fake_load_pbp)

    df = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)

    assert df.columns == ["game_id", "season", "week", "posteam", "epa"]
    assert df.height == 1


def test_load_pbp_normalizes_team_aliases(monkeypatch, tmp_path: Path) -> None:
    """Legacy team abbreviations are normalized to canonical values."""

    def fake_load_pbp(seasons: list[int]) -> pl.DataFrame:
        """Return play-by-play rows using relocated-franchise abbreviations."""
        _ = seasons
        return pl.DataFrame(
            {
                "game_id": ["2005_01"],
                "season": [2005],
                "season_type": ["REG"],
                "week": [1],
                "posteam": ["OAK"],
                "defteam": ["SD"],
                "home_team": ["STL"],
                "away_team": ["OAK"],
                "td_team": ["STL"],
                "penalty_team": ["SD"],
            }
        )

    monkeypatch.setattr(loaders.nfl, "load_pbp", fake_load_pbp)

    df = loaders.load_pbp([2005], cache_dir=tmp_path, current_season=2024)

    assert df["posteam"][0] == constants.ALIAS_TO_CANONICAL["OAK"]
    assert df["defteam"][0] == constants.ALIAS_TO_CANONICAL["SD"]
    assert df["home_team"][0] == constants.ALIAS_TO_CANONICAL["STL"]
    assert df["away_team"][0] == constants.ALIAS_TO_CANONICAL["OAK"]
    assert df["td_team"][0] == constants.ALIAS_TO_CANONICAL["STL"]
    assert df["penalty_team"][0] == constants.ALIAS_TO_CANONICAL["SD"]


def test_load_pbp_regular_season_filter(monkeypatch, tmp_path: Path) -> None:
    """The regular-season filter drops postseason plays and switches the cache filename."""

    def fake_load_pbp(seasons: list[int]) -> pl.DataFrame:
        """Return one regular season and one postseason play."""
        _ = seasons
        return _pbp_payload(season_types=["REG", "POST"])

    monkeypatch.setattr(loaders.nfl, "load_pbp", fake_load_pbp)

    reg_only = loaders.load_pbp([2020], cache_dir=tmp_path, current_season=2024)
    assert reg_only["season_type"].to_list() == ["REG"]
    assert (tmp_path / "pbp_2020_reg.parquet").exists()

    all_types = loaders.load_pbp(
        [2020], cache_dir=tmp_path, current_season=2024, regular_season_only=False
    )
    assert all_types["season_type"].to_list() == ["REG", "POST"]
    assert (tmp_path / "pbp_2020_all.parquet").exists()


def test_load_pbp_empty_seasons_returns_empty_frame() -> None:
    """An empty season list short-circuits to an empty frame."""
    assert loaders.load_pbp([]).height == 0


def test_load_pbp_current_season_out_of_range_is_non_fatal(monkeypatch, tmp_path: Path) -> None:
    """A current season nflreadpy refuses to serve degrades instead of failing the run.

    Before kickoff the current season has no play-by-play at all, and nflreadpy signals
    that with a ValueError rather than a connection failure. The pipeline must still run.
    """

    def out_of_range_load_pbp(seasons: list[int]) -> pl.DataFrame:
        msg = "Season must be between 1999 and 2025"
        raise ValueError(msg)

    monkeypatch.setattr(loaders.nfl, "load_pbp", out_of_range_load_pbp)

    df = loaders.load_pbp([2026], cache_dir=tmp_path, current_season=2026)

    assert df.height == 0
    assert "posteam" in df.columns


def test_load_pbp_historical_out_of_range_still_raises(monkeypatch, tmp_path: Path) -> None:
    """A historical season that cannot be served is a real failure and is not swallowed."""

    def out_of_range_load_pbp(seasons: list[int]) -> pl.DataFrame:
        msg = "Season must be between 1999 and 2025"
        raise ValueError(msg)

    monkeypatch.setattr(loaders.nfl, "load_pbp", out_of_range_load_pbp)

    with pytest.raises(ValueError, match="Season must be between"):
        loaders.load_pbp([2015], cache_dir=tmp_path, current_season=2026)


def _coverage_schedule() -> pl.DataFrame:
    """Build a three-week schedule whose last game has not been played yet.

    Returns:
        Schedule frame with two completed games and one scheduled game

    """
    return pl.DataFrame(
        {
            "season": [2001, 2001, 2001],
            "week": [1, 2, 3],
            "game_type": ["REG", "REG", "REG"],
            "home_abbr": ["JAX", "CLE", "JAX"],
            "away_abbr": ["CLE", "JAX", "CLE"],
            "home_score": [20, 17, None],
            "away_score": [10, 14, None],
        }
    )


def _coverage_team_stats() -> pl.DataFrame:
    """Build team stats that are missing the home team's first-week row.

    Returns:
        Team-stat frame with three of the four completed team-games

    """
    return pl.DataFrame(
        {
            "season": [2001, 2001, 2001],
            "week": [1, 2, 2],
            "team_abbr": ["CLE", "CLE", "JAX"],
            "opponent_abbr": ["JAX", "JAX", "CLE"],
            "season_type": ["REG", "REG", "REG"],
            "penalties": [5.0, 7.0, 9.0],
            "penalty_yards": [40.0, 60.0, 81.0],
        }
    )


def test_build_team_game_skeleton_has_two_rows_per_completed_game() -> None:
    """The skeleton carries exactly two team rows for every completed game."""
    skeleton = loaders.build_team_game_skeleton(_coverage_schedule())

    assert skeleton.height == 4
    per_game = skeleton.group_by(["season", "week"]).len().sort("week")
    assert per_game["len"].to_list() == [2, 2]
    assert sorted(skeleton.filter(pl.col("week") == 1)["team_abbr"].to_list()) == ["CLE", "JAX"]
    jax_week1 = skeleton.filter((pl.col("week") == 1) & (pl.col("team_abbr") == "JAX"))
    assert jax_week1["opponent_abbr"].to_list() == ["CLE"]
    assert jax_week1["season_type"].to_list() == ["REG"]


def test_attach_team_stats_to_schedule_keeps_the_missing_team_game() -> None:
    """A team-game absent from team stats survives as a row with null stats."""
    team_stats = _coverage_team_stats()

    attached = loaders.attach_team_stats_to_schedule(team_stats, _coverage_schedule())

    assert set(attached.columns) == set(team_stats.columns)
    assert attached.height == 4
    jax = attached.filter(pl.col("team_abbr") == "JAX").sort("week")
    assert jax["week"].to_list() == [1, 2]
    assert jax["penalties"].to_list() == [None, 9.0]
    assert jax["opponent_abbr"].to_list() == ["CLE", "CLE"]


def test_attach_team_stats_to_schedule_counts_games_from_the_schedule() -> None:
    """Season-to-date counts follow the schedule and rates stay ratios of sums."""
    attached = loaders.attach_team_stats_to_schedule(_coverage_team_stats(), _coverage_schedule())

    agg = teamrankings.aggregate_team_stats_to_week(attached, target_week=3, season=2001)
    jax = agg.filter(pl.col("team_abbr") == "JAX")

    assert jax["games_played"].to_list() == [2]
    assert jax["penalty_yards_per_penalty"][0] == pytest.approx(81.0 / 9.0)


def test_attach_team_stats_to_schedule_warns_about_coverage_gaps(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every season and team whose stat rows differ from the schedule is logged."""
    caplog.set_level(logging.WARNING)

    loaders.attach_team_stats_to_schedule(_coverage_team_stats(), _coverage_schedule())

    messages = [record.message for record in caplog.records if "coverage gap" in record.message]
    gap_messages = [message for message in messages if "JAX" in message]
    assert gap_messages, messages
    assert "2001" in gap_messages[0]
    assert "1" in gap_messages[0]
    assert "2" in gap_messages[0]
    # The fully covered team is not reported as a gap.
    assert not [message for message in messages if "CLE" in message]


def test_attach_team_stats_to_schedule_keeps_rows_the_schedule_does_not_cover() -> None:
    """Stat rows from seasons the schedule omits are never dropped."""
    team_stats = pl.concat(
        [
            _coverage_team_stats(),
            pl.DataFrame(
                {
                    "season": [2000],
                    "week": [5],
                    "team_abbr": ["CLE"],
                    "opponent_abbr": ["JAX"],
                    "season_type": ["REG"],
                    "penalties": [4.0],
                    "penalty_yards": [30.0],
                }
            ),
        ]
    )

    attached = loaders.attach_team_stats_to_schedule(team_stats, _coverage_schedule())

    assert attached.filter(pl.col("season") == 2000).height == 1
    assert attached.height == 5


def test_attach_team_stats_to_schedule_keeps_a_stat_row_the_schedule_omits() -> None:
    """A stat row with no scheduled counterpart is kept rather than dropped."""
    team_stats = pl.concat(
        [
            _coverage_team_stats(),
            pl.DataFrame(
                {
                    "season": [2001],
                    "week": [9],
                    "team_abbr": ["CLE"],
                    "opponent_abbr": ["JAX"],
                    "season_type": ["REG"],
                    "penalties": [3.0],
                    "penalty_yards": [25.0],
                }
            ),
        ]
    )

    attached = loaders.attach_team_stats_to_schedule(team_stats, _coverage_schedule())

    assert attached.filter(pl.col("week") == 9).height == 1
    assert attached.height == 5


def _collapsed_row_schedule() -> pl.DataFrame:
    """Build a two-game schedule whose first game has one team-stats row.

    Returns:
        Schedule frame with two completed games

    """
    return pl.DataFrame(
        {
            "season": [2001, 2001],
            "week": [1, 2],
            "game_type": ["REG", "REG"],
            "home_abbr": ["JAX", "CLE"],
            "away_abbr": ["CLE", "JAX"],
            "home_score": [20, 17],
            "away_score": [10, 14],
        }
    )


def _collapsed_row_team_stats() -> pl.DataFrame:
    """Build team stats whose first-week row carries both teams' production.

    Returns:
        Team-stat frame with a two-team row in week 1 and a clean week 2

    """
    return pl.DataFrame(
        {
            "season": [2001, 2001, 2001],
            "week": [1, 2, 2],
            "team_abbr": ["CLE", "CLE", "JAX"],
            "opponent_abbr": ["JAX", "JAX", "CLE"],
            "season_type": ["REG", "REG", "REG"],
            # Week 1 holds both teams' penalties because the source dropped the other row.
            "penalties": [12.0, 7.0, 9.0],
            "penalty_yards": [100.0, 60.0, 81.0],
        }
    )


def test_attach_team_stats_to_schedule_nulls_a_box_score_that_covers_both_teams() -> None:
    """The surviving row of a one-sided game loses a box score it cannot own."""
    team_stats = _collapsed_row_team_stats()

    attached = loaders.attach_team_stats_to_schedule(team_stats, _collapsed_row_schedule())

    assert set(attached.columns) == set(team_stats.columns)
    week_one = attached.filter(pl.col("week") == 1).sort("team_abbr")
    assert week_one["team_abbr"].to_list() == ["CLE", "JAX"]
    # Neither side of the collapsed game claims the two-team totals.
    assert week_one["penalties"].to_list() == [None, None]
    assert week_one["penalty_yards"].to_list() == [None, None]
    # Identity survives the repair.
    assert week_one["opponent_abbr"].to_list() == ["JAX", "CLE"]
    assert week_one["season_type"].to_list() == ["REG", "REG"]
    # A game both teams are covered for is untouched.
    week_two = attached.filter(pl.col("week") == 2).sort("team_abbr")
    assert week_two["penalties"].to_list() == [7.0, 9.0]


def test_attach_team_stats_to_schedule_leaves_a_game_neither_team_covers() -> None:
    """A game missing from the source on both sides has no box score to repair."""
    schedule = _collapsed_row_schedule()
    team_stats = _collapsed_row_team_stats().filter(pl.col("week") == 2)

    attached = loaders.attach_team_stats_to_schedule(team_stats, schedule)

    week_one = attached.filter(pl.col("week") == 1)
    assert week_one.height == 2
    assert week_one["penalties"].to_list() == [None, None]
    week_two = attached.filter(pl.col("week") == 2).sort("team_abbr")
    assert week_two["penalties"].to_list() == [7.0, 9.0]


def test_attach_team_stats_to_schedule_logs_the_repaired_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The repaired season, week and team are named, with how many rows were repaired."""
    caplog.set_level(logging.WARNING)

    loaders.attach_team_stats_to_schedule(_collapsed_row_team_stats(), _collapsed_row_schedule())

    messages = [record.message for record in caplog.records]
    repaired = [message for message in messages if "CLE" in message and "2001" in message]
    assert repaired, messages
    assert any("1" in message for message in repaired)
    assert any(message.count("1") and "row" in message for message in messages)


def test_attach_team_stats_to_schedule_without_a_schedule_is_a_no_op() -> None:
    """With no schedule rows the team stats are returned untouched."""
    team_stats = _coverage_team_stats()

    attached = loaders.attach_team_stats_to_schedule(team_stats, pl.DataFrame())

    assert attached.equals(team_stats)
