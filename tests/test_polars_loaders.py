"""Tests for Polars loader helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.polars import loaders, teamrankings


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


def test_load_schedule_uses_cache_for_historical_seasons(monkeypatch, tmp_path: Path) -> None:
    """Schedule loader prefers cache for historical seasons."""
    cached = pl.DataFrame(
        {
            "game_id": ["cached_game"],
            "season": [2020],
            "game_type": ["REG"],
            "week": [1],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
        }
    )
    cache_path = tmp_path / "schedule_2020.parquet"
    cached.write_parquet(cache_path)

    def fail_load_schedules(*_args, **_kwargs):
        """Fail if nflreadpy is called."""
        raise AssertionError("nflreadpy schedule load should not be called")

    monkeypatch.setattr(loaders.nfl, "load_schedules", fail_load_schedules)

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


def test_load_team_stats_uses_cache_for_historical_seasons(monkeypatch, tmp_path: Path) -> None:
    """Team stats loader prefers cache for historical seasons."""
    cached = pl.DataFrame(
        {
            "season": [2021],
            "week": [1],
            "team_abbr": ["AAA"],
            "opponent_abbr": ["BBB"],
            "fumbles": [1],
        }
    )
    cache_path = tmp_path / "team_stats_2021_reg.parquet"
    cached.write_parquet(cache_path)

    def fail_load_team_stats(*_args, **_kwargs):
        """Fail if nflreadpy is called."""
        raise AssertionError("nflreadpy team stats load should not be called")

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
        raise ConnectionError("404 Client Error: stats_team_week_2026.parquet")

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
        raise ConnectionError("404 Client Error: stats_team_week_2026.parquet")

    monkeypatch.setattr(loaders.nfl, "load_team_stats", fail_load_team_stats)

    df = loaders.load_team_stats(
        [2026], regular_season_only=True, cache_dir=tmp_path, current_season=2026
    )

    assert df["team_abbr"][0] == "AAA"


def test_load_team_stats_reraises_historical_download_failures(monkeypatch, tmp_path: Path) -> None:
    """Historical-season download failures should still surface instead of being hidden."""

    def fail_load_team_stats(*_args, **_kwargs):
        """Simulate a broken historical load."""
        raise ConnectionError("historical load failed")

    monkeypatch.setattr(loaders.nfl, "load_team_stats", fail_load_team_stats)

    with pytest.raises(ConnectionError, match="historical load failed"):
        loaders.load_team_stats(
            [2024], regular_season_only=True, cache_dir=tmp_path, current_season=2026
        )


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

    monkeypatch.setattr(constants, "DATA_PATH", str(tmp_path))
    elo_df = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": ["1", "2"],
            "team1": ["AAA", "AAA"],
            "team2": ["BBB", "BBB"],
            "elo1_pre": [1500, 1510],
            "elo2_pre": [1450, 1440],
            "qb1": ["QB1", "QB1"],
            "qb2": ["QB2", "QB2"],
            "qb1_value_pre": ["1.5", "1.6"],
            "qb2_value_pre": ["1.0", "0.9"],
            "qbelo1_pre": ["1400", "1405"],
            "qbelo2_pre": ["1350", "1345"],
        }
    )
    monkeypatch.setattr(loaders.pl, "read_csv", lambda _path: elo_df)

    elo = loaders.load_elo_ratings([2023])

    assert "away_abbr" in elo.columns
    assert "home_abbr" in elo.columns
    assert elo["week"].dtype in (pl.Int64, pl.Int32)

    raw = loaders.load_raw_elo_data()
    assert "team1" in raw.columns

    latest = loaders.get_latest_elo_by_team(elo, season=2023)
    assert latest.height == 2


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


def test_load_pbp_uses_cache_for_historical_seasons(monkeypatch, tmp_path: Path) -> None:
    """Play-by-play loader prefers the cache for historical seasons."""
    cached = pl.DataFrame({"season": [2020], "week": [1], "posteam": ["CACHED"]})
    cached.write_parquet(tmp_path / "pbp_2020_reg.parquet")

    def fail_load_pbp(*_args, **_kwargs) -> pl.DataFrame:
        """Fail if nflreadpy is called."""
        raise AssertionError("nflreadpy play-by-play load should not be called")

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


def test_load_pbp_current_season_failure_uses_cache(monkeypatch, tmp_path: Path) -> None:
    """A current-season connection failure falls back to cached play-by-play data."""
    cached = pl.DataFrame({"season": [2024], "week": [1], "posteam": ["CACHED"]})
    cached.write_parquet(tmp_path / "pbp_2024_reg.parquet")

    def failing_load_pbp(seasons: list[int]) -> pl.DataFrame:
        """Simulate an nflreadpy outage."""
        _ = seasons
        raise ConnectionError("boom")

    monkeypatch.setattr(loaders.nfl, "load_pbp", failing_load_pbp)

    df = loaders.load_pbp([2024], cache_dir=tmp_path, current_season=2024)

    assert df["posteam"].to_list() == ["CACHED"]


def test_load_pbp_current_season_failure_without_cache(monkeypatch, tmp_path: Path) -> None:
    """A current-season failure without a cache returns a typed empty frame."""

    def failing_load_pbp(seasons: list[int]) -> pl.DataFrame:
        """Simulate an nflreadpy outage."""
        _ = seasons
        raise ConnectionError("boom")

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
        raise ConnectionError("boom")

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
        raise ValueError("Season must be between 1999 and 2025")

    monkeypatch.setattr(loaders.nfl, "load_pbp", out_of_range_load_pbp)

    df = loaders.load_pbp([2026], cache_dir=tmp_path, current_season=2026)

    assert df.height == 0
    assert "posteam" in df.columns


def test_load_pbp_historical_out_of_range_still_raises(monkeypatch, tmp_path: Path) -> None:
    """A historical season that cannot be served is a real failure and is not swallowed."""

    def out_of_range_load_pbp(seasons: list[int]) -> pl.DataFrame:
        raise ValueError("Season must be between 1999 and 2025")

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
