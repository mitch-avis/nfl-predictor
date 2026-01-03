"""Tests for Polars loader helpers."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.polars import loaders

# pylint: disable=protected-access


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
    stadium_id = next(iter(constants.STADIUM_LOCATIONS.keys()))
    df = pl.DataFrame({"stadium_id": [stadium_id]})

    out = loaders._add_stadium_location(df)

    assert "stadium_city" in out.columns
    assert "stadium_state" in out.columns
    assert out["stadium_city"][0] == constants.STADIUM_LOCATIONS[stadium_id]["city"]


def test_load_schedule_transforms(monkeypatch) -> None:
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
                "away_team": ["AAA"],
                "home_team": ["BBB"],
                "location": ["Neutral"],
                "spread_line": [-3.5],
            }
        )

    monkeypatch.setattr(loaders.nfl, "load_schedules", fake_load_schedules)
    monkeypatch.setattr(loaders, "normalize_team_column", lambda df, _col: df)

    df = loaders.load_schedule([2023])

    assert "away_abbr" in df.columns
    assert "home_abbr" in df.columns
    assert df["neutral"][0] == 1
    assert df["home_spread"][0] == 3.5
    assert df["date"].dtype == pl.Date


def test_load_team_stats_combines(monkeypatch) -> None:
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

    df = loaders.load_team_stats([2023], regular_season_only=True)

    assert "team_abbr" in df.columns
    assert "opponent_abbr" in df.columns
    assert "fumbles" in df.columns
    assert "sack_fumbles" not in df.columns


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


def test_aggregate_pbp_stats() -> None:
    """Play-by-play stats are aggregated correctly."""
    pbp = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "season_type": ["REG", "REG"],
            "posteam": ["AAA", "AAA"],
            "play_type": ["run", "pass"],
            "third_down_converted": [1, 0],
            "third_down_failed": [0, 1],
            "fourth_down_converted": [0, 1],
            "fourth_down_failed": [0, 0],
            "yardline_100": [10, 30],
            "td_team": ["AAA", None],
            "two_point_attempt": [0, 1],
            "two_point_conv_result": [None, "success"],
        }
    )

    out = loaders.aggregate_pbp_stats(pbp, seasons=[2023])

    assert out.height == 1
    assert out["third_down_attempts"][0] == 2
    assert out["fourth_down_attempts"][0] == 1
    assert out["red_zone_plays"][0] == 1


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
