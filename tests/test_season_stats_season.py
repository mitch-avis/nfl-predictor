"""Tests for one season's tables, assembled as ``nfl-sos-ratings`` writes its data files."""

from __future__ import annotations

import itertools

import polars as pl
import pytest

from nfl_predictor.season_stats import season as season_tables
from nfl_predictor.season_stats.metrics import get_registry
from nfl_predictor.season_stats.pooled_rates import (
    denominator_column,
    is_rate_part,
    numerator_column,
)
from nfl_predictor.season_stats.qb_stats import CPOE_COLUMN
from nfl_predictor.season_stats.sources import SeasonInputs
from nfl_predictor.season_stats.team_stats import add_per_snap_rates
from nfl_predictor.season_stats.wp_bins import compute_qb_wp_bins, compute_team_wp_bins

_TEAMS = ("BUF", "MIA", "NE", "NYJ")
_STRENGTH = {"BUF": 0.08, "MIA": -0.02, "NE": 0.04, "NYJ": -0.10}


def _games() -> list[tuple[int, str, str, str]]:
    """Return (week, game_id, home, away) for a double round-robin."""
    return [
        (week, f"2025_{week:02d}_{away}_{home}", home, away)
        for week, (home, away) in enumerate(itertools.permutations(_TEAMS, 2), start=1)
    ]


def _snaps(week: int, team: str) -> int:
    """Return a team's offensive snaps in one week's game: 56 to 68, different game to game."""
    return 56 + (week * (1 + _TEAMS.index(team))) % 13


def _weekly_df() -> pl.DataFrame:
    """Return team-game rows; snaps differ game to game, so pooled and mean rates differ."""
    rows: list[dict[str, object]] = []
    for week, game_id, home, away in _games():
        margin = round((_STRENGTH[home] - _STRENGTH[away]) * 100) + 2
        for team, opponent, is_home, team_margin in (
            (home, away, True, margin),
            (away, home, False, -margin),
        ):
            points_for = 20 + max(team_margin, 0)
            rows.append(
                {
                    "game_id": game_id,
                    "season": 2025,
                    "season_type": "REG",
                    "week": week,
                    "team": team,
                    "opponent_team": opponent,
                    "is_home": is_home,
                    "points_for": points_for,
                    "points_allowed": points_for - team_margin,
                    "point_margin": team_margin,
                    "win_value": 1.0 if team_margin > 0 else 0.0,
                    "offensive_snaps": _snaps(week, team),
                    "defensive_snaps": _snaps(week, opponent),
                }
            )
    return add_per_snap_rates(pl.DataFrame(rows))


def _qb_df() -> pl.DataFrame:
    """Return one starting passer per team-game."""
    rows: list[dict[str, object]] = [
        {
            "game_id": row["game_id"],
            "week": row["week"],
            "team_abbr": row["team"],
            "qb_id": f"qb-{row['team']}",
            "qb_name": f"{row['team']} Passer",
            "qb_dropbacks": 38,
            "qb_attempts": 35,
            "qb_passing_epa": 3.8,
            "qb_spike_epa": 0.0,
            "qb_epa_per_dropback": 0.1,
            CPOE_COLUMN: 2.0,
            numerator_column(CPOE_COLUMN): 60.0,
            denominator_column(CPOE_COLUMN): 30,
        }
        for row in _weekly_df().iter_rows(named=True)
    ]
    return pl.DataFrame(rows)


def _schedule_df() -> pl.DataFrame:
    """Return the schedule of the synthetic games, every one played."""
    return pl.DataFrame(
        {
            "game_id": [game_id for _, game_id, _, _ in _games()],
            "week": [week for week, _, _, _ in _games()],
            "home_team": [home for _, _, home, _ in _games()],
            "away_team": [away for _, _, _, away in _games()],
            "home_score": [24] * len(_games()),
            "away_score": [17] * len(_games()),
        }
    )


def _pbp() -> pl.DataFrame:
    """Return two plays of the first game: a BUF dropback and a MIA punt, both with a bin."""
    return pl.DataFrame(
        {
            "game_id": ["2025_01_MIA_BUF"] * 2,
            "week": [1, 1],
            "posteam": ["BUF", "MIA"],
            "defteam": ["MIA", "BUF"],
            "wp": [0.6, 0.3],
            "qb_dropback": [1, 0],
            "special": [0, 1],
            "epa": [0.5, -0.2],
            "qb_epa": [0.5, 0.0],
            "passer_player_id": ["qb-BUF", None],
            "passer_player_name": ["B.Passer", None],
        }
    )


def _team_ratings() -> pl.DataFrame:
    """Return a ratings table as the ratings package publishes it, best first."""
    return pl.DataFrame(
        {
            "team": ["BUF", "NE", "MIA", "NYJ"],
            "games_played": [6, 6, 6, 6],
            **{
                column: [3.0, 1.0, -1.0, -3.0]
                for column in season_tables.TEAM_RATING_INPUT_COLUMNS
                if column != "team"
            },
        }
    )


def _qb_ratings() -> pl.DataFrame:
    """Return each passer's rating and faced pass defense."""
    return pl.DataFrame(
        {
            "qb_id": [f"qb-{team}" for team in _TEAMS],
            "adj_qb_epa_per_dropback": [0.2, 0.0, 0.1, -0.1],
            "qb_faced_pass_defense": [0.01, -0.01, 0.0, 0.02],
        }
    )


def _season_games() -> season_tables.SeasonGames:
    """Return the synthetic league's game rows and schedule."""
    return season_tables.SeasonGames(
        team_games=_weekly_df(), qb_games=_qb_df(), schedule=_schedule_df()
    )


def _tables() -> season_tables.SeasonTables:
    """Return the synthetic league's season tables."""
    return season_tables.build_season_tables(
        _season_games(), _pbp(), team_ratings=_team_ratings(), qb_ratings=_qb_ratings()
    )


def test_played_schedule_drops_games_without_final_scores() -> None:
    # Arrange
    schedule = pl.DataFrame(
        {
            "home_team": ["BUF", "NE"],
            "away_team": ["MIA", "NYJ"],
            "home_score": [24, None],
            "away_score": [17, None],
        }
    )

    # Act
    played = season_tables.played_schedule(schedule)

    # Assert
    assert played.select("home_team", "away_team").rows() == [("BUF", "MIA")]


def test_played_schedule_keeps_a_schedule_without_score_columns() -> None:
    # Arrange
    schedule = _schedule_df().drop("home_score", "away_score")

    # Act
    played = season_tables.played_schedule(schedule)

    # Assert
    assert played.equals(schedule)


def test_finalize_table_rejects_columns_missing_from_the_registry() -> None:
    # Arrange
    frame = pl.DataFrame({"team": ["NE"], "not_a_registered_column": [1]})

    # Act & Assert
    with pytest.raises(ValueError, match="not_a_registered_column"):
        season_tables.finalize_table(frame, "team_per_game_stats")


def test_finalize_table_orders_team_rows_by_team_then_week() -> None:
    # Arrange
    frame = pl.DataFrame(
        {"week": [2, 1, 2, 1], "team": ["NE", "NE", "BUF", "BUF"], "points_for": [1, 2, 3, 4]}
    )

    # Act
    ordered = season_tables.finalize_table(frame, "team_game_logs")

    # Assert
    assert ordered.select("team", "week").rows() == [("BUF", 1), ("BUF", 2), ("NE", 1), ("NE", 2)]


def test_finalize_table_orders_qb_rows_by_passer_id_before_team() -> None:
    # Arrange
    frame = pl.DataFrame(
        {
            "qb_id": ["00-2", "00-1", "00-2", "00-1"],
            "team": ["BUF", "NE", "BUF", "NE"],
            "week": [2, 2, 1, 1],
        }
    )

    # Act
    ordered = season_tables.finalize_table(frame, "qb_game_logs")

    # Assert
    assert ordered.select("qb_id", "week").rows() == [
        ("00-1", 1),
        ("00-1", 2),
        ("00-2", 1),
        ("00-2", 2),
    ]


def test_finalize_table_rejects_a_frame_without_a_row_key() -> None:
    # Arrange
    frame = pl.DataFrame({"week": [1, 2]})

    # Act & Assert
    with pytest.raises(ValueError, match="row order"):
        season_tables.finalize_table(frame, "weeks")


def test_build_season_games_profiles_opponents_from_played_games_only() -> None:
    # Arrange
    unplayed = pl.DataFrame(
        {
            "game_id": ["2025_18_BUF_PIT"],
            "week": [18],
            "home_team": ["PIT"],
            "away_team": ["BUF"],
            "home_score": [None],
            "away_score": [None],
        },
        schema=_schedule_df().schema,
    )
    inputs = SeasonInputs(
        season=2025,
        pbp=pl.DataFrame(),
        player_stats=pl.DataFrame(),
        schedule=pl.concat([_schedule_df(), unplayed]),
        official_team_stats=pl.DataFrame(),
        snap_counts=pl.DataFrame(),
        qb_identity=pl.DataFrame(),
    )

    # Act
    games = season_tables.build_season_games(inputs)

    # Assert
    assert "PIT" not in games.schedule.get_column("home_team").to_list()


def test_build_season_tables_names_each_table_after_its_published_file() -> None:
    # Act
    names = list(_tables().tables())

    # Assert
    assert names == [
        "team_game_logs",
        "qb_game_logs",
        "team_per_game_stats",
        "qb_per_game_stats",
        "qb_opponent_profiles",
        "opponent_profiles",
        "combined",
        "qb_combined",
        "team_wp_bins",
        "qb_wp_bins",
    ]


def test_build_season_tables_publish_no_hidden_rate_parts() -> None:
    # Act
    tables = _tables().tables()

    # Assert
    parts = {
        name: [column for column in frame.columns if is_rate_part(column)]
        for name, frame in tables.items()
    }
    assert {name: columns for name, columns in parts.items() if columns} == {}


def test_build_season_tables_publish_only_registered_columns() -> None:
    # Act
    tables = _tables().tables()

    # Assert
    unknown = {
        name: get_registry().validate_columns(frame.columns) for name, frame in tables.items()
    }
    assert {name: columns for name, columns in unknown.items() if columns} == {}


def test_build_season_tables_pool_a_season_rate_over_the_teams_games() -> None:
    # Arrange
    expected = dict(
        _weekly_df()
        .group_by("team")
        .agg((pl.col("points_for").sum() / pl.col("offensive_snaps").sum()).alias("rate"))
        .rows()
    )

    # Act
    per_game = _tables().team_per_game_stats

    # Assert
    assert dict(per_game.select("team", "points_per_offensive_snap").rows()) == pytest.approx(
        expected
    )


def test_build_season_tables_combine_stats_profiles_and_ratings_in_that_order() -> None:
    # Act
    combined = _tables().combined

    # Assert
    rating_columns = list(season_tables.TEAM_RATING_INPUT_COLUMNS[1:])
    assert combined.columns[-len(rating_columns) :] == rating_columns
    assert "games_played" in combined.columns
    first_profile = combined.columns.index("opp_points_for")
    assert first_profile < combined.columns.index("offense_rating")
    assert combined.select("team", "team_rating").rows() == [
        ("BUF", 3.0),
        ("MIA", -1.0),
        ("NE", 1.0),
        ("NYJ", -3.0),
    ]


def test_build_season_tables_give_each_passer_his_rating() -> None:
    # Act
    qb_combined = _tables().qb_combined

    # Assert
    assert qb_combined.columns[-2:] == list(season_tables.QB_RATING_INPUT_COLUMNS[1:])
    assert qb_combined.select("qb_id", "adj_qb_epa_per_dropback").rows() == [
        ("qb-BUF", 0.2),
        ("qb-MIA", 0.0),
        ("qb-NE", 0.1),
        ("qb-NYJ", -0.1),
    ]


def test_build_season_tables_refuse_ratings_without_a_rating_column() -> None:
    # Arrange
    ratings = _team_ratings().drop("sos")

    # Act & Assert
    with pytest.raises(ValueError, match="sos"):
        season_tables.build_season_tables(
            _season_games(), _pbp(), team_ratings=ratings, qb_ratings=_qb_ratings()
        )


def test_build_season_tables_bin_the_seasons_plays_by_win_probability() -> None:
    # Act
    tables = _tables()

    # Assert
    assert tables.team_wp_bins.equals(
        season_tables.finalize_table(compute_team_wp_bins(_pbp()), "team_wp_bins")
    )
    assert tables.qb_wp_bins.equals(
        season_tables.finalize_table(compute_qb_wp_bins(_pbp()), "qb_wp_bins")
    )


def test_build_season_tables_leave_out_profiles_no_opponent_game_supports() -> None:
    """A season with one game per team has no opponent games left once head-to-head is out."""
    # Arrange
    first_week = _weekly_df().filter(pl.col("week") == 1)
    games = season_tables.SeasonGames(
        team_games=first_week,
        qb_games=_qb_df().filter(pl.col("week") == 1),
        schedule=_schedule_df().filter(pl.col("week") == 1),
    )

    # Act
    tables = season_tables.build_season_tables(
        games, _pbp(), team_ratings=_team_ratings(), qb_ratings=_qb_ratings()
    )

    # Assert
    assert tables.opponent_profiles is None
    assert "opponent_profiles" not in tables.tables()
    assert not [column for column in tables.combined.columns if column.startswith("opp_")]


def test_build_season_tables_sum_each_teams_games_in_one_order_whatever_the_input_order() -> None:
    """Floats sum to other last bits in another order: 0.1 + 0.2 + 0.3 is not 0.3 + 0.2 + 0.1."""
    # Arrange
    epa = [0.1, 0.2, 0.3, 0.7, 0.11, 0.13]
    weekly = _weekly_df().with_columns(
        pl.Series("offensive_epa", [epa[index % len(epa)] for index in range(_weekly_df().height)])
    )
    shuffled = season_tables.SeasonGames(
        team_games=weekly.reverse(), qb_games=_qb_df().reverse(), schedule=_schedule_df()
    )
    ordered = season_tables.SeasonGames(
        team_games=weekly, qb_games=_qb_df(), schedule=_schedule_df()
    )
    expected = season_tables.build_season_tables(
        ordered, _pbp(), team_ratings=_team_ratings(), qb_ratings=_qb_ratings()
    )

    # Act
    tables = season_tables.build_season_tables(
        shuffled, _pbp(), team_ratings=_team_ratings(), qb_ratings=_qb_ratings()
    )

    # Assert
    assert {
        name: frame.equals(expected.tables()[name]) for name, frame in tables.tables().items()
    } == (dict.fromkeys(expected.tables(), True))


def _before_kickoff() -> SeasonInputs:
    """Return a current season's inputs before kickoff: a schedule and nothing else."""
    schedule = _schedule_df().with_columns(
        pl.lit(None, dtype=pl.Int64).alias("home_score"),
        pl.lit(None, dtype=pl.Int64).alias("away_score"),
    )
    return SeasonInputs(
        season=2026,
        pbp=pl.DataFrame(schema={"game_id": pl.String, "week": pl.Int64, "posteam": pl.String}),
        player_stats=pl.DataFrame(),
        schedule=schedule,
        official_team_stats=pl.DataFrame(),
        snap_counts=pl.DataFrame(),
        qb_identity=pl.DataFrame(),
    )


def test_a_season_before_kickoff_has_no_game_played() -> None:
    # Act
    games = season_tables.build_season_games(_before_kickoff())

    # Assert
    assert games.played is False
    assert games.schedule.is_empty()


def test_a_season_with_game_rows_has_a_game_played() -> None:
    # Act
    played = _season_games().played

    # Assert
    assert played is True


def test_build_season_tables_refuse_a_season_without_a_game_played() -> None:
    # Arrange
    games = season_tables.build_season_games(_before_kickoff())

    # Act & Assert
    with pytest.raises(ValueError, match="no game played"):
        season_tables.build_season_tables(
            games, pl.DataFrame(), team_ratings=_team_ratings(), qb_ratings=_qb_ratings()
        )
