"""Tests for one team's or quarterback's sections: logs, histories, ranges and pairs."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_predictor.api.ratings import files, sections
from tests.api import ratings_factories as factories


def _team_logs(team: str) -> pl.DataFrame:
    """Return one team's game logs from the factory season."""
    return files.entity_rows(factories.team_game_logs(2025), "team", team)


def test_team_game_logs_lead_with_identity_and_results() -> None:
    """A team's games show who and when, then the score and result, then the stats."""
    # Act
    table = sections.game_log_table("teams", _team_logs("WSH"), factories.ratings(17))

    # Assert
    groups = table.column_groups
    assert groups["identity"] == ["game_id", "week", "team", "opponent_team"]
    assert groups["results"] == [
        "points_for",
        "points_allowed",
        "point_margin",
        "win_value",
        "turnover_margin",
    ]
    assert groups["per_snap_rates"] == ["passing_yards_per_offensive_snap"]
    assert "passing_yards" in groups["raw_totals"]
    assert [row["week"] for row in table.rows] == [1, 2, 3, 4, 5, 6]


def test_game_logs_attach_each_opponents_season_ratings() -> None:
    """Every game carries its opponent's ratings for the whole season."""
    # Act
    table = sections.game_log_table("teams", _team_logs("WSH"), factories.ratings(17))

    # Assert
    assert table.column_groups["opponent_ratings"] == [
        "opp_team_rating",
        "opp_team_rank",
        "opp_offense_rating",
        "opp_defense_rating",
        "opp_special_teams_rating",
        "opp_SRS",
    ]
    by_opponent = {row["opponent_team"]: row["opp_team_rating"] for row in table.rows}
    assert by_opponent == {"BUF": 6.5, "KC": 4.0, "NE": -2.0}


def test_game_logs_rank_each_opponent_in_the_league() -> None:
    """Each game carries its opponent's league rank on Team Rating, 1 the best."""
    # Act
    table = sections.game_log_table("teams", _team_logs("WSH"), factories.ratings(17))

    # Assert
    by_opponent = {row["opponent_team"]: row["opp_team_rank"] for row in table.rows}
    assert by_opponent == {"BUF": 1, "KC": 2, "NE": 3}
    meta = table.column_metadata["opp_team_rank"]
    assert (meta.group, meta.contextual, meta.polarity) == ("opponent_ratings", True, "lower")
    assert "this game's opponent" in meta.description.lower()


def test_game_log_opponent_ratings_describe_the_opponent_not_the_schedule() -> None:
    """The attached ratings are one opponent's season, not the average over every opponent."""
    # Act
    meta = sections.game_log_table(
        "teams", _team_logs("WSH"), factories.ratings(17)
    ).column_metadata["opp_team_rating"]

    # Assert
    assert "this game's opponent" in meta.description.lower()
    assert "average over the opponents faced" not in meta.description.lower()
    assert meta.contextual is True


def test_game_logs_without_a_ratings_table_have_no_opponent_ratings() -> None:
    """A season without ratings still shows its game logs."""
    # Act
    table = sections.game_log_table("teams", _team_logs("WSH"), None)

    # Assert
    assert table.column_groups["opponent_ratings"] == []
    assert len(table.rows) == 6


def test_qb_game_logs_group_per_dropback_rates_and_late_game_credits() -> None:
    """A quarterback's games show his comeback credits with the results."""
    # Arrange
    logs = files.entity_rows(factories.qb_game_logs(2025), "qb_id", "00-0039910")

    # Act
    table = sections.game_log_table("qbs", logs, factories.ratings(17))

    # Assert
    groups = table.column_groups
    assert groups["identity"] == ["game_id", "week", "team", "opponent_team", "qb_id", "qb_name"]
    assert groups["results"] == [
        "points_for",
        "points_allowed",
        "qb_fourth_quarter_comeback",
        "qb_game_winning_drive",
    ]
    assert groups["per_dropback_rates"] == ["qb_epa_per_dropback"]
    assert "opp_defense_rating" in groups["opponent_ratings"]


def test_rating_history_lists_the_sample_then_the_ratings() -> None:
    """A team's weekly history shows games played, then each rating, headline first."""
    # Arrange
    rows = files.entity_rows(factories.ratings_by_week(), "team", "KC")

    # Act
    table = sections.rating_history_table("teams", rows)

    # Assert
    assert table.column_groups == {
        "identity": ["week", "team"],
        "sample": ["games_played"],
        "ratings": ["team_rating", "offense_rating", "defense_rating", "special_teams_rating"],
    }
    assert [row["week"] for row in table.rows] == [1, 2, 3, 4, 5, 6]


def test_qb_rating_history_lists_games_and_dropbacks() -> None:
    """A quarterback's weekly history shows his games and dropbacks behind the rating."""
    # Arrange
    rows = files.entity_rows(factories.qb_ratings_by_week(), "qb_id", "00-0034857")

    # Act
    table = sections.rating_history_table("qbs", rows)

    # Assert
    assert table.column_groups["sample"] == ["qb_games_played", "qb_dropbacks"]
    assert table.column_groups["ratings"] == ["adj_qb_epa_per_dropback"]


def test_rank_ranges_group_the_rating_rank_chances_and_units() -> None:
    """A team's rank range carries rating and rank percentiles, chances and unit ranges."""
    # Act
    table = sections.rank_range_table("teams", factories.rating_ranges())

    # Assert
    groups = table.column_groups
    assert groups["identity"] == ["team"]
    assert groups["published"] == ["team_rank"]
    assert groups["rating_range"][0] == "team_rating_q025"
    assert groups["rank_range"][-1] == "team_rank_q975"
    assert groups["rank_chances"] == [
        "team_rank_top5_probability",
        "team_rank_top10_probability",
        "team_rank_missing_share",
        "team_rank_probabilities",
    ]
    assert groups["offense_range"][0] == "offense_rank"
    assert set(groups) >= {"defense_range", "special_teams_range"}
    assert [row["team"] for row in table.rows] == ["BUF", "KC", "NE", "WSH"]
    assert len(table.rows[0]["team_rank_probabilities"]) == 4


def test_rank_ranges_sort_by_published_rank() -> None:
    """Rows come back best first whatever order the file holds."""
    # Arrange
    frame = factories.qb_rating_ranges().reverse()

    # Act
    table = sections.rank_range_table("qbs", frame)

    # Assert
    assert [row["qb_rank"] for row in table.rows] == [1, 2, 3, 4]
    assert table.column_groups["identity"] == ["qb_id", "qb_name", "team"]


def test_rank_history_keeps_the_weeks_from_the_first_week_given() -> None:
    """Weeks before every team had three games are left out."""
    # Arrange
    rows = files.entity_rows(factories.rating_ranges_by_week(), "team", "BUF")

    # Act
    table = sections.rank_history_table("teams", rows, first_week=3)

    # Assert
    assert [row["week"] for row in table.rows] == [3, 4, 5, 6]
    assert table.column_groups["rank_range"][:2] == ["team_rank", "team_rank_q025"]
    assert table.column_groups["rank_chances"] == [
        "team_rank_top5_probability",
        "team_rank_top10_probability",
        "team_rank_missing_share",
    ]


@pytest.mark.parametrize(
    ("buffalo", "kansas_city", "first"),
    [((1, 2, 3, 4), (1, 2, 2, 3), 4), ((1, 2, 2, 3), (1, 1, 2, 2), 5)],
)
def test_first_rank_history_week_is_the_week_every_team_reaches_three_games(
    buffalo: tuple[int, ...], kansas_city: tuple[int, ...], first: int
) -> None:
    """The weekly rank chart starts once the fewest games any team has played reaches three.

    With no such week yet, it starts after the last week played, so no week shows.
    """
    # Arrange
    weeks = range(1, len(buffalo) + 1)
    history = pl.DataFrame(
        {
            "week": [*weeks, *weeks],
            "team": ["BUF"] * len(buffalo) + ["KC"] * len(kansas_city),
            "games_played": [*buffalo, *kansas_city],
        }
    )

    # Act
    week = sections.first_rank_history_week(history)

    # Assert
    assert week == first


def test_first_rank_history_week_without_a_history_shows_every_week() -> None:
    """A season without a weekly rating history puts no floor under the rank chart."""
    # Act
    week = sections.first_rank_history_week(None)

    # Assert
    assert week is None


def test_head_to_head_lists_one_team_against_every_other() -> None:
    """A team's pairs name each other team with the chance and the rating-gap percentiles."""
    # Arrange
    rows = files.entity_rows(factories.rating_pairs(), "team", "KC")

    # Act
    table = sections.pair_table("teams", rows)

    # Assert
    assert table.column_groups == {
        "identity": ["team", "other_team"],
        "chance": ["team_rated_above_probability", "team_pair_share"],
        "rating_gap": ["team_rating_gap_q025", "team_rating_gap_q500", "team_rating_gap_q975"],
    }
    assert [row["other_team"] for row in table.rows] == ["BUF", "NE", "WSH"]
    assert table.rows[0]["team_rated_above_probability"] == 0.2
