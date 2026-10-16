"""Tests for the team and quarterback index tables and the season's progress."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_predictor.api.ratings import tables
from tests.api import ratings_factories as factories

if TYPE_CHECKING:
    from nfl_predictor.api.schemas.ratings import EntityKind


def test_team_index_groups_ratings_rates_and_opponent_profiles() -> None:
    """The team table leads with the ratings, then per-game and per-snap stats, then opponents."""
    # Arrange
    frame = factories.combined(games_played=17)

    # Act
    table = tables.index_table("teams", frame)

    # Assert
    groups = table.column_groups
    assert groups["identity"] == ["team"]
    assert groups["ratings"] == [
        "team_rating",
        "offense_rating",
        "defense_rating",
        "special_teams_rating",
        "sos",
        "SRS",
    ]
    assert groups["rating_companions"] == []
    assert groups["per_snap_rates"] == ["passing_yards_per_offensive_snap"]
    assert groups["opponent_context"] == ["opp_points_for", "opp_completion_pct"]
    assert "points_for" in groups["per_game_rates"]
    assert table.visible_columns[0] == "team"
    assert len(table.rows) == 4


def test_team_index_orders_stats_by_the_registry_categories() -> None:
    """Per-game stats follow the registry's category order: overall results before offense."""
    # Arrange
    frame = factories.combined(games_played=17)

    # Act
    per_game = tables.index_table("teams", frame).column_groups["per_game_rates"]

    # Assert
    assert per_game.index("points_for") < per_game.index("passing_yards")
    assert per_game.index("games_played") < per_game.index("completion_pct")


def test_qb_index_groups_ratings_companions_totals_and_faced_defenses() -> None:
    """The quarterback table splits totals, per-game and per-dropback stats from the ratings."""
    # Arrange
    frame = factories.qb_combined()

    # Act
    table = tables.index_table("qbs", frame)

    # Assert
    groups = table.column_groups
    assert groups["identity"] == ["qb_id", "qb_name", "team"]
    assert groups["ratings"] == ["adj_qb_epa_per_dropback", "qb_faced_pass_defense"]
    assert groups["rating_companions"] == ["qb_epa_per_dropback", "qb_dropbacks_total"]
    assert groups["per_game_rates"] == ["qb_pass_yards_per_game"]
    assert groups["per_dropback_rates"] == ["qb_epa_per_dropback"]
    assert groups["opponent_context"] == ["qopp_points_allowed", "qopp_qb_epa_per_dropback"]
    assert {"qb_attempts_total", "qb_is_eligible", "qb_attempt_qualifier"} <= set(
        groups["raw_totals"]
    )
    assert len(table.rows) == 5


def test_qb_index_pins_the_name_not_the_id() -> None:
    """The quarterback's name stays in view; his player id does not."""
    # Arrange
    frame = factories.qb_combined()

    # Act
    metadata = tables.index_table("qbs", frame).column_metadata

    # Assert
    assert (metadata["qb_name"].sticky, metadata["qb_id"].sticky) == (True, False)


@pytest.mark.parametrize("kind", ["teams", "qbs"])
def test_categories_follow_the_registry_order(kind: EntityKind) -> None:
    """Each page's categories come from the registry, the ratings category first."""
    # Act
    categories = tables.categories(kind)

    # Assert
    assert categories[0].name == "Schedule-Adjusted Ratings"
    assert all(category.description for category in categories)


def test_team_categories_carry_the_offense_subcategories() -> None:
    """The team page's offense filter lists the registry's subcategories."""
    # Act
    offense = next(c for c in tables.categories("teams") if c.name == "Offense")

    # Assert
    assert offense.subcategories[:3] == ["Total", "Passing", "Rushing"]


@pytest.mark.parametrize(
    ("season", "current", "games", "expected"),
    [
        (2025, 2025, 5, (True, 5)),
        (2025, 2025, 17, (False, 17)),
        (2024, 2025, 5, (False, 5)),
        (2020, 2020, 16, (False, 16)),
        (2020, 2020, 15, (True, 15)),
    ],
)
def test_season_progress_marks_only_the_current_season_with_games_left(
    season: int, current: int, games: int, expected: tuple[bool, int]
) -> None:
    """A season is in progress while it is the current one and a team has games left."""
    # Arrange
    frame = factories.combined(games_played=games)

    # Act
    progress = tables.season_progress(frame, season, current)

    # Assert
    assert progress == expected


def test_season_progress_without_games_played_counts_the_current_season_as_in_progress() -> None:
    """A team table without games played cannot show the season complete."""
    # Arrange
    frame = pl.DataFrame({"team": ["BUF"]})

    # Act
    progress = tables.season_progress(frame, 2026, 2026)

    # Assert
    assert progress == (True, None)


def test_team_names_name_every_team_of_the_table() -> None:
    """Each team code in the table gets its franchise name from the canonical team mapping."""
    # Arrange
    frame = factories.qb_combined()

    # Act
    names = tables.team_names(frame)

    # Assert
    assert names == {
        "BUF": "Buffalo Bills",
        "KC": "Kansas City Chiefs",
        "NE": "New England Patriots",
        "WSH": "Washington Commanders",
    }


def test_team_names_keep_an_unknown_code_as_it_is() -> None:
    """A code the mapping does not know names itself rather than disappearing."""
    # Arrange
    frame = pl.DataFrame({"team": ["KC", "XYZ", None]})

    # Act
    names = tables.team_names(frame)

    # Assert
    assert names == {"KC": "Kansas City Chiefs", "XYZ": "XYZ"}


@pytest.mark.parametrize(("season", "games"), [(2020, 16), (2021, 17), (1999, 16)])
def test_games_per_team_is_one_bye_short_of_the_weeks(season: int, games: int) -> None:
    """Teams play every regular-season week but one."""
    # Act
    result = tables.games_per_team(season)

    # Assert
    assert result == games
