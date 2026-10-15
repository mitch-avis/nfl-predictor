"""Tests for the ported quarterback rating as the own Elo's quarterback adjustment."""

from __future__ import annotations

import polars as pl
import pytest
from tests.sos_port_scripts import load_sos_port_script

elo_qb_alt = load_sos_port_script("elo_qb_alt")

TABLE_SCHEMA = {
    "week": pl.Int64,
    "qb_id": pl.String,
    "qb_dropbacks": pl.Int64,
    "adj_qb_epa_per_dropback": pl.Float64,
}


GAME_SCHEMA = {
    "game_id": pl.String,
    "season": pl.Int64,
    "week": pl.Int64,
    "game_type": pl.String,
    "home_abbr": pl.String,
    "away_abbr": pl.String,
    "home_qb_id": pl.String,
    "away_qb_id": pl.String,
}
# 25 Elo points per point, times 30 dropbacks a team-game (600 dropbacks over 10 games in 2001).
ROLLING_ELO_PER_RATING_POINT = 750.0


def _table(rows: list[tuple[int, str, int, float | None]]) -> pl.DataFrame:
    """Return a by-week quarterback table from ``(week, qb_id, dropbacks, rating)`` rows."""
    return pl.DataFrame(rows, schema=TABLE_SCHEMA, orient="row")


def _games(rows: list[tuple[str, int, int, str, str, str, str | None, str | None]]) -> pl.DataFrame:
    """Return schedule rows with each side's starter, in rating order."""
    return pl.DataFrame(rows, schema=GAME_SCHEMA, orient="row")


def _rolling_tables(week_two_rating: float = 0.4) -> dict[int, pl.DataFrame]:
    """Return 2001 (600 dropbacks) and 2002 tables where A rates 0.2, then the given value."""
    return {
        2001: _table([(17, "A", 400, 0.1), (17, "B", 200, 0.3)]),
        2002: _table([(1, "A", 30, 0.2), (2, "A", 60, week_two_rating)]),
    }


def _rolling_adjustments(tables: dict[int, pl.DataFrame], scale: float = 1.0) -> pl.DataFrame:
    """Return three 2002 games of home team X started by A, rated from ``tables``."""
    games = _games(
        [(f"g{week}", 2002, week, "REG", "X", "Y", "A", "B") for week in (1, 2, 3)],
    )
    starts = {(2002, week, team): qb for week in (1, 2, 3) for team, qb in (("X", "A"), ("Y", "B"))}
    return elo_qb_alt.alternative_adjustments(
        games, starts, elo_qb_alt.QuarterbackRatings(tables), {2001: 10, 2002: 10}, scale=scale
    )


def test_rating_with_dropback_this_season_reads_the_row_before_the_week() -> None:
    # Arrange
    ratings = elo_qb_alt.QuarterbackRatings(
        {2010: _table([(1, "A", 30, 0.1), (2, "A", 60, 0.2), (3, "A", 90, 0.3)])}
    )

    # Act
    rating = ratings.rating("A", 2010, 3, playoff=False)

    # Assert
    assert rating == pytest.approx(0.2)


def test_rating_without_dropback_this_season_reads_latest_rated_earlier_season() -> None:
    # Arrange
    ratings = elo_qb_alt.QuarterbackRatings(
        {
            2008: _table([(1, "A", 30, 0.5), (17, "A", 400, 0.1)]),
            2009: _table([(1, "A", 30, 0.6), (17, "A", 300, None)]),
            2010: _table([(2, "A", 40, 0.9)]),
        }
    )

    # Act
    rating = ratings.rating("A", 2010, 2, playoff=False)

    # Assert
    assert rating == pytest.approx(0.1)


def test_rating_of_unrated_quarterback_is_mean_latest_debut_rating() -> None:
    # Arrange
    ratings = elo_qb_alt.QuarterbackRatings(
        {
            2000: _table([(17, "V", 500, 0.4)]),
            2001: _table([(17, "B", 300, 0.2), (17, "V", 500, 0.5)]),
            2002: _table([(1, "C", 20, 0.7), (17, "C", 100, -0.1), (17, "B", 200, 0.3)]),
            2003: _table([(1, "X", 20, 0.0)]),
        }
    )

    # Act
    rating = ratings.rating("D", 2003, 1, playoff=False)

    # Assert
    assert rating == pytest.approx(0.05)


def test_replacement_level_without_debut_season_is_zero() -> None:
    # Arrange
    ratings = elo_qb_alt.QuarterbackRatings({2000: _table([(17, "V", 500, 0.4)])})

    # Act
    level = ratings.replacement_level(2001)

    # Assert
    assert level == 0.0


def test_playoff_rating_reads_latest_regular_season_row() -> None:
    # Arrange
    ratings = elo_qb_alt.QuarterbackRatings(
        {2010: _table([(1, "A", 30, 0.1), (16, "A", 450, 0.2), (17, "A", 480, 0.3)])}
    )

    # Act
    rating = ratings.rating("A", 2010, 19, playoff=True)

    # Assert
    assert rating == pytest.approx(0.3)


def test_rating_with_label_before_the_week_reads_the_week_row() -> None:
    # Arrange
    ratings = elo_qb_alt.QuarterbackRatings(
        {2010: _table([(1, "A", 30, 0.1), (2, "A", 60, 0.2), (3, "A", 90, 0.3)])}, offset=0
    )

    # Act
    rating = ratings.rating("A", 2010, 3, playoff=False)

    # Assert
    assert rating == pytest.approx(0.3)


def test_dropbacks_per_team_game_means_earlier_seasons() -> None:
    # Arrange
    ratings = elo_qb_alt.QuarterbackRatings(
        {
            2000: _table([(1, "A", 30, 0.1), (17, "A", 400, 0.1), (17, "B", 200, 0.2)]),
            2001: _table([(17, "A", 500, 0.3)]),
        }
    )

    # Act
    scale = ratings.dropbacks_per_team_game(2002, {2000: 10, 2001: 10})

    # Assert
    assert scale == pytest.approx(27.5)


def test_dropbacks_per_team_game_without_earlier_season_is_none() -> None:
    # Arrange
    ratings = elo_qb_alt.QuarterbackRatings({2000: _table([(17, "A", 400, 0.1)])})

    # Act
    scale = ratings.dropbacks_per_team_game(2000, {2000: 10})

    # Assert
    assert scale is None


def test_rating_with_attached_value_reads_it_over_the_table() -> None:
    # Arrange
    ratings = elo_qb_alt.QuarterbackRatings(
        {2010: _table([(1, "A", 30, 0.1), (2, "A", 60, 0.2)])},
        attached={(2010, 3, "A"): 0.25},
    )

    # Act
    rating = ratings.rating("A", 2010, 3, playoff=False)

    # Assert
    assert rating == pytest.approx(0.25)


def test_rating_without_attached_value_reads_latest_rated_earlier_season() -> None:
    # Arrange
    ratings = elo_qb_alt.QuarterbackRatings(
        {
            2009: _table([(17, "A", 300, 0.3)]),
            2010: _table([(1, "A", 30, 0.1), (2, "A", 60, 0.2)]),
        },
        attached={},
    )

    # Act
    rating = ratings.rating("A", 2010, 3, playoff=False)

    # Assert
    assert rating == pytest.approx(0.3)


def test_season_check_counts_equal_and_different_ratings_per_season() -> None:
    # Arrange
    tables = {
        2010: _table([(1, "A", 30, 0.1), (2, "A", 60, 0.2)]),
        2011: _table([(1, "A", 30, 0.5)]),
    }
    sides = pl.DataFrame(
        {
            "season": [2010, 2010, 2011, 2011],
            "week": [2, 1, 2, 1],
            "qb_id": ["A", "A", "A", "A"],
            "attached": [0.1, None, 0.4, None],
        },
        schema={"season": pl.Int64, "week": pl.Int64, "qb_id": pl.String, "attached": pl.Float64},
    )

    # Act
    counts = elo_qb_alt.season_check(sides, tables)

    # Assert
    assert counts == {
        2010: {"equal": 1, "different": 0, "both_null": 1, "one_null": 0},
        2011: {"equal": 0, "different": 1, "both_null": 1, "one_null": 0},
    }


def test_starter_rating_check_counts_matches_per_week_label() -> None:
    # Arrange
    tables = {2010: _table([(1, "A", 30, 0.1), (2, "A", 60, 0.2)])}
    sides = pl.DataFrame(
        {"season": [2010, 2010], "week": [2, 1], "qb_id": ["A", "A"], "attached": [0.1, None]},
        schema={"season": pl.Int64, "week": pl.Int64, "qb_id": pl.String, "attached": pl.Float64},
    )

    # Act
    counts = elo_qb_alt.starter_rating_check(sides, tables)

    # Assert
    assert counts == {
        0: {"matches": 0, "mismatches": 2},
        1: {"matches": 2, "mismatches": 0},
    }


def test_adjustment_rolls_team_rating_toward_each_start() -> None:
    # Arrange
    tables = _rolling_tables()

    # Act
    adjustments = _rolling_adjustments(tables)

    # Assert
    assert adjustments["home_team_rating"].to_list() == pytest.approx([0.1, 0.1, 0.105])
    assert adjustments["home_qb_adj"].to_list() == pytest.approx(
        [0.0, 0.1 * ROLLING_ELO_PER_RATING_POINT, 0.295 * ROLLING_ELO_PER_RATING_POINT]
    )


def test_adjustment_scale_multiplies_every_adjustment() -> None:
    # Arrange
    tables = _rolling_tables()

    # Act
    adjustments = _rolling_adjustments(tables, scale=0.5)

    # Assert
    assert adjustments["home_qb_adj"].to_list() == pytest.approx(
        [0.0, 0.05 * ROLLING_ELO_PER_RATING_POINT, 0.1475 * ROLLING_ELO_PER_RATING_POINT]
    )


def test_adjustment_ignores_ratings_of_later_weeks() -> None:
    # Arrange
    tables = _rolling_tables(week_two_rating=9.9)

    # Act
    changed = _rolling_adjustments(tables)

    # Assert
    assert changed["home_qb_rating"].to_list() == pytest.approx([0.1, 0.2, 9.9])
    assert changed["home_team_rating"].to_list() == pytest.approx([0.1, 0.1, 0.105])
    assert changed["home_qb_adj"].to_list()[:2] == pytest.approx(
        [0.0, 0.1 * ROLLING_ELO_PER_RATING_POINT]
    )


def test_preseason_team_rating_blends_end_of_season_with_first_starter() -> None:
    # Arrange
    games = _games(
        [
            ("g1", 2001, 1, "REG", "X", "Y", "A", None),
            ("g2", 2002, 1, "REG", "X", "Y", "C", None),
        ]
    )
    ratings = elo_qb_alt.QuarterbackRatings(
        {2000: _table([(17, "A", 400, 0.3)]), 2001: _table([(1, "A", 30, 0.6)])}
    )

    # Act
    adjustments = elo_qb_alt.alternative_adjustments(
        games, {(2001, 1, "X"): "A"}, ratings, {2000: 10, 2001: 10}
    )

    # Assert
    assert adjustments["home_team_rating"].to_list() == pytest.approx([0.3, 0.1])
    assert adjustments["home_qb_adj"].to_list() == pytest.approx([0.0, 25.0 * 10.75 * -0.1])


def test_adjustment_without_starter_is_zero() -> None:
    # Arrange
    games = _games([("g1", 2002, 3, "REG", "X", "Y", None, "B")])

    # Act
    adjustments = elo_qb_alt.alternative_adjustments(
        games, {}, elo_qb_alt.QuarterbackRatings(_rolling_tables()), {2001: 10}
    )

    # Assert
    assert adjustments["home_qb_adj"].to_list() == [0.0]


def test_adjustment_in_first_season_is_zero() -> None:
    # Arrange
    games = _games(
        [("g1", 2001, 3, "REG", "X", "Y", "A", "B"), ("g2", 2001, 4, "REG", "X", "Y", "B", "A")]
    )

    # Act
    adjustments = elo_qb_alt.alternative_adjustments(
        games, {}, elo_qb_alt.QuarterbackRatings(_rolling_tables()), {2001: 10}
    )

    # Assert
    assert adjustments["home_qb_adj"].to_list() == [0.0, 0.0]


@pytest.mark.parametrize(
    ("brier_ci", "log_loss_ci", "expected"),
    [
        ([-0.002, -0.001], [-0.001, 0.001], "would replace VALUE"),
        ([-0.002, 0.001], [-0.001, 0.001], "measures at least as well; VALUE stays"),
        ([-0.002, -0.001], [0.001, 0.002], "VALUE stays"),
        ([0.001, 0.002], [-0.001, 0.001], "VALUE stays"),
    ],
)
def test_verdict_applies_the_rule_to_both_intervals(
    brier_ci: list[float], log_loss_ci: list[float], expected: str
) -> None:
    # Arrange
    comparison = {"brier_ci": brier_ci, "log_loss_ci": log_loss_ci}

    # Act
    outcome = elo_qb_alt.verdict(comparison)

    # Assert
    assert outcome == expected


def test_rating_without_dropback_this_season_reads_most_recent_rated_season() -> None:
    # Arrange
    ratings = elo_qb_alt.QuarterbackRatings(
        {
            2008: _table([(17, "A", 400, 0.1)]),
            2009: _table([(17, "A", 300, 0.3)]),
            2010: _table([(2, "A", 40, 0.9)]),
        }
    )

    # Act
    rating = ratings.rating("A", 2010, 2, playoff=False)

    # Assert
    assert rating == pytest.approx(0.3)


def _playoff_adjustments() -> pl.DataFrame:
    """Return X's games of weeks 1 and 3 around a playoff game, season-to-date from attached.

    The attached map also holds a value for the playoff week, which the playoff game must not
    read, and the starts hold a playoff start, which must not roll the team's baseline.
    """
    games = _games(
        [
            ("g1", 2002, 1, "REG", "X", "Y", "A", "B"),
            ("g2", 2002, 19, "POST", "X", "Y", "A", "B"),
            ("g3", 2002, 3, "REG", "X", "Y", "A", "B"),
        ]
    )
    attached = {(2002, 1, "A"): None, (2002, 19, "A"): 0.9, (2002, 3, "A"): 0.25}
    ratings = elo_qb_alt.QuarterbackRatings(_rolling_tables(), attached=attached)
    starts = {(2002, week, "X"): "A" for week in (1, 19, 3)}
    return elo_qb_alt.alternative_adjustments(games, starts, ratings, {2001: 10})


def test_playoff_adjustment_reads_latest_regular_season_row() -> None:
    # Act
    adjustments = _playoff_adjustments()

    # Assert
    assert adjustments["home_qb_rating"].to_list() == pytest.approx([0.1, 0.4, 0.25])


def test_playoff_adjustment_leaves_team_rating_unrolled() -> None:
    # Act
    adjustments = _playoff_adjustments()

    # Assert
    assert adjustments["home_team_rating"].to_list() == pytest.approx([0.1, 0.1, 0.1])


def test_team_first_seen_without_starter_starts_at_replacement_level() -> None:
    # Arrange
    games = _games([("g1", 2003, 1, "REG", "X", "Y", None, None)])
    ratings = elo_qb_alt.QuarterbackRatings(
        {
            2000: _table([(17, "V", 500, 0.4)]),
            2001: _table([(17, "B", 300, 0.2)]),
            2002: _table([(17, "C", 100, -0.1)]),
        }
    )

    # Act
    adjustments = elo_qb_alt.alternative_adjustments(
        games, {}, ratings, {2000: 10, 2001: 10, 2002: 10}
    )

    # Assert
    assert adjustments["home_team_rating"].to_list() == pytest.approx([0.05])


def test_check_sides_drops_side_whose_starter_is_not_play_by_play_starter() -> None:
    # Arrange
    games = _games([("g1", 2010, 2, "REG", "X", "Y", "A", "B")])
    starts = {(2010, 2, "X"): "Z", (2010, 2, "Y"): "B"}
    dataset = pl.DataFrame({"game_id": ["g1"], "home_ratings_qb": [0.1], "away_ratings_qb": [0.2]})

    # Act
    sides = elo_qb_alt.check_sides(games, starts, dataset)

    # Assert
    assert sides.rows() == [(2010, 2, "B", 0.2)]
