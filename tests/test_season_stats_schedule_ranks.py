"""Tests for the all-time strength-of-schedule leaderboard."""

import polars as pl

from nfl_predictor.season_stats.schedule_ranks import rank_schedules


def _ratings(sos: dict[str, float], games_played: int = 17) -> pl.DataFrame:
    """Return a minimal ratings table with the given schedule strengths."""
    return pl.DataFrame(
        {
            "team": list(sos),
            "sos": list(sos.values()),
            "team_rating": [0.0] * len(sos),
            "games_played": [games_played] * len(sos),
        }
    )


def _two_seasons() -> dict[int, pl.DataFrame]:
    """Return two seasons of ratings."""
    return {
        2024: _ratings({"AAA": 1.5, "BBB": -2.0, "CCC": 0.5}),
        2025: _ratings({"AAA": -3.0, "BBB": 2.5, "CCC": 0.0}),
    }


def test_rank_schedules_ranks_every_team_season_softest_first() -> None:
    # Act
    ranked = rank_schedules(_two_seasons())

    # Assert
    assert ranked.select("season", "team", "softest_rank").rows()[:2] == [
        (2025, "AAA", 1),
        (2024, "BBB", 2),
    ]


def test_rank_schedules_reports_the_hardest_rank_too() -> None:
    # Act
    ranked = rank_schedules(_two_seasons())

    # Assert
    hardest = ranked.filter(pl.col("hardest_rank") == 1).row(0, named=True)
    assert (hardest["season"], hardest["team"]) == (2025, "BBB")


def test_rank_schedules_ignores_ratings_without_schedule_strength() -> None:
    # Arrange
    ratings = {
        2024: pl.DataFrame({"team": ["AAA"], "SaCR": [1.0]}),
        2025: _ratings({"AAA": 1.0, "BBB": -1.0}),
    }

    # Act
    ranked = rank_schedules(ratings)

    # Assert
    assert ranked.get_column("season").unique().to_list() == [2025]


def test_rank_schedules_skips_seasons_still_in_progress() -> None:
    # Arrange
    ratings = {
        2025: _ratings({"AAA": 1.0, "BBB": -1.0}),
        2026: _ratings({"AAA": -9.0, "BBB": 9.0}, games_played=3),
    }

    # Act
    ranked = rank_schedules(ratings)

    # Assert
    assert ranked.get_column("season").unique().to_list() == [2025]


def test_rank_schedules_without_ratings_returns_a_typed_empty_frame() -> None:
    # Act
    ranked = rank_schedules({})

    # Assert
    assert ranked.is_empty()
    assert ranked.columns == ["season", "team", "sos", "softest_rank", "hardest_rank"]
