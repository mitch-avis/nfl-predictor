"""Tests for pre-week quarterback rows that depend on what was known before the week."""

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from nfl_predictor.ratings.pre_week import RatingsHistory

_TEAMS = ("AAA", "BBB", "CCC", "DDD")
_FIRST = 2000
_SEASON = 2001
_WEEKS = 6


def _schedule() -> list[list[tuple[str, str]]]:
    """Return a double round robin of four teams by week, ``(home, away)`` pairs."""
    rounds = [
        [("AAA", "BBB"), ("CCC", "DDD")],
        [("AAA", "CCC"), ("BBB", "DDD")],
        [("AAA", "DDD"), ("BBB", "CCC")],
    ]
    return rounds + [[(away, home) for home, away in games] for games in rounds]


def _team_games(season: int) -> pl.DataFrame:
    """Return one season of team-game rows with deterministic, uneven values."""
    rows: list[dict[str, object]] = []
    for week_index, games in enumerate(_schedule()):
        week = week_index + 1
        for home, away in games:
            for team, opponent, is_home in ((home, away, True), (away, home, False)):
                index = _TEAMS.index(team)
                plays = 60 + (index * 3 + week) % 7
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{away}_{home}",
                        "week": week,
                        "team": team,
                        "opponent_team": opponent,
                        "is_home": is_home,
                        "offensive_snaps": plays,
                        "offensive_epa": 0.02 * plays * ((index * 5 + week + season) % 7 - 3),
                        "st_plays": 12,
                        "st_epa": 0.1 * ((index + week) % 5 - 2),
                        "point_margin": (index * 7 + week) % 15 - 7,
                    }
                )
    return pl.DataFrame(rows).sort("team", "week", "game_id")


def _qb_games(season: int) -> pl.DataFrame:
    """Return one passer per team-game, plus AAA's trick-play receiver in weeks 1 and 2."""
    rows: list[dict[str, object]] = []
    for row in _team_games(season).iter_rows(named=True):
        week = int(row["week"])
        passers = [(f"qb-{row['team']}", 30 + week % 4, 0.05 * (_TEAMS.index(row["team"]) - 1.5))]
        if row["team"] == "AAA" and week <= 2:
            passers.append(("wr-AAA", 1, 2.0))
        for qb_id, dropbacks, rate in passers:
            rows.append(
                {
                    "game_id": row["game_id"],
                    "week": week,
                    "team": row["team"],
                    "opponent_team": row["opponent_team"],
                    "is_home": row["is_home"],
                    "qb_id": qb_id,
                    "qb_name": f"{qb_id} Passer",
                    "qb_dropbacks": dropbacks,
                    "qb_dropback_epa": rate * dropbacks + 0.01 * week,
                    "qb_attempts": dropbacks - 1,
                }
            )
    return pl.DataFrame(rows).sort("team", "week", "game_id", "qb_name")


def _without_receiver(season: int, week: int) -> pl.DataFrame:
    """Return the season's rows as known before ``week``: the receiver is not a quarterback."""
    del week
    return _qb_games(season).filter(pl.col("qb_id") != "wr-AAA")


def test_pre_week_quarterbacks_read_the_rows_known_before_the_week() -> None:
    # Arrange
    history = RatingsHistory(
        _team_games, _qb_games, first_season=_FIRST, load_qb_games_before=_without_receiver
    )

    # Act
    quarterbacks = history.pre_week_quarterbacks(_SEASON, 4)

    # Assert
    assert quarterbacks.get_column("qb_id").to_list() == [f"qb-{team}" for team in _TEAMS]


def test_pre_week_quarterbacks_read_the_season_rows_without_a_loader_for_the_week() -> None:
    # Arrange
    history = RatingsHistory(_team_games, _qb_games, first_season=_FIRST)

    # Act
    quarterbacks = history.pre_week_quarterbacks(_SEASON, 4)

    # Assert
    assert "wr-AAA" in quarterbacks.get_column("qb_id").to_list()


def test_pre_week_quarterbacks_ask_for_the_season_and_week_being_rated_only() -> None:
    # Arrange
    calls: list[tuple[int, int]] = []

    def recording(season: int, week: int) -> pl.DataFrame:
        calls.append((season, week))
        return _without_receiver(season, week)

    history = RatingsHistory(
        _team_games, _qb_games, first_season=_FIRST, load_qb_games_before=recording
    )

    # Act
    history.pre_week(_SEASON, 5, _TEAMS)

    # Assert
    assert calls == [(_SEASON, 5)]


def test_pre_week_quarterbacks_take_the_penalty_from_the_previous_season_rows() -> None:
    # Arrange
    with_loader = RatingsHistory(
        _team_games, _qb_games, first_season=_FIRST, load_qb_games_before=_without_receiver
    )
    plain = RatingsHistory(_team_games, _qb_games, first_season=_FIRST)

    # Act
    penalty = with_loader.qb_penalty(_SEASON)

    # Assert
    assert penalty == plain.qb_penalty(_SEASON)


def test_pre_week_quarterbacks_ignore_rows_the_loader_returns_from_the_week_on() -> None:
    # Arrange
    def with_later_rows(season: int, week: int) -> pl.DataFrame:
        rows = _without_receiver(season, week)
        later = pl.col("week") >= week
        return rows.with_columns(
            pl.when(later)
            .then(pl.col("qb_dropback_epa") * -9.0 + 3.0)
            .otherwise(pl.col("qb_dropback_epa"))
            .alias("qb_dropback_epa")
        )

    expected = RatingsHistory(
        _team_games, _qb_games, first_season=_FIRST, load_qb_games_before=_without_receiver
    ).pre_week_quarterbacks(_SEASON, 4)
    history = RatingsHistory(
        _team_games, _qb_games, first_season=_FIRST, load_qb_games_before=with_later_rows
    )

    # Act
    quarterbacks = history.pre_week_quarterbacks(_SEASON, 4)

    # Assert
    assert_frame_equal(quarterbacks, expected, check_exact=True)
