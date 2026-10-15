"""The ETL warns when ``qb_elos.csv`` has no rows for the week being predicted.

Without them the week builder falls back for that week's games (``week_rows._merge_elo``): each
team's latest listed pre-game rating, a team ELO trend of 0.0 and no quarterback trends, values a
completed row never carries once its teams have played. The warning names the season and the
week and says to refresh the file before the run; it changes no value.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor import constants, data_collection
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars import loaders, ratings_build, week_rows
from tests.fakes import returning

if TYPE_CHECKING:
    import pytest

_SEASON = 2026
_PREDICTED_WEEK = 6
_EXPECTED_WARNING = (
    "qb_elos.csv has no rows for 2026 week 6, the week being predicted: its 2 unplayed games "
    "fall back to each team's latest listed ELO rating, a 0.0 ELO trend and no quarterback "
    "trends. Refresh data/qb_elos.csv before the run."
)


def _schedule(*, played_through: int) -> pl.DataFrame:
    """Return 2026 weeks 5-7, two games a week, played through ``played_through``."""
    weeks = [5, 5, 6, 6, 7, 7]
    played = [week <= played_through for week in weeks]
    return pl.DataFrame(
        {
            "game_id": [f"2026_{week:02d}_{index}" for index, week in enumerate(weeks)],
            "season": [_SEASON] * len(weeks),
            "week": weeks,
            "game_type": ["REG"] * len(weeks),
            "away_abbr": ["AAA", "CCC"] * 3,
            "home_abbr": ["BBB", "DDD"] * 3,
            "away_score": [20 if done else None for done in played],
            "home_score": [17 if done else None for done in played],
        }
    )


def _elo(*, listed_through: int) -> pl.DataFrame:
    """Return the ELO rows of the schedule's games through week ``listed_through``."""
    games = _schedule(played_through=0).filter(pl.col("week") <= listed_through)
    return games.select(
        "season",
        "week",
        "away_abbr",
        "home_abbr",
        pl.lit(1500.0).alias("away_elo_pre"),
        pl.lit(1510.0).alias("home_elo_pre"),
    )


def _recorded_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the project logger's warnings, formatted."""
    messages: list[str] = []

    def record(message: str, *args: object) -> None:
        messages.append(message % args if args else message)

    monkeypatch.setattr(log, "warning", record)
    return messages


def test_warns_when_the_elo_rows_stop_before_the_week_being_predicted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Week 7 is not listed either, but only the week being predicted is checked."""
    # Arrange
    warnings = _recorded_warnings(monkeypatch)

    # Act
    week_rows.warn_if_elo_lacks_week(
        _elo(listed_through=5),
        _schedule(played_through=5),
        season=_SEASON,
        week=_PREDICTED_WEEK,
    )

    # Assert
    assert warnings == [_EXPECTED_WARNING]


def test_stays_quiet_when_the_elo_rows_list_the_week_being_predicted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refreshed file lists the coming week, and the weeks after it are never listed yet."""
    # Arrange
    warnings = _recorded_warnings(monkeypatch)

    # Act
    week_rows.warn_if_elo_lacks_week(
        _elo(listed_through=6),
        _schedule(played_through=5),
        season=_SEASON,
        week=_PREDICTED_WEEK,
    )

    # Assert
    assert warnings == []


def test_stays_quiet_when_the_week_being_predicted_has_no_game_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With every game of the week played there is nothing to predict, so nothing falls back."""
    # Arrange
    warnings = _recorded_warnings(monkeypatch)

    # Act
    week_rows.warn_if_elo_lacks_week(
        _elo(listed_through=5),
        _schedule(played_through=6),
        season=_SEASON,
        week=_PREDICTED_WEEK,
    )

    # Assert
    assert warnings == []


def test_stays_quiet_without_any_elo_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no ELO rows at all nothing falls back, and the loader already warns about that."""
    # Arrange
    warnings = _recorded_warnings(monkeypatch)

    # Act
    week_rows.warn_if_elo_lacks_week(
        pl.DataFrame(),
        _schedule(played_through=5),
        season=_SEASON,
        week=_PREDICTED_WEEK,
    )

    # Assert
    assert warnings == []


def test_load_sources_warns_when_the_elo_file_lacks_the_week_being_predicted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source load checks the ELO rows against the current week, once."""
    # Arrange
    schedule = _schedule(played_through=5)

    def load_schedule(seasons: list[int], **_kwargs: object) -> pl.DataFrame:
        return schedule.filter(pl.col("season").is_in(seasons))

    def build_team_game_frame(stats: pl.DataFrame, *_a: object) -> pl.DataFrame:
        return stats

    def add_scoring_data(df: pl.DataFrame, _sched: pl.DataFrame) -> pl.DataFrame:
        return df

    def add_opponent_stats(df: pl.DataFrame) -> pl.DataFrame:
        return df

    monkeypatch.setattr(polars_utils, "get_current_nfl_week", lambda: (_SEASON, _PREDICTED_WEEK))
    monkeypatch.setattr(polars_utils, "load_schedule", load_schedule)
    monkeypatch.setattr(polars_utils, "load_team_stats", returning(pl.DataFrame()))
    monkeypatch.setattr(polars_utils, "load_pbp", returning(loaders._empty_pbp_frame()))
    monkeypatch.setattr(data_collection, "_build_team_game_frame", build_team_game_frame)
    monkeypatch.setattr(polars_utils, "add_scoring_data_to_team_stats", add_scoring_data)
    monkeypatch.setattr(polars_utils, "add_per_game_opponent_stats", add_opponent_stats)
    monkeypatch.setattr(polars_utils, "load_elo_ratings", returning(_elo(listed_through=5)))
    monkeypatch.setattr(polars_utils, "load_raw_elo_data", pl.DataFrame)
    monkeypatch.setattr(ratings_build, "load_ratings_sources", returning(None))
    warnings = _recorded_warnings(monkeypatch)
    config = dataclasses.replace(
        data_collection._default_config([_SEASON]), line_source=constants.LINE_SOURCE_STORED
    )

    # Act
    sources = data_collection._load_sources([_SEASON], config)

    # Assert
    assert [message for message in warnings if "qb_elos.csv" in message] == [_EXPECTED_WARNING]
    assert sources.elo_df.equals(_elo(listed_through=5))
