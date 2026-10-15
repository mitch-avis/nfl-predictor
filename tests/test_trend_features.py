"""Tests for trend/recency feature helpers."""

from __future__ import annotations

from typing import cast

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils

_PHASE_SEASON = 2023
_PHASE_REGULAR_WEEKS = constants.get_regular_season_weeks(_PHASE_SEASON)


def test_build_qb_trends_groups_by_qb_name() -> None:
    """QB trends are computed per QB name, not team."""
    # Arrange
    elo_df = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 2],
            "away_abbr": ["AAA", "BBB"],
            "home_abbr": ["CCC", "DDD"],
            "away_elo_pre": [1500.0, 1510.0],
            "home_elo_pre": [1490.0, 1485.0],
            "away_qb": ["QB1", "QB1"],
            "home_qb": ["QB2", "QB3"],
            "away_qb_elo_pre": [1000.0, 1100.0],
            "home_qb_elo_pre": [900.0, 950.0],
            "away_qb_value_pre": [1.0, 2.0],
            "home_qb_value_pre": [0.5, 0.6],
        }
    )

    # Act
    qb_trends = polars_utils.build_qb_trends(elo_df, season=2023)

    # Assert
    qb1_week2 = qb_trends.filter((pl.col("qb_name") == "QB1") & (pl.col("week") == 2))
    assert qb1_week2.height == 1
    qb1_trend = cast("float", qb1_week2["qb_elo_4wk_trend"][0])
    assert qb1_trend == pytest.approx(100.0)


def test_build_team_elo_trends_use_prior_games_only() -> None:
    """Team ELO trend uses prior games only and is zero for the first game."""
    # Arrange
    elo_df = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 2],
            "away_abbr": ["AAA", "EEE"],
            "home_abbr": ["FFF", "AAA"],
            "away_elo_pre": [1500.0, 1400.0],
            "home_elo_pre": [1490.0, 1510.0],
        }
    )

    # Act
    trends = polars_utils.build_team_elo_trends(elo_df, season=2023)

    # Assert
    aaa_week1 = trends.filter((pl.col("team_abbr") == "AAA") & (pl.col("week") == 1))
    aaa_week2 = trends.filter((pl.col("team_abbr") == "AAA") & (pl.col("week") == 2))
    assert aaa_week1.height == 1
    assert aaa_week2.height == 1
    assert aaa_week1["elo_4wk_trend"][0] == pytest.approx(0.0)
    assert aaa_week2["elo_4wk_trend"][0] == pytest.approx(10.0)


def _team_stat_games(weeks: int) -> pl.DataFrame:
    """Return AAA's first ``weeks`` games of 2023: four quiet games, then big margins."""
    return pl.DataFrame(
        {
            "season": [2023] * weeks,
            "week": list(range(1, weeks + 1)),
            "team_abbr": ["AAA"] * weeks,
            "scoring_margin": [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0][:weeks],
            "turnover_margin": [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0][:weeks],
        }
    )


def _schedule(weeks: list[int], game_type: str = "REG") -> pl.DataFrame:
    """Return AAA's 2023 games against BBB in ``weeks``, played or not."""
    return pl.DataFrame(
        {
            "season": [2023] * len(weeks),
            "week": weeks,
            "game_type": [game_type] * len(weeks),
            "away_abbr": ["AAA"] * len(weeks),
            "home_abbr": ["BBB"] * len(weeks),
        }
    )


def _team_week(trends: pl.DataFrame, team: str, week: int) -> pl.DataFrame:
    return trends.filter((pl.col("team_abbr") == team) & (pl.col("week") == week))


def test_build_team_stat_trends_compare_recent_to_season_mean() -> None:
    """Performance trends compare recent mean to season-to-date mean."""
    # Arrange
    stats = _team_stat_games(6)

    # Act
    trends = polars_utils.build_team_stat_trends(
        stats,
        season=2023,
        stats=["scoring_margin", "turnover_margin"],
        schedule_df=_schedule(list(range(1, 7))),
        window=4,
    )

    # Assert
    week6 = _team_week(trends, "AAA", 6)
    assert week6.height == 1
    scoring_trend = cast("float", week6["scoring_margin_4wk_trend"][0])
    turnover_trend = cast("float", week6["turnover_margin_4wk_trend"][0])
    assert scoring_trend == pytest.approx(0.45, abs=1e-2)
    assert turnover_trend == pytest.approx(0.1, abs=1e-2)


def test_build_team_stat_trends_gives_an_unplayed_week_the_trend_from_earlier_games() -> None:
    """A scheduled week without a played game gets the trend from the team's earlier games."""
    # Arrange
    stats = _team_stat_games(6)

    # Act
    trends = polars_utils.build_team_stat_trends(
        stats,
        season=2023,
        stats=["scoring_margin", "turnover_margin"],
        schedule_df=_schedule(list(range(1, 8))),
        window=4,
    )

    # Assert
    week7 = _team_week(trends, "AAA", 7)
    assert week7.height == 1
    # Weeks 1-6: the last four games average 5.5 against a season mean of 4.0, and turnovers
    # 1.0 against 4/6.
    assert week7["scoring_margin_4wk_trend"][0] == pytest.approx(1.5)
    assert week7["turnover_margin_4wk_trend"][0] == pytest.approx(1.0 - 4.0 / 6.0)


def test_build_team_stat_trends_gives_a_playoff_week_the_full_regular_season_trend() -> None:
    """A playoff week's trend comes from every regular-season game, the only games counted."""
    # Arrange
    stats = _team_stat_games(7)
    playoff_week = constants.get_regular_season_weeks(2023) + 1
    schedule = pl.concat([_schedule(list(range(1, 8))), _schedule([playoff_week], "WC")])

    # Act
    trends = polars_utils.build_team_stat_trends(
        stats,
        season=2023,
        stats=["scoring_margin", "turnover_margin"],
        schedule_df=schedule,
        window=4,
    )

    # Assert
    playoff = _team_week(trends, "AAA", playoff_week)
    assert playoff.height == 1
    # Weeks 1-7: the last four games average 7.75 against a season mean of 34/7, and
    # turnovers 1.5 against 6/7.
    assert playoff["scoring_margin_4wk_trend"][0] == pytest.approx(7.75 - 34.0 / 7.0)
    assert playoff["turnover_margin_4wk_trend"][0] == pytest.approx(1.5 - 6.0 / 7.0)


def test_build_team_stat_trends_keeps_a_game_without_a_value_in_the_window() -> None:
    """A game without a value fills a window slot but counts in neither mean.

    The week right after it has no season mean, so no trend (the week builder publishes 0.0).
    """
    # Arrange
    stats = _team_stat_games(6).with_columns(
        pl.when(pl.col("week") == 3)
        .then(None)
        .otherwise(pl.col("scoring_margin"))
        .alias("scoring_margin")
    )

    # Act
    trends = polars_utils.build_team_stat_trends(
        stats,
        season=2023,
        stats=["scoring_margin"],
        schedule_df=_schedule(list(range(1, 8))),
        window=4,
    )

    # Assert
    aaa = trends.filter(pl.col("team_abbr") == "AAA")
    by_week = dict(zip(aaa["week"], aaa["scoring_margin_4wk_trend"], strict=True))
    assert by_week[4] is None
    # Week 6: weeks 2-5 hold 1, (none), 1, 10 against 13 over four values.
    assert by_week[6] == pytest.approx(12.0 / 3.0 - 13.0 / 4.0)
    # Week 7: weeks 3-6 hold (none), 1, 10, 10 against 23 over five values.
    assert by_week[7] == pytest.approx(21.0 / 3.0 - 23.0 / 5.0)


@pytest.mark.parametrize("missing", ["season", "week", "away_abbr", "home_abbr"])
def test_build_team_stat_trends_rejects_a_schedule_it_cannot_key_on(missing: str) -> None:
    """A schedule without the team-week columns is an error, not a season of 0.0 trends."""
    # Arrange
    schedule = _schedule(list(range(1, 8))).drop(missing)

    # Act & Assert
    with pytest.raises(ValueError, match=missing):
        polars_utils.build_team_stat_trends(
            _team_stat_games(6),
            season=2023,
            stats=["scoring_margin"],
            schedule_df=schedule,
        )


def test_build_team_stat_trends_is_zero_for_a_team_without_an_earlier_game() -> None:
    """A team-week with no earlier game has no trend signal yet, published as 0.0."""
    # Arrange
    stats = _team_stat_games(6)

    # Act
    trends = polars_utils.build_team_stat_trends(
        stats,
        season=2023,
        stats=["scoring_margin", "turnover_margin"],
        schedule_df=_schedule(list(range(1, 8))),
        window=4,
    )

    # Assert
    aaa_week1 = _team_week(trends, "AAA", 1)
    bbb_week7 = _team_week(trends, "BBB", 7)
    assert aaa_week1.select("scoring_margin_4wk_trend", "turnover_margin_4wk_trend").row(0) == (
        0.0,
        0.0,
    )
    assert bbb_week7.select("scoring_margin_4wk_trend", "turnover_margin_4wk_trend").row(0) == (
        0.0,
        0.0,
    )


@pytest.mark.parametrize(
    ("week", "expected_flags"),
    [
        (1, (1, 0, 0)),
        (int((_PHASE_REGULAR_WEEKS / 3.0) + 1), (0, 1, 0)),
        (int((2.0 * _PHASE_REGULAR_WEEKS / 3.0) + 1), (0, 0, 1)),
    ],
    ids=["early", "mid", "late"],
)
def test_add_season_phase_features_buckets(week: int, expected_flags: tuple[int, ...]) -> None:
    """Season-phase buckets align with early/mid/late week ranges."""
    # Arrange
    base = pl.DataFrame({"game_id": [1]})

    # Act
    phased = polars_utils.add_season_phase_features(base, season=_PHASE_SEASON, week=week)

    # Assert
    flags = (
        phased["season_phase_early"][0],
        phased["season_phase_mid"][0],
        phased["season_phase_late"][0],
    )
    assert flags == expected_flags


def test_add_season_phase_features_normalizes_the_week() -> None:
    """The normalized week is the week over the regular season's length."""
    # Arrange
    base = pl.DataFrame({"game_id": [1]})

    # Act
    early = polars_utils.add_season_phase_features(base, season=_PHASE_SEASON, week=1)

    # Assert
    assert early["week_in_season_norm"][0] == pytest.approx(1.0 / _PHASE_REGULAR_WEEKS)


def test_add_season_phase_features_caps_postseason() -> None:
    """Season phase caps at 1.0 for postseason weeks."""
    # Arrange
    base = pl.DataFrame({"game_id": [1]})
    season = 2023
    regular_weeks = constants.get_regular_season_weeks(season)
    postseason_week = regular_weeks + 2

    # Act
    result = polars_utils.add_season_phase_features(base, season=season, week=postseason_week)

    # Assert
    assert result["week_in_season_norm"][0] == pytest.approx(1.0)
    assert result["season_phase_early"][0] == 0
    assert result["season_phase_mid"][0] == 0
    assert result["season_phase_late"][0] == 1
