"""Tests for the team-rating sources a power ranking can rank on."""

from __future__ import annotations

import itertools
import json
from typing import TYPE_CHECKING

import polars as pl
import pytest
from tests.rating_source_fixtures import fold_frame, write_fold_checkpoints

from nfl_predictor import constants
from nfl_predictor.ratings.team_rating import TEAM_RATING_COLUMNS
from nfl_predictor.reporting import rating_sources

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

STRENGTHS = {"AAA": 4.0, "BBB": 1.5, "CCC": -0.5, "DDD": -5.0}
HOME_TERM = 2.0


def _round_robin(strengths: dict[str, float], *, neutral_game: tuple[str, str] | None = None):
    """Return every pairing once, home side alternating, margins additive in the strengths."""
    rows = []
    for index, (first, second) in enumerate(itertools.combinations(sorted(strengths), 2)):
        home, away = (first, second) if index % 2 == 0 else (second, first)
        neutral = (home, away) == neutral_game
        rows.append(
            {
                "away_abbr": away,
                "home_abbr": home,
                "neutral": int(neutral),
                "predicted_margin": strengths[home]
                - strengths[away]
                + (0.0 if neutral else HOME_TERM),
            }
        )
    return pl.DataFrame(rows)


def test_fit_model_ratings_additive_margins_recovers_the_strengths() -> None:
    # Arrange
    games = _round_robin(STRENGTHS)

    # Act
    fit = rating_sources.fit_model_ratings(games, sorted(STRENGTHS))

    # Assert
    for team, strength in STRENGTHS.items():
        assert fit.ratings[team] == pytest.approx(strength, abs=1e-4)
    assert fit.home_term == pytest.approx(HOME_TERM, abs=1e-4)


def test_fit_model_ratings_neutral_site_game_carries_no_home_term() -> None:
    # Arrange
    games = _round_robin(STRENGTHS, neutral_game=("AAA", "BBB"))

    # Act
    fit = rating_sources.fit_model_ratings(games, sorted(STRENGTHS))

    # Assert
    for team, strength in STRENGTHS.items():
        assert fit.ratings[team] == pytest.approx(strength, abs=1e-4)


def test_fit_model_ratings_team_without_a_game_rates_average() -> None:
    # Arrange
    games = _round_robin(STRENGTHS)
    teams = [*sorted(STRENGTHS), "EEE"]

    # Act
    fit = rating_sources.fit_model_ratings(games, teams)

    # Assert
    assert fit.ratings["EEE"] == pytest.approx(0.0, abs=1e-9)
    assert sum(fit.ratings.values()) == pytest.approx(0.0, abs=1e-6)


def test_fit_model_ratings_one_game_each_splits_the_margin_around_the_home_term() -> None:
    # Arrange
    games = pl.DataFrame(
        {
            "away_abbr": ["BBB", "DDD"],
            "home_abbr": ["AAA", "CCC"],
            "neutral": [0, 0],
            "predicted_margin": [9.0, 1.0],
        }
    )

    # Act
    fit = rating_sources.fit_model_ratings(games, ["AAA", "BBB", "CCC", "DDD"])

    # Assert
    assert fit.home_term == pytest.approx(5.0, abs=1e-4)
    assert fit.ratings["AAA"] == pytest.approx(2.0, abs=1e-4)
    assert fit.ratings["BBB"] == pytest.approx(-2.0, abs=1e-4)
    assert fit.ratings["CCC"] == pytest.approx(-2.0, abs=1e-4)
    assert fit.ratings["DDD"] == pytest.approx(2.0, abs=1e-4)


def test_fit_model_ratings_without_games_raises() -> None:
    # Arrange
    games = _round_robin(STRENGTHS).head(0)

    # Act & Assert
    with pytest.raises(ValueError, match="no games"):
        rating_sources.fit_model_ratings(games, sorted(STRENGTHS))


def _season_predictions(season: int, weeks: dict[int, list[tuple[str, str, float]]]):
    """Return prediction rows for ``season``: per week, (away, home, predicted margin) games."""
    return pl.DataFrame(
        [
            {
                "game_id": f"{season}_{week:02d}_{away}_{home}",
                "season": season,
                "week": week,
                "away_abbr": away,
                "home_abbr": home,
                "neutral": 0,
                "predicted_margin": margin,
            }
            for week, games in weeks.items()
            for away, home, margin in games
        ]
    )


WEEK_GAMES = {
    1: [("BBB", "AAA", 6.0), ("DDD", "CCC", -1.0)],
    2: [("CCC", "AAA", 9.0), ("DDD", "BBB", 3.0)],
    3: [("AAA", "DDD", -100.0), ("BBB", "CCC", 100.0)],
}
TEAMS = ["AAA", "BBB", "CCC", "DDD"]


def _by_team(ratings: pl.DataFrame, season: int, week: int) -> dict[str, float]:
    rows = ratings.filter((pl.col("season") == season) & (pl.col("week") == week))
    return dict(zip(rows["team"].to_list(), rows["rating"].to_list(), strict=True))


def test_model_ratings_by_week_fits_only_the_games_before_each_week() -> None:
    # Arrange
    predictions = _season_predictions(2020, WEEK_GAMES)
    before_two = rating_sources.fit_model_ratings(predictions.filter(pl.col("week") < 2), TEAMS)
    before_three = rating_sources.fit_model_ratings(predictions.filter(pl.col("week") < 3), TEAMS)

    # Act
    ratings = rating_sources.model_ratings_by_week(predictions)

    # Assert
    assert sorted(set(ratings.get_column("week").to_list())) == [2, 3]
    assert _by_team(ratings, 2020, 2) == pytest.approx(before_two.ratings)
    assert _by_team(ratings, 2020, 3) == pytest.approx(before_three.ratings)


def test_model_ratings_by_week_keeps_each_season_to_its_own_games() -> None:
    # Arrange
    first = _season_predictions(2020, WEEK_GAMES)
    second = _season_predictions(
        2021, {1: [("BBB", "AAA", -4.0), ("DDD", "CCC", 2.0)], 2: [("CCC", "AAA", 1.0)]}
    )
    expected = rating_sources.fit_model_ratings(second.filter(pl.col("week") == 1), TEAMS)

    # Act
    ratings = rating_sources.model_ratings_by_week(pl.concat([first, second]))

    # Assert
    assert _by_team(ratings, 2021, 2) == pytest.approx(expected.ratings)


def test_model_ratings_by_week_rates_a_listed_team_without_a_game_as_average() -> None:
    # Arrange
    predictions = _season_predictions(2020, WEEK_GAMES)
    teams = {2020: [*TEAMS, "EEE"]}

    # Act
    ratings = rating_sources.model_ratings_by_week(predictions, teams_by_season=teams)

    # Assert
    assert _by_team(ratings, 2020, 2)["EEE"] == pytest.approx(0.0, abs=1e-9)


def test_carryover_factor_is_the_slope_through_the_origin() -> None:
    # Arrange
    previous = {"AAA": 6.0, "BBB": -1.0, "CCC": -5.0}
    week_one = {team: 0.6 * rating for team, rating in previous.items()}

    # Act
    factor = rating_sources.carryover_factor(week_one, previous)

    # Assert
    assert factor == pytest.approx(0.6)


def test_carryover_factor_reads_only_the_teams_in_both_seasons() -> None:
    # Arrange
    previous = {"AAA": 6.0, "BBB": -6.0, "OLD": 50.0}
    week_one = {"AAA": 3.0, "BBB": -3.0, "NEW": -40.0}

    # Act
    factor = rating_sources.carryover_factor(week_one, previous)

    # Assert
    assert factor == pytest.approx(0.5)


def test_carryover_factor_without_a_previous_spread_raises() -> None:
    # Arrange
    previous = {"AAA": 0.0, "BBB": 0.0}
    week_one = {"AAA": 1.0, "BBB": -1.0}

    # Act & Assert
    with pytest.raises(ValueError, match="spread"):
        rating_sources.carryover_factor(week_one, previous)


def test_week_one_model_ratings_regresses_last_season_and_centers() -> None:
    # Arrange
    previous_final = {"AAA": 4.0, "BBB": -1.0, "CCC": -1.0, "OLD": 9.0}
    teams = ["AAA", "BBB", "CCC", "NEW"]

    # Act
    ratings = rating_sources.week_one_model_ratings(previous_final, 0.5, teams)

    # Assert
    assert ratings == pytest.approx({"AAA": 1.75, "BBB": -0.75, "CCC": -0.75, "NEW": -0.25})


def test_ported_column_names_match_the_ratings_tables_and_features() -> None:
    # Act
    names = (rating_sources.PORTED_RATING_COLUMN, rating_sources.PORTED_FEATURE)

    # Assert
    assert names[0] in TEAM_RATING_COLUMNS
    assert names[1] in constants.RATINGS_FEATURE_STATS


def _by_week_table() -> pl.DataFrame:
    """Return a published ``ratings_by_week`` table: ratings through weeks 1 and 2."""
    return pl.DataFrame(
        {
            "week": [1, 1, 2, 2, 2],
            "team": ["AAA", "BBB", "AAA", "BBB", "CCC"],
            "games_played": [1, 1, 2, 2, 1],
            "team_rating": [3.0, -3.0, 2.5, -2.0, -0.5],
        }
    )


def test_ported_ratings_from_by_week_rates_the_week_after_the_games() -> None:
    # Arrange
    table = _by_week_table()

    # Act
    ratings = rating_sources.ported_ratings_from_by_week(table, season=2020)

    # Assert
    assert _by_team(ratings, 2020, 2) == {"AAA": 3.0, "BBB": -3.0}
    assert _by_team(ratings, 2020, 3) == {"AAA": 2.5, "BBB": -2.0, "CCC": -0.5}


def _feature_rows() -> pl.DataFrame:
    """Return game rows carrying the ratings feature group's pre-week team ratings."""
    return pl.DataFrame(
        {
            "season": [2020, 2020, 2020],
            "week": [3, 3, 3],
            "away_abbr": ["BBB", "DDD", "EEE"],
            "home_abbr": ["AAA", "CCC", "FFF"],
            "away_ratings_team": [-2.0, 1.0, None],
            "home_ratings_team": [2.5, -0.5, None],
        }
    )


def test_ported_ratings_from_games_reads_both_sides_and_skips_nulls() -> None:
    # Arrange
    games = _feature_rows()

    # Act
    ratings = rating_sources.ported_ratings_from_games(games)

    # Assert
    assert _by_team(ratings, 2020, 3) == {"AAA": 2.5, "BBB": -2.0, "CCC": -0.5, "DDD": 1.0}


def test_combine_ported_ratings_fills_byes_and_counts_disagreements() -> None:
    # Arrange
    from_games = rating_sources.ported_ratings_from_games(_feature_rows())
    by_week = _by_week_table().with_columns(
        pl.when(pl.col("team") == "CCC")
        .then(-0.25)
        .otherwise(pl.col("team_rating"))
        .alias("team_rating")
    )
    bye_team = pl.DataFrame(
        {"week": [2], "team": ["GGG"], "games_played": [1], "team_rating": [7.0]}
    )
    from_by_week = rating_sources.ported_ratings_from_by_week(
        pl.concat([by_week, bye_team]), season=2020
    )

    # Act
    combined = rating_sources.combine_ported_ratings(from_games, from_by_week)

    # Assert
    week_three = _by_team(combined.ratings, 2020, 3)
    assert week_three == {"AAA": 2.5, "BBB": -2.0, "CCC": -0.5, "DDD": 1.0, "GGG": 7.0}
    assert combined.compared == 3
    assert combined.disagreements == 1


def _rating_rows(values: dict[str, float], week: int = 5) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2020] * len(values),
            "week": [week] * len(values),
            "team": list(values),
            "rating": list(values.values()),
        },
        schema=rating_sources.RATING_SCHEMA,
    )


def test_blend_ratings_weights_the_ported_rating_by_w() -> None:
    # Arrange
    ported = _rating_rows({"AAA": 4.0, "BBB": -2.0})
    model = _rating_rows({"AAA": 2.0, "BBB": 1.0})

    # Act
    blended = rating_sources.blend_ratings(ported, model, 0.3)

    # Assert
    assert _by_team(blended, 2020, 5) == pytest.approx({"AAA": 2.6, "BBB": 0.1})


def test_blend_ratings_needs_both_sources_for_a_team_week() -> None:
    # Arrange
    ported = _rating_rows({"AAA": 4.0, "BBB": -2.0})
    model = _rating_rows({"AAA": 2.0, "CCC": 1.0})

    # Act
    blended = rating_sources.blend_ratings(ported, model, 0.5)

    # Assert
    assert _by_team(blended, 2020, 5) == pytest.approx({"AAA": 3.0})


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_blend_ratings_weight_outside_zero_to_one_raises(weight: float) -> None:
    # Arrange
    ratings = _rating_rows({"AAA": 1.0})

    # Act & Assert
    with pytest.raises(ValueError, match="between 0 and 1"):
        rating_sources.blend_ratings(ratings, ratings, weight)


def _fold_frames() -> list[pd.DataFrame]:
    return [
        fold_frame(2020, 2, [("CCC", "AAA", 9.0), ("DDD", "BBB", 3.0)]),
        fold_frame(2020, 1, [("BBB", "AAA", 6.0), ("DDD", "CCC", -1.0)]),
    ]


def test_read_fold_predictions_from_a_run_directory_reads_every_fold(tmp_path: Path) -> None:
    # Arrange
    checkpoints = write_fold_checkpoints(tmp_path / "wf_checkpoints" / "abc", _fold_frames())
    run_dir = tmp_path / "ratings_anchored_seed42"
    run_dir.mkdir()
    metadata = {
        "dataset_hash": "d" * 64,
        "git_commit_hash": "0123456",
        "config": {"random_seed": 42, "checkpoint": {"dir": str(checkpoints)}},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    # Act
    read = rating_sources.read_fold_predictions(run_dir)

    # Assert
    assert read.predictions.columns == list(rating_sources.PREDICTION_COLUMNS)
    assert read.predictions.get_column("game_id").to_list() == [
        "2020_01_BBB_AAA",
        "2020_01_DDD_CCC",
        "2020_02_CCC_AAA",
        "2020_02_DDD_BBB",
    ]
    assert read.predictions.get_column("predicted_margin").to_list() == [6.0, -1.0, 9.0, 3.0]
    assert read.folds == 2
    assert read.checkpoint_dir == checkpoints
    assert read.metadata["dataset_hash"] == "d" * 64


def test_read_fold_predictions_from_a_checkpoint_directory(tmp_path: Path) -> None:
    # Arrange
    checkpoints = write_fold_checkpoints(tmp_path / "abc", _fold_frames())

    # Act
    read = rating_sources.read_fold_predictions(checkpoints)

    # Assert
    assert read.predictions.height == 4
    assert read.metadata == {}


def test_read_fold_predictions_repeated_game_raises(tmp_path: Path) -> None:
    # Arrange
    frames = _fold_frames()
    frames.append(fold_frame(2020, 3, [("CCC", "AAA", 1.0)]).assign(game_id="2020_01_BBB_AAA"))
    checkpoints = write_fold_checkpoints(tmp_path / "abc", frames)

    # Act & Assert
    with pytest.raises(ValueError, match="more than once"):
        rating_sources.read_fold_predictions(checkpoints)


def _write_ratings_dir(directory: Path) -> Path:
    """Write 2024's weekly history (weeks 1-2) and 2023's season ratings."""
    directory.mkdir()
    by_week = pl.DataFrame(
        {
            "week": [1, 1, 1, 1, 2, 2, 2, 2, 2],
            "team": ["AAA", "BBB", "CCC", "DDD", "AAA", "BBB", "CCC", "DDD", "EEE"],
            "games_played": [1, 1, 1, 1, 2, 2, 2, 2, 1],
            "team_rating": [3.0, -3.0, 1.0, -1.0, 2.5, -2.0, 0.5, -1.5, 0.5],
        }
    )
    by_week.write_parquet(directory / "2024_ratings_by_week.parquet")
    season = pl.DataFrame({"team": ["AAA", "BBB", "CCC"], "team_rating": [6.0, -2.0, -4.0]})
    season.write_parquet(directory / "2023_ratings.parquet")
    return directory


def _write_game_rows(path: Path) -> Path:
    """Write ML game rows for 2024 weeks 1 and 3 with the pre-week team ratings."""
    pl.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 1, 3, 3],
            "away_abbr": ["BBB", "DDD", "BBB", "DDD"],
            "home_abbr": ["AAA", "CCC", "AAA", "CCC"],
            "away_ratings_team": [-1.2, 0.1, -2.0, -1.5],
            "home_ratings_team": [2.4, -1.3, 2.5, 0.5],
        }
    ).write_csv(path)
    return path


def test_ported_ratings_for_week_fills_a_bye_from_the_weekly_history(tmp_path: Path) -> None:
    # Arrange
    ratings_dir = _write_ratings_dir(tmp_path / "ratings")
    game_rows = _write_game_rows(tmp_path / "games.csv")

    # Act
    ratings = rating_sources.ported_ratings_for_week(ratings_dir, game_rows, season=2024, week=3)

    # Assert
    assert ratings == {"AAA": 2.5, "BBB": -2.0, "CCC": 0.5, "DDD": -1.5, "EEE": 0.5}


def test_ported_ratings_for_week_one_reads_the_preseason_prior(tmp_path: Path) -> None:
    # Arrange
    ratings_dir = _write_ratings_dir(tmp_path / "ratings")
    game_rows = _write_game_rows(tmp_path / "games.csv")

    # Act
    ratings = rating_sources.ported_ratings_for_week(ratings_dir, game_rows, season=2024, week=1)

    # Assert
    assert ratings == {"AAA": 2.4, "BBB": -1.2, "CCC": -1.3, "DDD": 0.1}


def test_ported_ratings_for_a_week_past_the_history_takes_its_last_week(tmp_path: Path) -> None:
    # Arrange
    ratings_dir = _write_ratings_dir(tmp_path / "ratings")
    game_rows = _write_game_rows(tmp_path / "games.csv")

    # Act
    ratings = rating_sources.ported_ratings_for_week(ratings_dir, game_rows, season=2024, week=20)

    # Assert
    assert ratings == {"AAA": 2.5, "BBB": -2.0, "CCC": 0.5, "DDD": -1.5, "EEE": 0.5}


def test_ported_ratings_for_week_without_any_source_raises(tmp_path: Path) -> None:
    # Arrange
    ratings_dir = _write_ratings_dir(tmp_path / "ratings")
    game_rows = _write_game_rows(tmp_path / "games.csv")

    # Act & Assert
    with pytest.raises(rating_sources.RatingSourceUnavailableError, match="2025"):
        rating_sources.ported_ratings_for_week(ratings_dir, game_rows, season=2025, week=3)


def test_season_ported_ratings_reads_the_published_season_table(tmp_path: Path) -> None:
    # Arrange
    ratings_dir = _write_ratings_dir(tmp_path / "ratings")

    # Act
    ratings = rating_sources.season_ported_ratings(ratings_dir, 2023)

    # Assert
    assert ratings == {"AAA": 6.0, "BBB": -2.0, "CCC": -4.0}


def test_read_predictions_joins_sources_and_refuses_a_repeated_game(tmp_path: Path) -> None:
    # Arrange
    first = write_fold_checkpoints(tmp_path / "first", _fold_frames()[:1])
    second = write_fold_checkpoints(tmp_path / "second", _fold_frames())

    # Act & Assert
    with pytest.raises(ValueError, match="2020_02_CCC_AAA"):
        rating_sources.read_predictions([first, second])


def test_model_ratings_for_week_one_carries_the_last_season_forward() -> None:
    # Arrange
    predictions = _season_predictions(2020, WEEK_GAMES)
    final = rating_sources.fit_model_ratings(predictions, TEAMS)
    expected = rating_sources.week_one_model_ratings(final.ratings, 0.5, TEAMS)

    # Act
    ratings = rating_sources.model_ratings_for_week(
        predictions, season=2021, week=1, teams=TEAMS, carryover=0.5
    )

    # Assert
    assert ratings == pytest.approx(expected)


def test_model_ratings_for_week_one_leave_out_last_seasons_playoffs() -> None:
    # Arrange
    regular_season = _season_predictions(2020, WEEK_GAMES)
    playoffs = _season_predictions(2020, {19: [("AAA", "DDD", 60.0), ("CCC", "BBB", -60.0)]})
    final = rating_sources.fit_model_ratings(regular_season, TEAMS)
    expected = rating_sources.week_one_model_ratings(final.ratings, 0.5, TEAMS)

    # Act
    ratings = rating_sources.model_ratings_for_week(
        pl.concat([regular_season, playoffs]), season=2021, week=1, teams=TEAMS, carryover=0.5
    )

    # Assert
    assert ratings == pytest.approx(expected)


def test_model_ratings_for_a_later_week_fits_the_season_so_far() -> None:
    # Arrange
    predictions = _season_predictions(2020, WEEK_GAMES)
    expected = rating_sources.fit_model_ratings(predictions.filter(pl.col("week") < 3), TEAMS)

    # Act
    ratings = rating_sources.model_ratings_for_week(predictions, season=2020, week=3, teams=TEAMS)

    # Assert
    assert ratings == pytest.approx(expected.ratings)


def test_model_ratings_for_week_one_without_a_carryover_raises() -> None:
    # Arrange
    predictions = _season_predictions(2020, WEEK_GAMES)

    # Act & Assert
    with pytest.raises(rating_sources.RatingSourceUnavailableError, match="carryover"):
        rating_sources.model_ratings_for_week(predictions, season=2021, week=1, teams=TEAMS)


def test_read_fold_predictions_missing_column_raises(tmp_path: Path) -> None:
    # Arrange
    frames = [frame.drop(columns="predicted_margin") for frame in _fold_frames()]
    checkpoints = write_fold_checkpoints(tmp_path / "abc", frames)

    # Act & Assert
    with pytest.raises(ValueError, match="predicted_margin"):
        rating_sources.read_fold_predictions(checkpoints)


def test_read_fold_predictions_without_a_neutral_column_is_refused(tmp_path: Path) -> None:
    # Arrange
    frames = [frame.drop(columns="neutral") for frame in _fold_frames()]
    checkpoints = write_fold_checkpoints(tmp_path / "abc", frames)

    # Act & Assert
    with pytest.raises(rating_sources.RatingSourceUnavailableError, match="neutral"):
        rating_sources.read_fold_predictions(checkpoints)


def test_fit_model_ratings_with_a_missing_margin_raises() -> None:
    # Arrange
    games = _round_robin(STRENGTHS).with_columns(
        pl.when(pl.col("home_abbr") == "AAA")
        .then(None)
        .otherwise(pl.col("predicted_margin"))
        .alias("predicted_margin")
    )

    # Act & Assert
    with pytest.raises(ValueError, match="margin for every game"):
        rating_sources.fit_model_ratings(games, sorted(STRENGTHS))


def test_ported_ratings_from_games_two_ratings_for_a_team_week_raise() -> None:
    # Arrange
    games = pl.concat(
        [
            _feature_rows(),
            _feature_rows().head(1).with_columns(pl.lit(9.0).alias("home_ratings_team")),
        ]
    )

    # Act & Assert
    with pytest.raises(ValueError, match="two different pre-week ratings"):
        rating_sources.ported_ratings_from_games(games)


def test_read_fold_predictions_from_a_run_without_folds_raises(tmp_path: Path) -> None:
    # Arrange
    checkpoints = tmp_path / "empty"
    checkpoints.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metadata = {"config": {"checkpoint": {"dir": str(checkpoints)}}}
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    # Act & Assert
    with pytest.raises(ValueError, match="no fold checkpoints"):
        rating_sources.read_fold_predictions(run_dir)


def test_season_ported_ratings_without_the_table_raises(tmp_path: Path) -> None:
    # Arrange
    ratings_dir = _write_ratings_dir(tmp_path / "ratings")

    # Act & Assert
    with pytest.raises(rating_sources.RatingSourceUnavailableError, match="2019_ratings"):
        rating_sources.season_ported_ratings(ratings_dir, 2019)


def test_model_ratings_for_a_week_without_earlier_games_raises() -> None:
    # Arrange
    predictions = _season_predictions(2020, WEEK_GAMES)

    # Act & Assert
    with pytest.raises(rating_sources.RatingSourceUnavailableError, match="2021"):
        rating_sources.model_ratings_for_week(predictions, season=2021, week=4, teams=TEAMS)


def _shifted(predictions: pl.DataFrame, shift: float) -> pl.DataFrame:
    return predictions.with_columns(pl.col("predicted_margin") + shift)


def test_average_predictions_averages_each_games_margin_over_the_runs() -> None:
    # Arrange
    first = _season_predictions(2020, WEEK_GAMES)
    second = _shifted(first, 2.0).reverse()

    # Act
    averaged = rating_sources.average_predictions([first, second])

    # Assert
    assert averaged.get_column("game_id").to_list() == first.get_column("game_id").to_list()
    expected = (first.get_column("predicted_margin") + 1.0).to_list()
    assert averaged.get_column("predicted_margin").to_list() == pytest.approx(expected)


def test_average_predictions_refuses_runs_of_different_games() -> None:
    # Arrange
    first = _season_predictions(2020, WEEK_GAMES)
    second = first.head(first.height - 1)

    # Act & Assert
    with pytest.raises(ValueError, match="same games"):
        rating_sources.average_predictions([first, second])


def test_average_predictions_refuses_runs_that_disagree_on_a_games_teams() -> None:
    # Arrange
    first = _season_predictions(2020, WEEK_GAMES)
    second = first.with_columns(
        pl.when(pl.col("game_id") == first.get_column("game_id")[0])
        .then(pl.lit("ZZZ"))
        .otherwise(pl.col("home_abbr"))
        .alias("home_abbr")
    )

    # Act & Assert
    with pytest.raises(ValueError, match="disagree"):
        rating_sources.average_predictions([first, second])


def test_read_split_predictions_takes_earlier_seasons_from_the_averaged_reference(
    tmp_path: Path,
) -> None:
    # Arrange
    current = write_fold_checkpoints(
        tmp_path / "stage1",
        [
            fold_frame(2023, 1, [("BBB", "AAA", 100.0)]),
            fold_frame(2024, 1, [("DDD", "CCC", 3.0)]),
        ],
    )
    seed_42 = write_fold_checkpoints(
        tmp_path / "seed42", [fold_frame(2023, 1, [("BBB", "AAA", 4.0)])]
    )
    seed_7 = write_fold_checkpoints(
        tmp_path / "seed7", [fold_frame(2023, 1, [("BBB", "AAA", 6.0)])]
    )

    # Act
    predictions = rating_sources.read_split_predictions([current], [seed_42, seed_7], season=2024)

    # Assert
    assert predictions.columns == list(rating_sources.PREDICTION_COLUMNS)
    assert predictions.get_column("game_id").to_list() == ["2023_01_BBB_AAA", "2024_01_DDD_CCC"]
    assert predictions.get_column("predicted_margin").to_list() == [5.0, 3.0]
