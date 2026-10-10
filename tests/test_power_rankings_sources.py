"""Tests for ranking on a rating source in points: the composite, ported, model and blend."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import norm
from tests.rating_source_fixtures import fold_frame, write_fold_checkpoints

from nfl_predictor import constants
from nfl_predictor.reporting import power_rankings, rating_sources

if TYPE_CHECKING:
    from pathlib import Path

    from nfl_predictor.ml.ml_model_core import MarginTotalModel


def _snapshot(composites: dict[str, float], *, points_per_unit: float) -> pd.DataFrame:
    """Return one week's strength snapshot whose SRS is an exact multiple of the composite."""
    teams = sorted(composites)
    return pd.DataFrame(
        {
            "team_abbr": teams,
            "adj_strength_composite": [composites[team] for team in teams],
            "adj_srs": [composites[team] * points_per_unit for team in teams],
        }
    )


def test_composite_points_vs_average_keeps_every_digit() -> None:
    # Arrange
    snapshot = _snapshot({"AAA": -1.0, "BBB": 0.5, "CCC": 2.0}, points_per_unit=6.123457)

    # Act
    points = power_rankings.composite_points_vs_average(snapshot)

    # Assert
    assert points == pytest.approx([-9.1851855, 0.0, 9.1851855], rel=1e-9, abs=1e-12)


def test_composite_points_vs_average_without_a_scale_is_all_nan() -> None:
    # Arrange
    snapshot = _snapshot({"AAA": -1.0, "BBB": 0.5, "CCC": 2.0}, points_per_unit=-2.0)

    # Act
    points = power_rankings.composite_points_vs_average(snapshot)

    # Assert
    assert np.isnan(points).all()


TEAMS = ["AAA", "BBB", "CCC", "DDD"]


def _records(teams: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_abbr": teams,
            "wins": [1] * len(teams),
            "losses": [1] * len(teams),
            "ties": [0] * len(teams),
            "games_played": [2] * len(teams),
        }
    )


def _rank_on(ratings: dict[str, float], teams: list[str]) -> pd.DataFrame:
    standings = power_rankings.LeagueStandings(2024, 2, _records(teams), pd.DataFrame())
    team_ratings = power_rankings.TeamRatings(ratings=ratings, source="ported", rating_week=3)
    return power_rankings.build_power_rankings_and_standings(
        standings, team_ratings=team_ratings
    ).power_rankings


def test_points_ranking_orders_teams_and_maps_points_through_the_margin_curve() -> None:
    # Arrange
    ratings = {"AAA": 4.0, "BBB": -2.0, "CCC": 1.0, "DDD": -3.0}

    # Act
    rankings = _rank_on(ratings, TEAMS)

    # Assert
    assert rankings["team_abbr"].tolist() == ["AAA", "CCC", "BBB", "DDD"]
    assert rankings["rank"].tolist() == [1, 2, 3, 4]
    top = rankings.iloc[0]
    assert top["points_vs_average"] == pytest.approx(4.0)
    expected = float(norm.cdf(4.0 / constants.SCORE_DIFF_STD_DEV))
    assert top["power_rating_1_10"] == pytest.approx(round(1.0 + 9.0 * expected, 2))
    assert top["power_rating_0_10"] == pytest.approx(round(10.0 * expected, 2))
    assert set(rankings["rating_source"]) == {"ported"}
    assert set(rankings["rating_week"]) == {3}


def test_points_ranking_centers_points_on_the_rated_teams() -> None:
    # Arrange
    ratings = {"AAA": 5.0, "BBB": 3.0, "CCC": 1.0, "DDD": -1.0}

    # Act
    rankings = _rank_on(ratings, TEAMS).set_index("team_abbr")

    # Assert
    assert rankings.loc["AAA", "points_vs_average"] == pytest.approx(3.0)
    assert rankings.loc["AAA", "team_rating"] == pytest.approx(5.0)


def test_points_ranking_keeps_a_team_without_a_rating_last() -> None:
    # Arrange
    ratings = {"AAA": 4.0, "BBB": -2.0, "CCC": 1.0}

    # Act
    rankings = _rank_on(ratings, TEAMS)

    # Assert
    assert rankings["team_abbr"].tolist()[-1] == "DDD"
    assert np.isnan(rankings.iloc[-1]["power_rating_1_10"])


def test_ranking_request_accepts_the_ported_method() -> None:
    # Arrange
    request = power_rankings.RankingRequest(method="ported", legacy_franchise_fit=False)

    # Act
    options = power_rankings.resolve_ranking_options(request)

    # Assert
    assert options.method == "ported"


def test_legacy_franchise_fit_cannot_be_combined_with_the_ported_method() -> None:
    # Arrange
    request = power_rankings.RankingRequest(method="ported", legacy_franchise_fit=True)

    # Act & Assert
    with pytest.raises(ValueError, match="--method ported"):
        power_rankings.resolve_ranking_options(request)


def test_model_method_without_predictions_is_rejected() -> None:
    # Arrange
    request = power_rankings.RankingRequest(method="model", legacy_franchise_fit=False)

    # Act & Assert
    with pytest.raises(ValueError, match="predictions"):
        power_rankings.resolve_ranking_options(request)


@pytest.mark.parametrize("weight", [None, 1.5])
def test_blend_method_needs_a_weight_between_zero_and_one(
    weight: float | None, tmp_path: Path
) -> None:
    # Arrange
    request = power_rankings.RankingRequest(
        method="blend",
        legacy_franchise_fit=False,
        model_predictions=(tmp_path,),
        blend_weight=weight,
    )

    # Act & Assert
    with pytest.raises(ValueError, match="blend weight"):
        power_rankings.resolve_ranking_options(request)


# (season, week, away, home, away score, home score, away rating, home rating, predicted margin)
LEAGUE = (
    (2023, 1, "AAA", "BBB", 10, 27, -1.0, 1.0, 2.0),
    (2023, 1, "CCC", "DDD", 24, 17, 0.5, -0.5, -1.0),
    (2023, 2, "BBB", "CCC", 20, 23, 1.0, 0.5, 1.5),
    (2023, 2, "DDD", "AAA", 13, 30, -0.5, -1.0, 4.0),
    (2024, 1, "AAA", "CCC", 21, 20, 2.4, -1.3, -3.5),
    (2024, 1, "BBB", "DDD", 35, 14, -1.2, 0.1, 1.0),
    (2024, 2, "CCC", "BBB", 17, 31, -1.0, 1.5, 6.0),
    (2024, 2, "DDD", "AAA", 28, 24, 0.2, 0.8, 3.0),
    (2024, 3, "AAA", "BBB", None, None, 2.5, -2.0, None),
    (2024, 3, "CCC", "DDD", None, None, 0.5, -1.5, None),
)


def _write_league(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write the league's game rows, published ratings and walk-forward folds."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    games = pl.DataFrame(
        [
            {
                "season": row[0],
                "week": row[1],
                "game_type": "REG",
                "away_abbr": row[2],
                "home_abbr": row[3],
                "away_score": row[4],
                "home_score": row[5],
                "neutral": 0,
                "away_ratings_team": row[6],
                "home_ratings_team": row[7],
            }
            for row in LEAGUE
        ]
    )
    games_path = tmp_path / "all_data_ml.csv"
    games.write_csv(games_path)
    ratings_dir = tmp_path / "ratings"
    ratings_dir.mkdir()
    pl.DataFrame(
        {
            "week": [1, 1, 1, 1, 2, 2, 2, 2],
            "team": TEAMS * 2,
            "games_played": [1] * 4 + [2] * 4,
            "team_rating": [1.0, -1.0, 2.0, -2.0, 2.5, -2.0, 0.5, -1.5],
        }
    ).write_parquet(ratings_dir / "2024_ratings_by_week.parquet")
    pl.DataFrame({"team": TEAMS, "team_rating": [-2.0, 2.0, 1.0, -1.0]}).write_parquet(
        ratings_dir / "2023_ratings.parquet"
    )
    weeks = ((2023, 1), (2023, 2), (2024, 1), (2024, 2))
    folds = write_fold_checkpoints(tmp_path / "folds", [_league_fold(*key) for key in weeks])
    return games_path, ratings_dir, folds


def _league_fold(season: int, week: int) -> pd.DataFrame:
    """Return the league's predictions for one week as a fold frame."""
    games = [
        (row[2], row[3], float(row[8]))
        for row in LEAGUE
        if row[:2] == (season, week) and row[8] is not None
    ]
    return fold_frame(season, week, games)


def _no_future_games(*_args: object, **_kwargs: object) -> pd.DataFrame:
    return pd.DataFrame(columns=["season", "week", "away_abbr", "home_abbr", "home_win_prob"])


def _rank_league(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    *,
    through_week: int = 2,
    blend_weight: float | None = None,
) -> pd.DataFrame:
    monkeypatch.setattr(power_rankings, "_predict_future_games", _no_future_games)
    games_path, ratings_dir, folds = _write_league(tmp_path)
    options = power_rankings.resolve_ranking_options(
        power_rankings.RankingRequest(
            method=method,
            legacy_franchise_fit=False,
            ratings_dir=ratings_dir,
            model_predictions=(folds,),
            blend_weight=blend_weight,
        )
    )
    inputs = power_rankings.RankingInputs(
        data_ml=games_path, data_schedule=games_path, season=2024, through_week=through_week
    )
    model = cast("MarginTotalModel", None)
    return power_rankings.compute_power_rankings(model, inputs, options).power_rankings


def _ratings_of(rankings: pd.DataFrame) -> dict[str, float]:
    return dict(zip(rankings["team_abbr"], rankings["team_rating"], strict=True))


def test_ported_method_ranks_on_the_ratings_before_the_next_week(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Act
    rankings = _rank_league(tmp_path, monkeypatch, "ported")

    # Assert
    assert _ratings_of(rankings) == {"AAA": 2.5, "BBB": -2.0, "CCC": 0.5, "DDD": -1.5}
    assert rankings["team_abbr"].tolist() == ["AAA", "CCC", "DDD", "BBB"]


def test_model_method_ranks_on_the_fit_to_the_season_so_far(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    folds = write_fold_checkpoints(
        tmp_path / "expected", [_league_fold(2024, 1), _league_fold(2024, 2)]
    )
    predictions = rating_sources.read_predictions([folds])
    expected = rating_sources.model_ratings_for_week(predictions, season=2024, week=3, teams=TEAMS)

    # Act
    rankings = _rank_league(tmp_path / "run", monkeypatch, "model")

    # Assert
    assert _ratings_of(rankings) == pytest.approx(expected)


def test_model_method_in_week_one_carries_last_season_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    week_one = {"AAA": 2.4, "BBB": -1.2, "CCC": -1.3, "DDD": 0.1}
    factor = rating_sources.carryover_factor(
        week_one, {"AAA": -2.0, "BBB": 2.0, "CCC": 1.0, "DDD": -1.0}
    )
    folds = write_fold_checkpoints(
        tmp_path / "expected", [_league_fold(2023, 1), _league_fold(2023, 2)]
    )
    predictions = rating_sources.read_predictions([folds])
    expected = rating_sources.model_ratings_for_week(
        predictions, season=2024, week=1, teams=TEAMS, carryover=factor
    )

    # Act
    rankings = _rank_league(tmp_path / "run", monkeypatch, "model", through_week=0)

    # Assert
    assert _ratings_of(rankings) == pytest.approx(expected)


def test_blend_method_weights_the_ported_and_model_ratings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    ported = _ratings_of(_rank_league(tmp_path / "ported", monkeypatch, "ported"))
    model = _ratings_of(_rank_league(tmp_path / "model", monkeypatch, "model"))
    expected = {team: 0.3 * ported[team] + 0.7 * model[team] for team in TEAMS}

    # Act
    rankings = _rank_league(tmp_path / "blend", monkeypatch, "blend", blend_weight=0.3)

    # Assert
    assert _ratings_of(rankings) == pytest.approx(expected)
    assert set(rankings["blend_weight"]) == {0.3}


def test_model_method_refuses_predictions_missing_a_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(power_rankings, "_predict_future_games", _no_future_games)
    games_path, ratings_dir, _ = _write_league(tmp_path)
    partial = write_fold_checkpoints(tmp_path / "partial", [_league_fold(2024, 1)])
    options = power_rankings.resolve_ranking_options(
        power_rankings.RankingRequest(
            method="model",
            legacy_franchise_fit=False,
            ratings_dir=ratings_dir,
            model_predictions=(partial,),
        )
    )
    inputs = power_rankings.RankingInputs(
        data_ml=games_path, data_schedule=games_path, season=2024, through_week=2
    )

    # Act & Assert
    with pytest.raises(rating_sources.RatingSourceUnavailableError, match="2 of 4"):
        power_rankings.compute_power_rankings(cast("MarginTotalModel", None), inputs, options)


def test_model_method_in_week_one_without_a_carryover_factor_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(power_rankings, "_predict_future_games", _no_future_games)
    games_path, ratings_dir, folds = _write_league(tmp_path)
    pl.DataFrame({"team": ["XXX", "YYY"], "team_rating": [1.0, -1.0]}).write_parquet(
        ratings_dir / "2023_ratings.parquet"
    )
    options = power_rankings.resolve_ranking_options(
        power_rankings.RankingRequest(
            method="model",
            legacy_franchise_fit=False,
            ratings_dir=ratings_dir,
            model_predictions=(folds,),
        )
    )
    inputs = power_rankings.RankingInputs(
        data_ml=games_path, data_schedule=games_path, season=2024, through_week=0
    )

    # Act & Assert
    with pytest.raises(rating_sources.RatingSourceUnavailableError, match="carryover"):
        power_rankings.compute_power_rankings(cast("MarginTotalModel", None), inputs, options)
