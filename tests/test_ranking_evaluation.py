"""Tests for scoring the candidate power rankings on next-week margins."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_predictor.reporting import ranking_evaluation, rating_sources, run_comparison


def _schedule_rows() -> pl.DataFrame:
    """Return game rows: scored games plus a playoff game, a week-1 game and an unplayed one."""
    return pl.DataFrame(
        {
            "game_id": ["a", "b", "c", "d", "e", "f"],
            "season": [2008, 2008, 2008, 2008, 2009, 2007],
            "week": [2, 1, 3, 20, 4, 5],
            "game_type": ["REG", "REG", "REG", "WC", "REG", "REG"],
            "away_abbr": ["BBB", "BBB", "CCC", "AAA", "DDD", "AAA"],
            "home_abbr": ["AAA", "AAA", "DDD", "BBB", "AAA", "CCC"],
            "away_score": [10, 3, None, 20, 24, 13],
            "home_score": [17, 7, None, 27, 21, 13],
            "neutral": [0, 0, 0, 0, None, 1],
        }
    )


def test_scored_games_keeps_played_regular_season_games_from_week_two() -> None:
    # Arrange
    rows = _schedule_rows()

    # Act
    games = ranking_evaluation.scored_games(rows, seasons=range(2007, 2010))

    # Assert
    assert games.get_column("game_id").to_list() == ["f", "a", "e"]
    assert games.get_column("actual_margin").to_list() == [0.0, 7.0, -3.0]
    assert games.get_column("neutral").to_list() == [1, 0, 0]


def _ratings(values: dict[tuple[int, int, str], float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [key[0] for key in values],
            "week": [key[1] for key in values],
            "team": [key[2] for key in values],
            "rating": list(values.values()),
        },
        schema=rating_sources.RATING_SCHEMA,
    )


def test_rating_differences_is_home_minus_away_and_null_without_both() -> None:
    # Arrange
    games = ranking_evaluation.scored_games(_schedule_rows(), seasons=range(2007, 2010))
    ratings = _ratings(
        {
            (2008, 2, "AAA"): 3.0,
            (2008, 2, "BBB"): -1.5,
            (2009, 4, "AAA"): 1.0,
            (2007, 5, "CCC"): 2.0,
        }
    )

    # Act
    scored = ranking_evaluation.rating_differences(games, ratings)

    # Assert
    differences = dict(zip(scored["game_id"], scored["difference"], strict=True))
    assert differences == {"f": None, "a": 4.5, "e": None}


def _scored(rows: list[tuple[int, int, float, float]]) -> pl.DataFrame:
    """Return scored games from (season, neutral, actual margin, rating difference) tuples."""
    return pl.DataFrame(
        {
            "game_id": [f"g{index}" for index in range(len(rows))],
            "season": [row[0] for row in rows],
            "week": [5] * len(rows),
            "neutral": [row[1] for row in rows],
            "actual_margin": [row[2] for row in rows],
            "difference": [row[3] for row in rows],
        }
    )


def test_fit_home_term_is_the_mean_residual_of_non_neutral_games() -> None:
    # Arrange
    games = _scored([(2007, 0, 10.0, 4.0), (2007, 0, -3.0, -5.0), (2007, 1, 30.0, 0.0)])

    # Act
    home_term = ranking_evaluation.fit_home_term(games)

    # Assert
    assert home_term == pytest.approx(4.0)


def test_home_terms_by_season_reads_only_earlier_seasons() -> None:
    # Arrange
    games = _scored(
        [(2007, 0, 3.0, 0.0), (2008, 0, 7.0, 0.0), (2009, 0, 100.0, 0.0), (2009, 0, 1.0, 0.0)]
    )

    # Act
    home_terms = ranking_evaluation.home_terms_by_season(games, [2008, 2009])

    # Assert
    assert home_terms == pytest.approx({2008: 3.0, 2009: 5.0})


def test_absolute_errors_apply_the_home_term_off_neutral_sites_only() -> None:
    # Arrange
    games = _scored([(2008, 0, 10.0, 4.0), (2008, 1, 10.0, 4.0)])

    # Act
    errors = ranking_evaluation.absolute_errors(games, {2008: 2.5})

    # Assert
    assert errors.get_column("error").to_list() == pytest.approx([3.5, 6.0])


def test_absolute_errors_without_a_season_home_term_raises() -> None:
    # Arrange
    games = _scored([(2008, 0, 10.0, 4.0)])

    # Act & Assert
    with pytest.raises(ValueError, match="2008"):
        ranking_evaluation.absolute_errors(games, {2009: 2.5})


def _blend_games(rows: list[tuple[int, float, float, float]]) -> pl.DataFrame:
    """Return scored games from (season, actual margin, ported, model difference) tuples."""
    return pl.DataFrame(
        {
            "game_id": [f"g{index}" for index in range(len(rows))],
            "season": [row[0] for row in rows],
            "week": [5] * len(rows),
            "neutral": [0] * len(rows),
            "actual_margin": [row[1] for row in rows],
            "ported_difference": [row[2] for row in rows],
            "model_difference": [row[3] for row in rows],
        }
    )


EARLIER_GAMES = [
    (2007, 12.0, 0.0, 10.0),
    (2007, -8.0, 0.0, -10.0),
    (2007, 2.0, 5.0, 0.0),
    (2007, 2.0, -5.0, 0.0),
]


def test_choose_blend_weights_picks_the_weight_with_the_lowest_earlier_error() -> None:
    # Arrange
    games = _blend_games(EARLIER_GAMES)

    # Act
    choices = ranking_evaluation.choose_blend_weights(games, [2008])

    # Assert
    assert choices[2008].weight == pytest.approx(0.0)
    assert choices[2008].home_term == pytest.approx(2.0)
    assert choices[2008].mae == pytest.approx(0.0)


def test_choose_blend_weights_ignores_the_season_being_chosen_for() -> None:
    # Arrange
    swapped = [(2008, margin, model, ported) for _, margin, ported, model in EARLIER_GAMES]
    games = _blend_games([*EARLIER_GAMES, *swapped])

    # Act
    choices = ranking_evaluation.choose_blend_weights(games, [2008])

    # Assert
    assert choices[2008].weight == pytest.approx(0.0)


def test_choose_blend_weights_breaks_a_tie_toward_the_ported_rating() -> None:
    # Arrange
    games = _blend_games([(2007, 3.0, 1.0, 1.0), (2007, -1.0, -2.0, -2.0)])

    # Act
    choices = ranking_evaluation.choose_blend_weights(games, [2008])

    # Assert
    assert choices[2008].weight == pytest.approx(1.0)


def test_blend_differences_apply_each_season_its_own_weight() -> None:
    # Arrange
    games = _blend_games([(2008, 0.0, 4.0, 0.0), (2009, 0.0, 4.0, 0.0)])
    choices = {
        2008: ranking_evaluation.BlendChoice(weight=0.25, home_term=1.0, mae=0.0, games=1),
        2009: ranking_evaluation.BlendChoice(weight=0.5, home_term=1.0, mae=0.0, games=1),
    }

    # Act
    blended = ranking_evaluation.blend_differences(games, choices)

    # Assert
    assert blended.get_column("difference").to_list() == pytest.approx([1.0, 2.0])


def test_season_bootstrap_with_equal_season_means_has_no_spread() -> None:
    # Arrange
    differences = np.array([1.0, -1.0, 3.0, -3.0, 0.5, -0.5])
    seasons = np.array([2008, 2008, 2009, 2009, 2010, 2010])

    # Act
    mean, low, high = ranking_evaluation.season_bootstrap(differences, seasons, 2000, 42)

    # Assert
    assert (mean, low, high) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


def test_season_bootstrap_weights_each_season_by_its_games() -> None:
    # Arrange
    differences = np.array([1.0, 1.0, 1.0, -2.0])
    seasons = np.array([2008, 2008, 2008, 2009])

    # Act
    mean, low, high = ranking_evaluation.season_bootstrap(differences, seasons, 2000, 42)

    # Assert
    assert mean == pytest.approx(0.25)
    assert low == pytest.approx(-2.0)
    assert high == pytest.approx(1.0)


def _paired_games() -> pl.DataFrame:
    rng = np.random.default_rng(3)
    size = 60
    return pl.DataFrame(
        {
            "season": np.repeat([2008, 2009, 2010], size // 3),
            "week": np.tile(np.arange(2, 22), 3),
            "candidate_error": rng.uniform(0.0, 20.0, size),
            "reference_error": rng.uniform(0.0, 20.0, size),
        }
    )


def test_paired_comparison_scores_only_the_window() -> None:
    # Arrange
    games = _paired_games()
    window = games.filter((pl.col("week") >= 5) & (pl.col("week") <= 8))
    differences = (window["candidate_error"] - window["reference_error"]).to_numpy()

    # Act
    comparison = ranking_evaluation.paired_comparison(
        games, first_week=5, last_week=8, resamples=500, seed=42
    )

    # Assert
    assert comparison.games == window.height
    assert comparison.candidate_mae == pytest.approx(window["candidate_error"].mean())
    assert comparison.reference_mae == pytest.approx(window["reference_error"].mean())
    assert comparison.game_interval == pytest.approx(
        run_comparison.bootstrap_mean(differences, 500, 42)
    )


def test_paired_comparison_season_interval_resamples_seasons() -> None:
    # Arrange
    games = _paired_games()
    differences = (games["candidate_error"] - games["reference_error"]).to_numpy()
    seasons = games.get_column("season").to_numpy()

    # Act
    comparison = ranking_evaluation.paired_comparison(
        games, first_week=2, last_week=None, resamples=500, seed=42
    )

    # Assert
    assert comparison.season_interval == pytest.approx(
        ranking_evaluation.season_bootstrap(differences, seasons, 500, 42)
    )


def _week_ratings(values: dict[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2010] * len(values),
            "week": [6] * len(values),
            "team": list(values),
            "rating": list(values.values()),
        },
        schema=rating_sources.RATING_SCHEMA,
    )


SIX_TEAMS = {"AAA": 6.0, "BBB": 5.0, "CCC": 4.0, "DDD": 3.0, "EEE": 2.0, "FFF": 1.0}


def test_rank_agreement_identical_order_correlates_fully() -> None:
    # Arrange
    reference = _week_ratings(SIX_TEAMS)
    candidate = _week_ratings({team: 10.0 * value for team, value in SIX_TEAMS.items()})

    # Act
    agreement = ranking_evaluation.rank_agreement(candidate, reference, [(2010, 6)])

    # Assert
    assert agreement.get_column("spearman").to_list() == pytest.approx([1.0])
    assert agreement.get_column("top_differs").to_list() == [False]
    assert agreement.get_column("teams").to_list() == [6]


def test_rank_agreement_reports_a_changed_top_five() -> None:
    # Arrange
    reference = _week_ratings(SIX_TEAMS)
    candidate = _week_ratings({**SIX_TEAMS, "FFF": 9.0, "EEE": 0.0})

    # Act
    agreement = ranking_evaluation.rank_agreement(candidate, reference, [(2010, 6)])

    # Assert
    assert agreement.get_column("top_differs").to_list() == [True]
    assert agreement.get_column("spearman").item() < 1.0


def test_composite_ratings_by_week_expresses_each_week_in_its_own_points() -> None:
    # Arrange
    snapshots = pl.DataFrame(
        {
            "season": [2010] * 6,
            "week": [2, 2, 2, 3, 3, 3],
            "team_abbr": ["AAA", "BBB", "CCC"] * 2,
            "adj_strength_composite": [1.0, 0.0, -1.0, 2.0, 0.0, -1.0],
            "adj_srs": [7.0, 0.0, -7.0, 6.0, 0.0, -3.0],
        }
    )

    # Act
    ratings = ranking_evaluation.composite_ratings_by_week(snapshots)

    # Assert
    week_two = ratings.filter(pl.col("week") == 2)
    week_three = ratings.filter(pl.col("week") == 3)
    assert week_two.get_column("rating").to_list() == pytest.approx([7.0, 0.0, -7.0])
    assert week_three.get_column("team").to_list() == ["AAA", "BBB", "CCC"]
    assert week_three.get_column("rating").to_list() == pytest.approx(
        [3.0 * (2.0 - 1.0 / 3.0), 3.0 * (0.0 - 1.0 / 3.0), 3.0 * (-1.0 - 1.0 / 3.0)]
    )


def _verdict(name: str, mae: float, game_high: float, season_high: float):
    return ranking_evaluation.CandidateVerdict(
        name=name, mae=mae, reference_mae=10.0, game_high=game_high, season_high=season_high
    )


def test_decide_keeps_the_reference_when_no_interval_excludes_zero() -> None:
    # Arrange
    verdicts = [_verdict("composite", 9.9, 0.01, -0.02), _verdict("model", 10.2, 0.5, 0.4)]

    # Act
    decision = ranking_evaluation.decide(verdicts, reference="ported")

    # Assert
    assert decision.winner == "ported"
    assert decision.qualified == ()


def test_decide_takes_the_lowest_error_among_candidates_that_beat_the_reference() -> None:
    # Arrange
    verdicts = [
        _verdict("composite", 9.8, -0.01, -0.02),
        _verdict("model", 9.5, -0.2, -0.1),
        _verdict("blend", 9.4, -0.3, 0.05),
    ]

    # Act
    decision = ranking_evaluation.decide(verdicts, reference="ported")

    # Assert
    assert decision.winner == "model"
    assert decision.qualified == ("composite", "model")


def test_fit_home_term_with_only_neutral_sites_raises() -> None:
    # Arrange
    games = _scored([(2007, 1, 10.0, 4.0)])

    # Act & Assert
    with pytest.raises(ValueError, match="non-neutral"):
        ranking_evaluation.fit_home_term(games)


def test_choose_blend_weights_without_earlier_games_raises() -> None:
    # Arrange
    games = _blend_games([(2008, 3.0, 1.0, 2.0)])

    # Act & Assert
    with pytest.raises(ValueError, match="before 2008"):
        ranking_evaluation.choose_blend_weights(games, [2008])


def test_blend_differences_without_a_season_weight_raises() -> None:
    # Arrange
    games = _blend_games([(2009, 0.0, 4.0, 0.0)])

    # Act & Assert
    with pytest.raises(ValueError, match="2009"):
        ranking_evaluation.blend_differences(games, {})


def test_paired_comparison_of_an_empty_window_raises() -> None:
    # Arrange
    games = _paired_games()

    # Act & Assert
    with pytest.raises(ValueError, match="no scored game"):
        ranking_evaluation.paired_comparison(
            games, first_week=30, last_week=31, resamples=10, seed=42
        )


def test_composite_ratings_by_week_drops_a_week_without_a_points_scale() -> None:
    # Arrange
    snapshots = pl.DataFrame(
        {
            "season": [2010] * 3,
            "week": [2] * 3,
            "team_abbr": ["AAA", "BBB", "CCC"],
            "adj_strength_composite": [1.0, 0.0, -1.0],
            "adj_srs": [-7.0, 0.0, 7.0],
        }
    )

    # Act
    ratings = ranking_evaluation.composite_ratings_by_week(snapshots)

    # Assert
    assert ratings.is_empty()
