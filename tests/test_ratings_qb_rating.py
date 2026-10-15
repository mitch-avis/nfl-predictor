"""Tests for the adjusted-EPA-per-dropback quarterback rating and its faced pass defense."""

import itertools

import numpy as np
import polars as pl
import pytest

from nfl_predictor.ratings.qb_rating import (
    QB_EPA_PER_DROPBACK_COLUMN,
    QbRatingResampler,
    bootstrap_qb_ratings,
    compute_qb_faced_pass_defense,
    fit_qb_ratings,
    fit_qb_ratings_by_week,
    qb_rating_rows,
    qb_season_summary,
)

# Each team has one passer, named after it. The effects average to zero across teams.
_PASSER = {"AAA": 0.15, "BBB": 0.05, "CCC": -0.05, "DDD": -0.15}
_DEFENSE = {"AAA": 0.04, "BBB": -0.02, "CCC": 0.06, "DDD": -0.08}
_LEAGUE = 0.03
_TINY_LAMBDA = 1e-6


def _qb_games(dropbacks: dict[str, int] | None = None) -> pl.DataFrame:
    """Return noiseless QB-game rows for a double round-robin built from known effects."""
    volume = dropbacks or {}
    rows: list[dict[str, object]] = []
    for game_number, (home, away) in enumerate(itertools.permutations(_PASSER, 2)):
        for team, opponent in ((home, away), (away, home)):
            plays = volume.get(opponent, 35)
            rows.append(
                {
                    "game_id": f"g{game_number:02d}",
                    "week": game_number + 1,
                    "qb_id": f"qb_{team}",
                    "qb_name": f"Q.{team}",
                    "team": team,
                    "opponent_team": opponent,
                    "qb_dropbacks": plays,
                    "qb_attempts": plays - 2,
                    "qb_dropback_epa": (_LEAGUE + _PASSER[team] - _DEFENSE[opponent]) * plays,
                }
            )
    return pl.DataFrame(rows)


def _value(frame: pl.DataFrame, qb_id: str, column: str) -> float:
    """Return one QB's value for one column."""
    return float(frame.filter(pl.col("qb_id") == qb_id).get_column(column).item())


def test_fit_qb_ratings_adjusted_epa_is_league_average_plus_passer_effect() -> None:
    # Arrange
    qb_games = _qb_games()

    # Act
    ratings = fit_qb_ratings(qb_games, ridge_lambda=_TINY_LAMBDA).ratings

    # Assert
    assert _value(ratings, "qb_AAA", "adj_qb_epa_per_dropback") == pytest.approx(
        _LEAGUE + 0.15, abs=1e-6
    )


def test_compute_qb_faced_pass_defense_weights_defenses_by_dropbacks() -> None:
    # Arrange
    qb_games = _qb_games(dropbacks={"CCC": 70})
    fit = fit_qb_ratings(qb_games, ridge_lambda=_TINY_LAMBDA)
    # qb_AAA faces BBB, CCC, and DDD twice each; CCC games carry 70 dropbacks, the others 35.
    expected = (35 * _DEFENSE["BBB"] + 70 * _DEFENSE["CCC"] + 35 * _DEFENSE["DDD"]) / 140

    # Act
    faced = compute_qb_faced_pass_defense(qb_games, fit)

    # Assert
    assert _value(faced, "qb_AAA", "qb_faced_pass_defense") == pytest.approx(expected, abs=1e-6)


def test_compute_qb_faced_pass_defense_ignores_the_passers_own_games() -> None:
    # Arrange
    qb_games = _qb_games()
    shredded = qb_games.with_columns(
        pl.when(pl.col("qb_id") == "qb_AAA")
        .then(pl.col("qb_dropback_epa") + pl.col("qb_dropbacks"))
        .otherwise(pl.col("qb_dropback_epa"))
        .alias("qb_dropback_epa")
    )
    baseline = compute_qb_faced_pass_defense(qb_games, fit_qb_ratings(qb_games, ridge_lambda=10.0))

    # Act
    result = compute_qb_faced_pass_defense(shredded, fit_qb_ratings(shredded, ridge_lambda=10.0))

    # Assert
    assert _value(result, "qb_AAA", "qb_faced_pass_defense") == pytest.approx(
        _value(baseline, "qb_AAA", "qb_faced_pass_defense")
    )


def test_fit_qb_ratings_missing_dropbacks_raises_value_error() -> None:
    # Arrange
    qb_games = _qb_games().drop("qb_dropbacks")

    # Act & Assert
    with pytest.raises(ValueError, match="qb_dropbacks"):
        fit_qb_ratings(qb_games)


def test_fit_qb_ratings_ignores_game_outcomes() -> None:
    # Arrange
    qb_games = _qb_games().with_columns(pl.lit(0).alias("qb_wins"))
    flipped = qb_games.with_columns(pl.lit(1).alias("qb_wins"))
    baseline = fit_qb_ratings(qb_games).ratings

    # Act
    result = fit_qb_ratings(flipped).ratings

    # Assert
    assert result.equals(baseline)


def test_compute_qb_faced_pass_defense_skips_defenses_without_other_passers() -> None:
    # Arrange
    qb_games = pl.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2"],
            "qb_id": ["qb_AAA", "qb_BBB", "qb_CCC", "qb_DDD"],
            "opponent_team": ["BBB", "AAA", "DDD", "CCC"],
            "qb_dropbacks": [35, 35, 35, 35],
            "qb_dropback_epa": [3.5, 0.0, 1.75, -1.75],
        }
    )
    fit = fit_qb_ratings(qb_games, ridge_lambda=10.0)

    # Act
    faced = compute_qb_faced_pass_defense(qb_games, fit)

    # Assert
    assert faced.get_column("qb_faced_pass_defense").null_count() == faced.height


def test_fit_qb_ratings_by_week_refits_each_week_with_the_season_penalty() -> None:
    # Arrange
    qb_games = _qb_games()
    season_fit = fit_qb_ratings(qb_games, ridge_lambda=10.0)
    through_week_seven = fit_qb_ratings(
        qb_games.filter(pl.col("week") <= 7), ridge_lambda=10.0
    ).ratings

    # Act
    history = fit_qb_ratings_by_week(qb_games, season_fit)

    # Assert
    week_seven = history.filter(pl.col("week") == 7).select(through_week_seven.columns)
    assert week_seven.sort("qb_id").equals(through_week_seven.sort("qb_id"))


def test_fit_qb_ratings_by_week_counts_games_and_dropbacks_through_each_week() -> None:
    # Arrange
    qb_games = _qb_games(dropbacks={"CCC": 70})

    # Act
    history = fit_qb_ratings_by_week(qb_games, fit_qb_ratings(qb_games))

    # Assert
    # qb_AAA hosts BBB, CCC, and DDD in weeks 1-3.
    week_three = history.filter((pl.col("week") == 3) & (pl.col("qb_id") == "qb_AAA"))
    assert week_three.select("qb_games_played", "qb_dropbacks").row(0) == (3, 35 + 70 + 35)


def test_fit_qb_ratings_by_week_publishes_columns_in_order() -> None:
    # Arrange
    qb_games = _qb_games()

    # Act
    history = fit_qb_ratings_by_week(qb_games, fit_qb_ratings(qb_games))

    # Assert
    assert history.columns == [
        "week",
        "qb_id",
        "qb_games_played",
        "qb_dropbacks",
        "adj_qb_epa_per_dropback",
    ]


def test_fit_qb_ratings_by_week_without_a_week_column_raises_value_error() -> None:
    # Arrange
    qb_games = _qb_games()
    season_fit = fit_qb_ratings(qb_games)

    # Act & Assert
    with pytest.raises(ValueError, match="week"):
        fit_qb_ratings_by_week(qb_games.drop("week"), season_fit)


def test_qb_rating_rows_keep_only_rows_with_a_dropback() -> None:
    # Arrange
    qb_games = _qb_games().with_columns(
        pl.when(pl.col("game_id") == "g00")
        .then(0)
        .otherwise(pl.col("qb_dropbacks"))
        .alias("qb_dropbacks")
    )

    # Act
    rows = qb_rating_rows(qb_games)

    # Assert
    assert rows.height == qb_games.height - 2


def test_qb_rating_rows_divide_the_dropback_epa_by_the_dropbacks() -> None:
    # Arrange
    qb_games = _qb_games(dropbacks={"CCC": 70})
    expected = (
        qb_games.get_column("qb_dropback_epa") / qb_games.get_column("qb_dropbacks")
    ).to_list()

    # Act
    rows = qb_rating_rows(qb_games)

    # Assert
    assert rows.get_column(QB_EPA_PER_DROPBACK_COLUMN).to_list() == expected


def test_qb_rating_resampler_matches_a_fit_on_duplicated_games() -> None:
    # Arrange
    qb_games = _qb_games(dropbacks={"CCC": 70})
    fit = fit_qb_ratings(qb_games, ridge_lambda=10.0)
    resampler = QbRatingResampler(qb_games, fit)
    counts = np.array([(index * 5) % 3 for index in range(len(resampler.game_ids))])
    duplicated = pl.concat(
        [
            qb_games.filter(pl.col("game_id") == game_id).with_columns(
                pl.lit(f"{game_id}#{copy}").alias("game_id")
            )
            for game_id, count in zip(resampler.game_ids, counts, strict=True)
            for copy in range(int(count))
        ]
    )
    expected = fit_qb_ratings(duplicated, ridge_lambda=10.0).ratings.sort("qb_id")

    # Act
    ratings = resampler.ratings(counts).sort("qb_id")

    # Assert
    assert ratings.get_column("qb_id").to_list() == expected.get_column("qb_id").to_list()
    assert ratings.get_column("adj_qb_epa_per_dropback").to_list() == pytest.approx(
        expected.get_column("adj_qb_epa_per_dropback").to_list()
    )


def test_bootstrap_qb_ratings_is_reproducible_for_a_seed() -> None:
    # Arrange
    qb_games = _qb_games()
    fit = fit_qb_ratings(qb_games, ridge_lambda=10.0)
    first = bootstrap_qb_ratings(qb_games, fit, resamples=4, seed=1)

    # Act
    second = bootstrap_qb_ratings(qb_games, fit, resamples=4, seed=1)

    # Assert
    assert second.equals(first)
    assert second.columns == ["draw", "qb_id", "adj_qb_epa_per_dropback"]


def _team_games(qb_games: pl.DataFrame) -> pl.DataFrame:
    """Return one team-game row per QB-game row's team and game, as the qualifier counts them."""
    return qb_games.select("game_id", "week", "team").unique(maintain_order=True)


def test_qb_season_summary_totals_each_passers_games_dropbacks_and_attempts() -> None:
    # Arrange
    qb_games = _qb_games(dropbacks={"CCC": 70})

    # Act
    summary = qb_season_summary(qb_games, _team_games(qb_games))

    # Assert
    row = summary.filter(pl.col("qb_id") == "qb_AAA").row(0, named=True)
    # qb_AAA plays BBB, CCC, and DDD twice each; the CCC games carry 70 dropbacks.
    assert (row["qb_games_played"], row["qb_dropbacks_total"], row["qb_attempts_total"]) == (
        6,
        4 * 35 + 2 * 70,
        4 * 33 + 2 * 68,
    )
    assert (row["qb_name"], row["team"]) == ("Q.AAA", "AAA")


def test_qb_season_summary_rate_is_the_dropback_epa_over_the_dropbacks() -> None:
    # Arrange
    qb_games = _qb_games(dropbacks={"CCC": 70})
    own = qb_games.filter(pl.col("qb_id") == "qb_BBB")
    expected = float(own.get_column("qb_dropback_epa").sum()) / float(
        own.get_column("qb_dropbacks").sum()
    )

    # Act
    summary = qb_season_summary(qb_games, _team_games(qb_games))

    # Assert
    assert _value(summary, "qb_BBB", "qb_epa_per_dropback") == pytest.approx(expected)


def test_qb_season_summary_qualifies_with_14_attempts_per_team_game() -> None:
    # Arrange
    qb_games = _qb_games().with_columns(
        pl.when(pl.col("qb_id") == "qb_DDD").then(13).otherwise(14).alias("qb_attempts")
    )

    # Act
    summary = qb_season_summary(qb_games, _team_games(qb_games))

    # Assert
    eligible = dict(summary.select("qb_id", "qb_is_eligible").iter_rows())
    assert eligible == {"qb_AAA": True, "qb_BBB": True, "qb_CCC": True, "qb_DDD": False}
    assert summary.get_column("qb_attempt_qualifier").to_list() == [6 * 14] * 4


def test_qb_season_summary_primary_team_breaks_a_tie_by_the_latest_game() -> None:
    # Arrange
    qb_games = _qb_games().with_columns(
        pl.when((pl.col("qb_id") == "qb_AAA") & (pl.col("week") > 3))
        .then(pl.lit("ZZZ"))
        .otherwise(pl.col("team"))
        .alias("team")
    )

    # Act
    summary = qb_season_summary(qb_games, _team_games(qb_games))

    # Assert
    # qb_AAA plays three games for AAA (weeks 1-3) and three for ZZZ (weeks 4, 7, and 10).
    assert _value_text(summary, "qb_AAA", "team") == "ZZZ"


def _value_text(frame: pl.DataFrame, qb_id: str, column: str) -> str:
    """Return one QB's text value for one column."""
    return str(frame.filter(pl.col("qb_id") == qb_id).get_column(column).item())


def test_qb_season_summary_counts_games_without_a_dropback_but_not_their_rate() -> None:
    # Arrange
    qb_games = _qb_games().with_columns(
        pl.when((pl.col("qb_id") == "qb_CCC") & (pl.col("week") == 5))
        .then(0)
        .otherwise(pl.col("qb_dropbacks"))
        .alias("qb_dropbacks"),
        pl.when((pl.col("qb_id") == "qb_CCC") & (pl.col("week") == 5))
        .then(0.0)
        .otherwise(pl.col("qb_dropback_epa"))
        .alias("qb_dropback_epa"),
    )

    # Act
    summary = qb_season_summary(qb_games, _team_games(qb_games))

    # Assert
    row = summary.filter(pl.col("qb_id") == "qb_CCC").row(0, named=True)
    assert (row["qb_games_played"], row["qb_dropbacks_total"]) == (6, 5 * 35)


def test_qb_season_summary_team_without_team_games_takes_the_most_games() -> None:
    # Arrange
    qb_games = _qb_games()
    team_games = _team_games(qb_games).filter(pl.col("team") != "DDD")

    # Act
    summary = qb_season_summary(qb_games, team_games)

    # Assert
    assert _value(summary, "qb_DDD", "qb_attempt_qualifier") == 6 * 14


def test_qb_season_summary_without_team_games_assumes_a_full_season() -> None:
    # Arrange
    qb_games = _qb_games()

    # Act
    summary = qb_season_summary(qb_games, _team_games(qb_games).clear())

    # Assert
    assert summary.get_column("qb_attempt_qualifier").to_list() == [17 * 14] * 4
