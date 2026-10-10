"""Tests for the shared offense-versus-defense ridge solver of the ratings package."""

import itertools

import numpy as np
import polars as pl
import pytest

from nfl_predictor.ratings.ridge import (
    UnitColumns,
    UnitPrior,
    build_unit_design,
    fit_unit_ridge,
    predict_unit,
    solve_unit_design,
)

_TEAMS = ("AAA", "BBB", "CCC", "DDD")
_OFFENSE = {"AAA": 0.10, "BBB": 0.05, "CCC": -0.05, "DDD": -0.10}
_DEFENSE = {"AAA": -0.04, "BBB": 0.08, "CCC": 0.00, "DDD": -0.04}
_INTERCEPT = 0.02
_HOME = 0.01
_COLUMNS = UnitColumns(response="epa_per_play", weight="plays")


def _schedule_rows(*, rounds: int = 1) -> pl.DataFrame:
    """Return ``rounds`` double round-robins of offense rows built from the known effects."""
    rows: list[dict[str, object]] = []
    matchups = itertools.product(range(rounds), itertools.permutations(_TEAMS, 2))
    for game_number, (_, (home, away)) in enumerate(matchups):
        game_id = f"g{game_number:03d}"
        for offense, defense, is_home in ((home, away, True), (away, home, False)):
            value = (
                _INTERCEPT + _OFFENSE[offense] - _DEFENSE[defense] + (_HOME if is_home else -_HOME)
            )
            rows.append(
                {
                    "game_id": game_id,
                    "team": offense,
                    "opponent_team": defense,
                    "is_home": is_home,
                    "plays": 60,
                    "epa_per_play": value,
                }
            )
    return pl.DataFrame(rows)


def _noisy_rows(seed: int) -> pl.DataFrame:
    """Return one double round-robin of the known effects plus per-row noise."""
    rows = _schedule_rows()
    noise = np.random.default_rng(seed).normal(0.0, 0.05, rows.height)
    return rows.with_columns(pl.col("epa_per_play") + pl.Series(noise))


def test_fit_unit_ridge_noiseless_schedule_recovers_offense_effects() -> None:
    # Arrange
    rows = _schedule_rows()

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=1e-6)

    # Assert
    assert fit.offense == pytest.approx(_OFFENSE, abs=1e-6)


def test_fit_unit_ridge_noiseless_schedule_recovers_defense_effects() -> None:
    # Arrange
    rows = _schedule_rows()

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=1e-6)

    # Assert
    assert fit.defense == pytest.approx(_DEFENSE, abs=1e-6)


def test_fit_unit_ridge_noiseless_schedule_recovers_intercept_and_home_field() -> None:
    # Arrange
    rows = _schedule_rows()

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=1e-6)

    # Assert
    assert (fit.intercept, fit.home_field) == pytest.approx((_INTERCEPT, _HOME), abs=1e-6)


def test_fit_unit_ridge_heavy_penalty_shrinks_effects_but_keeps_league_mean() -> None:
    # Arrange
    rows = _schedule_rows()

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=1e9)

    # Assert
    assert max(abs(value) for value in fit.offense.values()) < 1e-4
    assert fit.intercept == pytest.approx(_INTERCEPT, abs=1e-6)


def test_fit_unit_ridge_penalized_effects_sum_to_zero() -> None:
    """The unpenalized intercept absorbs the league mean, so each effect set sums to zero."""
    # Arrange
    rows = _noisy_rows(seed=6)

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=25.0)

    # Assert
    assert sum(fit.offense.values()) == pytest.approx(0.0, abs=1e-12)
    assert sum(fit.defense.values()) == pytest.approx(0.0, abs=1e-12)


def test_fit_unit_ridge_zero_weight_rows_do_not_move_the_fit() -> None:
    # Arrange
    rows = _schedule_rows()
    outlier = rows.head(1).with_columns(
        pl.lit(0, dtype=pl.Int64).alias("plays"), pl.lit(99.0).alias("epa_per_play")
    )
    padded = pl.concat([rows, outlier])

    # Act
    fit = fit_unit_ridge(padded, _COLUMNS, ridge_lambda=1e-6)

    # Assert
    assert fit.offense == pytest.approx(_OFFENSE, abs=1e-6)


def test_fit_unit_ridge_noiseless_schedule_cross_validates_to_smallest_penalty() -> None:
    # Arrange
    rows = _schedule_rows()
    grid = np.array([1e-3, 1.0, 1e3])

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, candidate_lambdas=grid)

    # Assert
    assert fit.ridge_lambda == pytest.approx(1e-3)


def test_fit_unit_ridge_pure_noise_cross_validates_to_largest_penalty() -> None:
    # Arrange
    rows = _schedule_rows(rounds=5)
    noise_only = rows.with_columns(
        pl.Series("epa_per_play", np.random.default_rng(7).normal(0.0, 0.5, rows.height))
    )
    grid = np.array([1e-3, 1.0, 1e6])

    # Act
    fit = fit_unit_ridge(noise_only, _COLUMNS, candidate_lambdas=grid)

    # Assert
    assert fit.ridge_lambda == pytest.approx(1e6)


def test_fit_unit_ridge_cross_validation_folds_group_whole_games() -> None:
    """Shuffling the rows leaves the folds, and so the chosen penalty, unchanged."""
    # Arrange
    rows = _noisy_rows(seed=8)
    shuffled = rows.sample(fraction=1.0, shuffle=True, seed=3)
    expected = fit_unit_ridge(rows, _COLUMNS).ridge_lambda

    # Act
    chosen = fit_unit_ridge(shuffled, _COLUMNS).ridge_lambda

    # Assert
    assert chosen == expected


def test_fit_unit_ridge_empty_rows_raises_value_error() -> None:
    # Arrange
    rows = _schedule_rows().clear()

    # Act & Assert
    with pytest.raises(ValueError, match="no rows"):
        fit_unit_ridge(rows, _COLUMNS)


def test_fit_unit_ridge_without_penalty_falls_back_to_least_squares() -> None:
    # Arrange
    rows = _schedule_rows()

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=0.0)

    # Assert
    assert np.isfinite(list(fit.offense.values())).all()


def test_fit_unit_ridge_single_game_uses_the_smallest_candidate_penalty() -> None:
    # Arrange
    rows = _schedule_rows().filter(pl.col("game_id") == "g000")
    grid = np.array([1e-3, 1.0, 1e3])

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, candidate_lambdas=grid)

    # Assert
    assert fit.ridge_lambda == pytest.approx(1e-3)


def test_fit_unit_ridge_all_zero_weights_uses_the_smallest_candidate_penalty() -> None:
    # Arrange
    rows = _schedule_rows().with_columns(pl.lit(0, dtype=pl.Int64).alias("plays"))
    grid = np.array([1e-3, 1.0, 1e3])

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, candidate_lambdas=grid)

    # Assert
    assert fit.ridge_lambda == pytest.approx(1e-3)


def test_fit_unit_ridge_without_a_home_column_fits_no_home_field() -> None:
    # Arrange
    rows = _schedule_rows().drop("is_home")

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=1e-6)

    # Assert
    assert fit.home_field == 0.0


def test_fit_unit_ridge_without_a_weight_column_weights_rows_equally() -> None:
    # Arrange
    rows = _schedule_rows()
    unweighted = UnitColumns(response="epa_per_play")

    # Act
    fit = fit_unit_ridge(rows.drop("plays"), unweighted, ridge_lambda=1e-6)

    # Assert
    assert fit.offense == pytest.approx(_OFFENSE, abs=1e-6)


def test_predict_unit_adds_intercept_offense_defense_and_home_field() -> None:
    # Arrange
    fit = fit_unit_ridge(_schedule_rows(rounds=2), _COLUMNS, ridge_lambda=1e-6)
    rows = pl.DataFrame({"team": ["AAA"], "opponent_team": ["BBB"], "is_home": [True]})
    expected = _INTERCEPT + _OFFENSE["AAA"] - _DEFENSE["BBB"] + _HOME

    # Act
    predicted = predict_unit(fit, rows, _COLUMNS)

    # Assert
    assert predicted.tolist() == pytest.approx([expected], abs=1e-6)


def test_predict_unit_treats_a_neutral_site_as_no_home_field() -> None:
    # Arrange
    fit = fit_unit_ridge(_schedule_rows(rounds=2), _COLUMNS, ridge_lambda=1e-6)
    rows = pl.DataFrame({"team": ["AAA"], "opponent_team": ["BBB"], "is_home": [None]})
    expected = _INTERCEPT + _OFFENSE["AAA"] - _DEFENSE["BBB"]

    # Act
    predicted = predict_unit(fit, rows, _COLUMNS)

    # Assert
    assert predicted.tolist() == pytest.approx([expected], abs=1e-6)


def test_predict_unit_unknown_unit_is_nan() -> None:
    # Arrange
    fit = fit_unit_ridge(_schedule_rows(), _COLUMNS, ridge_lambda=1.0)
    rows = pl.DataFrame({"team": ["ZZZ", "AAA"], "opponent_team": ["AAA", "YYY"]})

    # Act
    predicted = predict_unit(fit, rows, _COLUMNS)

    # Assert
    assert np.isnan(predicted).all()


def test_solve_unit_design_with_multiplicities_matches_a_fit_on_duplicated_rows() -> None:
    # Arrange
    rows = _schedule_rows()
    counts = np.array([2, 0, 1, 3, 1, 0, 1, 2, 0, 1, 1, 2], dtype=np.float64)
    game_ids = sorted(rows.get_column("game_id").unique().to_list())
    duplicated = pl.concat(
        [
            rows.filter(pl.col("game_id") == game_id).with_columns(
                pl.lit(f"{game_id}#{copy}").alias("game_id")
            )
            for game_id, count in zip(game_ids, counts, strict=True)
            for copy in range(int(count))
        ]
    )
    expected = fit_unit_ridge(duplicated, _COLUMNS, ridge_lambda=5.0)
    design = build_unit_design(rows, _COLUMNS)
    multipliers = counts[[game_ids.index(g) for g in rows.get_column("game_id").to_list()]]

    # Act
    fit = solve_unit_design(design, 5.0, multipliers)

    # Assert
    assert fit.offense == pytest.approx(expected.offense)
    assert fit.defense == pytest.approx(expected.defense)
    assert (fit.intercept, fit.home_field) == pytest.approx(
        (expected.intercept, expected.home_field)
    )


def test_solve_unit_design_leaves_out_units_without_weight() -> None:
    # Arrange
    rows = _schedule_rows()
    design = build_unit_design(rows, _COLUMNS)
    multipliers = np.where(
        (rows.get_column("team") == "AAA").to_numpy()
        | (rows.get_column("opponent_team") == "AAA").to_numpy(),
        0.0,
        1.0,
    )

    # Act
    fit = solve_unit_design(design, 5.0, multipliers)

    # Assert
    assert "AAA" not in fit.offense
    assert "AAA" not in fit.defense


def test_fit_unit_ridge_zero_prior_matches_the_plain_fit() -> None:
    # Arrange
    rows = _noisy_rows(seed=1)
    plain = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=50.0)
    zero = UnitPrior(offense=dict.fromkeys(_TEAMS, 0.0), defense=dict.fromkeys(_TEAMS, 0.0))

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=50.0, prior=zero)

    # Assert
    assert fit == plain


def test_fit_unit_ridge_overwhelming_penalty_returns_the_prior() -> None:
    # Arrange
    prior = UnitPrior(offense={"AAA": 0.3, "BBB": -0.1}, defense={"CCC": 0.2})

    # Act
    fit = fit_unit_ridge(_noisy_rows(seed=2), _COLUMNS, ridge_lambda=1e12, prior=prior)

    # Assert
    assert fit.offense["AAA"] == pytest.approx(0.3, abs=1e-6)
    assert fit.offense["BBB"] == pytest.approx(-0.1, abs=1e-6)
    assert fit.offense["CCC"] == pytest.approx(0.0, abs=1e-6)
    assert fit.defense["CCC"] == pytest.approx(0.2, abs=1e-6)


def test_fit_unit_ridge_prior_equals_the_plain_fit_on_residuals_plus_the_prior() -> None:
    # Arrange
    rows = _noisy_rows(seed=3)
    prior = UnitPrior(
        offense={"AAA": 0.06, "BBB": 0.02, "CCC": -0.03}, defense={"BBB": 0.05, "DDD": -0.02}
    )
    residual = rows.with_columns(
        pl.col("epa_per_play")
        - pl.col("team").replace_strict(dict(prior.offense), default=0.0)
        + pl.col("opponent_team").replace_strict(dict(prior.defense), default=0.0)
    )
    plain = fit_unit_ridge(residual, _COLUMNS, ridge_lambda=40.0)

    # Act
    fit = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=40.0, prior=prior)

    # Assert
    for team in _TEAMS:
        assert fit.offense[team] == pytest.approx(
            plain.offense[team] + prior.offense.get(team, 0.0), abs=1e-12
        )
        assert fit.defense[team] == pytest.approx(
            plain.defense[team] + prior.defense.get(team, 0.0), abs=1e-12
        )
    assert fit.intercept == pytest.approx(plain.intercept, abs=1e-12)
    assert fit.home_field == pytest.approx(plain.home_field, abs=1e-12)


def test_solve_unit_design_with_a_prior_matches_fit_unit_ridge() -> None:
    # Arrange
    rows = _noisy_rows(seed=4)
    prior = UnitPrior(offense={"AAA": 0.05}, defense={"DDD": -0.03})
    design = build_unit_design(rows, _COLUMNS)
    expected = fit_unit_ridge(rows, _COLUMNS, ridge_lambda=30.0, prior=prior)

    # Act
    fit = solve_unit_design(design, 30.0, np.ones(rows.height), prior=prior)

    # Assert
    assert fit.offense == pytest.approx(expected.offense, abs=1e-12)
    assert fit.defense == pytest.approx(expected.defense, abs=1e-12)


def test_fit_unit_ridge_refuses_a_prior_without_a_fixed_penalty() -> None:
    """Cross-validation sets the penalty for a prior of zero, so it never sees a prior."""
    # Arrange
    prior = UnitPrior(offense=dict(_OFFENSE), defense=dict(_DEFENSE))

    # Act & Assert
    with pytest.raises(ValueError, match="fixed penalty"):
        fit_unit_ridge(_noisy_rows(seed=5), _COLUMNS, prior=prior)
