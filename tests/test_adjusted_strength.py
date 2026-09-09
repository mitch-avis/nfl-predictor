"""Tests for the simultaneous opponent-adjusted team strength solvers."""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nfl_predictor.utils.polars import adjusted_strength

TEAMS: tuple[str, ...] = ("ARI", "BAL", "CHI", "DAL", "GB", "KC")
TRUE_OFFENSE: dict[str, float] = {
    "ARI": -3.0,
    "BAL": 5.0,
    "CHI": -1.5,
    "DAL": 2.5,
    "GB": 0.5,
    "KC": 6.0,
}
TRUE_DEFENSE: dict[str, float] = {
    "ARI": 4.0,
    "BAL": -2.0,
    "CHI": 1.0,
    "DAL": -3.5,
    "GB": 2.0,
    "KC": -5.0,
}


def _centered(values: dict[str, float]) -> dict[str, float]:
    """Return a copy of ``values`` shifted so its entries average to zero."""
    mean = sum(values.values()) / len(values)
    return {key: value - mean for key, value in values.items()}


def _double_round_robin(
    offense: dict[str, float],
    defense: dict[str, float],
    home_field_advantage: float,
) -> pl.DataFrame:
    """Build a noiseless double round-robin team-game frame from known strengths."""
    rows: list[dict[str, object]] = []
    for home in offense:
        for away in offense:
            if home == away:
                continue
            rows.append(
                {
                    "team_abbr": home,
                    "opponent_abbr": away,
                    "is_home": True,
                    "response": offense[home] - defense[away] + home_field_advantage,
                }
            )
            rows.append(
                {
                    "team_abbr": away,
                    "opponent_abbr": home,
                    "is_home": False,
                    "response": offense[away] - defense[home] - home_field_advantage,
                }
            )
    return pl.DataFrame(rows)


def _ratings_lookup(ratings: pl.DataFrame, column: str) -> dict[str, float]:
    """Return a team-keyed lookup for one rating column of a solved frame."""
    return dict(
        zip(
            ratings.get_column("team_abbr").to_list(),
            ratings.get_column(column).to_list(),
            strict=True,
        )
    )


def test_round_robin_recovers_known_offense_and_defense_strengths() -> None:
    """A noiseless double round-robin recovers the centered generating strengths."""
    team_games = _double_round_robin(TRUE_OFFENSE, TRUE_DEFENSE, 2.0)

    ratings, _ = adjusted_strength.solve_team_ridge(team_games, "response", ridge_lambda=1e-6)

    offense = _ratings_lookup(ratings, "offense_rating")
    defense = _ratings_lookup(ratings, "defense_rating")
    expected_offense = _centered(TRUE_OFFENSE)
    expected_defense = _centered(TRUE_DEFENSE)
    for team in TEAMS:
        assert offense[team] == pytest.approx(expected_offense[team], abs=1e-4)
        assert defense[team] == pytest.approx(expected_defense[team], abs=1e-4)


def test_home_field_term_matches_positive_truth() -> None:
    """A positive generating home-field edge is recovered with a positive sign."""
    team_games = _double_round_robin(TRUE_OFFENSE, TRUE_DEFENSE, 2.5)

    _, home_field_advantage = adjusted_strength.solve_team_ridge(
        team_games, "response", ridge_lambda=1e-6
    )

    assert home_field_advantage > 0.0
    assert home_field_advantage == pytest.approx(2.5, abs=1e-4)


def test_home_field_term_flips_sign_for_negative_truth() -> None:
    """A negative generating home-field edge is recovered with a negative sign."""
    team_games = _double_round_robin(TRUE_OFFENSE, TRUE_DEFENSE, -1.75)

    _, home_field_advantage = adjusted_strength.solve_team_ridge(
        team_games, "response", ridge_lambda=1e-6
    )

    assert home_field_advantage < 0.0
    assert home_field_advantage == pytest.approx(-1.75, abs=1e-4)


def test_offense_and_defense_coefficients_are_independently_centered() -> None:
    """Offense and defense coefficients each average to zero on noisy input."""
    rng = np.random.default_rng(1234)
    team_games = _double_round_robin(TRUE_OFFENSE, TRUE_DEFENSE, 2.0)
    noisy = team_games.with_columns(
        (pl.col("response") + pl.Series(rng.normal(0.0, 3.0, team_games.height))).alias("response")
    )

    ratings, _ = adjusted_strength.solve_team_ridge(noisy, "response", ridge_lambda=0.5)

    assert ratings.get_column("offense_rating").sum() == pytest.approx(0.0, abs=1e-5)
    assert ratings.get_column("defense_rating").sum() == pytest.approx(0.0, abs=1e-5)


def test_empty_input_returns_typed_empty_ratings_frame() -> None:
    """Empty input yields a typed empty ratings frame and a zero home-field term."""
    ratings, home_field_advantage = adjusted_strength.solve_team_ridge(
        pl.DataFrame(), "response", ridge_lambda=1.0
    )

    assert ratings.is_empty()
    assert ratings.schema == pl.Schema(
        {
            "team_abbr": pl.String,
            "offense_rating": pl.Float64,
            "defense_rating": pl.Float64,
        }
    )
    assert home_field_advantage == 0.0


def test_all_null_rows_return_typed_empty_ratings_frame() -> None:
    """Input that becomes empty after dropping nulls yields the typed empty frame."""
    team_games = pl.DataFrame(
        {
            "team_abbr": ["KC", "GB"],
            "opponent_abbr": ["GB", "KC"],
            "is_home": [True, False],
            "response": [None, None],
        },
        schema_overrides={"response": pl.Float64},
    )

    ratings, home_field_advantage = adjusted_strength.solve_team_ridge(
        team_games, "response", ridge_lambda=1.0
    )

    assert ratings.is_empty()
    assert ratings.schema == pl.Schema(
        {
            "team_abbr": pl.String,
            "offense_rating": pl.Float64,
            "defense_rating": pl.Float64,
        }
    )
    assert home_field_advantage == 0.0


def test_single_game_input_returns_two_team_frame() -> None:
    """A one-game frame solves without raising and returns both teams."""
    team_games = pl.DataFrame(
        {
            "team_abbr": ["KC", "GB"],
            "opponent_abbr": ["GB", "KC"],
            "is_home": [True, False],
            "response": [7.0, -7.0],
        }
    )

    ratings, home_field_advantage = adjusted_strength.solve_team_ridge(
        team_games, "response", ridge_lambda=1.0
    )

    assert ratings.height == 2
    assert ratings.get_column("team_abbr").to_list() == ["GB", "KC"]
    assert ratings.schema == pl.Schema(
        {
            "team_abbr": pl.String,
            "offense_rating": pl.Float64,
            "defense_rating": pl.Float64,
        }
    )
    assert isinstance(home_field_advantage, float)


def test_missing_home_column_solves_without_home_field_term() -> None:
    """A frame without the home indicator solves with a zero home-field column."""
    team_games = _double_round_robin(TRUE_OFFENSE, TRUE_DEFENSE, 0.0).drop("is_home")

    ratings, home_field_advantage = adjusted_strength.solve_team_ridge(
        team_games, "response", ridge_lambda=1e-6
    )

    assert ratings.height == len(TEAMS)
    assert home_field_advantage == 0.0


def test_repeated_solves_are_deterministic() -> None:
    """Solving the same frame twice produces frame-equal ratings."""
    team_games = _double_round_robin(TRUE_OFFENSE, TRUE_DEFENSE, 2.0)

    first, first_hfa = adjusted_strength.solve_team_ridge(team_games, "response", ridge_lambda=0.25)
    second, second_hfa = adjusted_strength.solve_team_ridge(
        team_games, "response", ridge_lambda=0.25
    )

    assert_frame_equal(first, second)
    assert first_hfa == second_hfa


def test_opponent_only_team_receives_defense_coefficient() -> None:
    """A team seen only in the opponent column still appears with both coefficients."""
    team_games = pl.DataFrame(
        {
            "team_abbr": ["KC", "GB", "KC"],
            "opponent_abbr": ["GB", "KC", "SEA"],
            "is_home": [True, False, True],
            "response": [7.0, -7.0, 3.0],
        }
    )

    ratings, _ = adjusted_strength.solve_team_ridge(team_games, "response", ridge_lambda=1.0)

    assert "SEA" in ratings.get_column("team_abbr").to_list()
    seattle = ratings.filter(pl.col("team_abbr") == "SEA")
    assert seattle.get_column("offense_rating").item() is not None
    assert seattle.get_column("defense_rating").item() is not None


def test_rank_deficient_design_falls_back_to_least_squares() -> None:
    """A zero ridge penalty on a rank-deficient design solves instead of raising."""
    team_games = _double_round_robin(TRUE_OFFENSE, TRUE_DEFENSE, 2.0)

    ratings, home_field_advantage = adjusted_strength.solve_team_ridge(
        team_games, "response", ridge_lambda=0.0
    )

    offense = _ratings_lookup(ratings, "offense_rating")
    expected_offense = _centered(TRUE_OFFENSE)
    for team in TEAMS:
        assert offense[team] == pytest.approx(expected_offense[team], abs=1e-4)
    assert home_field_advantage == pytest.approx(2.0, abs=1e-4)


def test_singular_matrix_error_falls_back_to_least_squares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A singular normal-equation solve degrades to least squares instead of raising."""

    def _raise_singular(
        matrix: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Stand in for ``numpy.linalg.solve`` and always report a singular matrix."""
        raise np.linalg.LinAlgError("singular matrix")

    monkeypatch.setattr(np.linalg, "solve", _raise_singular)
    team_games = _double_round_robin(TRUE_OFFENSE, TRUE_DEFENSE, 2.0)

    ratings, home_field_advantage = adjusted_strength.solve_team_ridge(
        team_games, "response", ridge_lambda=1e-6
    )

    offense = _ratings_lookup(ratings, "offense_rating")
    expected_offense = _centered(TRUE_OFFENSE)
    for team in TEAMS:
        assert offense[team] == pytest.approx(expected_offense[team], abs=1e-4)
    assert home_field_advantage == pytest.approx(2.0, abs=1e-4)


def test_null_rows_are_dropped_before_solving() -> None:
    """A row with a null response leaves the solved ratings unchanged."""
    team_games = _double_round_robin(TRUE_OFFENSE, TRUE_DEFENSE, 2.0)
    polluted = pl.concat(
        [
            team_games,
            pl.DataFrame(
                {
                    "team_abbr": ["KC", None],
                    "opponent_abbr": [None, "GB"],
                    "is_home": [True, False],
                    "response": [None, 12.0],
                },
                schema=team_games.schema,
            ),
        ]
    )

    clean_ratings, clean_hfa = adjusted_strength.solve_team_ridge(
        team_games, "response", ridge_lambda=0.25
    )
    polluted_ratings, polluted_hfa = adjusted_strength.solve_team_ridge(
        polluted, "response", ridge_lambda=0.25
    )

    assert_frame_equal(clean_ratings, polluted_ratings)
    assert clean_hfa == polluted_hfa


def test_missing_required_column_raises_value_error() -> None:
    """A frame without the requested response column reports a clear error."""
    team_games = pl.DataFrame({"team_abbr": ["KC"], "opponent_abbr": ["GB"]})

    with pytest.raises(ValueError, match="response"):
        adjusted_strength.solve_team_ridge(team_games, "response", ridge_lambda=1.0)


def test_build_team_design_matrix_encodes_signs_and_team_universe() -> None:
    """The design matrix encodes offense, defense, and home-field columns as expected."""
    team_games = pl.DataFrame(
        {
            "team_abbr": ["KC", "GB"],
            "opponent_abbr": ["GB", "SEA"],
            "is_home": [True, None],
            "response": [7.0, -3.0],
        }
    )

    design, response, teams = adjusted_strength.build_team_design_matrix(
        team_games,
        "response",
        team_col="team_abbr",
        opponent_col="opponent_abbr",
        home_col="is_home",
    )

    assert teams == ["GB", "KC", "SEA"]
    assert design.shape == (2, len(teams) * 2 + 1)
    assert response.tolist() == [7.0, -3.0]
    assert design[0, teams.index("KC")] == 1.0
    assert design[0, len(teams) + teams.index("GB")] == -1.0
    assert design[0, -1] == 1.0
    assert design[1, teams.index("GB")] == 1.0
    assert design[1, len(teams) + teams.index("SEA")] == -1.0
    assert design[1, -1] == 0.0


def test_solve_srs_recovers_known_ratings() -> None:
    """A noiseless round-robin margin response recovers the centered generating ratings."""
    true_ratings = _centered({"ARI": -6.0, "BAL": 1.0, "CHI": 3.0, "DAL": 8.0})
    rows: list[dict[str, object]] = []
    for team in true_ratings:
        for opponent in true_ratings:
            if team == opponent:
                continue
            rows.append(
                {
                    "team_abbr": team,
                    "opponent_abbr": opponent,
                    "margin": true_ratings[team] - true_ratings[opponent],
                }
            )
    team_games = pl.DataFrame(rows)

    ratings = adjusted_strength.solve_srs(team_games, "margin")

    solved = dict(
        zip(
            ratings.get_column("team_abbr").to_list(),
            ratings.get_column("srs_rating").to_list(),
            strict=True,
        )
    )
    for team, expected in true_ratings.items():
        assert solved[team] == pytest.approx(expected, abs=1e-6)


def test_solve_srs_ratings_are_centered() -> None:
    """Simple rating system output averages to zero."""
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for team in TEAMS:
        for opponent in TEAMS:
            if team == opponent:
                continue
            rows.append(
                {
                    "team_abbr": team,
                    "opponent_abbr": opponent,
                    "margin": float(rng.normal(0.0, 10.0)),
                }
            )

    ratings = adjusted_strength.solve_srs(pl.DataFrame(rows), "margin")

    assert ratings.get_column("srs_rating").sum() == pytest.approx(0.0, abs=1e-5)


def test_solve_srs_empty_input_returns_typed_empty_frame() -> None:
    """Empty simple rating system input yields a typed empty frame."""
    ratings = adjusted_strength.solve_srs(pl.DataFrame(), "margin")

    assert ratings.is_empty()
    assert ratings.schema == pl.Schema({"team_abbr": pl.String, "srs_rating": pl.Float64})


def test_solve_srs_drops_null_rows() -> None:
    """Null rows do not change the simple rating system solution."""
    base = pl.DataFrame(
        {
            "team_abbr": ["KC", "GB", "KC", "GB"],
            "opponent_abbr": ["GB", "KC", "GB", "KC"],
            "margin": [7.0, -7.0, 3.0, -3.0],
        }
    )
    polluted = pl.concat(
        [
            base,
            pl.DataFrame(
                {"team_abbr": ["KC"], "opponent_abbr": [None], "margin": [None]},
                schema=base.schema,
            ),
        ]
    )

    assert_frame_equal(
        adjusted_strength.solve_srs(base, "margin"),
        adjusted_strength.solve_srs(polluted, "margin"),
    )


def test_tune_ridge_lambda_returns_grid_member_and_is_deterministic() -> None:
    """The tuner returns a candidate from the grid and repeats its choice exactly."""
    team_games = _double_round_robin(TRUE_OFFENSE, TRUE_DEFENSE, 2.0)
    design, response, _ = adjusted_strength.build_team_design_matrix(
        team_games,
        "response",
        team_col="team_abbr",
        opponent_col="opponent_abbr",
        home_col="is_home",
    )

    first = adjusted_strength.tune_ridge_lambda(design, response)
    second = adjusted_strength.tune_ridge_lambda(design, response)

    assert first == second
    assert any(
        first == pytest.approx(float(candidate))
        for candidate in adjusted_strength.DEFAULT_RIDGE_LAMBDAS
    )


def test_tune_ridge_lambda_handles_short_inputs() -> None:
    """A single-row system returns the smallest candidate rather than raising."""
    design = np.array([[1.0, -1.0]], dtype=np.float64)
    response = np.array([3.0], dtype=np.float64)

    chosen = adjusted_strength.tune_ridge_lambda(design, response)

    assert chosen == pytest.approx(float(adjusted_strength.DEFAULT_RIDGE_LAMBDAS[0]))


def test_tune_ridge_lambda_prefers_a_larger_penalty_under_heavy_noise() -> None:
    """Heavier response noise pushes the tuner toward a larger ridge penalty."""
    rng = np.random.default_rng(20240909)
    team_games = _double_round_robin(TRUE_OFFENSE, TRUE_DEFENSE, 2.0)
    design, response, _ = adjusted_strength.build_team_design_matrix(
        team_games,
        "response",
        team_col="team_abbr",
        opponent_col="opponent_abbr",
        home_col="is_home",
    )
    noisy_response = response + rng.normal(0.0, 40.0, response.shape[0])

    quiet_lambda = adjusted_strength.tune_ridge_lambda(design, response)
    noisy_lambda = adjusted_strength.tune_ridge_lambda(design, noisy_response)

    assert noisy_lambda > quiet_lambda


def test_tune_ridge_lambda_rejects_mismatched_shapes() -> None:
    """Mismatched design and response row counts report a clear error."""
    design = np.zeros((3, 2), dtype=np.float64)
    response = np.zeros(2, dtype=np.float64)

    with pytest.raises(ValueError, match="same number of rows"):
        adjusted_strength.tune_ridge_lambda(design, response)


def test_tune_ridge_lambda_rejects_non_matrix_design() -> None:
    """A one-dimensional design reports a clear error."""
    with pytest.raises(ValueError, match="2D matrix"):
        adjusted_strength.tune_ridge_lambda(
            np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        )


def test_tune_ridge_lambda_rejects_non_vector_response() -> None:
    """A two-dimensional response reports a clear error."""
    with pytest.raises(ValueError, match="1D vector"):
        adjusted_strength.tune_ridge_lambda(
            np.zeros((3, 2), dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        )


def test_tune_ridge_lambda_rejects_empty_candidate_grid() -> None:
    """An empty candidate grid reports a clear error."""
    with pytest.raises(ValueError, match="at least one value"):
        adjusted_strength.tune_ridge_lambda(
            np.zeros((3, 2), dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            candidate_lambdas=np.array([], dtype=np.float64),
        )


def test_solve_srs_all_null_rows_return_typed_empty_frame() -> None:
    """Simple rating system input that empties after dropping nulls returns the typed frame."""
    team_games = pl.DataFrame(
        {"team_abbr": ["KC", "GB"], "opponent_abbr": ["GB", "KC"], "margin": [None, None]},
        schema_overrides={"margin": pl.Float64},
    )

    ratings = adjusted_strength.solve_srs(team_games, "margin")

    assert ratings.is_empty()
    assert ratings.schema == pl.Schema({"team_abbr": pl.String, "srs_rating": pl.Float64})
