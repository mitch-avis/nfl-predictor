"""Schedule-adjusted team strength solvers built on a simultaneous ridge fit.

The solvers here estimate every team's offense and defense coefficient jointly from one row per
team-game. That joint fit is the all-hops generalization of a one-hop opponent-profiling method,
in which each opponent's profile is built only from its games against the rest of the league, with
every head-to-head game against the subject excluded so that the two sides of a matchup are
independent. Solving all teams at once extends that independence to any depth: because a team's
own games contribute to its own coefficients rather than to the opponent profile it is being
measured against, no game can contaminate a rating through the team it was played against, one hop
away or many.

The design is ported from the reference implementation in the separate, read-only
``nfl-sos-ratings`` repository (``nfl_sos_ratings/simultaneous_adjustment.py``), which is credited
as the source of the simultaneous-adjustment method, the deterministic ridge tuner, and the
independent offense/defense centering convention. Nothing from that repository is imported here;
only the design is reproduced in this repository's Polars ETL layer.

Model
-----
Each team-game row contributes one equation::

    response = offense[team] - defense[opponent] + home_field_advantage * home_sign

where ``home_sign`` is ``+1`` for a home row, ``-1`` for an away row, and ``0`` when the home
indicator is null or absent. The stacked system is solved with ridge-regularized normal equations
and the resulting offense and defense blocks are centered independently.
"""

import numpy as np
import numpy.typing as npt
import polars as pl

from nfl_predictor.utils.logger import log

DEFAULT_RIDGE_LAMBDAS: npt.NDArray[np.float64] = np.logspace(-6, 2, 17, dtype=np.float64)
"""Default candidate ridge penalties evaluated by :func:`tune_ridge_lambda`."""

_COEFFICIENT_DECIMALS = 6

FloatArray = npt.NDArray[np.float64]


def _sorted_teams(team_games: pl.DataFrame, *columns: str) -> list[str]:
    """Return the sorted union of distinct labels found in the given string columns."""
    labels: set[str] = set()
    for column in columns:
        if column in team_games.columns:
            labels.update(
                team_games.get_column(column).drop_nulls().cast(pl.String).unique().to_list()
            )
    return sorted(labels)


def _require_columns(team_games: pl.DataFrame, *columns: str) -> None:
    """Raise when any required column is missing from the team-game frame."""
    missing = [column for column in columns if column not in team_games.columns]
    if missing:
        raise ValueError(f"team_games is missing required column(s): {', '.join(missing)}")


def _response_array(team_games: pl.DataFrame, response_col: str) -> FloatArray:
    """Return the response column as a contiguous float array."""
    return team_games.get_column(response_col).cast(pl.Float64).to_numpy().astype(np.float64)


def _solve_linear_system(
    design: FloatArray, response: FloatArray, ridge_lambda: float
) -> FloatArray:
    """Solve an ordinary or ridge least-squares system.

    A non-positive ``ridge_lambda`` uses ``numpy.linalg.lstsq`` directly. Otherwise the ridge
    normal equations are solved with ``numpy.linalg.solve``, falling back to least squares if the
    penalized matrix is reported as singular rather than propagating the error.
    """
    if ridge_lambda <= 0.0:
        return _least_squares(design, response)

    gram = design.T @ design
    penalty = ridge_lambda * np.eye(design.shape[1], dtype=np.float64)
    try:
        return np.linalg.solve(gram + penalty, design.T @ response)
    except np.linalg.LinAlgError:
        log.warning("Ridge normal equations were singular; falling back to least squares.")
        return _least_squares(design, response)


def _least_squares(design: FloatArray, response: FloatArray) -> FloatArray:
    """Return the minimum-norm least-squares solution for a possibly rank-deficient design."""
    solution, *_ = np.linalg.lstsq(design, response, rcond=None)
    return np.asarray(solution, dtype=np.float64)


def _centered(coefficients: FloatArray) -> FloatArray:
    """Return coefficients shifted so that they average to zero."""
    if coefficients.size == 0:
        return coefficients
    return coefficients - coefficients.mean()


def _rounded(coefficients: FloatArray) -> list[float]:
    """Return coefficients rounded to the shared reporting precision."""
    return [float(value) for value in np.round(coefficients, _COEFFICIENT_DECIMALS)]


def _empty_team_ratings(team_col: str) -> pl.DataFrame:
    """Return the typed empty frame produced by :func:`solve_team_ridge`."""
    return pl.DataFrame(
        schema={
            team_col: pl.String,
            "offense_rating": pl.Float64,
            "defense_rating": pl.Float64,
        }
    )


def build_team_design_matrix(
    team_games: pl.DataFrame,
    response_col: str,
    *,
    team_col: str = "team_abbr",
    opponent_col: str = "opponent_abbr",
    home_col: str = "is_home",
) -> tuple[FloatArray, FloatArray, list[str]]:
    """Build the offense/defense/home design matrix and response vector for a team-game frame.

    Rows with a null team, opponent, or response are dropped first. The returned matrix has one
    column per team offense coefficient, one column per team defense coefficient, and a single
    trailing shared home-field column. The team universe is the sorted union of the team and
    opponent columns, so a team that appears only as an opponent still receives a column pair.

    Args:
        team_games: One row per team-game.
        response_col: Name of the numeric response column.
        team_col: Column holding the subject team label.
        opponent_col: Column holding the opposing team label.
        home_col: Optional boolean column that is true for the subject team's home games.

    Returns:
        A ``(design, response, teams)`` tuple where ``design`` has shape
        ``(rows, 2 * len(teams) + 1)`` and ``teams`` is the sorted team universe.

    """
    _require_columns(team_games, team_col, opponent_col, response_col)
    frame = team_games.drop_nulls([team_col, opponent_col, response_col])
    teams = _sorted_teams(frame, team_col, opponent_col)
    team_count = len(teams)
    team_index = {team: index for index, team in enumerate(teams)}

    design = np.zeros((frame.height, team_count * 2 + 1), dtype=np.float64)
    response = _response_array(frame, response_col)

    selected = [team_col, opponent_col]
    has_home = home_col in frame.columns
    if has_home:
        selected.append(home_col)

    for row_index, row in enumerate(frame.select(selected).iter_rows()):
        team, opponent, *home_value = row
        design[row_index, team_index[str(team)]] = 1.0
        design[row_index, team_count + team_index[str(opponent)]] = -1.0
        if has_home and home_value[0] is not None:
            design[row_index, -1] = 1.0 if bool(home_value[0]) else -1.0

    return design, response, teams


def tune_ridge_lambda(
    design: FloatArray,
    response: FloatArray,
    *,
    candidate_lambdas: FloatArray | None = None,
    folds: int = 5,
) -> float:
    """Choose a ridge penalty by deterministic k-fold cross-validation.

    This is an offline calibration helper. It is intended to be run once, out of band, to pick a
    single frozen penalty constant; it is not called from the ETL hot path, where the chosen
    constant is passed to :func:`solve_team_ridge` directly.

    Folds are assigned deterministically as ``arange(rows) % folds`` so repeated calls on the same
    inputs always return the same penalty. The candidate with the lowest mean validation mean
    squared error wins, and ties resolve toward the smaller penalty.

    Args:
        design: Design matrix with one row per observation.
        response: Response vector aligned with ``design``.
        candidate_lambdas: Penalties to evaluate; defaults to :data:`DEFAULT_RIDGE_LAMBDAS`.
        folds: Requested fold count, clamped to at least two and at most the row count.

    Returns:
        The selected ridge penalty.

    Raises:
        ValueError: If the inputs have the wrong dimensions, disagree on row count, or no
            candidate penalties are supplied.

    """
    if design.ndim != 2:
        raise ValueError("design must be a 2D matrix")
    if response.ndim != 1:
        raise ValueError("response must be a 1D vector")
    if design.shape[0] != response.shape[0]:
        raise ValueError("design and response must have the same number of rows")

    lambdas = (
        np.asarray(candidate_lambdas, dtype=np.float64)
        if candidate_lambdas is not None
        else DEFAULT_RIDGE_LAMBDAS
    )
    if lambdas.size == 0:
        raise ValueError("candidate_lambdas must contain at least one value")

    row_count = design.shape[0]
    if row_count < 2:
        return float(lambdas[0])

    effective_folds = min(max(folds, 2), row_count)
    fold_ids = np.arange(row_count, dtype=np.int64) % effective_folds
    best_lambda = float(lambdas[0])
    best_error = float("inf")

    for candidate in lambdas:
        ridge_lambda = float(candidate)
        fold_errors: list[float] = []
        for fold_id in range(effective_folds):
            validation_mask = fold_ids == fold_id
            training_mask = ~validation_mask
            if not validation_mask.any() or not training_mask.any():
                continue
            coefficients = _solve_linear_system(
                design[training_mask], response[training_mask], ridge_lambda
            )
            residuals = response[validation_mask] - (design[validation_mask] @ coefficients)
            fold_errors.append(float(np.mean(residuals**2)))

        if not fold_errors:
            continue

        mean_error = float(np.mean(fold_errors))
        if mean_error < best_error or (
            bool(np.isclose(mean_error, best_error)) and ridge_lambda < best_lambda
        ):
            best_error = mean_error
            best_lambda = ridge_lambda

    return best_lambda


def solve_team_ridge(
    team_games: pl.DataFrame,
    response_col: str,
    *,
    ridge_lambda: float,
    team_col: str = "team_abbr",
    opponent_col: str = "opponent_abbr",
    home_col: str = "is_home",
) -> tuple[pl.DataFrame, float]:
    """Jointly estimate offense and defense coefficients for one response column.

    Every team's offense and defense effect is estimated in a single ridge solve, so an opponent's
    contribution to a rating is never built from games shared with the team being rated, at any
    depth of the schedule graph. Offense and defense coefficients are centered independently, which
    also removes the constant that the offense-minus-defense parameterization leaves unidentified.

    Args:
        team_games: One row per team-game.
        response_col: Name of the numeric response column.
        ridge_lambda: Ridge penalty; values at or below zero fall back to least squares.
        team_col: Column holding the subject team label.
        opponent_col: Column holding the opposing team label.
        home_col: Optional boolean column that is true for the subject team's home games.

    Returns:
        A ``(ratings, home_field_advantage)`` tuple. ``ratings`` holds one row per team with
        ``offense_rating`` and ``defense_rating`` columns sorted by team label; both it and the
        home-field term are rounded to six decimals. Empty input, or input that becomes empty once
        null rows are dropped, returns a typed empty frame and ``0.0``.

    """
    if team_games.is_empty():
        return _empty_team_ratings(team_col), 0.0

    design, response, teams = build_team_design_matrix(
        team_games,
        response_col,
        team_col=team_col,
        opponent_col=opponent_col,
        home_col=home_col,
    )
    if not teams or design.shape[0] == 0:
        return _empty_team_ratings(team_col), 0.0

    team_count = len(teams)
    coefficients = _solve_linear_system(design, response, ridge_lambda)
    offense = _centered(coefficients[:team_count])
    defense = _centered(coefficients[team_count : team_count * 2])
    home_field_advantage = float(coefficients[-1])

    ratings = pl.DataFrame(
        {
            team_col: teams,
            "offense_rating": _rounded(offense),
            "defense_rating": _rounded(defense),
        },
        schema={team_col: pl.String, "offense_rating": pl.Float64, "defense_rating": pl.Float64},
    ).sort(team_col)
    return ratings, round(home_field_advantage, _COEFFICIENT_DECIMALS)


def solve_srs(
    team_games: pl.DataFrame,
    response_col: str,
    *,
    team_col: str = "team_abbr",
    opponent_col: str = "opponent_abbr",
    rating_col: str = "srs_rating",
) -> pl.DataFrame:
    """Solve a centered simple rating system from team-game responses.

    Each row contributes ``response = rating[team] - rating[opponent]``. The stacked system is
    solved by least squares and the ratings are centered on their mean. This is the single-effect
    companion to :func:`solve_team_ridge`, used for symmetric margin-style responses such as point
    margin or a special-teams expected-points-added margin, where a separate offense and defense
    split is not meaningful.

    Args:
        team_games: One row per team-game.
        response_col: Name of the numeric response column.
        team_col: Column holding the subject team label.
        opponent_col: Column holding the opposing team label.
        rating_col: Name of the solved rating column in the returned frame.

    Returns:
        One row per team with the centered rating, sorted by team label and rounded to six
        decimals. Empty input, or input that becomes empty once null rows are dropped, returns a
        typed empty frame.

    """
    empty = pl.DataFrame(schema={team_col: pl.String, rating_col: pl.Float64})
    if team_games.is_empty():
        return empty

    _require_columns(team_games, team_col, opponent_col, response_col)
    frame = team_games.drop_nulls([team_col, opponent_col, response_col])
    teams = _sorted_teams(frame, team_col, opponent_col)
    if not teams or frame.height == 0:
        return empty

    team_index = {team: index for index, team in enumerate(teams)}
    design = np.zeros((frame.height, len(teams)), dtype=np.float64)
    response = _response_array(frame, response_col)

    for row_index, (team, opponent) in enumerate(
        frame.select([team_col, opponent_col]).iter_rows()
    ):
        design[row_index, team_index[str(team)]] = 1.0
        design[row_index, team_index[str(opponent)]] = -1.0

    ratings = _centered(_least_squares(design, response))
    return pl.DataFrame(
        {team_col: teams, rating_col: _rounded(ratings)},
        schema={team_col: pl.String, rating_col: pl.Float64},
    ).sort(team_col)
