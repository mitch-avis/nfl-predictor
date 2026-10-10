"""Weighted offense-versus-defense ridge solver shared by the team and quarterback ratings.

Every rating in this package uses the same model: one row per matchup of an offensive unit against
a defensive unit in one game, a response measured per play, and::

    response = intercept + offense[unit] - defense[opponent] + home_field * home_sign

fit by weighted least squares with a ridge penalty on the offense and defense effects only. The
intercept carries the league average and home field is one shared effect, so neither is shrunk.
A positive offense effect is a better offense; a positive defense effect is a better defense.
Because the intercept is unpenalized, both effect sets sum to zero across their units.

Solving every unit at once is what makes the adjustment recursive: an offense is judged against
the defenses it faced, and each of those defenses is judged against every offense it faced.

The penalty is chosen by deterministic k-fold cross-validation with folds grouped by game, so the
plays of one game never sit on both sides of a fold: game ``i`` of the sorted game ids goes to fold
``i mod 5``.

An optional prior (``UnitPrior``) moves the penalty's target from zero to a mean per unit: the fit
then minimizes the weighted squared error plus ``penalty * (effect - prior mean)^2``, so thin
evidence leaves an effect near its prior instead of near average. A zero prior is the plain fit.

Floating-point note: the normal equations sum over the rows in the order given, so the same rows in
another order can move a solution in its last bits. The published-log adapter
(:mod:`nfl_predictor.ratings.published_logs`) keeps the row order ``nfl-sos-ratings`` fit with.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

type FloatArray = npt.NDArray[np.float64]

# Candidate penalties, in units of the row weights (plays). The span covers small-sample special
# teams fits through play-level offense fits so the cross-validated choice lands inside the grid.
DEFAULT_RIDGE_LAMBDAS: FloatArray = np.logspace(-2.0, 5.0, 29, dtype=np.float64)
CROSS_VALIDATION_FOLDS = 5
_MIN_GROUPS = 2


@dataclass(frozen=True, slots=True)
class UnitColumns:
    """Column names for one offense-versus-defense ridge fit."""

    response: str
    weight: str | None = None
    offense: str = "team"
    defense: str = "opponent_team"
    home: str | None = "is_home"
    group: str = "game_id"


@dataclass(frozen=True, slots=True)
class UnitFit:
    """Solved effects for one offense-versus-defense ridge fit, in response units."""

    intercept: float
    home_field: float
    ridge_lambda: float
    offense: dict[str, float]
    defense: dict[str, float]


@dataclass(frozen=True, slots=True)
class UnitPrior:
    """Prior means for one fit's effects, in response units; a unit not listed has a mean of zero.

    The penalty pulls each offense effect toward ``offense[unit]`` and each defense effect toward
    ``defense[unit]`` instead of toward zero.
    """

    offense: Mapping[str, float]
    defense: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class UnitDesign:
    """A weighted design matrix with its labels and penalty mask, built once and refit often."""

    design: FloatArray
    response: FloatArray
    weights: FloatArray
    groups: npt.NDArray[np.int64]
    penalized: FloatArray
    offense_labels: list[str]
    defense_labels: list[str]
    has_home: bool


def _values(rows: pl.DataFrame, column: str, dtype: pl.DataType) -> list[Any]:
    """Return a column's values as Python objects of ``dtype``, ``None`` where missing.

    A column already of ``dtype`` is read as it is: a fit builds thousands of small designs, and
    each Polars expression on a column costs a query, which with a large thread pool dwarfs the
    arithmetic.
    """
    series = rows.get_column(column)
    return (series if series.dtype == dtype else series.cast(dtype)).to_list()


def _home_signs(rows: pl.DataFrame, home_column: str | None) -> FloatArray:
    """Return +1 for home rows, -1 for away rows, and 0 for neutral or unknown sites."""
    if home_column is None or home_column not in rows.columns:
        return np.zeros(rows.height, dtype=np.float64)
    flags = _values(rows, home_column, pl.Boolean())
    return np.array(
        [0.0 if flag is None else 1.0 if flag else -1.0 for flag in flags], dtype=np.float64
    )


def build_unit_design(rows: pl.DataFrame, columns: UnitColumns) -> UnitDesign:
    """Return the design matrix, response, weights, and fold groups for ``rows``.

    Rows without an offense, defense, or response are dropped. Weights below zero count as zero,
    and a null weight as zero.

    Raises:
        ValueError: If no row has an offense, defense, and response.

    """
    required = (columns.offense, columns.defense, columns.response)
    if any(rows.get_column(column).null_count() for column in required):
        rows = rows.drop_nulls(list(required))
    if rows.is_empty():
        msg = "fit_unit_ridge received no rows with an offense, defense, and response"
        raise ValueError(msg)

    offense_units = _values(rows, columns.offense, pl.String())
    defense_units = _values(rows, columns.defense, pl.String())
    offense_labels = sorted(set(offense_units))
    defense_labels = sorted(set(defense_units))
    home = _home_signs(rows, columns.home)
    has_home = bool(np.any(home != 0.0))

    fixed_count = 2 if has_home else 1
    offense_start = fixed_count
    defense_start = offense_start + len(offense_labels)
    design = np.zeros((rows.height, defense_start + len(defense_labels)), dtype=np.float64)
    row_index = np.arange(rows.height)
    design[:, 0] = 1.0
    if has_home:
        design[:, 1] = home
    design[row_index, offense_start + _indices(offense_units, offense_labels)] = 1.0
    design[row_index, defense_start + _indices(defense_units, defense_labels)] = -1.0

    weights = (
        np.ones(rows.height, dtype=np.float64)
        if columns.weight is None
        else np.clip(
            np.array(
                [
                    0.0 if weight is None else weight
                    for weight in _values(rows, columns.weight, pl.Float64())
                ],
                dtype=np.float64,
            ),
            0.0,
            None,
        )
    )
    group_values = _values(rows, columns.group, pl.String())
    groups = _indices(group_values, sorted(set(group_values)))
    penalized = np.zeros(design.shape[1], dtype=np.float64)
    penalized[fixed_count:] = 1.0
    return UnitDesign(
        design=design,
        response=np.array(_values(rows, columns.response, pl.Float64()), dtype=np.float64),
        weights=weights,
        groups=groups,
        penalized=penalized,
        offense_labels=offense_labels,
        defense_labels=defense_labels,
        has_home=has_home,
    )


def _indices(values: list[Any], labels: list[str]) -> npt.NDArray[np.int64]:
    """Return each value's position in ``labels``."""
    position = {label: index for index, label in enumerate(labels)}
    return np.array([position[value] for value in values], dtype=np.int64)


def _normal_equations(
    design: FloatArray, response: FloatArray, weights: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return ``X'WX`` and ``X'Wy`` for a weighted least-squares fit."""
    weighted_design = design * weights[:, np.newaxis]
    return weighted_design.T @ design, weighted_design.T @ response


def _prior_vector(system: UnitDesign, prior: UnitPrior | None) -> FloatArray | None:
    """Return the prior mean of every design column (zero for the intercept and home field)."""
    if prior is None:
        return None
    vector = np.zeros(system.design.shape[1], dtype=np.float64)
    offense_start = 2 if system.has_home else 1
    defense_start = offense_start + len(system.offense_labels)
    for index, label in enumerate(system.offense_labels):
        vector[offense_start + index] = prior.offense.get(label, 0.0)
    for index, label in enumerate(system.defense_labels):
        vector[defense_start + index] = prior.defense.get(label, 0.0)
    return vector


def _solve(
    gram: FloatArray,
    moment: FloatArray,
    penalty: FloatArray,
    prior: FloatArray | None = None,
) -> FloatArray:
    """Solve the penalized normal equations, falling back to least squares if singular.

    With ``prior``, the penalty pulls each coefficient toward its prior mean instead of zero:
    ``(X'WX + P) b = X'Wy + P m``.
    """
    system = gram + np.diag(penalty)
    target = moment if prior is None else moment + penalty * prior
    try:
        return np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(system, target, rcond=None)
        return solution


def _cross_validated_lambda(system: UnitDesign, candidates: FloatArray) -> float:
    """Return the candidate penalty whose held-out error, summed over the folds, is lowest.

    Each fold's error is the weighted mean squared residual of its held-out rows; a tie keeps the
    smallest penalty.
    """
    ordered = np.sort(np.asarray(candidates, dtype=np.float64))
    group_count = int(system.groups.max()) + 1
    if group_count < _MIN_GROUPS:
        return float(ordered[0])

    folds = system.groups % min(CROSS_VALIDATION_FOLDS, group_count)
    errors = np.zeros(ordered.size, dtype=np.float64)
    used_folds = 0
    for fold in np.unique(folds):
        held_out = folds == fold
        held_out_weights = system.weights[held_out]
        if held_out_weights.sum() <= 0.0:
            continue
        train = ~held_out
        gram, moment = _normal_equations(
            system.design[train], system.response[train], system.weights[train]
        )
        for position, ridge_lambda in enumerate(ordered):
            coefficients = _solve(gram, moment, system.penalized * ridge_lambda)
            residuals = system.response[held_out] - system.design[held_out] @ coefficients
            errors[position] += float(np.average(residuals**2, weights=held_out_weights))
        used_folds += 1

    if used_folds == 0:
        return float(ordered[0])
    return float(ordered[int(np.argmin(errors))])


def fit_unit_ridge(
    rows: pl.DataFrame,
    columns: UnitColumns,
    *,
    ridge_lambda: float | None = None,
    candidate_lambdas: FloatArray | None = None,
    prior: UnitPrior | None = None,
) -> UnitFit:
    """Fit offense, defense, intercept, and home-field effects for one response.

    Args:
        rows: One row per offense-versus-defense matchup with the columns named in ``columns``.
        columns: Column names for the fit.
        ridge_lambda: Fixed penalty; when ``None`` it is chosen by grouped cross-validation.
        candidate_lambdas: Penalty grid for cross-validation (default ``DEFAULT_RIDGE_LAMBDAS``).
        prior: Prior means the penalty pulls the effects toward (zero when ``None``). It needs a
            fixed ``ridge_lambda``: cross-validation sets the penalty for a prior of zero.

    Returns:
        The solved effects in response units.

    Raises:
        ValueError: If no row has an offense, defense, and response, or a prior comes without a
            fixed penalty.

    """
    if prior is not None and ridge_lambda is None:
        msg = "a prior needs a fixed penalty; cross-validation sets it for a prior of zero"
        raise ValueError(msg)
    system = build_unit_design(rows, columns)
    prior_vector = _prior_vector(system, prior)
    resolved_lambda = (
        ridge_lambda
        if ridge_lambda is not None
        else _cross_validated_lambda(
            system, DEFAULT_RIDGE_LAMBDAS if candidate_lambdas is None else candidate_lambdas
        )
    )
    gram, moment = _normal_equations(system.design, system.response, system.weights)
    coefficients = _solve(gram, moment, system.penalized * resolved_lambda, prior_vector)

    offense_start = 2 if system.has_home else 1
    defense_start = offense_start + len(system.offense_labels)
    return UnitFit(
        intercept=float(coefficients[0]),
        home_field=float(coefficients[1]) if system.has_home else 0.0,
        ridge_lambda=float(resolved_lambda),
        offense={
            label: float(coefficients[offense_start + index])
            for index, label in enumerate(system.offense_labels)
        },
        defense={
            label: float(coefficients[defense_start + index])
            for index, label in enumerate(system.defense_labels)
        },
    )


def solve_unit_design(
    design: UnitDesign,
    ridge_lambda: float,
    multipliers: FloatArray,
    prior: UnitPrior | None = None,
) -> UnitFit:
    """Refit a prebuilt design with each row's weight scaled by ``multipliers``.

    With a fixed penalty, a row counted k times enters the fit exactly as one row with k times the
    weight, so a bootstrap resample of games is a vector of per-row game counts, and a refit that
    leaves games out gives them a multiplier of 0. Units whose rows all have zero weight are left
    out of the result, as they would be absent from a fit on the resampled rows themselves.

    Args:
        design: From :func:`build_unit_design`.
        ridge_lambda: The fixed penalty.
        multipliers: One non-negative factor per design row.
        prior: Prior means the penalty pulls the effects toward (zero when ``None``).

    Returns:
        The solved effects for the units with weight.

    """
    weights = design.weights * multipliers
    gram, moment = _normal_equations(design.design, design.response, weights)
    coefficients = _solve(
        gram, moment, design.penalized * ridge_lambda, _prior_vector(design, prior)
    )
    column_weight = (design.design != 0.0).T.astype(np.float64) @ weights
    offense_start = 2 if design.has_home else 1
    defense_start = offense_start + len(design.offense_labels)
    return UnitFit(
        intercept=float(coefficients[0]),
        home_field=float(coefficients[1]) if design.has_home else 0.0,
        ridge_lambda=float(ridge_lambda),
        offense={
            label: float(coefficients[offense_start + index])
            for index, label in enumerate(design.offense_labels)
            if column_weight[offense_start + index] > 0.0
        },
        defense={
            label: float(coefficients[defense_start + index])
            for index, label in enumerate(design.defense_labels)
            if column_weight[defense_start + index] > 0.0
        },
    )


def predict_unit(fit: UnitFit, rows: pl.DataFrame, columns: UnitColumns) -> FloatArray:
    """Predict the response for each row from fitted effects.

    Args:
        fit: Effects from :func:`fit_unit_ridge`.
        rows: Offense-versus-defense rows with the offense, defense, and (optional) home columns
            named in ``columns``; a null or missing home flag is a neutral site.
        columns: Column names, as passed to :func:`fit_unit_ridge`.

    Returns:
        ``intercept + offense - defense + home_field * home_sign`` per row, or NaN where the fit
        has no effect for the row's offense or defense.

    """
    offense = np.asarray(
        rows.get_column(columns.offense)
        .cast(pl.String)
        .replace_strict(fit.offense, default=None, return_dtype=pl.Float64)
        .to_numpy(),
        dtype=np.float64,
    )
    defense = np.asarray(
        rows.get_column(columns.defense)
        .cast(pl.String)
        .replace_strict(fit.defense, default=None, return_dtype=pl.Float64)
        .to_numpy(),
        dtype=np.float64,
    )
    return fit.intercept + offense - defense + fit.home_field * _home_signs(rows, columns.home)


__all__ = [
    "CROSS_VALIDATION_FOLDS",
    "DEFAULT_RIDGE_LAMBDAS",
    "FloatArray",
    "UnitColumns",
    "UnitDesign",
    "UnitFit",
    "UnitPrior",
    "build_unit_design",
    "fit_unit_ridge",
    "predict_unit",
    "solve_unit_design",
]
