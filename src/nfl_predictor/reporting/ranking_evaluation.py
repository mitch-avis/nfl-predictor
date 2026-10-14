"""Score candidate power rankings on how well they predict next week's margins.

Each candidate rates every team before every week, in points
(:mod:`nfl_predictor.reporting.rating_sources`). For every regular-season game from week 2 on, a
candidate predicts the home margin as

    rating[home] - rating[away] + h            (h is zero at a neutral site)

where ``h`` is the league home term the candidate's own rating differences leave over, fit by
least squares on the scored games of earlier seasons only (:func:`home_terms_by_season`). The
score is the mean absolute error against the actual margin. Paired candidate-minus-reference
differences are bootstrapped over games and, as a check, over seasons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl

from nfl_predictor.reporting import power_rankings, rating_sources, run_comparison

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# The first week scored: week 1 has no game of its season for a rating to learn from.
FIRST_SCORED_WEEK = 2

# The blend's grid for the ported rating's weight: 0.0 to 1.0 in steps of 0.1.
BLEND_WEIGHTS: tuple[float, ...] = tuple(step / 10 for step in range(11))
# Mean absolute errors this close count as a tie when choosing the blend weight.
_TIE_TOLERANCE = 1e-12

# How many top teams the rank agreement compares, and the fewest teams a correlation needs.
TOP_TEAMS = 5
_MIN_CORRELATION_TEAMS = 2
_AGREEMENT_SCHEMA = pl.Schema(
    {
        "season": pl.Int64(),
        "week": pl.Int64(),
        "teams": pl.Int64(),
        "spearman": pl.Float64(),
        "top_differs": pl.Boolean(),
    }
)

# Scored-game windows: (label, first week, last week or None for no upper bound).
WINDOWS: tuple[tuple[str, int, int | None], ...] = (
    ("weeks 2-4", 2, 4),
    ("weeks 5-8", 5, 8),
    ("weeks 9-18", 9, 18),
    ("all weeks", FIRST_SCORED_WEEK, None),
)

_REGULAR_SEASON = "REG"


def scored_games(
    games: pl.DataFrame, *, seasons: Iterable[int], first_week: int = FIRST_SCORED_WEEK
) -> pl.DataFrame:
    """Return the played regular-season games of ``seasons`` from ``first_week`` on.

    Args:
        games: Game rows with ``game_id``, ``season``, ``week``, ``game_type``, ``away_abbr``,
            ``home_abbr``, ``away_score``, ``home_score`` and ``neutral`` (1 at a neutral site;
            a null flag is not neutral).
        seasons: The seasons to keep.
        first_week: The first week to keep.

    Returns:
        ``game_id``, ``season``, ``week``, ``away_abbr``, ``home_abbr``, ``neutral`` (0 or 1) and
        ``actual_margin`` (home minus away), sorted by season, week and game.

    """
    wanted = sorted(set(seasons))
    return (
        games.filter(
            (pl.col("game_type") == _REGULAR_SEASON)
            & pl.col("season").is_in(wanted)
            & (pl.col("week") >= first_week)
            & pl.col("away_score").is_not_null()
            & pl.col("home_score").is_not_null()
        )
        .select(
            pl.col("game_id").cast(pl.String),
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("away_abbr").cast(pl.String),
            pl.col("home_abbr").cast(pl.String),
            pl.col("neutral").cast(pl.Int64).fill_null(0),
            (pl.col("home_score").cast(pl.Float64) - pl.col("away_score").cast(pl.Float64)).alias(
                "actual_margin"
            ),
        )
        .sort("season", "week", "game_id")
    )


def composite_ratings_by_week(snapshots: pl.DataFrame) -> pl.DataFrame:
    """Return the adjusted-strength composite in points for every week of a snapshot file.

    Each week is expressed in points through its own SRS, as the composite ranking does
    (``power_rankings.composite_points_vs_average``), on every digit. A week whose points scale
    is unavailable has no rows.

    Args:
        snapshots: The ETL's strength snapshots (``season``, ``week``, ``team_abbr``,
            ``adj_strength_composite``, ``adj_srs``); week ``N`` is solved from the games
            before week ``N``.

    Returns:
        A rating frame, sorted by season, week and team.

    """
    parts: list[pl.DataFrame] = []
    ordered = snapshots.sort("season", "week", "team_abbr")
    for rows in ordered.partition_by(["season", "week"], maintain_order=True):
        frame = pd.DataFrame(
            {
                column: rows.get_column(column).cast(pl.Float64).to_numpy()
                for column in (power_rankings.COMPOSITE_COLUMN, "adj_srs")
            }
        )
        points = power_rankings.composite_points_vs_average(frame)
        week_rows = pl.DataFrame(
            {
                "season": rows.get_column("season").cast(pl.Int64),
                "week": rows.get_column("week").cast(pl.Int64),
                "team": rows.get_column("team_abbr").cast(pl.String),
                "rating": pl.Series(points, dtype=pl.Float64, nan_to_null=True),
            },
            schema=rating_sources.RATING_SCHEMA,
        )
        parts.append(week_rows.drop_nulls("rating"))
    if not parts:
        return pl.DataFrame(schema=rating_sources.RATING_SCHEMA)
    return pl.concat(parts).sort("season", "week", "team")


def rating_differences(games: pl.DataFrame, ratings: pl.DataFrame) -> pl.DataFrame:
    """Return ``games`` with ``difference``: the home team's rating minus the away team's.

    Args:
        games: :func:`scored_games`.
        ratings: A rating frame (``rating_sources.RATING_SCHEMA``).

    Returns:
        ``games`` in the same order, ``difference`` null where either team has no rating.

    """
    keys = ["season", "week"]
    home = ratings.select(
        *keys, pl.col("team").alias("home_abbr"), pl.col("rating").alias("_home_rating")
    )
    away = ratings.select(
        *keys, pl.col("team").alias("away_abbr"), pl.col("rating").alias("_away_rating")
    )
    return (
        games.join(home, on=[*keys, "home_abbr"], how="left", maintain_order="left")
        .join(away, on=[*keys, "away_abbr"], how="left", maintain_order="left")
        .with_columns((pl.col("_home_rating") - pl.col("_away_rating")).alias("difference"))
        .drop("_home_rating", "_away_rating")
    )


def fit_home_term(games: pl.DataFrame) -> float:
    """Return the least-squares home term: the mean residual of the non-neutral games.

    Args:
        games: Scored games with ``neutral``, ``actual_margin`` and ``difference``; games
            without a difference are left out.

    Raises:
        ValueError: If no non-neutral game has a difference.

    """
    rows = games.filter((pl.col("neutral") == 0) & pl.col("difference").is_not_null())
    if rows.is_empty():
        msg = "no non-neutral game with a rating difference to fit the home term on"
        raise ValueError(msg)
    residuals = (
        rows.get_column("actual_margin").to_numpy() - rows.get_column("difference").to_numpy()
    )
    return float(np.mean(residuals))


def home_terms_by_season(games: pl.DataFrame, seasons: Iterable[int]) -> dict[int, float]:
    """Return each season's home term, fit on the scored games of every earlier season.

    Raises:
        ValueError: If a season has no earlier non-neutral game with a difference.

    """
    return {season: fit_home_term(games.filter(pl.col("season") < season)) for season in seasons}


def absolute_errors(games: pl.DataFrame, home_terms: Mapping[int, float]) -> pl.DataFrame:
    """Return ``games`` with ``error``: ``|actual - difference - h * (1 - neutral)|``.

    Args:
        games: Scored games with ``season``, ``neutral``, ``actual_margin`` and ``difference``.
        home_terms: Each season's home term.

    Returns:
        ``games`` with ``error``, null where the difference is.

    Raises:
        ValueError: If a game's season has no home term.

    """
    seasons = set(games.get_column("season").unique().to_list())
    missing = sorted(seasons - set(home_terms))
    if missing:
        msg = f"no home term for seasons {missing}"
        raise ValueError(msg)
    home_term = pl.col("season").replace_strict(dict(home_terms), return_dtype=pl.Float64)
    predicted = pl.col("difference") + home_term * (1 - pl.col("neutral")).cast(pl.Float64)
    return games.with_columns((pl.col("actual_margin") - predicted).abs().alias("error"))


@dataclass(frozen=True, slots=True)
class BlendChoice:
    """The blend weight chosen for one season on the earlier seasons' scored games.

    Attributes:
        weight: The ported rating's weight ``w``.
        home_term: The home term fit with that weight on the same games.
        mae: Its mean absolute error on those games.
        games: How many games it was chosen on.

    """

    weight: float
    home_term: float
    mae: float
    games: int


def _blended(games: pl.DataFrame, weight: pl.Expr | float) -> pl.DataFrame:
    """Return ``games`` with ``difference``, the weighted ported and model differences."""
    return games.with_columns(
        (weight * pl.col("ported_difference") + (1.0 - weight) * pl.col("model_difference")).alias(
            "difference"
        )
    )


def choose_blend_weights(
    games: pl.DataFrame, seasons: Iterable[int], *, weights: Iterable[float] = BLEND_WEIGHTS
) -> dict[int, BlendChoice]:
    """Choose each season's blend weight on the scored games of every earlier season.

    For each weight, the blend's difference is ``w * ported + (1 - w) * model``; its home term
    is fit on the earlier games (:func:`fit_home_term`) and its mean absolute error measured on
    them with that home term. The lowest error wins; weights within ``1e-12`` of it tie, and a
    tie goes to the largest weight, the one nearest the ported rating alone.

    Args:
        games: Scored games with ``season``, ``neutral``, ``actual_margin``,
            ``ported_difference`` and ``model_difference``; games missing either difference are
            left out.
        seasons: The seasons to choose a weight for.
        weights: The grid.

    Returns:
        Each season's choice.

    Raises:
        ValueError: If a season has no earlier game with both differences.

    """
    grid = sorted(set(weights))
    choices: dict[int, BlendChoice] = {}
    for season in seasons:
        earlier = games.filter(
            (pl.col("season") < season)
            & pl.col("ported_difference").is_not_null()
            & pl.col("model_difference").is_not_null()
        )
        if earlier.is_empty():
            msg = f"no scored game before {season} with both differences to choose a weight on"
            raise ValueError(msg)
        tried: list[BlendChoice] = []
        for weight in grid:
            blended = _blended(earlier, weight)
            home_term = fit_home_term(blended)
            errors = absolute_errors(
                blended, dict.fromkeys(blended.get_column("season").unique().to_list(), home_term)
            )
            mae = float(np.mean(errors.get_column("error").to_numpy()))
            tried.append(BlendChoice(weight, home_term, mae, earlier.height))
        best = min(choice.mae for choice in tried)
        choices[season] = max(
            (choice for choice in tried if choice.mae <= best + _TIE_TOLERANCE),
            key=lambda choice: choice.weight,
        )
    return choices


def blend_differences(games: pl.DataFrame, choices: Mapping[int, BlendChoice]) -> pl.DataFrame:
    """Return ``games`` with ``difference``: each season's blend at its chosen weight.

    Raises:
        ValueError: If a game's season has no choice.

    """
    missing = sorted(set(games.get_column("season").unique().to_list()) - set(choices))
    if missing:
        msg = f"no blend weight for seasons {missing}"
        raise ValueError(msg)
    weight = pl.col("season").replace_strict(
        {season: choice.weight for season, choice in choices.items()}, return_dtype=pl.Float64
    )
    return _blended(games, weight)


def season_bootstrap(
    differences: np.ndarray, seasons: np.ndarray, resamples: int, seed: int
) -> tuple[float, float, float]:
    """Return the mean difference and its 95% season-cluster bootstrap interval.

    Each resample draws as many seasons as there are, with replacement
    (``numpy.random.default_rng(seed).integers`` in one call), and takes the mean over every
    game of the drawn seasons, so a season counts by its games as in the point estimate.

    Args:
        differences: One paired difference per game.
        seasons: Each game's season.
        resamples: How many resamples.
        seed: The generator's seed.

    Returns:
        ``(mean, 2.5th percentile, 97.5th percentile)``.

    """
    labels, inverse = np.unique(seasons, return_inverse=True)
    sums = np.bincount(inverse, weights=differences)
    counts = np.bincount(inverse).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, labels.size, size=(resamples, labels.size))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return (
        float(np.mean(differences)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """A candidate against the reference on one window's games.

    Attributes:
        games: Games in the window.
        candidate_mae: The candidate's mean absolute error.
        reference_mae: The reference's.
        game_interval: The mean candidate-minus-reference difference and its 95% game-bootstrap
            interval (``run_comparison.bootstrap_mean``).
        season_interval: The same with the season-cluster bootstrap (:func:`season_bootstrap`).

    """

    games: int
    candidate_mae: float
    reference_mae: float
    game_interval: tuple[float, float, float]
    season_interval: tuple[float, float, float]


def paired_comparison(
    games: pl.DataFrame, *, first_week: int, last_week: int | None, resamples: int, seed: int
) -> PairedComparison:
    """Compare ``candidate_error`` with ``reference_error`` over one window's games.

    Args:
        games: One row per scored game with ``season``, ``week``, ``candidate_error`` and
            ``reference_error``.
        first_week: The window's first week.
        last_week: Its last week, or ``None`` for no upper bound.
        resamples: Resamples for both bootstraps.
        seed: Their seed.

    Raises:
        ValueError: If the window has no game.

    """
    in_window = pl.col("week") >= first_week
    if last_week is not None:
        in_window &= pl.col("week") <= last_week
    rows = games.filter(in_window)
    if rows.is_empty():
        msg = f"no scored game in weeks {first_week}-{last_week}"
        raise ValueError(msg)
    candidate = rows.get_column("candidate_error").to_numpy()
    reference = rows.get_column("reference_error").to_numpy()
    differences = candidate - reference
    return PairedComparison(
        games=rows.height,
        candidate_mae=float(np.mean(candidate)),
        reference_mae=float(np.mean(reference)),
        game_interval=run_comparison.bootstrap_mean(differences, resamples, seed),
        season_interval=season_bootstrap(
            differences, rows.get_column("season").to_numpy(), resamples, seed
        ),
    )


def _spearman(first: pl.Series, second: pl.Series) -> float:
    """Return the Spearman rank correlation (average ranks for ties), NaN without spread."""
    first_ranks = first.rank("average").to_numpy().astype(np.float64)
    second_ranks = second.rank("average").to_numpy().astype(np.float64)
    if first_ranks.size < _MIN_CORRELATION_TEAMS or not (
        np.std(first_ranks) > 0.0 and np.std(second_ranks) > 0.0
    ):
        return float("nan")
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def _top_teams(ratings: pl.DataFrame, column: str, top: int) -> set[str]:
    """Return the ``top`` highest-rated teams, ties broken by team code."""
    ordered = ratings.sort([column, "team"], descending=[True, False])
    return set(ordered.get_column("team").head(top).to_list())


def rank_agreement(
    candidate: pl.DataFrame,
    reference: pl.DataFrame,
    weeks: Iterable[tuple[int, int]],
    *,
    top: int = TOP_TEAMS,
) -> pl.DataFrame:
    """Return, for each week, how closely a candidate's order follows the reference's.

    Only the teams both rate that week count.

    Returns:
        One row per week: ``season``, ``week``, ``teams``, ``spearman`` (rank correlation) and
        ``top_differs`` (whether the two top-``top`` sets differ).

    """
    keys = ["season", "week", "team"]
    rows: list[dict[str, object]] = []
    for season, week in weeks:
        joined = candidate.filter((pl.col("season") == season) & (pl.col("week") == week)).join(
            reference, on=keys, how="inner", suffix="_reference"
        )
        rows.append(
            {
                "season": season,
                "week": week,
                "teams": joined.height,
                "spearman": _spearman(joined["rating"], joined["rating_reference"]),
                "top_differs": _top_teams(joined, "rating", top)
                != _top_teams(joined, "rating_reference", top),
            }
        )
    return pl.DataFrame(rows, schema=_AGREEMENT_SCHEMA)


@dataclass(frozen=True, slots=True)
class CandidateVerdict:
    """A candidate's all-weeks result, as the decision rule reads it.

    Attributes:
        name: The candidate.
        mae: Its mean absolute error over every scored game.
        reference_mae: The reference's on the same games.
        game_high: The upper end of the candidate-minus-reference game-bootstrap interval.
        season_high: The upper end of the season-cluster interval.

    """

    name: str
    mae: float
    reference_mae: float
    game_high: float
    season_high: float

    @property
    def beats_reference(self) -> bool:
        """Return whether the error is lower with both intervals wholly below zero."""
        return self.mae < self.reference_mae and self.game_high < 0.0 and self.season_high < 0.0


@dataclass(frozen=True, slots=True)
class Decision:
    """The ranking the rule picks, and every candidate that beat the reference."""

    winner: str
    qualified: tuple[str, ...]


def decide(verdicts: Iterable[CandidateVerdict], *, reference: str) -> Decision:
    """Apply the decision rule: replace the reference only with a clear all-weeks win.

    A candidate replaces the reference only if its error is lower and both all-weeks intervals
    exclude zero in its favor; among several, the lowest error wins. Anything else is a tie,
    and a tie goes to the reference.
    """
    qualified = [verdict for verdict in verdicts if verdict.beats_reference]
    if not qualified:
        return Decision(winner=reference, qualified=())
    best = min(qualified, key=lambda verdict: verdict.mae)
    return Decision(winner=best.name, qualified=tuple(verdict.name for verdict in qualified))


__all__ = [
    "BLEND_WEIGHTS",
    "FIRST_SCORED_WEEK",
    "TOP_TEAMS",
    "WINDOWS",
    "BlendChoice",
    "CandidateVerdict",
    "Decision",
    "PairedComparison",
    "absolute_errors",
    "blend_differences",
    "choose_blend_weights",
    "composite_ratings_by_week",
    "decide",
    "fit_home_term",
    "home_terms_by_season",
    "paired_comparison",
    "rank_agreement",
    "rating_differences",
    "scored_games",
    "season_bootstrap",
]
