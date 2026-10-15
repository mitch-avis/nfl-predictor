"""Rank ranges: how much a team's or quarterback's rating and rank move under game resampling.

A season's games are resampled with replacement and the published fit is rerun on each resample
with the season's fixed ridge penalties
(:func:`nfl_predictor.ratings.team_rating.bootstrap_team_ratings`,
:func:`nfl_predictor.ratings.qb_rating.bootstrap_qb_ratings`). Every resample is ranked, and this
module summarizes the draws per team or quarterback: rating and rank quantiles, the probability of
each rank, and the chance of a top-5 or top-10 finish. It also summarizes every ordered pair from
the same draws: how often one is rated above the other and the range of their rating difference,
which two overlapping rank ranges cannot show because both units move together in each resample.

The ranges describe game-to-game sampling noise in the shrunken estimate. They say nothing about
whether the model itself is right, and a season in progress shows very wide ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Collection

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
RANGE_QUANTILES = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)
TOP_RANKS = (5, 10)
PAIR_QUANTILES = (0.025, 0.50, 0.975)


@dataclass(frozen=True, slots=True)
class RangeColumns:
    """Column names for one entity's rank ranges."""

    id: str
    rating: str
    rank: str


TEAM_RANGE_COLUMNS = RangeColumns(id="team", rating="team_rating", rank="team_rank")
QB_RANGE_COLUMNS = RangeColumns(id="qb_id", rating="adj_qb_epa_per_dropback", rank="qb_rank")
UNIT_RANGE_COLUMNS: tuple[RangeColumns, ...] = tuple(
    RangeColumns(id="team", rating=f"{unit}_rating", rank=f"{unit}_rank")
    for unit in ("offense", "defense", "special_teams")
)


@dataclass(frozen=True, slots=True)
class PairColumns:
    """Column names for one entity's head-to-head pairs."""

    id: str
    other: str
    rating: str
    above: str
    gap: str
    share: str


TEAM_PAIR_COLUMNS = PairColumns(
    id="team",
    other="other_team",
    rating="team_rating",
    above="team_rated_above_probability",
    gap="team_rating_gap",
    share="team_pair_share",
)
QB_PAIR_COLUMNS = PairColumns(
    id="qb_id",
    other="other_qb_id",
    rating="adj_qb_epa_per_dropback",
    above="qb_rated_above_probability",
    gap="qb_rating_gap",
    share="qb_pair_share",
)


def quantile_suffix(level: float) -> str:
    """Return the column suffix for a quantile level, ``_q025`` for 0.025 through ``_q975``."""
    return f"_q{round(level * 1000):03d}"


def summarize_rank_ranges(
    draws: pl.DataFrame,
    published: pl.DataFrame,
    columns: RangeColumns,
    *,
    eligible: Collection[str] | None = None,
) -> pl.DataFrame:
    """Summarize bootstrap draws into rating and rank ranges, one row per ranked unit.

    Args:
        draws: Long rows with ``draw`` and the ``columns`` id and rating (higher is better).
        published: The published id and rating from the full fit.
        columns: The id column, the rating to rank on, and the name for the rank columns.
        eligible: Units to rank (for example published-eligible quarterbacks); every published
            unit when ``None``. Ranks in each draw are among the eligible units present in it.

    Returns:
        Per unit, ordered by published rank: the published rank (``rank_column``), rating and rank
        quantiles (``{rating_column}_q025`` ... and ``{rank_column}_q025`` ...), the share of
        draws without the unit, P(top 5) and P(top 10), and ``{rank_column}_probabilities``, whose
        k-th entry is P(rank = k + 1). Probabilities count every draw, so a unit missing from some
        draws has probabilities summing to less than one.

    """
    id_column, rating_column, rank_column = columns.id, columns.rating, columns.rank
    units = published.filter(
        pl.lit(value=True) if eligible is None else pl.col(id_column).is_in(list(eligible))
    ).with_columns(
        pl.col(rating_column).rank(method="min", descending=True).cast(pl.Int64).alias(rank_column)
    )
    unit_count = units.height
    draw_count = draws.get_column("draw").n_unique()
    ranked = draws.filter(
        pl.col(id_column).is_in(units.get_column(id_column).to_list())
    ).with_columns(
        pl.col(rating_column)
        .rank(method="min", descending=True)
        .over("draw")
        .cast(pl.Int64)
        .alias("_rank")
    )
    by_unit = {str(key[0]): frame for key, frame in ranked.group_by(id_column, maintain_order=True)}
    rows: list[dict[str, object]] = []
    for unit in units.sort(rank_column).iter_rows(named=True):
        frame = by_unit.get(str(unit[id_column]))
        ratings = np.empty(0) if frame is None else frame.get_column(rating_column).to_numpy()
        ranks = (
            np.empty(0, dtype=np.int64) if frame is None else frame.get_column("_rank").to_numpy()
        )
        row: dict[str, object] = {id_column: unit[id_column], rank_column: unit[rank_column]}
        for level in RANGE_QUANTILES:
            suffix = quantile_suffix(level)
            row[f"{rating_column}{suffix}"] = (
                float(np.quantile(ratings, level)) if ratings.size else None
            )
            row[f"{rank_column}{suffix}"] = (
                int(np.quantile(ranks, level, method="inverted_cdf")) if ranks.size else None
            )
        row[f"{rank_column}_missing_share"] = 1.0 - ranks.size / draw_count
        for top in TOP_RANKS:
            row[f"{rank_column}_top{top}_probability"] = float((ranks <= top).sum() / draw_count)
        row[f"{rank_column}_probabilities"] = (
            np.bincount(ranks, minlength=unit_count + 1)[1:] / draw_count
        ).tolist()
        rows.append(row)
    return pl.DataFrame(rows)


def summarize_unit_rank_ranges(draws: pl.DataFrame, published: pl.DataFrame) -> pl.DataFrame:
    """Summarize the offense, defense, and special-teams ratings' rank ranges, one row per team.

    Per unit: the published rank and the rating and rank quantiles, as
    :func:`summarize_rank_ranges` gives them for ``team_rating``; the rank chances stay with the
    team rating. Percentiles of the units do not add up to the team rating's: the median of a sum
    is not the sum of the medians.
    """
    suffixes = [quantile_suffix(level) for level in RANGE_QUANTILES]
    summary = published.select("team")
    for columns in UNIT_RANGE_COLUMNS:
        unit = summarize_rank_ranges(draws, published, columns).select(
            "team",
            columns.rank,
            *(f"{columns.rating}{suffix}" for suffix in suffixes),
            *(f"{columns.rank}{suffix}" for suffix in suffixes),
        )
        summary = summary.join(unit, on="team", how="left", maintain_order="left")
    return summary


def _draw_matrix(
    draws: pl.DataFrame, id_column: str, rating_column: str, units: list[str]
) -> npt.NDArray[np.float64]:
    """Return a draws-by-units matrix of ratings, NaN where a unit is absent from a draw."""
    draw_ids = np.unique(draws.get_column("draw").to_numpy())
    matrix = np.full((draw_ids.size, len(units)), np.nan)
    present = draws.filter(pl.col(id_column).is_in(units))
    column_of = {unit: index for index, unit in enumerate(units)}
    rows = np.searchsorted(draw_ids, present.get_column("draw").to_numpy())
    columns = np.array(
        [column_of[str(unit)] for unit in present.get_column(id_column).to_list()], dtype=np.int64
    )
    matrix[rows, columns] = present.get_column(rating_column).cast(pl.Float64).to_numpy()
    return matrix


def summarize_rank_pairs(
    draws: pl.DataFrame,
    published: pl.DataFrame,
    columns: PairColumns,
    *,
    eligible: Collection[str] | None = None,
) -> pl.DataFrame:
    """Summarize bootstrap draws into head-to-head chances, one row per ordered pair of units.

    Args:
        draws: Long rows with ``draw`` and the ``columns`` id and rating (higher is better).
        published: The published units (their id column); ratings are not read.
        columns: The id, other-unit, and rating columns and the names of the output columns.
        eligible: Units to pair (for example published-eligible quarterbacks); every published
            unit when ``None``.

    Returns:
        Rows ordered by unit, then other unit: ``above``, the share of draws with both units in
        which the unit is rated above the other (a tie counts half, so the two chances of a pair
        add up to one); ``gap`` quantiles at ``PAIR_QUANTILES`` of the unit's rating minus the
        other's; and ``share``, the share of all draws holding both units. A pair that never
        shares a draw has null chances and quantiles.

    """
    id_column = columns.id
    units = sorted(
        {
            str(unit)
            for unit in published.get_column(id_column).to_list()
            if eligible is None or unit in eligible
        }
    )
    matrix = _draw_matrix(draws, id_column, columns.rating, units)
    gap_columns = [f"{columns.gap}{quantile_suffix(level)}" for level in PAIR_QUANTILES]
    rows: list[dict[str, object]] = []
    for first, unit in enumerate(units):
        for second, other in enumerate(units):
            if first == second:
                continue
            both = ~np.isnan(matrix[:, first]) & ~np.isnan(matrix[:, second])
            gaps = matrix[both, first] - matrix[both, second]
            row: dict[str, object] = {id_column: unit, columns.other: other}
            if gaps.size:
                row[columns.above] = float(((gaps > 0).sum() + 0.5 * (gaps == 0).sum()) / gaps.size)
                quantiles = np.quantile(gaps, PAIR_QUANTILES).tolist()
            else:
                row[columns.above] = None
                quantiles = [None] * len(PAIR_QUANTILES)
            row.update(zip(gap_columns, quantiles, strict=True))
            row[columns.share] = float(gaps.size / matrix.shape[0]) if matrix.shape[0] else 0.0
            rows.append(row)
    schema: dict[str, type[pl.DataType]] = {
        id_column: pl.String,
        columns.other: pl.String,
        columns.above: pl.Float64,
        **dict.fromkeys(gap_columns, pl.Float64),
        columns.share: pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "PAIR_QUANTILES",
    "QB_PAIR_COLUMNS",
    "QB_RANGE_COLUMNS",
    "RANGE_QUANTILES",
    "TEAM_PAIR_COLUMNS",
    "TEAM_RANGE_COLUMNS",
    "TOP_RANKS",
    "UNIT_RANGE_COLUMNS",
    "PairColumns",
    "RangeColumns",
    "quantile_suffix",
    "summarize_rank_pairs",
    "summarize_rank_ranges",
    "summarize_unit_rank_ranges",
]
