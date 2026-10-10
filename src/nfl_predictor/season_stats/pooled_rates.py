"""Rates pooled over games: summed numerators over summed denominators, never a mean of rates.

Every rate computed for one game (completion percentage, EPA per snap, points per drive, the mean
yards to go on third down) carries its numerator and denominator beside it as hidden columns,
``_num_<rate>`` and ``_den_<rate>``. A row for several games (a team's season, or an opponent's
games without the head-to-head ones) sums both over the games where both are known and divides,
so a 40-attempt game weighs four times a 10-attempt game, as the season's own totals do. A mean
over events (yards per drive, expected yards after catch) is a rate too: the sum of its values
over their count.

The hidden columns never reach a data file: writers drop them with ``drop_rate_parts``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable

NUMERATOR_PREFIX = "_num_"
DENOMINATOR_PREFIX = "_den_"


def numerator_column(rate: str) -> str:
    """Return the name of the hidden column that holds ``rate``'s numerator."""
    return f"{NUMERATOR_PREFIX}{rate}"


def denominator_column(rate: str) -> str:
    """Return the name of the hidden column that holds ``rate``'s denominator."""
    return f"{DENOMINATOR_PREFIX}{rate}"


def is_rate_part(column: str) -> bool:
    """Return whether ``column`` is a hidden numerator or denominator."""
    return column.startswith((NUMERATOR_PREFIX, DENOMINATOR_PREFIX))


def part_rate(column: str) -> str:
    """Return the rate a hidden numerator or denominator column belongs to."""
    return column.removeprefix(NUMERATOR_PREFIX).removeprefix(DENOMINATOR_PREFIX)


def guarded_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """Return ``numerator / denominator``, or null unless the denominator is positive."""
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)


def ratio_with_parts(rate: str, numerator: str, denominator: str) -> list[pl.Expr]:
    """Return ``rate`` as one column over another in a row of game totals, with its two parts."""
    return expression_ratio_with_parts(rate, pl.col(numerator), pl.col(denominator))


def expression_ratio_with_parts(
    rate: str, numerator: pl.Expr, denominator: pl.Expr
) -> list[pl.Expr]:
    """Return ``rate`` as one expression over another in a row of game totals, with its parts.

    For a ratio of combined totals, such as yards less sack yards over attempts plus sacks.
    """
    return [
        guarded_ratio(numerator, denominator).alias(rate),
        numerator.alias(numerator_column(rate)),
        denominator.alias(denominator_column(rate)),
    ]


def numerator_for_value(frame: pl.DataFrame, rate: str) -> pl.DataFrame:
    """Return ``frame`` with ``rate``'s numerator set to its value times its denominator.

    For a game rate replaced by an official value (CPOE): the denominator still counts the plays
    behind it, and pooling then weighs the official value by them. A frame without the rate's
    parts is returned unchanged.
    """
    numerator = numerator_column(rate)
    denominator = denominator_column(rate)
    if not {rate, numerator, denominator}.issubset(frame.columns):
        return frame
    return frame.with_columns((pl.col(rate) * pl.col(denominator)).alias(numerator))


def summed_ratio_with_parts(rate: str, numerator: pl.Expr, denominator: pl.Expr) -> list[pl.Expr]:
    """Return ``rate`` in a group-by aggregation: the summed numerator over the summed denominator.

    ``numerator`` and ``denominator`` are per-play expressions, such as a completion flag and a
    pass-attempt flag.
    """
    total_numerator = numerator.sum()
    total_denominator = denominator.sum()
    return [
        guarded_ratio(total_numerator, total_denominator).alias(rate),
        total_numerator.alias(numerator_column(rate)),
        total_denominator.alias(denominator_column(rate)),
    ]


def mean_with_parts(rate: str, values: pl.Expr) -> list[pl.Expr]:
    """Return a mean over events in a group-by aggregation: its known values' sum over their count.

    Equal to ``values.mean()``; the parts let several games pool it.
    """
    return summed_ratio_with_parts(rate, values, values.is_not_null().cast(pl.Int64))


def pooled_rate(rate: str) -> pl.Expr:
    """Return ``rate`` pooled over the rows of a group-by aggregation of game rows.

    The summed numerator over the summed denominator, both over the games where both are known;
    null when no such game has a positive denominator.
    """
    numerator = pl.col(numerator_column(rate))
    denominator = pl.col(denominator_column(rate))
    known = numerator.is_not_null() & denominator.is_not_null()
    return guarded_ratio(numerator.filter(known).sum(), denominator.filter(known).sum()).alias(rate)


def rate_parts(columns: Iterable[str]) -> list[str]:
    """Return the rates with both a numerator and a denominator column among ``columns``."""
    names = list(columns)
    present = set(names)
    return [
        name.removeprefix(NUMERATOR_PREFIX)
        for name in names
        if name.startswith(NUMERATOR_PREFIX)
        and denominator_column(name.removeprefix(NUMERATOR_PREFIX)) in present
    ]


def drop_rate_parts(frame: pl.DataFrame) -> pl.DataFrame:
    """Return ``frame`` without its hidden numerator and denominator columns."""
    return frame.drop([column for column in frame.columns if is_rate_part(column)])
