"""Season-table inputs rebuilt from the game logs ``nfl-sos-ratings`` published.

The season rows and opponent profiles pool each rate over games through its hidden numerator and
denominator (``pooled_rates``), and the published game logs leave those parts out. This adapter
turns published logs back into the rows the season tables read, rebuilding exactly the parts the
published columns determine, as the code that wrote them defines them:

- the per-snap rates (``team_stats.PER_SNAP_RATES``), whose numerator and denominator are both
  published columns;
- the offense ratios whose numerator and denominator are published columns
  (``team_stats_expanded.OFFENSE_RATIO_SPECS`` and ``GIVEAWAYS_PER_DRIVE_SPEC``);
- the defensive mirror of each of those (``team_stats_expanded.DEFENSE_MIRROR_RENAMES``), whose
  parts are the opponent's in the same game.

Every other rate keeps no parts: its numerator or denominator is a working column the logs drop
(the dropbacks' EPA behind EPA per dropback) or a count the logs do not carry (the passes with a
completion probability over expected). A season row built from these inputs averages such a rate
per game instead, so it is a check of that rate only where :func:`recovered_team_rates` lists it.
The quarterback rows rebuild every pooled rate from published totals (``qb_stats.QB_RATES``)
except the completion percentage over expected, whose parts the logs drop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor.season_stats.game_logs import (
    QB_GAME_ORDER,
    QB_LOG_CONTEXT_COLUMNS,
    TEAM_GAME_ORDER,
)
from nfl_predictor.season_stats.pooled_rates import denominator_column, numerator_column
from nfl_predictor.season_stats.team_stats import PER_SNAP_RATES
from nfl_predictor.season_stats.team_stats_expanded import (
    DEFENSE_MIRROR_PARTS_ONLY,
    DEFENSE_MIRROR_RENAMES,
    GIVEAWAYS_PER_DRIVE_SPEC,
    OFFENSE_RATIO_SPECS,
)

if TYPE_CHECKING:
    from collections.abc import Collection

_GAME_KEYS = ("game_id", "team", "opponent_team")


def _own_specs(columns: Collection[str]) -> list[tuple[str, str, str]]:
    """Return the (numerator, denominator, rate) specs whose three columns are all published."""
    available = set(columns)
    return [
        spec
        for spec in (*PER_SNAP_RATES, *OFFENSE_RATIO_SPECS, GIVEAWAYS_PER_DRIVE_SPEC)
        if set(spec) <= available
    ]


def _mirror_specs(columns: Collection[str]) -> list[tuple[str, str, str]]:
    """Return (numerator, denominator, mirrored rate) for each published mirror of an own spec.

    The numerator and denominator name the opponent's columns in the same game.
    """
    available = set(columns)
    own = {rate: (numerator, denominator) for numerator, denominator, rate in _own_specs(columns)}
    mirrors = {**DEFENSE_MIRROR_RENAMES, **DEFENSE_MIRROR_PARTS_ONLY}
    return [
        (*own[source], target)
        for source, target in mirrors.items()
        if source in own and target in available
    ]


def recovered_team_rates(columns: Collection[str]) -> tuple[str, ...]:
    """Return the rates whose hidden parts :func:`team_games_from_published` rebuilds."""
    return (
        *(rate for _, _, rate in _own_specs(columns)),
        *(rate for _, _, rate in _mirror_specs(columns)),
    )


def team_games_from_published(logs: pl.DataFrame) -> pl.DataFrame:
    """Return published team game logs as season-table input rows, recoverable parts rebuilt.

    Args:
        logs: One season's published team game logs (``{season}_team_game_logs``).

    Returns:
        ``logs`` with the hidden numerator and denominator of every rate of
        :func:`recovered_team_rates`, in the builders' row order (``game_logs.TEAM_GAME_ORDER``)

    """
    own = [
        expr
        for numerator, denominator, rate in _own_specs(logs.columns)
        for expr in (
            pl.col(numerator).alias(numerator_column(rate)),
            pl.col(denominator).alias(denominator_column(rate)),
        )
    ]
    mirrors = _mirror_specs(logs.columns)
    # Sorted as the builders sort, not only ordered so: Polars sums a group's floats in another
    # order when the frame is flagged sorted by the group key, as the builders' frames are.
    games = logs.with_columns(own).sort(TEAM_GAME_ORDER)
    if not mirrors:
        return games
    opponent_parts = logs.select(
        "game_id",
        pl.col("opponent_team").alias("team"),
        pl.col("team").alias("opponent_team"),
        *(
            expr
            for numerator, denominator, rate in mirrors
            for expr in (
                pl.col(numerator).alias(numerator_column(rate)),
                pl.col(denominator).alias(denominator_column(rate)),
            )
        ),
    )
    return games.join(opponent_parts, on=list(_GAME_KEYS), how="left", maintain_order="left").sort(
        TEAM_GAME_ORDER
    )


def qb_games_from_published(logs: pl.DataFrame) -> pl.DataFrame:
    """Return published QB game logs as the season tables read passer-game rows.

    The passer's team is ``team_abbr``, and the game's opponent, venue, and result, which the logs
    join from the team rows (``game_logs.QB_LOG_CONTEXT_COLUMNS``), are left out. The rows are in
    the builders' order (``game_logs.QB_GAME_ORDER``).
    """
    return (
        logs.drop(QB_LOG_CONTEXT_COLUMNS, strict=False)
        .rename({"team": "team_abbr"})
        .sort(QB_GAME_ORDER)
    )


__all__ = [
    "qb_games_from_published",
    "recovered_team_rates",
    "team_games_from_published",
]
