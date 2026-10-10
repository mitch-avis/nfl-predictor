"""Season quarterback identity from the nflverse players file and weekly rosters.

The players file gives each player's latest position only, so a quarterback later listed at
tight end (Taysom Hill) or wide receiver (Terrelle Pryor) would lose his quarterback seasons. The
rule, ported from ``nfl-sos-ratings`` (``data_loader.load_qb_identity_crosswalk`` there): a player
the players file or any regular-season week of the season's weekly roster lists at
``constants.QB_POSITION`` is a quarterback that season, whatever else he plays. Players are
matched by their GSIS id, never by name. nflverse publishes weekly rosters from 2002, so an
earlier season reads the players file alone.

The crosswalk keeps the column names of the ``nfl-sos-ratings`` code it was ported from, so the
quarterback stats ported after it read it unchanged: ``qb_id`` is the GSIS id of any player, not
only of quarterbacks, and ``snap_player_id`` his Pro Football Reference id, the key of the snap
counts.
"""

from __future__ import annotations

import polars as pl

from nfl_predictor import constants

QB_IDENTITY_SCHEMA = pl.Schema(
    {
        "qb_id": pl.String(),
        "snap_player_id": pl.String(),
        "qb_name": pl.String(),
        "qb_position": pl.String(),
    }
)
# The players file outranks the weekly roster wherever both have a value.
_PLAYERS_FILE_PRIORITY = 0
_WEEKLY_ROSTER_PRIORITY = 1


def _identity_rows(frame: pl.DataFrame, *, name_column: str, priority: int) -> pl.DataFrame:
    """Return one source's rows in the crosswalk columns with its priority, ids present only."""

    def text(column: str) -> pl.Expr:
        """Return ``column`` as text, or a null text column when the source lacks it."""
        if column in frame.columns:
            return pl.col(column).cast(pl.String)
        return pl.lit(None, dtype=pl.String)

    return frame.select(
        text("gsis_id").alias("qb_id"),
        text("pfr_id").alias("snap_player_id"),
        text(name_column).alias("qb_name"),
        text("position").alias("qb_position"),
        pl.lit(priority, dtype=pl.Int64).alias("_priority"),
    ).filter(pl.col("qb_id").is_not_null())


def qb_identity_crosswalk(players: pl.DataFrame, weekly_rosters: pl.DataFrame) -> pl.DataFrame:
    """Return each player's snap-count id, name and position for one season.

    ``players`` is the players file (``gsis_id``, ``display_name``, ``position``, ``pfr_id``) and
    ``weekly_rosters`` the season's regular-season weekly roster (``gsis_id``, ``full_name``,
    ``position``, ``pfr_id``), empty before 2002. ``qb_position`` is ``QB`` when either source
    lists the player there in the season. Otherwise it is, like ``snap_player_id`` and
    ``qb_name``, the first value present: the players file's, then the roster's in row order.

    Returns:
        One row per GSIS id (``QB_IDENTITY_SCHEMA``), sorted by ``qb_id``

    """
    rows = pl.concat(
        [
            _identity_rows(players, name_column="display_name", priority=_PLAYERS_FILE_PRIORITY),
            _identity_rows(
                weekly_rosters, name_column="full_name", priority=_WEEKLY_ROSTER_PRIORITY
            ),
        ]
    )
    listed_at_qb = (pl.col("qb_position") == constants.QB_POSITION).any()
    return (
        rows.sort("_priority", maintain_order=True)
        .group_by("qb_id", maintain_order=True)
        .agg(
            pl.col("snap_player_id").drop_nulls().first(),
            pl.col("qb_name").drop_nulls().first(),
            pl.when(listed_at_qb)
            .then(pl.lit(constants.QB_POSITION))
            .otherwise(pl.col("qb_position").drop_nulls().first())
            .alias("qb_position"),
        )
        .sort("qb_id")
        .cast(QB_IDENTITY_SCHEMA)
    )


def season_quarterback_ids(crosswalk: pl.DataFrame) -> list[str]:
    """Return the sorted GSIS ids of the players who are quarterbacks in the crosswalk's season."""
    return (
        crosswalk.filter(pl.col("qb_position") == constants.QB_POSITION)
        .get_column("qb_id")
        .sort()
        .to_list()
    )
