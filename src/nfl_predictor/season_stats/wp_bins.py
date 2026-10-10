"""Plays and EPA split into win-probability bins, the inputs to the garbage-time filter.

A play's bin is how far from decided the game was before the snap, in whole percentage points:
``floor(100 * min(wp, 1 - wp))``, so bin 0 means one side was already more than 99% to win and
bin 50 is a toss-up. A filter at X% keeps the plays in bins X and above, which is the rule "keep a
play when ``min(wp, 1 - wp) >= X / 100``"; the product is rounded to 9 decimals before the floor so
an exact percentage such as ``wp = 0.29`` lands in its own bin despite binary floating point.
Plays without a win probability have a null bin and are kept at every threshold, so a 0% filter
reproduces the published totals exactly.

The bins use the same play filters as the totals the ratings fit (``scrimmage_snap_expr`` and
``special_teams_play_expr`` on plays with both teams known, and a passer's dropbacks grouped by
id), so summed over every bin they equal ``offensive_snaps``, ``offensive_epa``, ``st_plays``,
``st_epa``, and each QB-game's ``qb_dropbacks``.
"""

import polars as pl

from nfl_predictor.season_stats.pbp_expressions import (
    scrimmage_snap_expr,
    special_teams_play_expr,
    value_expr,
)

WP_COLUMN = "wp"
SCRIMMAGE_UNIT = "scrimmage"
SPECIAL_TEAMS_UNIT = "special_teams"
# Decimal places the percentage keeps before the floor (see the module docstring).
_BIN_ROUNDING_DECIMALS = 9

TEAM_WP_BIN_SCHEMA = pl.Schema(
    {
        "game_id": pl.String(),
        "week": pl.Int64(),
        "team": pl.String(),
        "opponent_team": pl.String(),
        "wp_unit": pl.String(),
        "wp_bin": pl.Int64(),
        "wp_bin_plays": pl.Int64(),
        "wp_bin_epa": pl.Float64(),
    }
)
QB_WP_BIN_SCHEMA = pl.Schema(
    {
        "game_id": pl.String(),
        "week": pl.Int64(),
        "qb_id": pl.String(),
        "wp_bin": pl.Int64(),
        "qb_wp_bin_dropbacks": pl.Int64(),
        "qb_wp_bin_epa": pl.Float64(),
    }
)
_REQUIRED_PBP_COLUMNS = frozenset({"game_id", "week", "posteam", WP_COLUMN})


def wp_bin_expr() -> pl.Expr:
    """Return each play's win-probability bin, 0-50, or null when ``wp`` is missing or NaN."""
    wp = pl.col(WP_COLUMN).cast(pl.Float64).fill_nan(None)
    return (
        (pl.min_horizontal(wp, 1.0 - wp) * 100.0)
        .round(_BIN_ROUNDING_DECIMALS)
        .floor()
        .cast(pl.Int64)
        .alias("wp_bin")
    )


def _has_win_probability(pbp_df: pl.DataFrame, extra: set[str]) -> bool:
    """Return whether ``pbp_df`` has rows and every column binning needs."""
    return not pbp_df.is_empty() and _REQUIRED_PBP_COLUMNS | extra <= set(pbp_df.columns)


def _unit_bins(plays: pl.DataFrame, condition: pl.Expr, unit: str) -> pl.DataFrame:
    """Return one unit's plays and EPA per team-game and bin, for the plays ``condition`` flags."""
    return (
        plays.filter(condition)
        .group_by("game_id", "week", "posteam", "defteam", "wp_bin")
        .agg(
            pl.len().cast(pl.Int64).alias("wp_bin_plays"),
            value_expr(plays.columns, "epa", 0.0).cast(pl.Float64).sum().alias("wp_bin_epa"),
        )
        .with_columns(pl.lit(unit).alias("wp_unit"))
    )


def compute_team_wp_bins(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Return each team-game's scrimmage and special-teams plays and EPA per win-probability bin.

    Args:
        pbp_df: Regular-season play-by-play with normalized team abbreviations.

    Returns:
        One row per game, team (the possession team), unit, and bin with ``opponent_team``,
        ``wp_bin_plays``, and ``wp_bin_epa``; an empty frame with ``TEAM_WP_BIN_SCHEMA`` when the
        play-by-play has no rows or no ``wp`` column.

    """
    if not _has_win_probability(pbp_df, {"defteam"}):
        return pl.DataFrame(schema=TEAM_WP_BIN_SCHEMA)
    columns = pbp_df.columns
    plays = pbp_df.filter(
        pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null()
    ).with_columns(wp_bin_expr())
    # Two separate passes, not one labeled pass: a fake punt is both a scrimmage snap and a
    # special-teams play, and the totals count it in both.
    return (
        pl.concat(
            [
                _unit_bins(plays, scrimmage_snap_expr(columns), SCRIMMAGE_UNIT),
                _unit_bins(plays, special_teams_play_expr(columns), SPECIAL_TEAMS_UNIT),
            ]
        )
        .rename({"posteam": "team", "defteam": "opponent_team"})
        .select(TEAM_WP_BIN_SCHEMA.names())
        .cast(TEAM_WP_BIN_SCHEMA)
    )


def compute_qb_wp_bins(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Return each passer-game's dropbacks and play-level passing EPA per win-probability bin.

    Dropbacks are grouped the way ``qb_stats`` groups them: by passer id, falling back to the name
    only to keep unidentified passers apart, and rows without an id are left out because no
    published QB row can match them. EPA is the play-by-play ``qb_epa``.

    Args:
        pbp_df: Regular-season play-by-play with normalized team abbreviations.

    Returns:
        One row per game, passer, and bin with ``qb_wp_bin_dropbacks`` and ``qb_wp_bin_epa``; an
        empty frame with ``QB_WP_BIN_SCHEMA`` when the play-by-play has no rows or lacks ``wp`` or
        the passer columns.

    """
    if not _has_win_probability(pbp_df, {"qb_dropback", "passer_player_id", "passer_player_name"}):
        return pl.DataFrame(schema=QB_WP_BIN_SCHEMA)
    return (
        pbp_df.filter(
            pl.col("posteam").is_not_null()
            & pl.col("passer_player_name").is_not_null()
            & (pl.col("qb_dropback").fill_null(0) > 0)
        )
        .with_columns(
            wp_bin_expr(),
            pl.coalesce(pl.col("passer_player_id"), pl.col("passer_player_name")).alias(
                "_passer_key"
            ),
        )
        .group_by("game_id", "week", "posteam", "_passer_key", "wp_bin")
        .agg(
            pl.col("passer_player_id").drop_nulls().first().alias("qb_id"),
            pl.col("qb_dropback").sum().cast(pl.Int64).alias("qb_wp_bin_dropbacks"),
            value_expr(pbp_df.columns, "qb_epa", 0.0).cast(pl.Float64).sum().alias("qb_wp_bin_epa"),
        )
        .filter(pl.col("qb_id").is_not_null())
        .select(QB_WP_BIN_SCHEMA.names())
        .cast(QB_WP_BIN_SCHEMA)
    )


__all__ = [
    "QB_WP_BIN_SCHEMA",
    "SCRIMMAGE_UNIT",
    "SPECIAL_TEAMS_UNIT",
    "TEAM_WP_BIN_SCHEMA",
    "WP_COLUMN",
    "compute_qb_wp_bins",
    "compute_team_wp_bins",
    "wp_bin_expr",
]
