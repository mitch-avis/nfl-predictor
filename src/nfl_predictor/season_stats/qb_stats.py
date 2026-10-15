"""Quarterback season-level aggregation helpers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor.season_stats.pbp_expressions import (
    lost_fumble_team_expr,
    passer_rating_from_rates,
)
from nfl_predictor.season_stats.pooled_rates import (
    denominator_column,
    guarded_ratio,
    is_rate_part,
    mean_with_parts,
    numerator_column,
    pooled_rate,
    rate_parts,
)

if TYPE_CHECKING:
    from collections.abc import Callable

type PolarsCastType = type[pl.Int64 | pl.Float64]
type TotalOf = Callable[[str], pl.Expr]
"""Return the total of one game-row column over the games a row covers."""

CPOE_COLUMN = "qb_completion_percentage_above_expectation"

# Comebacks and game-winning drives count plays from the fourth quarter on (overtime included).
_FOURTH_QUARTER = 4

# A quarterback is ranked with 14 pass attempts per game his team has played (the usual NFL
# qualifier); a season without weekly team rows is taken as a full 17-game season.
QUALIFIER_ATTEMPTS_PER_TEAM_GAME = 14
FULL_SEASON_TEAM_GAMES = 17
QUALIFIER_COLUMN = "qb_attempt_qualifier"

_QB_TOTAL_COLUMNS: dict[str, tuple[str, PolarsCastType]] = {
    "qb_attempts": ("qb_attempts_total", pl.Int64),
    "qb_completions": ("qb_completions_total", pl.Int64),
    "qb_dropbacks": ("qb_dropbacks_total", pl.Int64),
    "qb_offense_snaps": ("qb_offense_snaps_total", pl.Int64),
    "qb_fourth_quarter_comeback": ("qb_fourth_quarter_comebacks", pl.Int64),
    "qb_game_winning_drive": ("qb_game_winning_drives", pl.Int64),
    "qb_pass_yards": ("qb_pass_yards_total", pl.Float64),
    "qb_pass_touchdowns": ("qb_pass_touchdowns_total", pl.Float64),
    "qb_interceptions": ("qb_interceptions_total", pl.Float64),
    "qb_sacks": ("qb_sacks_total", pl.Float64),
    "qb_sack_yards_lost": ("qb_sack_yards_lost_total", pl.Float64),
    "qb_sack_fumbles_lost": ("qb_sack_fumbles_lost_total", pl.Float64),
    "qb_passing_epa": ("qb_passing_epa_total", pl.Float64),
    "qb_spike_epa": ("qb_spike_epa_total", pl.Float64),
    "qb_carries": ("qb_carries_total", pl.Int64),
    "qb_rushing_yards": ("qb_rushing_yards_total", pl.Float64),
    "qb_rushing_tds": ("qb_rushing_tds_total", pl.Int64),
    "qb_rushing_first_downs": ("qb_rushing_first_downs_total", pl.Int64),
    "qb_rushing_epa": ("qb_rushing_epa_total", pl.Float64),
    "qb_rushing_fumbles": ("qb_rushing_fumbles_total", pl.Int64),
    "qb_rushing_fumbles_lost": ("qb_rushing_fumbles_lost_total", pl.Int64),
    "qb_rushing_2pt_conversions": ("qb_rushing_2pt_conversions_total", pl.Int64),
    "qb_designed_carries": ("qb_designed_carries_total", pl.Int64),
    "qb_designed_rush_yards": ("qb_designed_rush_yards_total", pl.Float64),
    "qb_designed_rush_epa": ("qb_designed_rush_epa_total", pl.Float64),
    "qb_scrambles": ("qb_scrambles_total", pl.Int64),
    "qb_scramble_yards": ("qb_scramble_yards_total", pl.Float64),
    "qb_kneels": ("qb_kneels_total", pl.Int64),
}

_QB_PER_GAME_COLUMNS: dict[str, str] = {
    "qb_attempts": "qb_attempts_per_game",
    "qb_completions": "qb_completions_per_game",
    "qb_dropbacks": "qb_dropbacks_per_game",
    "qb_offense_snaps": "qb_offense_snaps_per_game",
    "qb_fourth_quarter_comeback": "qb_fourth_quarter_comebacks_per_game",
    "qb_game_winning_drive": "qb_game_winning_drives_per_game",
    "qb_pass_yards": "qb_pass_yards_per_game",
    "qb_pass_touchdowns": "qb_pass_touchdowns_per_game",
    "qb_interceptions": "qb_interceptions_per_game",
    "qb_sacks": "qb_sacks_per_game",
    "qb_sack_yards_lost": "qb_sack_yards_lost_per_game",
    "qb_sack_fumbles_lost": "qb_sack_fumbles_lost_per_game",
    "qb_passing_epa": "qb_passing_epa_per_game",
    "qb_spike_epa": "qb_spike_epa_per_game",
    "qb_carries": "qb_carries_per_game",
    "qb_rushing_yards": "qb_rushing_yards_per_game",
    "qb_rushing_tds": "qb_rushing_tds_per_game",
    "qb_rushing_first_downs": "qb_rushing_first_downs_per_game",
    "qb_rushing_epa": "qb_rushing_epa_per_game",
    "qb_rushing_fumbles": "qb_rushing_fumbles_per_game",
    "qb_rushing_fumbles_lost": "qb_rushing_fumbles_lost_per_game",
    "qb_rushing_2pt_conversions": "qb_rushing_2pt_conversions_per_game",
    "qb_designed_carries": "qb_designed_carries_per_game",
    "qb_designed_rush_yards": "qb_designed_rush_yards_per_game",
    "qb_designed_rush_epa": "qb_designed_rush_epa_per_game",
    "qb_scrambles": "qb_scrambles_per_game",
    "qb_scramble_yards": "qb_scramble_yards_per_game",
    "qb_kneels": "qb_kneels_per_game",
}


def select_primary_qb_rows(qb_df: pl.DataFrame) -> pl.DataFrame:
    """Return one primary QB row per team-week: most snaps, then dropbacks, then attempts.

    The primary QB takes the game's result and late-game credit. Unknown (null) snaps, as before
    snap counts exist in 2013, sort last. Remaining ties (common before 2013) go to the lowest
    ``qb_id`` (or ``qb_name``), so the pick never depends on row order.
    """
    if not {"team_abbr", "week"}.issubset(set(qb_df.columns)):
        return qb_df

    sort_keys = [
        column
        for column in ("qb_offense_snaps", "qb_dropbacks", "qb_attempts")
        if column in qb_df.columns
    ]
    if not sort_keys:
        return qb_df

    tie_keys = [column for column in ("qb_id", "qb_name") if column in qb_df.columns][:1]
    return (
        qb_df.sort(
            [*sort_keys, *tie_keys],
            descending=[True] * len(sort_keys) + [False] * len(tie_keys),
            nulls_last=True,
            maintain_order=True,
        )
        .group_by(["team_abbr", "week"], maintain_order=True)
        .first()
    )


def _compute_team_late_game_flags_from_pbp(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Return team-game 4QC and GWD flags from late-game PBP score states.

    Fourth-quarter comeback = eventual winner with a quarter-4-or-later
    offensive snap where `score_differential < 0`.
    Game-winning drive = eventual winner with a quarter-4-or-later scoring play
    where `score_differential <= 0` before the play and
    `score_differential_post > 0` after the play.
    """
    required_cols = {
        "game_id",
        "posteam",
        "qtr",
        "score_differential",
        "score_differential_post",
        "posteam_score",
        "posteam_score_post",
    }
    if pbp_df.is_empty() or not required_cols.issubset(set(pbp_df.columns)):
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "team_abbr": pl.String,
                "qb_fourth_quarter_comeback": pl.Int64,
                "qb_game_winning_drive": pl.Int64,
            }
        )

    offense = (
        pbp_df.filter(pl.col("posteam").is_not_null() & (pl.col("posteam") != ""))
        .with_row_index("play_order")
        .sort("play_order")
    )

    team_scores = offense.group_by(["game_id", "posteam"]).agg(
        pl.col("posteam_score_post").drop_nulls().max().alias("final_points_for")
    )
    final_state = (
        team_scores.join(
            team_scores.rename(
                {
                    "posteam": "opponent_team_abbr",
                    "final_points_for": "final_points_allowed",
                }
            ),
            on="game_id",
            how="inner",
        )
        .filter(pl.col("posteam") != pl.col("opponent_team_abbr"))
        .rename({"posteam": "team_abbr"})
        .select(["game_id", "team_abbr", "final_points_for", "final_points_allowed"])
    )
    trailing_late = (
        offense.filter((pl.col("qtr") >= _FOURTH_QUARTER) & (pl.col("score_differential") < 0))
        .group_by(["game_id", "posteam"])
        .agg(pl.lit(1).alias("had_fourth_quarter_deficit"))
        .rename({"posteam": "team_abbr"})
    )
    lead_taking = (
        offense.filter(
            (pl.col("qtr") >= _FOURTH_QUARTER)
            & (pl.col("posteam_score_post") > pl.col("posteam_score"))
            & (pl.col("score_differential") <= 0)
            & (pl.col("score_differential_post") > 0)
        )
        .group_by(["game_id", "posteam"])
        .agg(pl.lit(1).alias("had_game_winning_drive"))
        .rename({"posteam": "team_abbr"})
    )

    return (
        final_state.join(trailing_late, on=["game_id", "team_abbr"], how="left")
        .join(lead_taking, on=["game_id", "team_abbr"], how="left")
        .with_columns(
            (pl.col("final_points_for") > pl.col("final_points_allowed")).alias("team_won"),
            pl.col("had_fourth_quarter_deficit").fill_null(0),
            pl.col("had_game_winning_drive").fill_null(0),
        )
        .with_columns(
            pl.when(pl.col("team_won"))
            .then(pl.col("had_fourth_quarter_deficit"))
            .otherwise(0)
            .cast(pl.Int64)
            .alias("qb_fourth_quarter_comeback"),
            pl.when(pl.col("team_won"))
            .then(pl.col("had_game_winning_drive"))
            .otherwise(0)
            .cast(pl.Int64)
            .alias("qb_game_winning_drive"),
        )
        .select(["game_id", "team_abbr", "qb_fourth_quarter_comeback", "qb_game_winning_drive"])
    )


def _canonicalize_qb_rows(
    qb_rows: pl.DataFrame,
    qb_identity_df: pl.DataFrame | None,
    *,
    join_key: str,
) -> pl.DataFrame:
    """Attach canonical QB identifiers and names from the GSIS/PFR crosswalk."""
    if qb_rows.is_empty():
        return qb_rows.with_columns(pl.lit(None, dtype=pl.String).alias("qb_position"))

    if "qb_id" not in qb_rows.columns:
        qb_rows = qb_rows.with_columns(pl.lit(None, dtype=pl.String).alias("qb_id"))
    if "snap_player_id" not in qb_rows.columns:
        qb_rows = qb_rows.with_columns(pl.lit(None, dtype=pl.String).alias("snap_player_id"))
    if "qb_name" not in qb_rows.columns:
        qb_rows = qb_rows.with_columns(pl.lit(None, dtype=pl.String).alias("qb_name"))

    if qb_identity_df is None or qb_identity_df.is_empty() or join_key not in qb_rows.columns:
        return qb_rows.with_columns(pl.lit(None, dtype=pl.String).alias("qb_position"))

    if join_key == "qb_id":
        lookup = (
            qb_identity_df.filter(pl.col("qb_id").is_not_null())
            .select(["qb_id", "snap_player_id", "qb_name", "qb_position"])
            .unique(subset=["qb_id"], keep="first")
            .rename(
                {
                    "snap_player_id": "identity_snap_player_id",
                    "qb_name": "identity_qb_name",
                    "qb_position": "identity_qb_position",
                }
            )
        )
        return (
            qb_rows.join(lookup, on="qb_id", how="left")
            .with_columns(
                pl.coalesce([pl.col("snap_player_id"), pl.col("identity_snap_player_id")]).alias(
                    "snap_player_id"
                ),
                pl.coalesce([pl.col("identity_qb_name"), pl.col("qb_name")]).alias("qb_name"),
                pl.col("identity_qb_position").alias("qb_position"),
            )
            .drop(["identity_snap_player_id", "identity_qb_name", "identity_qb_position"])
        )

    lookup = (
        qb_identity_df.filter(pl.col("snap_player_id").is_not_null())
        .select(["snap_player_id", "qb_id", "qb_name", "qb_position"])
        .unique(subset=["snap_player_id"], keep="first")
        .rename(
            {
                "qb_id": "identity_qb_id",
                "qb_name": "identity_qb_name",
                "qb_position": "identity_qb_position",
            }
        )
    )
    return (
        qb_rows.join(lookup, on="snap_player_id", how="left")
        .with_columns(
            pl.coalesce([pl.col("identity_qb_id"), pl.col("qb_id")]).alias("qb_id"),
            pl.coalesce([pl.col("identity_qb_name"), pl.col("qb_name")]).alias("qb_name"),
            pl.col("identity_qb_position").alias("qb_position"),
        )
        .drop(["identity_qb_id", "identity_qb_name", "identity_qb_position"])
    )


def compute_qb_game_volumes_from_pbp(
    pbp_df: pl.DataFrame,
    snap_counts_df: pl.DataFrame | None = None,
    qb_identity_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Combine PBP dropbacks and snap counts into one row per quarterback game.

    Without snap-count data (``snap_counts_df`` absent or empty, as in seasons before nflverse
    has it) every ``qb_offense_snaps`` is null: the snaps are unknown, not zero. With it, a passer
    who has no snap-count row gets 0.
    """
    schema = {
        "game_id": pl.String,
        "week": pl.Int64,
        "team_abbr": pl.String,
        "qb_name": pl.String,
        "qb_id": pl.String,
        "snap_player_id": pl.String,
        "qb_dropbacks": pl.Int64,
        "qb_offense_snaps": pl.Int64,
    }
    parts: list[pl.DataFrame] = []

    if not pbp_df.is_empty():
        dropbacks = (
            pbp_df.filter(
                pl.col("posteam").is_not_null()
                & pl.col("passer_player_name").is_not_null()
                & (pl.col("qb_dropback").fill_null(0) > 0)
            )
            # One group per passer even when the play-by-play tags his name two ways in a game.
            .with_columns(
                pl.coalesce([pl.col("passer_player_id"), pl.col("passer_player_name")]).alias(
                    "_passer_key"
                )
            )
            .group_by(["game_id", "week", "posteam", "_passer_key"])
            .agg(
                pl.col("passer_player_id").drop_nulls().first(),
                pl.col("passer_player_name").drop_nulls().sort().first(),
                pl.col("qb_dropback").sum().cast(pl.Int64).alias("qb_dropbacks"),
            )
            .drop("_passer_key")
            .rename(
                {
                    "posteam": "team_abbr",
                    "passer_player_id": "qb_id",
                    "passer_player_name": "qb_name",
                }
            )
            .with_columns(
                pl.lit(None, dtype=pl.String).alias("snap_player_id"),
                pl.lit(0).cast(pl.Int64).alias("qb_offense_snaps"),
            )
            .select(schema.keys())
        )
        dropbacks = _canonicalize_qb_rows(dropbacks, qb_identity_df, join_key="qb_id")
        if not dropbacks.is_empty():
            parts.append(dropbacks)

    if snap_counts_df is not None and not snap_counts_df.is_empty():
        snap_counts = (
            snap_counts_df.filter(
                (pl.col("position") == "QB") & (pl.col("offense_snaps").fill_null(0) > 0)
            )
            .group_by(["game_id", "week", "team", "player", "pfr_player_id"])
            .agg(
                pl.col("offense_snaps")
                .fill_null(0)
                .sum()
                .round(0)
                .cast(pl.Int64)
                .alias("qb_offense_snaps")
            )
            .rename(
                {
                    "team": "team_abbr",
                    "player": "qb_name",
                    "pfr_player_id": "snap_player_id",
                }
            )
            .with_columns(
                pl.lit(None, dtype=pl.String).alias("qb_id"),
                pl.lit(0).cast(pl.Int64).alias("qb_dropbacks"),
            )
            .select(schema.keys())
        )
        snap_counts = _canonicalize_qb_rows(snap_counts, qb_identity_df, join_key="snap_player_id")
        if not snap_counts.is_empty():
            parts.append(snap_counts)

    if not parts:
        return pl.DataFrame(schema=schema)

    combined = pl.concat(parts, how="diagonal_relaxed")
    if qb_identity_df is not None and not qb_identity_df.is_empty():
        combined = combined.with_columns(
            pl.coalesce([pl.col("qb_id"), pl.col("snap_player_id"), pl.col("qb_name")]).alias(
                "_qb_identity_key"
            )
        )
        group_keys = ["game_id", "week", "team_abbr", "_qb_identity_key"]
        agg_exprs: list[pl.Expr] = [pl.col("qb_name").drop_nulls().first().alias("qb_name")]
    else:
        group_keys = ["game_id", "week", "team_abbr", "qb_name"]
        agg_exprs = []

    agg_exprs.extend(
        [
            pl.col("qb_id").drop_nulls().first().alias("qb_id"),
            pl.col("snap_player_id").drop_nulls().first().alias("snap_player_id"),
            pl.col("qb_position").drop_nulls().first().alias("qb_position"),
            pl.col("qb_dropbacks").sum().cast(pl.Int64).alias("qb_dropbacks"),
            pl.col("qb_offense_snaps").sum().cast(pl.Int64).alias("qb_offense_snaps"),
        ]
    )

    result = combined.group_by(group_keys).agg(agg_exprs)
    if "_qb_identity_key" in result.columns:
        result = result.drop("_qb_identity_key")
    if snap_counts_df is None or snap_counts_df.is_empty():
        result = result.with_columns(pl.lit(None, dtype=pl.Int64).alias("qb_offense_snaps"))

    return (
        result.filter(pl.col("qb_position").is_null() | (pl.col("qb_position") == "QB"))
        .drop("qb_position")
        .sort(["team_abbr", "week", "game_id", "qb_name"])
    )


# CPOE and its hidden numerator and denominator (``pooled_rates``): the CPOE summed over the
# passer's plays that have one, and their count, so rows for several games pool it.
CPOE_PARTS = (CPOE_COLUMN, numerator_column(CPOE_COLUMN), denominator_column(CPOE_COLUMN))

# The columns and types `compute_qb_game_stats_from_pbp` returns, with or without plays.
_QB_GAME_STATS_SCHEMA: dict[str, type[pl.DataType]] = {
    "game_id": pl.String,
    "week": pl.Int64,
    "team_abbr": pl.String,
    "qb_name": pl.String,
    "qb_id": pl.String,
    "snap_player_id": pl.String,
    "qb_dropbacks": pl.Int64,
    "qb_offense_snaps": pl.Int64,
    "qb_attempts": pl.Int64,
    "qb_completions": pl.Int64,
    "qb_pass_yards": pl.Float64,
    "qb_pass_touchdowns": pl.Int64,
    "qb_interceptions": pl.Int64,
    "qb_sacks": pl.Int64,
    "qb_sack_yards_lost": pl.Float64,
    "qb_sack_fumbles_lost": pl.Int64,
    "qb_passing_epa": pl.Float64,
    "qb_spike_epa": pl.Float64,
    "qb_designed_carries": pl.Int64,
    "qb_designed_rush_yards": pl.Float64,
    "qb_designed_rush_epa": pl.Float64,
    "qb_scrambles": pl.Int64,
    "qb_scramble_yards": pl.Float64,
    "qb_kneels": pl.Int64,
    "qb_epa_per_dropback": pl.Float64,
    "qb_pass_yards_per_dropback": pl.Float64,
    "qb_td_int_margin_rate": pl.Float64,
    "qb_sack_rate": pl.Float64,
    "qb_any_a": pl.Float64,
    "qb_scramble_rate": pl.Float64,
    "qb_yards_per_scramble": pl.Float64,
    "qb_designed_yards_per_carry": pl.Float64,
    "qb_designed_epa_per_carry": pl.Float64,
    "qb_fourth_quarter_comeback": pl.Int64,
    "qb_game_winning_drive": pl.Int64,
    "qb_completion_percentage_above_expectation": pl.Float64,
    numerator_column(CPOE_COLUMN): pl.Float64,
    denominator_column(CPOE_COLUMN): pl.Int64,
}


def qb_epa_per_dropback_expr() -> pl.Expr:
    """Return a QB-game's EPA per dropback: passing EPA without spikes over dropbacks.

    A spike is a called incompletion to stop the clock, not a dropback, so its EPA, which
    nflverse's passing EPA counts, stays out of the rate, as it does out of the dropbacks.
    """
    return (
        pl.when(pl.col("qb_dropbacks") > 0)
        .then((pl.col("qb_passing_epa") - pl.col("qb_spike_epa")) / pl.col("qb_dropbacks"))
        .otherwise(None)
        .alias("qb_epa_per_dropback")
    )


def _qb_spike_stats(pbp_df: pl.DataFrame, qb_identity_df: pl.DataFrame | None) -> pl.DataFrame:
    """Return each passer's spike EPA per game (``qb_spike_epa``), keyed like the QB rows.

    A spike is a play nflverse flags ``qb_spike`` (or, without that column, types ``qb_spike``);
    its EPA is the passer's ``qb_epa``, as in nflverse's passing EPA.
    """
    schema = {
        "game_id": pl.String,
        "week": pl.Int64,
        "team_abbr": pl.String,
        "qb_id": pl.String,
        "qb_spike_epa": pl.Float64,
    }
    columns = set(pbp_df.columns)
    if "qb_spike" in columns:
        spike = pl.col("qb_spike").fill_null(0) > 0
    elif "play_type" in columns:
        spike = (pl.col("play_type") == "qb_spike").fill_null(value=False)
    else:
        return pl.DataFrame(schema=schema)
    spikes = (
        pbp_df.filter(
            pl.col("posteam").is_not_null() & pl.col("passer_player_name").is_not_null() & spike
        )
        .with_columns(
            pl.coalesce([pl.col("passer_player_id"), pl.col("passer_player_name")]).alias(
                "_passer_key"
            )
        )
        .group_by(["game_id", "week", "posteam", "_passer_key"])
        .agg(
            pl.col("passer_player_id").drop_nulls().first(),
            pl.col("passer_player_name").drop_nulls().sort().first(),
            pl.col("qb_epa").fill_null(0.0).sum().alias("qb_spike_epa"),
        )
        .drop("_passer_key")
        .rename(
            {"posteam": "team_abbr", "passer_player_id": "qb_id", "passer_player_name": "qb_name"}
        )
    )
    return _canonicalize_qb_rows(spikes, qb_identity_df, join_key="qb_id").select(list(schema))


def compute_qb_game_stats_from_pbp(
    pbp_df: pl.DataFrame,
    snap_counts_df: pl.DataFrame | None = None,
    qb_identity_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Derive per-game quarterback stats from PBP, supplemented with snap counts.

    Derived rate fields use dropbacks as the denominator.
    `qb_any_a` uses the standard formula:
    `(pass_yards + 20 * pass_tds - 45 * interceptions - sack_yards_lost) / (attempts + sacks)`.
    """
    volumes = compute_qb_game_volumes_from_pbp(pbp_df, snap_counts_df, qb_identity_df)
    if volumes.is_empty():
        return pl.DataFrame(schema=_QB_GAME_STATS_SCHEMA)

    pbp_columns = set(pbp_df.columns)
    sack_yards = (
        pl.col("yards_gained").fill_null(0.0) if "yards_gained" in pbp_columns else pl.lit(0.0)
    )
    cpoe = pl.col("cpoe") if "cpoe" in pbp_columns else pl.lit(None, dtype=pl.Float64)

    pbp_stats = (
        pbp_df.filter(
            pl.col("posteam").is_not_null()
            & pl.col("passer_player_name").is_not_null()
            & (pl.col("qb_dropback").fill_null(0) > 0)
        )
        # One group per passer: play-by-play sometimes tags the same passer two ways in one game
        # ("T.Pike" and "T.Pike (3rd QB)"), so the name is only the fallback key for a missing id.
        .with_columns(
            pl.coalesce([pl.col("passer_player_id"), pl.col("passer_player_name")]).alias(
                "_passer_key"
            )
        )
        .group_by(["game_id", "week", "posteam", "_passer_key"])
        .agg(
            [
                pl.col("passer_player_id").drop_nulls().first(),
                pl.col("passer_player_name").drop_nulls().sort().first(),
                pl.col("pass").fill_null(0).sum().cast(pl.Int64).alias("qb_attempts"),
                pl.col("complete_pass").fill_null(0).sum().cast(pl.Int64).alias("qb_completions"),
                pl.col("passing_yards").fill_null(0.0).sum().alias("qb_pass_yards"),
                pl.col("pass_touchdown")
                .fill_null(0)
                .sum()
                .cast(pl.Int64)
                .alias("qb_pass_touchdowns"),
                pl.col("interception").fill_null(0).sum().cast(pl.Int64).alias("qb_interceptions"),
                pl.col("sack").fill_null(0).sum().cast(pl.Int64).alias("qb_sacks"),
                pl.when(pl.col("sack").fill_null(0) > 0)
                .then((-sack_yards).clip(0.0, None))
                .otherwise(0.0)
                .sum()
                .alias("qb_sack_yards_lost"),
                # The quarterback's own strip-sacks: not a defender's fumble after recovering one.
                (
                    (pl.col("sack").fill_null(0) > 0)
                    & (lost_fumble_team_expr(list(pbp_columns)) == pl.col("posteam")).fill_null(
                        value=False
                    )
                )
                .sum()
                .cast(pl.Int64)
                .alias("qb_sack_fumbles_lost"),
                pl.col("qb_epa").fill_null(0.0).sum().alias("qb_passing_epa"),
                *mean_with_parts(CPOE_COLUMN, cpoe),
            ]
        )
        .drop("_passer_key")
        .rename(
            {
                "posteam": "team_abbr",
                "passer_player_id": "qb_id",
                "passer_player_name": "qb_name",
            }
        )
    )
    pbp_stats = _canonicalize_qb_rows(pbp_stats, qb_identity_df, join_key="qb_id")
    pbp_stats = pbp_stats.select(
        [
            "game_id",
            "week",
            "team_abbr",
            "qb_id",
            "qb_attempts",
            "qb_completions",
            "qb_pass_yards",
            "qb_pass_touchdowns",
            "qb_interceptions",
            "qb_sacks",
            "qb_sack_yards_lost",
            "qb_sack_fumbles_lost",
            "qb_passing_epa",
            *CPOE_PARTS,
        ]
    )

    rushing_stats = pl.DataFrame(
        schema={
            "game_id": pl.String,
            "week": pl.Int64,
            "team_abbr": pl.String,
            "qb_id": pl.String,
            "qb_name": pl.String,
            "snap_player_id": pl.String,
            "qb_designed_carries": pl.Int64,
            "qb_designed_rush_yards": pl.Float64,
            "qb_designed_rush_epa": pl.Float64,
            "qb_scrambles": pl.Int64,
            "qb_scramble_yards": pl.Float64,
            "qb_kneels": pl.Int64,
        }
    )
    if {
        "game_id",
        "week",
        "posteam",
        "rusher_player_id",
        "rush",
    }.issubset(pbp_columns):
        rush_flag = pl.col("rush").fill_null(0) > 0
        scramble_flag = (
            pl.col("qb_scramble").fill_null(0) > 0
            if "qb_scramble" in pbp_columns
            else pl.lit(False)
        )
        kneel_flag = (
            pl.col("qb_kneel").fill_null(0) > 0 if "qb_kneel" in pbp_columns else pl.lit(False)
        )
        two_point_flag = (
            pl.col("two_point_attempt").fill_null(0) > 0
            if "two_point_attempt" in pbp_columns
            else pl.lit(False)
        )
        designed_rush_flag = rush_flag & ~scramble_flag & ~kneel_flag & ~two_point_flag
        rush_yards = (
            pl.col("rushing_yards").fill_null(0.0)
            if "rushing_yards" in pbp_columns
            else (
                pl.col("yards_gained").fill_null(0.0)
                if "yards_gained" in pbp_columns
                else pl.lit(0.0)
            )
        )
        rush_epa = (
            pl.col("epa").fill_null(0.0)
            if "epa" in pbp_columns
            else (pl.col("qb_epa").fill_null(0.0) if "qb_epa" in pbp_columns else pl.lit(0.0))
        )
        rusher_name = pl.coalesce(
            [
                pl.col("rusher_player_name").cast(pl.String)
                if "rusher_player_name" in pbp_columns
                else pl.lit(None, dtype=pl.String),
                pl.col("passer_player_name").cast(pl.String)
                if "passer_player_name" in pbp_columns
                else pl.lit(None, dtype=pl.String),
            ]
        )
        # nflverse sets rush = 0 on scrambles and kneels (a scramble counts as a pass play), so
        # their own flags bring them in beside the designed runs.
        rushing_stats = (
            pbp_df.filter(
                pl.col("posteam").is_not_null()
                & pl.col("rusher_player_id").is_not_null()
                & (rush_flag | scramble_flag | kneel_flag)
            )
            .group_by(["game_id", "week", "posteam", "rusher_player_id"])
            .agg(
                rusher_name.drop_nulls().first().alias("qb_name"),
                designed_rush_flag.cast(pl.Int64).sum().alias("qb_designed_carries"),
                rush_yards.filter(designed_rush_flag)
                .sum()
                .fill_null(0.0)
                .alias("qb_designed_rush_yards"),
                rush_epa.filter(designed_rush_flag)
                .sum()
                .fill_null(0.0)
                .alias("qb_designed_rush_epa"),
                (scramble_flag & ~two_point_flag).cast(pl.Int64).sum().alias("qb_scrambles"),
                rush_yards.filter(scramble_flag & ~two_point_flag)
                .sum()
                .fill_null(0.0)
                .alias("qb_scramble_yards"),
                (kneel_flag & ~two_point_flag).cast(pl.Int64).sum().alias("qb_kneels"),
            )
            .rename(
                {
                    "posteam": "team_abbr",
                    "rusher_player_id": "qb_id",
                }
            )
        )
        rushing_stats = _canonicalize_qb_rows(rushing_stats, qb_identity_df, join_key="qb_id")
        rushing_stats = rushing_stats.select(
            [
                "game_id",
                "week",
                "team_abbr",
                "qb_id",
                "qb_designed_carries",
                "qb_designed_rush_yards",
                "qb_designed_rush_epa",
                "qb_scrambles",
                "qb_scramble_yards",
                "qb_kneels",
            ]
        )

    spike_stats = _qb_spike_stats(pbp_df, qb_identity_df)

    return (
        volumes.join(
            pbp_stats,
            on=["game_id", "week", "team_abbr", "qb_id"],
            how="left",
        )
        .join(
            rushing_stats,
            on=["game_id", "week", "team_abbr", "qb_id"],
            how="left",
        )
        .join(spike_stats, on=["game_id", "week", "team_abbr", "qb_id"], how="left")
        .with_columns(
            # Passing EPA counts spikes, as nflverse's official passing EPA does.
            (pl.col("qb_passing_epa").fill_null(0.0) + pl.col("qb_spike_epa").fill_null(0.0)).alias(
                "qb_passing_epa"
            ),
            pl.col("qb_spike_epa").fill_null(0.0),
        )
        .with_columns(
            pl.col("qb_attempts").fill_null(0).cast(pl.Int64),
            pl.col("qb_completions").fill_null(0).cast(pl.Int64),
            pl.col("qb_pass_yards").fill_null(0.0),
            pl.col("qb_pass_touchdowns").fill_null(0).cast(pl.Int64),
            pl.col("qb_interceptions").fill_null(0).cast(pl.Int64),
            pl.col("qb_sacks").fill_null(0).cast(pl.Int64),
            pl.col("qb_sack_yards_lost").fill_null(0.0),
            pl.col("qb_sack_fumbles_lost").fill_null(0).cast(pl.Int64),
            pl.col("qb_designed_carries").fill_null(0).cast(pl.Int64),
            pl.col("qb_designed_rush_yards").fill_null(0.0),
            pl.col("qb_designed_rush_epa").fill_null(0.0),
            pl.col("qb_scrambles").fill_null(0).cast(pl.Int64),
            pl.col("qb_scramble_yards").fill_null(0.0),
            pl.col("qb_kneels").fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            qb_epa_per_dropback_expr(),
            pl.when(pl.col("qb_dropbacks") > 0)
            .then(pl.col("qb_pass_yards") / pl.col("qb_dropbacks"))
            .otherwise(None)
            .alias("qb_pass_yards_per_dropback"),
            pl.when(pl.col("qb_dropbacks") > 0)
            .then(
                (pl.col("qb_pass_touchdowns") - pl.col("qb_interceptions")) / pl.col("qb_dropbacks")
            )
            .otherwise(None)
            .alias("qb_td_int_margin_rate"),
            pl.when(pl.col("qb_dropbacks") > 0)
            .then(pl.col("qb_sacks") / pl.col("qb_dropbacks"))
            .otherwise(None)
            .alias("qb_sack_rate"),
            pl.when((pl.col("qb_attempts") + pl.col("qb_sacks")) > 0)
            .then(
                (
                    pl.col("qb_pass_yards")
                    + (20.0 * pl.col("qb_pass_touchdowns"))
                    - (45.0 * pl.col("qb_interceptions"))
                    - pl.col("qb_sack_yards_lost")
                )
                / (pl.col("qb_attempts") + pl.col("qb_sacks"))
            )
            .otherwise(None)
            .alias("qb_any_a"),
            pl.when(pl.col("qb_dropbacks") > 0)
            .then(pl.col("qb_scrambles") / pl.col("qb_dropbacks"))
            .otherwise(None)
            .alias("qb_scramble_rate"),
            pl.when(pl.col("qb_scrambles") > 0)
            .then(pl.col("qb_scramble_yards") / pl.col("qb_scrambles"))
            .otherwise(None)
            .alias("qb_yards_per_scramble"),
            pl.when(pl.col("qb_designed_carries") > 0)
            .then(pl.col("qb_designed_rush_yards") / pl.col("qb_designed_carries"))
            .otherwise(None)
            .alias("qb_designed_yards_per_carry"),
            pl.when(pl.col("qb_designed_carries") > 0)
            .then(pl.col("qb_designed_rush_epa") / pl.col("qb_designed_carries"))
            .otherwise(None)
            .alias("qb_designed_epa_per_carry"),
        )
        .join(
            _compute_team_late_game_flags_from_pbp(pbp_df),
            on=["game_id", "team_abbr"],
            how="left",
        )
        .join(
            select_primary_qb_rows(volumes)
            .select(["game_id", "team_abbr", "qb_name"])
            .with_columns(pl.lit(1).alias("_is_primary_qb")),
            on=["game_id", "team_abbr", "qb_name"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("_is_primary_qb").fill_null(0) > 0)
            .then(pl.col("qb_fourth_quarter_comeback").fill_null(0))
            .otherwise(0)
            .cast(pl.Int64)
            .alias("qb_fourth_quarter_comeback"),
            pl.when(pl.col("_is_primary_qb").fill_null(0) > 0)
            .then(pl.col("qb_game_winning_drive").fill_null(0))
            .otherwise(0)
            .cast(pl.Int64)
            .alias("qb_game_winning_drive"),
        )
        .select(list(_QB_GAME_STATS_SCHEMA))
        .sort(["team_abbr", "week", "game_id", "qb_name"])
    )


# Floating-point noise in a passer rating sits far below this many decimals.
_RATING_NOISE_DECIMALS = 9


def qb_passer_rating_expr(
    completions: pl.Expr,
    attempts: pl.Expr,
    yards: pl.Expr,
    touchdowns: pl.Expr,
    interceptions: pl.Expr,
) -> pl.Expr:
    """Return passer rating over passing totals, to one decimal as the NFL publishes it.

    A tie rounds up (78.75 shows as 78.8). Rounding to nine decimals first removes floating-point
    noise, which otherwise puts an exact tie just below it (78.74999999999999) or not, depending on
    how many rows Polars evaluates together. Null without attempts.
    """
    rating = passer_rating_from_rates(
        completions / attempts,
        yards / attempts,
        touchdowns / attempts,
        interceptions / attempts,
    )
    shown = rating.round(_RATING_NOISE_DECIMALS).round(1, mode="half_away_from_zero")
    return pl.when(attempts > 0).then(shown).otherwise(None)


@dataclass(frozen=True, slots=True)
class QbRate:
    """A quarterback stat rebuilt from the totals of game-row columns over the games a row covers.

    ``build`` takes a ``TotalOf`` and returns the stat from the ``inputs``' totals.
    """

    inputs: tuple[str, ...]
    build: Callable[[TotalOf], pl.Expr]


def _ratio(numerator: str, denominator: str) -> QbRate:
    """Return the rate of one total over another, null when the denominator is not positive."""
    return QbRate(
        (numerator, denominator),
        lambda total: guarded_ratio(total(numerator), total(denominator)),
    )


# ANY/A credits 20 yards per passing touchdown and charges 45 per interception.
_ANY_A_TD_BONUS = 20.0
_ANY_A_INT_PENALTY = 45.0
# The passing totals passer rating is built from, in ``qb_passer_rating_expr``'s order.
QB_PASSING_TOTALS = (
    "qb_completions",
    "qb_attempts",
    "qb_pass_yards",
    "qb_pass_touchdowns",
    "qb_interceptions",
)


def _td_int_differential(total: TotalOf) -> pl.Expr:
    """Return touchdown passes minus interceptions."""
    return total("qb_pass_touchdowns") - total("qb_interceptions")


def _epa_per_dropback(total: TotalOf) -> pl.Expr:
    """Return passing EPA without spikes per dropback (pass attempts and sacks)."""
    return guarded_ratio(total("qb_passing_epa") - total("qb_spike_epa"), total("qb_dropbacks"))


def _td_int_margin_rate(total: TotalOf) -> pl.Expr:
    """Return touchdown passes minus interceptions per dropback."""
    return guarded_ratio(_td_int_differential(total), total("qb_dropbacks"))


def _any_a(total: TotalOf) -> pl.Expr:
    """Return adjusted net yards per attempt: yards with TD and INT adjustments, less sacks."""
    return guarded_ratio(
        total("qb_pass_yards")
        + (_ANY_A_TD_BONUS * total("qb_pass_touchdowns"))
        - (_ANY_A_INT_PENALTY * total("qb_interceptions"))
        - total("qb_sack_yards_lost"),
        total("qb_attempts") + total("qb_sacks"),
    )


def _passer_rating(total: TotalOf) -> pl.Expr:
    """Return passer rating over the passing totals."""
    return qb_passer_rating_expr(*(total(column) for column in QB_PASSING_TOTALS))


# Every quarterback rate rebuilt from totals for a row covering several games (a season row, or
# the passers a defense faced), so each is its season numerator over its season denominator. The
# order fixes the season row's column order for the ones a game row lacks.
QB_RATES: dict[str, QbRate] = {
    "qb_yards_per_attempt": _ratio("qb_pass_yards", "qb_attempts"),
    "qb_touchdown_rate": _ratio("qb_pass_touchdowns", "qb_attempts"),
    "qb_interception_rate": _ratio("qb_interceptions", "qb_attempts"),
    "qb_completion_pct": _ratio("qb_completions", "qb_attempts"),
    "qb_yards_per_carry": _ratio("qb_rushing_yards", "qb_carries"),
    "qb_epa_per_carry": _ratio("qb_rushing_epa", "qb_carries"),
    "qb_designed_yards_per_carry": _ratio("qb_designed_rush_yards", "qb_designed_carries"),
    "qb_designed_epa_per_carry": _ratio("qb_designed_rush_epa", "qb_designed_carries"),
    "qb_yards_per_scramble": _ratio("qb_scramble_yards", "qb_scrambles"),
    "qb_scramble_rate": _ratio("qb_scrambles", "qb_dropbacks"),
    "qb_epa_per_dropback": QbRate(
        ("qb_passing_epa", "qb_spike_epa", "qb_dropbacks"), _epa_per_dropback
    ),
    "qb_pass_yards_per_dropback": _ratio("qb_pass_yards", "qb_dropbacks"),
    "qb_td_int_margin_rate": QbRate(
        ("qb_pass_touchdowns", "qb_interceptions", "qb_dropbacks"), _td_int_margin_rate
    ),
    "qb_sack_rate": _ratio("qb_sacks", "qb_dropbacks"),
    "qb_any_a": QbRate(
        (
            "qb_pass_yards",
            "qb_pass_touchdowns",
            "qb_interceptions",
            "qb_sack_yards_lost",
            "qb_attempts",
            "qb_sacks",
        ),
        _any_a,
    ),
    "qb_passer_rating": QbRate(QB_PASSING_TOTALS, _passer_rating),
}


# Season-row totals built from other totals, after the rates in the season row's column order.
# Kept out of ``QB_RATES``, which the QB opponent profiles publish per game.
_QB_SEASON_TOTALS: dict[str, QbRate] = {
    "qb_td_int_differential": QbRate(
        ("qb_pass_touchdowns", "qb_interceptions"), _td_int_differential
    ),
}


def qb_rate_exprs(available: set[str], total: TotalOf, rates: tuple[str, ...]) -> list[pl.Expr]:
    """Return each of ``rates`` whose game-row inputs are all ``available``, in order.

    ``total`` returns the total of a game-row column over the games the row covers.
    """
    return [
        QB_RATES[rate].build(total).alias(rate)
        for rate in rates
        if available.issuperset(QB_RATES[rate].inputs)
    ]


def _qb_season_rate_exprs(columns: set[str]) -> list[pl.Expr]:
    """Return every season rate and derived total whose totals are present, in column order."""
    totals = {source: total for source, (total, _) in _QB_TOTAL_COLUMNS.items()}
    available = {source for source, total in totals.items() if total in columns}

    def season_total(column: str) -> pl.Expr:
        return pl.col(totals[column])

    stats = {**QB_RATES, **_QB_SEASON_TOTALS}
    return [
        stats[name].build(season_total).alias(name)
        for name in stats
        if available.issuperset(stats[name].inputs)
    ]


def _known_total(column: str) -> pl.Expr:
    """Return the column's sum, or null when every value is null: a total of unknowns is unknown.

    Quarterback snaps are null in seasons before snap counts exist, and a plain sum would make
    them 0.
    """
    return pl.when(pl.col(column).is_not_null().any()).then(pl.col(column).sum())


def _qb_season_agg_exprs(qb_df: pl.DataFrame) -> list[pl.Expr]:
    """Return games played, season totals, and every other numeric stat over the season.

    A stat with hidden rate parts (CPOE) is pooled over the season (``pooled_rates``); the rest
    start as per-game means, and ``_qb_season_rate_exprs`` then rebuilds the rates from totals.
    """
    pooled = set(rate_parts(qb_df.columns))
    qb_stat_cols = [
        col
        for col, dtype in zip(qb_df.columns, qb_df.dtypes, strict=True)
        if dtype.is_numeric()
        and col not in {"week", *set(_QB_PER_GAME_COLUMNS)}
        and not is_rate_part(col)
    ]
    agg_exprs: list[pl.Expr] = [pl.len().alias("qb_games_played")]
    agg_exprs.extend(
        pooled_rate(col) if col in pooled else pl.col(col).mean().alias(col) for col in qb_stat_cols
    )
    for source_col, (total_col, total_dtype) in _QB_TOTAL_COLUMNS.items():
        if source_col in qb_df.columns:
            agg_exprs.append(_known_total(source_col).cast(total_dtype).alias(total_col))
        elif total_col == "qb_attempts_total":
            agg_exprs.append(pl.lit(0).cast(pl.Int64).alias(total_col))
    return agg_exprs


def _qb_primary_team_map(qb_df: pl.DataFrame, qb_keys: list[str]) -> pl.DataFrame:
    """Return each QB's team for labels and the qualifier: the team he played the most games for.

    A tie (a mid-season trade with equal games on both teams) goes to the team he played for most
    recently, then to the alphabetically first team, so the pick never depends on row order.
    """
    last_week = pl.col("week").max() if "week" in qb_df.columns else pl.lit(0)
    return (
        qb_df.group_by([*qb_keys, "team_abbr"])
        .agg(pl.len().alias("_games"), last_week.alias("_last_week"))
        .sort(
            ["_games", "_last_week", "team_abbr"],
            descending=[True, True, False],
            nulls_last=True,
            maintain_order=True,
        )
        .group_by(qb_keys, maintain_order=True)
        .first()
        .select([*qb_keys, pl.col("team_abbr").alias("team")])
    )


def _with_qb_results(
    season_stats: pl.DataFrame,
    qb_df: pl.DataFrame,
    weekly_df: pl.DataFrame | None,
    qb_keys: list[str],
) -> pl.DataFrame:
    """Add primary-QB wins, losses, ties, and win percentage.

    The win percentage is null for a quarterback without a decision: one who was never the primary
    passer, or whose results are unknown.
    """
    required_weekly_cols = {"team", "week", "points_for", "points_allowed"}
    if weekly_df is None or not required_weekly_cols.issubset(set(weekly_df.columns)):
        return season_stats.with_columns(pl.lit(None, dtype=pl.Float64).alias("qb_win_pct"))

    decisions = pl.col("qb_wins") + pl.col("qb_losses") + pl.col("qb_ties")
    qb_results = (
        select_primary_qb_rows(qb_df)
        .join(
            weekly_df.select(["team", "week", "points_for", "points_allowed"]),
            left_on=["team_abbr", "week"],
            right_on=["team", "week"],
            how="left",
        )
        .with_columns(
            [
                (pl.col("points_for") > pl.col("points_allowed")).cast(pl.Int64).alias("qb_win"),
                (pl.col("points_for") < pl.col("points_allowed")).cast(pl.Int64).alias("qb_loss"),
                (pl.col("points_for") == pl.col("points_allowed")).cast(pl.Int64).alias("qb_tie"),
            ]
        )
        .group_by(qb_keys)
        .agg(
            [
                pl.col("qb_win").sum().alias("qb_wins"),
                pl.col("qb_loss").sum().alias("qb_losses"),
                pl.col("qb_tie").sum().alias("qb_ties"),
            ]
        )
        .with_columns(
            pl.when(decisions > 0)
            .then((pl.col("qb_wins") + 0.5 * pl.col("qb_ties")) / decisions)
            .otherwise(None)
            .alias("qb_win_pct")
        )
    )
    return season_stats.join(qb_results, on=qb_keys, how="left").with_columns(
        pl.col("qb_wins").fill_null(0).cast(pl.Int64),
        pl.col("qb_losses").fill_null(0).cast(pl.Int64),
        pl.col("qb_ties").fill_null(0).cast(pl.Int64),
    )


def compute_qb_season_stats(
    qb_df: pl.DataFrame,
    weekly_df: pl.DataFrame | None = None,
    min_games: int = 0,
    min_attempts: int | None = None,
) -> pl.DataFrame:
    """Aggregate per-quarterback season stats with volume and eligibility fields."""
    if qb_df.is_empty():
        return pl.DataFrame(
            schema={
                "qb_id": pl.String,
                "qb_name": pl.String,
                "team": pl.String,
                "qb_games_played": pl.Int64,
                "qb_attempts_total": pl.Int64,
                "qb_win_pct": pl.Float64,
                QUALIFIER_COLUMN: pl.Int64,
                "qb_is_eligible": pl.Boolean,
            }
        )

    if "qb_id" not in qb_df.columns:
        qb_df = qb_df.with_columns(pl.col("team_abbr").alias("qb_id"))
    if "qb_name" not in qb_df.columns:
        qb_df = qb_df.with_columns(pl.col("team_abbr").alias("qb_name"))

    qb_keys = ["qb_id", "qb_name"]

    season_stats = (
        qb_df.group_by(qb_keys)
        .agg(_qb_season_agg_exprs(qb_df))
        .join(_qb_primary_team_map(qb_df, qb_keys), on=qb_keys, how="left")
    )
    season_stats = _with_qb_results(season_stats, qb_df, weekly_df, qb_keys)

    per_game_exprs = [
        pl.when(pl.col("qb_games_played") > 0)
        .then(pl.col(total_col) / pl.col("qb_games_played"))
        .otherwise(None)
        .alias(per_game_col)
        for source_col, per_game_col in _QB_PER_GAME_COLUMNS.items()
        if (total_col := _QB_TOTAL_COLUMNS[source_col][0]) in season_stats.columns
    ]
    season_stats = season_stats.with_columns(per_game_exprs)

    rate_exprs = _qb_season_rate_exprs(set(season_stats.columns))
    if rate_exprs:
        season_stats = season_stats.with_columns(rate_exprs)
    season_stats = _with_attempt_qualifier(season_stats, weekly_df, min_attempts)

    return season_stats.with_columns(
        (
            (pl.col("qb_games_played") >= min_games)
            & (pl.col("qb_attempts_total") >= pl.col(QUALIFIER_COLUMN))
        ).alias("qb_is_eligible")
    ).sort("team")


def _with_attempt_qualifier(
    season_stats: pl.DataFrame, weekly_df: pl.DataFrame | None, min_attempts: int | None
) -> pl.DataFrame:
    """Add ``qb_attempt_qualifier``, the pass attempts a quarterback needs to be ranked.

    The rule is 14 attempts per game the quarterback's team (his primary team, ``team``) has played,
    so in a season in progress each team's own game count applies, byes included. ``min_attempts``
    replaces it with one number for every quarterback. Without weekly team rows, a full season of
    17 games is assumed; a team missing from them gets the most games any team has played.
    """
    if min_attempts is not None:
        return season_stats.with_columns(
            pl.lit(min_attempts, dtype=pl.Int64).alias(QUALIFIER_COLUMN)
        )
    if weekly_df is None or weekly_df.is_empty() or "team" not in weekly_df.columns:
        full_season = FULL_SEASON_TEAM_GAMES * QUALIFIER_ATTEMPTS_PER_TEAM_GAME
        return season_stats.with_columns(
            pl.lit(full_season, dtype=pl.Int64).alias(QUALIFIER_COLUMN)
        )

    team_games = weekly_df.group_by("team").len("_team_games")
    most_games = team_games.get_column("_team_games").max()
    return (
        season_stats.join(team_games, on="team", how="left")
        .with_columns(
            (pl.col("_team_games").fill_null(most_games) * QUALIFIER_ATTEMPTS_PER_TEAM_GAME)
            .cast(pl.Int64)
            .alias(QUALIFIER_COLUMN)
        )
        .drop("_team_games")
    )
