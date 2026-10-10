"""Team and quarterback game logs from play-by-play, official weekly stats, and schedules.

The builders of ``nfl-sos-ratings``' game logs (``data_loader.py`` and ``main.py`` there), ported
as functions of frames: the season's inputs come from :mod:`nfl_predictor.season_stats.sources`.

- :func:`build_weekly_team_stats` builds one row per team-game from play-by-play
  (``team_stats.compute_team_game_stats_from_pbp``), then replaces the passing, rushing,
  first-down, sack, interception and lost-fumble totals with nflverse's official weekly team stats
  (:func:`override_team_game_stats_with_official_weekly`). A game with official rows for only one
  team (both teams' totals credited to it, as in Jacksonville's 2001-2002 home games) keeps its
  play-by-play values (:func:`two_team_games_only`).
- :func:`build_qb_game_stats` builds one row per passer-game from play-by-play and snap counts
  (``qb_stats.compute_qb_game_stats_from_pbp``), then replaces the attempt-based passing fields
  and adds the rushing fields from the official weekly player stats.
- :func:`build_team_game_logs` and :func:`build_qb_game_logs` turn those rows into the published
  game logs: identity columns first, the rates' hidden parts dropped.

The frames these functions return keep the hidden numerator and denominator of every rate
(``pooled_rates``), which the season rows and opponent profiles pool over games.
"""

from __future__ import annotations

import polars as pl

from nfl_predictor import constants
from nfl_predictor.season_stats.pooled_rates import drop_rate_parts, numerator_for_value
from nfl_predictor.season_stats.qb_stats import (
    CPOE_COLUMN,
    CPOE_PARTS,
    QB_PASSING_TOTALS,
    compute_qb_game_stats_from_pbp,
    qb_epa_per_dropback_expr,
    qb_passer_rating_expr,
)
from nfl_predictor.season_stats.team_stats import (
    add_per_snap_rates,
    compute_team_game_stats_from_pbp,
)

# The position a source lists a quarterback at (``constants.QB_POSITION``).
QB_POSITION = constants.QB_POSITION
# Teams in one game: a game whose official weekly team stats name fewer is credited to one team.
TEAMS_PER_GAME = 2
# The official weekly team stat columns the override reads, besides the game keys.
OFFICIAL_TEAM_STAT_COLUMNS = (
    "passing_yards",
    "rushing_yards",
    "passing_tds",
    "rushing_tds",
    "passing_first_downs",
    "rushing_first_downs",
    "passing_epa",
    "rushing_epa",
    "passing_cpoe",
    "sacks_suffered",
    "passing_interceptions",
    "sack_fumbles_lost",
    "rushing_fumbles_lost",
)


def two_team_games_only(team_stats: pl.DataFrame) -> pl.DataFrame:
    """Return weekly team stats without the games that have a row for only one team."""
    complete = (
        team_stats.group_by("game_id")
        .agg(pl.col("team").n_unique().alias("teams"))
        .filter(pl.col("teams") == TEAMS_PER_GAME)
        .get_column("game_id")
    )
    return team_stats.filter(pl.col("game_id").is_in(complete.to_list()))


_OFFICIAL_QB_RUSHING_FIELDS: dict[str, tuple[str, type[pl.Int64 | pl.Float64]]] = {
    "carries": ("official_qb_carries", pl.Int64),
    "rushing_yards": ("official_qb_rushing_yards", pl.Float64),
    "rushing_tds": ("official_qb_rushing_tds", pl.Int64),
    "rushing_first_downs": ("official_qb_rushing_first_downs", pl.Int64),
    "rushing_epa": ("official_qb_rushing_epa", pl.Float64),
    "rushing_fumbles": ("official_qb_rushing_fumbles", pl.Int64),
    "rushing_fumbles_lost": ("official_qb_rushing_fumbles_lost", pl.Int64),
    "rushing_2pt_conversions": ("official_qb_rushing_2pt_conversions", pl.Int64),
}


def _official_rushing_selection(columns: list[str]) -> list[pl.Expr]:
    """Return the official QB rushing selection, tolerating absent columns."""
    return [
        (pl.col(source).cast(dtype) if source in columns else pl.lit(None, dtype=dtype)).alias(
            target
        )
        for source, (target, dtype) in _OFFICIAL_QB_RUSHING_FIELDS.items()
    ]


def official_weekly_qb_stats(
    weekly_player_stats_df: pl.DataFrame,
    qb_identity_df: pl.DataFrame,
) -> pl.DataFrame:
    """Return authoritative weekly QB passing stats keyed to canonical QB IDs."""
    if weekly_player_stats_df.is_empty() or "player_id" not in weekly_player_stats_df.columns:
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "week": pl.Int64,
                "team_abbr": pl.String,
                "qb_id": pl.String,
                "official_qb_attempts": pl.Int64,
                "official_qb_completions": pl.Int64,
                "official_qb_pass_yards": pl.Float64,
                "official_qb_pass_touchdowns": pl.Int64,
                "official_qb_interceptions": pl.Int64,
                "official_qb_sacks": pl.Int64,
                "official_qb_sack_yards_lost": pl.Float64,
                "official_qb_passing_epa": pl.Float64,
                "official_qb_completion_percentage_above_expectation": pl.Float64,
                "official_qb_carries": pl.Int64,
                "official_qb_rushing_yards": pl.Float64,
                "official_qb_rushing_tds": pl.Int64,
                "official_qb_rushing_first_downs": pl.Int64,
                "official_qb_rushing_epa": pl.Float64,
                "official_qb_rushing_fumbles": pl.Int64,
                "official_qb_rushing_fumbles_lost": pl.Int64,
                "official_qb_rushing_2pt_conversions": pl.Int64,
            }
        )

    qb_name_expr = (
        pl.col("player_display_name").cast(pl.String)
        if "player_display_name" in weekly_player_stats_df.columns
        else (
            pl.col("player_name").cast(pl.String)
            if "player_name" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.String)
        )
    )

    # Player stats give a player's latest position, so the crosswalk's season positions also count:
    # a quarterback later listed at tight end keeps his official stats.
    crosswalk_qbs = (
        qb_identity_df.filter(pl.col("qb_position") == QB_POSITION).get_column("qb_id")
        if {"qb_id", "qb_position"} <= set(qb_identity_df.columns)
        else pl.Series("qb_id", [], dtype=pl.String)
    )
    listed_qb = (
        (pl.col("position") == QB_POSITION) | pl.col("player_id").is_in(crosswalk_qbs.implode())
        if "position" in weekly_player_stats_df.columns
        else pl.lit(value=True)
    )
    official_qb_stats = weekly_player_stats_df.filter(
        listed_qb
        & pl.col("team").is_not_null()
        & pl.col("week").is_not_null()
        & pl.col("player_id").is_not_null()
    ).select(
        pl.col("game_id").cast(pl.String),
        pl.col("week").cast(pl.Int64),
        pl.col("team").cast(pl.String).alias("team_abbr"),
        pl.col("player_id").cast(pl.String).alias("qb_id"),
        qb_name_expr.alias("qb_name"),
        (
            pl.col("attempts").cast(pl.Int64)
            if "attempts" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_qb_attempts"),
        (
            pl.col("completions").cast(pl.Int64)
            if "completions" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_qb_completions"),
        (
            pl.col("passing_yards").cast(pl.Float64)
            if "passing_yards" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("official_qb_pass_yards"),
        (
            pl.col("passing_tds").cast(pl.Int64)
            if "passing_tds" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_qb_pass_touchdowns"),
        (
            pl.col("passing_interceptions").cast(pl.Int64)
            if "passing_interceptions" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_qb_interceptions"),
        (
            pl.col("sacks_suffered").cast(pl.Int64)
            if "sacks_suffered" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_qb_sacks"),
        # nflverse stores the yards lost on sacks as a negative number; the magnitude keeps the
        # column in yards lost, as the play-by-play fallback counts them and ANY/A subtracts them.
        (
            pl.col("sack_yards_lost").cast(pl.Float64).abs()
            if "sack_yards_lost" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("official_qb_sack_yards_lost"),
        (
            pl.col("passing_epa").cast(pl.Float64)
            if "passing_epa" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("official_qb_passing_epa"),
        (
            pl.col("passing_cpoe").cast(pl.Float64)
            if "passing_cpoe" in weekly_player_stats_df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("official_qb_completion_percentage_above_expectation"),
        *_official_rushing_selection(weekly_player_stats_df.columns),
    )
    if official_qb_stats.is_empty() or qb_identity_df.is_empty():
        return official_qb_stats.drop("qb_name")

    return (
        official_qb_stats.join(
            qb_identity_df.select(["qb_id", "qb_name"])
            .unique(subset=["qb_id"], keep="first")
            .rename({"qb_name": "canonical_qb_name"}),
            on="qb_id",
            how="left",
        )
        .with_columns(
            pl.coalesce([pl.col("canonical_qb_name"), pl.col("qb_name")]).alias("qb_name")
        )
        .drop(["canonical_qb_name", "qb_name"])
    )


def override_qb_game_stats_with_official_weekly(
    qb_df: pl.DataFrame,
    official_qb_stats_df: pl.DataFrame,
) -> pl.DataFrame:
    """Replace attempt-based QB game fields with authoritative weekly player stats."""
    if qb_df.is_empty() or official_qb_stats_df.is_empty():
        return qb_df

    overridden = (
        qb_df.join(
            official_qb_stats_df,
            on=["game_id", "week", "team_abbr", "qb_id"],
            how="left",
        )
        .with_columns(
            pl.coalesce([pl.col("official_qb_attempts"), pl.col("qb_attempts")])
            .cast(pl.Int64)
            .alias("qb_attempts"),
            pl.coalesce([pl.col("official_qb_completions"), pl.col("qb_completions")])
            .cast(pl.Int64)
            .alias("qb_completions"),
            pl.coalesce([pl.col("official_qb_pass_yards"), pl.col("qb_pass_yards")]).alias(
                "qb_pass_yards"
            ),
            pl.coalesce([pl.col("official_qb_pass_touchdowns"), pl.col("qb_pass_touchdowns")])
            .cast(pl.Int64)
            .alias("qb_pass_touchdowns"),
            pl.coalesce([pl.col("official_qb_interceptions"), pl.col("qb_interceptions")])
            .cast(pl.Int64)
            .alias("qb_interceptions"),
            pl.coalesce([pl.col("official_qb_sacks"), pl.col("qb_sacks")])
            .cast(pl.Int64)
            .alias("qb_sacks"),
            pl.coalesce(
                [pl.col("official_qb_sack_yards_lost"), pl.col("qb_sack_yards_lost")]
            ).alias("qb_sack_yards_lost"),
            pl.coalesce([pl.col("official_qb_passing_epa"), pl.col("qb_passing_epa")]).alias(
                "qb_passing_epa"
            ),
            pl.coalesce(
                [
                    pl.col("official_qb_completion_percentage_above_expectation"),
                    pl.col("qb_completion_percentage_above_expectation"),
                ]
            ).alias("qb_completion_percentage_above_expectation"),
            pl.col("official_qb_carries").fill_null(0).cast(pl.Int64).alias("qb_carries"),
            pl.col("official_qb_rushing_yards").fill_null(0.0).alias("qb_rushing_yards"),
            pl.col("official_qb_rushing_tds").fill_null(0).cast(pl.Int64).alias("qb_rushing_tds"),
            pl.col("official_qb_rushing_first_downs")
            .fill_null(0)
            .cast(pl.Int64)
            .alias("qb_rushing_first_downs"),
            pl.col("official_qb_rushing_epa").fill_null(0.0).alias("qb_rushing_epa"),
            pl.col("official_qb_rushing_fumbles")
            .fill_null(0)
            .cast(pl.Int64)
            .alias("qb_rushing_fumbles"),
            pl.col("official_qb_rushing_fumbles_lost")
            .fill_null(0)
            .cast(pl.Int64)
            .alias("qb_rushing_fumbles_lost"),
            pl.col("official_qb_rushing_2pt_conversions")
            .fill_null(0)
            .cast(pl.Int64)
            .alias("qb_rushing_2pt_conversions"),
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
        )
        .with_columns(
            pl.when(pl.col("qb_attempts") > 0)
            .then(pl.col("qb_completions") / pl.col("qb_attempts"))
            .otherwise(None)
            .alias("qb_completion_pct"),
            pl.when(pl.col("qb_carries") > 0)
            .then(pl.col("qb_rushing_yards") / pl.col("qb_carries"))
            .otherwise(None)
            .alias("qb_yards_per_carry"),
            pl.when(pl.col("qb_carries") > 0)
            .then(pl.col("qb_rushing_epa") / pl.col("qb_carries"))
            .otherwise(None)
            .alias("qb_epa_per_carry"),
        )
        .drop(
            [
                "official_qb_attempts",
                "official_qb_completions",
                "official_qb_pass_yards",
                "official_qb_pass_touchdowns",
                "official_qb_interceptions",
                "official_qb_sacks",
                "official_qb_sack_yards_lost",
                "official_qb_passing_epa",
                "official_qb_completion_percentage_above_expectation",
                *[target for target, _ in _OFFICIAL_QB_RUSHING_FIELDS.values()],
            ]
        )
    )
    # The official CPOE is weighed by the passer's plays with a CPOE when games are pooled.
    return numerator_for_value(overridden, CPOE_COLUMN)


def official_weekly_team_surface(official_team_stats_df: pl.DataFrame) -> pl.DataFrame:
    """Return authoritative weekly team offense stats for published columns."""
    if official_team_stats_df.is_empty():
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "week": pl.Int64,
                "team": pl.String,
                "opponent_team": pl.String,
                "official_passing_yards": pl.Float64,
                "official_rushing_yards": pl.Float64,
                "official_total_yards": pl.Float64,
                "official_passing_tds": pl.Int64,
                "official_rushing_tds": pl.Int64,
                "official_passing_first_downs": pl.Int64,
                "official_rushing_first_downs": pl.Int64,
                "official_passing_epa": pl.Float64,
                "official_rushing_epa": pl.Float64,
                "official_passing_cpoe": pl.Float64,
                "official_sacks_suffered": pl.Int64,
                "official_passing_interceptions": pl.Int64,
                "official_sack_fumbles_lost": pl.Int64,
                "official_rushing_fumbles_lost": pl.Int64,
            }
        )

    passing_yards_expr = (
        pl.col("passing_yards").cast(pl.Float64)
        if "passing_yards" in official_team_stats_df.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    rushing_yards_expr = (
        pl.col("rushing_yards").cast(pl.Float64)
        if "rushing_yards" in official_team_stats_df.columns
        else pl.lit(None, dtype=pl.Float64)
    )

    return official_team_stats_df.select(
        pl.col("game_id").cast(pl.String),
        pl.col("week").cast(pl.Int64),
        pl.col("team").cast(pl.String),
        pl.col("opponent_team").cast(pl.String),
        passing_yards_expr.alias("official_passing_yards"),
        rushing_yards_expr.alias("official_rushing_yards"),
        (passing_yards_expr + rushing_yards_expr).alias("official_total_yards"),
        (
            pl.col("passing_tds").cast(pl.Int64)
            if "passing_tds" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_passing_tds"),
        (
            pl.col("rushing_tds").cast(pl.Int64)
            if "rushing_tds" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_rushing_tds"),
        (
            pl.col("passing_first_downs").cast(pl.Int64)
            if "passing_first_downs" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_passing_first_downs"),
        (
            pl.col("rushing_first_downs").cast(pl.Int64)
            if "rushing_first_downs" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_rushing_first_downs"),
        (
            pl.col("passing_epa").cast(pl.Float64)
            if "passing_epa" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("official_passing_epa"),
        (
            pl.col("rushing_epa").cast(pl.Float64)
            if "rushing_epa" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("official_rushing_epa"),
        (
            pl.col("passing_cpoe").cast(pl.Float64)
            if "passing_cpoe" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("official_passing_cpoe"),
        (
            pl.col("sacks_suffered").cast(pl.Int64)
            if "sacks_suffered" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_sacks_suffered"),
        (
            pl.col("passing_interceptions").cast(pl.Int64)
            if "passing_interceptions" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_passing_interceptions"),
        (
            pl.col("sack_fumbles_lost").cast(pl.Int64)
            if "sack_fumbles_lost" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_sack_fumbles_lost"),
        (
            pl.col("rushing_fumbles_lost").cast(pl.Int64)
            if "rushing_fumbles_lost" in official_team_stats_df.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("official_rushing_fumbles_lost"),
    )


def override_team_game_stats_with_official_weekly(
    team_df: pl.DataFrame,
    official_team_stats_df: pl.DataFrame,
) -> pl.DataFrame:
    """Replace published team stat columns with authoritative weekly team stats."""
    if team_df.is_empty() or official_team_stats_df.is_empty():
        return team_df

    official_offense = official_weekly_team_surface(official_team_stats_df)
    official_allowed = official_offense.select(
        pl.col("game_id"),
        pl.col("week"),
        pl.col("opponent_team").alias("team"),
        pl.col("team").alias("opponent_team"),
        pl.col("official_passing_yards").alias("official_passing_yards_allowed"),
        pl.col("official_rushing_yards").alias("official_rushing_yards_allowed"),
        pl.col("official_total_yards").alias("official_total_yards_allowed"),
        pl.col("official_passing_tds").alias("official_passing_tds_allowed"),
        pl.col("official_rushing_tds").alias("official_rushing_tds_allowed"),
        pl.col("official_passing_first_downs").alias("official_passing_first_downs_allowed"),
        pl.col("official_rushing_first_downs").alias("official_rushing_first_downs_allowed"),
        pl.col("official_passing_epa").alias("official_passing_epa_allowed"),
        pl.col("official_rushing_epa").alias("official_rushing_epa_allowed"),
        pl.col("official_passing_cpoe").alias("official_passing_cpoe_allowed"),
    )

    overridden = (
        team_df.join(official_offense, on=["game_id", "week", "team", "opponent_team"], how="left")
        .join(official_allowed, on=["game_id", "week", "team", "opponent_team"], how="left")
        .with_columns(
            pl.coalesce([pl.col("official_passing_yards"), pl.col("passing_yards")]).alias(
                "passing_yards"
            ),
            pl.coalesce([pl.col("official_rushing_yards"), pl.col("rushing_yards")]).alias(
                "rushing_yards"
            ),
            pl.coalesce([pl.col("official_total_yards"), pl.col("total_yards")]).alias(
                "total_yards"
            ),
            pl.coalesce([pl.col("official_passing_tds"), pl.col("passing_tds")])
            .cast(pl.Int64)
            .alias("passing_tds"),
            pl.coalesce([pl.col("official_rushing_tds"), pl.col("rushing_tds")])
            .cast(pl.Int64)
            .alias("rushing_tds"),
            pl.coalesce([pl.col("official_passing_first_downs"), pl.col("passing_first_downs")])
            .cast(pl.Int64)
            .alias("passing_first_downs"),
            pl.coalesce([pl.col("official_rushing_first_downs"), pl.col("rushing_first_downs")])
            .cast(pl.Int64)
            .alias("rushing_first_downs"),
            pl.coalesce([pl.col("official_passing_epa"), pl.col("passing_epa")]).alias(
                "passing_epa"
            ),
            pl.coalesce([pl.col("official_rushing_epa"), pl.col("rushing_epa")]).alias(
                "rushing_epa"
            ),
            pl.coalesce([pl.col("official_passing_cpoe"), pl.col("passing_cpoe")]).alias(
                "passing_cpoe"
            ),
            pl.coalesce([pl.col("official_sacks_suffered"), pl.col("sacks_suffered")])
            .cast(pl.Int64)
            .alias("sacks_suffered"),
            pl.coalesce([pl.col("official_passing_interceptions"), pl.col("passing_interceptions")])
            .cast(pl.Int64)
            .alias("passing_interceptions"),
            pl.coalesce([pl.col("official_sack_fumbles_lost"), pl.col("sack_fumbles_lost")])
            .cast(pl.Int64)
            .alias("sack_fumbles_lost"),
            pl.coalesce([pl.col("official_rushing_fumbles_lost"), pl.col("rushing_fumbles_lost")])
            .cast(pl.Int64)
            .alias("rushing_fumbles_lost"),
            pl.coalesce(
                [pl.col("official_passing_yards_allowed"), pl.col("passing_yards_allowed")]
            ).alias("passing_yards_allowed"),
            pl.coalesce(
                [pl.col("official_rushing_yards_allowed"), pl.col("rushing_yards_allowed")]
            ).alias("rushing_yards_allowed"),
            pl.coalesce(
                [pl.col("official_total_yards_allowed"), pl.col("total_yards_allowed")]
            ).alias("total_yards_allowed"),
            pl.coalesce([pl.col("official_passing_tds_allowed"), pl.col("passing_tds_allowed")])
            .cast(pl.Int64)
            .alias("passing_tds_allowed"),
            pl.coalesce([pl.col("official_rushing_tds_allowed"), pl.col("rushing_tds_allowed")])
            .cast(pl.Int64)
            .alias("rushing_tds_allowed"),
            pl.coalesce(
                [
                    pl.col("official_passing_first_downs_allowed"),
                    pl.col("passing_first_downs_allowed"),
                ]
            )
            .cast(pl.Int64)
            .alias("passing_first_downs_allowed"),
            pl.coalesce(
                [
                    pl.col("official_rushing_first_downs_allowed"),
                    pl.col("rushing_first_downs_allowed"),
                ]
            )
            .cast(pl.Int64)
            .alias("rushing_first_downs_allowed"),
            pl.coalesce(
                [pl.col("official_passing_epa_allowed"), pl.col("passing_epa_allowed")]
            ).alias("passing_epa_allowed"),
            pl.coalesce(
                [pl.col("official_rushing_epa_allowed"), pl.col("rushing_epa_allowed")]
            ).alias("rushing_epa_allowed"),
            pl.coalesce(
                [pl.col("official_passing_cpoe_allowed"), pl.col("passing_cpoe_allowed")]
            ).alias("passing_cpoe_allowed"),
        )
        .drop(
            [
                "official_passing_yards",
                "official_rushing_yards",
                "official_total_yards",
                "official_passing_tds",
                "official_rushing_tds",
                "official_passing_first_downs",
                "official_rushing_first_downs",
                "official_passing_epa",
                "official_rushing_epa",
                "official_passing_cpoe",
                "official_sacks_suffered",
                "official_passing_interceptions",
                "official_sack_fumbles_lost",
                "official_rushing_fumbles_lost",
                "official_passing_yards_allowed",
                "official_rushing_yards_allowed",
                "official_total_yards_allowed",
                "official_passing_tds_allowed",
                "official_rushing_tds_allowed",
                "official_passing_first_downs_allowed",
                "official_rushing_first_downs_allowed",
                "official_passing_epa_allowed",
                "official_rushing_epa_allowed",
                "official_passing_cpoe_allowed",
            ]
        )
    )
    # The per-snap rates divide the official totals now, with their parts beside them, and the
    # official CPOE is weighed by the plays behind it.
    for cpoe in ("passing_cpoe", "passing_cpoe_allowed"):
        overridden = numerator_for_value(overridden, cpoe)
    return add_per_snap_rates(overridden)


def build_weekly_team_stats(
    pbp_df: pl.DataFrame,
    player_stats_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
    official_team_stats_df: pl.DataFrame,
) -> pl.DataFrame:
    """Return one season's team-game rows: play-by-play totals, then the official weekly stats.

    Args:
        pbp_df: The season's regular-season play-by-play, as ``sources.prepare_pbp`` leaves it.
        player_stats_df: The season's regular-season weekly player stats (defense-only stats).
        schedule_df: The season's played regular-season games (``home_team``, ``away_team``,
            ``home_score``, ``away_score``).
        official_team_stats_df: The season's regular-season official weekly team stats; a game
            with rows for only one team is left out, so its play-by-play values stand.

    Returns:
        One row per team-game, sorted by team, week, and game, with every rate's hidden parts

    """
    weekly_team_stats_df = compute_team_game_stats_from_pbp(pbp_df, player_stats_df, schedule_df)
    if "game_id" in official_team_stats_df.columns:
        official_team_stats_df = two_team_games_only(official_team_stats_df)
    return override_team_game_stats_with_official_weekly(
        weekly_team_stats_df,
        official_team_stats_df,
    )


def build_qb_game_stats(
    pbp_df: pl.DataFrame,
    snap_counts_df: pl.DataFrame,
    qb_identity_df: pl.DataFrame,
    weekly_player_stats_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build QB game stats from play-by-play, overriding passing fields with official stats."""
    qb_df = compute_qb_game_stats_from_pbp(pbp_df, snap_counts_df, qb_identity_df)
    official_qb_stats_df = official_weekly_qb_stats(weekly_player_stats_df, qb_identity_df)
    qb_df = override_qb_game_stats_with_official_weekly(qb_df, official_qb_stats_df)

    passer_rating = qb_passer_rating_expr(
        *(pl.col(column).cast(pl.Float64) for column in QB_PASSING_TOTALS)
    )

    return qb_df.with_columns(
        pl.coalesce([pl.col("qb_id"), pl.col("snap_player_id"), pl.col("qb_name")]).alias("qb_id"),
        passer_rating.alias("qb_passer_rating"),
    ).select(
        [
            "game_id",
            "week",
            "team_abbr",
            "qb_name",
            "qb_id",
            "snap_player_id",
            "qb_dropbacks",
            "qb_offense_snaps",
            "qb_attempts",
            "qb_completions",
            "qb_pass_yards",
            "qb_pass_touchdowns",
            "qb_interceptions",
            "qb_sacks",
            "qb_sack_yards_lost",
            "qb_sack_fumbles_lost",
            "qb_passing_epa",
            "qb_spike_epa",
            "qb_epa_per_dropback",
            "qb_pass_yards_per_dropback",
            "qb_td_int_margin_rate",
            "qb_sack_rate",
            "qb_any_a",
            "qb_fourth_quarter_comeback",
            "qb_game_winning_drive",
            *CPOE_PARTS,
            "qb_passer_rating",
        ]
        + [
            column
            for column in (
                "qb_completion_pct",
                "qb_carries",
                "qb_rushing_yards",
                "qb_rushing_tds",
                "qb_rushing_first_downs",
                "qb_rushing_epa",
                "qb_rushing_fumbles",
                "qb_rushing_fumbles_lost",
                "qb_rushing_2pt_conversions",
                "qb_designed_carries",
                "qb_designed_rush_yards",
                "qb_designed_rush_epa",
                "qb_scrambles",
                "qb_scramble_yards",
                "qb_kneels",
                "qb_yards_per_carry",
                "qb_epa_per_carry",
                "qb_scramble_rate",
                "qb_yards_per_scramble",
                "qb_designed_yards_per_carry",
                "qb_designed_epa_per_carry",
            )
            if column in qb_df.columns
        ]
    )


def build_team_game_logs(weekly_df: pl.DataFrame) -> pl.DataFrame:
    """Return team game logs with identity columns first, sorted by team and week.

    The rates' hidden numerators and denominators (``pooled_rates``) stay out of the logs.
    """
    published = drop_rate_parts(weekly_df)
    leading = [c for c in ("game_id", "week", "team", "opponent_team") if c in published.columns]
    rest = [c for c in published.columns if c not in leading and c not in {"season", "season_type"}]
    return published.select(leading + rest).sort(
        [c for c in ("team", "week", "game_id") if c in published.columns]
    )


def build_qb_game_logs(qb_df: pl.DataFrame, weekly_df: pl.DataFrame) -> pl.DataFrame:
    """Return QB game logs enriched with opponent, venue, and game-result context.

    The rates' hidden numerators and denominators (``pooled_rates``) stay out of the logs.
    """
    context = weekly_df.select(
        [
            column
            for column in (
                "team",
                "week",
                "opponent_team",
                "is_home",
                "points_for",
                "points_allowed",
                "point_margin",
                "win_value",
                "turnover_margin",
            )
            if column in weekly_df.columns
        ]
    ).rename({"team": "team_abbr"})
    qb_game_logs = (
        drop_rate_parts(qb_df)
        .join(context, on=["team_abbr", "week"], how="left")
        .rename({"team_abbr": "team"})
    )
    leading = [
        c
        for c in ("game_id", "week", "team", "opponent_team", "qb_id", "qb_name")
        if c in qb_game_logs.columns
    ]
    rest = [
        c
        for c in qb_game_logs.columns
        if c not in leading and c not in {"season", "season_type", "snap_player_id"}
    ]
    return qb_game_logs.select(leading + rest).sort(["team", "week", "game_id", "qb_name"])


__all__ = [
    "OFFICIAL_TEAM_STAT_COLUMNS",
    "QB_POSITION",
    "TEAMS_PER_GAME",
    "build_qb_game_logs",
    "build_qb_game_stats",
    "build_team_game_logs",
    "build_weekly_team_stats",
    "official_weekly_qb_stats",
    "official_weekly_team_surface",
    "override_qb_game_stats_with_official_weekly",
    "override_team_game_stats_with_official_weekly",
    "two_team_games_only",
]
