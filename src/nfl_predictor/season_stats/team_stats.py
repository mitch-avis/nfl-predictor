"""Compute per-game team statistics (team-level and QB-level)."""

import polars as pl

from nfl_predictor.season_stats.pbp_expressions import (
    dropback_expr,
    lost_fumble_team_expr,
    scrimmage_snap_expr,
    value_expr,
)
from nfl_predictor.season_stats.pooled_rates import (
    is_rate_part,
    mean_with_parts,
    pooled_rate,
    rate_parts,
    ratio_with_parts,
)
from nfl_predictor.season_stats.team_stats_expanded import (
    TEAM_FORMULA_RATES,
    compute_expanded_team_game_stats,
)

_DEFENSE_ONLY_PLAYER_STATS = [
    "def_tackles_for_loss",
    "def_fumbles_forced",
    "def_sacks",
    "def_qb_hits",
    "def_interceptions",
    "def_pass_defended",
    "def_safeties",
]


# Per-snap rates of the team-game totals: (numerator, denominator, rate). Each carries its numerator
# and denominator as hidden parts, so rows for several games pool it (``pooled_rates``).
PER_SNAP_RATES: tuple[tuple[str, str, str], ...] = (
    ("points_for", "offensive_snaps", "points_per_offensive_snap"),
    ("total_yards", "offensive_snaps", "total_yards_per_offensive_snap"),
    ("passing_yards", "offensive_snaps", "passing_yards_per_offensive_snap"),
    ("rushing_yards", "offensive_snaps", "rushing_yards_per_offensive_snap"),
    ("passing_epa", "offensive_snaps", "passing_epa_per_offensive_snap"),
    ("rushing_epa", "offensive_snaps", "rushing_epa_per_offensive_snap"),
    ("passing_tds", "offensive_snaps", "passing_tds_per_offensive_snap"),
    ("rushing_tds", "offensive_snaps", "rushing_tds_per_offensive_snap"),
    ("sacks_suffered", "offensive_snaps", "sacks_suffered_per_offensive_snap"),
    (
        "passing_interceptions",
        "offensive_snaps",
        "passing_interceptions_per_offensive_snap",
    ),
    ("sack_fumbles_lost", "offensive_snaps", "sack_fumbles_lost_per_offensive_snap"),
    (
        "rushing_fumbles_lost",
        "offensive_snaps",
        "rushing_fumbles_lost_per_offensive_snap",
    ),
    (
        "passing_first_downs",
        "offensive_snaps",
        "passing_first_downs_per_offensive_snap",
    ),
    (
        "rushing_first_downs",
        "offensive_snaps",
        "rushing_first_downs_per_offensive_snap",
    ),
    ("points_allowed", "defensive_snaps", "points_allowed_per_defensive_snap"),
    (
        "total_yards_allowed",
        "defensive_snaps",
        "total_yards_allowed_per_defensive_snap",
    ),
    (
        "passing_yards_allowed",
        "defensive_snaps",
        "passing_yards_allowed_per_defensive_snap",
    ),
    (
        "rushing_yards_allowed",
        "defensive_snaps",
        "rushing_yards_allowed_per_defensive_snap",
    ),
    (
        "passing_epa_allowed",
        "defensive_snaps",
        "passing_epa_allowed_per_defensive_snap",
    ),
    (
        "rushing_epa_allowed",
        "defensive_snaps",
        "rushing_epa_allowed_per_defensive_snap",
    ),
    (
        "passing_tds_allowed",
        "defensive_snaps",
        "passing_tds_allowed_per_defensive_snap",
    ),
    (
        "rushing_tds_allowed",
        "defensive_snaps",
        "rushing_tds_allowed_per_defensive_snap",
    ),
    (
        "passing_first_downs_allowed",
        "defensive_snaps",
        "passing_first_downs_allowed_per_defensive_snap",
    ),
    (
        "rushing_first_downs_allowed",
        "defensive_snaps",
        "rushing_first_downs_allowed_per_defensive_snap",
    ),
    ("def_sacks", "defensive_snaps", "def_sacks_per_defensive_snap"),
    (
        "def_interceptions",
        "defensive_snaps",
        "def_interceptions_per_defensive_snap",
    ),
    (
        "def_pass_defended",
        "defensive_snaps",
        "def_pass_defended_per_defensive_snap",
    ),
    (
        "def_tackles_for_loss",
        "defensive_snaps",
        "def_tackles_for_loss_per_defensive_snap",
    ),
    ("def_qb_hits", "defensive_snaps", "def_qb_hits_per_defensive_snap"),
    (
        "def_fumbles_forced",
        "defensive_snaps",
        "def_fumbles_forced_per_defensive_snap",
    ),
    ("def_safeties", "defensive_snaps", "def_safeties_per_defensive_snap"),
)


def _get_numeric_stat_cols(df: pl.DataFrame) -> list[str]:
    """Return numeric stat column names, leaving out identifiers and hidden rate parts."""
    exclude = {"season", "week", "season_type", "games"}
    return [
        col
        for col, dtype in zip(df.columns, df.dtypes, strict=True)
        if dtype.is_numeric() and col not in exclude and not is_rate_part(col)
    ]


def _games_agg_exprs(games: pl.DataFrame, *, longest_as_max: bool = True) -> list[pl.Expr]:
    """Return one aggregation per stat column for rows of several games.

    A rate with its numerator and denominator (``pooled_rates``) is pooled over the games, and a
    rate built from other rates (``TEAM_FORMULA_RATES``) is rebuilt from their pooled values. A
    ``longest_`` stat keeps the largest game value unless ``longest_as_max`` is false (opponent
    rows average it per game, as published); every other stat is averaged per game.
    """
    pooled = set(rate_parts(games.columns))
    exprs: list[pl.Expr] = []
    for column in _get_numeric_stat_cols(games):
        formula = TEAM_FORMULA_RATES.get(column)
        if column in pooled:
            exprs.append(pooled_rate(column))
        elif formula is not None and pooled.issuperset(formula.inputs):
            inputs = [pooled_rate(name) for name in formula.inputs]
            exprs.append(formula.combine(*inputs).alias(column))
        elif longest_as_max and column.startswith("longest_"):
            exprs.append(pl.col(column).max().alias(column))
        else:
            exprs.append(pl.col(column).mean().alias(column))
    return exprs


def _pass_cpoe_expr(columns: list[str]) -> pl.Expr:
    """Return each pass play's completion probability over expected (null on other plays)."""
    cpoe = pl.col("cpoe") if "cpoe" in columns else pl.lit(None, dtype=pl.Float64)
    return pl.when(value_expr(columns, "pass") > 0).then(cpoe).otherwise(None)


def add_per_snap_rates(team_games: pl.DataFrame) -> pl.DataFrame:
    """Return ``team_games`` with every per-snap rate its totals allow, each with its parts."""
    return team_games.with_columns(
        [
            expr
            for numerator, denominator, rate in PER_SNAP_RATES
            if {numerator, denominator}.issubset(team_games.columns)
            for expr in ratio_with_parts(rate, numerator, denominator)
        ]
    )


def _extract_points_per_team_game(schedule_df: pl.DataFrame) -> pl.DataFrame:
    """Pivot schedule scores into one row per team-game with points for and allowed."""
    select_keys = [key for key in ("game_id", "week") if key in schedule_df.columns]
    home = schedule_df.select(
        [
            *select_keys,
            pl.col("home_team").alias("team"),
            pl.col("away_team").alias("opponent_team"),
            pl.lit(True).alias("is_home"),
            pl.col("home_score").alias("points_for"),
            pl.col("away_score").alias("points_allowed"),
        ]
    )
    away = schedule_df.select(
        [
            *select_keys,
            pl.col("away_team").alias("team"),
            pl.col("home_team").alias("opponent_team"),
            pl.lit(False).alias("is_home"),
            pl.col("away_score").alias("points_for"),
            pl.col("home_score").alias("points_allowed"),
        ]
    )
    return pl.concat([home, away])


def _charted_defense_stats(frame: pl.DataFrame) -> list[str]:
    """Return the defense-only player stats ``frame`` has any value for.

    The weekly player stats loader leaves a stat nflverse credits to no one that season null for
    every player, and such a stat stays null for every team, never 0.
    """
    return [
        column
        for column in _DEFENSE_ONLY_PLAYER_STATS
        if column in frame.columns and frame.get_column(column).is_not_null().any()
    ]


def _aggregate_defense_only_player_stats(player_stats_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate defense-only player stats to one row per team-week-opponent.

    A stat with no value for any player (one the season lacks) stays null.
    """
    if player_stats_df.is_empty():
        return pl.DataFrame(
            schema={"team": pl.String, "opponent_team": pl.String, "week": pl.Int64}
        )

    defense_cols = [
        column for column in _DEFENSE_ONLY_PLAYER_STATS if column in player_stats_df.columns
    ]
    if not defense_cols:
        return pl.DataFrame(
            schema={"team": pl.String, "opponent_team": pl.String, "week": pl.Int64}
        )

    group_keys = [
        key
        for key in ("season", "season_type", "week", "team", "opponent_team")
        if key in player_stats_df.columns
    ]
    charted = _charted_defense_stats(player_stats_df)
    return player_stats_df.group_by(group_keys).agg(
        [
            pl.col(column).fill_null(0).sum().alias(column)
            if column in charted
            else pl.lit(None, dtype=player_stats_df.schema[column]).alias(column)
            for column in defense_cols
        ]
    )


def compute_team_game_stats_from_pbp(
    pbp_df: pl.DataFrame,
    player_stats_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
) -> pl.DataFrame:
    """Derive one row per team-game from PBP, plus defense-only player-stat add-ons.

    Per-snap rate fields are computed as the relevant game total divided by
    offensive or defensive snaps for that team-game.
    """
    if pbp_df.is_empty():
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "season": pl.Int64,
                "season_type": pl.String,
                "week": pl.Int64,
                "team": pl.String,
                "opponent_team": pl.String,
                "is_home": pl.Boolean,
                "games": pl.Int64,
            }
        )

    group_keys = [
        key for key in ("game_id", "season", "season_type", "week") if key in pbp_df.columns
    ]
    # A strip-sack or run fumble the defense recovers and fumbles back is the defense's lost fumble.
    offense_lost_fumble = (lost_fumble_team_expr(pbp_df.columns) == pl.col("posteam")).fill_null(
        value=False
    )
    offense_stats = (
        pbp_df.filter(
            pl.col("posteam").is_not_null()
            & pl.col("defteam").is_not_null()
            & scrimmage_snap_expr(pbp_df.columns)
        )
        .group_by([*group_keys, "posteam", "defteam"])
        .agg(
            [
                value_expr(pbp_df.columns, "passing_yards", 0.0).sum().alias("passing_yards"),
                value_expr(pbp_df.columns, "rushing_yards", 0.0).sum().alias("rushing_yards"),
                pl.when(dropback_expr(pbp_df.columns))
                .then(value_expr(pbp_df.columns, "epa", 0.0))
                .otherwise(0.0)
                .sum()
                .alias("passing_epa"),
                pl.when(value_expr(pbp_df.columns, "rush") > 0)
                .then(value_expr(pbp_df.columns, "epa", 0.0))
                .otherwise(0.0)
                .sum()
                .alias("rushing_epa"),
                value_expr(pbp_df.columns, "pass_touchdown")
                .sum()
                .cast(pl.Int64)
                .alias("passing_tds"),
                value_expr(pbp_df.columns, "rush_touchdown")
                .sum()
                .cast(pl.Int64)
                .alias("rushing_tds"),
                pl.when(value_expr(pbp_df.columns, "pass") > 0)
                .then(value_expr(pbp_df.columns, "first_down"))
                .otherwise(0)
                .sum()
                .cast(pl.Int64)
                .alias("passing_first_downs"),
                pl.when(value_expr(pbp_df.columns, "rush") > 0)
                .then(value_expr(pbp_df.columns, "first_down"))
                .otherwise(0)
                .sum()
                .cast(pl.Int64)
                .alias("rushing_first_downs"),
                *mean_with_parts("passing_cpoe", _pass_cpoe_expr(pbp_df.columns)),
                value_expr(pbp_df.columns, "sack").sum().cast(pl.Int64).alias("sacks_suffered"),
                value_expr(pbp_df.columns, "interception")
                .sum()
                .cast(pl.Int64)
                .alias("passing_interceptions"),
                ((value_expr(pbp_df.columns, "sack") > 0) & offense_lost_fumble)
                .sum()
                .cast(pl.Int64)
                .alias("sack_fumbles_lost"),
                ((value_expr(pbp_df.columns, "rush") > 0) & offense_lost_fumble)
                .sum()
                .cast(pl.Int64)
                .alias("rushing_fumbles_lost"),
            ]
        )
        .rename({"posteam": "team", "defteam": "opponent_team"})
        .with_columns((pl.col("passing_yards") + pl.col("rushing_yards")).alias("total_yards"))
    )

    allowed_stats = (
        pbp_df.filter(
            pl.col("posteam").is_not_null()
            & pl.col("defteam").is_not_null()
            & scrimmage_snap_expr(pbp_df.columns)
        )
        .group_by([*group_keys, "defteam", "posteam"])
        .agg(
            [
                value_expr(pbp_df.columns, "passing_yards", 0.0)
                .sum()
                .alias("passing_yards_allowed"),
                value_expr(pbp_df.columns, "rushing_yards", 0.0)
                .sum()
                .alias("rushing_yards_allowed"),
                pl.when(dropback_expr(pbp_df.columns))
                .then(value_expr(pbp_df.columns, "epa", 0.0))
                .otherwise(0.0)
                .sum()
                .alias("passing_epa_allowed"),
                pl.when(value_expr(pbp_df.columns, "rush") > 0)
                .then(value_expr(pbp_df.columns, "epa", 0.0))
                .otherwise(0.0)
                .sum()
                .alias("rushing_epa_allowed"),
                value_expr(pbp_df.columns, "pass_touchdown")
                .sum()
                .cast(pl.Int64)
                .alias("passing_tds_allowed"),
                value_expr(pbp_df.columns, "rush_touchdown")
                .sum()
                .cast(pl.Int64)
                .alias("rushing_tds_allowed"),
                pl.when(value_expr(pbp_df.columns, "pass") > 0)
                .then(value_expr(pbp_df.columns, "first_down"))
                .otherwise(0)
                .sum()
                .cast(pl.Int64)
                .alias("passing_first_downs_allowed"),
                pl.when(value_expr(pbp_df.columns, "rush") > 0)
                .then(value_expr(pbp_df.columns, "first_down"))
                .otherwise(0)
                .sum()
                .cast(pl.Int64)
                .alias("rushing_first_downs_allowed"),
                *mean_with_parts("passing_cpoe_allowed", _pass_cpoe_expr(pbp_df.columns)),
            ]
        )
        .rename({"defteam": "team", "posteam": "opponent_team"})
        .with_columns(
            (pl.col("passing_yards_allowed") + pl.col("rushing_yards_allowed")).alias(
                "total_yards_allowed"
            )
        )
    )

    snap_counts = compute_team_snap_counts_from_pbp(pbp_df)
    points = _extract_points_per_team_game(schedule_df)
    defense_only = _aggregate_defense_only_player_stats(player_stats_df)

    join_keys = [key for key in group_keys if key in points.columns]
    join_keys.extend(["team", "opponent_team"])

    result = (
        offense_stats.join(allowed_stats, on=[*group_keys, "team", "opponent_team"], how="left")
        .join(
            snap_counts,
            on=[key for key in ("game_id", "week", "team") if key in offense_stats.columns],
            how="left",
        )
        .join(points, on=join_keys, how="left")
        .join(
            defense_only,
            on=[
                key for key in [*group_keys, "team", "opponent_team"] if key in defense_only.columns
            ],
            how="left",
        )
        .with_columns(pl.lit(1).cast(pl.Int64).alias("games"))
    )

    fill_zero_exprs = [
        pl.col(column).fill_null(0)
        for column in [
            "offensive_snaps",
            "defensive_snaps",
            "passing_yards_allowed",
            "rushing_yards_allowed",
            "total_yards_allowed",
            "passing_epa_allowed",
            "rushing_epa_allowed",
            "passing_tds_allowed",
            "rushing_tds_allowed",
            "passing_first_downs_allowed",
            "rushing_first_downs_allowed",
            # A team without player-stat rows made none of a stat the season has.
            *_charted_defense_stats(result),
        ]
        if column in result.columns
    ]
    result = result.with_columns(fill_zero_exprs).with_columns(
        (pl.col("points_for") - pl.col("points_allowed")).alias("point_margin"),
        pl.when(pl.col("points_for") > pl.col("points_allowed"))
        .then(1.0)
        .when(pl.col("points_for") < pl.col("points_allowed"))
        .then(0.0)
        .otherwise(0.5)
        .alias("win_value"),
    )

    result = add_per_snap_rates(result)

    expanded = compute_expanded_team_game_stats(pbp_df)
    if "team" in expanded.columns and not expanded.is_empty():
        expansion_keys = [
            key
            for key in [*group_keys, "team", "opponent_team"]
            if key in expanded.columns and key in result.columns
        ]
        result = result.join(expanded, on=expansion_keys, how="left")

    result = _add_receiving_display_mirrors(result)

    return result.sort([key for key in ("team", "week", "game_id") if key in result.columns])


# Receiving display mirrors: at team level these receiving columns restate the
# passing surface exactly (verified: team receiving yards equal gross passing
# yards). Kept for display only; every alias is duplicate_of its source in the
# metric registry and is never ratings-eligible. Targets and catch rate are not
# aliases: throwaways and spikes are attempts but not targets.
_RECEIVING_ALIAS_SOURCES = {
    "receptions": "completions",
    "receiving_yards": "passing_yards",
    "receiving_tds": "passing_tds",
    "receiving_air_yards": "passing_air_yards",
    "receiving_yards_after_catch": "passing_yards_after_catch",
    "receiving_first_downs": "passing_first_downs",
    "receptions_allowed": "completions_allowed",
    "receiving_yards_allowed": "passing_yards_allowed",
}


def _add_receiving_display_mirrors(result: pl.DataFrame) -> pl.DataFrame:
    """Add display-only receiving aliases of the passing surface."""
    aliases = [
        pl.col(source).alias(alias)
        for alias, source in _RECEIVING_ALIAS_SOURCES.items()
        if source in result.columns
    ]
    return result.with_columns(aliases) if aliases else result


def compute_team_snap_counts_from_pbp(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Compute team offensive and defensive snap counts from play-by-play data."""
    if pbp_df.is_empty():
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "week": pl.Int64,
                "team": pl.String,
                "offensive_snaps": pl.Int64,
                "defensive_snaps": pl.Int64,
            }
        )

    scrimmage_plays = pbp_df.filter(
        pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & scrimmage_snap_expr(pbp_df.columns)
    )

    offense = scrimmage_plays.select(
        "game_id",
        "week",
        pl.col("posteam").alias("team"),
        pl.lit(1).alias("offensive_snaps"),
        pl.lit(0).alias("defensive_snaps"),
    )
    defense = scrimmage_plays.select(
        "game_id",
        "week",
        pl.col("defteam").alias("team"),
        pl.lit(0).alias("offensive_snaps"),
        pl.lit(1).alias("defensive_snaps"),
    )

    return (
        pl.concat([offense, defense])
        .group_by(["game_id", "week", "team"])
        .agg(
            [
                pl.col("offensive_snaps").sum().cast(pl.Int64).alias("offensive_snaps"),
                pl.col("defensive_snaps").sum().cast(pl.Int64).alias("defensive_snaps"),
            ]
        )
        .sort(["team", "week", "game_id"])
    )


def compute_all_teams_per_game(weekly_df: pl.DataFrame) -> pl.DataFrame:
    """Compute every team's season row from its game rows.

    One row per team: counts averaged per game, rates pooled over the season (its summed
    numerator over its summed denominator), and the longest plays as season maxima
    (``_games_agg_exprs``).
    """
    return (
        weekly_df.group_by("team")
        .agg([*_games_agg_exprs(weekly_df), pl.col("team").count().alias("games_played")])
        .sort("team")
    )


def compute_win_totals(weekly_df: pl.DataFrame) -> pl.DataFrame:
    """Compute wins, losses, ties, and win_pct per team from weekly game results.

    A game counts when both scores are present; a team held scoreless still records the loss.
    """
    valid = weekly_df.filter(
        pl.col("points_for").is_not_null() & pl.col("points_allowed").is_not_null()
    )
    return (
        valid.with_columns(
            [
                (pl.col("points_for") > pl.col("points_allowed")).cast(pl.Int32).alias("win"),
                (pl.col("points_for") < pl.col("points_allowed")).cast(pl.Int32).alias("loss"),
                (pl.col("points_for") == pl.col("points_allowed")).cast(pl.Int32).alias("tie"),
            ]
        )
        .group_by("team")
        .agg(
            [
                pl.col("win").sum().alias("wins"),
                pl.col("loss").sum().alias("losses"),
                pl.col("tie").sum().alias("ties"),
            ]
        )
        .with_columns(
            (
                (pl.col("wins") + 0.5 * pl.col("ties"))
                / (pl.col("wins") + pl.col("losses") + pl.col("ties"))
            ).alias("win_pct")
        )
        .sort("team")
    )


def compute_team_stats_excluding_opponents(
    weekly_df: pl.DataFrame, pairs: pl.DataFrame
) -> pl.DataFrame:
    """Return each team's row over its games not against the opponent it is paired with.

    ``pairs`` holds ``team`` and ``excluded_opponent``. The result has one row per pair with a game
    left, keyed by both columns: the team's remaining games aggregated as a season row is
    (``_games_agg_exprs``: counts per game, rates pooled over the games), except that a longest
    play is averaged per game, and ``games_included``. One aggregation covers every pair.
    """
    games = (
        pairs.select("team", "excluded_opponent")
        .unique(maintain_order=True)
        .join(weekly_df, on="team", how="inner")
        .filter(pl.col("opponent_team") != pl.col("excluded_opponent"))
    )
    return games.group_by(["team", "excluded_opponent"], maintain_order=True).agg(
        [
            *_games_agg_exprs(weekly_df, longest_as_max=False),
            pl.len().cast(pl.Int64).alias("games_included"),
        ]
    )
