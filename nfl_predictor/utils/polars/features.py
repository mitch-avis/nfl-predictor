"""
Polars-based utility functions for NFL data processing.

This module provides functions for loading, transforming, and processing NFL game data
using nflreadpy as the primary data source and Polars for high-performance data manipulation.

Key functionality includes:
    - Loading NFL schedules with game metadata, lines/odds, and stadium information
    - Loading per-game team statistics from nflreadpy
    - Loading ELO and QB ELO ratings from local CSV files
    - Loading TeamRankings data from scraped CSV files
    - Aggregating per-game stats into rolling averages for ML features
    - Computing derived metrics (yards per point, penalty efficiency, etc.)
    - Normalizing team abbreviations across different data sources

Data Flow:
    1. load_schedule() -> Schedule with lines, venue info, city/state
    2. load_team_stats() -> Per-game offensive/defensive stats
    3. add_scoring_data_to_team_stats() -> Adds points scored/allowed from schedule
    4. aggregate_team_stats_to_week() -> Rolling averages for prediction
    5. merge_schedule_with_team_stats() -> Combines schedule with aggregated stats
    6. calculate_stat_differentials() -> Computes away - home differences

Example:
    >>> from nfl_predictor.utils import polars_utils
    >>> schedule = polars_utils.load_schedule([2023, 2024])
    >>> team_stats = polars_utils.load_team_stats([2023, 2024])
    >>> agg_stats = polars_utils.aggregate_team_stats_to_week(team_stats, week=10, season=2024)
"""

from typing import Sequence

import polars as pl
from polars.datatypes.classes import DataTypeClass

from nfl_predictor import constants

NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


def compute_team_records_before_week(
    schedule_df: pl.DataFrame,
    *,
    season: int,
    week: int,
    include_postseason: bool = False,
) -> pl.DataFrame:
    """Compute season-to-date W-L-T records for each team strictly before a week.

    This function is time-safe: it only uses games with `week < week` and requires scores
    to be present.

    Args:
        schedule_df: Schedule DataFrame containing at least season/week/team/score columns.
        season: Season year to compute records for.
        week: Week number for which records should reflect games strictly before this week.
        include_postseason: If True, include non-REG games in the record computation.

    Returns:
        DataFrame with one row per team (`team_abbr`) and record columns:
        wins/losses/ties, division_*, conference_*.
    """

    required = {
        "season",
        "week",
        "game_type",
        "away_abbr",
        "home_abbr",
        "away_score",
        "home_score",
    }
    missing = sorted(required - set(schedule_df.columns))
    if missing:
        raise ValueError(f"schedule_df missing required columns: {missing}")

    prior_games = schedule_df.filter(
        (pl.col("season") == season)
        & (pl.col("week") < week)
        & (pl.col("away_score").is_not_null())
        & (pl.col("home_score").is_not_null())
    )
    if not include_postseason:
        prior_games = prior_games.filter(pl.col("game_type") == "REG")

    if prior_games.height == 0:
        return pl.DataFrame(
            {
                "team_abbr": pl.Series([], dtype=pl.Utf8),
                "wins": pl.Series([], dtype=pl.Int32),
                "losses": pl.Series([], dtype=pl.Int32),
                "ties": pl.Series([], dtype=pl.Int32),
                "games_played": pl.Series([], dtype=pl.Int32),
                "win_pct": pl.Series([], dtype=pl.Float32),
                "division_wins": pl.Series([], dtype=pl.Int32),
                "division_losses": pl.Series([], dtype=pl.Int32),
                "division_ties": pl.Series([], dtype=pl.Int32),
                "conference_wins": pl.Series([], dtype=pl.Int32),
                "conference_losses": pl.Series([], dtype=pl.Int32),
                "conference_ties": pl.Series([], dtype=pl.Int32),
            }
        )

    base = prior_games.select(
        [
            "away_abbr",
            "home_abbr",
            "away_score",
            "home_score",
        ]
    )

    home_rows = base.select(
        [
            pl.col("home_abbr").alias("team_abbr"),
            pl.col("away_abbr").alias("opponent_abbr"),
            (pl.col("home_score") > pl.col("away_score")).cast(pl.Int32).alias("win"),
            (pl.col("home_score") < pl.col("away_score")).cast(pl.Int32).alias("loss"),
            (pl.col("home_score") == pl.col("away_score")).cast(pl.Int32).alias("tie"),
        ]
    )
    away_rows = base.select(
        [
            pl.col("away_abbr").alias("team_abbr"),
            pl.col("home_abbr").alias("opponent_abbr"),
            (pl.col("away_score") > pl.col("home_score")).cast(pl.Int32).alias("win"),
            (pl.col("away_score") < pl.col("home_score")).cast(pl.Int32).alias("loss"),
            (pl.col("away_score") == pl.col("home_score")).cast(pl.Int32).alias("tie"),
        ]
    )

    team_games = pl.concat([home_rows, away_rows], how="vertical")

    division_map = constants.TEAM_TO_DIVISION
    conference_map = constants.TEAM_TO_CONFERENCE
    team_games = team_games.with_columns(
        [
            pl.col("team_abbr").replace_strict(division_map, default=None).alias("team_division"),
            pl.col("opponent_abbr")
            .replace_strict(division_map, default=None)
            .alias("opp_division"),
            pl.col("team_abbr")
            .replace_strict(conference_map, default=None)
            .alias("team_conference"),
            pl.col("opponent_abbr")
            .replace_strict(conference_map, default=None)
            .alias("opp_conference"),
        ]
    ).with_columns(
        [
            (pl.col("team_division") == pl.col("opp_division"))
            .fill_null(False)
            .alias("is_division_game"),
            (pl.col("team_conference") == pl.col("opp_conference"))
            .fill_null(False)
            .alias("is_conference_game"),
        ]
    )

    aggregated = team_games.group_by("team_abbr").agg(
        [
            pl.col("win").sum().cast(pl.Int32).alias("wins"),
            pl.col("loss").sum().cast(pl.Int32).alias("losses"),
            pl.col("tie").sum().cast(pl.Int32).alias("ties"),
            pl.when(pl.col("is_division_game"))
            .then(pl.col("win"))
            .otherwise(0)
            .sum()
            .cast(pl.Int32)
            .alias("division_wins"),
            pl.when(pl.col("is_division_game"))
            .then(pl.col("loss"))
            .otherwise(0)
            .sum()
            .cast(pl.Int32)
            .alias("division_losses"),
            pl.when(pl.col("is_division_game"))
            .then(pl.col("tie"))
            .otherwise(0)
            .sum()
            .cast(pl.Int32)
            .alias("division_ties"),
            pl.when(pl.col("is_conference_game"))
            .then(pl.col("win"))
            .otherwise(0)
            .sum()
            .cast(pl.Int32)
            .alias("conference_wins"),
            pl.when(pl.col("is_conference_game"))
            .then(pl.col("loss"))
            .otherwise(0)
            .sum()
            .cast(pl.Int32)
            .alias("conference_losses"),
            pl.when(pl.col("is_conference_game"))
            .then(pl.col("tie"))
            .otherwise(0)
            .sum()
            .cast(pl.Int32)
            .alias("conference_ties"),
        ]
    )

    aggregated = aggregated.with_columns(
        [
            (pl.col("wins") + pl.col("losses") + pl.col("ties"))
            .cast(pl.Int32)
            .alias("games_played"),
            pl.when(pl.col("wins") + pl.col("losses") + pl.col("ties") > 0)
            .then(
                (pl.col("wins") / (pl.col("wins") + pl.col("losses") + pl.col("ties"))).cast(
                    pl.Float32
                )
            )
            .otherwise(pl.lit(0.0, dtype=pl.Float32))
            .alias("win_pct"),
        ]
    )

    return aggregated.sort("team_abbr")


def add_divisional_matchup_feature(df: pl.DataFrame) -> pl.DataFrame:
    """Add an `is_divisional_matchup` feature for each game.

    A divisional matchup is defined as away/home teams sharing the same division, based on
    `constants.TEAM_TO_DIVISION`.

    Args:
        df: DataFrame containing `away_abbr` and `home_abbr`.

    Returns:
        DataFrame with `is_divisional_matchup` added as an Int32 0/1 column.
    """

    required = {"away_abbr", "home_abbr"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"df missing required columns: {missing}")

    division_map = constants.TEAM_TO_DIVISION
    away_div = pl.col("away_abbr").replace_strict(division_map, default=None)
    home_div = pl.col("home_abbr").replace_strict(division_map, default=None)

    return df.with_columns(
        [(away_div == home_div).fill_null(False).cast(pl.Int32).alias("is_divisional_matchup")]
    )


def compute_team_next_week_context(
    schedule_df: pl.DataFrame,
    *,
    season: int,
    week: int,
    include_postseason: bool = False,
) -> pl.DataFrame:
    """Compute per-team next-week opponent context for a given season/week.

    This uses only schedule information (no game results) and is time-safe.

    Args:
        schedule_df: Schedule DataFrame with season/week/game_type/date/away_abbr/home_abbr.
        season: Season year.
        week: Current week number.
        include_postseason: If True, include non-REG schedule rows.

    Returns:
        DataFrame keyed by `team_abbr` with next-week opponent fields.
    """

    required = {"season", "week", "game_type", "date", "away_abbr", "home_abbr"}
    missing = sorted(required - set(schedule_df.columns))
    if missing:
        raise ValueError(f"schedule_df missing required columns: {missing}")

    schedule = schedule_df.filter(pl.col("season") == season)
    if not include_postseason:
        schedule = schedule.filter(pl.col("game_type") == "REG")

    current = schedule.filter(pl.col("week") == week).select(["date", "away_abbr", "home_abbr"])
    nxt = schedule.filter(pl.col("week") == week + 1).select(["date", "away_abbr", "home_abbr"])

    current_team = pl.concat(
        [
            current.select(
                [
                    pl.col("date").alias("current_date"),
                    pl.col("away_abbr").alias("team_abbr"),
                    pl.col("home_abbr").alias("opponent_abbr"),
                    pl.lit(0, dtype=pl.Int32).alias("current_is_home"),
                ]
            ),
            current.select(
                [
                    pl.col("date").alias("current_date"),
                    pl.col("home_abbr").alias("team_abbr"),
                    pl.col("away_abbr").alias("opponent_abbr"),
                    pl.lit(1, dtype=pl.Int32).alias("current_is_home"),
                ]
            ),
        ],
        how="vertical",
    )

    next_team = pl.concat(
        [
            nxt.select(
                [
                    pl.col("date").alias("next_date"),
                    pl.col("away_abbr").alias("team_abbr"),
                    pl.col("home_abbr").alias("next_opponent_abbr"),
                    pl.lit(0, dtype=pl.Int32).alias("next_is_home"),
                ]
            ),
            nxt.select(
                [
                    pl.col("date").alias("next_date"),
                    pl.col("home_abbr").alias("team_abbr"),
                    pl.col("away_abbr").alias("next_opponent_abbr"),
                    pl.lit(1, dtype=pl.Int32).alias("next_is_home"),
                ]
            ),
        ],
        how="vertical",
    )

    joined = current_team.join(next_team, on="team_abbr", how="left")

    division_map = constants.TEAM_TO_DIVISION
    next_is_div = pl.col("team_abbr").replace_strict(division_map, default=None) == pl.col(
        "next_opponent_abbr"
    ).replace_strict(division_map, default=None)

    return joined.with_columns(
        [
            (pl.col("next_date") - pl.col("current_date"))
            .dt.total_days()
            .cast(pl.Int32)
            .alias("days_to_next_game"),
            (pl.col("current_is_home") != pl.col("next_is_home"))
            .cast(pl.Int32)
            .alias("next_location_change"),
            pl.when(pl.col("next_opponent_abbr").is_null())
            .then(pl.lit(None, dtype=pl.Int32))
            .otherwise(next_is_div.fill_null(False).cast(pl.Int32))
            .alias("next_is_divisional_matchup"),
        ]
    ).select(
        [
            "team_abbr",
            "next_opponent_abbr",
            "next_is_home",
            "days_to_next_game",
            "next_location_change",
            "next_is_divisional_matchup",
        ]
    )


def add_lookahead_features(
    games_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
    *,
    season: int,
    week: int,
    include_postseason: bool = False,
) -> pl.DataFrame:
    """Join lookahead/trap-style features (next-week context) onto game rows.

    Next opponent win% is taken from the current week’s pre-game record features present in
    `games_df` (e.g., `away_win_pct` / `home_win_pct`), avoiding any use of future results.

    Missing-data behavior:
    - If schedule context is missing for the week or the next week, columns are added as nulls.

    Args:
        games_df: Game rows for a single week.
        schedule_df: Season schedule.
        season: Season year.
        week: Week number.
        include_postseason: If True, include non-REG schedule rows.

    Returns:
        `games_df` with all `constants.LOOKAHEAD_FEATURE_COLUMNS` present.
    """

    def _expected_dtype(column: str) -> DataTypeClass:
        if column.endswith("_abbr"):
            return pl.Utf8
        if column.endswith("_win_pct"):
            return pl.Float32
        return pl.Int32

    def _ensure_null_cols(df: pl.DataFrame) -> pl.DataFrame:
        exprs: list[pl.Expr] = []
        for col in constants.LOOKAHEAD_FEATURE_COLUMNS:
            dtype = _expected_dtype(col)
            if col in df.columns:
                exprs.append(pl.col(col).cast(dtype, strict=False).alias(col))
            else:
                exprs.append(pl.lit(None, dtype=dtype).alias(col))
        return df.with_columns(exprs)

    try:
        team_context = compute_team_next_week_context(
            schedule_df,
            season=season,
            week=week,
            include_postseason=include_postseason,
        )
    except ValueError:
        return _ensure_null_cols(games_df)

    if team_context.height == 0:
        return _ensure_null_cols(games_df)

    # Build a per-team win% table from the current week’s rows.
    win_pct_rows = []
    if "away_abbr" in games_df.columns and "away_win_pct" in games_df.columns:
        win_pct_rows.append(
            games_df.select(
                pl.col("away_abbr").alias("team_abbr"),
                pl.col("away_win_pct").cast(pl.Float32).alias("win_pct"),
            )
        )
    if "home_abbr" in games_df.columns and "home_win_pct" in games_df.columns:
        win_pct_rows.append(
            games_df.select(
                pl.col("home_abbr").alias("team_abbr"),
                pl.col("home_win_pct").cast(pl.Float32).alias("win_pct"),
            )
        )

    if win_pct_rows:
        team_win_pct = pl.concat(win_pct_rows, how="vertical").unique(subset=["team_abbr"])
        team_context = team_context.join(
            team_win_pct.rename(
                {"team_abbr": "next_opponent_abbr", "win_pct": "next_opponent_win_pct"}
            ),
            on="next_opponent_abbr",
            how="left",
        )
    else:
        team_context = team_context.with_columns(
            pl.lit(None, dtype=pl.Float32).alias("next_opponent_win_pct")
        )

    away_ctx = team_context.rename(
        {
            "team_abbr": "away_abbr",
            "next_opponent_abbr": "away_next_opponent_abbr",
            "next_is_home": "away_next_is_home",
            "days_to_next_game": "away_days_to_next_game",
            "next_location_change": "away_next_location_change",
            "next_is_divisional_matchup": "away_next_is_divisional_matchup",
            "next_opponent_win_pct": "away_next_opponent_win_pct",
        }
    )
    home_ctx = team_context.rename(
        {
            "team_abbr": "home_abbr",
            "next_opponent_abbr": "home_next_opponent_abbr",
            "next_is_home": "home_next_is_home",
            "days_to_next_game": "home_days_to_next_game",
            "next_location_change": "home_next_location_change",
            "next_is_divisional_matchup": "home_next_is_divisional_matchup",
            "next_opponent_win_pct": "home_next_opponent_win_pct",
        }
    )

    out = games_df.join(away_ctx, on="away_abbr", how="left").join(
        home_ctx, on="home_abbr", how="left"
    )
    return _ensure_null_cols(out)


def compute_team_standings_before_week(
    schedule_df: pl.DataFrame,
    *,
    season: int,
    week: int,
    include_postseason: bool = False,
) -> pl.DataFrame:
    """Compute standings-based features strictly before a week.

    This function is time-safe: it only uses games with `week < week` and requires scores.

    Args:
        schedule_df: Schedule DataFrame with season/week/game_type/teams/scores.
        season: Season year.
        week: Week number.
        include_postseason: If True, include non-REG games.

    Returns:
        Per-team standings table with ranks, games-behind, and simple clinch/elimination proxies.
    """

    required = {
        "season",
        "week",
        "game_type",
        "away_abbr",
        "home_abbr",
        "away_score",
        "home_score",
    }
    missing = sorted(required - set(schedule_df.columns))
    if missing:
        raise ValueError(f"schedule_df missing required columns: {missing}")

    # Base records from prior games; then expand to all teams with 0s.
    records = compute_team_records_before_week(
        schedule_df,
        season=season,
        week=week,
        include_postseason=include_postseason,
    )
    all_teams = pl.DataFrame({"team_abbr": constants.TEAM_ABBR})
    records = all_teams.join(records, on="team_abbr", how="left")
    records = records.with_columns(
        [
            pl.col("wins").fill_null(0).cast(pl.Int32),
            pl.col("losses").fill_null(0).cast(pl.Int32),
            pl.col("ties").fill_null(0).cast(pl.Int32),
            pl.col("games_played").fill_null(0).cast(pl.Int32),
            pl.col("win_pct").fill_null(0.0).cast(pl.Float32),
        ]
    )

    records = records.with_columns(
        [
            pl.col("team_abbr")
            .replace_strict(constants.TEAM_TO_DIVISION, default=None)
            .alias("division"),
            pl.col("team_abbr")
            .replace_strict(constants.TEAM_TO_CONFERENCE, default=None)
            .alias("conference"),
        ]
    )

    # Remaining games proxy: use schedule-based total games when available.
    # Note: regular-season weeks are not the same as games played (e.g., 18-week season but 17
    # games per team). We count scheduled REG games per team from the schedule to avoid
    # over/under-stating games remaining.
    expected_games_per_team = 17 if season >= 2021 else 16
    season_schedule = schedule_df.filter(pl.col("season") == season)
    if not include_postseason:
        season_schedule = season_schedule.filter(pl.col("game_type") == "REG")

    scheduled = pl.concat(
        [
            season_schedule.select(pl.col("away_abbr").alias("team_abbr")),
            season_schedule.select(pl.col("home_abbr").alias("team_abbr")),
        ],
        how="vertical",
    )
    scheduled_games = scheduled.group_by("team_abbr").agg(
        pl.len().cast(pl.Int32).alias("season_games_scheduled")
    )

    records = records.join(scheduled_games, on="team_abbr", how="left").with_columns(
        pl.col("season_games_scheduled")
        .fill_null(pl.lit(expected_games_per_team, dtype=pl.Int32))
        .cast(pl.Int32)
    )
    records = records.with_columns(
        (
            pl.when(pl.col("season_games_scheduled") - pl.col("games_played") < 0)
            .then(pl.lit(0, dtype=pl.Int32))
            .otherwise(pl.col("season_games_scheduled") - pl.col("games_played"))
        )
        .cast(pl.Int32)
        .alias("games_remaining")
    ).with_columns((pl.col("wins") + pl.col("games_remaining")).alias("max_wins"))

    # Division rank by win_pct then wins.
    records = records.with_columns(
        pl.struct(["win_pct", "wins"])
        .rank(method="dense", descending=True)
        .over("division")
        .cast(pl.Int32)
        .alias("division_rank")
    )
    records = records.with_columns(
        pl.struct(["win_pct", "wins"])
        .rank(method="dense", descending=True)
        .over("conference")
        .cast(pl.Int32)
        .alias("conference_rank")
    )

    # Division games behind: (leader_w - team_w + team_l - leader_l) / 2
    division_leaders = (
        records.sort(["division", "win_pct", "wins"], descending=[False, True, True])
        .group_by("division")
        .agg(
            [
                pl.col("wins").first().alias("division_leader_wins"),
                pl.col("losses").first().alias("division_leader_losses"),
            ]
        )
    )
    records = records.join(division_leaders, on="division", how="left").with_columns(
        (
            (
                (pl.col("division_leader_wins") - pl.col("wins"))
                + (pl.col("losses") - pl.col("division_leader_losses"))
            )
            / 2.0
        )
        .cast(pl.Float32)
        .alias("division_games_behind")
    )

    # Conference cutoff (seed 7) by win_pct then wins.
    seed7 = (
        records.sort(["conference", "win_pct", "wins"], descending=[False, True, True])
        .with_columns(pl.int_range(1, pl.len() + 1).over("conference").alias("conf_order"))
        .filter(pl.col("conf_order") == 7)
        .select(
            [
                pl.col("conference"),
                pl.col("wins").alias("seed7_wins"),
                pl.col("max_wins").alias("seed7_max_wins"),
            ]
        )
    )
    records = records.join(seed7, on="conference", how="left").with_columns(
        ((pl.col("seed7_wins") - pl.col("wins")).cast(pl.Float32)).alias(
            "conference_games_behind_seed7"
        )
    )

    # Division clinch proxy: clinched when the current division leader has more *current* wins
    # than any other team can possibly reach.
    top2 = records.group_by("division").agg(
        pl.col("max_wins").sort(descending=True).head(2).alias("division_top2_max_wins")
    )
    records = records.join(top2, on="division", how="left").with_columns(
        [
            pl.col("division_top2_max_wins")
            .list.get(0)
            .cast(pl.Int32)
            .alias("division_max_wins_1"),
            pl.col("division_top2_max_wins")
            .list.get(1)
            .cast(pl.Int32)
            .alias("division_max_wins_2"),
        ]
    )
    records = records.with_columns(
        pl.when(pl.col("max_wins") == pl.col("division_max_wins_1"))
        .then(pl.col("division_max_wins_2"))
        .otherwise(pl.col("division_max_wins_1"))
        .fill_null(pl.col("division_max_wins_1"))
        .cast(pl.Int32)
        .alias("division_max_wins_other")
    )
    records = records.with_columns(
        [
            pl.when(
                (pl.col("division_rank") == 1)
                & (pl.col("wins") > pl.col("division_max_wins_other"))
            )
            .then(1)
            .otherwise(0)
            .cast(pl.Int32)
            .alias("division_clinched_proxy"),
            pl.when(pl.col("max_wins") < pl.col("division_leader_wins"))
            .then(1)
            .otherwise(0)
            .cast(pl.Int32)
            .alias("division_eliminated_proxy"),
        ]
    )

    # Conference clinch/elimination proxies relative to seed7 wins.
    # If a team cannot reach the current seed7 wins, mark eliminated.
    records = records.with_columns(
        [
            pl.when(pl.col("seed7_wins").is_null())
            .then(pl.lit(None, dtype=pl.Int32))
            .otherwise((pl.col("max_wins") < pl.col("seed7_wins")).cast(pl.Int32))
            .alias("conference_eliminated_proxy"),
            pl.when(pl.col("seed7_max_wins").is_null())
            .then(pl.lit(None, dtype=pl.Int32))
            .otherwise((pl.col("wins") > pl.col("seed7_max_wins")).cast(pl.Int32))
            .alias("conference_clinched_proxy"),
        ]
    )

    return records.select(
        [
            "team_abbr",
            "division_rank",
            "conference_rank",
            "division_games_behind",
            "conference_games_behind_seed7",
            "division_clinched_proxy",
            "division_eliminated_proxy",
            "conference_clinched_proxy",
            "conference_eliminated_proxy",
        ]
    )


def add_motivation_features(
    games_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
    *,
    season: int,
    week: int,
    include_postseason: bool = False,
) -> pl.DataFrame:
    """Join motivation/standings proxy features onto game rows.

    Missing-data behavior:
    - If schedule schema is incomplete (e.g., missing scores), motivation columns are added as
      nulls.

    Args:
        games_df: Week game rows.
        schedule_df: Season schedule.
        season: Season year.
        week: Week number.
        include_postseason: If True, include non-REG games.

    Returns:
        `games_df` with `constants.MOTIVATION_FEATURE_COLUMNS` present.
    """

    def _ensure_null_cols(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [
                (
                    pl.col(c)
                    if c in df.columns
                    else pl.lit(None, dtype=pl.Float32 if "behind" in c else pl.Int32).alias(c)
                )
                for c in constants.MOTIVATION_FEATURE_COLUMNS
            ]
        )

    try:
        standings = compute_team_standings_before_week(
            schedule_df,
            season=season,
            week=week,
            include_postseason=include_postseason,
        )
    except ValueError:
        return _ensure_null_cols(games_df)

    away = standings.rename(
        {
            "team_abbr": "away_abbr",
            "division_rank": "away_division_rank",
            "conference_rank": "away_conference_rank",
            "division_games_behind": "away_division_games_behind",
            "conference_games_behind_seed7": "away_conference_games_behind_seed7",
            "division_clinched_proxy": "away_division_clinched_proxy",
            "division_eliminated_proxy": "away_division_eliminated_proxy",
            "conference_clinched_proxy": "away_conference_clinched_proxy",
            "conference_eliminated_proxy": "away_conference_eliminated_proxy",
        }
    )
    home = standings.rename(
        {
            "team_abbr": "home_abbr",
            "division_rank": "home_division_rank",
            "conference_rank": "home_conference_rank",
            "division_games_behind": "home_division_games_behind",
            "conference_games_behind_seed7": "home_conference_games_behind_seed7",
            "division_clinched_proxy": "home_division_clinched_proxy",
            "division_eliminated_proxy": "home_division_eliminated_proxy",
            "conference_clinched_proxy": "home_conference_clinched_proxy",
            "conference_eliminated_proxy": "home_conference_eliminated_proxy",
        }
    )

    out = games_df.join(away, on="away_abbr", how="left").join(home, on="home_abbr", how="left")
    return _ensure_null_cols(out)


def add_season_phase_features(
    games_df: pl.DataFrame,
    *,
    season: int,
    week: int,
) -> pl.DataFrame:
    """Add normalized season phase features to game rows.

    The season phase is derived from the regular-season length for the given season.
    Postseason weeks are capped at 1.0 (late season).

    Args:
        games_df: Week game rows.
        season: Season year.
        week: Week number.

    Returns:
        `games_df` with season phase columns added.
    """

    regular_weeks = constants.get_regular_season_weeks(season)
    if regular_weeks <= 0:
        week_norm = 0.0
    else:
        week_norm = min(float(week) / float(regular_weeks), 1.0)

    early = int(week_norm <= (1.0 / 3.0))
    mid = int((1.0 / 3.0) < week_norm <= (2.0 / 3.0))
    late = int(week_norm > (2.0 / 3.0))

    return games_df.with_columns(
        [
            pl.lit(week_norm).cast(pl.Float32).alias("week_in_season_norm"),
            pl.lit(early).cast(pl.Int8).alias("season_phase_early"),
            pl.lit(mid).cast(pl.Int8).alias("season_phase_mid"),
            pl.lit(late).cast(pl.Int8).alias("season_phase_late"),
        ]
    )


def build_team_elo_trends(
    elo_df: pl.DataFrame,
    season: int,
    *,
    window: int = 4,
) -> pl.DataFrame:
    """Compute per-team ELO trends based on prior games.

    The trend is defined as `elo_pre - rolling_mean(elo_pre)` over the prior `window`
    games, with week-1 values defaulting to 0.0 (no trend signal yet).

    Args:
        elo_df: Full ELO DataFrame with away/home columns.
        season: Season year to compute.
        window: Rolling window size in games.

    Returns:
        DataFrame with columns: season, week, team_abbr, elo_4wk_trend.
    """

    schema = {
        "season": pl.Int64,
        "week": pl.Int64,
        "team_abbr": pl.Utf8,
        "elo_4wk_trend": pl.Float64,
    }
    if elo_df.height == 0:
        return pl.DataFrame(schema=schema)

    required = {"season", "week", "away_abbr", "home_abbr", "away_elo_pre", "home_elo_pre"}
    if not required.issubset(set(elo_df.columns)):
        return pl.DataFrame(schema=schema)

    season_elo = elo_df.filter(pl.col("season") == season)
    if season_elo.height == 0:
        return pl.DataFrame(schema=schema)

    away = season_elo.select(
        [
            "season",
            "week",
            pl.col("away_abbr").alias("team_abbr"),
            pl.col("away_elo_pre").alias("elo_pre"),
        ]
    )
    home = season_elo.select(
        [
            "season",
            "week",
            pl.col("home_abbr").alias("team_abbr"),
            pl.col("home_elo_pre").alias("elo_pre"),
        ]
    )
    long_df = pl.concat([away, home], how="vertical").filter(pl.col("elo_pre").is_not_null())
    if long_df.height == 0:
        return pl.DataFrame(schema=schema)

    long_df = long_df.sort(["team_abbr", "week"])
    prior_mean = (
        pl.col("elo_pre").rolling_mean(window_size=window, min_samples=1).shift(1).over("team_abbr")
    )
    trend = (
        pl.when(prior_mean.is_null())
        .then(pl.lit(0.0))
        .otherwise(pl.col("elo_pre") - prior_mean)
        .alias("elo_4wk_trend")
    )

    return long_df.with_columns(trend).select(["season", "week", "team_abbr", "elo_4wk_trend"])


def build_qb_trends(
    elo_df: pl.DataFrame,
    season: int,
    *,
    window: int = 4,
) -> pl.DataFrame:
    """Compute per-QB rolling trends for QB ELO/value.

    Trends are calculated per QB name (not team) using prior games only.

    Args:
        elo_df: Full ELO DataFrame with away/home columns.
        season: Season year to compute.
        window: Rolling window size in games.

    Returns:
        DataFrame with columns: season, week, qb_name, qb_elo_4wk_trend, qb_value_4wk_trend.
    """

    schema = {
        "season": pl.Int64,
        "week": pl.Int64,
        "qb_name": pl.Utf8,
        "qb_elo_4wk_trend": pl.Float64,
        "qb_value_4wk_trend": pl.Float64,
    }
    if elo_df.height == 0:
        return pl.DataFrame(schema=schema)

    required = {"season", "week", "away_qb", "home_qb", "away_qb_elo_pre", "home_qb_elo_pre"}
    if not required.issubset(set(elo_df.columns)):
        return pl.DataFrame(schema=schema)

    season_elo = elo_df.filter(pl.col("season") == season)
    if season_elo.height == 0:
        return pl.DataFrame(schema=schema)

    has_qb_value = "away_qb_value_pre" in season_elo.columns and "home_qb_value_pre" in (
        season_elo.columns
    )
    away_cols = [
        "season",
        "week",
        pl.col("away_qb").alias("qb_name"),
        pl.col("away_qb_elo_pre").alias("qb_elo_pre"),
    ]
    home_cols = [
        "season",
        "week",
        pl.col("home_qb").alias("qb_name"),
        pl.col("home_qb_elo_pre").alias("qb_elo_pre"),
    ]
    if has_qb_value:
        away_cols.append(pl.col("away_qb_value_pre").alias("qb_value_pre"))
        home_cols.append(pl.col("home_qb_value_pre").alias("qb_value_pre"))
    else:
        away_cols.append(pl.lit(None).alias("qb_value_pre"))
        home_cols.append(pl.lit(None).alias("qb_value_pre"))

    away = season_elo.select(away_cols)
    home = season_elo.select(home_cols)
    long_df = pl.concat([away, home], how="vertical").filter(pl.col("qb_name").is_not_null())
    if long_df.height == 0:
        return pl.DataFrame(schema=schema)

    long_df = long_df.sort(["qb_name", "week"])

    exprs = []
    if "qb_elo_pre" in long_df.columns:
        prior_elo = (
            pl.col("qb_elo_pre")
            .rolling_mean(window_size=window, min_samples=1)
            .shift(1)
            .over("qb_name")
        )
        exprs.append(
            pl.when(prior_elo.is_null())
            .then(pl.lit(0.0))
            .otherwise(pl.col("qb_elo_pre") - prior_elo)
            .alias("qb_elo_4wk_trend")
        )

    if "qb_value_pre" in long_df.columns:
        prior_value = (
            pl.col("qb_value_pre")
            .rolling_mean(window_size=window, min_samples=1)
            .shift(1)
            .over("qb_name")
        )
        exprs.append(
            pl.when(prior_value.is_null())
            .then(pl.lit(0.0))
            .otherwise(pl.col("qb_value_pre") - prior_value)
            .alias("qb_value_4wk_trend")
        )

    if not exprs:
        return pl.DataFrame(schema=schema)

    out = long_df.with_columns(exprs).select(
        [
            "season",
            "week",
            "qb_name",
            "qb_elo_4wk_trend",
            "qb_value_4wk_trend",
        ]
    )
    return out


def build_team_stat_trends(
    team_stats_df: pl.DataFrame,
    season: int,
    *,
    stats: Sequence[str],
    window: int = 4,
) -> pl.DataFrame:
    """Compute team-level performance trend features.

    Trend is defined as the prior `window`-game mean minus season-to-date mean
    (both computed strictly before the current week).

    Args:
        team_stats_df: Per-game team stats DataFrame.
        season: Season year to compute.
        stats: Stat column names to compute trends for.
        window: Rolling window size in games.

    Returns:
        DataFrame with columns: season, week, team_abbr, <stat>_4wk_trend.
    """

    schema = {"season": pl.Int64, "week": pl.Int64, "team_abbr": pl.Utf8}
    for stat in stats:
        schema[f"{stat}_4wk_trend"] = pl.Float64

    if team_stats_df.height == 0:
        return pl.DataFrame(schema=schema)

    available_stats = [stat for stat in stats if stat in team_stats_df.columns]
    if not available_stats:
        return pl.DataFrame(schema=schema)

    season_stats = team_stats_df.filter(pl.col("season") == season)
    if season_stats.height == 0:
        return pl.DataFrame(schema=schema)

    season_stats = season_stats.select(["season", "week", "team_abbr", *available_stats]).sort(
        ["team_abbr", "week"]
    )

    exprs = []
    for stat in available_stats:
        shifted = pl.col(stat).shift(1)
        recent_mean = shifted.rolling_mean(window_size=window, min_samples=1).over("team_abbr")
        cum_sum = shifted.cum_sum().over("team_abbr")
        cum_count = shifted.cum_count().over("team_abbr")
        season_mean = pl.when(cum_count > 0).then(cum_sum / cum_count)
        exprs.append(
            pl.when(cum_count == 0)
            .then(pl.lit(0.0))
            .otherwise(recent_mean - season_mean)
            .alias(f"{stat}_4wk_trend")
        )

    return season_stats.with_columns(exprs).select(
        ["season", "week", "team_abbr", *[f"{stat}_4wk_trend" for stat in available_stats]]
    )


def build_coach_features(
    schedule_df: pl.DataFrame,
    *,
    season: int,
) -> pl.DataFrame:
    """Compute per-coach prior win-rate features (career and team-specific)."""

    schema = {
        "season": pl.Int64,
        "week": pl.Int64,
        "team_abbr": pl.Utf8,
        "coach_name": pl.Utf8,
        "coach_games_prior": pl.Int32,
        "coach_win_pct_prior": pl.Float32,
        "coach_team_games_prior": pl.Int32,
        "coach_team_win_pct_prior": pl.Float32,
    }

    required = {
        "season",
        "week",
        "away_abbr",
        "home_abbr",
        "away_score",
        "home_score",
        "away_coach",
        "home_coach",
    }
    if not required.issubset(set(schedule_df.columns)):
        return pl.DataFrame(schema=schema)

    base = schedule_df.filter(pl.col("season") <= season).select(
        [
            "season",
            "week",
            "away_abbr",
            "home_abbr",
            "away_score",
            "home_score",
            "away_coach",
            "home_coach",
        ]
    )
    if base.height == 0:
        return pl.DataFrame(schema=schema)

    base = base.with_columns(
        (pl.col("away_score").is_not_null() & pl.col("home_score").is_not_null())
        .cast(pl.Int32)
        .alias("played")
    )

    home_rows = base.select(
        [
            "season",
            "week",
            pl.col("home_abbr").alias("team_abbr"),
            pl.col("home_coach").alias("coach_name"),
            (pl.col("home_score") > pl.col("away_score"))
            .cast(pl.Int32)
            .mul(pl.col("played"))
            .alias("win"),
            (pl.col("home_score") < pl.col("away_score"))
            .cast(pl.Int32)
            .mul(pl.col("played"))
            .alias("loss"),
            (pl.col("home_score") == pl.col("away_score"))
            .cast(pl.Int32)
            .mul(pl.col("played"))
            .alias("tie"),
            pl.col("played").alias("games"),
        ]
    )
    away_rows = base.select(
        [
            "season",
            "week",
            pl.col("away_abbr").alias("team_abbr"),
            pl.col("away_coach").alias("coach_name"),
            (pl.col("away_score") > pl.col("home_score"))
            .cast(pl.Int32)
            .mul(pl.col("played"))
            .alias("win"),
            (pl.col("away_score") < pl.col("home_score"))
            .cast(pl.Int32)
            .mul(pl.col("played"))
            .alias("loss"),
            (pl.col("away_score") == pl.col("home_score"))
            .cast(pl.Int32)
            .mul(pl.col("played"))
            .alias("tie"),
            pl.col("played").alias("games"),
        ]
    )

    long_df = (
        pl.concat([home_rows, away_rows], how="vertical")
        .filter(pl.col("coach_name").is_not_null())
        .filter(pl.col("team_abbr").is_not_null())
    )
    if long_df.height == 0:
        return pl.DataFrame(schema=schema)

    coach_weekly = (
        long_df.group_by(["coach_name", "season", "week"])
        .agg(
            [
                pl.col("win").sum().cast(pl.Int32).alias("wins"),
                pl.col("loss").sum().cast(pl.Int32).alias("losses"),
                pl.col("tie").sum().cast(pl.Int32).alias("ties"),
                pl.col("games").sum().cast(pl.Int32).alias("games"),
            ]
        )
        .sort(["coach_name", "season", "week"])
        .with_columns(
            [
                pl.col("games")
                .cum_sum()
                .shift(1)
                .over("coach_name")
                .fill_null(0)
                .alias("coach_games_prior"),
                pl.col("wins")
                .cum_sum()
                .shift(1)
                .over("coach_name")
                .fill_null(0)
                .alias("coach_wins_prior"),
                pl.col("ties")
                .cum_sum()
                .shift(1)
                .over("coach_name")
                .fill_null(0)
                .alias("coach_ties_prior"),
            ]
        )
        .with_columns(
            pl.when(pl.col("coach_games_prior") > 0)
            .then(pl.col("coach_wins_prior") / pl.col("coach_games_prior"))
            .otherwise(pl.lit(0.0))
            .cast(pl.Float32)
            .alias("coach_win_pct_prior")
        )
        .select(["coach_name", "season", "week", "coach_games_prior", "coach_win_pct_prior"])
    )

    team_weekly = (
        long_df.group_by(["coach_name", "team_abbr", "season", "week"])
        .agg(
            [
                pl.col("win").sum().cast(pl.Int32).alias("wins"),
                pl.col("loss").sum().cast(pl.Int32).alias("losses"),
                pl.col("tie").sum().cast(pl.Int32).alias("ties"),
                pl.col("games").sum().cast(pl.Int32).alias("games"),
            ]
        )
        .sort(["coach_name", "team_abbr", "season", "week"])
        .with_columns(
            [
                pl.col("games")
                .cum_sum()
                .shift(1)
                .over(["coach_name", "team_abbr"])
                .fill_null(0)
                .alias("coach_team_games_prior"),
                pl.col("wins")
                .cum_sum()
                .shift(1)
                .over(["coach_name", "team_abbr"])
                .fill_null(0)
                .alias("coach_team_wins_prior"),
            ]
        )
        .with_columns(
            pl.when(pl.col("coach_team_games_prior") > 0)
            .then(pl.col("coach_team_wins_prior") / pl.col("coach_team_games_prior"))
            .otherwise(pl.lit(0.0))
            .cast(pl.Float32)
            .alias("coach_team_win_pct_prior")
        )
    )

    combined = team_weekly.join(
        coach_weekly, on=["coach_name", "season", "week"], how="left"
    ).select(
        [
            "season",
            "week",
            "team_abbr",
            "coach_name",
            "coach_games_prior",
            "coach_win_pct_prior",
            "coach_team_games_prior",
            "coach_team_win_pct_prior",
        ]
    )

    return combined.filter(pl.col("season") == season).select(list(schema.keys()))
