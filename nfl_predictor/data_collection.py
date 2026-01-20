"""
Data collection module for NFL game prediction using nflreadpy and Polars.

This module orchestrates the collection, processing, and storage of NFL game data
for use in prediction models. It uses nflreadpy as the primary data source and
Polars for high-performance data manipulation.

Key Features:
    - Collects historical game data from 2006 to present (configurable via `constants.MIN_SEASON`)
    - Includes both regular season (weeks 1-18) and playoff games (WC, DIV, CON, SB)
    - Aggregates per-game team statistics into rolling averages
    - Merges ELO ratings and TeamRankings data for enhanced features
    - Computes statistical differentials between away and home teams
    - Produces separate ML-ready (with diffs) and analysis (without diffs) outputs

Output Files:
    - data/all_data_ml.csv: Full dataset with differential columns for ML
    - data/all_data.csv: Dataset without differential columns for analysis
    - data/completed_games_ml.csv: Only completed games with diffs
    - data/completed_games.csv: Only completed games without diffs
    - data/predict/week_XX_games_to_predict.csv: Upcoming games for prediction

Usage:
    Run directly to collect and process all data:
        python -m nfl_predictor.data_collection

    Optional flags (when run as a script):
        --timing --debug-logs --refresh-nflreadpy --min-season --max-season

    Or import and call programmatically:
        from nfl_predictor.data_collection import collect_all_data
        df = collect_all_data([2023, 2024])
"""

import argparse
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import game_utils, polars_utils
from nfl_predictor.utils.logger import log


def _current_nfl_season(today: date) -> int:
    """Return the current NFL season for a given date."""

    return today.year if today.month > constants.SEASON_END_MONTH else today.year - 1


def _default_max_season(today: Optional[date] = None) -> int:
    """Return the default max season (inclusive) based on today's date."""

    if today is None:
        today = date.today()
    return _current_nfl_season(today)


# Configuration: default season bounds (inclusive)
DEFAULT_MIN_SEASON = constants.MIN_SEASON

# Data collection tuning toggles (overridable via CLI when run as a script).
ENABLE_DATA_COLLECTION_TIMING = False
ENABLE_DATA_COLLECTION_DEBUG = False
FORCE_REFRESH_NFLREADPY = False


@dataclass(frozen=True)
class DataCollectionConfig:
    """Runtime configuration for data collection."""

    enable_timing: bool
    enable_debug: bool
    force_refresh_nflreadpy: bool
    min_season: int
    max_season: int


def _prefix_team_records(records_df: pl.DataFrame, team_side: str) -> pl.DataFrame:
    """Return a record DataFrame with columns prefixed for a specific team side.

    Args:
        records_df: DataFrame returned by `polars_utils.compute_team_records_before_week`.
        team_side: Either "away" or "home".

    Returns:
        DataFrame with `team_abbr` renamed to `{team_side}_abbr` and record columns renamed to
        `{team_side}_<field>`.
    """

    if team_side not in {"away", "home"}:
        raise ValueError(f"team_side must be 'away' or 'home', got: {team_side}")

    prefix = f"{team_side}_"
    base_cols = [
        col[len(prefix) :] for col in constants.RECORD_FEATURE_COLUMNS if col.startswith(prefix)
    ]
    rename_map = {
        "team_abbr": f"{team_side}_abbr",
        **{col: f"{prefix}{col}" for col in base_cols},
    }
    return records_df.rename(rename_map)


def _configure_logging(enable_debug: bool) -> None:
    """Adjust logging verbosity for data collection runs."""

    if not enable_debug:
        return
    log.setLevel(logging.DEBUG)
    for handler in log.handlers:
        handler.setLevel(logging.DEBUG)
    log.debug("Debug logging enabled for data collection.")


def _parse_args(argv: list[str]) -> DataCollectionConfig:
    """Parse CLI args when data collection is run as a script."""

    parser = argparse.ArgumentParser(description="Run nflreadpy data collection.")
    parser.add_argument(
        "--min-season",
        type=int,
        default=None,
        help="Minimum season to include (inclusive).",
    )
    parser.add_argument(
        "--max-season",
        type=int,
        default=None,
        help="Maximum season to include (inclusive).",
    )
    parser.add_argument(
        "--timing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable per-step timing logs.",
    )
    parser.add_argument(
        "--debug-logs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable debug-level logs for data collection.",
    )
    parser.add_argument(
        "--refresh-nflreadpy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force refresh nflreadpy data even when cache exists.",
    )
    args = parser.parse_args(argv)
    default_min = DEFAULT_MIN_SEASON
    default_max = _default_max_season()
    min_season = default_min if args.min_season is None else args.min_season
    max_season = default_max if args.max_season is None else args.max_season

    return DataCollectionConfig(
        enable_timing=ENABLE_DATA_COLLECTION_TIMING if args.timing is None else args.timing,
        enable_debug=ENABLE_DATA_COLLECTION_DEBUG if args.debug_logs is None else args.debug_logs,
        force_refresh_nflreadpy=(
            FORCE_REFRESH_NFLREADPY if args.refresh_nflreadpy is None else args.refresh_nflreadpy
        ),
        min_season=int(min_season),
        max_season=int(max_season),
    )


def _resolve_config(argv: Optional[list[str]]) -> DataCollectionConfig:
    """Resolve data collection config from defaults and optional CLI args."""

    if argv is None:
        return DataCollectionConfig(
            enable_timing=ENABLE_DATA_COLLECTION_TIMING,
            enable_debug=ENABLE_DATA_COLLECTION_DEBUG,
            force_refresh_nflreadpy=FORCE_REFRESH_NFLREADPY,
            min_season=DEFAULT_MIN_SEASON,
            max_season=_default_max_season(),
        )
    return _parse_args(argv)


def _resolve_seasons(min_season: int, max_season: int) -> list[int]:
    """Resolve the list of seasons to process (inclusive bounds)."""

    if max_season < min_season:
        raise ValueError("max_season must be >= min_season.")
    return list(range(min_season, max_season + 1))


@contextmanager
def _timed_step(label: str, enabled: bool) -> Iterator[None]:
    """Time a step and log duration when enabled."""

    if not enabled:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("Timing: %s took %.2fs", label, elapsed)


@contextmanager
def _timed_substep(
    label: str,
    enabled: bool,
    totals: Optional[dict[str, float]],
) -> Iterator[None]:
    """Accumulate timing for sub-steps without per-call logging."""

    if not enabled:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if totals is not None:
            totals[label] = totals.get(label, 0.0) + elapsed


def _log_df_stats(label: str, df: pl.DataFrame, enabled: bool) -> None:
    """Log dataframe shape/columns when debug logging is enabled."""

    if not enabled:
        return
    log.debug("%s: %d rows, %d cols", label, df.height, len(df.columns))


def main(argv: Optional[list[str]] = None) -> None:
    """
    Main entry point for data collection using nflreadpy.

    Orchestrates the data collection, processing, and storage for NFL game predictions.
    """

    config = _resolve_config(argv)
    _configure_logging(config.enable_debug)

    log.info("Starting data collection with nflreadpy...")
    log.info(
        "NFLreadpy cache enabled (historical seasons). Force refresh: %s",
        config.force_refresh_nflreadpy,
    )

    # Determine current season and week
    today = date.today()
    current_season = today.year if today.month > constants.SEASON_END_MONTH else today.year - 1
    current_week = _determine_nfl_week(today)

    log.info("Current season: %s, week: %s", current_season, current_week)

    seasons_to_process = _resolve_seasons(config.min_season, config.max_season)
    log.info(
        "Season range: %s-%s (%d seasons)",
        config.min_season,
        config.max_season,
        len(seasons_to_process),
    )

    with _timed_step("collect_all_data", config.enable_timing):
        all_data_df = collect_all_data(seasons_to_process, config=config)

    _log_df_stats("all_data", all_data_df, config.enable_debug)

    # Create version without diff columns (for non-ML local usage)
    no_diff_df = polars_utils.remove_diff_columns(all_data_df)

    # Save all data (ML version with diffs)
    save_dataframe(all_data_df, "all_data_ml")

    # Save all data (non-ML version without diffs)
    save_dataframe(no_diff_df, "all_data")

    # Filter and save completed games (both versions)
    completed_df = polars_utils.filter_completed_games(all_data_df)
    completed_no_diff_df = polars_utils.remove_diff_columns(completed_df)
    save_dataframe(completed_df, "completed_games_ml")
    save_dataframe(completed_no_diff_df, "completed_games")

    # Filter and save upcoming games for prediction (ML version only)
    upcoming_df = polars_utils.filter_upcoming_games(all_data_df, current_season, current_week)
    save_dataframe(upcoming_df, f"predict/week_{current_week:>02}_games_to_predict")

    log.info("Data collection complete.")


def collect_all_data(
    seasons: list[int],
    *,
    config: Optional[DataCollectionConfig] = None,
) -> pl.DataFrame:
    """
    Collect and combine all data for specified seasons.

    Args:
        seasons: List of season years to process
        config: Optional runtime config for logging/timing and cache refresh

    Returns:
        Combined DataFrame with all game data and features
    """

    if config is None:
        min_season = min(seasons)
        max_season = max(seasons)
        config = DataCollectionConfig(
            enable_timing=ENABLE_DATA_COLLECTION_TIMING,
            enable_debug=ENABLE_DATA_COLLECTION_DEBUG,
            force_refresh_nflreadpy=FORCE_REFRESH_NFLREADPY,
            min_season=min_season,
            max_season=max_season,
        )

    log.info(
        "Collecting data for %d seasons: %s - %s",
        len(seasons),
        min(seasons),
        max(seasons),
    )

    current_season, current_week = polars_utils.get_current_nfl_week()
    log.info("Current season: %d, week: %d (for TR scraping)", current_season, current_week)

    # Load full schedule with lines/odds directly from nflreadpy
    # Includes both regular season (REG) and playoff games (WC, DIV, CON, SB)
    with _timed_step("load_schedule", config.enable_timing):
        schedule_df = polars_utils.load_schedule(
            seasons,
            force_refresh=config.force_refresh_nflreadpy,
            current_season=current_season,
        )
    log.info(
        "Loaded schedule: %d total games (regular season + playoffs)",
        schedule_df.height,
    )
    _log_df_stats("schedule_df", schedule_df, config.enable_debug)

    # Determine seasons to load for team stats
    # Include previous season for week 1 regression if not processing from the beginning
    stats_seasons = list(seasons)
    min_season = min(seasons)
    if min_season > constants.MIN_SEASON - 1:  # Need prior season for week 1 regression
        stats_seasons = [min_season - 1] + stats_seasons

    # Load team statistics (regular season only - used for building features)
    # Playoff games use cumulative stats from the regular season
    with _timed_step("load_team_stats", config.enable_timing):
        team_stats_df = polars_utils.load_team_stats(
            stats_seasons,
            regular_season_only=True,
            force_refresh=config.force_refresh_nflreadpy,
            current_season=current_season,
        )
    log.info("Loaded team stats: %d regular season team-game records", team_stats_df.height)
    _log_df_stats("team_stats_df", team_stats_df, config.enable_debug)

    # Add scoring data (points scored/allowed) to team stats from schedule
    # This enables computing points-related metrics like scoring margin
    # Need to also load schedule for previous season for scoring data
    with _timed_step("add_scoring_data", config.enable_timing):
        if min_season > constants.MIN_SEASON - 1:
            prev_schedule = polars_utils.load_schedule(
                [min_season - 1],
                force_refresh=config.force_refresh_nflreadpy,
                current_season=current_season,
            )
            full_schedule = pl.concat([prev_schedule, schedule_df], how="diagonal")
            team_stats_df = polars_utils.add_scoring_data_to_team_stats(
                team_stats_df, full_schedule
            )
        else:
            team_stats_df = polars_utils.add_scoring_data_to_team_stats(team_stats_df, schedule_df)
    _log_df_stats("team_stats_with_scores", team_stats_df, config.enable_debug)

    # Add per-game opponent stats AFTER scoring data is added
    # This ensures opponent_points_scored, opponent_points_allowed, etc. are included
    with _timed_step("add_per_game_opponent_stats", config.enable_timing):
        team_stats_df = polars_utils.add_per_game_opponent_stats(team_stats_df)
    _log_df_stats("team_stats_with_opponents", team_stats_df, config.enable_debug)

    # Load ELO ratings
    with _timed_step("load_elo_ratings", config.enable_timing):
        elo_df = polars_utils.load_elo_ratings(seasons)
    if elo_df.height > 0:
        log.info("Loaded ELO ratings: %d game records", elo_df.height)
    else:
        log.warning("No ELO ratings loaded")
    _log_df_stats("elo_df", elo_df, config.enable_debug)

    # Load raw ELO data for QB lookups (needed for fill_future_qb_data)
    with _timed_step("load_raw_elo_data", config.enable_timing):
        raw_elo_df = polars_utils.load_raw_elo_data()

    min_season = min(seasons)

    # Process each season
    all_seasons_data = []

    for season in seasons:
        log.info("Processing season %d...", season)

        # Load TeamRankings for this season (will scrape if current/future week)
        with _timed_step(f"load_team_rankings_{season}", config.enable_timing):
            tr_df = polars_utils.load_team_rankings(season, current_season, current_week)

        # Load previous season's TeamRankings for week 1 regression
        prev_tr_df = None
        if season > min_season:
            prev_tr_df = polars_utils.load_team_rankings(season - 1, current_season, current_week)

        with _timed_step(f"process_season_{season}", config.enable_timing):
            season_data = process_season(
                season,
                schedule_df,
                team_stats_df,
                min_season=min_season,
                timing_enabled=config.enable_timing,
                elo_df=elo_df,
                tr_df=tr_df,
                prev_tr_df=prev_tr_df,
            )
        if season_data.height > 0:
            all_seasons_data.append(season_data)

    # Combine all seasons
    if all_seasons_data:
        combined_df = pl.concat(all_seasons_data, how="diagonal")
        # Sort by date, newest first
        if "date" in combined_df.columns:
            combined_df = combined_df.sort("date", descending=True)
        # Remove exact duplicate games if any exist (defensive cleanup)
        if "game_id" in combined_df.columns:
            combined_df = combined_df.unique(subset=["game_id"], keep="first")
        else:
            combined_df = combined_df.unique(
                subset=["season", "week", "away_abbr", "home_abbr"],
                keep="first",
            )
        # Fill in QB data for future games using most recent starters
        combined_df = game_utils.fill_future_qb_data(combined_df, raw_elo_df)
        # Fill in lines for future games from SurvivorGrid
        combined_df = game_utils.fill_future_game_lines(combined_df)
        # Fill missing moneylines by calculating from spreads
        combined_df = game_utils.fill_missing_moneylines(combined_df)
        # Select final columns in correct order
        combined_df = polars_utils.select_final_columns(combined_df)
        # Ensure final ordering by date after any dedupe/transforms
        if "date" in combined_df.columns:
            combined_df = combined_df.sort("date", descending=True)
        return combined_df

    return pl.DataFrame()


def process_season(
    season: int,
    schedule_df: pl.DataFrame,
    team_stats_df: pl.DataFrame,
    *,
    min_season: int,
    timing_enabled: bool = False,
    elo_df: Optional[pl.DataFrame] = None,
    tr_df: Optional[pl.DataFrame] = None,
    prev_tr_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Process a single season's data.

    Args:
        season: Season year to process
        schedule_df: Full schedule DataFrame
        team_stats_df: Full team stats DataFrame
        min_season: Earliest season included in this run
        timing_enabled: Whether to accumulate per-season timing summaries
        elo_df: ELO ratings DataFrame
        tr_df: TeamRankings DataFrame for this season
        prev_tr_df: TeamRankings DataFrame for previous season (for week 1)

    Returns:
        Processed DataFrame for the season
    """

    # Filter to this season
    season_schedule = schedule_df.filter(pl.col("season") == season)

    if season_schedule.height == 0:
        log.warning("No schedule data for season %d", season)
        return pl.DataFrame()

    # Get unique weeks in the schedule
    weeks = sorted(season_schedule.select("week").unique().to_series().to_list())

    # Process each week
    weekly_data = []
    timing_totals: Optional[dict[str, float]] = {} if timing_enabled else None
    for week in weeks:
        week_data = process_week(
            season,
            week,
            season_schedule,
            team_stats_df,
            min_season=min_season,
            timing_enabled=timing_enabled,
            timing_totals=timing_totals,
            elo_df=elo_df,
            tr_df=tr_df,
            prev_tr_df=prev_tr_df,
        )
        if week_data.height > 0:
            weekly_data.append(week_data)

    if timing_enabled and timing_totals:
        summary = ", ".join(
            f"{label}={timing_totals[label]:.2f}s"
            for label in sorted(
                timing_totals,
                key=lambda label: timing_totals[label],
                reverse=True,
            )
        )
        log.info("Timing summary season %d: %s", season, summary)

    if weekly_data:
        return pl.concat(weekly_data, how="diagonal")

    return pl.DataFrame()


def process_week(
    season: int,
    week: int,
    schedule_df: pl.DataFrame,
    team_stats_df: pl.DataFrame,
    *,
    min_season: int,
    timing_enabled: bool = False,
    timing_totals: Optional[dict[str, float]] = None,
    elo_df: Optional[pl.DataFrame] = None,
    tr_df: Optional[pl.DataFrame] = None,
    prev_tr_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Process a single week's games with aggregated stats from prior weeks.

    For teams that have no prior games in the current season (e.g., Week 1, or
    teams whose games were postponed like MIA/TB in 2017), we fall back to using
    the previous season's stats with regression to mean.

    Args:
        season: Season year
        week: Week number
        schedule_df: Season schedule DataFrame
        team_stats_df: Full team stats DataFrame (all seasons for week-1 lookback)
        min_season: Earliest season included in this run
        timing_enabled: Whether to accumulate timing totals
        timing_totals: Optional dict to accumulate timing totals
        elo_df: ELO ratings DataFrame
        tr_df: TeamRankings DataFrame for this season
        prev_tr_df: TeamRankings DataFrame for previous season (for week 1)

    Returns:
        DataFrame with week's games and features
    """

    # Skip the very first week of the first processed season (no prior data to aggregate)
    if season == min_season and week == 1:
        log.info(
            "Skipping season %d week %d (no prior games to build features)",
            season,
            week,
        )
        return pl.DataFrame()

    # Get this week's games
    week_games = schedule_df.filter((pl.col("season") == season) & (pl.col("week") == week))

    if week_games.height == 0:
        return pl.DataFrame()

    # Get season-specific stats
    season_stats = team_stats_df.filter(pl.col("season") == season)

    # Aggregate stats from prior weeks
    with _timed_substep("aggregate_team_stats", timing_enabled, timing_totals):
        agg_stats = polars_utils.aggregate_team_stats_to_week(season_stats, week, season)

    # Get teams playing this week
    teams_this_week = set(
        week_games.select("away_abbr").to_series().to_list()
        + week_games.select("home_abbr").to_series().to_list()
    )

    # Check which teams have prior stats
    teams_with_stats = set()
    if agg_stats.height > 0 and "team_abbr" in agg_stats.columns:
        teams_with_stats = set(agg_stats.select("team_abbr").to_series().to_list())

    # Find teams that need fallback to previous season
    teams_needing_fallback = teams_this_week - teams_with_stats

    # If any teams need fallback (week 1, or teams with postponed first games like MIA/TB 2017)
    if teams_needing_fallback and season > min_season:
        prev_season_stats = team_stats_df.filter(pl.col("season") == season - 1)

        if prev_season_stats.height > 0:
            # Use full previous season for teams that need fallback
            # Use a high week number to get all regular season games
            prev_agg = polars_utils.aggregate_team_stats_to_week(prev_season_stats, 99, season - 1)

            if prev_agg.height > 0:
                # Calculate league means from previous season
                league_means = polars_utils.calculate_league_means(team_stats_df, season - 1)

                # Regress toward league mean
                prev_agg = polars_utils.regress_to_mean(
                    prev_agg,
                    league_means,
                    constants.WEEK1_REGRESSION_FACTOR,
                )

                # Filter to only teams that need fallback
                fallback_stats = prev_agg.filter(
                    pl.col("team_abbr").is_in(list(teams_needing_fallback))
                )

                if fallback_stats.height > 0:
                    if agg_stats.height > 0:
                        # Combine current season stats with fallback stats
                        agg_stats = pl.concat([agg_stats, fallback_stats])
                    else:
                        agg_stats = fallback_stats

    if agg_stats.height == 0:
        log.debug("No aggregated stats available for season %d week %d", season, week)
        return pl.DataFrame()

    # Merge schedule with aggregated team stats
    with _timed_substep("merge_team_stats", timing_enabled, timing_totals):
        merged = polars_utils.merge_schedule_with_team_stats(week_games, agg_stats)

    # Season-to-date W-L-T record features (time-safe: strictly before this week)
    records_df = pl.DataFrame()
    with _timed_substep("record_features", timing_enabled, timing_totals):
        try:
            records_df = polars_utils.compute_team_records_before_week(
                schedule_df,
                season=season,
                week=week,
                include_postseason=False,
            )
        except ValueError:
            # Some unit tests use a minimal schedule fixture without scores/game_type.
            log.debug(
                "Skipping record feature computation for season %d week %d (schedule incomplete)",
                season,
                week,
            )

    if records_df.height > 0:
        away_records = _prefix_team_records(records_df, "away")
        home_records = _prefix_team_records(records_df, "home")
        merged = merged.join(away_records, on="away_abbr", how="left").join(
            home_records, on="home_abbr", how="left"
        )

    # Week 1 (and edge cases) may have no record rows; ensure columns exist and fill with 0.
    merged = merged.with_columns(
        [
            (
                pl.col(c)
                .fill_null(0.0 if c.endswith("_win_pct") else 0)
                .cast(pl.Float32 if c.endswith("_win_pct") else pl.Int32)
                if c in merged.columns
                else pl.lit(
                    0.0 if c.endswith("_win_pct") else 0,
                    dtype=pl.Float32 if c.endswith("_win_pct") else pl.Int32,
                ).alias(c)
            )
            for c in constants.RECORD_FEATURE_COLUMNS
        ]
    )

    # Merge with ELO ratings
    if elo_df is not None and elo_df.height > 0:
        with _timed_substep("merge_elo", timing_enabled, timing_totals):
            season_elo = elo_df.filter(pl.col("season") == season)
            if "week" in season_elo.columns:
                week_elo = season_elo.filter(pl.col("week") == week)
                if week_elo.height > 0:
                    # Exact week match - drop season/week from ELO before merge
                    elo_cols = [c for c in week_elo.columns if c not in ["season", "week"]]
                    week_elo = week_elo.select(elo_cols)
                    merged = merged.join(
                        week_elo,
                        on=["away_abbr", "home_abbr"],
                        how="left",
                    )
                else:
                    # No ELO for this specific week - use most recent ELO per team
                    # This handles future weeks and playoff games
                    latest_elo = polars_utils.get_latest_elo_by_team(elo_df, season)
                    if latest_elo.height > 0:
                        # Join for away team
                        away_elo = latest_elo.rename(
                            {
                                "team_abbr": "away_abbr",
                                "elo_pre": "away_elo_pre",
                                "qb_value_pre": "away_qb_value_pre",
                                "qb_elo_pre": "away_qb_elo_pre",
                            }
                        )
                        merged = merged.join(away_elo, on="away_abbr", how="left")

                        # Join for home team
                        home_elo = latest_elo.rename(
                            {
                                "team_abbr": "home_abbr",
                                "elo_pre": "home_elo_pre",
                                "qb_value_pre": "home_qb_value_pre",
                                "qb_elo_pre": "home_qb_elo_pre",
                            }
                        )
                        merged = merged.join(home_elo, on="home_abbr", how="left")

    # Merge with TeamRankings
    with _timed_substep("merge_team_rankings", timing_enabled, timing_totals):
        merged = _merge_team_rankings(merged, season, week, tr_df, prev_tr_df)

    # Divisional rivalry feature
    with _timed_substep("add_divisional_feature", timing_enabled, timing_totals):
        merged = polars_utils.add_divisional_matchup_feature(merged)

    # Lookahead / next-week context features (null when schedule context is unavailable)
    with _timed_substep("add_lookahead_features", timing_enabled, timing_totals):
        try:
            merged = polars_utils.add_lookahead_features(
                merged,
                schedule_df,
                season=season,
                week=week,
                include_postseason=False,
            )
        except ValueError:
            log.debug(
                "Skipping lookahead features for season %d week %d (schedule incomplete)",
                season,
                week,
            )

    # Motivation / standings proxy features (null when schedule results are unavailable)
    with _timed_substep("add_motivation_features", timing_enabled, timing_totals):
        try:
            merged = polars_utils.add_motivation_features(
                merged,
                schedule_df,
                season=season,
                week=week,
                include_postseason=False,
            )
        except ValueError:
            log.debug(
                "Skipping motivation features for season %d week %d (schedule incomplete)",
                season,
                week,
            )

    # Calculate stat differentials
    with _timed_substep("calculate_differentials", timing_enabled, timing_totals):
        stats_to_diff = polars_utils.get_stats_for_diff()
        merged = polars_utils.calculate_stat_differentials(merged, stats_to_diff)

    return merged


def _merge_team_rankings(
    merged: pl.DataFrame,
    season: int,
    week: int,
    tr_df: Optional[pl.DataFrame],
    prev_tr_df: Optional[pl.DataFrame],
) -> pl.DataFrame:
    """
    Merge TeamRankings data into the game DataFrame.

    Handles three scenarios:
    1. Regular season (week 2+): Use current season's TR for that week
    2. Week 1: Use previous season's final TR values
    3. Playoffs (week > regular season weeks): Use TR data for that specific playoff week
       (which should be freshly scraped during the playoff week)

    Note: Before 2021, playoffs started in week 18 (17-week season).
          From 2021 onwards, playoffs start in week 19 (18-week season).

    Args:
        merged: Game DataFrame to merge TR data into
        season: Season year (used to determine regular season length)
        week: Week number
        tr_df: TeamRankings DataFrame for current season
        prev_tr_df: TeamRankings DataFrame for previous season

    Returns:
        DataFrame with TR columns merged
    """

    tr_to_use = None
    regular_season_weeks = constants.get_regular_season_weeks(season)

    # Determine which TR data to use based on week
    if week == 1 and prev_tr_df is not None and prev_tr_df.height > 0:
        # Week 1: Use previous season's final TR values
        tr_to_use = polars_utils.get_latest_team_rankings(prev_tr_df)
    elif tr_df is not None and tr_df.height > 0 and "week" in tr_df.columns:
        # Try to get specific week's TR data (works for regular season AND playoffs)
        week_tr = tr_df.filter(pl.col("week") == week)
        if week_tr.height > 0:
            # Drop week column since we're joining on team only
            tr_to_use = week_tr.drop("week")
        elif week > regular_season_weeks:
            # Fallback for playoffs: use most recent available TR data
            tr_to_use = polars_utils.get_latest_team_rankings(tr_df)

    # Merge TR data if available
    if tr_to_use is not None and tr_to_use.height > 0:
        # Only include the TR columns we actually want (ratings + stats)
        expected_tr_cols = set(polars_utils.get_tr_columns())
        available_tr_cols = [c for c in tr_to_use.columns if c in expected_tr_cols]

        if not available_tr_cols:
            log.warning(
                "TR data has no expected columns. Available: %s",
                tr_to_use.columns[:5],
            )
            return merged

        # Select only the columns we want plus team_abbr
        tr_to_use = tr_to_use.select(["team_abbr"] + available_tr_cols)

        # Join for away team
        away_tr = tr_to_use.rename(
            {c: f"away_{c}" for c in tr_to_use.columns if c != "team_abbr"}
        ).rename({"team_abbr": "away_abbr"})
        merged = merged.join(away_tr, on="away_abbr", how="left")

        # Join for home team
        home_tr = tr_to_use.rename(
            {c: f"home_{c}" for c in tr_to_use.columns if c != "team_abbr"}
        ).rename({"team_abbr": "home_abbr"})
        merged = merged.join(home_tr, on="home_abbr", how="left")

    return merged


def save_dataframe(df: pl.DataFrame, name: str) -> None:
    """
    Save a Polars DataFrame to CSV.

    Args:
        df: DataFrame to save
        name: Base name for the file (without extension)
    """

    file_path = f"{constants.DATA_PATH}/{name}.csv"

    # Create directory if needed
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save to CSV
    df.write_csv(file_path)
    log.info("Saved %s (%d rows) to %s", name, df.height, file_path)


def load_dataframe(name: str) -> Optional[pl.DataFrame]:
    """
    Load a Polars DataFrame from CSV.

    Args:
        name: Base name for the file (without extension)

    Returns:
        DataFrame or None if file doesn't exist
    """

    file_path = f"{constants.DATA_PATH}/{name}.csv"

    if not os.path.isfile(file_path):
        log.warning("File not found: %s", file_path)
        return None

    return pl.read_csv(file_path)


def _determine_nfl_week(given_date: date) -> int:
    """
    Determine the current NFL week for a given date.

    Args:
        given_date: Date to check

    Returns:
        Week number (1..regular season weeks + playoff weeks).

        For in-season dates in January/February, this can return playoff weeks
        (e.g., 19-22 for seasons with an 18-week regular season).
    """

    def get_season_start(year: int) -> date:
        sept_first = date(year, 9, 1)
        first_monday = sept_first + timedelta((7 - sept_first.weekday()) % 7)
        return first_monday + timedelta(days=3)

    def adjust_to_tuesday(start_date: date) -> date:
        return start_date - timedelta(days=(start_date.weekday() - 1) % 7)

    season_start = adjust_to_tuesday(get_season_start(given_date.year))

    current_season = (
        given_date.year if given_date.month > constants.SEASON_END_MONTH else given_date.year - 1
    )
    max_week = constants.get_regular_season_weeks(current_season) + 4

    if given_date < season_start:
        if given_date.month in (1, 2):
            previous_season_start = adjust_to_tuesday(get_season_start(given_date.year - 1))
            week_number = ((given_date - previous_season_start).days // 7) + 1
            return max(1, min(week_number, max_week))
        previous_season_start = adjust_to_tuesday(get_season_start(given_date.year - 1))
        if given_date >= previous_season_start:
            return 1
        return 0

    week_number = ((given_date - season_start).days // 7) + 1
    return max(1, min(week_number, max_week))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
