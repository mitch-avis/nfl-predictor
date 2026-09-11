"""Data collection module for NFL game prediction using nflreadpy and Polars.

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

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import game_utils, polars_utils
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars import schedule_strength, strength_snapshot


def _current_nfl_season(today: date) -> int:
    """Return the current NFL season for a given date."""
    return today.year if today.month > constants.SEASON_END_MONTH else today.year - 1


def _default_max_season(today: date | None = None) -> int:
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
    # Set to False to ablate the early-season strength prior and publish the raw
    # in-season solve, so the blend can be measured on its own.
    blend_strength_prior: bool = True
    # Set to False to ablate the early-season blend of season-to-date stats toward the
    # regressed previous season, restoring the plain in-season mean from week 2 on.
    blend_stat_prior: bool = True
    stat_prior_blend_games: float = constants.PRIOR_BLEND_GAMES


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
    parser.add_argument(
        "--strength-prior-blend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Blend the previous season's final adjusted-strength snapshot into early-season "
            "weeks. Use --no-strength-prior-blend to publish the raw in-season solve instead."
        ),
    )
    parser.add_argument(
        "--stat-prior-blend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Blend season-to-date stats toward the regressed previous season, weighting the "
            "in-season sample games / (games + K). Use --no-stat-prior-blend to publish the "
            "plain in-season mean from week 2 on."
        ),
    )
    parser.add_argument(
        "--stat-prior-blend-games",
        type=float,
        default=constants.PRIOR_BLEND_GAMES,
        help=(
            "K for the season-to-date stat blend: the game count at which the in-season "
            f"sample and the prior are weighted equally (default {constants.PRIOR_BLEND_GAMES:g})."
        ),
    )
    args = parser.parse_args(argv)
    if args.stat_prior_blend_games <= 0:
        parser.error("--stat-prior-blend-games must be positive.")
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
        blend_strength_prior=bool(args.strength_prior_blend),
        blend_stat_prior=bool(args.stat_prior_blend),
        stat_prior_blend_games=float(args.stat_prior_blend_games),
    )


def _resolve_config(argv: list[str] | None) -> DataCollectionConfig:
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
    if min_season < constants.NFLREADPY_MIN_SEASON:
        raise ValueError(
            f"min_season must be >= {constants.NFLREADPY_MIN_SEASON} "
            "(nflreadpy data availability starts in 1999)."
        )
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
    totals: dict[str, float] | None,
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


def main(argv: list[str] | None = None) -> None:
    """Run the nflreadpy-backed data collection pipeline.

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


def _log_pbp_null_rates(team_stats_df: pl.DataFrame, enable_debug: bool) -> None:
    """Log the per-season null rate of the play-by-play count columns.

    Args:
        team_stats_df: Team stats after the play-by-play join
        enable_debug: Whether debug diagnostics are enabled

    """
    if not enable_debug or team_stats_df.height == 0:
        return
    if "offensive_snaps" not in team_stats_df.columns:
        return

    per_season = (
        team_stats_df.group_by("season")
        .agg(pl.col("offensive_snaps").is_null().mean().alias("null_rate"))
        .sort("season")
    )
    for row in per_season.iter_rows(named=True):
        log.debug(
            "Play-by-play null rate for season %s: %.4f",
            row["season"],
            row["null_rate"],
        )


# Context flag carried alongside the play-by-play counts; see the join docstring below.
_PBP_HOME_COLUMN = "is_home"


def _join_pbp_team_game_stats(
    team_stats_df: pl.DataFrame,
    pbp_team_games: pl.DataFrame,
) -> pl.DataFrame:
    """Attach per-team-game play-by-play counts to the team stats frame.

    The counts join on `(season, week, team_abbr)`; `opponent_abbr` is dropped from the
    play-by-play side because team stats already carry it. The `is_home` context flag rides
    along with the counts: season-to-date aggregation drops it because it is not numeric, so
    it never reaches the published schema, but the opponent-adjusted solves read it here.
    Teams without play-by-play for a game keep nulls, and when no play-by-play is available
    at all every count column and `is_home` are still added as nulls so the downstream
    schema stays invariant. The join is guaranteed
    not to change the row count: duplicate team-week keys are collapsed with a warning
    rather than multiplying the team-stats frame.

    Args:
        team_stats_df: Per-game team statistics
        pbp_team_games: Per-team-game play-by-play counts, possibly empty

    Returns:
        Team stats with the play-by-play count columns attached

    """
    join_keys = ["season", "week", "team_abbr"]

    if pbp_team_games.height == 0:
        log.warning("No play-by-play team-game rows available; emitting null count columns.")
        empty_columns: list[pl.Expr] = [
            pl.lit(None, dtype=pl.Float64).alias(col)
            for col in constants.PBP_COUNT_COLUMNS
            if col not in team_stats_df.columns
        ]
        # `is_home` is a context flag rather than a count, so it is not in
        # PBP_COUNT_COLUMNS and needs its own null fill to keep the schema invariant.
        if _PBP_HOME_COLUMN not in team_stats_df.columns:
            empty_columns.append(pl.lit(None, dtype=pl.Boolean).alias(_PBP_HOME_COLUMN))
        return team_stats_df.with_columns(empty_columns)

    countable = [col for col in pbp_team_games.columns if col not in {*join_keys, "opponent_abbr"}]
    lookup = pbp_team_games.select([*join_keys, *countable])

    # A duplicate team-week key would multiply team-stat rows and silently corrupt every
    # downstream season-to-date mean, so collapse duplicates and say so loudly.
    deduped = lookup.unique(subset=join_keys, keep="first")
    if deduped.height != lookup.height:
        log.warning(
            "Play-by-play produced %d duplicate team-week keys; keeping the first of each.",
            lookup.height - deduped.height,
        )

    merged = team_stats_df.join(deduped, on=join_keys, how="left")

    missing: list[pl.Expr] = [
        pl.lit(None, dtype=pl.Float64).alias(col)
        for col in constants.PBP_COUNT_COLUMNS
        if col not in merged.columns
    ]
    if _PBP_HOME_COLUMN not in merged.columns:
        missing.append(pl.lit(None, dtype=pl.Boolean).alias(_PBP_HOME_COLUMN))
    if missing:
        merged = merged.with_columns(missing)

    return merged


def collect_all_data(
    seasons: list[int],
    *,
    config: DataCollectionConfig | None = None,
) -> pl.DataFrame:
    """Collect and combine all data for specified seasons.

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
    if min_season > constants.NFLREADPY_MIN_SEASON:  # Need prior season for week 1 regression
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

    # Load play-by-play and attach per-team-game counts before any downstream enrichment.
    # Uses the same season window as team stats so the week-1 previous-season fallback is
    # covered, and degrades to null columns when the source is unavailable.
    with _timed_step("load_pbp", config.enable_timing):
        pbp_df = polars_utils.load_pbp(
            stats_seasons,
            force_refresh=config.force_refresh_nflreadpy,
            current_season=current_season,
        )
    log.info("Loaded play-by-play: %d regular season plays", pbp_df.height)

    with _timed_step("aggregate_pbp_team_game_stats", config.enable_timing):
        pbp_team_games = polars_utils.aggregate_pbp_team_game_stats(pbp_df)
        team_stats_df = _join_pbp_team_game_stats(team_stats_df, pbp_team_games)
    log.info("Aggregated play-by-play: %d team-game records", pbp_team_games.height)
    _log_pbp_null_rates(team_stats_df, config.enable_debug)
    _log_df_stats("team_stats_with_pbp", team_stats_df, config.enable_debug)

    # Add scoring data (points scored/allowed) to team stats from schedule
    # This enables computing points-related metrics like scoring margin
    # Need to also load schedule for previous season for scoring data
    with _timed_step("add_scoring_data", config.enable_timing):
        if min_season > constants.NFLREADPY_MIN_SEASON:
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
    team_rankings_cache: dict[int, pl.DataFrame] = {}

    def _load_team_rankings_cached(season: int) -> pl.DataFrame:
        """Load TeamRankings data once per season to avoid redundant scraping."""
        cached_df = team_rankings_cache.get(season)
        if cached_df is not None:
            return cached_df
        if season < constants.TEAMRANKINGS_MIN_SEASON:
            log.info(
                "Skipping TeamRankings for season %d (data starts in %d).",
                season,
                constants.TEAMRANKINGS_MIN_SEASON,
            )
            team_rankings_cache[season] = pl.DataFrame()
            return team_rankings_cache[season]

        min_week = 1
        if season == constants.TEAMRANKINGS_MIN_SEASON:
            min_week = max(min_week, constants.TEAMRANKINGS_MIN_WEEK)
        if season == min_season and season < current_season:
            min_week = max(min_week, constants.TEAMRANKINGS_MIN_WEEK)
        with _timed_step(f"load_team_rankings_{season}", config.enable_timing):
            tr_df = polars_utils.load_team_rankings(
                season,
                current_season,
                current_week,
                min_week=min_week,
            )
        team_rankings_cache[season] = tr_df
        return tr_df

    for season in seasons:
        log.info("Processing season %d...", season)

        # Load TeamRankings for this season (will scrape if current/future week)
        tr_df = _load_team_rankings_cached(season)

        # Load previous season's TeamRankings for week 1 regression
        prev_tr_df = None
        if season > min_season:
            prev_tr_df = _load_team_rankings_cached(season - 1)

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
                blend_strength_prior=config.blend_strength_prior,
                blend_stat_prior=config.blend_stat_prior,
                stat_prior_blend_games=config.stat_prior_blend_games,
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


# Per-game components of the one-hop schedule-strength margin. The published
# `epa_margin_per_play` is a difference of two per-side rates with different
# denominators, so it is not expressible as a single ratio of sums. These two columns
# are, which is what the opponent profiling below needs: net EPA over every play the
# team was involved in, offense and defense pooled.
_STRENGTH_MARGIN_NUMERATOR = "_strength_epa_margin_sum"
_STRENGTH_MARGIN_DENOMINATOR = "_strength_total_plays"

_STRENGTH_MARGIN_SOURCES = (
    "pass_epa_sum",
    "rush_epa_sum",
    "pass_epa_allowed_sum",
    "rush_epa_allowed_sum",
    "offensive_snaps",
    "defensive_snaps",
)


def _schedule_teams(schedule_df: pl.DataFrame, season: int) -> list[str]:
    """Return every team on the season's schedule, sorted.

    The schedule is published before kickoff, so this is the team universe even when no
    game has been played yet. Taking it from here rather than from played games is what
    lets a Week-1 row carry the regressed prior in production instead of nulls.
    """
    sides = [side for side in ("away_abbr", "home_abbr") if side in schedule_df.columns]
    if not sides or "season" not in schedule_df.columns:
        return []
    season_rows = schedule_df.filter(pl.col("season") == season)
    teams: set[str] = set()
    for side in sides:
        teams.update(season_rows.get_column(side).drop_nulls().cast(pl.String).to_list())
    return sorted(teams)


def _regular_season_schedule(schedule_df: pl.DataFrame, season: int) -> pl.DataFrame:
    """Return only the season's regular-season rows.

    The postseason bracket is a result of the season, not schedule context known in
    advance, so it must not reach any pre-week feature. Filtering by week rather than by
    `game_type` keeps this working on the minimal schedule frames used in tests, which do
    not always carry a game-type column.
    """
    if "week" not in schedule_df.columns:
        return schedule_df
    return schedule_df.filter(pl.col("week") <= constants.get_regular_season_weeks(season))


def _with_strength_margin_components(team_stats_df: pl.DataFrame) -> pl.DataFrame:
    """Attach the net-EPA numerator and play-count denominator used by schedule strength.

    Formulas:
        numerator   = (pass_epa_sum + rush_epa_sum)
                      - (pass_epa_allowed_sum + rush_epa_allowed_sum)
        denominator = offensive_snaps + defensive_snaps

    Returns the frame unchanged when any source column is missing, so a season without
    play-by-play still flows through and simply yields a null schedule strength.
    """
    if any(column not in team_stats_df.columns for column in _STRENGTH_MARGIN_SOURCES):
        return team_stats_df

    def value(column: str) -> pl.Expr:
        return pl.col(column).cast(pl.Float64, strict=False)

    return team_stats_df.with_columns(
        (
            (value("pass_epa_sum") + value("rush_epa_sum"))
            - (value("pass_epa_allowed_sum") + value("rush_epa_allowed_sum"))
        ).alias(_STRENGTH_MARGIN_NUMERATOR),
        (value("offensive_snaps") + value("defensive_snaps")).alias(_STRENGTH_MARGIN_DENOMINATOR),
    )


def build_prior_strength_snapshot(
    team_stats_df: pl.DataFrame,
    season: int,
    *,
    min_season: int,
) -> pl.DataFrame | None:
    """Return the previous season's final strength snapshot, or None when unavailable.

    "Final" means the whole regular season, which is what the playoff branch of the
    snapshot builder returns for any week past the regular season.
    """
    if season <= min_season or team_stats_df.height == 0:
        return None
    previous = season - 1
    if team_stats_df.filter(pl.col("season") == previous).height == 0:
        return None
    snapshot = strength_snapshot.build_strength_snapshot(
        team_stats_df,
        season=previous,
        week=constants.get_regular_season_weeks(previous) + 1,
        blend_prior=False,
    )
    return snapshot if snapshot.height > 0 else None


def build_prior_season_stats(
    team_stats_df: pl.DataFrame,
    season: int,
    *,
    min_season: int,
) -> pl.DataFrame | None:
    """Return the previous regular season's per-game stats regressed toward the league mean.

    Formula, per aggregated column:
        ``regressed = team_mean * (1 - WEEK1_REGRESSION_FACTOR)
        + league_mean * WEEK1_REGRESSION_FACTOR``

    Derived ratios are then recomputed from the regressed components. This frame is both
    the Week-1 fallback and the prior the season-to-date blend leans on in early weeks.

    Returns:
        One row per team, or None for the first season in the run or when the previous
        season has no rows.

    """
    if season <= min_season or team_stats_df.height == 0:
        return None
    previous = season - 1
    previous_stats = team_stats_df.filter(pl.col("season") == previous)
    if previous_stats.height == 0:
        return None
    # A target week past the regular season selects every regular-season game.
    prior = polars_utils.aggregate_team_stats_to_week(previous_stats, 99, previous)
    if prior.height == 0:
        return None
    league_means = polars_utils.calculate_league_means(team_stats_df, previous)
    prior = polars_utils.regress_to_mean(prior, league_means, constants.WEEK1_REGRESSION_FACTOR)
    # Regression rewrites the summed components, so the ratios derived from them are
    # recomputed against the regressed sums.
    return polars_utils.recompute_derived_metrics(prior)


def build_strength_features(
    team_stats_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
    *,
    season: int,
    week: int,
    prior_snapshot: pl.DataFrame | None = None,
    blend_prior: bool = True,
) -> pl.DataFrame:
    """Build every published schedule-adjusted strength column for one season week.

    Combines the pre-week ridge snapshot with the two schedule-strength lenses: the
    ridge-based mean of opponents' pre-week composite, and the one-hop companion that
    profiles each faced opponent from its other games only.

    Both lenses are restricted to the regular season. That matters for the
    games-remaining side: the regular-season schedule is fixed before kickoff and so is
    legitimately known, but *which* postseason games a team will play, and against whom,
    is an outcome of the very season being predicted. Letting the bracket into a
    week-`N` feature would leak the season's result backwards into it.

    Args:
        team_stats_df: Per-game team stats carrying the play-by-play sums, `is_home`
            and the scoring columns. Rows outside the requested window are filtered
            downstream, so the full history may be passed.
        schedule_df: The season's schedule, used for the games-remaining lens.
        season: Season being processed.
        week: Week being processed; every value is solved from earlier games only.
        prior_snapshot: Previous season's final snapshot for the early-season blend.
        blend_prior: Set to False to ablate the prior and publish the raw in-season solve.

    Returns:
        One row per team with `team_abbr` and `constants.ADJUSTED_STRENGTH_STATS`.

    """
    snapshot = strength_snapshot.build_strength_snapshot(
        team_stats_df,
        season=season,
        week=week,
        prior_snapshot=prior_snapshot,
        blend_prior=blend_prior,
        teams=_schedule_teams(schedule_df, season),
    )
    if snapshot.height == 0:
        return pl.DataFrame(
            schema={
                "team_abbr": pl.String,
                **dict.fromkeys(constants.ADJUSTED_STRENGTH_STATS, pl.Float64),
            }
        )

    features = snapshot.select("team_abbr", *constants.STRENGTH_TEAM_STATS)

    ratings = snapshot.select("team_abbr", "adj_strength_composite")
    try:
        adjusted = schedule_strength.compute_schedule_strength_adjusted(
            _regular_season_schedule(schedule_df, season),
            ratings,
            season=season,
            week=week,
            rating_col="adj_strength_composite",
            team_col="team_abbr",
        )
        features = features.join(adjusted, on="team_abbr", how="left")
    except ValueError as error:
        log.warning(
            "Skipping adjusted schedule strength for season %d week %d: %s",
            season,
            week,
            error,
        )

    prepared = _with_strength_margin_components(team_stats_df)
    if _STRENGTH_MARGIN_NUMERATOR in prepared.columns:
        raw = schedule_strength.compute_schedule_strength_raw(
            prepared,
            season=season,
            week=week,
            team_col="team_abbr",
            opponent_col="opponent_abbr",
            numerator_col=_STRENGTH_MARGIN_NUMERATOR,
            denominator_col=_STRENGTH_MARGIN_DENOMINATOR,
        )
        features = features.join(raw, on="team_abbr", how="left")

    missing = [
        pl.lit(None, dtype=pl.Float64).alias(column)
        for column in constants.ADJUSTED_STRENGTH_STATS
        if column not in features.columns
    ]
    if missing:
        features = features.with_columns(missing)

    return features.select("team_abbr", *constants.ADJUSTED_STRENGTH_STATS)


def _merge_strength_features(
    merged: pl.DataFrame,
    features: pl.DataFrame,
) -> pl.DataFrame:
    """Join the per-team strength columns onto a week's games as away_/home_ pairs.

    Every published column is added on both sides even when the join finds nothing, so
    the output schema stays invariant across seasons and the model sees nulls rather
    than a missing column.

    A duplicate team key on the feature side would multiply this week's games instead of
    annotating them, silently corrupting every downstream row, so duplicates are collapsed
    with a warning rather than joined.
    """
    if features.height > 0:
        deduped = features.unique(subset=["team_abbr"], keep="first")
        if deduped.height != features.height:
            log.warning(
                "Strength features produced %d duplicate team keys; keeping the first of each.",
                features.height - deduped.height,
            )
        features = deduped

    row_count = merged.height
    for side in ("away", "home"):
        renamed = features.rename(
            {"team_abbr": f"{side}_abbr"}
            | {column: f"{side}_{column}" for column in constants.ADJUSTED_STRENGTH_STATS}
        )
        merged = merged.join(renamed, on=f"{side}_abbr", how="left")

    if merged.height != row_count:
        raise ValueError(
            f"Strength feature join changed the row count from {row_count} to {merged.height}"
        )

    return merged.with_columns(
        [
            pl.lit(None, dtype=pl.Float64).alias(f"{side}_{column}")
            for side in ("away", "home")
            for column in constants.ADJUSTED_STRENGTH_STATS
            if f"{side}_{column}" not in merged.columns
        ]
    )


def process_season(
    season: int,
    schedule_df: pl.DataFrame,
    team_stats_df: pl.DataFrame,
    *,
    min_season: int,
    timing_enabled: bool = False,
    elo_df: pl.DataFrame | None = None,
    tr_df: pl.DataFrame | None = None,
    prev_tr_df: pl.DataFrame | None = None,
    blend_strength_prior: bool = True,
    blend_stat_prior: bool = True,
    stat_prior_blend_games: float = constants.PRIOR_BLEND_GAMES,
) -> pl.DataFrame:
    """Process a single season's data.

    Args:
        season: Season year to process
        schedule_df: Full schedule DataFrame
        team_stats_df: Full team stats DataFrame
        min_season: Earliest season included in this run
        timing_enabled: Whether to accumulate per-season timing summaries
        elo_df: ELO ratings DataFrame
        tr_df: TeamRankings DataFrame for this season
        prev_tr_df: TeamRankings DataFrame for previous season (for week 1)
        blend_strength_prior: Set to False to ablate the strength prior blend
        blend_stat_prior: Set to False to ablate the season-to-date stat prior blend
        stat_prior_blend_games: K in the stat blend weight ``games / (games + K)``

    Returns:
        Processed DataFrame for the season

    """
    # Filter to this season
    season_schedule = schedule_df.filter(pl.col("season") == season)

    if season_schedule.height == 0:
        log.warning("No schedule data for season %d", season)
        return pl.DataFrame()

    # Precompute trend features for the season (time-safe, prior weeks only)
    team_elo_trends = pl.DataFrame()
    qb_trends = pl.DataFrame()
    team_stat_trends = pl.DataFrame()
    coach_features = pl.DataFrame()
    if elo_df is not None and elo_df.height > 0:
        team_elo_trends = polars_utils.build_team_elo_trends(elo_df, season)
        qb_trends = polars_utils.build_qb_trends(elo_df, season)
    if team_stats_df.height > 0:
        team_stat_trends = polars_utils.build_team_stat_trends(
            team_stats_df,
            season,
            stats=["scoring_margin", "turnover_margin"],
        )
    if schedule_df.height > 0:
        coach_features = polars_utils.build_coach_features(schedule_df, season=season)

    # Solved once per season rather than per week: it depends only on the prior season.
    prior_strength_snapshot = (
        build_prior_strength_snapshot(team_stats_df, season, min_season=min_season)
        if blend_strength_prior
        else None
    )
    # Also built once per season: the Week-1 fallback and the stat blend's prior.
    prior_season_stats = build_prior_season_stats(team_stats_df, season, min_season=min_season)

    # Get unique weeks in the schedule
    weeks = sorted(season_schedule.select("week").unique().to_series().to_list())

    # Process each week
    weekly_data = []
    timing_totals: dict[str, float] | None = {} if timing_enabled else None
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
            team_elo_trends=team_elo_trends,
            qb_trends=qb_trends,
            team_stat_trends=team_stat_trends,
            coach_features=coach_features,
            prior_strength_snapshot=prior_strength_snapshot,
            blend_strength_prior=blend_strength_prior,
            prior_season_stats=prior_season_stats,
            blend_stat_prior=blend_stat_prior,
            stat_prior_blend_games=stat_prior_blend_games,
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
    timing_totals: dict[str, float] | None = None,
    elo_df: pl.DataFrame | None = None,
    tr_df: pl.DataFrame | None = None,
    prev_tr_df: pl.DataFrame | None = None,
    team_elo_trends: pl.DataFrame | None = None,
    qb_trends: pl.DataFrame | None = None,
    team_stat_trends: pl.DataFrame | None = None,
    coach_features: pl.DataFrame | None = None,
    prior_strength_snapshot: pl.DataFrame | None = None,
    blend_strength_prior: bool = True,
    prior_season_stats: pl.DataFrame | None = None,
    blend_stat_prior: bool = True,
    stat_prior_blend_games: float = constants.PRIOR_BLEND_GAMES,
) -> pl.DataFrame:
    """Process a single week's games with aggregated stats from prior weeks.

    For teams that have no prior games in the current season (e.g., Week 1, or
    teams whose games were postponed like MIA/TB in 2017), we fall back to using
    the previous season's stats with regression to mean.

    Teams that have played are blended toward that same regressed prior season:
    ``weight = games / (games + stat_prior_blend_games)`` and
    ``blended = weight * in_season_mean + (1 - weight) * regressed_prior_mean``,
    with every derived ratio recomputed from the blended sums. A team with zero games
    has weight zero, so the Week-1 fallback is the limit of the same blend.

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
        team_elo_trends: Optional rolling ELO trend features for the current season
        qb_trends: Optional rolling quarterback trend features for the current season
        team_stat_trends: Optional rolling team-stat trend features for the current season
        coach_features: Optional per-team coach feature DataFrame
        prior_strength_snapshot: Optional previous-season final strength snapshot used
            by the early-season prior blend. Computed here when not supplied.
        blend_strength_prior: Set to False to ablate the strength prior blend
        prior_season_stats: Optional regressed previous-season stats from
            `build_prior_season_stats`. Computed here when not supplied.
        blend_stat_prior: Set to False to ablate the season-to-date stat prior blend
        stat_prior_blend_games: K in the stat blend weight ``games / (games + K)``

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

    # Teams that have played lean on the regressed prior season in early weeks; teams
    # that have not (week 1, or postponed first games like MIA/TB 2017) use it outright.
    blend_played_teams = blend_stat_prior and agg_stats.height > 0
    if (teams_needing_fallback or blend_played_teams) and season > min_season:
        if prior_season_stats is None:
            prior_season_stats = build_prior_season_stats(
                team_stats_df, season, min_season=min_season
            )

        if prior_season_stats is not None:
            if blend_played_teams:
                with _timed_substep("blend_stat_prior", timing_enabled, timing_totals):
                    agg_stats = polars_utils.blend_with_prior_stats(
                        agg_stats, prior_season_stats, stat_prior_blend_games
                    )

            fallback_stats = prior_season_stats.filter(
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

    # Season phase features (normalized week + early/mid/late buckets)
    merged = polars_utils.add_season_phase_features(merged, season=season, week=week)

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

    # Merge trend features derived from ELO/QB history and team stats
    merged = _merge_team_trends(merged, team_elo_trends)
    merged = _merge_team_trends(merged, team_stat_trends)
    merged = _merge_qb_trends(merged, qb_trends)
    merged = _merge_coach_features(merged, coach_features)

    # Merge with TeamRankings
    with _timed_substep("merge_team_rankings", timing_enabled, timing_totals):
        merged = _merge_team_rankings(merged, season, week, tr_df, prev_tr_df)

    # TeamRankings trend: last-5 vs last-10 rating
    if {
        "away_last_5_games_rating",
        "away_last_10_games_rating",
        "home_last_5_games_rating",
        "home_last_10_games_rating",
    }.issubset(merged.columns):
        merged = merged.with_columns(
            [
                (pl.col("away_last_5_games_rating") - pl.col("away_last_10_games_rating")).alias(
                    "away_last_5_games_rating_trend"
                ),
                (pl.col("home_last_5_games_rating") - pl.col("home_last_10_games_rating")).alias(
                    "home_last_5_games_rating_trend"
                ),
            ]
        )

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

    # Schedule-adjusted team strength, solved from games strictly before this week
    with _timed_substep("strength_features", timing_enabled, timing_totals):
        if prior_strength_snapshot is None and blend_strength_prior:
            prior_strength_snapshot = build_prior_strength_snapshot(
                team_stats_df, season, min_season=min_season
            )
        strength_features = build_strength_features(
            team_stats_df,
            schedule_df,
            season=season,
            week=week,
            prior_snapshot=prior_strength_snapshot,
            blend_prior=blend_strength_prior,
        )
        merged = _merge_strength_features(merged, strength_features)

    # Calculate stat differentials
    with _timed_substep("calculate_differentials", timing_enabled, timing_totals):
        stats_to_diff = polars_utils.get_stats_for_diff()
        merged = polars_utils.calculate_stat_differentials(merged, stats_to_diff)

    # Fill missing trend features with neutral defaults
    trend_cols = []
    for base in constants.TREND_FEATURE_COLUMNS:
        trend_cols.extend([f"away_{base}", f"home_{base}", f"{base}_diff"])
    merged = merged.with_columns(
        [pl.col(col).fill_null(0.0).cast(pl.Float32) for col in trend_cols if col in merged.columns]
    )

    coach_int_cols = [
        "away_coach_games_prior",
        "home_coach_games_prior",
        "away_coach_team_games_prior",
        "home_coach_team_games_prior",
    ]
    coach_float_cols = [
        "away_coach_win_pct_prior",
        "home_coach_win_pct_prior",
        "away_coach_team_win_pct_prior",
        "home_coach_team_win_pct_prior",
    ]
    merged = merged.with_columns(
        [
            *[
                pl.col(col).fill_null(0).cast(pl.Int32)
                for col in coach_int_cols
                if col in merged.columns
            ],
            *[
                pl.col(col).fill_null(0.0).cast(pl.Float32)
                for col in coach_float_cols
                if col in merged.columns
            ],
        ]
    )

    return merged


def _merge_team_trends(
    merged: pl.DataFrame,
    trend_df: pl.DataFrame | None,
) -> pl.DataFrame:
    """Merge per-team trend features for away/home teams."""
    if trend_df is None or trend_df.height == 0:
        return merged

    required = {"season", "week", "team_abbr"}
    if not required.issubset(trend_df.columns):
        return merged

    value_cols = [c for c in trend_df.columns if c not in required]
    if not value_cols:
        return merged

    away_map = {"team_abbr": "away_abbr", **{c: f"away_{c}" for c in value_cols}}
    home_map = {"team_abbr": "home_abbr", **{c: f"home_{c}" for c in value_cols}}

    away_trends = trend_df.rename(away_map)
    home_trends = trend_df.rename(home_map)

    merged = merged.join(away_trends, on=["season", "week", "away_abbr"], how="left")
    merged = merged.join(home_trends, on=["season", "week", "home_abbr"], how="left")
    return merged


def _merge_qb_trends(
    merged: pl.DataFrame,
    trend_df: pl.DataFrame | None,
) -> pl.DataFrame:
    """Merge per-QB trend features for away/home QBs."""
    if trend_df is None or trend_df.height == 0:
        return merged

    required = {"season", "week", "qb_name"}
    if not required.issubset(trend_df.columns):
        return merged

    if "away_qb" not in merged.columns or "home_qb" not in merged.columns:
        return merged

    value_cols = [c for c in trend_df.columns if c not in required]
    if not value_cols:
        return merged

    away_map = {"qb_name": "away_qb", **{c: f"away_{c}" for c in value_cols}}
    home_map = {"qb_name": "home_qb", **{c: f"home_{c}" for c in value_cols}}

    away_trends = trend_df.rename(away_map)
    home_trends = trend_df.rename(home_map)

    merged = merged.join(away_trends, on=["season", "week", "away_qb"], how="left")
    merged = merged.join(home_trends, on=["season", "week", "home_qb"], how="left")
    return merged


def _merge_coach_features(
    merged: pl.DataFrame,
    coach_df: pl.DataFrame | None,
) -> pl.DataFrame:
    """Merge per-coach features for away/home teams."""
    if coach_df is None or coach_df.height == 0:
        return merged

    required = {"season", "week", "team_abbr"}
    if not required.issubset(coach_df.columns):
        return merged

    drop_cols = {"season", "week", "team_abbr", "coach_name"}
    value_cols = [c for c in coach_df.columns if c not in drop_cols]
    if not value_cols:
        return merged

    coach_df = coach_df.select(["season", "week", "team_abbr", *value_cols])
    away_map = {"team_abbr": "away_abbr", **{c: f"away_{c}" for c in value_cols}}
    home_map = {"team_abbr": "home_abbr", **{c: f"home_{c}" for c in value_cols}}

    away_features = coach_df.rename(away_map)
    home_features = coach_df.rename(home_map)

    merged = merged.join(away_features, on=["season", "week", "away_abbr"], how="left")
    merged = merged.join(home_features, on=["season", "week", "home_abbr"], how="left")
    return merged


def _merge_team_rankings(
    merged: pl.DataFrame,
    season: int,
    week: int,
    tr_df: pl.DataFrame | None,
    prev_tr_df: pl.DataFrame | None,
) -> pl.DataFrame:
    """Merge TeamRankings data into the game DataFrame.

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
    """Save a Polars DataFrame to CSV.

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


def load_dataframe(name: str) -> pl.DataFrame | None:
    """Load a Polars DataFrame from CSV.

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
    """Determine the current NFL week for a given date.

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
