"""Polars-based data loading and merge helpers.

These functions load and prepare schedule/team stats and related data sources.
Implementation was split out of `nfl_predictor.utils.polars_utils`.
"""

import os
from pathlib import Path
from typing import Optional

import nflreadpy as nfl
import polars as pl
from polars.datatypes import DataType

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.scraping_utils import get_current_nfl_week, normalize_team_column

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


def _is_numeric_dtype(dtype: DataType) -> bool:
    """Return True if dtype is numeric."""

    return isinstance(dtype, pl.Decimal) or dtype in NUMERIC_DTYPES


def _resolve_cache_dir(cache_dir: Optional[os.PathLike | str]) -> Path:
    """Resolve the nflreadpy cache directory and ensure it exists."""

    resolved = Path(cache_dir) if cache_dir is not None else Path(constants.NFLREADPY_CACHE_DIR)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_current_season(current_season: Optional[int]) -> int:
    """Resolve the current NFL season for cache decisions."""

    if current_season is not None:
        return int(current_season)
    season, _week = get_current_nfl_week()
    return int(season)


def _schedule_cache_path(cache_dir: Path, season: int) -> Path:
    """Build the cache path for a season schedule."""

    return cache_dir / f"schedule_{season}.parquet"


def _team_stats_cache_path(cache_dir: Path, season: int, regular_season_only: bool) -> Path:
    """Build the cache path for season team stats."""

    suffix = "reg" if regular_season_only else "all"
    return cache_dir / f"team_stats_{season}_{suffix}.parquet"


def _read_cached_frame(path: Path) -> Optional[pl.DataFrame]:
    """Read a cached parquet file if it exists."""

    if not path.exists():
        return None
    try:
        return pl.read_parquet(path)
    except (OSError, pl.exceptions.ComputeError, pl.exceptions.NoDataError) as exc:
        log.warning("Failed to read nflreadpy cache file %s: %s", path, exc)
        return None


def _write_cached_frame(df: pl.DataFrame, path: Path) -> None:
    """Write a cached parquet file, logging any failures."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
    except (OSError, pl.exceptions.ComputeError) as exc:
        log.warning("Failed to write nflreadpy cache file %s: %s", path, exc)


def _prepare_schedule(schedule_df: pl.DataFrame) -> pl.DataFrame:
    """Normalize raw nflreadpy schedule data to the project schema."""

    # Select only the columns we need (if they exist)
    available_cols = set(schedule_df.columns)
    cols_to_select = [c for c in constants.NFLREADPY_SCHEDULE_COLUMNS if c in available_cols]
    schedule_df = schedule_df.select(cols_to_select)

    # Rename columns to match our internal naming
    rename_mapping = {
        k: v for k, v in constants.NFLREADPY_SCHEDULE_RENAME.items() if k in schedule_df.columns
    }
    schedule_df = schedule_df.rename(rename_mapping)

    # Normalize kickoff time columns.
    # nflreadpy schedule schemas vary a bit across versions; coalesce to `gametime`.
    time_candidates = [
        c
        for c in ("gametime", "game_time", "kickoff_time", "start_time")
        if c in schedule_df.columns
    ]
    if time_candidates:
        # Prefer an existing `gametime` column when present.
        exprs = [pl.col(c).cast(pl.Utf8) for c in time_candidates]
        schedule_df = schedule_df.with_columns(pl.coalesce(exprs).alias("gametime"))
        # Drop alternate raw time columns to avoid schema clutter.
        drop_cols = [c for c in time_candidates if c != "gametime"]
        if drop_cols:
            schedule_df = schedule_df.drop(drop_cols)

    # Normalize team abbreviations
    if "away_abbr" in schedule_df.columns:
        schedule_df = normalize_team_column(schedule_df, "away_abbr")
    if "home_abbr" in schedule_df.columns:
        schedule_df = normalize_team_column(schedule_df, "home_abbr")

    # Transform neutral column: "Home" -> 0, "Neutral" -> 1
    if "neutral" in schedule_df.columns:
        schedule_df = schedule_df.with_columns(
            pl.when(pl.col("neutral") == "Neutral")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("neutral")
        )

    # Calculate home_spread from away_spread (nflreadpy spread_line is away perspective)
    if "away_spread" in schedule_df.columns:
        schedule_df = schedule_df.with_columns((-pl.col("away_spread")).alias("home_spread"))

    # Parse date column
    if "date" in schedule_df.columns:
        schedule_df = schedule_df.with_columns(pl.col("date").str.to_date("%Y-%m-%d").alias("date"))

    # Optional: combine date + gametime into a sortable datetime.
    # We keep `gametime` as the raw string and add `game_datetime` when parsing succeeds.
    if "date" in schedule_df.columns and "gametime" in schedule_df.columns:
        dt_str = pl.concat_str([pl.col("date").cast(pl.Utf8), pl.col("gametime")], separator=" ")
        dt_24 = dt_str.str.strptime(pl.Datetime, "%Y-%m-%d %H:%M", strict=False)
        dt_ampm = dt_str.str.strptime(pl.Datetime, "%Y-%m-%d %I:%M%p", strict=False)
        dt_ampm_sp = dt_str.str.strptime(pl.Datetime, "%Y-%m-%d %I:%M %p", strict=False)
        schedule_df = schedule_df.with_columns(
            pl.coalesce([dt_24, dt_ampm, dt_ampm_sp]).alias("game_datetime")
        )

    # Add stadium city and state from stadium_id
    if "stadium_id" in schedule_df.columns:
        schedule_df = _add_stadium_location(schedule_df)

    return schedule_df


def _prepare_team_stats(team_stats_df: pl.DataFrame, regular_season_only: bool) -> pl.DataFrame:
    """Normalize raw nflreadpy team stats to the project schema."""

    # Filter to regular season only (exclude preseason and postseason)
    if regular_season_only and "season_type" in team_stats_df.columns:
        team_stats_df = team_stats_df.filter(pl.col("season_type") == "REG")
        log.debug("Filtered to regular season only: %d records", team_stats_df.height)

    # Normalize team abbreviations
    if "team" in team_stats_df.columns:
        team_stats_df = normalize_team_column(team_stats_df, "team")
        team_stats_df = team_stats_df.rename({"team": "team_abbr"})

    if "opponent_team" in team_stats_df.columns:
        team_stats_df = normalize_team_column(team_stats_df, "opponent_team")
        team_stats_df = team_stats_df.rename({"opponent_team": "opponent_abbr"})

    # Rename columns using our mapping
    available_cols = set(team_stats_df.columns)
    rename_mapping = {
        k: v for k, v in constants.NFLREADPY_TEAM_STATS_MAPPING.items() if k in available_cols
    }
    team_stats_df = team_stats_df.rename(rename_mapping)

    # Combine stats as specified
    team_stats_df = combine_stats(team_stats_df)

    return team_stats_df


def load_schedule(
    seasons: list[int],
    *,
    cache_dir: Optional[os.PathLike | str] = None,
    force_refresh: bool = False,
    current_season: Optional[int] = None,
) -> pl.DataFrame:
    """
    Load NFL schedule data for specified seasons using nflreadpy.

    Cached schedules are used for historical seasons when available. Current and future
    seasons are always refreshed to keep upcoming games up to date.

    Args:
        seasons: List of season years to load
        cache_dir: Optional cache directory override for nflreadpy outputs
        force_refresh: If True, refresh schedules even when cache exists
        current_season: Optional current season override for cache decisions

    Returns:
        Polars DataFrame with schedule data including lines/odds
    """

    if not seasons:
        return pl.DataFrame()

    resolved_cache_dir = _resolve_cache_dir(cache_dir)
    resolved_current_season = _resolve_current_season(current_season)

    log.info("Loading schedule for seasons: %s", seasons)

    schedule_frames: list[pl.DataFrame] = []
    for season in seasons:
        cache_path = _schedule_cache_path(resolved_cache_dir, season)
        use_cache = (season < resolved_current_season) and not force_refresh
        cached = _read_cached_frame(cache_path) if use_cache else None
        if cached is not None:
            log.info(
                "Using cached nflreadpy schedule for season %d from %s",
                season,
                cache_path,
            )
            schedule_frames.append(cached)
            continue

        if season < resolved_current_season and not force_refresh:
            log.info("Schedule cache miss for season %d; loading via nflreadpy.", season)
        else:
            log.info("Refreshing schedule via nflreadpy for season %d.", season)

        season_df = nfl.load_schedules(seasons=[season])
        season_df = _prepare_schedule(season_df)
        _write_cached_frame(season_df, cache_path)
        schedule_frames.append(season_df)

    return pl.concat(schedule_frames, how="diagonal")


def _add_stadium_location(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add stadium city and state columns based on stadium_id.

    Uses the STADIUM_LOCATIONS mapping in constants to look up
    city and state for each stadium.

    Args:
        df: DataFrame with stadium_id column

    Returns:
        DataFrame with stadium_city and stadium_state columns added
    """

    # Create mapping dictionaries for city and state
    city_map = {k: v["city"] for k, v in constants.STADIUM_LOCATIONS.items()}
    state_map = {k: v["state"] for k, v in constants.STADIUM_LOCATIONS.items()}

    # Add city and state columns using replace_strict (Polars >=1.0)
    df = df.with_columns(
        [
            pl.col("stadium_id").replace_strict(city_map, default=None).alias("stadium_city"),
            pl.col("stadium_id").replace_strict(state_map, default=None).alias("stadium_state"),
        ]
    )

    return df


def load_team_stats(
    seasons: list[int],
    regular_season_only: bool = True,
    *,
    cache_dir: Optional[os.PathLike | str] = None,
    force_refresh: bool = False,
    current_season: Optional[int] = None,
) -> pl.DataFrame:
    """
    Load team statistics for specified seasons using nflreadpy.

    Cached stats are used for historical seasons when available. Current and future
    seasons are always refreshed to keep upcoming games up to date.

    Args:
        seasons: List of season years to load
        regular_season_only: If True, filter to only regular season games
        cache_dir: Optional cache directory override for nflreadpy outputs
        force_refresh: If True, refresh stats even when cache exists
        current_season: Optional current season override for cache decisions

    Returns:
        Polars DataFrame with team statistics per game
    """

    if not seasons:
        return pl.DataFrame()

    resolved_cache_dir = _resolve_cache_dir(cache_dir)
    resolved_current_season = _resolve_current_season(current_season)

    log.info("Loading team stats for seasons: %s", seasons)

    team_frames: list[pl.DataFrame] = []
    for season in seasons:
        cache_path = _team_stats_cache_path(resolved_cache_dir, season, regular_season_only)
        use_cache = (season < resolved_current_season) and not force_refresh
        cached = _read_cached_frame(cache_path) if use_cache else None
        if cached is not None:
            log.info(
                "Using cached nflreadpy team stats for season %d from %s",
                season,
                cache_path,
            )
            team_frames.append(cached)
            continue

        if season < resolved_current_season and not force_refresh:
            log.info("Team stats cache miss for season %d; loading via nflreadpy.", season)
        else:
            log.info("Refreshing team stats via nflreadpy for season %d.", season)

        season_df = nfl.load_team_stats(seasons=[season])
        season_df = _prepare_team_stats(season_df, regular_season_only=regular_season_only)
        _write_cached_frame(season_df, cache_path)
        team_frames.append(season_df)

    return pl.concat(team_frames, how="diagonal")


def add_scoring_data_to_team_stats(
    team_stats_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Add points scored and points allowed from schedule to team stats.

    The schedule has per-game scores (away_score, home_score). This function
    extracts that into per-team format and merges with team_stats so we can
    compute scoring-related metrics.

    Args:
        team_stats_df: Per-team per-game stats DataFrame
        schedule_df: Schedule DataFrame with scores

    Returns:
        team_stats_df with points_scored and points_allowed columns added
    """

    if schedule_df.height == 0:
        return team_stats_df

    # Extract scoring for away teams
    away_scores = schedule_df.select(
        [
            pl.col("season"),
            pl.col("week"),
            pl.col("away_abbr").alias("team_abbr"),
            pl.col("away_score").alias("points_scored"),
            pl.col("home_score").alias("points_allowed"),
        ]
    )

    # Extract scoring for home teams
    home_scores = schedule_df.select(
        [
            pl.col("season"),
            pl.col("week"),
            pl.col("home_abbr").alias("team_abbr"),
            pl.col("home_score").alias("points_scored"),
            pl.col("away_score").alias("points_allowed"),
        ]
    )

    # Combine into per-team scoring
    all_scores = pl.concat([away_scores, home_scores])

    # Compute scoring margin
    all_scores = all_scores.with_columns(
        [
            (pl.col("points_scored") - pl.col("points_allowed")).alias("scoring_margin"),
        ]
    )

    # Merge with team_stats
    result = team_stats_df.join(
        all_scores,
        on=["season", "week", "team_abbr"],
        how="left",
    )

    return result


def combine_stats(df: pl.DataFrame) -> pl.DataFrame:
    """
    Combine related stats into single columns and remove originals.

    Combines:
    - sack + rushing + receiving fumbles -> fumbles
    - sack + rushing + receiving fumbles_lost -> fumbles_lost
    - passing + rushing + receiving first_downs -> first_downs
    - passing + rushing + receiving 2pt_conversions -> 2pt_conversions
    - fumble_recovery_own + fumble_recovery_opp -> fumble_recoveries

    Args:
        df: DataFrame with raw stats

    Returns:
        DataFrame with combined stats
    """

    combine_operations = []

    # Fumbles (sack + rushing + receiving)
    fumble_cols = ["sack_fumbles", "rushing_fumbles", "receiving_fumbles"]
    if all(c in df.columns for c in fumble_cols):
        combine_operations.append(
            (
                pl.col("sack_fumbles") + pl.col("rushing_fumbles") + pl.col("receiving_fumbles")
            ).alias("fumbles")
        )

    # Fumbles lost (sack + rushing + receiving)
    fumble_lost_cols = [
        "sack_fumbles_lost",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
    ]
    if all(c in df.columns for c in fumble_lost_cols):
        combine_operations.append(
            (
                pl.col("sack_fumbles_lost")
                + pl.col("rushing_fumbles_lost")
                + pl.col("receiving_fumbles_lost")
            ).alias("fumbles_lost")
        )

    # First downs (passing + rushing only - receiving first downs overlap with passing)
    first_down_cols = ["passing_first_downs", "rushing_first_downs"]
    if all(c in df.columns for c in first_down_cols):
        combine_operations.append(
            (pl.col("passing_first_downs") + pl.col("rushing_first_downs")).alias("first_downs")
        )

    # 2pt conversions (passing + rushing + receiving)
    two_pt_cols = [
        "passing_2pt_conversions",
        "rushing_2pt_conversions",
        "receiving_2pt_conversions",
    ]
    if all(c in df.columns for c in two_pt_cols):
        combine_operations.append(
            (
                pl.col("passing_2pt_conversions")
                + pl.col("rushing_2pt_conversions")
                + pl.col("receiving_2pt_conversions")
            ).alias("2pt_conversions")
        )

    # Fumble recoveries (own + opponent)
    fumble_recovery_cols = ["fumble_recovery_own", "fumble_recovery_opp"]
    if all(c in df.columns for c in fumble_recovery_cols):
        combine_operations.append(
            (pl.col("fumble_recovery_own") + pl.col("fumble_recovery_opp")).alias(
                "fumble_recoveries"
            )
        )

    # Computed: Turnover margin = turnovers gained - turnovers lost
    # turnovers_gained = def_interceptions + fumble_recovery_opp
    # turnovers_lost = interceptions_thrown + fumbles_lost (after combining)
    # Note: passing_interceptions is renamed to interceptions_thrown before this function
    turnover_gain_cols = ["def_interceptions", "fumble_recovery_opp"]
    turnover_loss_cols = ["interceptions_thrown"]
    fumbles_lost_available = any(
        c in df.columns for c in ["sack_fumbles_lost", "rushing_fumbles_lost", "fumbles_lost"]
    )
    has_turnover_cols = all(c in df.columns for c in turnover_gain_cols + turnover_loss_cols)
    if has_turnover_cols and fumbles_lost_available:
        # We'll compute this after fumbles_lost is created, so use a placeholder
        pass  # Will be computed after with_columns

    # Apply combinations
    if combine_operations:
        df = df.with_columns(combine_operations)

    # Now compute turnover_margin (after fumbles_lost exists)
    has_fumbles_lost = "fumbles_lost" in df.columns
    has_all_turnover = all(c in df.columns for c in turnover_gain_cols + turnover_loss_cols)
    if has_fumbles_lost and has_all_turnover:
        df = df.with_columns(
            (
                (pl.col("def_interceptions") + pl.col("fumble_recovery_opp"))
                - (pl.col("interceptions_thrown") + pl.col("fumbles_lost"))
            ).alias("turnover_margin")
        )

    # Compute total yards (passing + rushing - sack yards lost)
    yards_cols = ["pass_yards", "rush_yards"]
    if all(c in df.columns for c in yards_cols):
        total_yards_expr = pl.col("pass_yards") + pl.col("rush_yards")
        # Subtract sack yards lost if available
        if "yards_lost_from_sacks" in df.columns:
            total_yards_expr = total_yards_expr - pl.col("yards_lost_from_sacks")
        df = df.with_columns(total_yards_expr.alias("total_yards"))

    # Drop original columns that were combined
    cols_to_drop = (
        fumble_cols + fumble_lost_cols + first_down_cols + two_pt_cols + fumble_recovery_cols
    )
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop)

    return df


def add_per_game_opponent_stats(team_stats_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add opponent's stats for each game to the team's record.

    For each team-game record, looks up the opponent's stats for that same game
    and adds them as opponent_* columns. This allows aggregating "stats of teams
    this team has faced" when computing rolling averages.

    Note: Some stats are excluded from opponent generation because they would be
    exact duplicates or inverses of existing stats.
    See constants.EXCLUDE_FROM_OPPONENT_STATS.

    Args:
        team_stats_df: DataFrame with per-game team statistics

    Returns:
        DataFrame with added opponent_* columns for each game
    """

    # Identify stat columns to copy from opponent (exclude identifiers and duplicate-prone stats)
    exclude_cols = {
        "season",
        "week",
        "team_abbr",
        "opponent_abbr",
        "season_type",
        "games_played",
    }
    exclude_cols.update(constants.EXCLUDE_FROM_OPPONENT_STATS)

    stat_cols = [col for col in team_stats_df.columns if col not in exclude_cols]

    # Create a lookup table with opponent stats
    # Key: (season, week, team_abbr) -> opponent's stats for that game
    opponent_lookup = team_stats_df.select(
        [pl.col("season"), pl.col("week"), pl.col("team_abbr")]
        + [pl.col(c).alias(f"opponent_{c}") for c in stat_cols]
    )

    # Join: for each team's game, look up the opponent's record using opponent_abbr
    # The join key is: this team's opponent_abbr = lookup's team_abbr (same game)
    result = team_stats_df.join(
        opponent_lookup,
        left_on=["season", "week", "opponent_abbr"],
        right_on=["season", "week", "team_abbr"],
        how="left",
        suffix="_opp_lookup",
    )

    return result


def load_pbp(seasons: list[int]) -> pl.DataFrame:
    """
    Load play-by-play data for specified seasons using nflreadpy.

    Args:
        seasons: List of season years to load

    Returns:
        Polars DataFrame with play-by-play data
    """

    log.info("Loading play-by-play for seasons: %s", seasons)
    pbp_df = nfl.load_pbp(seasons=seasons)

    # Normalize team abbreviations in relevant columns
    team_cols = ["home_team", "away_team", "posteam", "defteam"]
    for col in team_cols:
        if col in pbp_df.columns:
            pbp_df = normalize_team_column(pbp_df, col)

    return pbp_df


def aggregate_pbp_stats(
    pbp_df: pl.DataFrame,
    seasons: list[int],
) -> pl.DataFrame:
    """
    Aggregate play-by-play data into per-team per-week statistics.

    Computes statistics that aren't directly available in load_team_stats:
    - Third down attempts and conversions
    - Fourth down attempts and conversions
    - Red zone attempts and touchdowns
    - Two-point conversion attempts and successes
    - Total plays (for points per play calculations)

    Args:
        pbp_df: Play-by-play DataFrame from load_pbp
        seasons: List of seasons to process

    Returns:
        DataFrame with columns: season, week, team_abbr, and computed stats
    """

    if pbp_df.height == 0:
        return pl.DataFrame()

    # Filter to real plays (exclude nulls, penalties, etc.)
    # Filter to requested seasons and regular season
    plays = pbp_df.filter(
        pl.col("season").is_in(seasons)
        & pl.col("season_type").eq("REG")
        & pl.col("posteam").is_not_null()
        & pl.col("play_type").is_in(["run", "pass", "qb_kneel", "qb_spike"])
    )

    if plays.height == 0:
        log.warning("No valid plays found in PBP data for aggregation")
        return pl.DataFrame()

    # Aggregate by team (posteam = team on offense)
    team_stats = plays.group_by(["season", "week", "posteam"]).agg(
        [
            # Third down stats
            pl.col("third_down_converted").sum().alias("third_down_conversions"),
            pl.col("third_down_failed").sum().alias("third_down_fails"),
            # Fourth down stats
            pl.col("fourth_down_converted").sum().alias("fourth_down_conversions"),
            pl.col("fourth_down_failed").sum().alias("fourth_down_fails"),
            # Red zone stats (plays inside opponent's 20)
            ((pl.col("yardline_100").le(20)) & (pl.col("td_team").is_not_null()))
            .sum()
            .alias("red_zone_tds"),
            (pl.col("yardline_100").le(20)).sum().alias("red_zone_plays"),
            # Two-point attempts
            pl.col("two_point_attempt").sum().alias("two_point_attempts"),
            (pl.col("two_point_conv_result") == "success").sum().alias("two_point_successes"),
            # Total plays for points per play
            pl.len().alias("total_plays"),
        ]
    )

    # Rename posteam to team_abbr
    team_stats = team_stats.rename({"posteam": "team_abbr"})

    # Calculate third and fourth down attempts
    team_stats = team_stats.with_columns(
        [
            (pl.col("third_down_conversions") + pl.col("third_down_fails")).alias(
                "third_down_attempts"
            ),
            (pl.col("fourth_down_conversions") + pl.col("fourth_down_fails")).alias(
                "fourth_down_attempts"
            ),
        ]
    )

    return team_stats


def load_elo_ratings(seasons: list[int]) -> pl.DataFrame:
    """
    Load ELO ratings from qb_elos.csv file.

    Args:
        seasons: List of season years to load

    Returns:
        Polars DataFrame with ELO ratings per game
    """

    elo_path = os.path.join(constants.DATA_PATH, "qb_elos.csv")

    if not os.path.exists(elo_path):
        log.warning("ELO file not found: %s", elo_path)
        return pl.DataFrame()

    log.info("Loading ELO ratings from %s", elo_path)
    elo_df = pl.read_csv(elo_path)

    # Filter to requested seasons
    if "season" in elo_df.columns:
        elo_df = elo_df.filter(pl.col("season").is_in(seasons))

    # Cast week to integer and filter out nulls
    if "week" in elo_df.columns:
        elo_df = elo_df.filter(pl.col("week").is_not_null() & (pl.col("week") != ""))
        # Week may be like "19.0" so cast to float first, then int
        elo_df = elo_df.with_columns(pl.col("week").cast(pl.Float64).cast(pl.Int64))

    # Normalize team abbreviations
    if "team1" in elo_df.columns:
        elo_df = elo_df.with_columns(
            pl.col("team1").replace(constants.ALIAS_TO_CANONICAL).alias("team1")
        )
    if "team2" in elo_df.columns:
        elo_df = elo_df.with_columns(
            pl.col("team2").replace(constants.ALIAS_TO_CANONICAL).alias("team2")
        )

    # Select and rename columns for away/home format
    # In qb_elos.csv: team1 = home, team2 = away
    cols_to_keep = [
        "season",
        "week",
        "team1",
        "team2",
        "elo1_pre",
        "elo2_pre",
        "qb1",
        "qb2",
        "qb1_value_pre",
        "qb2_value_pre",
        "qbelo1_pre",
        "qbelo2_pre",
    ]
    available = [c for c in cols_to_keep if c in elo_df.columns]
    elo_df = elo_df.select(available)

    # Cast numeric string columns to floats
    numeric_cols = ["qb1_value_pre", "qb2_value_pre", "qbelo1_pre", "qbelo2_pre"]
    for col in numeric_cols:
        if col in elo_df.columns:
            elo_df = elo_df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    # Rename to away/home format (team1=home, team2=away in ELO data)
    rename_map = {
        "team1": "home_abbr",
        "team2": "away_abbr",
        "elo1_pre": "home_elo_pre",
        "elo2_pre": "away_elo_pre",
        "qb1": "home_qb",
        "qb2": "away_qb",
        "qb1_value_pre": "home_qb_value_pre",
        "qb2_value_pre": "away_qb_value_pre",
        "qbelo1_pre": "home_qb_elo_pre",
        "qbelo2_pre": "away_qb_elo_pre",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in elo_df.columns}
    elo_df = elo_df.rename(rename_map)

    # Deduplicate any repeated games in the source ELO data
    subset_cols = [c for c in ["season", "week", "home_abbr", "away_abbr"] if c in elo_df.columns]
    if subset_cols:
        elo_df = elo_df.unique(subset=subset_cols, keep="last")

    return elo_df


def load_raw_elo_data() -> pl.DataFrame:
    """
    Load raw ELO data from qb_elos.csv file without transformations.

    This is used for QB-specific lookups where we need the original
    column names (qb1, qb2, qb1_value_pre, etc.).

    Returns:
        Raw Polars DataFrame with ELO data
    """

    elo_path = os.path.join(constants.DATA_PATH, "qb_elos.csv")

    if not os.path.exists(elo_path):
        log.warning("ELO file not found: %s", elo_path)
        return pl.DataFrame()

    elo_df = pl.read_csv(elo_path)

    # Cast numeric columns that may be strings
    numeric_cols = ["qb1_value_pre", "qb2_value_pre", "qbelo1_pre", "qbelo2_pre"]
    for col in numeric_cols:
        if col in elo_df.columns:
            elo_df = elo_df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    subset_cols = [c for c in ["season", "week", "team1", "team2"] if c in elo_df.columns]
    if subset_cols:
        elo_df = elo_df.unique(subset=subset_cols, keep="last")

    return elo_df


def get_latest_elo_by_team(elo_df: pl.DataFrame, season: int) -> pl.DataFrame:
    """
    Get the most recent ELO ratings for each team from a given season.

    This is used for future games that don't yet have specific week ELO data.
    For each team, finds their most recent ELO rating from the season.

    Args:
        elo_df: Full ELO DataFrame with season, week, and per-game ratings
        season: The season to get latest ratings from

    Returns:
        DataFrame with columns: team_abbr, elo_pre, qb_value_pre, qb_elo_pre
        One row per team with their most recent ELO values
    """

    if elo_df.height == 0:
        return pl.DataFrame()

    # Filter to the target season
    season_elo = elo_df.filter(pl.col("season") == season)

    if season_elo.height == 0:
        return pl.DataFrame()

    # We need to extract per-team ELO values. The data has away/home format.
    # Create a "melted" view with team_abbr and their ELO values
    away_elo = season_elo.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("away_abbr").alias("team_abbr"),
        pl.col("away_elo_pre").alias("elo_pre"),
        pl.col("away_qb_value_pre").alias("qb_value_pre"),
        pl.col("away_qb_elo_pre").alias("qb_elo_pre"),
    )

    home_elo = season_elo.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("home_abbr").alias("team_abbr"),
        pl.col("home_elo_pre").alias("elo_pre"),
        pl.col("home_qb_value_pre").alias("qb_value_pre"),
        pl.col("home_qb_elo_pre").alias("qb_elo_pre"),
    )

    # Combine and sort by week descending, then take first per team
    all_team_elo = pl.concat([away_elo, home_elo])
    all_team_elo = all_team_elo.sort("week", descending=True)

    # Group by team and take the first (most recent) row
    latest_elo = all_team_elo.group_by("team_abbr").agg(
        pl.col("elo_pre").first(),
        pl.col("qb_value_pre").first(),
        pl.col("qb_elo_pre").first(),
    )

    return latest_elo
