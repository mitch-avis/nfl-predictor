"""Polars-based data loading and merge helpers.

These functions load and prepare schedule/team stats and related data sources.
Implementation was split out of `nfl_predictor.utils.polars_utils`.
"""

import functools
import json
import operator
from pathlib import Path
from typing import TYPE_CHECKING

import nflreadpy as nfl
import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.scraping_utils import (
    get_current_nfl_week,
    normalize_team_column,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

    from polars.datatypes import DataType

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


# Columns of `constants.PBP_COLUMNS` that carry text rather than measurements.
# Everything else in that list is numeric; `season`/`week` are whole numbers and the
# remaining value columns (EPA, rates, indicator flags) are stored as floats because
# nflreadpy publishes several of them as nullable floats.
_PBP_STRING_COLUMNS = frozenset(
    {
        "game_id",
        "posteam",
        "defteam",
        "home_team",
        "away_team",
        "posteam_type",
        "play_type",
        "season_type",
        "two_point_conv_result",
        "td_team",
        "penalty_team",
        "fixed_drive_result",
        "drive_start_yard_line",
        "passer_player_id",
        "passer_player_name",
    }
)

_PBP_INTEGER_COLUMNS = frozenset({"season", "week"})

# Explicit schema used to build a typed but empty play-by-play frame so callers always
# see a stable set of columns and dtypes, even when no season could be loaded.
_PBP_COLUMN_DTYPES: dict[str, DataType] = {
    column: (
        pl.Utf8()
        if column in _PBP_STRING_COLUMNS
        else pl.Int64()
        if column in _PBP_INTEGER_COLUMNS
        else pl.Float64()
    )
    for column in constants.PBP_COLUMNS
}


# Parquet key-value metadata entry that records, in each nflreadpy cache file, the columns
# the loader requested when it wrote the file (a JSON list).
_REQUESTED_COLUMNS_METADATA_KEY = "nfl_predictor.requested_columns"

# Identity of a team-game everywhere in the per-team frames.
_TEAM_GAME_KEYS: tuple[str, str, str] = ("season", "week", "team_abbr")

# Matchup context the schedule supplies for a team-game the stat source does not cover.
_SKELETON_CONTEXT_COLUMNS: tuple[str, str] = ("opponent_abbr", "season_type")

_REGULAR_SEASON_TYPE = "REG"

# Explicit schema so an empty skeleton still carries its columns and dtypes.
_SKELETON_SCHEMA = pl.Schema(
    {
        "season": pl.Int64(),
        "week": pl.Int64(),
        "team_abbr": pl.Utf8(),
        "opponent_abbr": pl.Utf8(),
        "season_type": pl.Utf8(),
    }
)


def _is_numeric_dtype(dtype: DataType) -> bool:
    """Return True if dtype is numeric."""
    return isinstance(dtype, pl.Decimal) or dtype in NUMERIC_DTYPES


def _resolve_cache_dir(cache_dir: os.PathLike | str | None) -> Path:
    """Resolve the nflreadpy cache directory and ensure it exists."""
    resolved = Path(cache_dir) if cache_dir is not None else constants.NFLREADPY_CACHE_DIR
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_current_season(current_season: int | None) -> int:
    """Resolve the current NFL season for cache decisions."""
    if current_season is not None:
        return int(current_season)
    season, _week = get_current_nfl_week()
    return int(season)


def _schedule_cache_path(cache_dir: Path, season: int) -> Path:
    """Build the cache path for a season schedule."""
    return cache_dir / f"schedule_{season}.parquet"


def _team_stats_cache_path(cache_dir: Path, season: int, *, regular_season_only: bool) -> Path:
    """Build the cache path for season team stats."""
    suffix = "reg" if regular_season_only else "all"
    return cache_dir / f"team_stats_{season}_{suffix}.parquet"


def _pbp_cache_path(cache_dir: Path, season: int, *, regular_season_only: bool) -> Path:
    """Build the cache path for a season of play-by-play data."""
    suffix = "reg" if regular_season_only else "all"
    return cache_dir / f"pbp_{season}_{suffix}.parquet"


def _recorded_requested_columns(path: Path) -> frozenset[str]:
    """Return the columns requested when a cache file was written, or none if not recorded.

    Raises:
        ValueError: If the record is not a JSON list of column names.

    """
    recorded = pl.read_parquet_metadata(path).get(_REQUESTED_COLUMNS_METADATA_KEY)
    if recorded is None:
        return frozenset()
    columns = json.loads(recorded)
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        msg = f"recorded requested columns are not a list of names: {recorded[:80]}"
        raise ValueError(msg)
    return frozenset(columns)


def _read_cached_frame(path: Path, requested_columns: Sequence[str] = ()) -> pl.DataFrame | None:
    """Read a cached parquet file, or return None when it is missing, unreadable or stale.

    A file is stale when it lacks a requested column that was not requested when it was
    written: the code now asks for more than the file was built to hold. A requested column
    that was also requested at write time and is still absent is one the source never
    published, so it does not make the file stale. Such a column is never checked again: if
    nflverse later publishes it for that season, only a forced refresh (`force_refresh` on
    the loaders, `--refresh-nflreadpy` on the ETL) rewrites the file. A file written before
    the requested columns were recorded is current when it holds every requested column.

    Args:
        path: Cache file to read
        requested_columns: Columns the caller needs; empty skips the staleness check

    Returns:
        The cached frame, or None on a missing, unreadable or stale file

    """
    if not path.exists():
        return None
    try:
        schema = pl.read_parquet_schema(path)
        missing = {column for column in requested_columns if column not in schema}
        stale = sorted(missing - _recorded_requested_columns(path)) if missing else []
        if stale:
            log.info("Cache file %s predates the requested columns %s.", path, stale)
            return None
        return pl.read_parquet(path)
    except (
        OSError,
        ValueError,
        pl.exceptions.ComputeError,
        pl.exceptions.NoDataError,
    ) as exc:
        log.warning("Failed to read nflreadpy cache file %s: %s", path, exc)
        return None


def _write_cached_frame(
    df: pl.DataFrame, path: Path, requested_columns: Sequence[str] = ()
) -> None:
    """Write a cached parquet file, recording the requested columns, logging any failures."""
    metadata = (
        {_REQUESTED_COLUMNS_METADATA_KEY: json.dumps(list(requested_columns))}
        if requested_columns
        else None
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path, metadata=metadata)
    except (OSError, pl.exceptions.ComputeError) as exc:
        log.warning("Failed to write nflreadpy cache file %s: %s", path, exc)


def _add_stadium_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add stadium type and altitude features."""
    roof_expr = pl.lit(None)
    if "stadium_roof" in df.columns:
        roof_raw = pl.col("stadium_roof").cast(pl.Utf8).str.to_lowercase()
        roof_expr = (
            pl.when(roof_raw.str.contains("retract"))
            .then(pl.lit("retractable"))
            .when(roof_raw.str.contains("dome|indoor|closed"))
            .then(pl.lit("dome"))
            .when(roof_raw.str.contains("outdoor|open"))
            .then(pl.lit("open"))
            .otherwise(pl.lit(None))
        )

    if "stadium_id" in df.columns:
        altitude_map = {
            stadium_id: meta.get("elevation_ft", 0.0)
            for stadium_id, meta in constants.STADIUMS.items()
        }
        altitude = (
            pl.col("stadium_id")
            .replace_strict(altitude_map, default=0.0)
            .cast(pl.Float32)
            .alias("stadium_elevation")
        )
    else:
        altitude = pl.lit(0.0, dtype=pl.Float32).alias("stadium_elevation")

    stadium_type = (
        pl.when(roof_expr.is_null())
        .then(pl.lit("unknown"))
        .otherwise(roof_expr)
        .cast(pl.Utf8)
        .alias("stadium_type")
    )

    if "stadium_city" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("stadium_city"))
    if "stadium_state" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("stadium_state"))
    if "stadium_name" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("stadium_name"))

    if "stadium_surface" in df.columns:
        df = df.with_columns(
            pl.col("stadium_surface").cast(pl.Utf8).str.to_lowercase().alias("stadium_surface")
        )

    return df.with_columns([stadium_type, altitude])


def _apply_schedule_enrichments(df: pl.DataFrame) -> pl.DataFrame:
    """Add optional stadium features to a schedule DataFrame."""
    if "stadium_id" in df.columns and "stadium_city" not in df.columns:
        df = _add_stadium_location(df)
    return _add_stadium_features(df)


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

    schedule_df = _apply_schedule_enrichments(schedule_df)

    return schedule_df


def _prepare_team_stats(team_stats_df: pl.DataFrame, *, regular_season_only: bool) -> pl.DataFrame:
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
    return combine_stats(team_stats_df)


def _team_stats_requested_columns() -> tuple[str, ...]:
    """Return the columns a prepared team-stat frame holds when nflreadpy publishes them all.

    The list comes from running `_prepare_team_stats` on an empty frame that carries every
    raw column the preparation reads: the keys of `constants.NFLREADPY_TEAM_STATS_MAPPING`
    and the inputs `combine_stats` sums or derives from. A change to the mapping, to the
    summed stats or to the derived columns therefore invalidates the cache files written
    before it.

    Returns:
        Requested team-stat column names in the order the preparation produces them

    """
    mapping = constants.NFLREADPY_TEAM_STATS_MAPPING
    renamed = set(mapping.values())
    combine_inputs = [
        *(part for _, parts in _SUMMED_STATS for part in parts),
        *_TURNOVER_MARGIN_INPUTS,
        *_TOTAL_YARDS_INPUTS,
        _TOTAL_YARDS_SACK_INPUT,
    ]
    raw_columns = dict.fromkeys(
        [
            "season",
            "week",
            "season_type",
            "team",
            "opponent_team",
            *mapping,
            *(column for column in combine_inputs if column not in renamed),
        ]
    )
    raw = pl.DataFrame(schema=dict.fromkeys(raw_columns, pl.Int64()))
    raw = raw.with_columns(pl.col("season_type", "team", "opponent_team").cast(pl.Utf8))
    return tuple(_prepare_team_stats(raw, regular_season_only=False).columns)


def load_schedule(
    seasons: list[int],
    *,
    cache_dir: os.PathLike | str | None = None,
    force_refresh: bool = False,
    current_season: int | None = None,
) -> pl.DataFrame:
    """Load NFL schedule data for specified seasons using nflreadpy.

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
            schedule_frames.append(_apply_schedule_enrichments(cached))
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
    """Add stadium name, city, and state columns based on stadium_id.

    Uses the STADIUMS mapping in constants to look up
    city and state for each stadium.

    Args:
        df: DataFrame with stadium_id column

    Returns:
        DataFrame with stadium_name, stadium_city, and stadium_state columns added

    """
    # Create mapping dictionaries for name/city/state
    name_map = {k: v.get("name") for k, v in constants.STADIUMS.items()}
    city_map = {k: v.get("city") for k, v in constants.STADIUMS.items()}
    state_map = {k: v.get("state") for k, v in constants.STADIUMS.items()}

    name_expr = pl.col("stadium_id").replace_strict(name_map, default=None)
    if "stadium_name" in df.columns:
        name_expr = pl.coalesce([pl.col("stadium_name"), name_expr])

    # Add name/city/state columns using replace_strict (Polars >=1.0)
    return df.with_columns(
        [
            name_expr.alias("stadium_name"),
            pl.col("stadium_id").replace_strict(city_map, default=None).alias("stadium_city"),
            pl.col("stadium_id").replace_strict(state_map, default=None).alias("stadium_state"),
        ]
    )


def load_team_stats(
    seasons: list[int],
    *,
    regular_season_only: bool = True,
    cache_dir: os.PathLike | str | None = None,
    force_refresh: bool = False,
    current_season: int | None = None,
) -> pl.DataFrame:
    """Load team statistics for specified seasons using nflreadpy.

    Cached stats are used for historical seasons when available and current: a cache file
    that lacks a column the loader now produces (see `_team_stats_requested_columns`), and
    did not request when it was written, is downloaded again; a requested column the source
    did not publish then is not checked again, so `force_refresh` is what picks it up if
    nflverse publishes it later. Current and future seasons are always refreshed to keep
    upcoming games up to date; when that refresh fails, the cached file is used as it is,
    or the season is skipped.

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

    requested_columns = _team_stats_requested_columns()
    team_frames: list[pl.DataFrame] = []
    for season in seasons:
        cache_path = _team_stats_cache_path(
            resolved_cache_dir, season, regular_season_only=regular_season_only
        )
        use_cache = (season < resolved_current_season) and not force_refresh
        cached = _read_cached_frame(cache_path, requested_columns) if use_cache else None
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

        try:
            season_df = nfl.load_team_stats(seasons=[season])
        except ConnectionError as exc:
            if season < resolved_current_season:
                raise

            fallback_cached = _read_cached_frame(cache_path)
            if fallback_cached is not None:
                log.warning(
                    "Current-season team stats unavailable for season %d; "
                    "using cached data from %s. Error: %s",
                    season,
                    cache_path,
                    exc,
                )
                team_frames.append(fallback_cached)
            else:
                log.warning(
                    "Current-season team stats unavailable for season %d; "
                    "continuing without them. Error: %s",
                    season,
                    exc,
                )
            continue

        season_df = _prepare_team_stats(season_df, regular_season_only=regular_season_only)
        _write_cached_frame(season_df, cache_path, requested_columns)
        team_frames.append(season_df)

    if not team_frames:
        return pl.DataFrame()

    return pl.concat(team_frames, how="diagonal")


def build_team_game_skeleton(schedule_df: pl.DataFrame) -> pl.DataFrame:
    """Build one row per team per completed regular-season game from the schedule.

    The schedule is the authority on which games were played, so every completed game
    contributes exactly two rows, one per team, whether or not a statistical source
    covers it. Games without both scores are still to be played and are left out.

    Args:
        schedule_df: Schedule frame with `season`, `week`, `home_abbr`, `away_abbr` and
            both score columns; `game_type` is used to keep regular-season games when
            the column is present

    Returns:
        Frame with `season`, `week`, `team_abbr`, `opponent_abbr` and `season_type`,
        sorted by season, week and team; empty when no completed regular-season game
        is available

    """
    required = {"season", "week", "home_abbr", "away_abbr", "home_score", "away_score"}
    if schedule_df.height == 0 or not required <= set(schedule_df.columns):
        return pl.DataFrame(schema=_SKELETON_SCHEMA)

    completed = schedule_df.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    if "game_type" in completed.columns:
        completed = completed.filter(pl.col("game_type") == _REGULAR_SEASON_TYPE)
    if completed.height == 0:
        return pl.DataFrame(schema=_SKELETON_SCHEMA)

    def one_side(team_column: str, opponent_column: str) -> pl.DataFrame:
        """Project the schedule onto one team's perspective of each game."""
        return completed.select(
            pl.col("season"),
            pl.col("week"),
            pl.col(team_column).alias("team_abbr"),
            pl.col(opponent_column).alias("opponent_abbr"),
            pl.lit(_REGULAR_SEASON_TYPE).alias("season_type"),
        )

    both_sides = pl.concat(
        [
            one_side("home_abbr", "away_abbr"),
            one_side("away_abbr", "home_abbr"),
        ]
    )
    return both_sides.cast(_SKELETON_SCHEMA).sort(list(_TEAM_GAME_KEYS))


def _log_team_stats_coverage(team_stats_df: pl.DataFrame, skeleton: pl.DataFrame) -> None:
    """Warn for every season and team whose stat rows differ from the schedule.

    Args:
        team_stats_df: Per-team-game stats restricted to the seasons the skeleton covers
        skeleton: Schedule-derived per-team-game frame

    """
    group_keys = ["season", "team_abbr"]
    scheduled = skeleton.group_by(group_keys, maintain_order=True).agg(
        pl.len().alias("scheduled_games")
    )
    observed = team_stats_df.group_by(group_keys, maintain_order=True).agg(
        pl.len().alias("stat_rows")
    )
    mismatched = (
        scheduled.join(observed, on=group_keys, how="full", coalesce=True)
        .with_columns(
            pl.col("scheduled_games").fill_null(0),
            pl.col("stat_rows").fill_null(0),
        )
        .filter(pl.col("scheduled_games") != pl.col("stat_rows"))
        .sort(group_keys)
    )
    for row in mismatched.iter_rows(named=True):
        log.warning(
            "Team-stats coverage gap for %s %s: %d team-stat rows, %d scheduled games.",
            row["season"],
            row["team_abbr"],
            row["stat_rows"],
            row["scheduled_games"],
        )


def _repair_collapsed_box_scores(joined: pl.DataFrame, *, covered_flag: str) -> pl.DataFrame:
    """Null the box score of a stat row that describes both teams of one game.

    Where the source publishes a row for only one side of a scheduled game, that row's
    box score is the whole game's production rather than the team's own: the missing
    side's yards, plays and penalties are counted in it. Such a value is not a team-game
    statistic, so it is replaced with a null and the games it covers fall out of the
    season-to-date denominators along with it. Identity, the scoring columns the schedule
    supplies and the play-by-play counts are per-team correct either way and stay, as
    `constants.TEAM_GAME_NON_BOX_SCORE_COLUMNS` records.

    Args:
        joined: Per-team-game frame with one row per scheduled team-game
        covered_flag: Name of the boolean column marking rows the stats source covers

    Returns:
        The frame with the affected rows' box-score columns set to null

    """
    if "opponent_abbr" not in joined.columns:
        return joined

    opponent_flag = "_opponent_covered"
    opponent_coverage = joined.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("team_abbr").alias("opponent_abbr"),
        pl.col(covered_flag).alias(opponent_flag),
    )
    flagged = joined.join(
        opponent_coverage,
        on=["season", "week", "opponent_abbr"],
        how="left",
    ).with_columns(pl.col(opponent_flag).fill_null(value=False))

    collapsed = pl.col(covered_flag) & ~pl.col(opponent_flag)
    affected = flagged.filter(collapsed)
    if affected.height == 0:
        return flagged.drop(opponent_flag)

    log.warning(
        "Repairing %d team-stat row(s) whose box score covers both teams of the game.",
        affected.height,
    )
    for row in (
        affected.select("season", "week", "team_abbr")
        .sort(["season", "week", "team_abbr"])
        .iter_rows(named=True)
    ):
        log.warning(
            "Dropping the box score of %s week %s %s: the source has no row for its opponent.",
            row["season"],
            row["week"],
            row["team_abbr"],
        )

    box_score_columns = [
        column
        for column in joined.columns
        if column not in constants.TEAM_GAME_NON_BOX_SCORE_COLUMNS and column != covered_flag
    ]
    return flagged.with_columns(
        pl.when(collapsed)
        .then(pl.lit(None, dtype=joined.schema[column]))
        .otherwise(pl.col(column))
        .alias(column)
        for column in box_score_columns
    ).drop(opponent_flag)


def attach_team_stats_to_schedule(
    team_stats_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
) -> pl.DataFrame:
    """Left-join team stats onto the schedule-derived per-team-game frame.

    A team-game the statistical source does not cover keeps a row with null stats, so
    season-to-date game counts follow the schedule instead of the source's coverage.
    Season-to-date means and rates are unaffected by the added rows because Polars
    aggregations skip nulls, so every rate stays a ratio of sums over the games that do
    carry values.

    Rows the skeleton does not cover are never dropped: stats from a season the schedule
    does not span, and stat rows without a scheduled counterpart, are carried through
    unchanged. The result is sorted by season, week and team so the frame is
    deterministic regardless of source ordering.

    Args:
        team_stats_df: Per-team-game statistics
        schedule_df: Schedule covering the same seasons

    Returns:
        The team stats with the same columns, one row per completed scheduled team-game
        plus any uncovered rows

    """
    keys = list(_TEAM_GAME_KEYS)
    if team_stats_df.height == 0 or not set(keys) <= set(team_stats_df.columns):
        return team_stats_df

    skeleton = build_team_game_skeleton(schedule_df)
    if skeleton.height == 0:
        return team_stats_df

    skeleton = skeleton.cast({key: team_stats_df.schema[key] for key in keys})
    covered_seasons = skeleton["season"].unique(maintain_order=True).to_list()
    in_scope = team_stats_df.filter(pl.col("season").is_in(covered_seasons))
    out_of_scope = team_stats_df.filter(~pl.col("season").is_in(covered_seasons))

    _log_team_stats_coverage(in_scope, skeleton)

    context_columns = [col for col in _SKELETON_CONTEXT_COLUMNS if col in team_stats_df.columns]
    covered_flag = "_covered_by_team_stats"
    joined = (
        skeleton.select(*keys, *context_columns)
        .join(
            in_scope.drop(context_columns).with_columns(pl.lit(value=True).alias(covered_flag)),
            on=keys,
            how="left",
        )
        .with_columns(pl.col(covered_flag).fill_null(value=False))
    )
    joined = _repair_collapsed_box_scores(joined, covered_flag=covered_flag).drop(covered_flag)
    unscheduled = in_scope.join(skeleton.select(keys), on=keys, how="anti")

    return (
        pl.concat([joined, unscheduled, out_of_scope], how="diagonal")
        .select(team_stats_df.columns)
        .sort(keys)
    )


def add_scoring_data_to_team_stats(
    team_stats_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
) -> pl.DataFrame:
    """Add points scored and points allowed from schedule to team stats.

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
    return team_stats_df.join(
        all_scores,
        on=["season", "week", "team_abbr"],
        how="left",
    )


def combine_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Combine related stats into single columns and remove originals.

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
    # Each summed stat, from its parts, when every part is present; the parts are dropped.
    # First downs are passing + rushing only: receiving first downs overlap with passing.
    combine_operations = [
        functools.reduce(operator.add, [pl.col(part) for part in parts]).alias(name)
        for name, parts in _SUMMED_STATS
        if all(part in df.columns for part in parts)
    ]
    if combine_operations:
        df = df.with_columns(combine_operations)

    # Computed after fumbles_lost exists: turnover margin is turnovers gained minus turnovers
    # lost, where turnovers gained are def_interceptions plus fumble_recovery_opp and
    # turnovers lost are interceptions_thrown plus fumbles_lost.
    # Note: passing_interceptions is renamed to interceptions_thrown before this function
    if "fumbles_lost" in df.columns and all(c in df.columns for c in _TURNOVER_MARGIN_INPUTS):
        df = df.with_columns(
            (
                (pl.col("def_interceptions") + pl.col("fumble_recovery_opp"))
                - (pl.col("interceptions_thrown") + pl.col("fumbles_lost"))
            ).alias("turnover_margin")
        )

    # Compute total yards (passing + rushing - sack yards lost)
    if all(c in df.columns for c in _TOTAL_YARDS_INPUTS):
        total_yards_expr = pl.col("pass_yards") + pl.col("rush_yards")
        # Subtract sack yards lost if available
        if _TOTAL_YARDS_SACK_INPUT in df.columns:
            total_yards_expr = total_yards_expr - pl.col(_TOTAL_YARDS_SACK_INPUT)
        df = df.with_columns(total_yards_expr.alias("total_yards"))

    # Drop original columns that were combined
    cols_to_drop = [part for _, parts in _SUMMED_STATS for part in parts if part in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop)

    return df


# Columns `combine_stats` reads, besides the summed parts, to derive turnover margin and
# total yards (`fumbles_lost` is itself a summed stat). Total yards need passing and
# rushing yards; sack yards lost are subtracted only when present.
_TURNOVER_MARGIN_INPUTS: tuple[str, ...] = (
    "def_interceptions",
    "fumble_recovery_opp",
    "interceptions_thrown",
)
_TOTAL_YARDS_INPUTS: tuple[str, ...] = ("pass_yards", "rush_yards")
_TOTAL_YARDS_SACK_INPUT = "yards_lost_from_sacks"

# The stats `combine_stats` sums from their parts.
_SUMMED_STATS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fumbles", ("sack_fumbles", "rushing_fumbles", "receiving_fumbles")),
    ("fumbles_lost", ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")),
    ("first_downs", ("passing_first_downs", "rushing_first_downs")),
    (
        "2pt_conversions",
        ("passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions"),
    ),
    ("fumble_recoveries", ("fumble_recovery_own", "fumble_recovery_opp")),
)


def add_per_game_opponent_stats(team_stats_df: pl.DataFrame) -> pl.DataFrame:
    """Add opponent's stats for each game to the team's record.

    For each team-game record, looks up the opponent's stats for that same game
    and adds them as opponent_* columns. This allows aggregating "stats of teams
    this team has faced" when computing rolling averages.

    Note: Some stats are excluded from opponent generation because they would be
    exact duplicates or inverses of existing stats.
    See constants.EXCLUDE_FROM_OPPONENT_STATS. The mirrors named in
    constants.OPPONENT_MIRROR_INTERMEDIATES are built anyway because a derived metric
    reads them; the final schema selection drops them.

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
    exclude_cols.difference_update(constants.OPPONENT_MIRROR_INTERMEDIATES)

    stat_cols = [col for col in team_stats_df.columns if col not in exclude_cols]

    # Create a lookup table with opponent stats
    # Key: (season, week, team_abbr) -> opponent's stats for that game
    opponent_lookup = team_stats_df.select(
        [pl.col("season"), pl.col("week"), pl.col("team_abbr")]
        + [pl.col(c).alias(f"opponent_{c}") for c in stat_cols]
    )

    # Join: for each team's game, look up the opponent's record using opponent_abbr
    # The join key is: this team's opponent_abbr = lookup's team_abbr (same game)
    return team_stats_df.join(
        opponent_lookup,
        left_on=["season", "week", "opponent_abbr"],
        right_on=["season", "week", "team_abbr"],
        how="left",
        suffix="_opp_lookup",
    )


def _empty_pbp_frame() -> pl.DataFrame:
    """Build an empty play-by-play frame with the full expected schema."""
    return pl.DataFrame(schema=_PBP_COLUMN_DTYPES)


def _select_pbp_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the columns of `constants.PBP_COLUMNS` that the frame has, in that order."""
    available_cols = set(frame.columns)
    return frame.select([c for c in constants.PBP_COLUMNS if c in available_cols])


def _prepare_pbp(season_df: pl.DataFrame, *, regular_season_only: bool) -> pl.DataFrame:
    """Reduce raw nflreadpy play-by-play data to the cached project schema.

    Only the columns of `constants.PBP_COLUMNS` that are actually present are kept, because
    older seasons do not publish every column. Team abbreviations are normalized so cached
    frames already use canonical identifiers.

    Args:
        season_df: Raw play-by-play frame for a single season
        regular_season_only: If True, drop non-regular-season plays when `season_type` exists

    Returns:
        Prepared play-by-play DataFrame ready to cache

    """
    prepared = _select_pbp_columns(season_df)

    if regular_season_only and "season_type" in prepared.columns:
        prepared = prepared.filter(pl.col("season_type") == "REG")

    for col in ("posteam", "defteam", "home_team", "away_team", "td_team", "penalty_team"):
        if col in prepared.columns:
            prepared = normalize_team_column(prepared, col)

    return prepared


def load_pbp(
    seasons: list[int],
    *,
    cache_dir: os.PathLike | str | None = None,
    force_refresh: bool = False,
    current_season: int | None = None,
    regular_season_only: bool = True,
) -> pl.DataFrame:
    """Load play-by-play data for specified seasons using nflreadpy.

    Seasons are loaded one at a time because play-by-play frames are large. Each season is
    reduced to the guarded selection in `constants.PBP_COLUMNS`, filtered, normalized, and
    then cached, so a cache hit needs no further work.

    Caching contract:
        - Prepared frames are cached as `pbp_<season>_<reg|all>.parquet` in the resolved
          cache directory.
        - Historical seasons (`season < current_season`) read from that cache whenever it
          exists, is current and `force_refresh` is False. Each file records the columns
          requested when it was written; a file that lacks a column of
          `constants.PBP_COLUMNS` that was not requested then predates the current list and
          is downloaded again (see `_read_cached_frame`). A requested column the source did
          not publish when the file was written is not checked again; `force_refresh`
          picks it up if nflverse publishes it later. A cache hit keeps only the columns of
          `constants.PBP_COLUMNS`, as a download would.
        - The current and any future season is always refreshed, and `force_refresh` bypasses
          the cache for every season.

    Degrade-on-failure contract:
        - A failure for a historical season is re-raised, because historical data is expected
          to be available.
        - A failure for the current or a future season is logged as a warning and the cached
          frame is used when one exists, even one that predates the requested columns;
          otherwise that season is skipped. Both a `ConnectionError` and a `ValueError`
          count as a failure here: before kickoff the current season has no play-by-play
          published at all, and nflreadpy reports that by raising `ValueError` for an
          out-of-range season rather than by failing to connect.
        - When no season yields data, an empty frame carrying the full expected schema is
          returned (see `_PBP_COLUMN_DTYPES`) so callers always see stable columns and dtypes.

    Args:
        seasons: List of season years to load
        cache_dir: Optional cache directory override for nflreadpy outputs
        force_refresh: If True, refresh play-by-play data even when cache exists
        current_season: Optional current season override for cache decisions
        regular_season_only: If True, keep only regular season plays

    Returns:
        Polars DataFrame with prepared play-by-play data

    """
    if not seasons:
        return _empty_pbp_frame()

    resolved_cache_dir = _resolve_cache_dir(cache_dir)
    resolved_current_season = _resolve_current_season(current_season)

    log.info("Loading play-by-play for seasons: %s", seasons)

    requested_columns = tuple(constants.PBP_COLUMNS)
    pbp_frames: list[pl.DataFrame] = []
    for season in seasons:
        cache_path = _pbp_cache_path(
            resolved_cache_dir, season, regular_season_only=regular_season_only
        )
        use_cache = (season < resolved_current_season) and not force_refresh
        cached = _read_cached_frame(cache_path, requested_columns) if use_cache else None
        if cached is not None:
            log.info(
                "Using cached nflreadpy play-by-play for season %d from %s",
                season,
                cache_path,
            )
            pbp_frames.append(_select_pbp_columns(cached))
            continue

        if season < resolved_current_season and not force_refresh:
            log.info("Play-by-play cache miss for season %d; loading via nflreadpy.", season)
        else:
            log.info("Refreshing play-by-play via nflreadpy for season %d.", season)

        try:
            season_df = nfl.load_pbp(seasons=[season])
        except (ConnectionError, ValueError) as exc:
            if season < resolved_current_season:
                raise

            fallback_cached = _read_cached_frame(cache_path)
            if fallback_cached is not None:
                log.warning(
                    "Current-season play-by-play unavailable for season %d; "
                    "using cached data from %s. Error: %s",
                    season,
                    cache_path,
                    exc,
                )
                pbp_frames.append(_select_pbp_columns(fallback_cached))
            else:
                log.warning(
                    "Current-season play-by-play not published yet for season %d; "
                    "continuing without it. Error: %s",
                    season,
                    exc,
                )
            continue

        season_df = _prepare_pbp(season_df, regular_season_only=regular_season_only)
        _write_cached_frame(season_df, cache_path, requested_columns)
        pbp_frames.append(season_df)

    if not pbp_frames:
        log.warning("No play-by-play data available for seasons: %s", seasons)
        return _empty_pbp_frame()

    return pl.concat(pbp_frames, how="diagonal")


# Types of the `qb_elos.csv` columns this package reads. The file is the 538-style schema
# produced by `nfeloqb` (`team1` is the home team). Its rows start in 1920, decades before
# it carries quarterback ratings or weeks, so letting Polars guess from the first rows
# reads those columns as text. `week` is published as a float ("1.0") and blank where the
# source has no week; `date` stays ISO text, which sorts in date order.
_QB_ELO_COLUMN_TYPES: dict[str, type[pl.DataType]] = {
    "date": pl.String,
    "season": pl.Int64,
    "week": pl.Float64,
    "team1": pl.String,
    "team2": pl.String,
    "elo1_pre": pl.Float64,
    "elo2_pre": pl.Float64,
    "qb1": pl.String,
    "qb2": pl.String,
    "qb1_value_pre": pl.Float64,
    "qb2_value_pre": pl.Float64,
    "qbelo1_pre": pl.Float64,
    "qbelo2_pre": pl.Float64,
}


def _read_qb_elos(elo_path: Path) -> pl.DataFrame:
    """Read the `qb_elos.csv` columns this package uses, with their declared types.

    A blank value reads as null. A text token in a numeric column (for example `NA`)
    fails the read with a `ComputeError` that names the column, so a malformed copy of
    the file stops the ETL instead of silently blanking quarterback features.
    """
    header = pl.read_csv(elo_path, n_rows=0).columns
    column_types = {name: dtype for name, dtype in _QB_ELO_COLUMN_TYPES.items() if name in header}
    return pl.read_csv(elo_path, columns=list(column_types), schema_overrides=column_types)


def load_elo_ratings(seasons: list[int]) -> pl.DataFrame:
    """Load ELO ratings from qb_elos.csv file.

    Args:
        seasons: List of season years to load

    Returns:
        Polars DataFrame with ELO ratings per game

    """
    elo_path = constants.DATA_PATH / "qb_elos.csv"

    if not elo_path.exists():
        log.warning("ELO file not found: %s", elo_path)
        return pl.DataFrame()

    log.info("Loading ELO ratings from %s", elo_path)
    elo_df = _read_qb_elos(elo_path)

    # Filter to requested seasons
    if "season" in elo_df.columns:
        elo_df = elo_df.filter(pl.col("season").is_in(seasons))

    # Drop rows without a week (blank in the file) and store the published "19.0" as 19
    if "week" in elo_df.columns:
        elo_df = elo_df.filter(pl.col("week").is_not_null())
        elo_df = elo_df.with_columns(pl.col("week").cast(pl.Int64))

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
        elo_df = elo_df.unique(subset=subset_cols, keep="last", maintain_order=True)

    return elo_df


def load_raw_elo_data() -> pl.DataFrame:
    """Load the qb_elos.csv rows under their published column names.

    This is used for QB-specific lookups where we need the original
    column names (qb1, qb2, qb1_value_pre, etc.). Only the columns in
    `_QB_ELO_COLUMN_TYPES` are read, with those types.

    Returns:
        Raw Polars DataFrame with ELO data, deduplicated per game

    """
    elo_path = constants.DATA_PATH / "qb_elos.csv"

    if not elo_path.exists():
        log.warning("ELO file not found: %s", elo_path)
        return pl.DataFrame()

    elo_df = _read_qb_elos(elo_path)

    subset_cols = [c for c in ["season", "week", "team1", "team2"] if c in elo_df.columns]
    if subset_cols:
        elo_df = elo_df.unique(subset=subset_cols, keep="last", maintain_order=True)

    return elo_df


def get_latest_elo_by_team(elo_df: pl.DataFrame, season: int) -> pl.DataFrame:
    """Get the most recent ELO ratings for each team from a given season.

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
    return all_team_elo.group_by("team_abbr", maintain_order=True).agg(
        pl.col("elo_pre").first(),
        pl.col("qb_value_pre").first(),
        pl.col("qb_elo_pre").first(),
    )
