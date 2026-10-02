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
    - data/strength_snapshots.csv: Pre-week schedule-adjusted strength for every team on
      each season's schedule and every processed week, teams on a bye included

Usage:
    Run directly to collect and process all data:
        python -m nfl_predictor.data_collection

    Optional flags (when run as a script):
        --timing --debug-logs --refresh-nflreadpy --min-season --max-season --incremental

    Or import and call programmatically:
        from nfl_predictor.data_collection import collect_all_data
        df = collect_all_data([2023, 2024])
"""

import argparse
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import clock, game_utils, polars_utils, season_cache
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars import nfelo_lines, pbp, pick_time_lines, qb_stats
from nfl_predictor.utils.polars.strength_table import (
    build_prior_strength_snapshot,
    build_strength_table,
    combine_strength_snapshots,
    stamp_strength_snapshot,
)
from nfl_predictor.utils.polars.week_rows import build_prior_season_stats, process_week

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import date


def _default_max_season(today: date | None = None) -> int:
    """Return the default max season (inclusive) based on today's date."""
    if today is None:
        today = clock.local_today()
    return clock.nfl_season(today)


# Configuration: default season bounds (inclusive)
DEFAULT_MIN_SEASON = constants.MIN_SEASON

# Data collection tuning toggles (overridable via CLI when run as a script).
ENABLE_DATA_COLLECTION_TIMING = False
ENABLE_DATA_COLLECTION_DEBUG = False
FORCE_REFRESH_NFLREADPY = False


@dataclass(frozen=True, kw_only=True)
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
    # Directory the produced datasets are written to. ``None`` means the packaged
    # ``constants.DATA_PATH``, so an omitted --data-dir keeps the historical behaviour.
    data_dir: Path | None = None
    # Whether play-by-play should override derivable per-team-game box-score columns.
    # Default flipped to "pbp" on 2026-09-21: nflverse-comparison and walk-forward verified
    # (models/pbp_vs_nflverse_m54_2/COMPARISON.md, models/wf_m54_flip_*).
    team_stats_source: str = "pbp"
    # Whether the legacy TeamRankings situational percentage columns come from scrape or PBP.
    # Default flipped to "pbp" on 2026-09-21 (see team_stats_source above); play-by-play fills
    # 1999-2002, which the TeamRankings scrape (starts 2003) leaves null.
    tr_stats_source: str = "pbp"
    # Reuse finished seasons' builds from the season cache under `<data dir>/cache/` when
    # their inputs, the code and the options are unchanged (see `utils.season_cache`).
    incremental: bool = False
    # Which market line each game carries: the stored nflverse line ("stored") or the line
    # known at pick time ("pick_time", see `utils.polars.pick_time_lines`).
    line_source: str = constants.LINE_SOURCE_STORED


def _configure_logging(*, enable_debug: bool) -> None:
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
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory the collected datasets are written to (default: the packaged data "
            "directory). Cached upstream inputs are unaffected."
        ),
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
    parser.add_argument(
        "--team-stats-source",
        choices=("nflverse", "pbp"),
        default="pbp",
        help=(
            "Prefer nflverse or play-by-play for derivable per-team-game box-score columns "
            "(default pbp since 2026-09-21)."
        ),
    )
    parser.add_argument(
        "--tr-stats-source",
        choices=("scrape", "pbp"),
        default="pbp",
        help=(
            "Source for the legacy TeamRankings situational percentage columns "
            "(default pbp since 2026-09-21)."
        ),
    )
    parser.add_argument(
        "--line-source",
        choices=constants.LINE_SOURCES,
        default=constants.LINE_SOURCE_STORED,
        help=(
            "Market line each game carries: the stored nflverse line (default) or the line "
            "known at pick time (nfelo's opener for completed games, its latest line for "
            "upcoming ones, the stored line where nfelo has no real opener)."
        ),
    )
    parser.add_argument(
        "--incremental",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse each finished season's build from <data dir>/cache/"
            f"{constants.ETL_SEASON_CACHE_DIRNAME}/ when its inputs, the ETL code and these "
            "options are unchanged, and rebuild the rest. Off by default."
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
        data_dir=args.data_dir,
        team_stats_source=str(args.team_stats_source),
        tr_stats_source=str(args.tr_stats_source),
        incremental=bool(args.incremental),
        line_source=str(args.line_source),
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
            team_stats_source="pbp",
            tr_stats_source="pbp",
        )
    return _parse_args(argv)


def _overlay_pbp_team_box_scores(
    team_stats_df: pl.DataFrame,
    pbp_box_scores: pl.DataFrame,
) -> pl.DataFrame:
    """Prefer play-by-play team-game values while keeping nflverse fallbacks.

    The play-by-play frame carries only the derivable columns. Where it has a non-null value, it
    wins; where it does not, the nflverse value stays in place. Rows that exist only in the
    play-by-play frame are retained so the schedule skeleton can still carry them forward.
    """
    if pbp_box_scores.height == 0:
        return team_stats_df
    if team_stats_df.height == 0:
        return pbp_box_scores.sort(["season", "week", "team_abbr", "opponent_abbr"])

    keys = ["season", "week", "team_abbr", "opponent_abbr"]
    joined = team_stats_df.join(
        pbp_box_scores,
        on=keys,
        how="full",
        coalesce=True,
        suffix="_pbp",
    )
    output_columns = list(team_stats_df.columns)
    output_columns.extend(
        column for column in pbp_box_scores.columns if column not in output_columns
    )

    exprs: list[pl.Expr] = []
    for column in output_columns:
        if column in keys:
            exprs.append(pl.col(column))
            continue
        pbp_column = f"{column}_pbp"
        if pbp_column in joined.columns and column in joined.columns:
            exprs.append(pl.coalesce(pl.col(pbp_column), pl.col(column)).alias(column))
        elif pbp_column in joined.columns:
            exprs.append(pl.col(pbp_column).alias(column))
        else:
            exprs.append(pl.col(column))

    return joined.select(exprs).sort(keys)


def _resolve_seasons(min_season: int, max_season: int) -> list[int]:
    """Resolve the list of seasons to process (inclusive bounds)."""
    if min_season < constants.NFLREADPY_MIN_SEASON:
        msg = (
            f"min_season must be >= {constants.NFLREADPY_MIN_SEASON} "
            "(nflreadpy data availability starts in 1999)."
        )
        raise ValueError(msg)
    if max_season < min_season:
        msg = "max_season must be >= min_season."
        raise ValueError(msg)
    return list(range(min_season, max_season + 1))


@contextmanager
def _timed_step(label: str, *, enabled: bool) -> Iterator[None]:
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


def _log_df_stats(label: str, df: pl.DataFrame, *, enabled: bool) -> None:
    """Log dataframe shape/columns when debug logging is enabled."""
    if not enabled:
        return
    log.debug("%s: %d rows, %d cols", label, df.height, len(df.columns))


def main(argv: list[str] | None = None) -> None:
    """Run the nflreadpy-backed data collection pipeline.

    Orchestrates the data collection, processing, and storage for NFL game predictions.
    """
    config = _resolve_config(argv)
    _configure_logging(enable_debug=config.enable_debug)

    log.info("Starting data collection with nflreadpy...")
    log.info(
        "NFLreadpy cache enabled (historical seasons). Force refresh: %s",
        config.force_refresh_nflreadpy,
    )

    # Determine current season and week
    today = clock.local_today()
    current_season = today.year if today.month > constants.SEASON_END_MONTH else today.year - 1
    current_week = clock.nfl_week(today)

    log.info("Current season: %s, week: %s", current_season, current_week)

    seasons_to_process = _resolve_seasons(config.min_season, config.max_season)
    log.info(
        "Season range: %s-%s (%d seasons)",
        config.min_season,
        config.max_season,
        len(seasons_to_process),
    )

    strength_snapshots: list[pl.DataFrame] = []
    market_lines_metadata: dict[str, object] = {}
    with _timed_step("collect_all_data", enabled=config.enable_timing):
        all_data_df = collect_all_data(
            seasons_to_process,
            config=config,
            strength_snapshots=strength_snapshots,
            market_lines_metadata=market_lines_metadata,
        )

    _log_df_stats("all_data", all_data_df, enabled=config.enable_debug)

    # Create version without diff columns (for non-ML local usage)
    no_diff_df = polars_utils.remove_diff_columns(all_data_df)

    # Save all data (ML version with diffs)
    save_dataframe(all_data_df, "all_data_ml", config.data_dir)

    # Save all data (non-ML version without diffs)
    save_dataframe(no_diff_df, "all_data", config.data_dir)

    # Filter and save completed games (both versions)
    completed_df = polars_utils.filter_completed_games(all_data_df)
    completed_no_diff_df = polars_utils.remove_diff_columns(completed_df)
    save_dataframe(completed_df, "completed_games_ml", config.data_dir)
    save_dataframe(completed_no_diff_df, "completed_games", config.data_dir)

    # Per-team pre-week strength, bye teams included, for reports that rank teams.
    save_dataframe(
        combine_strength_snapshots(strength_snapshots),
        constants.STRENGTH_SNAPSHOTS_NAME,
        config.data_dir,
    )

    # Filter and save upcoming games for prediction (ML version only)
    upcoming_df = polars_utils.filter_upcoming_games(all_data_df, current_season, current_week)
    save_dataframe(
        upcoming_df, f"predict/week_{current_week:>02}_games_to_predict", config.data_dir
    )

    if market_lines_metadata:
        _save_market_lines_metadata(market_lines_metadata, config.data_dir)

    log.info("Data collection complete.")


def _save_market_lines_metadata(metadata: dict[str, object], data_dir: Path | None) -> None:
    """Write a pick-time build's line record beside the datasets."""
    path = _resolve_data_dir(data_dir) / f"{constants.MARKET_LINES_METADATA_NAME}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("Saved the market-line record to %s", path)


def _attach_qb_features(
    games: pl.DataFrame,
    pbp_df: pl.DataFrame,
    *,
    max_season: int,
    current_season: int,
    identity_path: Path | None = None,
) -> pl.DataFrame:
    """Attach the quarterback per-dropback family to the combined game rows.

    Career rates need every earlier regular season, so play-by-play seasons from
    ``constants.NFLREADPY_MIN_SEASON`` through ``max_season`` that are not already in
    ``pbp_df`` are loaded from the per-season cache before aggregating; a partial-season run
    therefore produces the same values as a full rebuild. Those history seasons are never
    force-refreshed: a refresh applies to the seasons being processed, which arrive in
    ``pbp_df``, and re-pulling every earlier season would turn a one-season refresh into a
    full-history download. Rows without ``away_qb`` / ``home_qb`` come back unchanged, and the
    final schema fills the columns with nulls.

    Args:
        games: Combined game rows after the future-week quarterback fill.
        pbp_df: Play-by-play already loaded for the team-stat seasons.
        max_season: Last season being processed.
        current_season: Passed to ``load_pbp`` for its cache decisions.
        identity_path: Quarterback identity file; defaults to
            ``DATA_PATH/<QB_META_DATA_NAME>.csv``.

    Returns:
        ``games`` with the quarterback columns from ``qb_stats.attach_qb_features``.

    """
    if not {"season", "week", "away_qb", "home_qb"}.issubset(games.columns):
        log.warning("Game rows have no away_qb/home_qb; skipping quarterback features")
        return games
    loaded = (
        set(pbp_df.get_column("season").unique(maintain_order=True).to_list())
        if "season" in pbp_df
        else set()
    )
    missing = [
        season
        for season in range(constants.NFLREADPY_MIN_SEASON, max_season + 1)
        if season not in loaded
    ]
    parts = [qb_stats.aggregate_qb_game_stats(pbp_df)]
    if missing:
        history = polars_utils.load_pbp(missing, force_refresh=False, current_season=current_season)
        parts.append(qb_stats.aggregate_qb_game_stats(history))
    qb_games = pl.concat(parts, how="vertical")
    path = identity_path or constants.DATA_PATH / f"{constants.QB_META_DATA_NAME}.csv"
    identity = qb_stats.load_qb_identity(path)
    log.info(
        "QB features: %d quarterback games from play-by-play, %d identity names",
        qb_games.height,
        identity.height,
    )
    return qb_stats.attach_qb_features(games, qb_games, identity)


def _log_pbp_null_rates(team_stats_df: pl.DataFrame, *, enable_debug: bool) -> None:
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
        team_stats_df.group_by("season", maintain_order=True)
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
    deduped = lookup.unique(subset=join_keys, keep="first", maintain_order=True)
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


def _build_team_game_frame(
    team_stats_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
    pbp_team_games: pl.DataFrame,
) -> pl.DataFrame:
    """Build the per-team-game frame the season-to-date features aggregate over.

    The schedule supplies the rows, so every completed game contributes exactly two
    team-games whether or not the statistical sources cover it; team stats and the
    play-by-play counts are attached to that frame and stay null where a source is
    missing.

    Args:
        team_stats_df: Per-team-game statistics from nflverse
        schedule_df: Schedule covering the same seasons as the team stats
        pbp_team_games: Per-team-game play-by-play counts, possibly empty

    Returns:
        Per-team-game frame with the team-stat columns and the play-by-play counts

    """
    framed = polars_utils.attach_team_stats_to_schedule(team_stats_df, schedule_df)
    return _join_pbp_team_game_stats(framed, pbp_team_games)


def _default_config(seasons: list[int]) -> DataCollectionConfig:
    """Return the module-default config over the requested seasons."""
    return DataCollectionConfig(
        enable_timing=ENABLE_DATA_COLLECTION_TIMING,
        enable_debug=ENABLE_DATA_COLLECTION_DEBUG,
        force_refresh_nflreadpy=FORCE_REFRESH_NFLREADPY,
        min_season=min(seasons),
        max_season=max(seasons),
    )


@dataclass(frozen=True)
class _EtlSources:
    """The loaded upstream data every season is built from."""

    current_season: int
    current_week: int
    schedule_df: pl.DataFrame
    team_stats_df: pl.DataFrame
    pbp_df: pl.DataFrame
    elo_df: pl.DataFrame
    raw_elo_df: pl.DataFrame
    # Set for a pick-time build: the nfelo snapshot, the line order's report and the maps.
    pick_time: pick_time_lines.PickTimeLines | None = None


def _stats_window(
    seasons: list[int], schedule_df: pl.DataFrame, config: DataCollectionConfig, current_season: int
) -> tuple[list[int], pl.DataFrame]:
    """Return the team-stat seasons and their schedule, with the prior season for week 1."""
    # Include previous season for week 1 regression if not processing from the beginning
    stats_seasons = list(seasons)
    min_season = min(seasons)
    if min_season > constants.NFLREADPY_MIN_SEASON:  # Need prior season for week 1 regression
        stats_seasons = [min_season - 1, *stats_seasons]

    # The schedule for every season the team stats cover, so the per-team-game frame and
    # the scoring merge below both span the week-1 previous-season fallback.
    if stats_seasons == list(seasons):
        return stats_seasons, schedule_df
    with _timed_step("load_prior_schedule", enabled=config.enable_timing):
        prior_schedule_df = polars_utils.load_schedule(
            [min_season - 1],
            force_refresh=config.force_refresh_nflreadpy,
            current_season=current_season,
        )
    return stats_seasons, pl.concat([prior_schedule_df, schedule_df], how="diagonal")


def _load_team_game_frame(
    stats_seasons: list[int],
    stats_schedule_df: pl.DataFrame,
    config: DataCollectionConfig,
    current_season: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load team stats and play-by-play and build the per-team-game frame.

    Returns the frame, with scoring and per-game opponent stats, and the play-by-play.
    """
    # Load team statistics (regular season only - used for building features)
    # Playoff games use cumulative stats from the regular season
    with _timed_step("load_team_stats", enabled=config.enable_timing):
        team_stats_df = polars_utils.load_team_stats(
            stats_seasons,
            regular_season_only=True,
            force_refresh=config.force_refresh_nflreadpy,
            current_season=current_season,
        )
    log.info("Loaded team stats: %d regular season team-game records", team_stats_df.height)
    _log_df_stats("team_stats_df", team_stats_df, enabled=config.enable_debug)

    # Load play-by-play and attach per-team-game counts before any downstream enrichment.
    # Uses the same season window as team stats so the week-1 previous-season fallback is
    # covered, and degrades to null columns when the source is unavailable.
    with _timed_step("load_pbp", enabled=config.enable_timing):
        pbp_df = polars_utils.load_pbp(
            stats_seasons,
            force_refresh=config.force_refresh_nflreadpy,
            current_season=current_season,
        )
    log.info("Loaded play-by-play: %d regular season plays", pbp_df.height)

    if config.team_stats_source == "pbp":
        with _timed_step("aggregate_pbp_team_box_score_stats", enabled=config.enable_timing):
            pbp_box_scores = pbp.aggregate_pbp_team_box_score_stats(pbp_df)
        team_stats_df = _overlay_pbp_team_box_scores(team_stats_df, pbp_box_scores)
        _log_df_stats("team_stats_with_pbp_box_scores", team_stats_df, enabled=config.enable_debug)

    with _timed_step("aggregate_pbp_team_game_stats", enabled=config.enable_timing):
        pbp_team_games = polars_utils.aggregate_pbp_team_game_stats(pbp_df)
        team_stats_df = _build_team_game_frame(team_stats_df, stats_schedule_df, pbp_team_games)
    log.info("Aggregated play-by-play: %d team-game records", pbp_team_games.height)
    log.info("Per-team-game frame: %d team-game rows", team_stats_df.height)
    _log_pbp_null_rates(team_stats_df, enable_debug=config.enable_debug)
    _log_df_stats("team_stats_with_pbp", team_stats_df, enabled=config.enable_debug)

    # Add scoring data (points scored/allowed) to team stats from schedule
    # This enables computing points-related metrics like scoring margin
    with _timed_step("add_scoring_data", enabled=config.enable_timing):
        team_stats_df = polars_utils.add_scoring_data_to_team_stats(
            team_stats_df, stats_schedule_df
        )
    _log_df_stats("team_stats_with_scores", team_stats_df, enabled=config.enable_debug)

    # Add per-game opponent stats AFTER scoring data is added
    # This ensures opponent_points_scored, opponent_points_allowed, etc. are included
    with _timed_step("add_per_game_opponent_stats", enabled=config.enable_timing):
        team_stats_df = polars_utils.add_per_game_opponent_stats(team_stats_df)
    _log_df_stats("team_stats_with_opponents", team_stats_df, enabled=config.enable_debug)
    return team_stats_df, pbp_df


def _load_elo(
    seasons: list[int], config: DataCollectionConfig
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the per-game ELO ratings and the raw ELO rows for quarterback lookups."""
    with _timed_step("load_elo_ratings", enabled=config.enable_timing):
        elo_df = polars_utils.load_elo_ratings(seasons)
    if elo_df.height > 0:
        log.info("Loaded ELO ratings: %d game records", elo_df.height)
    else:
        log.warning("No ELO ratings loaded")
    _log_df_stats("elo_df", elo_df, enabled=config.enable_debug)

    # Load raw ELO data for QB lookups (needed for fill_future_qb_data)
    with _timed_step("load_raw_elo_data", enabled=config.enable_timing):
        raw_elo_df = polars_utils.load_raw_elo_data()
    return elo_df, raw_elo_df


def _load_sources(seasons: list[int], config: DataCollectionConfig) -> _EtlSources:
    """Load the schedule, the per-team-game frame, the play-by-play and the ELO ratings."""
    current_season, current_week = polars_utils.get_current_nfl_week()
    log.info("Current season: %d, week: %d (for TR scraping)", current_season, current_week)

    # Load full schedule with lines/odds directly from nflreadpy
    # Includes both regular season (REG) and playoff games (WC, DIV, CON, SB)
    with _timed_step("load_schedule", enabled=config.enable_timing):
        schedule_df = polars_utils.load_schedule(
            seasons,
            force_refresh=config.force_refresh_nflreadpy,
            current_season=current_season,
        )
    log.info(
        "Loaded schedule: %d total games (regular season + playoffs)",
        schedule_df.height,
    )
    _log_df_stats("schedule_df", schedule_df, enabled=config.enable_debug)

    stats_seasons, stats_schedule_df = _stats_window(seasons, schedule_df, config, current_season)
    team_stats_df, pbp_df = _load_team_game_frame(
        stats_seasons, stats_schedule_df, config, current_season
    )
    elo_df, raw_elo_df = _load_elo(seasons, config)
    return _EtlSources(
        current_season=current_season,
        current_week=current_week,
        schedule_df=schedule_df,
        team_stats_df=team_stats_df,
        pbp_df=pbp_df,
        elo_df=elo_df,
        raw_elo_df=raw_elo_df,
    )


class _TeamRankingsLoader:
    """Load each season's TeamRankings once per run, so no season is scraped twice."""

    def __init__(self, min_season: int, sources: _EtlSources, *, enable_timing: bool) -> None:
        """Remember the run's first season, the current week and the timing switch."""
        self._min_season = min_season
        self._current_season = sources.current_season
        self._current_week = sources.current_week
        self._enable_timing = enable_timing
        self._cache: dict[int, pl.DataFrame] = {}

    def load(self, season: int) -> pl.DataFrame:
        """Return the season's TeamRankings rows (empty before the source's first season)."""
        cached_df = self._cache.get(season)
        if cached_df is not None:
            return cached_df
        if season < constants.TEAMRANKINGS_MIN_SEASON:
            log.info(
                "Skipping TeamRankings for season %d (data starts in %d).",
                season,
                constants.TEAMRANKINGS_MIN_SEASON,
            )
            self._cache[season] = pl.DataFrame()
            return self._cache[season]

        min_week = 1
        if season == constants.TEAMRANKINGS_MIN_SEASON:
            min_week = max(min_week, constants.TEAMRANKINGS_MIN_WEEK)
        if season == self._min_season and season < self._current_season:
            min_week = max(min_week, constants.TEAMRANKINGS_MIN_WEEK)
        with _timed_step(f"load_team_rankings_{season}", enabled=self._enable_timing):
            tr_df = polars_utils.load_team_rankings(
                season,
                self._current_season,
                self._current_week,
                min_week=min_week,
            )
        self._cache[season] = tr_df
        return tr_df


def _sort_newest_first(games: pl.DataFrame) -> pl.DataFrame:
    """Sort games by date, newest first, breaking same-date ties by game id.

    Several games share each kickoff date, so a date-only sort would leave their order to
    whatever order the rows arrived in; the game-id tie-break makes the published row order
    a fixed function of the games. Without a date column the frame is returned unchanged.
    """
    if "date" not in games.columns:
        return games
    if "game_id" not in games.columns:
        return games.sort("date", descending=True, maintain_order=True)
    return games.sort(["date", "game_id"], descending=[True, False])


def _combine_seasons(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Stack the seasons newest first and drop any duplicate game (defensive cleanup)."""
    combined_df = _sort_newest_first(pl.concat(frames, how="diagonal"))
    # Remove exact duplicate games if any exist (defensive cleanup)
    if "game_id" in combined_df.columns:
        return combined_df.unique(subset=["game_id"], keep="first", maintain_order=True)
    return combined_df.unique(
        subset=["season", "week", "away_abbr", "home_abbr"],
        keep="first",
        maintain_order=True,
    )


def _finish_games(
    combined_df: pl.DataFrame, sources: _EtlSources, max_season: int, *, enable_timing: bool
) -> pl.DataFrame:
    """Fill future games' quarterbacks and lines, add QB features, and order the columns."""
    # Fill in QB data for future games using most recent starters
    combined_df = game_utils.fill_future_qb_data(combined_df, sources.raw_elo_df)
    # Quarterback features key on the final starter assignment, future weeks included.
    with _timed_step("attach_qb_features", enabled=enable_timing):
        combined_df = _attach_qb_features(
            combined_df,
            sources.pbp_df,
            max_season=max_season,
            current_season=sources.current_season,
        )
    if sources.pick_time is None:
        # Fill in lines for future games from SurvivorGrid
        combined_df = game_utils.fill_future_game_lines(combined_df)
        # Fill missing moneylines by calculating from spreads
        combined_df = game_utils.fill_missing_moneylines(combined_df)
    else:
        # The same fills, pricing every derived moneyline with its season's fitted map.
        fill_moneylines = sources.pick_time.fill_moneylines
        combined_df = game_utils.fill_future_game_lines(
            combined_df, fill_moneylines=fill_moneylines
        )
        combined_df = fill_moneylines(combined_df)
    # Select final columns in correct order
    combined_df = polars_utils.select_final_columns(combined_df)
    # Ensure final ordering by date after any dedupe/transforms
    return _sort_newest_first(combined_df)


@dataclass(frozen=True)
class _SeasonCacheRun:
    """The season cache, this run's cache keys, and the first season never cached."""

    cache: season_cache.SeasonCache
    keys: season_cache.SeasonKeys
    current_season: int


def _open_season_cache(
    config: DataCollectionConfig, sources: _EtlSources, min_season: int
) -> _SeasonCacheRun | None:
    """Return the season cache of an incremental run, or None for a full rebuild.

    A season's build reads the schedule, the team stats and the ELO ratings through that
    season (the coach history reaches back to the run's first season), its own and the
    previous season's TeamRankings, and the options below; the keys cover exactly those.
    """
    if not config.incremental:
        return None
    keys = season_cache.SeasonKeys(
        {
            "schedule": sources.schedule_df,
            "team_stats": sources.team_stats_df,
            "elo": sources.elo_df,
        },
        {
            "min_season": min_season,
            "blend_strength_prior": config.blend_strength_prior,
            "blend_stat_prior": config.blend_stat_prior,
            "stat_prior_blend_games": config.stat_prior_blend_games,
            "team_stats_source": config.team_stats_source,
            "tr_stats_source": config.tr_stats_source,
            "line_source": config.line_source,
        },
    )
    directory = _resolve_data_dir(config.data_dir) / "cache" / constants.ETL_SEASON_CACHE_DIRNAME
    log.info("Incremental run: finished seasons are read from %s when unchanged", directory)
    return _SeasonCacheRun(season_cache.SeasonCache(directory), keys, sources.current_season)


def _build_season(
    season: int, sources: _EtlSources, inputs: SeasonInputs, cache: _SeasonCacheRun | None
) -> pl.DataFrame:
    """Build one season's game rows, reusing a finished season's cached build when valid.

    The season in progress is always rebuilt. A cached build carries the season's strength
    snapshots too, so they reach ``inputs.strength_snapshots`` exactly as a rebuild's would.
    """
    key = (
        None
        if cache is None or season >= cache.current_season
        else cache.keys.key(season, {"tr": inputs.tr_df, "prev_tr": inputs.prev_tr_df})
    )
    if cache is None or key is None:
        return process_season(season, sources.schedule_df, sources.team_stats_df, inputs)
    build = cache.cache.load(season, key)
    if build is None:
        recorded: list[pl.DataFrame] = []
        games = process_season(
            season,
            sources.schedule_df,
            sources.team_stats_df,
            replace(inputs, strength_snapshots=recorded),
        )
        snapshots = (
            pl.concat(recorded, how="vertical") if recorded else combine_strength_snapshots([])
        )
        build = season_cache.SeasonBuild(games=games, snapshots=snapshots)
        cache.cache.store(season, key, build)
    else:
        log.info("Season %d reused from the season cache", season)
    if inputs.strength_snapshots is not None and build.snapshots.height > 0:
        inputs.strength_snapshots.append(build.snapshots)
    return build.games


def collect_all_data(
    seasons: list[int],
    *,
    config: DataCollectionConfig | None = None,
    strength_snapshots: list[pl.DataFrame] | None = None,
    market_lines_metadata: dict[str, object] | None = None,
) -> pl.DataFrame:
    """Collect and combine all data for specified seasons.

    Args:
        seasons: List of season years to process
        config: Optional runtime config for logging/timing and cache refresh
        strength_snapshots: Optional list that receives each processed week's per-team
            strength snapshot; see `combine_strength_snapshots`
        market_lines_metadata: Optional dict that receives a pick-time build's line record
            (the nfelo snapshot and its hash, per-season counts, the moneyline maps); a
            stored-line build leaves it empty

    Returns:
        Combined DataFrame with all game data and features

    """
    if config is None:
        config = _default_config(seasons)

    log.info(
        "Collecting data for %d seasons: %s - %s",
        len(seasons),
        min(seasons),
        max(seasons),
    )
    sources = _load_sources(seasons, config)
    if config.line_source == constants.LINE_SOURCE_PICK_TIME:
        sources = _with_pick_time_lines(sources, seasons, enable_timing=config.enable_timing)
    min_season = min(seasons)
    rankings = _TeamRankingsLoader(min_season, sources, enable_timing=config.enable_timing)
    cache = _open_season_cache(config, sources, min_season)

    all_seasons_data = []
    for season in seasons:
        log.info("Processing season %d...", season)
        inputs = SeasonInputs(
            min_season=min_season,
            elo_df=sources.elo_df,
            # This season's TeamRankings (scraped if current), then the prior season's (week 1).
            tr_df=rankings.load(season),
            prev_tr_df=rankings.load(season - 1) if season > min_season else None,
            tr_stats_source=config.tr_stats_source,
            blend_strength_prior=config.blend_strength_prior,
            blend_stat_prior=config.blend_stat_prior,
            stat_prior_blend_games=config.stat_prior_blend_games,
            strength_snapshots=strength_snapshots,
            timing_enabled=config.enable_timing,
        )
        with _timed_step(f"process_season_{season}", enabled=config.enable_timing):
            season_data = _build_season(season, sources, inputs, cache)
        if season_data.height > 0:
            all_seasons_data.append(season_data)

    if not all_seasons_data:
        return pl.DataFrame()
    games = _finish_games(
        _combine_seasons(all_seasons_data),
        sources,
        max(seasons),
        enable_timing=config.enable_timing,
    )
    if sources.pick_time is not None and market_lines_metadata is not None:
        market_lines_metadata.update(sources.pick_time.metadata())
    return games


def _with_pick_time_lines(
    sources: _EtlSources, seasons: list[int], *, enable_timing: bool
) -> _EtlSources:
    """Replace the schedule's lines with the lines known at pick time.

    Reads nfelo's lines (cached, never failing the run) and, for the moneyline maps, the
    schedules of every earlier season the run does not build, from the per-season cache.
    """
    with _timed_step("load_nfelo_lines", enabled=enable_timing):
        snapshot = nfelo_lines.load_nfelo_lines()
    built = set(seasons)
    history_seasons = [
        season
        for season in range(constants.NFLREADPY_MIN_SEASON, max(seasons))
        if season not in built
    ]
    history = (
        polars_utils.load_schedule(
            history_seasons, force_refresh=False, current_season=sources.current_season
        )
        if history_seasons
        else pl.DataFrame()
    )
    schedule_df, pick_time = pick_time_lines.prepare_pick_time_lines(
        sources.schedule_df, history=history, snapshot=snapshot
    )
    for row in pick_time.report.iter_rows(named=True):
        log.info(
            "Pick-time lines, season %d: %d games, %d in nfelo, %d openers, %d stored "
            "fallback rows, upcoming %d nfelo / %d nflverse / %d without a line",
            row["season"],
            row["games"],
            row["matched"],
            row["opener"],
            row["stored_fallback"],
            row["upcoming_nfelo"],
            row["upcoming_nflverse"],
            row["upcoming_without_line"],
        )
    return replace(sources, schedule_df=schedule_df, pick_time=pick_time)


@dataclass(frozen=True, kw_only=True)
class SeasonInputs:
    """Everything a season's weeks are built from beyond the schedule and the team stats.

    Attributes:
        min_season: Earliest season included in this run.
        elo_df: ELO ratings.
        tr_df: TeamRankings rows for this season.
        prev_tr_df: TeamRankings rows for the previous season (for week 1).
        tr_stats_source: Source for the legacy TeamRankings stat columns.
        blend_strength_prior: Set to False to ablate the strength prior blend.
        blend_stat_prior: Set to False to ablate the season-to-date stat prior blend.
        stat_prior_blend_games: K in the stat blend weight ``games / (games + K)``.
        strength_snapshots: Optional list that receives each processed week's per-team
            strength snapshot, every scheduled team included (bye teams too), plus the week
            after the regular season when the schedule does not reach it yet.
        timing_enabled: Whether to accumulate per-step timing totals.
        timing_totals: The dict the timing totals accumulate in, when enabled.
        team_elo_trends: Rolling ELO trend features for the season.
        qb_trends: Rolling quarterback trend features for the season.
        team_stat_trends: Rolling team-stat trend features for the season.
        coach_features: Per-team coach features.
        prior_strength_snapshot: Previous season's final strength snapshot for the prior
            blend; built for the week when absent and the blend is on.
        prior_season_stats: Regressed previous-season stats from `build_prior_season_stats`;
            built for the week when absent.

    """

    min_season: int
    elo_df: pl.DataFrame | None = None
    tr_df: pl.DataFrame | None = None
    prev_tr_df: pl.DataFrame | None = None
    tr_stats_source: str = "scrape"
    blend_strength_prior: bool = True
    blend_stat_prior: bool = True
    stat_prior_blend_games: float = constants.PRIOR_BLEND_GAMES
    strength_snapshots: list[pl.DataFrame] | None = None
    timing_enabled: bool = False
    timing_totals: dict[str, float] | None = None
    team_elo_trends: pl.DataFrame | None = None
    qb_trends: pl.DataFrame | None = None
    team_stat_trends: pl.DataFrame | None = None
    coach_features: pl.DataFrame | None = None
    prior_strength_snapshot: pl.DataFrame | None = None
    prior_season_stats: pl.DataFrame | None = None


def _with_season_features(
    season: int, schedule_df: pl.DataFrame, team_stats_df: pl.DataFrame, inputs: SeasonInputs
) -> SeasonInputs:
    """Precompute the season's trend and coach features and its priors, once per season."""
    team_elo_trends = pl.DataFrame()
    qb_trends = pl.DataFrame()
    team_stat_trends = pl.DataFrame()
    coach_features = pl.DataFrame()
    if inputs.elo_df is not None and inputs.elo_df.height > 0:
        team_elo_trends = polars_utils.build_team_elo_trends(inputs.elo_df, season)
        qb_trends = polars_utils.build_qb_trends(inputs.elo_df, season)
    if team_stats_df.height > 0:
        team_stat_trends = polars_utils.build_team_stat_trends(
            team_stats_df,
            season,
            stats=["scoring_margin", "turnover_margin"],
        )
    if schedule_df.height > 0:
        coach_features = polars_utils.build_coach_features(schedule_df, season=season)

    return replace(
        inputs,
        team_elo_trends=team_elo_trends,
        qb_trends=qb_trends,
        team_stat_trends=team_stat_trends,
        coach_features=coach_features,
        # Solved once per season rather than per week: it depends only on the prior season.
        prior_strength_snapshot=(
            build_prior_strength_snapshot(team_stats_df, season, min_season=inputs.min_season)
            if inputs.blend_strength_prior
            else None
        ),
        # Also built once per season: the Week-1 fallback and the stat blend's prior.
        prior_season_stats=build_prior_season_stats(
            team_stats_df, season, min_season=inputs.min_season
        ),
    )


def _log_timing_summary(season: int, timing_totals: dict[str, float]) -> None:
    """Log the season's per-step timing totals, slowest first."""
    summary = ", ".join(
        f"{label}={timing_totals[label]:.2f}s"
        for label in sorted(timing_totals, key=lambda label: timing_totals[label], reverse=True)
    )
    log.info("Timing summary season %d: %s", season, summary)


def process_season(
    season: int, schedule_df: pl.DataFrame, team_stats_df: pl.DataFrame, inputs: SeasonInputs
) -> pl.DataFrame:
    """Process a single season's data.

    Args:
        season: Season year to process
        schedule_df: Full schedule DataFrame
        team_stats_df: Full team stats DataFrame
        inputs: The run's other inputs and options; the season's trend features and priors
            are precomputed here.

    Returns:
        Processed DataFrame for the season

    """
    # Filter to this season
    season_schedule = schedule_df.filter(pl.col("season") == season)

    if season_schedule.height == 0:
        log.warning("No schedule data for season %d", season)
        return pl.DataFrame()

    # Precompute trend features and priors for the season (time-safe, prior weeks only)
    inputs = _with_season_features(season, schedule_df, team_stats_df, inputs)
    inputs = replace(inputs, timing_totals={} if inputs.timing_enabled else None)

    # Get unique weeks in the schedule
    weeks = sorted(season_schedule.select("week").unique(maintain_order=True).to_series().to_list())

    # Process each week
    weekly_data = []
    for week in weeks:
        week_data = process_week(season, week, season_schedule, team_stats_df, inputs)
        if week_data.height > 0:
            weekly_data.append(week_data)

    # No game carries the week after the regular season until the playoff schedule is
    # published, yet a ranking through the final regular-season week needs that week's
    # snapshot (the whole regular season). Solve it directly when the schedule stops short.
    after_regular_season = constants.get_regular_season_weeks(season) + 1
    if inputs.strength_snapshots is not None and after_regular_season not in weeks:
        full_season = build_strength_table(
            team_stats_df,
            season_schedule,
            season=season,
            week=after_regular_season,
            prior_snapshot=inputs.prior_strength_snapshot,
        )
        inputs.strength_snapshots.append(
            stamp_strength_snapshot(full_season, season=season, week=after_regular_season)
        )

    if inputs.timing_enabled and inputs.timing_totals:
        _log_timing_summary(season, inputs.timing_totals)

    if weekly_data:
        return pl.concat(weekly_data, how="diagonal")

    return pl.DataFrame()


def _resolve_data_dir(data_dir: Path | str | None) -> Path:
    """Return the directory datasets are read from and written to.

    Args:
        data_dir: An explicit directory, or ``None`` for the packaged data directory.

    Returns:
        The directory to use.

    """
    return Path(data_dir) if data_dir is not None else constants.DATA_PATH


def save_dataframe(df: pl.DataFrame, name: str, data_dir: Path | str | None = None) -> None:
    """Save a Polars DataFrame to CSV.

    Args:
        df: DataFrame to save
        name: Base name for the file (without extension)
        data_dir: Directory to write into; defaults to the packaged data directory.

    """
    file_path = _resolve_data_dir(data_dir) / f"{name}.csv"

    # Create directory if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.write_csv(file_path)
    log.info("Saved %s (%d rows) to %s", name, df.height, file_path)


def load_dataframe(name: str, data_dir: Path | str | None = None) -> pl.DataFrame | None:
    """Load a Polars DataFrame from CSV.

    Args:
        name: Base name for the file (without extension)
        data_dir: Directory to read from; defaults to the packaged data directory.

    Returns:
        DataFrame or None if file doesn't exist

    """
    file_path = _resolve_data_dir(data_dir) / f"{name}.csv"

    if not file_path.is_file():
        log.warning("File not found: %s", file_path)
        return None

    return pl.read_csv(file_path, infer_schema_length=None)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
