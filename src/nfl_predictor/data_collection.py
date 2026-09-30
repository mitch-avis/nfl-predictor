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
        --timing --debug-logs --refresh-nflreadpy --min-season --max-season

    Or import and call programmatically:
        from nfl_predictor.data_collection import collect_all_data
        df = collect_all_data([2023, 2024])
"""

import argparse
import logging
import time
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import clock, game_utils, polars_utils
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars import pbp, qb_stats, schedule_strength, strength_snapshot

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
        msg = f"team_side must be 'away' or 'home', got: {team_side}"
        raise ValueError(msg)

    prefix = f"{team_side}_"
    base_cols = [
        col[len(prefix) :] for col in constants.RECORD_FEATURE_COLUMNS if col.startswith(prefix)
    ]
    rename_map = {
        "team_abbr": f"{team_side}_abbr",
        **{col: f"{prefix}{col}" for col in base_cols},
    }
    return records_df.rename(rename_map)


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


@contextmanager
def _timed_substep(
    label: str,
    *,
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
    with _timed_step("collect_all_data", enabled=config.enable_timing):
        all_data_df = collect_all_data(
            seasons_to_process, config=config, strength_snapshots=strength_snapshots
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

    log.info("Data collection complete.")


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
    loaded = set(pbp_df.get_column("season").unique().to_list()) if "season" in pbp_df else set()
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


def _combine_seasons(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Stack the seasons newest first and drop any duplicate game (defensive cleanup)."""
    combined_df = pl.concat(frames, how="diagonal")
    # Sort by date, newest first
    if "date" in combined_df.columns:
        combined_df = combined_df.sort("date", descending=True)
    # Remove exact duplicate games if any exist (defensive cleanup)
    if "game_id" in combined_df.columns:
        return combined_df.unique(subset=["game_id"], keep="first")
    return combined_df.unique(
        subset=["season", "week", "away_abbr", "home_abbr"],
        keep="first",
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


def collect_all_data(
    seasons: list[int],
    *,
    config: DataCollectionConfig | None = None,
    strength_snapshots: list[pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Collect and combine all data for specified seasons.

    Args:
        seasons: List of season years to process
        config: Optional runtime config for logging/timing and cache refresh
        strength_snapshots: Optional list that receives each processed week's per-team
            strength snapshot; see `combine_strength_snapshots`

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
    min_season = min(seasons)
    rankings = _TeamRankingsLoader(min_season, sources, enable_timing=config.enable_timing)

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
            season_data = process_season(season, sources.schedule_df, sources.team_stats_df, inputs)
        if season_data.height > 0:
            all_seasons_data.append(season_data)

    if not all_seasons_data:
        return pl.DataFrame()
    return _finish_games(
        _combine_seasons(all_seasons_data),
        sources,
        max(seasons),
        enable_timing=config.enable_timing,
    )


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


# Columns the season-to-date stat aggregation and the record features both produce. The
# record features own them: `games_played` there is the team's completed games this season
# (wins + losses + ties), while the stat frame reports the row count behind its means, which
# is the previous season's total on a fallback row.
_RECORD_OWNED_STAT_COLUMNS = ("away_games_played", "home_games_played")


# Every per-team value of one week's strength table: the published game-row columns plus
# the league-wide home-field term, which only the snapshot file carries.
_STRENGTH_TABLE_COLUMNS = (*constants.ADJUSTED_STRENGTH_STATS, "adj_hfa")

# Column types of the published strength snapshot file.
_STRENGTH_SNAPSHOT_FILE_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    column: {"season": pl.Int64, "week": pl.Int64, "team_abbr": pl.String}.get(column, pl.Float64)
    for column in constants.STRENGTH_SNAPSHOT_FILE_COLUMNS
}


def build_strength_features(
    team_stats_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
    *,
    season: int,
    week: int,
    prior_snapshot: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build the per-team strength columns published on game rows for one season week.

    This is `build_strength_table` without the home-field term, which is one league-wide
    value per week and so is not joined onto games. Arguments are those of
    `build_strength_table`.

    Returns:
        One row per team with `team_abbr` and `constants.ADJUSTED_STRENGTH_STATS`.

    """
    return build_strength_table(
        team_stats_df,
        schedule_df,
        season=season,
        week=week,
        prior_snapshot=prior_snapshot,
    ).select("team_abbr", *constants.ADJUSTED_STRENGTH_STATS)


def _stamp_strength_snapshot(table: pl.DataFrame, *, season: int, week: int) -> pl.DataFrame:
    """Key one week's strength table by season and week, in the published file layout."""
    return table.with_columns(
        pl.lit(season).alias("season"),
        pl.lit(week).alias("week"),
    ).select(pl.col(name).cast(dtype) for name, dtype in _STRENGTH_SNAPSHOT_FILE_SCHEMA.items())


def combine_strength_snapshots(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Stack weekly strength snapshots into the published file, typed and sorted.

    The result has one row per ``(season, week, team)`` for every team on each season's
    schedule, bye teams included, ordered by season, week and team. An empty list gives
    an empty frame that still carries every documented column, so the file keeps one
    schema whether or not any week was solved.
    """
    if not frames:
        return pl.DataFrame(schema=_STRENGTH_SNAPSHOT_FILE_SCHEMA)
    return pl.concat(frames, how="vertical").sort(["season", "week", "team_abbr"])


def build_strength_table(
    team_stats_df: pl.DataFrame,
    schedule_df: pl.DataFrame,
    *,
    season: int,
    week: int,
    prior_snapshot: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build every schedule-adjusted strength value for one season week, per team.

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
        prior_snapshot: Previous season's final snapshot for the early-season blend;
            ``None`` (the ablation) publishes the raw in-season solve.

    Returns:
        One row per team on the season's schedule, teams on a bye included, with
        `team_abbr`, `constants.ADJUSTED_STRENGTH_STATS` and the league-wide home-field
        term `adj_hfa`.

    """
    snapshot = strength_snapshot.build_strength_snapshot(
        team_stats_df,
        season=season,
        week=week,
        prior_snapshot=prior_snapshot,
        teams=_schedule_teams(schedule_df, season),
    )
    if snapshot.height == 0:
        return pl.DataFrame(
            schema={"team_abbr": pl.String, **dict.fromkeys(_STRENGTH_TABLE_COLUMNS, pl.Float64)}
        )

    features = snapshot.select("team_abbr", *constants.STRENGTH_TEAM_STATS, "adj_hfa")

    ratings = snapshot.select("team_abbr", "adj_strength_composite")
    try:
        adjusted = schedule_strength.compute_schedule_strength_adjusted(
            _regular_season_schedule(schedule_df, season),
            ratings,
            season=season,
            week=week,
            columns=schedule_strength.RatingColumns(
                rating="adj_strength_composite", team="team_abbr"
            ),
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
            columns=schedule_strength.TeamColumns(team="team_abbr", opponent="opponent_abbr"),
            margin=schedule_strength.MarginSource(
                numerator=_STRENGTH_MARGIN_NUMERATOR, denominator=_STRENGTH_MARGIN_DENOMINATOR
            ),
        )
        features = features.join(raw, on="team_abbr", how="left")

    missing = [
        pl.lit(None, dtype=pl.Float64).alias(column)
        for column in constants.ADJUSTED_STRENGTH_STATS
        if column not in features.columns
    ]
    if missing:
        features = features.with_columns(missing)

    return features.select("team_abbr", *_STRENGTH_TABLE_COLUMNS)


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
        msg = f"Strength feature join changed the row count from {row_count} to {merged.height}"
        raise ValueError(msg)

    return merged.with_columns(
        [
            pl.lit(None, dtype=pl.Float64).alias(f"{side}_{column}")
            for side in ("away", "home")
            for column in constants.ADJUSTED_STRENGTH_STATS
            if f"{side}_{column}" not in merged.columns
        ]
    )


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
    weeks = sorted(season_schedule.select("week").unique().to_series().to_list())

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
            _stamp_strength_snapshot(full_season, season=season, week=after_regular_season)
        )

    if inputs.timing_enabled and inputs.timing_totals:
        _log_timing_summary(season, inputs.timing_totals)

    if weekly_data:
        return pl.concat(weekly_data, how="diagonal")

    return pl.DataFrame()


@dataclass(frozen=True)
class _Week:
    """One week being built: its season and week, the season's frames, and the inputs."""

    season: int
    week: int
    schedule_df: pl.DataFrame
    team_stats_df: pl.DataFrame
    inputs: SeasonInputs

    def substep(self, label: str) -> AbstractContextManager[None]:
        """Time a step into the season's timing totals when timing is on."""
        return _timed_substep(
            label, enabled=self.inputs.timing_enabled, totals=self.inputs.timing_totals
        )


def _week_team_stats(week: _Week, week_games: pl.DataFrame) -> pl.DataFrame:
    """Return season-to-date team stats before the week, blended toward the regressed prior.

    Teams without an earlier game this season (week 1, or postponed first games like
    MIA/TB 2017) take the regressed previous season outright.
    """
    inputs = week.inputs
    season_stats = week.team_stats_df.filter(pl.col("season") == week.season)
    with week.substep("aggregate_team_stats"):
        agg_stats = polars_utils.aggregate_team_stats_to_week(season_stats, week.week, week.season)
    if inputs.tr_stats_source == "scrape":
        agg_stats = agg_stats.drop(
            [column for column in constants.TR_STATS if column in agg_stats.columns]
        )

    teams_this_week = set(
        week_games.select("away_abbr").to_series().to_list()
        + week_games.select("home_abbr").to_series().to_list()
    )
    teams_with_stats = set()
    if agg_stats.height > 0 and "team_abbr" in agg_stats.columns:
        teams_with_stats = set(agg_stats.select("team_abbr").to_series().to_list())
    teams_needing_fallback = teams_this_week - teams_with_stats

    blend_played_teams = inputs.blend_stat_prior and agg_stats.height > 0
    if not (teams_needing_fallback or blend_played_teams) or week.season <= inputs.min_season:
        return agg_stats
    prior_season_stats = inputs.prior_season_stats
    if prior_season_stats is None:
        prior_season_stats = build_prior_season_stats(
            week.team_stats_df, week.season, min_season=inputs.min_season
        )
    if prior_season_stats is None:
        return agg_stats

    if blend_played_teams:
        with week.substep("blend_stat_prior"):
            agg_stats = polars_utils.blend_with_prior_stats(
                agg_stats, prior_season_stats, inputs.stat_prior_blend_games
            )
    fallback_stats = prior_season_stats.filter(
        pl.col("team_abbr").is_in(list(teams_needing_fallback))
    )
    if fallback_stats.height == 0:
        return agg_stats
    # Combine current season stats with fallback stats
    return pl.concat([agg_stats, fallback_stats]) if agg_stats.height > 0 else fallback_stats


def _add_record_features(merged: pl.DataFrame, week: _Week) -> pl.DataFrame:
    """Add each side's season-to-date W-L-T record, strictly before the week."""
    records_df = pl.DataFrame()
    with week.substep("record_features"):
        try:
            records_df = polars_utils.compute_team_records_before_week(
                week.schedule_df,
                season=week.season,
                week=week.week,
                include_postseason=False,
            )
        except ValueError:
            # Some unit tests use a minimal schedule fixture without scores/game_type.
            log.debug(
                "Skipping record feature computation for season %d week %d (schedule incomplete)",
                week.season,
                week.week,
            )

    # The season-to-date stat frame carries its own `games_played`, which a prior-season
    # fallback row fills with the *previous* season's game count. The record columns below
    # own these names (`constants.RECORD_FEATURE_COLUMNS`), so drop the stat-frame copies
    # first; otherwise the join suffixes the record values away and the published column
    # contradicts the `wins` / `losses` / `ties` it should agree with.
    merged = merged.drop(
        [column for column in _RECORD_OWNED_STAT_COLUMNS if column in merged.columns]
    )

    if records_df.height > 0:
        away_records = _prefix_team_records(records_df, "away")
        home_records = _prefix_team_records(records_df, "home")
        merged = merged.join(away_records, on="away_abbr", how="left").join(
            home_records, on="home_abbr", how="left"
        )

    # Week 1 (and edge cases) may have no record rows; ensure columns exist and fill with 0.
    return merged.with_columns(
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


def _merge_elo(merged: pl.DataFrame, week: _Week) -> pl.DataFrame:
    """Join the week's ELO ratings, or each team's latest ELO when the week has none."""
    elo_df = week.inputs.elo_df
    if elo_df is None or elo_df.height == 0:
        return merged
    with week.substep("merge_elo"):
        season_elo = elo_df.filter(pl.col("season") == week.season)
        if "week" not in season_elo.columns:
            return merged
        week_elo = season_elo.filter(pl.col("week") == week.week)
        if week_elo.height > 0:
            # Exact week match - drop season/week from ELO before merge
            elo_cols = [c for c in week_elo.columns if c not in ["season", "week"]]
            return merged.join(week_elo.select(elo_cols), on=["away_abbr", "home_abbr"], how="left")
        # No ELO for this specific week - use most recent ELO per team
        # This handles future weeks and playoff games
        latest_elo = polars_utils.get_latest_elo_by_team(elo_df, week.season)
        if latest_elo.height == 0:
            return merged
        for side in ("away", "home"):
            side_elo = latest_elo.rename(
                {
                    "team_abbr": f"{side}_abbr",
                    "elo_pre": f"{side}_elo_pre",
                    "qb_value_pre": f"{side}_qb_value_pre",
                    "qb_elo_pre": f"{side}_qb_elo_pre",
                }
            )
            merged = merged.join(side_elo, on=f"{side}_abbr", how="left")
        return merged


def _add_team_rankings_features(merged: pl.DataFrame, week: _Week) -> pl.DataFrame:
    """Join the TeamRankings ratings and add the last-5 versus last-10 rating trend."""
    inputs = week.inputs
    with week.substep("merge_team_rankings"):
        merged = _merge_team_rankings(
            merged,
            week.season,
            week.week,
            TeamRankingsFrames(inputs.tr_df, inputs.prev_tr_df),
            tr_stats_source=inputs.tr_stats_source,
        )

    if not {
        "away_last_5_games_rating",
        "away_last_10_games_rating",
        "home_last_5_games_rating",
        "home_last_10_games_rating",
    }.issubset(merged.columns):
        return merged
    return merged.with_columns(
        [
            (pl.col("away_last_5_games_rating") - pl.col("away_last_10_games_rating")).alias(
                "away_last_5_games_rating_trend"
            ),
            (pl.col("home_last_5_games_rating") - pl.col("home_last_10_games_rating")).alias(
                "home_last_5_games_rating_trend"
            ),
        ]
    )


def _add_schedule_context_features(merged: pl.DataFrame, week: _Week) -> pl.DataFrame:
    """Add the divisional flag and the lookahead and motivation features."""
    with week.substep("add_divisional_feature"):
        merged = polars_utils.add_divisional_matchup_feature(merged)

    # Lookahead / next-week context features (null when schedule context is unavailable)
    with week.substep("add_lookahead_features"):
        try:
            merged = polars_utils.add_lookahead_features(
                merged,
                week.schedule_df,
                season=week.season,
                week=week.week,
                include_postseason=False,
            )
        except ValueError:
            log.debug(
                "Skipping lookahead features for season %d week %d (schedule incomplete)",
                week.season,
                week.week,
            )

    # Motivation / standings proxy features (null when schedule results are unavailable)
    with week.substep("add_motivation_features"):
        try:
            merged = polars_utils.add_motivation_features(
                merged,
                week.schedule_df,
                season=week.season,
                week=week.week,
                include_postseason=False,
            )
        except ValueError:
            log.debug(
                "Skipping motivation features for season %d week %d (schedule incomplete)",
                week.season,
                week.week,
            )
    return merged


def _add_strength_features(merged: pl.DataFrame, week: _Week) -> pl.DataFrame:
    """Join schedule-adjusted team strength, solved from games strictly before the week."""
    inputs = week.inputs
    with week.substep("strength_features"):
        prior_strength_snapshot = inputs.prior_strength_snapshot
        if prior_strength_snapshot is None and inputs.blend_strength_prior:
            prior_strength_snapshot = build_prior_strength_snapshot(
                week.team_stats_df, week.season, min_season=inputs.min_season
            )
        strength_table = build_strength_table(
            week.team_stats_df,
            week.schedule_df,
            season=week.season,
            week=week.week,
            prior_snapshot=prior_strength_snapshot if inputs.blend_strength_prior else None,
        )
        if inputs.strength_snapshots is not None:
            # Recorded before the join below keeps only the teams playing this week.
            inputs.strength_snapshots.append(
                _stamp_strength_snapshot(strength_table, season=week.season, week=week.week)
            )
        return _merge_strength_features(
            merged, strength_table.select("team_abbr", *constants.ADJUSTED_STRENGTH_STATS)
        )


def _fill_trend_and_coach_defaults(merged: pl.DataFrame) -> pl.DataFrame:
    """Fill missing trend features and coach history with neutral defaults."""
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
    return merged.with_columns(
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


def process_week(
    season: int,
    week: int,
    schedule_df: pl.DataFrame,
    team_stats_df: pl.DataFrame,
    inputs: SeasonInputs,
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
        inputs: The run's other inputs and options, with the season's precomputed
            features where `process_season` built them.

    Returns:
        DataFrame with week's games and features

    """
    # Skip the very first week of the first processed season (no prior data to aggregate)
    if season == inputs.min_season and week == 1:
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

    this_week = _Week(season, week, schedule_df, team_stats_df, inputs)
    agg_stats = _week_team_stats(this_week, week_games)
    if agg_stats.height == 0:
        log.debug("No aggregated stats available for season %d week %d", season, week)
        return pl.DataFrame()

    # Merge schedule with aggregated team stats
    with this_week.substep("merge_team_stats"):
        merged = polars_utils.merge_schedule_with_team_stats(week_games, agg_stats)

    # Season phase features (normalized week + early/mid/late buckets)
    merged = polars_utils.add_season_phase_features(merged, season=season, week=week)
    merged = _add_record_features(merged, this_week)
    merged = _merge_elo(merged, this_week)

    # Merge trend features derived from ELO/QB history and team stats
    merged = _merge_team_trends(merged, inputs.team_elo_trends)
    merged = _merge_team_trends(merged, inputs.team_stat_trends)
    merged = _merge_qb_trends(merged, inputs.qb_trends)
    merged = _merge_coach_features(merged, inputs.coach_features)

    merged = _add_team_rankings_features(merged, this_week)
    merged = _add_schedule_context_features(merged, this_week)
    merged = _add_strength_features(merged, this_week)

    # Calculate stat differentials
    with this_week.substep("calculate_differentials"):
        stats_to_diff = polars_utils.get_stats_for_diff()
        merged = polars_utils.calculate_stat_differentials(merged, stats_to_diff)

    return _fill_trend_and_coach_defaults(merged)


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
    return merged.join(home_trends, on=["season", "week", "home_abbr"], how="left")


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
    return merged.join(home_trends, on=["season", "week", "home_qb"], how="left")


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
    return merged.join(home_features, on=["season", "week", "home_abbr"], how="left")


class TeamRankingsFrames(NamedTuple):
    """TeamRankings rows for the season being built and for the season before it."""

    current: pl.DataFrame | None
    previous: pl.DataFrame | None


def _merge_team_rankings(
    merged: pl.DataFrame,
    season: int,
    week: int,
    rankings: TeamRankingsFrames,
    *,
    tr_stats_source: str = "scrape",
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
        rankings: TeamRankings rows for the current and the previous season
        tr_stats_source: Whether TR situational columns come from the scrape or from PBP

    Returns:
        DataFrame with TR columns merged

    """
    tr_df, prev_tr_df = rankings
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
        # When the PBP source is selected for the situational percentages, TeamRankings still
        # contributes only its ratings.
        expected_tr_cols = set(constants.TR_RATINGS)
        if tr_stats_source == "scrape":
            expected_tr_cols.update(constants.TR_STATS)
        available_tr_cols = [c for c in tr_to_use.columns if c in expected_tr_cols]

        if not available_tr_cols:
            log.warning(
                "TR data has no expected columns. Available: %s",
                tr_to_use.columns[:5],
            )
            return merged

        # Select only the columns we want plus team_abbr
        tr_to_use = tr_to_use.select(["team_abbr", *available_tr_cols])

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

    return pl.read_csv(file_path)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
