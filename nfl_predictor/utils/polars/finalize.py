"""Final schema/column ordering helpers (Polars).

Builds and enforces the invariant output schema ordering for downstream ML.
Implementation was split out of `nfl_predictor.utils.polars_utils`.
"""

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars.teamrankings import (
    get_elo_columns,
    get_pbp_columns,
    get_stat_columns,
    get_tr_columns,
)


def build_final_column_order() -> list[str]:
    """Build the final column order according to specification.

    Order:
    1. Metadata columns (as defined in constants)
    2. away_<stat> columns (alphabetically sorted)
    3. away_opponent_<stat> columns (alphabetically sorted)
    4. home_<stat> columns (alphabetically sorted)
    5. home_opponent_<stat> columns (alphabetically sorted)
    6. <stat>_diff columns (alphabetically sorted)
    7. Lines/odds columns
    8. Result columns

    Note: Deduplicates columns and excludes opponent stats that are duplicates.
    Opponent versions are only generated for nflreadpy stats, not for ELO, TR, or
    play-by-play columns.

    Returns:
        Ordered list of column names

    """
    columns = []

    # 1. Metadata columns (fixed order)
    columns.extend(constants.METADATA_COLUMNS)

    # Get each column category separately
    elo_cols = get_elo_columns()
    tr_cols = get_tr_columns()
    nflreadpy_stats = get_stat_columns()
    pbp_stats = get_pbp_columns()

    # Build the base stat columns (non-opponent): ELO + TR + trends + nflreadpy + PBP stats.
    # Play-by-play stats name their allowed variants explicitly, so they never receive the
    # generic opponent_ mirror generated below for nflreadpy stats.
    base_stats = []
    base_stats.extend(elo_cols)
    base_stats.extend(tr_cols)
    base_stats.extend(constants.TREND_FEATURE_COLUMNS)
    base_stats.extend(nflreadpy_stats)
    base_stats.extend(pbp_stats)

    # Deduplicate
    seen = set()
    unique_base = []
    for col in base_stats:
        if col not in seen:
            seen.add(col)
            unique_base.append(col)

    # Separate opponent_ stats from non-opponent stats
    # (opponent_ stats already exist in nflreadpy data like opponent_third_down_pct)
    non_opponent_stats = [s for s in unique_base if not s.startswith("opponent_")]
    opponent_stats = [s for s in unique_base if s.startswith("opponent_")]

    # Generate opponent versions ONLY for nflreadpy stats (not ELO or TR columns)
    # These are created by add_per_game_opponent_stats()
    excluded = set(constants.EXCLUDE_FROM_OPPONENT_STATS)
    nflreadpy_non_opponent = [s for s in nflreadpy_stats if not s.startswith("opponent_")]
    for stat in nflreadpy_non_opponent:
        opp_stat = f"opponent_{stat}"
        if stat not in excluded and opp_stat not in opponent_stats:
            opponent_stats.append(opp_stat)

    # Sort both lists alphabetically
    non_opponent_stats_sorted = sorted(non_opponent_stats)
    opponent_stats_sorted = sorted(opponent_stats)

    # 2. away_<stat> columns (alphabetically)
    columns.extend([f"away_{s}" for s in non_opponent_stats_sorted])

    # 3. away_opponent_<stat> columns (alphabetically)
    columns.extend([f"away_{s}" for s in opponent_stats_sorted])

    # 4. home_<stat> columns (alphabetically)
    columns.extend([f"home_{s}" for s in non_opponent_stats_sorted])

    # 5. home_opponent_<stat> columns (alphabetically)
    columns.extend([f"home_{s}" for s in opponent_stats_sorted])

    # 6. diff columns (all stats, alphabetically)
    all_stats_for_diff = non_opponent_stats_sorted + opponent_stats_sorted
    columns.extend([f"{s}_diff" for s in sorted(all_stats_for_diff)])

    # 7. Lines/odds
    columns.extend(constants.LINES_COLUMNS)

    # 8. Results
    columns.extend(constants.RESULT_COLUMNS)

    return columns


def select_final_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Select and order final columns according to specification.

    Args:
        df: DataFrame with all computed columns

    Returns:
        DataFrame with only specified columns in correct order

    """
    final_order = build_final_column_order()

    missing_cols = [c for c in final_order if c not in df.columns]
    if missing_cols:
        # Enforce invariant schema: add missing expected columns as nulls.
        df = df.with_columns([pl.lit(None).alias(c) for c in missing_cols])
        log.info(
            "Schema enforcement: added %d missing columns as nulls (showing up to 10): %s",
            len(missing_cols),
            missing_cols[:10],
        )

    return df.select(final_order)
