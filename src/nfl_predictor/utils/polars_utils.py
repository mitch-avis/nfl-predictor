"""Compatibility facade for Polars-first ETL helpers.

The helpers live in the modules under `nfl_predictor.utils.polars`; this module re-exports the
ones the pipeline and tests import from here.
"""

from __future__ import annotations

from nfl_predictor.utils.polars.features import (
    add_divisional_matchup_feature,
    add_lookahead_features,
    add_motivation_features,
    add_season_phase_features,
    build_coach_features,
    build_qb_trends,
    build_team_elo_trends,
    build_team_stat_trends,
    compute_team_next_week_context,
    compute_team_records_before_week,
)
from nfl_predictor.utils.polars.finalize import (
    build_final_column_order,
    select_final_columns,
)
from nfl_predictor.utils.polars.loaders import (
    add_per_game_opponent_stats,
    add_scoring_data_to_team_stats,
    attach_team_stats_to_schedule,
    combine_stats,
    get_current_nfl_week,
    get_latest_elo_by_team,
    load_elo_ratings,
    load_pbp,
    load_raw_elo_data,
    load_schedule,
    load_team_stats,
)
from nfl_predictor.utils.polars.pbp import (
    aggregate_pbp_team_game_stats,
)
from nfl_predictor.utils.polars.teamrankings import (
    aggregate_team_stats_to_week,
    blend_with_prior_stats,
    calculate_league_means,
    calculate_stat_differentials,
    filter_completed_games,
    filter_upcoming_games,
    get_latest_team_rankings,
    get_pbp_columns,
    get_stats_for_diff,
    get_tr_columns,
    load_team_rankings,
    merge_schedule_with_team_stats,
    recompute_derived_metrics,
    regress_to_mean,
    remove_diff_columns,
)

__all__ = [
    "add_divisional_matchup_feature",
    "add_lookahead_features",
    "add_motivation_features",
    "add_per_game_opponent_stats",
    "add_scoring_data_to_team_stats",
    "add_season_phase_features",
    "aggregate_pbp_team_game_stats",
    "aggregate_team_stats_to_week",
    "attach_team_stats_to_schedule",
    "blend_with_prior_stats",
    "build_coach_features",
    "build_final_column_order",
    "build_qb_trends",
    "build_team_elo_trends",
    "build_team_stat_trends",
    "calculate_league_means",
    "calculate_stat_differentials",
    "combine_stats",
    "compute_team_next_week_context",
    "compute_team_records_before_week",
    "filter_completed_games",
    "filter_upcoming_games",
    "get_current_nfl_week",
    "get_latest_elo_by_team",
    "get_latest_team_rankings",
    "get_pbp_columns",
    "get_stats_for_diff",
    "get_tr_columns",
    "load_elo_ratings",
    "load_pbp",
    "load_raw_elo_data",
    "load_schedule",
    "load_team_rankings",
    "load_team_stats",
    "merge_schedule_with_team_stats",
    "recompute_derived_metrics",
    "regress_to_mean",
    "remove_diff_columns",
    "select_final_columns",
]
