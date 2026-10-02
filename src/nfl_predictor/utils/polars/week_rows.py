"""One week's game rows for the ETL, with every pre-week feature joined on.

`process_week` takes a season's schedule and the per-game team stats and returns the week's
games with season-to-date team stats (blended toward the regressed previous season), records,
ELO, trend and coach features, TeamRankings ratings, schedule context and schedule-adjusted
strength, plus the away-minus-home differentials. Every value is known before the week's
kickoff: game results enter only from earlier weeks.
"""

import time
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars.strength_snapshot import StrengthPrior
from nfl_predictor.utils.polars.strength_table import (
    build_prior_strength_snapshot,
    build_strength_table,
    stamp_strength_snapshot,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nfl_predictor.data_collection import SeasonInputs


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
        deduped = features.unique(subset=["team_abbr"], keep="first", maintain_order=True)
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
            prior=(
                StrengthPrior(prior_strength_snapshot)
                if inputs.blend_strength_prior and prior_strength_snapshot is not None
                else None
            ),
        )
        if inputs.strength_snapshots is not None:
            # Recorded before the join below keeps only the teams playing this week.
            inputs.strength_snapshots.append(
                stamp_strength_snapshot(strength_table, season=week.season, week=week.week)
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
