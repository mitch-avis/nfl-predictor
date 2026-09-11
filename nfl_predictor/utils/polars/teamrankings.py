"""TeamRankings integration helpers (Polars).

Loads, validates, scrapes, and merges TeamRankings.com team-week data.
Implementation was split out of `nfl_predictor.utils.polars_utils`.
"""

import os

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars.loaders import _is_numeric_dtype
from nfl_predictor.utils.scraping_utils import (
    get_current_nfl_week,
    get_missing_tr_columns,
    get_week_date,
    merge_tr_data,
    normalize_team_column,
    save_team_rankings_week,
    scrape_team_rankings_for_week,
    update_season_team_rankings,
)


def _get_required_tr_columns() -> set[str]:
    """Get the set of required TeamRankings columns."""
    required = {"team_abbr", "week"}
    required.update(constants.TR_RATINGS)
    required.update(constants.TR_STATS)
    return required


def _validate_tr_dataframe(tr_df: pl.DataFrame) -> tuple[bool, list[str]]:
    """Validate that a TeamRankings DataFrame has all required columns.

    Args:
        tr_df: TeamRankings DataFrame to validate

    Returns:
        Tuple of (is_valid, list of missing columns)

    """
    if tr_df.height == 0:
        return False, list(_get_required_tr_columns())

    required_cols = _get_required_tr_columns()
    actual_cols = set(tr_df.columns)
    missing = required_cols - actual_cols

    return len(missing) == 0, list(missing)


def load_team_rankings(
    season: int,
    current_season: int | None = None,
    current_week: int | None = None,
    *,
    min_week: int = 1,
) -> pl.DataFrame:
    """Load TeamRankings data for a season, scraping fresh data for current/future weeks.

    For historical weeks (past seasons or completed weeks of current season), this loads
    from cached CSV files. For current week or future weeks of the current season, it
    scrapes fresh data from TeamRankings.com.

    If cached data is missing required columns, only the missing columns will be scraped
    and merged with the existing data (smart scraping).

    Supports playoff weeks (weeks 19-22 for seasons 2021+, weeks 18-21 for earlier seasons).

    Args:
        season: Season year to load
        current_season: Current NFL season (if None, will be determined)
        current_week: Current NFL week (if None, will be determined)
        min_week: Earliest week to load (used to skip known-missing weeks)

    Returns:
        Polars DataFrame with TeamRankings ratings/stats per team per week

    """
    if season < constants.TEAMRANKINGS_MIN_SEASON:
        log.info(
            "Skipping TeamRankings for season %d (data starts in %d).",
            season,
            constants.TEAMRANKINGS_MIN_SEASON,
        )
        return pl.DataFrame()

    # Determine current season/week if not provided
    if current_season is None or current_week is None:
        current_season, current_week = get_current_nfl_week()

    season_dir = os.path.join(constants.DATA_PATH, str(season))
    os.makedirs(season_dir, exist_ok=True)

    # Determine weeks to load for this season (including playoffs)
    regular_season_weeks = constants.get_regular_season_weeks(season)
    # Playoff weeks: WC, DIV, CON, SB = 4 additional weeks
    max_playoff_week = regular_season_weeks + 4

    min_week = max(min_week, 1)
    if season == constants.TEAMRANKINGS_MIN_SEASON:
        min_week = max(min_week, constants.TEAMRANKINGS_MIN_WEEK)

    if season < current_season:
        # Past season: load all regular season weeks plus playoff weeks
        weeks_to_load = list(range(min_week, max_playoff_week + 1))
    elif season == current_season:
        # Current season: load weeks 1 through current week + a few future weeks
        # Future weeks use current week's data as a placeholder
        weeks_to_load = list(range(min_week, min(current_week + 3, max_playoff_week + 1)))
    else:
        # Future season: no data to load
        return pl.DataFrame()

    if not weeks_to_load:
        return pl.DataFrame()

    existing_data = []
    weeks_to_full_scrape = []  # Weeks that need complete scraping (no file or empty)
    weeks_to_partial_scrape = []  # Weeks with files that need additional columns

    # Check each week file for validity
    for week in weeks_to_load:
        week_file = os.path.join(season_dir, f"{season}_week_{week:02d}_team_rankings.csv")

        force_refresh = season == current_season and week == current_week

        if os.path.exists(week_file):
            try:
                week_df = pl.read_csv(week_file)
            except (
                pl.exceptions.ComputeError,
                pl.exceptions.NoDataError,
                OSError,
            ) as e:
                log.warning("Failed to read TR file for week %d: %s", week, e)
                weeks_to_full_scrape.append((week, None))
                continue

            week_df = _normalize_tr_dataframe(week_df)

            if week_df.height == 0:
                # Empty file, need full scrape
                weeks_to_full_scrape.append((week, None))
                continue

            is_valid, missing_cols = _validate_tr_dataframe(week_df)
            if force_refresh:
                # Always refresh the current week for the in-progress season; keep cached
                # data as a fallback if scraping fails.
                weeks_to_full_scrape.append((week, week_df if is_valid else None))
            else:
                if is_valid:
                    log.debug(
                        "Using cached TeamRankings data for season %d week %d (%s)",
                        season,
                        week,
                        week_file,
                    )
                    existing_data.append(week_df)
                else:
                    # File exists but missing columns - need partial scrape
                    log.debug(
                        "Week %d TR file missing %d columns: %s",
                        week,
                        len(missing_cols),
                        missing_cols[:3],
                    )
                    weeks_to_partial_scrape.append((week, week_df, missing_cols))
        else:
            # No file - need full scrape for past weeks
            if season < current_season or week <= current_week:
                weeks_to_full_scrape.append((week, None))
            # Future weeks of current season will get current week's data copied later

    # Full scrape for weeks without any data
    if weeks_to_full_scrape:
        log.info(
            "Full scraping %d weeks of TR data for season %d",
            len(weeks_to_full_scrape),
            season,
        )
        for week, fallback_df in weeks_to_full_scrape:
            week_date = get_week_date(season, week)
            scraped_df = scrape_team_rankings_for_week(week, week_date)

            if scraped_df.height > 0:
                save_team_rankings_week(scraped_df, season, week)
                existing_data.append(scraped_df)
            else:
                if fallback_df is not None and fallback_df.height > 0:
                    log.warning(
                        "Failed to scrape TR data for season %d week %d; using cached fallback",
                        season,
                        week,
                    )
                    existing_data.append(fallback_df)
                else:
                    log.warning("Failed to scrape TR data for season %d week %d", season, week)

    # Partial scrape for weeks with files missing some columns
    if weeks_to_partial_scrape:
        log.info(
            "Partial scraping %d weeks of TR data for season %d (missing columns only)",
            len(weeks_to_partial_scrape),
            season,
        )
        for week, existing_week_df, _missing_cols in weeks_to_partial_scrape:
            # Determine which ratings and stats to scrape
            missing_ratings, missing_stats = get_missing_tr_columns(existing_week_df)

            if not missing_ratings and not missing_stats:
                # No columns to scrape, use existing data
                existing_data.append(existing_week_df)
                continue

            log.debug(
                "Week %d: scraping %d ratings, %d stats",
                week,
                len(missing_ratings),
                len(missing_stats),
            )

            week_date = get_week_date(season, week)
            scraped_df = scrape_team_rankings_for_week(
                week,
                week_date,
                ratings_to_scrape=missing_ratings,
                stats_to_scrape=missing_stats,
            )

            if scraped_df.height > 0:
                # Merge new columns with existing data
                merged_df = merge_tr_data(existing_week_df, scraped_df)
                save_team_rankings_week(merged_df, season, week)
                existing_data.append(merged_df)
            else:
                # Scraping failed, use existing data anyway
                log.warning(
                    "Failed to scrape missing columns for season %d week %d, using existing data",
                    season,
                    week,
                )
                existing_data.append(existing_week_df)

    # Update the consolidated season file if we scraped anything
    if weeks_to_full_scrape or weeks_to_partial_scrape:
        update_season_team_rankings(season)

    # For current season, copy current week data to future weeks if needed
    if season == current_season and existing_data:
        # Get the latest scraped data (current week)
        current_week_data = None
        for df in existing_data:
            if "week" in df.columns:
                week_vals = df.select(pl.col("week")).to_series().to_list()
                if current_week in week_vals:
                    current_week_data = df.filter(pl.col("week") == current_week)
                    break

        if current_week_data is not None and current_week_data.height > 0:
            start_week = max(current_week + 1, min_week)
            for future_week in range(start_week, max_playoff_week + 1):
                # Always overwrite future-week caches with the most recently scraped week.
                # This prevents stale future-week placeholders from persisting across runs.
                future_data = current_week_data.with_columns(pl.lit(future_week).alias("week"))
                save_team_rankings_week(future_data, season, future_week)
                existing_data.append(future_data)

    # Combine all data
    if existing_data:
        # Cast week column to consistent type before concat
        existing_data = [df.cast({"week": pl.Int64}) for df in existing_data]
        combined = pl.concat(existing_data, how="diagonal")
        # Remove duplicates by keeping latest data for each team/week
        combined = combined.unique(subset=["team_abbr", "week"], keep="last")
        return combined.sort(["week", "team_abbr"])

    log.debug("No TeamRankings data found for season %d", season)
    return pl.DataFrame()


def _normalize_tr_dataframe(tr_df: pl.DataFrame) -> pl.DataFrame:
    """Normalize a TeamRankings DataFrame.

    Args:
        tr_df: Raw TeamRankings DataFrame

    Returns:
        Normalized DataFrame

    """
    # Drop unnamed index column if present
    if "" in tr_df.columns:
        tr_df = tr_df.drop("")

    # Rename 'abbr' to 'team_abbr' for consistency (only if abbr exists and team_abbr doesn't)
    if "abbr" in tr_df.columns and "team_abbr" not in tr_df.columns:
        tr_df = tr_df.rename({"abbr": "team_abbr"})
    elif "abbr" in tr_df.columns and "team_abbr" in tr_df.columns:
        # If both exist, drop abbr
        tr_df = tr_df.drop("abbr")

    # Normalize team abbreviations
    if "team_abbr" in tr_df.columns:
        tr_df = normalize_team_column(tr_df, "team_abbr")

    return tr_df


def get_latest_team_rankings(tr_df: pl.DataFrame) -> pl.DataFrame:
    """Get the most recent TeamRankings values for each team.

    This is used for:
    - Playoff games (use end-of-regular-season TR values)
    - Future weeks without specific TR data

    Args:
        tr_df: TeamRankings DataFrame for a season

    Returns:
        DataFrame with latest TR values per team (week column removed)

    """
    if tr_df.height == 0:
        return pl.DataFrame()

    if "week" not in tr_df.columns:
        return tr_df

    # Sort by week descending and take first per team
    tr_sorted = tr_df.sort("week", descending=True)

    # Get non-week columns for aggregation
    non_key_cols = [c for c in tr_df.columns if c not in ["week", "team_abbr"]]

    # Group by team and take first (most recent) value for each column
    agg_exprs = [pl.col(c).first() for c in non_key_cols]
    latest_tr = tr_sorted.group_by("team_abbr").agg(agg_exprs)

    return latest_tr


def aggregate_team_stats_to_week(
    team_stats_df: pl.DataFrame,
    target_week: int,
    season: int,
) -> pl.DataFrame:
    """Aggregate team statistics up to (but not including) a target week.

    This creates rolling averages of team performance metrics that can be used
    as features for predicting the target week's games.

    For regular season games, uses stats from prior weeks within the season.
    For playoff games (after regular season), uses the full regular season stats.

    Note: Before 2021, the NFL had 17-week regular seasons (playoffs started week 18).
          From 2021 onwards, the NFL has 18-week regular seasons (playoffs start week 19).

    Args:
        team_stats_df: DataFrame with per-game team statistics
        target_week: The week we're predicting (stats up to week-1 are used)
        season: The season year

    Returns:
        DataFrame with aggregated team statistics

    """
    # Get the number of regular season weeks for this season
    regular_season_weeks = constants.get_regular_season_weeks(season)

    # For playoff games (week > regular season weeks), use full regular season stats
    # For regular season, use only games before target_week
    if target_week > regular_season_weeks:
        max_week_to_include = regular_season_weeks
    else:
        max_week_to_include = target_week - 1

    # Filter to games in the specified week range
    prior_games = team_stats_df.filter(
        (pl.col("season") == season) & (pl.col("week") <= max_week_to_include)
    )

    if prior_games.height == 0:
        # Use debug level for week 1 (expected) - this is handled by regression logic
        log.debug("No prior games found for season %s week %s", season, target_week)
        return pl.DataFrame()

    # Identify numeric columns for aggregation (excluding identifiers)
    exclude_cols = {"season", "week", "team_abbr", "opponent_abbr", "season_type"}
    numeric_cols = [
        col
        for col in prior_games.columns
        if col not in exclude_cols and prior_games[col].dtype in [pl.Float64, pl.Int32, pl.Int64]
    ]

    # Aggregate by team
    agg_exprs = [pl.col(c).mean().alias(c) for c in numeric_cols]
    agg_exprs.append(pl.len().alias("games_played"))

    agg_df = prior_games.group_by("team_abbr").agg(agg_exprs)

    # Compute derived ratio metrics from the aggregated averages
    agg_df = _compute_derived_metrics(agg_df)

    return agg_df


# Play-by-play rate specs: (output name, numerator column, denominator column).
# Each rate is a ratio of season-to-date sums. `aggregate_team_stats_to_week` stores the
# mean per-game count, and mean(numerator) / mean(denominator) equals
# sum(numerator) / sum(denominator) because the game count cancels, so dividing the means
# is the same number as the ratio of sums and never a mean of per-game rates.
_PBP_SIMPLE_RATE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("off_pass_epa_per_snap", "pass_epa_sum", "offensive_snaps"),
    ("off_rush_epa_per_snap", "rush_epa_sum", "offensive_snaps"),
    ("def_pass_epa_allowed_per_snap", "pass_epa_allowed_sum", "defensive_snaps"),
    ("def_rush_epa_allowed_per_snap", "rush_epa_allowed_sum", "defensive_snaps"),
    ("epa_per_dropback", "pass_epa_sum", "dropbacks"),
    ("epa_per_carry", "rush_epa_sum", "carries"),
    ("epa_per_dropback_allowed", "pass_epa_allowed_sum", "dropbacks_allowed"),
    ("epa_per_carry_allowed", "rush_epa_allowed_sum", "carries_allowed"),
    ("pass_success_rate", "pass_success_count", "dropbacks"),
    ("rush_success_rate", "rush_success_count", "carries"),
    ("pass_success_rate_allowed", "pass_success_allowed_count", "dropbacks_allowed"),
    ("rush_success_rate_allowed", "rush_success_allowed_count", "carries_allowed"),
    ("explosive_pass_rate", "explosive_pass_count", "dropbacks"),
    ("explosive_rush_rate", "explosive_rush_count", "carries"),
    ("explosive_pass_rate_allowed", "explosive_pass_allowed_count", "dropbacks_allowed"),
    ("explosive_rush_rate_allowed", "explosive_rush_allowed_count", "carries_allowed"),
    ("stuffed_rush_rate", "stuffed_rush_count", "carries"),
    ("stuffed_rush_rate_allowed", "stuffed_rush_allowed_count", "carries_allowed"),
    ("early_down_pass_rate", "early_down_passes", "early_down_plays"),
)


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr, output: str) -> pl.Expr:
    """Build a null-on-zero-denominator ratio expression.

    Args:
        numerator: Expression for the numerator
        denominator: Expression for the denominator
        output: Output column name

    Returns:
        Polars expression aliased to `output`, null where the denominator is not positive

    """
    return (
        pl.when(denominator > 0)
        .then(numerator / denominator)
        .otherwise(None)
        .cast(pl.Float64)
        .alias(output)
    )


def _compute_pbp_derived_metrics(agg_df: pl.DataFrame) -> pl.DataFrame:
    """Compute play-by-play rate metrics from aggregated per-game counts.

    Rates are ratios of season-to-date sums, never means of per-game rates. Because
    aggregation stores the mean per-game count, dividing two means is exactly the ratio of
    the underlying sums (the game count cancels).

    Formulas:
        - off_pass_epa_per_snap = pass_epa_sum / offensive_snaps
        - off_rush_epa_per_snap = rush_epa_sum / offensive_snaps
        - def_pass_epa_allowed_per_snap = pass_epa_allowed_sum / defensive_snaps
        - def_rush_epa_allowed_per_snap = rush_epa_allowed_sum / defensive_snaps
        - epa_per_dropback = pass_epa_sum / dropbacks
        - epa_per_carry = rush_epa_sum / carries
        - epa_per_dropback_allowed = pass_epa_allowed_sum / dropbacks_allowed
        - epa_per_carry_allowed = rush_epa_allowed_sum / carries_allowed
        - pass_success_rate = pass_success_count / dropbacks
        - rush_success_rate = rush_success_count / carries
        - pass_success_rate_allowed = pass_success_allowed_count / dropbacks_allowed
        - rush_success_rate_allowed = rush_success_allowed_count / carries_allowed
        - success_rate = (pass_success_count + rush_success_count) / (dropbacks + carries)
        - success_rate_allowed = (pass_success_allowed_count + rush_success_allowed_count)
          / (dropbacks_allowed + carries_allowed)
        - explosive_pass_rate = explosive_pass_count / dropbacks
        - explosive_rush_rate = explosive_rush_count / carries
        - explosive_pass_rate_allowed = explosive_pass_allowed_count / dropbacks_allowed
        - explosive_rush_rate_allowed = explosive_rush_allowed_count / carries_allowed
        - stuffed_rush_rate = stuffed_rush_count / carries
        - stuffed_rush_rate_allowed = stuffed_rush_allowed_count / carries_allowed
        - early_down_pass_rate = early_down_passes / early_down_plays
        - epa_margin_per_play = (pass_epa_sum + rush_epa_sum) / offensive_snaps
          - (pass_epa_allowed_sum + rush_epa_allowed_sum) / defensive_snaps
        - st_epa_margin_per_play = (st_epa_for - st_epa_against) / st_plays

    Every rate is null when its denominator is zero or when its source counts are absent,
    so a season without play-by-play still emits the invariant schema.

    Args:
        agg_df: DataFrame with averaged per-game play-by-play counts

    Returns:
        DataFrame with the play-by-play rate columns added

    """
    available = set(agg_df.columns)
    derived: list[pl.Expr] = []

    for output, numerator, denominator in _PBP_SIMPLE_RATE_SPECS:
        if numerator in available and denominator in available:
            derived.append(_safe_ratio(pl.col(numerator), pl.col(denominator), output))

    # Overall success rate combines the pass and rush components on both sides.
    success_specs = (
        ("success_rate", ("pass_success_count", "rush_success_count"), ("dropbacks", "carries")),
        (
            "success_rate_allowed",
            ("pass_success_allowed_count", "rush_success_allowed_count"),
            ("dropbacks_allowed", "carries_allowed"),
        ),
    )
    for output, numerators, denominators in success_specs:
        if available.issuperset(numerators) and available.issuperset(denominators):
            derived.append(
                _safe_ratio(
                    pl.col(numerators[0]) + pl.col(numerators[1]),
                    pl.col(denominators[0]) + pl.col(denominators[1]),
                    output,
                )
            )

    # EPA margin per play: offensive EPA per offensive snap minus EPA allowed per
    # defensive snap, so pace on either side of the ball cannot distort the margin.
    margin_sources = (
        "pass_epa_sum",
        "rush_epa_sum",
        "offensive_snaps",
        "pass_epa_allowed_sum",
        "rush_epa_allowed_sum",
        "defensive_snaps",
    )
    if available.issuperset(margin_sources):
        offense = pl.when(pl.col("offensive_snaps") > 0).then(
            (pl.col("pass_epa_sum") + pl.col("rush_epa_sum")) / pl.col("offensive_snaps")
        )
        defense = pl.when(pl.col("defensive_snaps") > 0).then(
            (pl.col("pass_epa_allowed_sum") + pl.col("rush_epa_allowed_sum"))
            / pl.col("defensive_snaps")
        )
        derived.append((offense - defense).cast(pl.Float64).alias("epa_margin_per_play"))

    # Special-teams EPA margin per special-teams play.
    if available.issuperset(("st_epa_for", "st_epa_against", "st_plays")):
        derived.append(
            _safe_ratio(
                pl.col("st_epa_for") - pl.col("st_epa_against"),
                pl.col("st_plays"),
                "st_epa_margin_per_play",
            )
        )

    if derived:
        agg_df = agg_df.with_columns(derived)

    # Keep the schema invariant when a season has no play-by-play source at all.
    missing = [
        pl.lit(None, dtype=pl.Float64).alias(stat)
        for stat in constants.PBP_STATS
        if stat not in agg_df.columns
    ]
    if missing:
        agg_df = agg_df.with_columns(missing)

    return agg_df


def _compute_derived_metrics(agg_df: pl.DataFrame) -> pl.DataFrame:
    """Compute derived ratio metrics from aggregated team statistics.

    These metrics require division and should be computed after averaging
    raw stats to avoid ratio-of-averages issues.

    Metrics computed:
        - yards_per_point: total_yards / points_scored
        - opponent_yards_per_point: opponent_total_yards / points_allowed
        - yards_per_point_margin: yards_per_point - opponent_yards_per_point
        - points_per_play: points_scored / (pass_attempts + rush_attempts + times_sacked)
        - opponent_points_per_play: points_allowed / opponent_total_plays
        - points_per_play_margin: points_per_play - opponent_points_per_play
        - penalty_yards_per_penalty: penalty_yards / penalties
        - opponent_penalty_yards_per_penalty: opponent equivalent
        - the play-by-play rate family (see `_compute_pbp_derived_metrics`)

    Args:
        agg_df: DataFrame with averaged team statistics

    Returns:
        DataFrame with additional derived metric columns

    """
    derived_cols = []

    # --- Yards per point metrics ---
    # Yards per point = total_yards / points_scored
    if "total_yards" in agg_df.columns and "points_scored" in agg_df.columns:
        derived_cols.append(
            pl.when(pl.col("points_scored") > 0)
            .then(pl.col("total_yards") / pl.col("points_scored"))
            .otherwise(pl.lit(0.0))
            .alias("yards_per_point")
        )

    # Opponent yards per point (from opponent_total_yards and points_allowed)
    if "opponent_total_yards" in agg_df.columns and "points_allowed" in agg_df.columns:
        derived_cols.append(
            pl.when(pl.col("points_allowed") > 0)
            .then(pl.col("opponent_total_yards") / pl.col("points_allowed"))
            .otherwise(pl.lit(0.0))
            .alias("opponent_yards_per_point")
        )

    # Apply yards_per_point metrics first so we can calculate margin
    if derived_cols:
        agg_df = agg_df.with_columns(derived_cols)
        derived_cols = []

    # Yards per point margin = yards_per_point - opponent_yards_per_point
    if "yards_per_point" in agg_df.columns and "opponent_yards_per_point" in agg_df.columns:
        derived_cols.append(
            (pl.col("yards_per_point") - pl.col("opponent_yards_per_point")).alias(
                "yards_per_point_margin"
            )
        )

    # --- Points per play metrics ---
    # Total plays = pass_attempts + rush_attempts + times_sacked
    has_plays = all(c in agg_df.columns for c in ["pass_attempts", "rush_attempts", "times_sacked"])
    if has_plays and "points_scored" in agg_df.columns:
        total_plays = pl.col("pass_attempts") + pl.col("rush_attempts") + pl.col("times_sacked")
        derived_cols.append(
            pl.when(total_plays > 0)
            .then(pl.col("points_scored") / total_plays)
            .otherwise(pl.lit(0.0))
            .alias("points_per_play")
        )

    # Opponent points per play
    has_opp_plays = all(
        c in agg_df.columns
        for c in [
            "opponent_pass_attempts",
            "opponent_rush_attempts",
            "opponent_times_sacked",
        ]
    )
    if has_opp_plays and "points_allowed" in agg_df.columns:
        opp_total_plays = (
            pl.col("opponent_pass_attempts")
            + pl.col("opponent_rush_attempts")
            + pl.col("opponent_times_sacked")
        )
        derived_cols.append(
            pl.when(opp_total_plays > 0)
            .then(pl.col("points_allowed") / opp_total_plays)
            .otherwise(pl.lit(0.0))
            .alias("opponent_points_per_play")
        )

    # Apply points_per_play metrics first so we can calculate margin
    if derived_cols:
        agg_df = agg_df.with_columns(derived_cols)
        derived_cols = []

    # Points per play margin
    if "points_per_play" in agg_df.columns and "opponent_points_per_play" in agg_df.columns:
        derived_cols.append(
            (pl.col("points_per_play") - pl.col("opponent_points_per_play")).alias(
                "points_per_play_margin"
            )
        )

    # --- Penalty efficiency metrics ---
    # Penalty yards per penalty = penalty_yards / penalties
    if "penalty_yards" in agg_df.columns and "penalties" in agg_df.columns:
        derived_cols.append(
            pl.when(pl.col("penalties") > 0)
            .then(pl.col("penalty_yards") / pl.col("penalties"))
            .otherwise(pl.lit(0.0))
            .alias("penalty_yards_per_penalty")
        )

    # Opponent penalty yards per penalty
    opp_pen_yards = "opponent_penalty_yards"
    opp_pens = "opponent_penalties"
    if opp_pen_yards in agg_df.columns and opp_pens in agg_df.columns:
        derived_cols.append(
            pl.when(pl.col(opp_pens) > 0)
            .then(pl.col(opp_pen_yards) / pl.col(opp_pens))
            .otherwise(pl.lit(0.0))
            .alias("opponent_penalty_yards_per_penalty")
        )

    if derived_cols:
        agg_df = agg_df.with_columns(derived_cols)

    return _compute_pbp_derived_metrics(agg_df)


def recompute_derived_metrics(agg_df: pl.DataFrame) -> pl.DataFrame:
    """Recompute derived ratio metrics after the aggregated counts have been rewritten.

    Regression toward the league mean rewrites the summed components a ratio is built
    from, so every derived rate has to be recomputed from the regressed sums. Without
    this the fallback would publish counts regressed toward the mean alongside ratios
    still computed from the unregressed values.

    Args:
        agg_df: Aggregated team statistics whose counts have been modified

    Returns:
        DataFrame with the derived ratio metrics recomputed

    """
    return _compute_derived_metrics(agg_df)


def blend_with_prior_stats(
    in_season: pl.DataFrame,
    prior: pl.DataFrame,
    prior_blend_games: float,
) -> pl.DataFrame:
    """Blend season-to-date per-game means toward the regressed previous season.

    Formulas:
        ``weight = games_played / (games_played + prior_blend_games)``
        ``blended = weight * in_season_mean + (1 - weight) * regressed_prior_mean``

    The blend is applied to the per-game means of the underlying counts and sums, and
    every derived ratio is then recomputed from the blended values, so a published rate
    stays a ratio of blended sums rather than a blend of two rates. `games_played` keeps
    the in-season count. When only one side has a value for a team and column, that side
    is published unchanged, so a team with no previous-season row keeps its raw in-season
    means.

    Args:
        in_season: Aggregated season-to-date stats, one row per team, with `games_played`
        prior: Regressed previous-season stats, one row per team
        prior_blend_games: Games at which the two sides are weighted equally; positive

    Returns:
        The in-season frame, blended, with the same columns in the same order

    """
    if in_season.height == 0 or prior.height == 0 or "team_abbr" not in prior.columns:
        return in_season

    blended_columns = [
        column
        for column in in_season.columns
        if column not in {"team_abbr", "games_played"}
        and column in prior.columns
        and in_season.schema[column].is_numeric()
        and prior.schema[column].is_numeric()
    ]
    if not blended_columns:
        return in_season

    prior_side = prior.select(
        pl.col("team_abbr"),
        *[pl.col(column).cast(pl.Float64).alias(f"_prior_{column}") for column in blended_columns],
    ).unique(subset=["team_abbr"], keep="first", maintain_order=True)

    games = pl.col("games_played").cast(pl.Float64)
    weight = games / (games + prior_blend_games)
    blended = (
        in_season.join(prior_side, on="team_abbr", how="left", maintain_order="left")
        .with_columns(
            pl.when(pl.col(f"_prior_{column}").is_null())
            .then(pl.col(column))
            .when(pl.col(column).is_null())
            .then(pl.col(f"_prior_{column}"))
            .otherwise(weight * pl.col(column) + (1.0 - weight) * pl.col(f"_prior_{column}"))
            .alias(column)
            for column in blended_columns
        )
        .select(in_season.columns)
    )
    return recompute_derived_metrics(blended)


def calculate_league_means(team_stats_df: pl.DataFrame, season: int) -> dict[str, float]:
    """Calculate league-wide mean statistics for a given season.

    Args:
        team_stats_df: DataFrame with per-game team statistics
        season: The season year to calculate means for

    Returns:
        Dictionary mapping stat names to their league-wide mean values

    """
    season_stats = team_stats_df.filter(pl.col("season") == season)

    if season_stats.height == 0:
        return {}

    # Identify numeric columns
    exclude_cols = {"season", "week", "team_abbr", "opponent_abbr", "season_type"}
    numeric_cols = [
        col
        for col in season_stats.columns
        if col not in exclude_cols and season_stats[col].dtype in [pl.Float64, pl.Int32, pl.Int64]
    ]

    # Calculate means
    means = {}
    for col in numeric_cols:
        mean_val = season_stats.select(pl.col(col).mean()).item()
        if mean_val is not None:
            means[col] = float(mean_val)

    return means


def regress_to_mean(
    team_stats: pl.DataFrame,
    league_means: dict[str, float],
    regression_factor: float = 1 / 3,
) -> pl.DataFrame:
    """Regress team statistics toward league mean.

    For week-1 games, we regress prior season stats toward the league mean
    to account for roster changes, coaching changes, and mean reversion.

    The formula is: regressed_stat = team_stat * (1 - factor) + league_mean * factor

    Args:
        team_stats: DataFrame with team statistics
        league_means: Dictionary of league-wide mean values
        regression_factor: Fraction to regress toward mean (default 1/3)

    Returns:
        DataFrame with regressed statistics

    """
    if not league_means:
        return team_stats

    regression_exprs = []
    for col in team_stats.columns:
        if col in league_means and col not in {"team_abbr", "games_played"}:
            # regressed = team_value * (1 - factor) + league_mean * factor
            regressed = (
                pl.col(col) * (1 - regression_factor)
                + pl.lit(league_means[col]) * regression_factor
            ).alias(col)
            regression_exprs.append(regressed)
        else:
            regression_exprs.append(pl.col(col))

    return team_stats.select(regression_exprs)


def get_stat_columns() -> list[str]:
    """Get the nflreadpy stat column names to use.

    Returns:
        List of stat column names

    """
    return constants.NFLREADPY_STATS.copy()


def get_pbp_columns() -> list[str]:
    """Get the play-by-play derived stat column names to publish.

    Returns:
        List of stat column names

    """
    return constants.PBP_STATS.copy()


def get_strength_columns() -> list[str]:
    """Get the schedule-adjusted strength stat column names to publish.

    These name their own offensive and defensive sides explicitly, so like the
    play-by-play stats they never receive the generic `opponent_` mirror.

    Returns:
        List of stat column names

    """
    return constants.ADJUSTED_STRENGTH_STATS.copy()


def get_elo_columns() -> list[str]:
    """Get the ELO column names.

    Returns:
        List of ELO column names

    """
    return constants.ELO_COLUMNS.copy()


def get_tr_columns() -> list[str]:
    """Get all TeamRankings column names (ratings + stats).

    Returns:
        List of TeamRankings column names

    """
    return constants.TR_RATINGS.copy() + constants.TR_STATS.copy()


def calculate_game_result(row_dict: dict) -> float | None:
    """Calculate game result from away team perspective.

    Args:
        row_dict: Dictionary with away_score and home_score

    Returns:
        1.0 if away team won, 0.0 if lost, 0.5 if tie, None if scores are missing

    """
    away_score = row_dict.get("away_score")
    home_score = row_dict.get("home_score")

    if away_score is None or home_score is None:
        return None

    if away_score > home_score:
        return 1.0
    if away_score < home_score:
        return 0.0
    return 0.5


def merge_schedule_with_team_stats(
    schedule_df: pl.DataFrame,
    agg_stats_df: pl.DataFrame,
) -> pl.DataFrame:
    """Merge schedule data with aggregated team statistics.

    Args:
        schedule_df: Schedule DataFrame with game matchups
        agg_stats_df: Aggregated team statistics DataFrame

    Returns:
        DataFrame with schedule and team stats merged for both away and home teams

    """
    # Prepare away team stats with prefix
    away_stats = agg_stats_df.rename(
        {col: f"away_{col}" for col in agg_stats_df.columns if col != "team_abbr"}
    ).rename({"team_abbr": "away_abbr"})

    # Prepare home team stats with prefix
    home_stats = agg_stats_df.rename(
        {col: f"home_{col}" for col in agg_stats_df.columns if col != "team_abbr"}
    ).rename({"team_abbr": "home_abbr"})

    # Join with schedule
    merged = schedule_df.join(away_stats, on="away_abbr", how="left")
    merged = merged.join(home_stats, on="home_abbr", how="left")

    return merged


def calculate_stat_differentials(
    df: pl.DataFrame,
    stats_to_diff: list[str],
) -> pl.DataFrame:
    """Calculate differentials between away and home team statistics.

    Args:
        df: DataFrame with away_* and home_* prefixed stat columns
        stats_to_diff: List of stat names (without prefix) to calculate differences for

    Returns:
        DataFrame with added *_diff columns

    """
    for stat in stats_to_diff:
        away_col = f"away_{stat}"
        home_col = f"home_{stat}"

        if away_col in df.columns and home_col in df.columns:
            away_dtype = df.schema.get(away_col)
            home_dtype = df.schema.get(home_col)

            # Only compute differentials on numeric columns; skip others to avoid type errors
            if away_dtype is None or home_dtype is None:
                continue

            if not (_is_numeric_dtype(away_dtype) and _is_numeric_dtype(home_dtype)):
                log.debug(
                    "Skipping diff for %s (non-numeric types: %s vs %s)",
                    stat,
                    away_dtype,
                    home_dtype,
                )
                continue

            df = df.with_columns((pl.col(away_col) - pl.col(home_col)).alias(f"{stat}_diff"))

    return df


def get_team_name(abbr: str) -> str | None:
    """Get the full team name from an abbreviation.

    Args:
        abbr: Team abbreviation (canonical or alias)

    Returns:
        Full team name or None if not found

    """
    canonical = constants.normalize_team_abbr(abbr)
    team_info = constants.TEAM_MAPPING.get(canonical)
    if team_info is None:
        return None
    team_name = team_info.get("name")
    return str(team_name) if team_name is not None else None


def filter_completed_games(schedule_df: pl.DataFrame) -> pl.DataFrame:
    """Filter schedule to only completed games (those with scores).

    Args:
        schedule_df: Schedule DataFrame

    Returns:
        DataFrame with only completed games

    """
    return schedule_df.filter(
        pl.col("away_score").is_not_null() & pl.col("home_score").is_not_null()
    )


def filter_upcoming_games(
    schedule_df: pl.DataFrame,
    season: int,
    week: int,
) -> pl.DataFrame:
    """Filter schedule to upcoming games for a specific week.

    Args:
        schedule_df: Schedule DataFrame
        season: Season year
        week: Week number

    Returns:
        DataFrame with upcoming games (no scores yet)

    """
    return schedule_df.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & (pl.col("away_score").is_null() | pl.col("home_score").is_null())
    )


def remove_diff_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Remove all differential columns (ending in _diff) from a DataFrame.

    This creates a non-ML version of the data for faster local usage.

    Args:
        df: DataFrame with differential columns

    Returns:
        DataFrame without _diff columns

    """
    non_diff_cols = [c for c in df.columns if not c.endswith("_diff")]
    return df.select(non_diff_cols)


def polars_to_pandas(df: pl.DataFrame):
    """Convert a Polars DataFrame to pandas for compatibility with existing code.

    Args:
        df: Polars DataFrame

    Returns:
        pandas DataFrame

    """
    return df.to_pandas()


def pandas_to_polars(df) -> pl.DataFrame:
    """Convert a pandas DataFrame to Polars.

    Args:
        df: pandas DataFrame

    Returns:
        Polars DataFrame

    """
    return pl.from_pandas(df)


def get_stats_for_diff() -> list[str]:
    """Get list of stats that should have differentials calculated.

    Excludes opponent stats that are duplicates of their non-opponent counterparts.

    Returns:
        List of stat names (without prefix) to calculate diffs for

    """
    all_stats = []

    # nflreadpy stats (base stats)
    all_stats.extend(get_stat_columns())

    # Play-by-play derived stats (allowed variants are explicit, so no opponent mirror)
    all_stats.extend(get_pbp_columns())

    # Schedule-adjusted strength stats (offense/defense named explicitly, no mirror)
    all_stats.extend(get_strength_columns())

    # Opponent stats (excluding duplicates)
    excluded = set(constants.EXCLUDE_FROM_OPPONENT_STATS)
    for stat in get_stat_columns():
        if stat not in excluded:
            all_stats.append(f"opponent_{stat}")

    # ELO columns
    all_stats.extend(get_elo_columns())

    # TeamRankings columns
    all_stats.extend(get_tr_columns())

    # Trend feature columns
    all_stats.extend(constants.TREND_FEATURE_COLUMNS)

    # Deduplicate
    seen = set()
    unique_stats = []
    for stat in all_stats:
        if stat not in seen:
            seen.add(stat)
            unique_stats.append(stat)

    return unique_stats
