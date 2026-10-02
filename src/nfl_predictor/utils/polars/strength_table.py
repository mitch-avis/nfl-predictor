"""Per-week schedule-adjusted strength table for the ETL.

For one season week this combines the pre-week ridge snapshot (`strength_snapshot`) with the
two schedule-strength lenses (`schedule_strength`) into one row per scheduled team, solved
from games strictly before the week. The ETL joins these values onto the week's game rows
and records every week's table, bye teams included, in the strength snapshot file.
"""

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars import schedule_strength, strength_snapshot

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


def stamp_strength_snapshot(table: pl.DataFrame, *, season: int, week: int) -> pl.DataFrame:
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
