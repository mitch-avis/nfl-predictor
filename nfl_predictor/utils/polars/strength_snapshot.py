"""Build the pre-week schedule-adjusted team strength snapshot for one season week.

A snapshot answers one question: given only the games played strictly before week
``N`` of a season, how strong is each team, with the difficulty of the games it has
already played divided out?

The backbone is the simultaneous ridge in
:mod:`nfl_predictor.utils.polars.adjusted_strength`, which estimates every team's
offense and defense coefficient jointly. That joint estimation is the all-hops
generalization of the head-to-head-excluded opponent-profiling method published by
the separate, read-only ``nfl-sos-ratings`` project, which is credited here as the
source of the method, of the independent per-side centering, and of the default
display-composite weights.

Leakage contract
----------------
A snapshot for week ``N`` is solved from games with ``week < N`` in that season and
nothing else. Playoff weeks are the one documented exception: they use the whole
regular season, matching how every other season-to-date feature in this repository
treats a postseason row. Games from other seasons never enter the solve; the
previous season reaches a snapshot only through the explicit, ablatable prior blend
below. The team universe is supplied by the caller from the season's schedule rather
than inferred from the prior-week games, so a team on a bye -- or a whole league before
week 1 has been played -- still gets a row carrying the prior. Which teams exist in a
season is known before kickoff and carries no outcome information.

Responses
---------
Two per-game rates drive the ridge, both expressed per offensive snap so that pace
does not masquerade as quality::

    pass_epa_per_snap = pass_epa_sum / offensive_snaps
    rush_epa_per_snap = rush_epa_sum / offensive_snaps

Two simple rating systems ride alongside on the same prior-week rows::

    adj_srs   = SRS over (points_scored - points_allowed)
    st_rating = SRS over (st_epa_for - st_epa_against) / st_plays

The shared home-field term
--------------------------
``adj_hfa`` is the home-field coefficient of the **passing** solve, not a blend of the
two responses and not a separate fit. The two responses are solved independently and
each yields its own home-field term; the passing one is published because passing
carries most of the per-snap EPA signal. It falls back to the rushing solve's term only
when the passing response has no usable rows. It is one league-wide value per
``(season, week)``, so it is not published to the game schema.

Sign convention
---------------
The ridge models a team-game as ``offense[team] - defense[opponent] + hfa``. The
defense coefficient therefore enters with a minus sign, so a **larger**
``adj_def_*`` value is a **better** defense: it is the quantity that suppresses the
opponent's per-snap output. The display composite adds the defensive components
with positive weights for the same reason.
"""

from collections.abc import Sequence

import numpy as np
import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars.adjusted_strength import solve_srs, solve_team_ridge

# Ridge penalty, frozen for reproducibility instead of tuned per solve.
#
# Provenance: chosen once offline by the deterministic 5-fold cross-validation in
# `adjusted_strength.tune_ridge_lambda` over `numpy.logspace(-6, 2, 17)`, evaluated on
# 128 real pre-week snapshots (seasons 2005, 2010, 2015, 2019, 2021, 2022, 2023 and 2024
# at week cutoffs 3, 5, 7, 9, 12, 14, 16 and 18, each solve seeing 32 to 512 team-game
# rows). The median selected penalty is 10.0 at *every* cutoff, early weeks included, so
# a single frozen value costs nothing against per-solve tuning while keeping the ETL
# deterministic. The reference implementation tunes per solve; this repository does not.
#
# Known property, not a defect: a penalty this size relative to per-snap EPA (~0.0x)
# shrinks coefficients noticeably toward zero, and it shrinks them *more* when fewer
# games have been played. Simulation against planted effects recovers roughly 30% of
# true coefficient magnitude at 4 games per team, rising to about 50% by 17. Team
# ordering is essentially unaffected (the property CV selects for), but the raw
# `adj_off_*`/`adj_def_*` columns are therefore on a scale that drifts across the season:
# the same numeric value means a stronger team in December than in September. Consumers
# that need a scale-stable measure should prefer `adj_strength_composite`, which is
# standardized within each snapshot and so is immune to this.
STRENGTH_RIDGE_LAMBDA = 10.0

# Default weights for the display composite, over within-snapshot standardized
# components. Source: the published team composite of the read-only `nfl-sos-ratings`
# project, whose weights are fit to predict the following season's adjusted
# performance rather than to correlate with wins. They are a display default only:
# every component is also published in raw adjusted units so the model can weight
# them itself.
COMPOSITE_WEIGHTS: dict[str, float] = {
    "adj_off_pass_epa_snap": 0.3828739475913225,
    "adj_off_rush_epa_snap": 0.19062479977967036,
    "adj_def_pass_epa_snap": 0.27163464954613765,
    "adj_def_rush_epa_snap": 0.0973640754908631,
    "st_rating": 0.0575025275920063,
}

_TEAM = "team_abbr"
_OPPONENT = "opponent_abbr"
_HOME = "is_home"
_PASSING_RESPONSE = "_pass_epa_per_snap"
_RUSH_RESPONSE = "_rush_epa_per_snap"
_MARGIN_RESPONSE = "_point_margin"
_ST_RESPONSE = "_st_epa_margin_per_play"

# Which solved coefficient feeds which published column, per ridge response.
_RIDGE_SPECS: tuple[tuple[str, str, str], ...] = (
    (_PASSING_RESPONSE, "adj_off_pass_epa_snap", "adj_def_pass_epa_snap"),
    (_RUSH_RESPONSE, "adj_off_rush_epa_snap", "adj_def_rush_epa_snap"),
)

# Rating columns the early-season prior blend applies to. `adj_strength_composite`
# is deliberately absent: it is rebuilt from the blended components afterwards so it
# always describes the values actually published.
_BLENDED_COLUMNS: tuple[str, ...] = (
    "adj_off_pass_epa_snap",
    "adj_off_rush_epa_snap",
    "adj_def_pass_epa_snap",
    "adj_def_rush_epa_snap",
    "adj_srs",
    "st_rating",
)

_SNAPSHOT_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    _TEAM: pl.String,
    **dict.fromkeys(constants.STRENGTH_SNAPSHOT_STATS, pl.Float64),
}


def empty_snapshot() -> pl.DataFrame:
    """Return a typed, empty snapshot frame with the published column order."""
    return pl.DataFrame(schema=_SNAPSHOT_SCHEMA)


def _cutoff_week(season: int, week: int) -> int:
    """Return the highest game week a snapshot for ``week`` is allowed to see.

    Regular-season rows see everything strictly before their own week. Playoff rows
    see the whole regular season, matching ``aggregate_team_stats_to_week``.
    """
    regular_season_weeks = constants.get_regular_season_weeks(season)
    if week > regular_season_weeks:
        return regular_season_weeks
    return week - 1


def _numeric(column: str) -> pl.Expr:
    """Return a column cast to float, or a typed null literal when it is absent."""
    return pl.col(column).cast(pl.Float64, strict=False)


def _with_responses(team_games: pl.DataFrame) -> pl.DataFrame:
    """Attach the four per-game solve responses as ratios of that game's own sums.

    Formulas:
        - ``pass_epa_per_snap = pass_epa_sum / offensive_snaps``
        - ``rush_epa_per_snap = rush_epa_sum / offensive_snaps``
        - ``point_margin = points_scored - points_allowed``
        - ``st_epa_margin_per_play = (st_epa_for - st_epa_against) / st_plays``

    A response whose source columns are missing is emitted as null, so the matching
    solve degrades to an all-null rating rather than raising.
    """
    columns = set(team_games.columns)

    def ratio(numerator: str, denominator: str) -> pl.Expr:
        if numerator not in columns or denominator not in columns:
            return pl.lit(None, dtype=pl.Float64)
        return (
            pl.when(_numeric(denominator) > 0.0)
            .then(_numeric(numerator) / _numeric(denominator))
            .otherwise(None)
        )

    if {"points_scored", "points_allowed"} <= columns:
        margin = _numeric("points_scored") - _numeric("points_allowed")
    else:
        margin = pl.lit(None, dtype=pl.Float64)

    if {"st_epa_for", "st_epa_against", "st_plays"} <= columns:
        st_margin = (
            pl.when(_numeric("st_plays") > 0.0)
            .then((_numeric("st_epa_for") - _numeric("st_epa_against")) / _numeric("st_plays"))
            .otherwise(None)
        )
    else:
        st_margin = pl.lit(None, dtype=pl.Float64)

    return team_games.with_columns(
        ratio("pass_epa_sum", "offensive_snaps").alias(_PASSING_RESPONSE),
        ratio("rush_epa_sum", "offensive_snaps").alias(_RUSH_RESPONSE),
        margin.alias(_MARGIN_RESPONSE),
        st_margin.alias(_ST_RESPONSE),
    )


def _season_teams(team_games: pl.DataFrame, season: int) -> list[str]:
    """Return every team appearing in the season's played games, sorted.

    This is the fallback when the caller does not supply the season's team list. It can
    only see teams that have already played, so before a season kicks off it is empty;
    callers that hold the schedule should pass `teams` instead.
    """
    if "season" not in team_games.columns:
        return []
    season_rows = team_games.filter(pl.col("season") == season)
    labels: set[str] = set()
    for column in (_TEAM, _OPPONENT):
        if column in season_rows.columns:
            labels.update(season_rows.get_column(column).drop_nulls().cast(pl.String).to_list())
    return sorted(labels)


def _usable_rows(prior_games: pl.DataFrame, response: str) -> pl.DataFrame:
    """Return the rows a solve can actually use for one response.

    Drops nulls *and* non-finite values. NaN is not null, so `drop_nulls` alone would let
    one bad cell reach the normal equations, and a single NaN there returns NaN for every
    team's coefficient rather than for the one row that caused it.
    """
    if response not in prior_games.columns:
        return prior_games.clear()
    return prior_games.filter(pl.col(response).is_finite())


def _solve_components(prior_games: pl.DataFrame, teams: list[str]) -> tuple[pl.DataFrame, float]:
    """Solve every rating component on the prior-week rows and return one row per team."""
    frame = pl.DataFrame({_TEAM: teams}, schema={_TEAM: pl.String})
    home_field = float("nan")

    # `_RIDGE_SPECS` is ordered so the passing response is solved first, which makes the
    # published home-field term the passing solve's; see the module docstring.
    for response, offense_col, defense_col in _RIDGE_SPECS:
        usable = _usable_rows(prior_games, response)
        if usable.is_empty():
            frame = frame.with_columns(
                pl.lit(None, dtype=pl.Float64).alias(offense_col),
                pl.lit(None, dtype=pl.Float64).alias(defense_col),
            )
            continue
        ratings, solved_hfa = solve_team_ridge(
            usable,
            response,
            ridge_lambda=STRENGTH_RIDGE_LAMBDA,
            team_col=_TEAM,
            opponent_col=_OPPONENT,
            home_col=_HOME,
        )
        if np.isnan(home_field):
            home_field = solved_hfa
        frame = frame.join(
            ratings.rename(
                {"offense_rating": offense_col, "defense_rating": defense_col},
            ),
            on=_TEAM,
            how="left",
        )

    for response, column in ((_MARGIN_RESPONSE, "adj_srs"), (_ST_RESPONSE, "st_rating")):
        usable = _usable_rows(prior_games, response)
        if usable.is_empty():
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
            continue
        frame = frame.join(
            solve_srs(usable, response, team_col=_TEAM, opponent_col=_OPPONENT).rename(
                {"srs_rating": column}
            ),
            on=_TEAM,
            how="left",
        )

    hfa = None if np.isnan(home_field) else home_field
    return frame.with_columns(pl.lit(hfa, dtype=pl.Float64).alias("adj_hfa")), home_field


def _games_played(prior_games: pl.DataFrame, teams: list[str]) -> pl.DataFrame:
    """Return the count of prior in-season games behind each team's solve."""
    base = pl.DataFrame({_TEAM: teams}, schema={_TEAM: pl.String})
    if _TEAM not in prior_games.columns:
        return base.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("strength_games_played"))
    counts = (
        prior_games.group_by(_TEAM)
        .agg(pl.len().cast(pl.Float64).alias("strength_games_played"))
        .cast({_TEAM: pl.String})
    )
    return base.join(counts, on=_TEAM, how="left").with_columns(
        pl.col("strength_games_played").fill_null(0.0)
    )


def _blend_prior(
    solved: pl.DataFrame,
    prior_snapshot: pl.DataFrame,
) -> pl.DataFrame:
    """Blend the in-season solve with the regressed previous-season snapshot.

    Formulas:
        ``regressed_prior = prior * (1 - WEEK1_REGRESSION_FACTOR)``
        ``weight = games / (games + PRIOR_BLEND_GAMES)``
        ``blended = weight * in_season + (1 - weight) * regressed_prior``

    Ratings are centered on zero by construction, so regressing toward the league
    mean is regressing toward zero. When only one side is available that side is
    used unchanged, which is what makes a Week-1 row equal the regressed prior and a
    team with no prior season equal its raw in-season solve.
    """
    available = [column for column in _BLENDED_COLUMNS if column in prior_snapshot.columns]
    prior = prior_snapshot.select(
        pl.col(_TEAM).cast(pl.String),
        *[
            (_numeric(column) * (1.0 - constants.WEEK1_REGRESSION_FACTOR)).alias(f"_prior_{column}")
            for column in available
        ],
    ).unique(subset=[_TEAM], keep="first")

    games = pl.col("strength_games_played")
    weight = games / (games + constants.PRIOR_BLEND_GAMES)
    return solved.join(prior, on=_TEAM, how="left").with_columns(
        *[
            pl.when(pl.col(column).is_null())
            .then(pl.col(f"_prior_{column}"))
            .when(pl.col(f"_prior_{column}").is_null())
            .then(pl.col(column))
            .otherwise(weight * pl.col(column) + (1.0 - weight) * pl.col(f"_prior_{column}"))
            .alias(column)
            for column in available
        ]
    )


def _with_composite(snapshot: pl.DataFrame) -> pl.DataFrame:
    """Attach the weighted display composite over standardized components.

    Formula: each component is standardized across the teams in this snapshot
    (``z = (value - mean) / population_std``), multiplied by its documented weight,
    and summed. Only components that are present and non-null for a team contribute,
    and the sum is divided by the weights actually used, so a season without
    special-teams data still yields a comparable composite instead of a null. A
    component with no spread across the league contributes nothing rather than
    dividing by zero.
    """
    weighted_total = pl.lit(0.0, dtype=pl.Float64)
    weight_total = pl.lit(0.0, dtype=pl.Float64)

    for column, weight in COMPOSITE_WEIGHTS.items():
        if column not in snapshot.columns:
            continue
        spread = pl.col(column).std(ddof=0)
        standardized = (
            pl.when(spread.is_null() | (spread == 0.0))
            .then(0.0)
            .otherwise((pl.col(column) - pl.col(column).mean()) / spread)
        )
        contributes = pl.col(column).is_not_null()
        weighted_total = weighted_total + pl.when(contributes).then(
            standardized * weight
        ).otherwise(0.0)
        weight_total = weight_total + pl.when(contributes).then(weight).otherwise(0.0)

    return snapshot.with_columns(
        pl.when(weight_total > 0.0)
        .then(weighted_total / weight_total)
        .otherwise(None)
        .alias("adj_strength_composite")
    )


def build_strength_snapshot(
    team_games: pl.DataFrame,
    *,
    season: int,
    week: int,
    prior_snapshot: pl.DataFrame | None = None,
    blend_prior: bool = True,
    teams: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Build the pre-week schedule-adjusted strength snapshot for one season week.

    Args:
        team_games: One row per team-game with ``season``, ``week``, ``team_abbr``,
            ``opponent_abbr``, ``is_home`` and the play-by-play sums and scoring
            columns the responses are built from. Rows from other seasons and from
            week ``week`` onward are filtered out here, so the caller may pass the
            whole history.
        season: Season the snapshot is for.
        week: Week the snapshot is for. Values are solved from games strictly before
            it, or from the whole regular season for a playoff week.
        prior_snapshot: The previous season's final snapshot, keyed by team. Supplies
            the early-season prior; ``None`` means a team with no in-season games
            gets nulls.
        blend_prior: Set to ``False`` to ablate the prior entirely and publish the
            raw in-season solve.
        teams: The season's full team list, normally taken from the schedule. Supply it
            so that a team yet to play still gets a row carrying the prior; without it
            the universe is drawn from played games only, which is empty before a season
            kicks off.

    Returns:
        One row per team in the season with ``team_abbr`` followed by
        ``constants.STRENGTH_SNAPSHOT_STATS``. Empty input returns a typed empty
        frame.

    """
    season_teams = sorted(set(teams)) if teams else _season_teams(team_games, season)
    if not season_teams:
        return empty_snapshot()

    cutoff = _cutoff_week(season, week)
    # A frame with no games (or without the identity columns to filter on) is not an
    # error: before a season kicks off there is nothing to solve, and the teams still
    # need rows so the prior can be published for them.
    if team_games.is_empty() or not {"season", "week"} <= set(team_games.columns):
        prior_games = _with_responses(team_games.clear())
    else:
        prior_games = _with_responses(
            team_games.filter((pl.col("season") == season) & (pl.col("week") <= cutoff))
        )

    solved, _ = _solve_components(prior_games, season_teams)
    solved = solved.join(_games_played(prior_games, season_teams), on=_TEAM, how="left")

    if blend_prior and prior_snapshot is not None and not prior_snapshot.is_empty():
        solved = _blend_prior(solved, prior_snapshot)

    solved = _with_composite(solved)
    return solved.select(
        pl.col(name).cast(_SNAPSHOT_SCHEMA[name])
        for name in (_TEAM, *constants.STRENGTH_SNAPSHOT_STATS)
    ).sort(_TEAM)
