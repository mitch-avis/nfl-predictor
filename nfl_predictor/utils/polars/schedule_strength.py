"""Opponent-averaged schedule-strength helpers.

The method implemented here is ported from the sibling ``nfl-sos-ratings`` project, which
originated the head-to-head-excluded opponent-profiling approach used by
:func:`compute_schedule_strength_raw`: for a subject team ``S`` and each opponent ``X`` that
``S`` has faced, ``X``'s profile is built from ``X``'s games against the rest of the league
with every ``S`` versus ``X`` meeting removed, so the profile cannot echo ``S``'s own play
back at ``S``. Those opponent profiles are then averaged with **equal weight per unique
opponent**, so a division rival faced twice counts once.

Three constructs live here, and all three are pure functions of their inputs:

``sos_played_adj``
    ``mean over unique opponents O faced in games with week < week of rating[O]``.

``sos_remaining_adj``
    ``mean over unique opponents O in games with week >= week of rating[O]``.

``sos_played_raw``
    ``mean over unique opponents X faced in games with week < week of margin(X | not S)``,
    where ``margin(X | not S)`` is ``X``'s EPA margin per play over ``X``'s games with
    ``week < week`` in the same season, excluding every game in which ``X``'s opponent was
    the subject ``S``.

The adjusted helpers never compute ratings; the caller supplies a pre-week ratings snapshot
and the same snapshot is applied to every opponent, which is what keeps the output free of
information from the week being predicted.

Aggregation rule: rates are ratios of sums taken after aggregation, never a mean of
per-game rates. :func:`compute_schedule_strength_raw` therefore prefers the sums form -
pass ``numerator_col`` and ``denominator_col`` (for example ``epa_margin_sum`` and
``total_play_count``) and each opponent's margin becomes ``sum(numerator) /
sum(denominator)``. Only when a caller has nothing but a pre-divided rate column does the
function fall back to the plain mean of that rate over the opponent's games.
"""

from __future__ import annotations

import polars as pl

PLAYED_ADJUSTED_COLUMN = "sos_played_adj"
REMAINING_ADJUSTED_COLUMN = "sos_remaining_adj"
PLAYED_RAW_COLUMN = "sos_played_raw"

_SCHEDULE_REQUIRED_COLUMNS = ("season", "week", "away_abbr", "home_abbr")

# Internal working column names. They are namespaced with a leading underscore so they do
# not collide with caller-supplied column names.
_SUBJECT = "_subject"
_OPPONENT = "_opponent"
_OPPONENT_FOE = "_opponent_foe"
_RATING = "_rating"
_OPPONENT_VALUE = "_opponent_value"


def _require_columns(frame: pl.DataFrame, required: tuple[str, ...], label: str) -> None:
    """Raise when ``frame`` is missing any of ``required``.

    Args:
        frame: DataFrame to inspect.
        required: Column names that must be present.
        label: Name used to describe the frame in the error message.

    Raises:
        ValueError: If any required column is absent.

    """
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _empty_adjusted_frame(team_col: str) -> pl.DataFrame:
    """Return the typed, empty result frame for :func:`compute_schedule_strength_adjusted`."""
    return pl.DataFrame(
        schema={
            team_col: pl.Utf8,
            PLAYED_ADJUSTED_COLUMN: pl.Float64,
            REMAINING_ADJUSTED_COLUMN: pl.Float64,
        }
    )


def _empty_raw_frame(team_col: str) -> pl.DataFrame:
    """Return the typed, empty result frame for :func:`compute_schedule_strength_raw`."""
    return pl.DataFrame(schema={team_col: pl.Utf8, PLAYED_RAW_COLUMN: pl.Float64})


def _schedule_to_team_games(schedule: pl.DataFrame, *, team_col: str) -> pl.DataFrame:
    """Expand one row per game into two rows, one per participating team.

    Args:
        schedule: Season-filtered schedule with `week`, `away_abbr` and `home_abbr`.
        team_col: Output name for the subject team column.

    Returns:
        DataFrame with `week`, ``team_col`` and the internal opponent column.

    """
    home_rows = schedule.select(
        pl.col("week"),
        pl.col("home_abbr").alias(team_col),
        pl.col("away_abbr").alias(_OPPONENT),
    )
    away_rows = schedule.select(
        pl.col("week"),
        pl.col("away_abbr").alias(team_col),
        pl.col("home_abbr").alias(_OPPONENT),
    )
    return pl.concat([home_rows, away_rows], how="vertical")


def _mean_opponent_rating(
    team_games: pl.DataFrame,
    rating_lookup: pl.DataFrame,
    *,
    team_col: str,
    out_col: str,
) -> pl.DataFrame:
    """Average opponent ratings with equal weight per unique opponent.

    Formula: ``out_col[team] = mean over unique opponents O of rating[O]``. Deduplicating
    the ``(team, opponent)`` pairs before the join is what makes a twice-faced opponent
    count once. Opponents absent from ``rating_lookup`` join to null and Polars' ``mean``
    skips nulls, so an unrated opponent is dropped from the mean rather than treated as a
    zero; a team whose opponents are all unrated averages an all-null group and yields null.

    Args:
        team_games: Team-game rows already restricted to the side being summarised.
        rating_lookup: Ratings keyed by the internal opponent column.
        team_col: Subject team column name.
        out_col: Name of the produced value column.

    Returns:
        DataFrame with ``team_col`` and ``out_col`` (Float64), one row per team present.

    """
    unique_pairs = team_games.select(team_col, _OPPONENT).unique()
    return (
        unique_pairs.join(rating_lookup, on=_OPPONENT, how="left")
        .group_by(team_col)
        .agg(pl.col(_RATING).mean().cast(pl.Float64).alias(out_col))
    )


def compute_schedule_strength_adjusted(
    schedule: pl.DataFrame,
    ratings: pl.DataFrame,
    *,
    season: int,
    week: int,
    rating_col: str = "rating",
    team_col: str = "team_abbr",
) -> pl.DataFrame:
    """Average a pre-week ratings snapshot over each team's played and remaining opponents.

    Formulas, both with equal weight per unique opponent::

        sos_played_adj[T]    = mean over unique O in games(T, season, week' <  week) rating[O]
        sos_remaining_adj[T] = mean over unique O in games(T, season, week' >= week) rating[O]

    A twice-faced opponent (a division rival, or a postseason rematch) counts **once** on
    each side, matching the equal-weight-per-opponent averaging of the ``nfl-sos-ratings``
    opponent profile. Every opponent is scored with the same supplied pre-week snapshot, so
    no rating from the week being predicted, or from any later week, can enter the result.

    Opponents that have no row in ``ratings`` contribute null and are excluded from the mean
    rather than counted as zero; if none of a team's opponents are rated the team's value is
    null. A team that has played nobody (week 1) gets a null ``sos_played_adj``, and a team
    with no games left gets a null ``sos_remaining_adj``.

    Args:
        schedule: Schedule with at least `season`, `week`, `away_abbr` and `home_abbr`.
            Rows from other seasons are ignored; every week of ``season`` is used, so
            postseason rows count toward the remaining side once they exist.
        ratings: Pre-week ratings snapshot with one row per team, keyed by ``team_col``.
        season: Season to summarise.
        week: Boundary week. Games strictly before it are "played", the rest "remaining".
        rating_col: Name of the rating value column in ``ratings``.
        team_col: Team key column in ``ratings`` and in the returned frame.

    Returns:
        DataFrame with ``[team_col, "sos_played_adj", "sos_remaining_adj"]``, one row per
        team appearing in the season's schedule, sorted by team, both value columns Float64.
        An empty, correctly typed frame when the season has no games.

    Raises:
        ValueError: If required columns are missing or ``ratings`` repeats a team.

    """
    _require_columns(schedule, _SCHEDULE_REQUIRED_COLUMNS, "schedule")
    _require_columns(ratings, (team_col, rating_col), "ratings")

    if ratings.select(team_col).n_unique() != ratings.height:
        raise ValueError("ratings must contain one rating per team")

    season_schedule = schedule.filter(pl.col("season") == season)
    if season_schedule.is_empty():
        return _empty_adjusted_frame(team_col)

    team_games = _schedule_to_team_games(season_schedule, team_col=team_col)
    teams = team_games.select(team_col).unique()
    rating_lookup = ratings.select(
        pl.col(team_col).alias(_OPPONENT),
        pl.col(rating_col).cast(pl.Float64).alias(_RATING),
    )

    played = _mean_opponent_rating(
        team_games.filter(pl.col("week") < week),
        rating_lookup,
        team_col=team_col,
        out_col=PLAYED_ADJUSTED_COLUMN,
    )
    remaining = _mean_opponent_rating(
        team_games.filter(pl.col("week") >= week),
        rating_lookup,
        team_col=team_col,
        out_col=REMAINING_ADJUSTED_COLUMN,
    )

    return (
        teams.join(played, on=team_col, how="left")
        .join(remaining, on=team_col, how="left")
        .select(team_col, PLAYED_ADJUSTED_COLUMN, REMAINING_ADJUSTED_COLUMN)
        .sort(team_col)
    )


def _opponent_margin_expr(
    *,
    margin_col: str,
    numerator_col: str | None,
    denominator_col: str | None,
) -> pl.Expr:
    """Build the per-opponent margin aggregation expression.

    Preferred (ratio of sums): ``sum(numerator) / sum(denominator)``, null when the summed
    denominator is not positive. Fallback (mean of rates): ``mean(margin_col)``, used only
    when the caller has a pre-divided per-game rate and no underlying sums.

    Args:
        margin_col: Per-game rate column used by the fallback path.
        numerator_col: Summable numerator column, or None.
        denominator_col: Summable denominator column, or None.

    Returns:
        Aggregation expression producing the internal per-opponent value column.

    """
    if numerator_col is not None and denominator_col is not None:
        denominator = pl.col(denominator_col).sum()
        return (
            pl.when(denominator > 0)
            .then(pl.col(numerator_col).sum() / denominator)
            .otherwise(None)
            .cast(pl.Float64)
            .alias(_OPPONENT_VALUE)
        )
    return pl.col(margin_col).mean().cast(pl.Float64).alias(_OPPONENT_VALUE)


def compute_schedule_strength_raw(
    team_games: pl.DataFrame,
    *,
    season: int,
    week: int,
    margin_col: str = "epa_margin_per_play",
    team_col: str = "team_abbr",
    opponent_col: str = "opponent_abbr",
    numerator_col: str | None = None,
    denominator_col: str | None = None,
) -> pl.DataFrame:
    """Average each faced opponent's head-to-head-excluded EPA margin per play.

    This is the one-hop opponent profile ported from ``nfl-sos-ratings``. For a subject
    team ``S`` at ``(season, week)``::

        opponents(S)      = unique X over S's games with week' < week
        margin(X | not S) = X's EPA margin per play over X's games with week' < week,
                            excluding every game whose opponent was S
        sos_played_raw[S] = mean over X in opponents(S) of margin(X | not S)

    The outer average is a plain mean with **equal weight per unique opponent**, so a
    division rival faced twice counts once. Excluding the head-to-head games is what makes
    the value a measure of the opponents rather than an echo of ``S``: perturbing an
    ``S``-versus-``X`` game cannot move ``sos_played_raw[S]``, while perturbing one of
    ``X``'s games against a third team does.

    ``margin(X | not S)`` follows the repo's aggregation rule. When ``numerator_col`` and
    ``denominator_col`` are both supplied it is the ratio of sums,
    ``sum(numerator) / sum(denominator)`` over the surviving games - the preferred form,
    because it weights each game by its play count. When they are omitted the function falls
    back to ``mean(margin_col)``, the plain mean of an already-divided per-game rate; that
    path weights every game equally regardless of play count, so prefer the sums form when
    the underlying sums are available.

    An opponent left with no games after the exclusion (its only prior meetings were with
    ``S``) contributes nothing and is dropped from the mean, as is an opponent whose summed
    denominator is zero. If no opponent survives, or the subject has no prior games at all
    (week 1), the value is null.

    Args:
        team_games: One row per team-game with `season`, `week`, ``team_col``,
            ``opponent_col`` and the margin inputs. Rows from other seasons are ignored.
        season: Season to summarise.
        week: Boundary week; only games strictly before it are used, on both hops, so
            current-week and postseason rows can never leak in.
        margin_col: Per-game EPA margin per play column, used for the mean-of-rates path.
        team_col: Subject team column, also the key of the returned frame.
        opponent_col: Opponent column.
        numerator_col: Optional summable margin numerator, e.g. ``epa_margin_sum``.
        denominator_col: Optional summable play count, e.g. ``total_play_count``.

    Returns:
        DataFrame with ``[team_col, "sos_played_raw"]``, one row per team appearing in the
        season, sorted by team, Float64 values. An empty, correctly typed frame when the
        season has no games.

    Raises:
        ValueError: If required columns are missing, or only one of ``numerator_col`` and
            ``denominator_col`` is supplied.

    """
    if (numerator_col is None) != (denominator_col is None):
        raise ValueError("numerator_col and denominator_col must be supplied together")

    required = ("season", "week", team_col, opponent_col)
    if numerator_col is not None and denominator_col is not None:
        required = (*required, numerator_col, denominator_col)
    else:
        required = (*required, margin_col)
    _require_columns(team_games, required, "team_games")

    season_games = team_games.filter(pl.col("season") == season)
    if season_games.is_empty():
        return _empty_raw_frame(team_col)

    teams = season_games.select(team_col).unique()
    prior_games = season_games.filter(pl.col("week") < week)

    subject_pairs = prior_games.select(
        pl.col(team_col).alias(_SUBJECT),
        pl.col(opponent_col).alias(_OPPONENT),
    ).unique()
    opponent_games = prior_games.select(
        pl.col(team_col).alias(_OPPONENT),
        pl.col(opponent_col).alias(_OPPONENT_FOE),
        pl.exclude(team_col, opponent_col),
    )

    per_opponent = (
        subject_pairs.join(opponent_games, on=_OPPONENT, how="inner")
        .filter(pl.col(_OPPONENT_FOE) != pl.col(_SUBJECT))
        .group_by(_SUBJECT, _OPPONENT)
        .agg(
            _opponent_margin_expr(
                margin_col=margin_col,
                numerator_col=numerator_col,
                denominator_col=denominator_col,
            )
        )
    )
    subject_means = per_opponent.group_by(_SUBJECT).agg(
        pl.col(_OPPONENT_VALUE).mean().cast(pl.Float64).alias(PLAYED_RAW_COLUMN)
    )

    return (
        teams.join(subject_means, left_on=team_col, right_on=_SUBJECT, how="left")
        .select(team_col, PLAYED_RAW_COLUMN)
        .sort(team_col)
    )
