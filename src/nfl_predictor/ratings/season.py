"""Every rating table of one season, built as ``nfl-sos-ratings`` writes its ratings files.

The team tables come from :func:`build_team_season`, the quarterback tables from
:func:`build_qb_season`; each table is returned in the row order that project writes it in, under
the name of its file (``{season}_<name>.parquet`` there):

- ``ratings``: one row per team, best first: games played, the offense, defense, special-teams and
  team ratings, the head-to-head-excluded ``sos``, and the score-based ``SRS``.
- ``team_prior``: the preseason prior means the season's fits used (empty once every team has
  played nine games): the season fit's (``excluded_team`` null, last) and each head-to-head
  refit's.
- ``ratings_by_week``: the team ratings fit on the games through each week.
- ``rating_ranges`` and ``rating_pairs``: the game bootstrap's rank ranges and head-to-head chances.
- ``rating_ranges_by_week``: each week's rank ranges, built only for a season in progress.
- ``qb_season``: every passer's season volume, qualifier, rating, and faced pass defense (the
  rating columns of that project's ``qb_combined`` file).
- ``qb_ratings``: the qualified passers, best first.
- ``qb_ratings_by_week``, ``qb_rating_ranges``, ``qb_rating_pairs``, and
  ``qb_rating_ranges_by_week``: as for teams, ranked among the qualified passers.

The team fit takes the previous season's cross-validated penalties and the season's preseason
prior from a :class:`~nfl_predictor.ratings.team_prior.PriorHistory`; the first season
cross-validates its own. The quarterback fit cross-validates the season's own penalty, as that
project's published quarterback ratings do.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor.ratings.qb_rating import (
    QB_FACED_DEFENSE_COLUMN,
    QB_ID_COLUMN,
    QB_RATING_COLUMN,
    QB_SUMMARY_COLUMNS,
    QbRatingFit,
    bootstrap_qb_ratings,
    compute_qb_faced_pass_defense,
    fit_qb_ratings,
    fit_qb_ratings_by_week,
    qb_season_summary,
)
from nfl_predictor.ratings.rating_ranges import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    QB_PAIR_COLUMNS,
    QB_RANGE_COLUMNS,
    RANGE_QUANTILES,
    TEAM_PAIR_COLUMNS,
    TEAM_RANGE_COLUMNS,
    TOP_RANKS,
    RangeColumns,
    quantile_suffix,
    summarize_rank_pairs,
    summarize_rank_ranges,
    summarize_unit_rank_ranges,
)
from nfl_predictor.ratings.team_prior import (
    PriorHistory,
    SeasonPrior,
    prior_without_team,
    snapshot_prior,
    team_prior_table,
)
from nfl_predictor.ratings.team_rating import (
    TEAM_RATING_COLUMNS,
    TeamRatingFit,
    bootstrap_team_ratings,
    compute_team_schedule_strength,
    fit_team_ratings,
    fit_team_ratings_by_week,
    fit_team_ratings_with_previous_penalties,
)
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars.adjusted_strength import solve_srs

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from nfl_predictor.ratings.ridge import UnitPrior

TEAM_RATINGS_ORDER = ("team", "games_played", *TEAM_RATING_COLUMNS, "sos", "SRS")
QB_RATINGS_ORDER = (
    QB_ID_COLUMN,
    "qb_name",
    "team",
    "qb_games_played",
    "qb_dropbacks_total",
    "qb_attempts_total",
    "qb_epa_per_dropback",
    QB_FACED_DEFENSE_COLUMN,
    QB_RATING_COLUMN,
)
_QB_IDENTITY = (QB_ID_COLUMN, "qb_name", "team")
# The row order of each table: ``(column, descending)`` pairs, nulls last, as written there.
PUBLISHED_ROW_ORDER: dict[str, tuple[tuple[str, bool], ...]] = {
    "ratings": (("team_rating", True), ("team", False)),
    "team_prior": (("excluded_team", False), ("team", False)),
    "ratings_by_week": (("team", False), ("week", False)),
    "rating_ranges": (("team_rank", False), ("team", False)),
    "rating_pairs": (("team", False), ("other_team", False)),
    "rating_ranges_by_week": (("team", False), ("week", False)),
    "qb_season": ((QB_ID_COLUMN, False),),
    "qb_ratings": ((QB_RATING_COLUMN, True), (QB_ID_COLUMN, False)),
    "qb_ratings_by_week": ((QB_ID_COLUMN, False), ("week", False)),
    "qb_rating_ranges": (("qb_rank", False), (QB_ID_COLUMN, False)),
    "qb_rating_pairs": ((QB_ID_COLUMN, False), ("other_qb_id", False)),
    "qb_rating_ranges_by_week": ((QB_ID_COLUMN, False), ("week", False)),
}


@dataclass(frozen=True, slots=True)
class BootstrapSettings:
    """How many game-bootstrap resamples to draw, and the seed of their generator.

    The defaults are the published ones: 1000 resamples from ``numpy.random.default_rng(0)``.
    """

    resamples: int = BOOTSTRAP_RESAMPLES
    seed: int = BOOTSTRAP_SEED


PUBLISHED_BOOTSTRAP = BootstrapSettings()


@dataclass(frozen=True, slots=True)
class TeamSeasonRatings:
    """One season's team rating tables, each in its published row order."""

    ratings: pl.DataFrame
    team_prior: pl.DataFrame
    ratings_by_week: pl.DataFrame
    rating_ranges: pl.DataFrame
    rating_pairs: pl.DataFrame
    rating_ranges_by_week: pl.DataFrame | None


@dataclass(frozen=True, slots=True)
class QbSeasonRatings:
    """One season's quarterback rating tables, each in its published row order."""

    qb_season: pl.DataFrame
    qb_ratings: pl.DataFrame
    qb_ratings_by_week: pl.DataFrame
    qb_rating_ranges: pl.DataFrame
    qb_rating_pairs: pl.DataFrame
    qb_rating_ranges_by_week: pl.DataFrame | None


def published_order(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    """Return ``frame`` in the row order of the table ``name`` (``PUBLISHED_ROW_ORDER``)."""
    order = PUBLISHED_ROW_ORDER[name]
    return frame.sort(
        [column for column, _ in order],
        descending=[descending for _, descending in order],
        nulls_last=True,
    )


def build_team_ratings(
    team_games: pl.DataFrame,
    fit: TeamRatingFit,
    *,
    prior_without: Callable[[str], UnitPrior | None] | None = None,
) -> pl.DataFrame:
    """Return one row per team with the season fit's ratings, ``sos``, and ``SRS``.

    Args:
        team_games: The season's team-game rows.
        fit: The season fit of ``team_games``.
        prior_without: The preseason prior source of the head-to-head-excluded refits.

    Returns:
        ``TEAM_RATINGS_ORDER`` columns, best team first. A team whose every game lacks a final
        score keeps its row with a null ``SRS`` (the score-based rating), and a warning names it.

    """
    games_played = team_games.group_by("team", maintain_order=True).len("games_played")
    schedule = compute_team_schedule_strength(team_games, fit, prior_without=prior_without)
    srs = solve_srs(team_games, "point_margin", team_col="team", opponent_col="opponent_team")
    ratings = (
        fit.ratings.join(schedule, on="team", how="left", maintain_order="left")
        .join(srs, on="team", how="left", maintain_order="left")
        .rename({"srs_rating": "SRS"})
        .join(games_played, on="team", how="left", maintain_order="left")
        .select(TEAM_RATINGS_ORDER)
        .sort("team_rating", descending=True)
    )
    unscored = sorted(ratings.filter(pl.col("SRS").is_null()).get_column("team").to_list())
    if unscored:
        log.warning("SRS: no final score for %s; their SRS is null", unscored)
    return ratings


def build_team_rank_summaries(
    team_games: pl.DataFrame,
    fit: TeamRatingFit,
    scrimmage_prior: UnitPrior | None = None,
    *,
    bootstrap: BootstrapSettings = PUBLISHED_BOOTSTRAP,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return every team's rank ranges and every ordered pair's head-to-head chances.

    Both come from one set of game-bootstrap resamples: each redraws the season's games with
    replacement and refits with ``fit``'s penalties and the season fit's preseason prior means;
    ranks are among all teams in the resample. The ranges cover the team rating and each unit.
    """
    draws = bootstrap_team_ratings(
        team_games,
        fit,
        resamples=bootstrap.resamples,
        seed=bootstrap.seed,
        scrimmage_prior=scrimmage_prior,
    )
    ranges = summarize_rank_ranges(draws, fit.ratings, TEAM_RANGE_COLUMNS).join(
        summarize_unit_rank_ranges(draws, fit.ratings), on="team", how="left", maintain_order="left"
    )
    return ranges, summarize_rank_pairs(draws, fit.ratings, TEAM_PAIR_COLUMNS)


def _weekly_rank_columns(columns: RangeColumns) -> list[str]:
    """Return a weekly rank-range table's columns after ``week``: rank percentiles and chances.

    Rating percentiles are left out, as each week's ratings use that week's own scale.
    """
    return [
        columns.id,
        columns.rank,
        *(f"{columns.rank}{quantile_suffix(level)}" for level in RANGE_QUANTILES),
        *(f"{columns.rank}_top{top}_probability" for top in TOP_RANKS),
        f"{columns.rank}_missing_share",
    ]


def _weekly_rank_schema(columns: RangeColumns) -> dict[str, pl.DataType]:
    """Return the column types of a weekly rank-range table with no rows."""
    schema: dict[str, pl.DataType] = {"week": pl.Int64(), columns.id: pl.String()}
    for column in _weekly_rank_columns(columns)[1:]:
        schema[column] = pl.Float64() if column.endswith(("_probability", "_share")) else pl.Int64()
    return schema


def build_team_rank_ranges_by_week(
    team_games: pl.DataFrame,
    fit: TeamRatingFit,
    season_prior: SeasonPrior | None = None,
    *,
    bootstrap: BootstrapSettings = PUBLISHED_BOOTSTRAP,
) -> pl.DataFrame:
    """Return every team's rank range as of each week, from the games through that week.

    Each week refits its games with ``fit``'s penalties and that week's preseason prior means, as
    the rating history does, and bootstraps them like the season's rank ranges; the last week
    matches the season's ranges.
    """
    frames: list[pl.DataFrame] = []
    for week in sorted(team_games.get_column("week").unique(maintain_order=True).to_list()):
        through_week = team_games.filter(pl.col("week") <= week)
        means = snapshot_prior(season_prior, through_week)
        week_fit = fit_team_ratings(
            through_week,
            scrimmage_lambda=fit.scrimmage_lambda,
            special_teams_lambda=fit.special_teams_lambda,
            scrimmage_prior=means,
        )
        draws = bootstrap_team_ratings(
            through_week,
            week_fit,
            resamples=bootstrap.resamples,
            seed=bootstrap.seed,
            scrimmage_prior=means,
        )
        frames.append(
            summarize_rank_ranges(draws, week_fit.ratings, TEAM_RANGE_COLUMNS)
            .select(_weekly_rank_columns(TEAM_RANGE_COLUMNS))
            .with_columns(pl.lit(week, dtype=pl.Int64).alias("week"))
        )
    return pl.concat(frames).select("week", pl.exclude("week"))


def build_team_season(
    season: int,
    team_games: pl.DataFrame,
    history: PriorHistory,
    *,
    weekly_ranges: bool = False,
    bootstrap: BootstrapSettings = PUBLISHED_BOOTSTRAP,
) -> TeamSeasonRatings:
    """Return one season's team rating tables.

    Args:
        season: The season rated.
        team_games: The season's team-game rows (``nfl_predictor.ratings.inputs``).
        history: Every season's team-game rows, for the penalties (the previous season's
            cross-validated fit) and the preseason prior (earlier seasons only).
        weekly_ranges: Also build each week's rank ranges, as for a season in progress.
        bootstrap: The game bootstrap behind the rank ranges and pairs (the published one by
            default).

    Returns:
        The tables, each in its published row order.

    """
    previous = history.penalties(season) if season > history.first_season else None
    season_prior = history.season_prior(season)
    means = snapshot_prior(season_prior, team_games)
    fit = fit_team_ratings_with_previous_penalties(team_games, previous, scrimmage_prior=means)
    prior_without = None if means is None else prior_without_team(history, season, team_games)
    ratings = build_team_ratings(team_games, fit, prior_without=prior_without)

    def week_prior(games: pl.DataFrame) -> UnitPrior | None:
        return snapshot_prior(season_prior, games)

    ranges, pairs = build_team_rank_summaries(team_games, fit, means, bootstrap=bootstrap)
    weekly = (
        published_order(
            build_team_rank_ranges_by_week(team_games, fit, season_prior, bootstrap=bootstrap),
            "rating_ranges_by_week",
        )
        if weekly_ranges
        else None
    )
    return TeamSeasonRatings(
        ratings=published_order(ratings, "ratings"),
        team_prior=published_order(
            team_prior_table(means, prior_without, ratings.get_column("team").sort().to_list()),
            "team_prior",
        ),
        ratings_by_week=published_order(
            fit_team_ratings_by_week(team_games, fit, prior_for=week_prior), "ratings_by_week"
        ),
        rating_ranges=published_order(ranges, "rating_ranges"),
        rating_pairs=published_order(pairs, "rating_pairs"),
        rating_ranges_by_week=weekly,
    )


def build_qb_ratings(qb_games: pl.DataFrame, fit: QbRatingFit) -> pl.DataFrame:
    """Return one row per passer with adjusted EPA per dropback and faced pass defense."""
    return fit.ratings.join(compute_qb_faced_pass_defense(qb_games, fit), on=QB_ID_COLUMN)


def _eligible_ids(qb_season: pl.DataFrame) -> list[str]:
    """Return the passers ``qb_season`` flags as qualified for the rankings."""
    return qb_season.filter(pl.col("qb_is_eligible")).get_column(QB_ID_COLUMN).to_list()


def build_qb_rank_summaries(
    qb_games: pl.DataFrame,
    fit: QbRatingFit,
    qb_season: pl.DataFrame,
    *,
    bootstrap: BootstrapSettings = PUBLISHED_BOOTSTRAP,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the qualified passers' rank ranges and their head-to-head chances.

    Both come from one set of game-bootstrap resamples. Ranks and pairs are among the passers
    ``qb_season`` flags ``qb_is_eligible`` for the full season, and each range row carries the
    passer's name and primary team.
    """
    eligible = qb_season.filter(pl.col("qb_is_eligible"))
    eligible_ids: Collection[str] = eligible.get_column(QB_ID_COLUMN).to_list()
    draws = bootstrap_qb_ratings(qb_games, fit, resamples=bootstrap.resamples, seed=bootstrap.seed)
    ranges = summarize_rank_ranges(draws, fit.ratings, QB_RANGE_COLUMNS, eligible=eligible_ids)
    identity = eligible.select(_QB_IDENTITY).unique(QB_ID_COLUMN, keep="first", maintain_order=True)
    ranges = ranges.join(identity, on=QB_ID_COLUMN, how="left", maintain_order="left").select(
        *identity.columns, pl.exclude(identity.columns)
    )
    return ranges, summarize_rank_pairs(draws, fit.ratings, QB_PAIR_COLUMNS, eligible=eligible_ids)


def build_qb_rank_ranges_by_week(
    qb_games: pl.DataFrame,
    fit: QbRatingFit,
    qb_season: pl.DataFrame,
    *,
    bootstrap: BootstrapSettings = PUBLISHED_BOOTSTRAP,
) -> pl.DataFrame:
    """Return the qualified passers' rank ranges as of each week, from the games through it.

    Each week refits its games with ``fit``'s penalty and bootstraps them; ranks are among the
    full-season qualified passers who have played by then, as in the season's rank ranges. Weeks
    before any of them has played are left out.
    """
    eligible = _eligible_ids(qb_season)
    frames: list[pl.DataFrame] = []
    for week in sorted(qb_games.get_column("week").unique(maintain_order=True).to_list()):
        through_week = qb_games.filter(pl.col("week") <= week)
        week_fit = fit_qb_ratings(through_week, ridge_lambda=fit.ridge_lambda)
        draws = bootstrap_qb_ratings(
            through_week, week_fit, resamples=bootstrap.resamples, seed=bootstrap.seed
        )
        summary = summarize_rank_ranges(
            draws, week_fit.ratings, QB_RANGE_COLUMNS, eligible=eligible
        )
        if summary.is_empty():
            continue
        frames.append(
            summary.select(_weekly_rank_columns(QB_RANGE_COLUMNS)).with_columns(
                pl.lit(week, dtype=pl.Int64).alias("week")
            )
        )
    if not frames:
        return pl.DataFrame(schema=_weekly_rank_schema(QB_RANGE_COLUMNS))
    return pl.concat(frames).select("week", pl.exclude("week"))


def build_qb_season(
    qb_games: pl.DataFrame,
    team_games: pl.DataFrame,
    *,
    weekly_ranges: bool = False,
    bootstrap: BootstrapSettings = PUBLISHED_BOOTSTRAP,
) -> QbSeasonRatings:
    """Return one season's quarterback rating tables.

    Args:
        qb_games: The season's quarterback-game rows (``nfl_predictor.ratings.inputs``).
        team_games: The season's team-game rows, which count each team's games for the qualifier.
        weekly_ranges: Also build each week's rank ranges, as for a season in progress.
        bootstrap: The game bootstrap behind the rank ranges and pairs (the published one by
            default).

    Returns:
        The tables, each in its published row order.

    """
    fit = fit_qb_ratings(qb_games)
    qb_season = qb_season_summary(qb_games, team_games).join(
        build_qb_ratings(qb_games, fit), on=QB_ID_COLUMN, how="left"
    )
    ranges, pairs = build_qb_rank_summaries(qb_games, fit, qb_season, bootstrap=bootstrap)
    weekly = (
        published_order(
            build_qb_rank_ranges_by_week(qb_games, fit, qb_season, bootstrap=bootstrap),
            "qb_rating_ranges_by_week",
        )
        if weekly_ranges
        else None
    )
    return QbSeasonRatings(
        qb_season=published_order(
            qb_season.select(*QB_SUMMARY_COLUMNS, QB_FACED_DEFENSE_COLUMN, QB_RATING_COLUMN),
            "qb_season",
        ),
        qb_ratings=published_order(
            qb_season.filter(pl.col("qb_is_eligible")).select(QB_RATINGS_ORDER), "qb_ratings"
        ),
        qb_ratings_by_week=published_order(
            fit_qb_ratings_by_week(qb_games, fit), "qb_ratings_by_week"
        ),
        qb_rating_ranges=published_order(ranges, "qb_rating_ranges"),
        qb_rating_pairs=published_order(pairs, "qb_rating_pairs"),
        qb_rating_ranges_by_week=weekly,
    )


__all__ = [
    "PUBLISHED_ROW_ORDER",
    "QB_RATINGS_ORDER",
    "TEAM_RATINGS_ORDER",
    "QbSeasonRatings",
    "TeamSeasonRatings",
    "build_qb_rank_ranges_by_week",
    "build_qb_rank_summaries",
    "build_qb_ratings",
    "build_qb_season",
    "build_team_rank_ranges_by_week",
    "build_team_rank_summaries",
    "build_team_ratings",
    "build_team_season",
    "published_order",
]
