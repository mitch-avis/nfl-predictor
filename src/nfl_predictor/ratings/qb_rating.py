"""Quarterback rating: EPA per dropback adjusted for the pass defenses actually faced.

``adj_qb_epa_per_dropback`` is the passer's EPA per dropback after the simultaneous solve in
:mod:`nfl_predictor.ratings.ridge`: every passer is judged against the defenses he faced, and every
defense against all the passers it faced. It reads on the same scale as raw EPA per dropback, with
the league-average passer at the fitted intercept. Rows are weighted by dropbacks, and the ridge
penalty pulls small samples toward average, so a few backup snaps cannot produce an extreme value.

``qb_faced_pass_defense`` is the dropback-weighted average quality of the pass defenses the
passer faced, in EPA per dropback (positive means tougher defenses). Each defense is rated from a
refit that leaves out the passer's own dropbacks, so a passer who shreds a defense cannot make
that defense look weaker in his own schedule.

The rating history (``fit_qb_ratings_by_week``) refits the rating on the games through each week
with the season's penalty, so each week's row shows the rating as the evidence then supported it.

The input rows are the quarterback-game rows of :mod:`nfl_predictor.ratings.inputs`. Each carries
the passer's dropbacks (pass attempts and sacks; spikes and kneel-downs are not dropbacks) and the
EPA on them (``qb_dropback_epa``, spikes left out); the fit's per-game response is their ratio,
``qb_epa_per_dropback``.

:func:`qb_season_summary` adds what a season's quarterback table shows beside the rating: games,
dropbacks and attempts, the raw season EPA per dropback, the passer's primary team, and whether he
qualifies for the rankings (14 pass attempts per game his primary team has played).
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from nfl_predictor.ratings.ridge import (
    UnitColumns,
    build_unit_design,
    fit_unit_ridge,
    solve_unit_design,
)

QB_ID_COLUMN = "qb_id"
QB_DROPBACKS_COLUMN = "qb_dropbacks"
QB_DROPBACK_EPA_COLUMN = "qb_dropback_epa"
QB_EPA_PER_DROPBACK_COLUMN = "qb_epa_per_dropback"
QB_RATING_COLUMN = "adj_qb_epa_per_dropback"
QB_FACED_DEFENSE_COLUMN = "qb_faced_pass_defense"
# A quarterback is ranked with 14 pass attempts per game his primary team has played; without any
# team-game rows a full season of 17 games is assumed.
QUALIFIER_ATTEMPTS_PER_TEAM_GAME = 14
FULL_SEASON_TEAM_GAMES = 17
QB_SUMMARY_COLUMNS = (
    QB_ID_COLUMN,
    "qb_name",
    "team",
    "qb_games_played",
    "qb_dropbacks_total",
    "qb_attempts_total",
    QB_EPA_PER_DROPBACK_COLUMN,
    "qb_attempt_qualifier",
    "qb_is_eligible",
)

_REQUIRED_COLUMNS = (
    "game_id",
    QB_ID_COLUMN,
    "opponent_team",
    QB_DROPBACKS_COLUMN,
    QB_DROPBACK_EPA_COLUMN,
)
QB_UNIT_COLUMNS = UnitColumns(
    response=QB_EPA_PER_DROPBACK_COLUMN,
    weight=QB_DROPBACKS_COLUMN,
    offense=QB_ID_COLUMN,
)


@dataclass(frozen=True, slots=True)
class QbRatingFit:
    """Adjusted QB ratings plus the penalty a head-to-head-excluded refit must reuse."""

    ratings: pl.DataFrame
    ridge_lambda: float


def _require_columns(qb_games: pl.DataFrame) -> None:
    """Raise ``ValueError`` naming every column the QB rating needs but lacks."""
    missing = sorted(set(_REQUIRED_COLUMNS) - set(qb_games.columns))
    if missing:
        msg = f"QB game rows are missing QB-rating columns: {', '.join(missing)}"
        raise ValueError(msg)


def _rated_rows(qb_games: pl.DataFrame) -> pl.DataFrame:
    """Return the QB-game rows with at least one dropback, with their EPA per dropback."""
    return qb_games.filter(pl.col(QB_DROPBACKS_COLUMN) > 0).with_columns(
        (pl.col(QB_DROPBACK_EPA_COLUMN) / pl.col(QB_DROPBACKS_COLUMN)).alias(
            QB_EPA_PER_DROPBACK_COLUMN
        )
    )


def qb_rating_rows(qb_games: pl.DataFrame) -> pl.DataFrame:
    """Return the passer-game rows the QB fit uses: those with a dropback, with their rate.

    ``qb_epa_per_dropback`` is ``qb_dropback_epa / qb_dropbacks``. Fit the rows with
    ``QB_UNIT_COLUMNS``.

    Raises:
        ValueError: If a QB-rating column is missing.

    """
    _require_columns(qb_games)
    return _rated_rows(qb_games)


def _ratings_frame(intercept: float, offense: dict[str, float]) -> pl.DataFrame:
    """Return ``qb_id`` and the adjusted rating (intercept plus passer effect), sorted by id."""
    passers = sorted(offense)
    return pl.DataFrame(
        {
            QB_ID_COLUMN: passers,
            QB_RATING_COLUMN: [intercept + offense[qb] for qb in passers],
        },
        schema={QB_ID_COLUMN: pl.String, QB_RATING_COLUMN: pl.Float64},
    )


def fit_qb_ratings(qb_games: pl.DataFrame, *, ridge_lambda: float | None = None) -> QbRatingFit:
    """Fit adjusted EPA per dropback for every passer with a dropback.

    Args:
        qb_games: One row per passer-game with ``game_id``, ``qb_id``, ``opponent_team``,
            ``qb_dropbacks``, ``qb_dropback_epa``, and optionally ``is_home``.
        ridge_lambda: Fixed ridge penalty; cross-validated when ``None``.

    Returns:
        One row per passer with ``qb_id`` and ``adj_qb_epa_per_dropback``, plus the penalty.

    Raises:
        ValueError: If a required column is missing or no row has a dropback.

    """
    _require_columns(qb_games)
    fit = fit_unit_ridge(_rated_rows(qb_games), QB_UNIT_COLUMNS, ridge_lambda=ridge_lambda)
    return QbRatingFit(
        ratings=_ratings_frame(fit.intercept, fit.offense), ridge_lambda=fit.ridge_lambda
    )


def fit_qb_ratings_by_week(qb_games: pl.DataFrame, fit: QbRatingFit) -> pl.DataFrame:
    """Return each passer's adjusted EPA per dropback as of every week.

    Every week refits the passer-games through that week with the season fit's penalty, for the
    reason :func:`nfl_predictor.ratings.team_rating.fit_team_ratings_by_week` gives: early weeks
    are too small for cross-validation to choose one reliably. Faced pass defense is not refit week
    by week.

    Args:
        qb_games: The passer-game rows passed to :func:`fit_qb_ratings`, plus ``week``.
        fit: The season fit of ``qb_games`` whose penalty every week reuses.

    Returns:
        One row per week and passer with a dropback by then: ``week``, ``qb_id``,
        ``qb_games_played`` and ``qb_dropbacks`` through that week, and
        ``adj_qb_epa_per_dropback``.

    Raises:
        ValueError: If ``week`` or a QB-rating column is missing.

    """
    _require_columns(qb_games)
    if "week" not in qb_games.columns:
        msg = "QB game rows need a week column for the weekly rating history"
        raise ValueError(msg)
    rows = _rated_rows(qb_games)
    weeks: list[int] = sorted(rows.get_column("week").unique().to_list())
    frames: list[pl.DataFrame] = []
    for week in weeks:
        through_week = rows.filter(pl.col("week") <= week)
        volume = through_week.group_by(QB_ID_COLUMN).agg(
            pl.len().cast(pl.Int64).alias("qb_games_played"),
            pl.col(QB_DROPBACKS_COLUMN).sum().cast(pl.Int64).alias("qb_dropbacks"),
        )
        ratings = fit_qb_ratings(through_week, ridge_lambda=fit.ridge_lambda).ratings
        frames.append(
            ratings.join(volume, on=QB_ID_COLUMN).select(
                pl.lit(week, dtype=pl.Int64).alias("week"),
                QB_ID_COLUMN,
                "qb_games_played",
                "qb_dropbacks",
                QB_RATING_COLUMN,
            )
        )
    return pl.concat(frames)


class QbRatingResampler:
    """Refit the QB rating on resampled games, reusing one fit's fixed penalty.

    The QB counterpart of :class:`nfl_predictor.ratings.team_rating.TeamRatingResampler`:
    :meth:`ratings` takes a count per game and returns what :func:`fit_qb_ratings` would return on
    the resampled games.
    """

    def __init__(self, qb_games: pl.DataFrame, fit: QbRatingFit) -> None:
        """Build the passer-versus-defense design for ``qb_games``."""
        _require_columns(qb_games)
        self._ridge_lambda = fit.ridge_lambda
        rows = _rated_rows(qb_games).drop_nulls(
            [QB_ID_COLUMN, "opponent_team", QB_EPA_PER_DROPBACK_COLUMN]
        )
        self.game_ids: list[str] = sorted(rows.get_column("game_id").unique().to_list())
        index = {game_id: position for position, game_id in enumerate(self.game_ids)}
        self._design = build_unit_design(rows, QB_UNIT_COLUMNS)
        self._games = np.array(
            [index[game_id] for game_id in rows.get_column("game_id")], dtype=np.int64
        )

    def ratings(self, game_counts: npt.ArrayLike) -> pl.DataFrame:
        """Return ``qb_id`` and ``adj_qb_epa_per_dropback`` for passers in the drawn games."""
        counts = np.asarray(game_counts, dtype=np.float64)
        fit = solve_unit_design(self._design, self._ridge_lambda, counts[self._games])
        return _ratings_frame(fit.intercept, fit.offense)


def bootstrap_qb_ratings(
    qb_games: pl.DataFrame, fit: QbRatingFit, *, resamples: int, seed: int
) -> pl.DataFrame:
    """Return ``adj_qb_epa_per_dropback`` for every passer in each game-bootstrap resample.

    Each resample draws the games with a dropback with replacement (``numpy``'s
    ``default_rng(seed)``, one ``integers`` call per resample) and refits with ``fit``'s penalty.

    Returns:
        Long rows with ``draw``, ``qb_id``, and ``adj_qb_epa_per_dropback``; a passer with no
        dropbacks in a resample has no row for that draw.

    """
    resampler = QbRatingResampler(qb_games, fit)
    rng = np.random.default_rng(seed)
    game_count = len(resampler.game_ids)
    frames = [
        resampler.ratings(
            np.bincount(rng.integers(0, game_count, game_count), minlength=game_count)
        ).select(pl.lit(draw).alias("draw"), QB_ID_COLUMN, QB_RATING_COLUMN)
        for draw in range(resamples)
    ]
    return pl.concat(frames)


def compute_qb_faced_pass_defense(qb_games: pl.DataFrame, fit: QbRatingFit) -> pl.DataFrame:
    """Return each passer's dropback-weighted faced pass defense, head-to-head excluded.

    Args:
        qb_games: The same passer-game rows passed to :func:`fit_qb_ratings`.
        fit: The full fit whose penalty the refits reuse.

    Returns:
        One row per passer with ``qb_id`` and ``qb_faced_pass_defense`` in EPA per dropback.
        Defenses that have faced no other passer yet are skipped, and the value is null when
        none of the passer's defenses can be rated.

    """
    _require_columns(qb_games)
    rows = _rated_rows(qb_games)
    passers: list[str] = fit.ratings.get_column(QB_ID_COLUMN).to_list()
    values: list[float | None] = []
    for passer in passers:
        others = rows.filter(pl.col(QB_ID_COLUMN) != passer)
        defense = (
            fit_unit_ridge(others, QB_UNIT_COLUMNS, ridge_lambda=fit.ridge_lambda).defense
            if not others.is_empty()
            else {}
        )
        own = rows.filter(
            (pl.col(QB_ID_COLUMN) == passer) & pl.col("opponent_team").is_in(list(defense))
        )
        if own.is_empty():
            values.append(None)
            continue
        weights = own.get_column(QB_DROPBACKS_COLUMN).cast(pl.Float64).to_numpy()
        effects = np.array(
            [defense[team] for team in own.get_column("opponent_team").to_list()], dtype=np.float64
        )
        values.append(float(np.average(effects, weights=weights)))
    return pl.DataFrame(
        {QB_ID_COLUMN: passers, QB_FACED_DEFENSE_COLUMN: values},
        schema={QB_ID_COLUMN: pl.String, QB_FACED_DEFENSE_COLUMN: pl.Float64},
    )


def _known_total(column: str) -> pl.Expr:
    """Return the column's sum, or null when every value is null: a total of unknowns is unknown."""
    return pl.when(pl.col(column).is_not_null().any()).then(pl.col(column).sum())


def _primary_teams(qb_games: pl.DataFrame) -> pl.DataFrame:
    """Return each passer's primary team: the team he played the most games for.

    A tie (a mid-season trade with equal games on both teams) goes to the team he played for most
    recently, then to the alphabetically first team, so the pick never depends on row order.
    """
    return (
        qb_games.group_by([QB_ID_COLUMN, "team"])
        .agg(pl.len().alias("_games"), pl.col("week").max().alias("_last_week"))
        .sort(
            ["_games", "_last_week", "team"],
            descending=[True, True, False],
            nulls_last=True,
            maintain_order=True,
        )
        .group_by(QB_ID_COLUMN, maintain_order=True)
        .first()
        .select(QB_ID_COLUMN, "team")
    )


def qb_season_summary(qb_games: pl.DataFrame, team_games: pl.DataFrame) -> pl.DataFrame:
    """Return each passer's season volume, raw rate, primary team, and ranking qualifier.

    Args:
        qb_games: Every passer-game row of the season so far, those without a dropback included,
            with ``week``, ``team``, ``qb_name`` and ``qb_attempts`` besides the rating columns.
        team_games: The season's team-game rows so far; only ``team`` is read, to count each
            team's games.

    Returns:
        One row per passer, sorted by primary team and then ``qb_id``, with the columns of
        ``QB_SUMMARY_COLUMNS``: ``qb_games_played`` counts every game row, totals are null when
        every game's value is unknown, ``qb_epa_per_dropback`` is the dropback EPA total over the
        dropback total, ``qb_name`` is the name on the passer's latest game, and
        ``qb_attempt_qualifier`` is 14 attempts per game of his primary team (a team without
        team-game rows takes the most games any team has, or 17 when there are none).
        ``qb_is_eligible`` is ``qb_attempts_total >= qb_attempt_qualifier``.

    Raises:
        ValueError: If a QB-rating column is missing.

    """
    _require_columns(qb_games)
    totals = (
        qb_games.sort("week", "game_id", maintain_order=True)
        .group_by(QB_ID_COLUMN, maintain_order=True)
        .agg(
            pl.col("qb_name").last(),
            pl.len().alias("qb_games_played"),
            _known_total(QB_DROPBACKS_COLUMN).cast(pl.Int64).alias("qb_dropbacks_total"),
            _known_total("qb_attempts").cast(pl.Int64).alias("qb_attempts_total"),
            _known_total(QB_DROPBACK_EPA_COLUMN).alias("_dropback_epa_total"),
        )
    )
    team_counts = team_games.group_by("team").len("_team_games")
    most_games = (
        team_counts.get_column("_team_games").max()
        if team_counts.height
        else FULL_SEASON_TEAM_GAMES
    )
    return (
        totals.join(_primary_teams(qb_games), on=QB_ID_COLUMN, how="left")
        .join(team_counts, on="team", how="left")
        .with_columns(
            pl.when(pl.col("qb_dropbacks_total") > 0)
            .then(pl.col("_dropback_epa_total") / pl.col("qb_dropbacks_total"))
            .alias(QB_EPA_PER_DROPBACK_COLUMN),
            (pl.col("_team_games").fill_null(most_games) * QUALIFIER_ATTEMPTS_PER_TEAM_GAME)
            .cast(pl.Int64)
            .alias("qb_attempt_qualifier"),
        )
        .with_columns(
            (pl.col("qb_attempts_total") >= pl.col("qb_attempt_qualifier")).alias("qb_is_eligible")
        )
        .select(QB_SUMMARY_COLUMNS)
        .sort("team", QB_ID_COLUMN, nulls_last=True)
    )


__all__ = [
    "FULL_SEASON_TEAM_GAMES",
    "QB_DROPBACKS_COLUMN",
    "QB_DROPBACK_EPA_COLUMN",
    "QB_EPA_PER_DROPBACK_COLUMN",
    "QB_FACED_DEFENSE_COLUMN",
    "QB_ID_COLUMN",
    "QB_RATING_COLUMN",
    "QB_SUMMARY_COLUMNS",
    "QB_UNIT_COLUMNS",
    "QUALIFIER_ATTEMPTS_PER_TEAM_GAME",
    "QbRatingFit",
    "QbRatingResampler",
    "bootstrap_qb_ratings",
    "compute_qb_faced_pass_defense",
    "fit_qb_ratings",
    "fit_qb_ratings_by_week",
    "qb_rating_rows",
    "qb_season_summary",
]
