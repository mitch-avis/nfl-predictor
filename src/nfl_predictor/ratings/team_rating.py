"""Points-based team ratings and head-to-head-excluded schedule strength.

A team's rating answers one question: how many points per game better than an average team was it,
after adjusting for every opponent it actually faced? It is built from expected points added
(EPA), which is already measured in points, so the pieces add up without any weighting:

- ``offense_rating``: scrimmage EPA per play the offense produced above an average offense,
  adjusted for the defenses it faced, times the league-average scrimmage plays per team-game.
- ``defense_rating``: scrimmage EPA per play the defense prevented relative to an average defense,
  adjusted for the offenses it faced, on the same per-game scale.
- ``special_teams_rating``: the same two-sided adjustment over special-teams plays (kicks, punts,
  returns, field goals, extra points), counting both the team's possession units and its coverage
  units, times the league-average special-teams plays per team-game.
- ``team_rating``: the sum of the three, in points per game on a neutral field.

Each piece comes from one simultaneous ridge fit (:mod:`nfl_predictor.ratings.ridge`), so
opponents are judged against their own opponents, and so on through the whole schedule.

``sos`` is the average ``team_rating`` of the opponents a team played, one entry per game. Each
opponent is rated from a refit that leaves out every game involving the team being evaluated, so a
team beating up on an opponent cannot make that opponent look weaker in its own schedule strength.

The ridge penalties come from the previous season: ``fit_team_ratings_with_previous_penalties``
reuses the penalties cross-validation chose for the whole previous season. A few weeks of games are
too few to choose a penalty reliably (cross-validation sometimes picks the largest, rating every
team near zero), and the previous season's choice predicted early-season games better without
costing anything later. The first play-by-play season cross-validates its own.

The rating history (``fit_team_ratings_by_week``) refits the ratings on the games through each
week with the season's penalties, so each week's row shows the rating as the evidence then
supported it.

Early in a season the scrimmage effects shrink toward a preseason prior instead of toward zero
(:mod:`nfl_predictor.ratings.team_prior`): every function here takes the prior means a caller
built for its snapshot, and none builds them itself.

The input rows are the team-game rows of :mod:`nfl_predictor.ratings.inputs`: one row per team and
game with the team on offense, its scrimmage plays and EPA, and its special-teams plays and EPA
from the possession side.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

from nfl_predictor.ratings.ridge import (
    UnitColumns,
    UnitDesign,
    UnitFit,
    UnitPrior,
    build_unit_design,
    fit_unit_ridge,
    solve_unit_design,
)

if TYPE_CHECKING:
    from collections.abc import Callable

SCRIMMAGE_PLAYS_COLUMN = "offensive_snaps"
SCRIMMAGE_EPA_COLUMN = "offensive_epa"
SPECIAL_TEAMS_PLAYS_COLUMN = "st_plays"
SPECIAL_TEAMS_EPA_COLUMN = "st_epa"
TEAM_RATING_COLUMNS = ("offense_rating", "defense_rating", "special_teams_rating", "team_rating")

_KEY_COLUMNS = ("game_id", "team", "opponent_team", "is_home")
_RESPONSE = "epa_per_play"
_WEIGHT = "plays"
# The ridge columns for the rows `scrimmage_rows` returns (special-teams rows use the same names).
TEAM_UNIT_COLUMNS = UnitColumns(response=_RESPONSE, weight=_WEIGHT)


@dataclass(frozen=True, slots=True)
class TeamRatingFit:
    """Team ratings plus the fitted constants a head-to-head-excluded refit must reuse."""

    ratings: pl.DataFrame
    scrimmage_lambda: float
    special_teams_lambda: float
    scrimmage_plays_per_game: float
    special_teams_plays_per_game: float


def _require_columns(game_logs: pl.DataFrame) -> None:
    """Raise ``ValueError`` naming every input column the team rating needs but lacks."""
    required = {
        *_KEY_COLUMNS,
        SCRIMMAGE_PLAYS_COLUMN,
        SCRIMMAGE_EPA_COLUMN,
        SPECIAL_TEAMS_PLAYS_COLUMN,
        SPECIAL_TEAMS_EPA_COLUMN,
    }
    missing = sorted(required - set(game_logs.columns))
    if missing:
        msg = f"team game logs are missing team-rating columns: {', '.join(missing)}"
        raise ValueError(msg)


def _unit_rows(game_logs: pl.DataFrame, plays_column: str, epa_column: str) -> pl.DataFrame:
    """Return one offense-versus-defense row per team-game with EPA per play and its play count."""
    return game_logs.filter(pl.col(plays_column) > 0).select(
        *_KEY_COLUMNS,
        pl.col(plays_column).cast(pl.Float64).alias(_WEIGHT),
        (pl.col(epa_column).cast(pl.Float64) / pl.col(plays_column)).alias(_RESPONSE),
    )


def scrimmage_rows(game_logs: pl.DataFrame) -> pl.DataFrame:
    """Return the rows the scrimmage fit uses, for analyses that refit or predict them.

    One row per team-game with plays: ``game_id``, ``team`` (the offense), ``opponent_team`` (the
    defense), ``is_home``, ``plays``, and ``epa_per_play``; fit them with ``TEAM_UNIT_COLUMNS``.

    Raises:
        ValueError: If a team-rating column is missing.

    """
    _require_columns(game_logs)
    return _unit_rows(game_logs, SCRIMMAGE_PLAYS_COLUMN, SCRIMMAGE_EPA_COLUMN)


def special_teams_rows(game_logs: pl.DataFrame) -> pl.DataFrame:
    """Return the rows the special-teams fit uses, shaped like :func:`scrimmage_rows`.

    ``team`` is the possession team and ``opponent_team`` its coverage units.

    Raises:
        ValueError: If a team-rating column is missing.

    """
    _require_columns(game_logs)
    return _unit_rows(game_logs, SPECIAL_TEAMS_PLAYS_COLUMN, SPECIAL_TEAMS_EPA_COLUMN)


def _plays_per_game(rows: pl.DataFrame) -> float:
    """Return the league-average play count per team-game row."""
    return float(rows.get_column(_WEIGHT).sum()) / rows.height


def _ratings_frame(
    scrimmage: UnitFit,
    special_teams: UnitFit,
    scrimmage_plays_per_game: float,
    special_teams_plays_per_game: float,
) -> pl.DataFrame:
    """Convert per-play effects into the four per-game rating columns, one row per team."""
    teams = sorted(
        set(scrimmage.offense)
        | set(scrimmage.defense)
        | set(special_teams.offense)
        | set(special_teams.defense)
    )
    offense = [scrimmage.offense.get(team, 0.0) * scrimmage_plays_per_game for team in teams]
    defense = [scrimmage.defense.get(team, 0.0) * scrimmage_plays_per_game for team in teams]
    special = [
        (special_teams.offense.get(team, 0.0) + special_teams.defense.get(team, 0.0))
        * special_teams_plays_per_game
        for team in teams
    ]
    return pl.DataFrame(
        {
            "team": teams,
            "offense_rating": offense,
            "defense_rating": defense,
            "special_teams_rating": special,
            "team_rating": [o + d + s for o, d, s in zip(offense, defense, special, strict=True)],
        }
    )


def team_ratings_from_unit_fits(
    scrimmage: UnitFit, special_teams: UnitFit, fit: TeamRatingFit
) -> pl.DataFrame:
    """Return the four rating columns from refit unit effects on ``fit``'s per-game scales.

    For refits of a season's games (head-to-head exclusions, filtered plays) that keep the season
    fit's scales, so their ratings read on the published scale.
    """
    return _ratings_frame(
        scrimmage, special_teams, fit.scrimmage_plays_per_game, fit.special_teams_plays_per_game
    )


def fit_team_ratings(
    game_logs: pl.DataFrame,
    *,
    scrimmage_lambda: float | None = None,
    special_teams_lambda: float | None = None,
    scrimmage_prior: UnitPrior | None = None,
) -> TeamRatingFit:
    """Fit points-based offense, defense, special-teams, and overall team ratings.

    Args:
        game_logs: One row per team-game (the team on offense), with ``game_id``, ``team``,
            ``opponent_team``, ``is_home``, scrimmage plays and EPA, and special-teams plays and EPA
            from the possession team's side.
        scrimmage_lambda: Fixed scrimmage ridge penalty; cross-validated when ``None``.
        special_teams_lambda: Fixed special-teams ridge penalty; cross-validated when ``None``.
        scrimmage_prior: Per-play prior means the scrimmage penalty pulls the offense and defense
            effects toward (``ridge.UnitPrior``; zero when ``None``). It needs a fixed
            ``scrimmage_lambda``. The special-teams fit keeps a prior of zero.

    Returns:
        The ratings frame plus the penalties and per-game scales used to build it.

    Raises:
        ValueError: If a required column is missing, no rows have plays, or a scrimmage prior
            comes without a fixed scrimmage penalty.

    """
    _require_columns(game_logs)
    scrimmage_rows = _unit_rows(game_logs, SCRIMMAGE_PLAYS_COLUMN, SCRIMMAGE_EPA_COLUMN)
    special_rows = _unit_rows(game_logs, SPECIAL_TEAMS_PLAYS_COLUMN, SPECIAL_TEAMS_EPA_COLUMN)
    scrimmage = fit_unit_ridge(
        scrimmage_rows, TEAM_UNIT_COLUMNS, ridge_lambda=scrimmage_lambda, prior=scrimmage_prior
    )
    special_teams = fit_unit_ridge(
        special_rows, TEAM_UNIT_COLUMNS, ridge_lambda=special_teams_lambda
    )
    scrimmage_plays_per_game = _plays_per_game(scrimmage_rows)
    special_teams_plays_per_game = _plays_per_game(special_rows)
    return TeamRatingFit(
        ratings=_ratings_frame(
            scrimmage, special_teams, scrimmage_plays_per_game, special_teams_plays_per_game
        ),
        scrimmage_lambda=scrimmage.ridge_lambda,
        special_teams_lambda=special_teams.ridge_lambda,
        scrimmage_plays_per_game=scrimmage_plays_per_game,
        special_teams_plays_per_game=special_teams_plays_per_game,
    )


class TeamRatingResampler:
    """Refit the team ratings on resampled games, reusing one fit's fixed penalties.

    Built once per season; each call to :meth:`ratings` takes a count per game (how many times a
    bootstrap resample drew it) and returns what :func:`fit_team_ratings` would return on the
    resampled games with duplicates relabeled, without rebuilding anything.
    """

    def __init__(
        self,
        game_logs: pl.DataFrame,
        fit: TeamRatingFit,
        *,
        scrimmage_prior: UnitPrior | None = None,
    ) -> None:
        """Build the scrimmage and special-teams designs for ``game_logs``.

        ``scrimmage_prior`` holds the snapshot's prior means, which every resample keeps: a
        resample does not recount games, so a team's prior does not change with its draws.
        """
        _require_columns(game_logs)
        self._fit = fit
        self._priors: tuple[UnitPrior | None, UnitPrior | None] = (scrimmage_prior, None)
        self.game_ids: list[str] = sorted(game_logs.get_column("game_id").unique().to_list())
        index = {game_id: position for position, game_id in enumerate(self.game_ids)}
        self._units: list[tuple[UnitDesign, npt.NDArray[np.int64], npt.NDArray[np.float64]]] = []
        for plays_column, epa_column in (
            (SCRIMMAGE_PLAYS_COLUMN, SCRIMMAGE_EPA_COLUMN),
            (SPECIAL_TEAMS_PLAYS_COLUMN, SPECIAL_TEAMS_EPA_COLUMN),
        ):
            rows = _unit_rows(game_logs, plays_column, epa_column).drop_nulls(
                ["team", "opponent_team", _RESPONSE]
            )
            self._units.append(
                (
                    build_unit_design(rows, TEAM_UNIT_COLUMNS),
                    np.array(
                        [index[game_id] for game_id in rows.get_column("game_id")], dtype=np.int64
                    ),
                    np.asarray(rows.get_column(_WEIGHT).to_numpy(), dtype=np.float64),
                )
            )

    def ratings(self, game_counts: npt.ArrayLike) -> pl.DataFrame:
        """Return the four rating columns for the games weighted by ``game_counts``.

        Args:
            game_counts: How many times each game in ``game_ids`` is drawn, in that order.

        """
        counts = np.asarray(game_counts, dtype=np.float64)
        fits: list[UnitFit] = []
        plays_per_game: list[float] = []
        for (design, games, plays), ridge_lambda, prior in zip(
            self._units,
            (self._fit.scrimmage_lambda, self._fit.special_teams_lambda),
            self._priors,
            strict=True,
        ):
            multipliers = counts[games]
            fits.append(solve_unit_design(design, ridge_lambda, multipliers, prior))
            plays_per_game.append(float((multipliers * plays).sum() / multipliers.sum()))
        return _ratings_frame(fits[0], fits[1], plays_per_game[0], plays_per_game[1])


def bootstrap_team_ratings(
    game_logs: pl.DataFrame,
    fit: TeamRatingFit,
    *,
    resamples: int,
    seed: int,
    scrimmage_prior: UnitPrior | None = None,
) -> pl.DataFrame:
    """Return every team's ratings in ``resamples`` game-bootstrap resamples.

    Each resample draws the season's games with replacement and refits with ``fit``'s penalties
    and the snapshot's ``scrimmage_prior``.

    Returns:
        Long rows with ``draw``, ``team``, and the four ``TEAM_RATING_COLUMNS`` (the three unit
        ratings and ``team_rating``, their sum).

    """
    resampler = TeamRatingResampler(game_logs, fit, scrimmage_prior=scrimmage_prior)
    rng = np.random.default_rng(seed)
    game_count = len(resampler.game_ids)
    frames = [
        resampler.ratings(
            np.bincount(rng.integers(0, game_count, game_count), minlength=game_count)
        ).select(pl.lit(draw).alias("draw"), "team", *TEAM_RATING_COLUMNS)
        for draw in range(resamples)
    ]
    return pl.concat(frames)


def fit_team_ratings_with_previous_penalties(
    game_logs: pl.DataFrame,
    previous: TeamRatingFit | None,
    *,
    scrimmage_prior: UnitPrior | None = None,
) -> TeamRatingFit:
    """Fit the published team ratings: this season's games with the previous season's penalties.

    Args:
        game_logs: This season's team-game rows, as :func:`fit_team_ratings` takes them.
        previous: The previous season's full-season fit (penalties cross-validated), or ``None``
            for a season without a previous one, which cross-validates its own penalties.
        scrimmage_prior: The snapshot's preseason prior means (``team_prior.snapshot_prior``), or
            ``None`` for none. A prior needs ``previous``: cross-validation never sees a prior.

    Returns:
        The season fit.

    """
    if previous is None:
        return fit_team_ratings(game_logs, scrimmage_prior=scrimmage_prior)
    return fit_team_ratings(
        game_logs,
        scrimmage_lambda=previous.scrimmage_lambda,
        special_teams_lambda=previous.special_teams_lambda,
        scrimmage_prior=scrimmage_prior,
    )


def fit_team_ratings_by_week(
    game_logs: pl.DataFrame,
    fit: TeamRatingFit,
    *,
    prior_for: Callable[[pl.DataFrame], UnitPrior | None] | None = None,
) -> pl.DataFrame:
    """Return each team's ratings as of every week, each fit on the games through that week.

    Every week reuses the season fit's penalties rather than cross-validating its own: after one
    or two weeks there are too few games for cross-validation to choose a penalty reliably. With
    the penalty held fixed, the pull toward average depends only on how many plays a team has, so
    early weeks sit close to average and spread out as games accumulate. The per-game scales are
    each week's own league averages, as a fit of those games alone would use. Head-to-head-excluded
    schedule strength is not refit week by week.

    Args:
        game_logs: The team-game rows passed to :func:`fit_team_ratings`, plus ``week``.
        fit: The season fit of ``game_logs`` whose penalties every week reuses.
        prior_for: Returns the preseason prior means for the games through a week (that week's
            games played set the fade), or ``None`` for none; no prior when omitted.

    Returns:
        One row per week and team that has played by then, with ``week``, ``team``,
        ``games_played`` through that week, and the four rating columns. The last week's ratings
        are the season fit's.

    Raises:
        ValueError: If ``week`` or a team-rating column is missing.

    """
    _require_columns(game_logs)
    if "week" not in game_logs.columns:
        msg = "team game logs need a week column for the weekly rating history"
        raise ValueError(msg)
    weeks: list[int] = sorted(game_logs.get_column("week").unique().to_list())
    frames: list[pl.DataFrame] = []
    for week in weeks:
        through_week = game_logs.filter(pl.col("week") <= week)
        games_played = through_week.group_by("team").len("games_played")
        ratings = fit_team_ratings(
            through_week,
            scrimmage_lambda=fit.scrimmage_lambda,
            special_teams_lambda=fit.special_teams_lambda,
            scrimmage_prior=None if prior_for is None else prior_for(through_week),
        ).ratings
        frames.append(
            ratings.join(games_played, on="team").select(
                pl.lit(week, dtype=pl.Int64).alias("week"),
                "team",
                pl.col("games_played").cast(pl.Int64),
                *TEAM_RATING_COLUMNS,
            )
        )
    return pl.concat(frames)


def _ratings_without(
    game_logs: pl.DataFrame, team: str, fit: TeamRatingFit, prior: UnitPrior | None
) -> pl.DataFrame:
    """Rate every other team from a refit that drops all games involving ``team``.

    ``prior`` holds the refit's prior means (none for ``None``). Returns an empty frame when no
    games remain, as can happen in the first weeks of a season.
    """
    others = game_logs.filter((pl.col("team") != team) & (pl.col("opponent_team") != team))
    if others.is_empty():
        return pl.DataFrame(schema={"team": pl.String, "team_rating": pl.Float64})
    scrimmage = fit_unit_ridge(
        _unit_rows(others, SCRIMMAGE_PLAYS_COLUMN, SCRIMMAGE_EPA_COLUMN),
        TEAM_UNIT_COLUMNS,
        ridge_lambda=fit.scrimmage_lambda,
        prior=prior,
    )
    special_teams = fit_unit_ridge(
        _unit_rows(others, SPECIAL_TEAMS_PLAYS_COLUMN, SPECIAL_TEAMS_EPA_COLUMN),
        TEAM_UNIT_COLUMNS,
        ridge_lambda=fit.special_teams_lambda,
    )
    return _ratings_frame(
        scrimmage,
        special_teams,
        fit.scrimmage_plays_per_game,
        fit.special_teams_plays_per_game,
    )


def compute_team_schedule_strength(
    game_logs: pl.DataFrame,
    fit: TeamRatingFit,
    *,
    prior_without: Callable[[str], UnitPrior | None] | None = None,
) -> pl.DataFrame:
    """Return each team's played-game mean opponent ``team_rating``, head-to-head excluded.

    Args:
        game_logs: The same team-game rows passed to :func:`fit_team_ratings`.
        fit: The full-season fit whose penalties and per-game scales the refits reuse.
        prior_without: Returns the preseason prior means for the refit that leaves out a team,
            built without that team's previous-season games too, or ``None`` for none; no prior
            when omitted.

    Returns:
        One row per team with ``team`` and ``sos`` in points per game. Early in a season an
        opponent may have played no one else yet; such games are skipped, and ``sos`` is null
        when no opponent can be rated.

    """
    _require_columns(game_logs)
    teams: list[str] = fit.ratings.get_column("team").to_list()
    values: list[float | None] = []
    for team in teams:
        prior = None if prior_without is None else prior_without(team)
        opponent_ratings = _ratings_without(game_logs, team, fit, prior).select(
            pl.col("team").alias("opponent_team"), pl.col("team_rating").alias("sos")
        )
        rated = (
            game_logs.filter(pl.col("team") == team)
            .select("opponent_team")
            .join(opponent_ratings, on="opponent_team", how="inner")
        )
        values.append(float(np.mean(rated.get_column("sos").to_numpy())) if rated.height else None)
    return pl.DataFrame({"team": teams, "sos": values})


__all__ = [
    "SCRIMMAGE_EPA_COLUMN",
    "SCRIMMAGE_PLAYS_COLUMN",
    "SPECIAL_TEAMS_EPA_COLUMN",
    "SPECIAL_TEAMS_PLAYS_COLUMN",
    "TEAM_RATING_COLUMNS",
    "TEAM_UNIT_COLUMNS",
    "TeamRatingFit",
    "TeamRatingResampler",
    "bootstrap_team_ratings",
    "compute_team_schedule_strength",
    "fit_team_ratings",
    "fit_team_ratings_by_week",
    "fit_team_ratings_with_previous_penalties",
    "scrimmage_rows",
    "special_teams_rows",
    "team_ratings_from_unit_fits",
]
