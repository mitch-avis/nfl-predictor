"""Preseason prior for the team fit: last season's scrimmage effects, carried over and faded out.

Early in a season the team fit's ridge penalty pulls every scrimmage effect toward zero, an average
team, because a few games say little. With a prior, it pulls each team's offense and defense
toward a mean built from the previous season instead (``ridge.UnitPrior``):

    raw_o[t] = fade(g_t, horizon) * rho_o * o_prev[t]        (defense likewise)
    fade(g)  = max(0, 1 - g / horizon)
    m_o[t]   = raw_o[t] - the mean of raw_o over the teams in the fit

``o_prev`` is the team's per-play scrimmage offense effect in the previous season's published fit,
``rho_o`` the carryover slope (season t's near-unpenalized effects regressed on season t-1's
published ones, through the origin, pooled over the season pairs before the season rated), and
``g_t`` the games the team has played. Once a team has played ``horizon`` games its prior is gone,
and centering keeps the effects averaging zero when byes leave teams with different game counts.
Special teams keep a prior of zero.

The published team fit uses the prior at ``PRIOR_HORIZON_GAMES``, the horizon a pre-registered
test in ``nfl-sos-ratings`` recommended: every fit of a season's games so far takes the means of
:func:`snapshot_prior`, and a head-to-head-excluded refit takes the means of the previous season
refit without the evaluated team (:meth:`PriorHistory.season_prior_without`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nfl_predictor import constants
from nfl_predictor.ratings.ridge import UnitFit, UnitPrior, fit_unit_ridge
from nfl_predictor.ratings.team_rating import (
    TEAM_UNIT_COLUMNS,
    TeamRatingFit,
    fit_team_ratings,
    fit_team_ratings_with_previous_penalties,
    scrimmage_rows,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

# The penalty of the fit that the carryover slope regresses: small enough not to shrink the
# effects, so they estimate a season's effects without bias, and large enough to pin the effects
# to average zero (the intercept is unpenalized).
NEAR_UNPENALIZED_LAMBDA = 1e-6
# Season pairs the carryover slope needs before a season gets a prior (2003 is the first).
MIN_CARRYOVER_PAIRS = 3
# Games after which a team's prior is gone: the published horizon, the one the pre-registered test
# recommended.
PRIOR_HORIZON_GAMES = 9.0


def fade(games: int, horizon: float) -> float:
    """Return the prior's weight after ``games`` games: 1 before any, 0 from ``horizon`` on."""
    return max(0.0, 1.0 - games / horizon)


def carryover_slope(previous: Sequence[float], current: Sequence[float]) -> float:
    """Return the least-squares slope through the origin of ``current`` on ``previous``.

    Raises:
        ValueError: If the two differ in length or ``previous`` has no spread (all zeros).

    """
    before = np.asarray(previous, dtype=np.float64)
    after = np.asarray(current, dtype=np.float64)
    if before.shape != after.shape:
        msg = f"carryover_slope needs paired values, got {before.size} and {after.size}"
        raise ValueError(msg)
    denominator = float(before @ before)
    if denominator == 0.0:
        msg = "the previous effects have no spread, so no carryover slope exists"
        raise ValueError(msg)
    return float(before @ after) / denominator


@dataclass(frozen=True, slots=True)
class CarryoverSlopes:
    """How much of a team's previous-season scrimmage effect carries into the next season."""

    offense: float
    defense: float
    pairs: int


@dataclass(frozen=True, slots=True)
class SeasonPrior:
    """What one season's prior is built from: the previous season's effects and the slopes."""

    season: int
    offense: Mapping[str, float]
    defense: Mapping[str, float]
    slopes: CarryoverSlopes


def games_played(game_logs: pl.DataFrame, teams: Iterable[str]) -> dict[str, int]:
    """Return how many games each of ``teams`` has in ``game_logs``, zero for a team without one."""
    counts: dict[str, int] = dict(
        game_logs.group_by("team").agg(pl.col("game_id").n_unique()).iter_rows()
    )
    return {team: int(counts.get(team, 0)) for team in teams}


def _centered(values: Mapping[str, float], fitted: Collection[str]) -> dict[str, float]:
    """Shift every value by the mean over ``fitted``, so the fitted teams' values average zero."""
    shift = float(np.mean([values[team] for team in fitted])) if fitted else 0.0
    return {team: value - shift for team, value in values.items()}


def prior_means(
    season_prior: SeasonPrior,
    games: Mapping[str, int],
    horizon: float,
    fitted: Collection[str],
) -> UnitPrior:
    """Return the centered prior means of every team in ``games``.

    A raw mean is ``fade(games, horizon) * slope * previous effect``; each side's means are then
    shifted so the ``fitted`` teams (those with games in the fit) average zero, and a team outside
    the fit gets the same shift.

    Raises:
        ValueError: If a team has no previous-season effect (the ridge would read it as zero), or a
            fitted team is not in ``games``.

    """
    missing = sorted(
        team
        for team in games
        if team not in season_prior.offense or team not in season_prior.defense
    )
    if missing:
        msg = (
            f"the {season_prior.season} prior has no {season_prior.season - 1} effect for "
            f"{', '.join(missing)}"
        )
        raise ValueError(msg)
    unknown = sorted(set(fitted) - set(games))
    if unknown:
        msg = f"fitted teams without a game count: {', '.join(unknown)}"
        raise ValueError(msg)
    slopes = season_prior.slopes
    offense = {
        team: fade(count, horizon) * slopes.offense * season_prior.offense[team]
        for team, count in games.items()
    }
    defense = {
        team: fade(count, horizon) * slopes.defense * season_prior.defense[team]
        for team, count in games.items()
    }
    return UnitPrior(offense=_centered(offense, fitted), defense=_centered(defense, fitted))


def snapshot_prior(
    season_prior: SeasonPrior | None,
    game_logs: pl.DataFrame,
    horizon: float = PRIOR_HORIZON_GAMES,
    games: Mapping[str, int] | None = None,
) -> UnitPrior | None:
    """Return the centered prior means for a fit of ``game_logs``, one season's games so far.

    Each team's games played come from ``games`` when given (a head-to-head-excluded refit keeps the
    snapshot's counts), otherwise from ``game_logs``. The means are centered over the teams with
    scrimmage rows in ``game_logs``. ``None`` means no prior: the season has none, or every team
    has played ``horizon`` games, where every mean is zero and the plain fit is the same.
    """
    if season_prior is None:
        return None
    rows = scrimmage_rows(game_logs)
    fitted = sorted(
        set(rows.get_column("team").to_list()) | set(rows.get_column("opponent_team").to_list())
    )
    counts = dict(games) if games is not None else games_played(game_logs, fitted)
    counts = {team: counts[team] for team in fitted}
    if all(fade(count, horizon) == 0.0 for count in counts.values()):
        return None
    return prior_means(season_prior, counts, horizon, fitted)


def snapshot_ratings(
    game_logs: pl.DataFrame,
    season_teams: Sequence[str],
    penalties: TeamRatingFit | None,
    season_prior: SeasonPrior | None,
    horizon: float,
) -> pl.DataFrame:
    """Return the team ratings fit on ``game_logs``, one season's games so far.

    The ratings of :func:`snapshot_fit`, without the prior means it fit with.
    """
    return snapshot_fit(game_logs, season_teams, penalties, season_prior, horizon)[0]


def snapshot_fit(
    game_logs: pl.DataFrame,
    season_teams: Sequence[str],
    penalties: TeamRatingFit | None,
    season_prior: SeasonPrior | None,
    horizon: float,
) -> tuple[pl.DataFrame, UnitPrior | None]:
    """Return the team ratings fit on ``game_logs`` and the prior means the fit used.

    Without a prior this is the published fit: ``penalties`` holds the season's penalties (the
    previous season's cross-validated fit), or the games cross-validate their own when it is
    ``None``. With a prior, the scrimmage effects shrink toward the centered prior means at those
    penalties, and every team in ``season_teams`` without a game rates at its prior means on these
    games' per-game scale, with special teams at zero.

    The means are ``None`` without a prior, so an audit can check the exact means a fit used.

    Raises:
        ValueError: If a prior comes without penalties, a team has no previous-season effect, or a
            game lacks one of its two teams' scrimmage rows.

    """
    if season_prior is None:
        return fit_team_ratings_with_previous_penalties(game_logs, penalties).ratings, None
    if penalties is None:
        msg = "a prior needs the season's penalties; cross-validation never sees a prior"
        raise ValueError(msg)
    rows = scrimmage_rows(game_logs)
    fitted = sorted(rows.get_column("team").unique().to_list())
    if fitted != sorted(rows.get_column("opponent_team").unique().to_list()):
        msg = "every game needs both teams' scrimmage rows to center the prior"
        raise ValueError(msg)
    means = prior_means(season_prior, games_played(game_logs, season_teams), horizon, fitted)
    fit = fit_team_ratings(
        game_logs,
        scrimmage_lambda=penalties.scrimmage_lambda,
        special_teams_lambda=penalties.special_teams_lambda,
        scrimmage_prior=means,
    )
    rated = set(fit.ratings.get_column("team").to_list())
    unplayed = [team for team in season_teams if team not in rated]
    if not unplayed:
        return fit.ratings, means
    scale = fit.scrimmage_plays_per_game
    offense = [means.offense[team] * scale for team in unplayed]
    defense = [means.defense[team] * scale for team in unplayed]
    prior_rows = pl.DataFrame(
        {
            "team": unplayed,
            "offense_rating": offense,
            "defense_rating": defense,
            "special_teams_rating": [0.0] * len(unplayed),
            "team_rating": [o + d for o, d in zip(offense, defense, strict=True)],
        }
    )
    return pl.concat([fit.ratings, prior_rows.select(fit.ratings.columns)]).sort("team"), means


# The columns of a season's ``team_prior`` file: the means a fit used, per team and side.
TEAM_PRIOR_COLUMNS = ("excluded_team", "team", "offense_prior", "defense_prior")
_TEAM_PRIOR_SCHEMA = {
    "excluded_team": pl.String,
    "team": pl.String,
    "offense_prior": pl.Float64,
    "defense_prior": pl.Float64,
}


def team_prior_table(
    season_means: UnitPrior | None,
    prior_without: Callable[[str], UnitPrior | None] | None,
    teams: Sequence[str],
) -> pl.DataFrame:
    """Return the prior means a season's fits used, for the garbage-time filter to refit with.

    One row per team with the season fit's means (``excluded_team`` null), then, for each of
    ``teams`` in order, one row per other team with the means of the refit that leaves it out.
    Empty without a prior, so a season whose priors have faded keeps no stale means.
    """
    rows: list[tuple[str | None, str, float, float]] = []
    if season_means is not None:
        rows.extend(
            (None, team, season_means.offense[team], season_means.defense[team])
            for team in sorted(season_means.offense)
        )
    if prior_without is not None:
        for left_out in teams:
            means = prior_without(left_out)
            if means is not None:
                rows.extend(
                    (left_out, team, means.offense[team], means.defense[team])
                    for team in sorted(means.offense)
                )
    return pl.DataFrame(rows, schema=_TEAM_PRIOR_SCHEMA, orient="row")


def read_team_prior_table(table: pl.DataFrame) -> tuple[UnitPrior | None, dict[str, UnitPrior]]:
    """Return the season fit's means and each left-out team's from a ``team_prior`` table."""
    by_group: dict[str | None, UnitPrior] = {}
    for (left_out,), group in table.group_by("excluded_team", maintain_order=True):
        teams = group.get_column("team").to_list()
        by_group[None if left_out is None else str(left_out)] = UnitPrior(
            offense=dict(zip(teams, group.get_column("offense_prior").to_list(), strict=True)),
            defense=dict(zip(teams, group.get_column("defense_prior").to_list(), strict=True)),
        )
    season_means = by_group.pop(None, None)
    return season_means, {str(team): means for team, means in by_group.items()}


def prior_without_team(
    history: PriorHistory,
    season: int,
    game_logs: pl.DataFrame,
    horizon: float = PRIOR_HORIZON_GAMES,
) -> Callable[[str], UnitPrior | None]:
    """Return the prior source for the refits of ``game_logs`` that leave out one team.

    For a left-out team, the means come from the previous season refit without that team
    (:meth:`PriorHistory.season_prior_without`), faded by each other team's games in the whole
    snapshot (its game against the left-out team included) and centered over the teams left in
    the refit. ``None`` when no other game remains or the snapshot has no prior.
    """
    teams = sorted(
        set(game_logs.get_column("team").to_list())
        | set(game_logs.get_column("opponent_team").to_list())
    )
    counts = games_played(game_logs, teams)

    def prior(team: str) -> UnitPrior | None:
        others = game_logs.filter((pl.col("team") != team) & (pl.col("opponent_team") != team))
        if others.is_empty():
            return None
        return snapshot_prior(
            history.season_prior_without(season, team), others, horizon, games=counts
        )

    return prior


class PriorHistory:
    """The prior's inputs season by season, from one source of seasons' team game logs.

    Each quantity is computed once and cached. ``load_game_logs`` is called only for the seasons a
    result needs, and a season's prior needs only earlier seasons, so a history whose source holds
    only the seasons before ``s`` gives the same prior for ``s`` as a full one.
    """

    def __init__(
        self,
        load_game_logs: Callable[[int], pl.DataFrame],
        *,
        first_season: int = constants.NFLREADPY_MIN_SEASON,
    ) -> None:
        """Remember the source of game logs and the first season it holds.

        ``first_season`` is the first play-by-play season (1999), which cross-validates its own
        penalties because no earlier season exists.
        """
        self._load = load_game_logs
        self._first_season = first_season
        self._logs: dict[int, pl.DataFrame] = {}
        self._cross_validated: dict[int, TeamRatingFit] = {}
        self._published: dict[int, UnitFit] = {}
        self._unpenalized: dict[int, UnitFit] = {}
        self._without: dict[tuple[int, str], SeasonPrior | None] = {}

    def game_logs(self, season: int) -> pl.DataFrame:
        """Return one season's team game logs."""
        if season not in self._logs:
            self._logs[season] = self._load(season)
        return self._logs[season]

    def cross_validated_fit(self, season: int) -> TeamRatingFit:
        """Return the season's full fit with its own cross-validated penalties."""
        if season not in self._cross_validated:
            self._cross_validated[season] = fit_team_ratings(self.game_logs(season))
        return self._cross_validated[season]

    def penalties(self, season: int) -> TeamRatingFit:
        """Return the fit whose penalties the season's published fit uses.

        That is the previous season's cross-validated fit, or the season's own for the first one.
        """
        if season > self._first_season:
            return self.cross_validated_fit(season - 1)
        return self.cross_validated_fit(season)

    def published_scrimmage(self, season: int) -> UnitFit:
        """Return the scrimmage effects of the season's published full-season fit."""
        if season not in self._published:
            self._published[season] = fit_unit_ridge(
                scrimmage_rows(self.game_logs(season)),
                TEAM_UNIT_COLUMNS,
                ridge_lambda=self.penalties(season).scrimmage_lambda,
            )
        return self._published[season]

    def near_unpenalized_scrimmage(self, season: int) -> UnitFit:
        """Return the season's scrimmage effects at ``NEAR_UNPENALIZED_LAMBDA``."""
        if season not in self._unpenalized:
            self._unpenalized[season] = fit_unit_ridge(
                scrimmage_rows(self.game_logs(season)),
                TEAM_UNIT_COLUMNS,
                ridge_lambda=NEAR_UNPENALIZED_LAMBDA,
            )
        return self._unpenalized[season]

    def slopes(self, season: int) -> CarryoverSlopes | None:
        """Return the carryover slopes for ``season``, or ``None`` without enough season pairs.

        Pools every pair ``(t - 1, t)`` with ``t`` before ``season``, team by team for the teams in
        both seasons: season ``t``'s near-unpenalized effects on season ``t - 1``'s published ones.
        """
        later_seasons = range(self._first_season + 1, season)
        if len(later_seasons) < MIN_CARRYOVER_PAIRS:
            return None
        offense: tuple[list[float], list[float]] = ([], [])
        defense: tuple[list[float], list[float]] = ([], [])
        for later in later_seasons:
            before = self.published_scrimmage(later - 1)
            after = self.near_unpenalized_scrimmage(later)
            for team in sorted(set(before.offense) & set(after.offense)):
                offense[0].append(before.offense[team])
                offense[1].append(after.offense[team])
            for team in sorted(set(before.defense) & set(after.defense)):
                defense[0].append(before.defense[team])
                defense[1].append(after.defense[team])
        return CarryoverSlopes(
            offense=carryover_slope(*offense),
            defense=carryover_slope(*defense),
            pairs=len(later_seasons),
        )

    def season_prior_without(self, season: int, team: str) -> SeasonPrior | None:
        """Return the season's prior built from a previous season that leaves out ``team``.

        A head-to-head-excluded refit (``sos``) rates the opponents without the evaluated team's
        games, so their prior means come from the previous season refit without that team's games
        too, at the published fit's penalty, and those games never shape them. The slopes are the
        season's own, pooled over every team as the penalty is. ``None`` when the season has no
        prior.
        """
        if (season, team) in self._without:
            return self._without[(season, team)]
        slopes = self.slopes(season)
        prior = None
        if slopes is not None:
            previous = self.game_logs(season - 1)
            others = previous.filter((pl.col("team") != team) & (pl.col("opponent_team") != team))
            fit = fit_unit_ridge(
                scrimmage_rows(others),
                TEAM_UNIT_COLUMNS,
                ridge_lambda=self.penalties(season - 1).scrimmage_lambda,
            )
            prior = SeasonPrior(
                season=season, offense=dict(fit.offense), defense=dict(fit.defense), slopes=slopes
            )
        self._without[(season, team)] = prior
        return prior

    def season_prior(self, season: int) -> SeasonPrior | None:
        """Return what the season's prior is built from, or ``None`` when it has no prior."""
        slopes = self.slopes(season)
        if slopes is None:
            return None
        previous = self.published_scrimmage(season - 1)
        return SeasonPrior(
            season=season,
            offense=dict(previous.offense),
            defense=dict(previous.defense),
            slopes=slopes,
        )


__all__ = [
    "MIN_CARRYOVER_PAIRS",
    "NEAR_UNPENALIZED_LAMBDA",
    "PRIOR_HORIZON_GAMES",
    "TEAM_PRIOR_COLUMNS",
    "CarryoverSlopes",
    "PriorHistory",
    "SeasonPrior",
    "carryover_slope",
    "fade",
    "games_played",
    "prior_means",
    "prior_without_team",
    "read_team_prior_table",
    "snapshot_fit",
    "snapshot_prior",
    "snapshot_ratings",
    "team_prior_table",
]
