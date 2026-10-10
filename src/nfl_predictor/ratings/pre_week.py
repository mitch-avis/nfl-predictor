"""Leakage-safe ratings for model features: each fit sees only the games before the week.

For one season and week, :meth:`RatingsHistory.pre_week` returns what the ratings knew before that
week's first game, so a feature built from them never sees the game it describes or any later one:

- Team ratings (offense, defense, special teams, and their sum, in points per game) are fit on the
  season's regular-season games with ``week`` before the requested week, with the previous
  season's cross-validated penalties and the preseason prior as of those games: each team's prior
  fades with the games it has played before the week (none after nine), built from earlier seasons
  only. A team without a game yet rates at its prior on the snapshot's scale, with special teams
  at zero. For a week after the first, the ratings of the teams that have played equal the
  published week-by-week history of the week before.
- ``sos`` is the mean rating of the opponents a team has played before the week, each from a refit
  of those games without the team's, with priors from the previous season refit without it.
- Quarterback ratings (adjusted EPA per dropback and the faced pass defense) are fit on the
  passer-games before the week with the previous season's cross-validated quarterback penalty.
  The published quarterback history uses the season's own cross-validated penalty instead, which
  depends on the whole season, so the two differ slightly.

What a week returns when there is little or nothing to fit:

- Week 1 (no games yet): every team at its preseason prior, centered over the teams given, on the
  previous season's league-average scrimmage plays per team-game, with special teams at zero,
  ``sos`` null, and no quarterback rows. A season without a prior (before the carryover slopes
  have three earlier season pairs, so before 2003) has null team ratings that week.
- The first season (1999): null team ratings and no quarterback rows in every week, because no
  earlier season exists to supply the penalties.
- A postseason week: the fit of the whole regular season, as in that season's published ratings.

The team universe is the caller's list of the season's teams (from its schedule, known before
kickoff), plus any team in the games so far.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor import constants
from nfl_predictor.ratings.inputs import qb_game_rows, team_game_rows
from nfl_predictor.ratings.qb_rating import (
    QB_DROPBACKS_COLUMN,
    QB_FACED_DEFENSE_COLUMN,
    QB_ID_COLUMN,
    QB_RATING_COLUMN,
    compute_qb_faced_pass_defense,
    fit_qb_ratings,
    qb_rating_rows,
)
from nfl_predictor.ratings.team_prior import (
    PRIOR_HORIZON_GAMES,
    PriorHistory,
    SeasonPrior,
    prior_means,
    prior_without_team,
    snapshot_fit,
    snapshot_prior,
)
from nfl_predictor.ratings.team_rating import (
    SCRIMMAGE_PLAYS_COLUMN,
    TEAM_RATING_COLUMNS,
    compute_team_schedule_strength,
    fit_team_ratings_with_previous_penalties,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

TEAM_PRE_WEEK_COLUMNS = ("team", "games_played", *TEAM_RATING_COLUMNS, "sos")
QB_PRE_WEEK_COLUMNS = (
    QB_ID_COLUMN,
    "qb_games_played",
    QB_DROPBACKS_COLUMN,
    QB_RATING_COLUMN,
    QB_FACED_DEFENSE_COLUMN,
)
_TEAM_PRE_WEEK_SCHEMA = pl.Schema(
    {
        "team": pl.String(),
        "games_played": pl.Int64(),
        **{column: pl.Float64() for column in (*TEAM_RATING_COLUMNS, "sos")},
    }
)
_QB_PRE_WEEK_SCHEMA = pl.Schema(
    {
        QB_ID_COLUMN: pl.String(),
        "qb_games_played": pl.Int64(),
        QB_DROPBACKS_COLUMN: pl.Int64(),
        QB_RATING_COLUMN: pl.Float64(),
        QB_FACED_DEFENSE_COLUMN: pl.Float64(),
    }
)


@dataclass(frozen=True, slots=True)
class PreWeekRatings:
    """The ratings known before one week's games, for the features of those games.

    Attributes:
        season: The season.
        week: The week whose games the ratings are for; they come from the games before it.
        teams: One row per team, sorted by team, with ``TEAM_PRE_WEEK_COLUMNS``.
        quarterbacks: One row per passer with a dropback before the week, sorted by ``qb_id``,
            with ``QB_PRE_WEEK_COLUMNS``.

    """

    season: int
    week: int
    teams: pl.DataFrame
    quarterbacks: pl.DataFrame


class RatingsHistory:
    """Every season's team and quarterback game rows, with the fits pre-week ratings reuse.

    Each season's rows are loaded once, checked against :mod:`nfl_predictor.ratings.inputs`, and
    every fit is cached, so a caller can ask for every week of every season in turn.
    """

    def __init__(
        self,
        load_team_games: Callable[[int], pl.DataFrame],
        load_qb_games: Callable[[int], pl.DataFrame],
        *,
        first_season: int = constants.NFLREADPY_MIN_SEASON,
    ) -> None:
        """Remember where each season's rows come from and the first season there is.

        Args:
            load_team_games: Returns one season's regular-season team-game rows.
            load_qb_games: Returns one season's regular-season quarterback-game rows.
            first_season: The first season with rows; it has no earlier season for penalties.

        """
        self._team = PriorHistory(
            lambda season: team_game_rows(load_team_games(season)), first_season=first_season
        )
        self._load_qb = load_qb_games
        self._qb_games: dict[int, pl.DataFrame] = {}
        self._qb_penalties: dict[int, float] = {}

    @property
    def team_priors(self) -> PriorHistory:
        """Return the team fits' history: penalties, carryover slopes, and season priors."""
        return self._team

    def qb_games(self, season: int) -> pl.DataFrame:
        """Return one season's quarterback-game rows."""
        if season not in self._qb_games:
            self._qb_games[season] = qb_game_rows(self._load_qb(season))
        return self._qb_games[season]

    def qb_penalty(self, season: int) -> float | None:
        """Return the quarterback penalty known before ``season``, or ``None`` for the first.

        That is the penalty cross-validation chose for the whole previous season.
        """
        if season <= self._team.first_season:
            return None
        previous = season - 1
        if previous not in self._qb_penalties:
            self._qb_penalties[previous] = fit_qb_ratings(self.qb_games(previous)).ridge_lambda
        return self._qb_penalties[previous]

    def pre_week(self, season: int, week: int, teams: Collection[str]) -> PreWeekRatings:
        """Return the ratings fit on ``season``'s games before ``week``.

        Args:
            season: The season.
            week: The week whose games the ratings are for (a postseason week takes the whole
                regular season).
            teams: The season's teams, so a team without a game yet still gets a row.

        Returns:
            The team and quarterback ratings known before the week.

        Raises:
            ValueError: If loaded rows break the input contract, or, in a season with a prior,
                a team has no previous-season effect or a game lacks one of its team rows.

        """
        return PreWeekRatings(
            season=season,
            week=week,
            teams=self.pre_week_teams(season, week, teams),
            quarterbacks=self.pre_week_quarterbacks(season, week),
        )

    def pre_week_teams(self, season: int, week: int, teams: Collection[str]) -> pl.DataFrame:
        """Return the team part of :meth:`pre_week`: one row per team, sorted by team.

        Raises:
            ValueError: As :meth:`pre_week` does for the team rows.

        """
        snapshot = self._team.game_logs(season).filter(pl.col("week") < week)
        universe = sorted(
            set(teams)
            | set(snapshot.get_column("team").to_list())
            | set(snapshot.get_column("opponent_team").to_list())
        )
        games_played = snapshot.group_by("team").len("games_played")
        frame = (
            pl.DataFrame({"team": universe}, schema={"team": pl.String()})
            .join(games_played, on="team", how="left")
            .with_columns(pl.col("games_played").fill_null(0).cast(pl.Int64))
        )
        if season <= self._team.first_season:
            return _with_null_ratings(frame)
        penalties = self._team.penalties(season)
        season_prior = self._team.season_prior(season)
        if snapshot.filter(pl.col(SCRIMMAGE_PLAYS_COLUMN) > 0).is_empty():
            if season_prior is None:
                return _with_null_ratings(frame)
            prior = _preseason_ratings(season_prior, universe, penalties.scrimmage_plays_per_game)
            return (
                frame.join(prior, on="team", how="left")
                .with_columns(pl.lit(None, dtype=pl.Float64).alias("sos"))
                .select(TEAM_PRE_WEEK_COLUMNS)
                .cast(_TEAM_PRE_WEEK_SCHEMA)
            )
        ratings, _ = snapshot_fit(snapshot, universe, penalties, season_prior, PRIOR_HORIZON_GAMES)
        means = snapshot_prior(season_prior, snapshot)
        fit = fit_team_ratings_with_previous_penalties(snapshot, penalties, scrimmage_prior=means)
        prior_without = None if means is None else prior_without_team(self._team, season, snapshot)
        schedule = compute_team_schedule_strength(snapshot, fit, prior_without=prior_without)
        return (
            frame.join(ratings, on="team", how="left")
            .join(schedule, on="team", how="left")
            .select(TEAM_PRE_WEEK_COLUMNS)
            .cast(_TEAM_PRE_WEEK_SCHEMA)
        )

    def pre_week_quarterbacks(self, season: int, week: int) -> pl.DataFrame:
        """Return the quarterback part of :meth:`pre_week`: one row per passer, sorted by id.

        Raises:
            ValueError: If loaded rows break the input contract.

        """
        penalty = self.qb_penalty(season)
        if penalty is None:
            return pl.DataFrame(schema=_QB_PRE_WEEK_SCHEMA)
        snapshot = self.qb_games(season).filter(pl.col("week") < week)
        rated = qb_rating_rows(snapshot)
        if rated.is_empty():
            return pl.DataFrame(schema=_QB_PRE_WEEK_SCHEMA)
        fit = fit_qb_ratings(snapshot, ridge_lambda=penalty)
        volume = rated.group_by(QB_ID_COLUMN).agg(
            pl.len().cast(pl.Int64).alias("qb_games_played"),
            pl.col(QB_DROPBACKS_COLUMN).sum().cast(pl.Int64),
        )
        return (
            fit.ratings.join(volume, on=QB_ID_COLUMN)
            .join(compute_qb_faced_pass_defense(snapshot, fit), on=QB_ID_COLUMN)
            .select(QB_PRE_WEEK_COLUMNS)
            .sort(QB_ID_COLUMN)
        )


def _preseason_ratings(season_prior: SeasonPrior, teams: list[str], scale: float) -> pl.DataFrame:
    """Return every team's ratings at its full preseason prior, before any game is played.

    The prior means are centered over ``teams`` and converted to points per game with ``scale``
    plays per team-game; special teams are zero.
    """
    means = prior_means(season_prior, dict.fromkeys(teams, 0), PRIOR_HORIZON_GAMES, teams)
    offense = [means.offense[team] * scale for team in teams]
    defense = [means.defense[team] * scale for team in teams]
    return pl.DataFrame(
        {
            "team": teams,
            "offense_rating": offense,
            "defense_rating": defense,
            "special_teams_rating": [0.0] * len(teams),
            "team_rating": [o + d for o, d in zip(offense, defense, strict=True)],
        }
    )


def _with_null_ratings(frame: pl.DataFrame) -> pl.DataFrame:
    """Return ``frame``'s teams and games played with every rating and ``sos`` null."""
    return frame.with_columns(
        pl.lit(None, dtype=pl.Float64).alias(column) for column in (*TEAM_RATING_COLUMNS, "sos")
    ).select(TEAM_PRE_WEEK_COLUMNS)


__all__ = [
    "QB_PRE_WEEK_COLUMNS",
    "TEAM_PRE_WEEK_COLUMNS",
    "PreWeekRatings",
    "RatingsHistory",
]
