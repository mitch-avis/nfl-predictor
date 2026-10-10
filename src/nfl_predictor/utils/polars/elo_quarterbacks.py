"""The own Elo's quarterback adjustment: FiveThirtyEight's VALUE model.

FiveThirtyEight (538) added a quarterback adjustment to its NFL Elo in 2019 ("How Our NFL
Predictions Work", version 2.0). This module computes it from this repository's play-by-play
and hands ``elo_ratings`` one number per side of every game: the Elo points the starting
quarterback adds to or takes from his team's rating. Any other quarterback rating can replace it
through the same frame (``game_id``, ``home_qb_adj``, ``away_qb_adj``).

VALUE
    538's per-game box-score value of a quarterback::

        -2.2 * pass attempts + 3.7 * completions + pass yards / 5 + 11.3 * pass touchdowns
        - 14.1 * interceptions - 8 * times sacked - 1.1 * rush attempts + 0.6 * rush yards
        + 15.9 * rush touchdowns

    The box scores come from the cached regular-season play-by-play (``quarterback_box_scores``):
    the passing lines are ``qb_stats.aggregate_qb_game_stats``'s, and a quarterback's rushing is
    his scrambles (dropbacks without a pass) and kneels. The cache keeps the passer id but not
    the rusher id, so designed quarterback runs are left out, and scrambles and kneels go to the
    team-game's primary passer (the most dropbacks with a passer id), as in ``qb_stats``. The
    starter is the passer of the team's first dropback with a passer id, in play order (the order
    the frame lists each game's plays in, which the cache keeps).

Defense
    Each team carries a rolling rating of the quarterback VALUE it allows, relative to the
    league: after a game, ``defense = 0.95 * defense + 0.05 * (allowed - league)``, where
    ``allowed`` is the opponent's team-game VALUE (all its quarterbacks) and ``league`` the mean
    team-game VALUE of that week. A starter's game value is adjusted by the opposing defense's
    pre-game rating: ``adjusted = VALUE - defense``. 538 publishes the adjustment but not its
    rolling rate; this uses its team rate (20 games), and reverts the rating a third of the way
    to 0 each offseason, like the team Elo.

Rolling ratings
    A quarterback's rating moves a tenth of the way to each start's adjusted value
    (``0.9 * old + 0.1 * game``, 538's "every 10 games"), his team's a twentieth of the way to
    the same value (``0.95 * old + 0.05 * game``). Only starts update anything, and only
    regular-season games: the play-by-play cache holds no playoff plays. The adjustment for a
    game is ``3.3 * (starter's rating - team's rating)``, 538's VALUE-to-Elo factor.

Preseason
    A quarterback with 10 to 100 career starts reverts a quarter of the way toward the mean
    adjusted start of the previous season; one with fewer or more keeps his rating. A team's
    rating is a third its end-of-season rating and two thirds its first starter's preseason
    rating, and a team seen for the first time starts at its starter's rating. 538 rates a
    rookie by draft position (0 undrafted, about 34 for the first pick), which this repository
    does not hold: a quarterback's first start in the data is rated at the mean adjusted value
    of every first start in earlier seasons (the frame's first season excluded, where veterans
    start too), or 0 before there are any.

A game's starter is the play-by-play starter for a regular-season game with plays, then the
starter ``projected_starters`` names, then the team's last starter. Every value for a game comes
from games strictly before it.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor.utils.polars import pbp, qb_stats

if TYPE_CHECKING:
    from collections.abc import Mapping

# 538's VALUE weights per box-score column.
VALUE_WEIGHTS: dict[str, float] = {
    "attempts": -2.2,
    "completions": 3.7,
    "pass_yards": 1.0 / 5.0,
    "pass_tds": 11.3,
    "interceptions": -14.1,
    "sacks": -8.0,
    "rush_attempts": -1.1,
    "rush_yards": 0.6,
    "rush_tds": 15.9,
}
# Elo points per point of VALUE.
VALUE_TO_ELO = 3.3
QUARTERBACK_WEIGHT = 0.1
TEAM_WEIGHT = 0.05
DEFENSE_WEIGHT = 0.05
DEFENSE_OFFSEASON_REVERSION = 1.0 / 3.0
VETERAN_REVERSION = 0.25
VETERAN_MIN_STARTS = 10
VETERAN_MAX_STARTS = 100
# A team's preseason rating keeps this share of its end-of-season rating.
TEAM_PRESEASON_CARRYOVER = 1.0 / 3.0

BOX_SCORE_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int64(),
    "week": pl.Int64(),
    "team_abbr": pl.String(),
    "qb_id": pl.String(),
    "passer_name": pl.String(),
    "starter": pl.Boolean(),
    **{name: pl.Float64() for name in VALUE_WEIGHTS},
    "value": pl.Float64(),
}
ADJUSTMENT_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.String(),
    "home_qb_id": pl.String(),
    "away_qb_id": pl.String(),
    "home_qb_value": pl.Float64(),
    "away_qb_value": pl.Float64(),
    "home_team_qb_value": pl.Float64(),
    "away_team_qb_value": pl.Float64(),
    "home_qb_adj": pl.Float64(),
    "away_qb_adj": pl.Float64(),
}
_TEAM_GAME = ["season", "week", "posteam"]
_RUSH_COLUMNS = ("rush_attempts", "rush_yards", "rush_tds")
_SORT_COLUMNS = ("season", "week", "date", "game_id")


def quarterback_value() -> pl.Expr:
    """Return 538's VALUE over the box-score columns of ``VALUE_WEIGHTS``, named ``value``."""
    terms = [weight * pl.col(name) for name, weight in VALUE_WEIGHTS.items()]
    return functools.reduce(operator.add, terms).alias("value")


def _rushing(plays: pl.DataFrame, dropbacks: pl.DataFrame) -> pl.DataFrame:
    """Return scramble and kneel rushing per team-game: attempts, yards and touchdowns."""
    columns = plays.columns
    scrambles = dropbacks.filter(
        pl.col("passer_player_id").is_null() & (pbp.flag_expr(columns, "rush_attempt") > 0)
    )
    kneels = plays.filter(
        (pbp.flag_expr(columns, "qb_kneel") > 0)
        & (pbp.flag_expr(columns, "two_point_attempt") == 0)
    )
    carries = pl.concat(
        [
            frame.select(
                *_TEAM_GAME,
                pbp.value_expr(columns, "yards_gained").alias("yards"),
                pbp.flag_expr(columns, "rush_touchdown").alias("touchdown"),
            )
            for frame in (scrambles, kneels)
        ]
    )
    return carries.group_by(_TEAM_GAME, maintain_order=True).agg(
        pl.len().cast(pl.Float64).alias("rush_attempts"),
        pl.col("yards").sum().alias("rush_yards"),
        pl.col("touchdown").sum().alias("rush_tds"),
    )


def quarterback_box_scores(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Return one row per quarterback per regular-season team-game, with VALUE.

    Args:
        pbp_df: Play-by-play rows with each game's plays in order (the cache's order).

    Returns:
        ``BOX_SCORE_SCHEMA``: the keys, the passer name, whether he started (passed on his
        team's first dropback), the nine box-score inputs and ``value``, sorted by season,
        week, team and id. Definitions are in the module docstring.

    """
    ordered = pbp_df.with_row_index("_play_order")
    passing = qb_stats.aggregate_qb_game_stats(ordered)
    if passing.height == 0:
        return pl.DataFrame(schema=BOX_SCORE_SCHEMA)
    plays = pbp.regular_season_plays(ordered)
    dropbacks = plays.filter(pbp.dropback_condition(plays.columns))
    with_passer = dropbacks.filter(pl.col("passer_player_id").is_not_null()).with_columns(
        pl.col("passer_player_id").cast(pl.String)
    )
    starters = (
        with_passer.sort("_play_order")
        .group_by(_TEAM_GAME, maintain_order=True)
        .agg(pl.col("passer_player_id").first().alias("starter_id"))
        .rename({"posteam": "team_abbr"})
    )
    primary = (
        with_passer.group_by([*_TEAM_GAME, "passer_player_id"], maintain_order=True)
        .len()
        .sort(
            [*_TEAM_GAME, "len", "passer_player_id"], descending=[False, False, False, True, False]
        )
        .unique(_TEAM_GAME, keep="first", maintain_order=True)
        .select(*_TEAM_GAME, pl.col("passer_player_id").alias("qb_id"))
    )
    rushing = (
        _rushing(plays, dropbacks)
        .join(primary, on=_TEAM_GAME, how="inner")
        .rename({"posteam": "team_abbr"})
    )
    keys = ["season", "week", "team_abbr"]
    box = (
        passing.join(rushing, on=[*keys, "qb_id"], how="left")
        .with_columns(pl.col(name).fill_null(0.0) for name in _RUSH_COLUMNS)
        .join(starters, on=keys, how="left")
        .with_columns(
            (pl.col("qb_id") == pl.col("starter_id")).fill_null(value=False).alias("starter")
        )
        .with_columns(quarterback_value())
    )
    return box.select(pl.col(name).cast(dtype) for name, dtype in BOX_SCORE_SCHEMA.items()).sort(
        "season", "week", "team_abbr", "qb_id"
    )


@dataclass
class _Quarterback:
    """A quarterback's rolling rating, career starts and the season he was last rated for."""

    rating: float
    season: int
    starts: int = 0


@dataclass
class _Team:
    """A team's rolling quarterback rating, its defense rating and its current season."""

    rating: float
    season: int
    defense: float = 0.0


@dataclass
class _PreGame:
    """One side's pre-game values."""

    starter: str | None
    quarterback: float | None
    team: float
    defense: float

    @property
    def adjustment(self) -> float:
        """Return the Elo points the starter adds to his team: 3.3 * (starter - team)."""
        if self.quarterback is None:
            return 0.0
        return VALUE_TO_ELO * (self.quarterback - self.team)


@dataclass
class _Book:
    """Every quarterback's and team's state and the league's per-season start values."""

    first_season: int
    quarterbacks: dict[str, _Quarterback] = field(default_factory=dict)
    teams: dict[str, _Team] = field(default_factory=dict)
    start_totals: dict[int, list[float]] = field(default_factory=dict)
    debut_totals: dict[int, list[float]] = field(default_factory=dict)

    def debut_rating(self, season: int) -> float:
        """Return the mean adjusted first start of earlier seasons (the first excluded), or 0."""
        values = [
            value
            for year, totals in self.debut_totals.items()
            if self.first_season < year < season
            for value in totals
        ]
        return sum(values) / len(values) if values else 0.0

    def quarterback(self, qb_id: str, season: int) -> float:
        """Return a quarterback's rating for ``season``, debuting or reverting him first."""
        state = self.quarterbacks.get(qb_id)
        if state is None:
            state = _Quarterback(rating=self.debut_rating(season), season=season)
            self.quarterbacks[qb_id] = state
        elif state.season < season:
            previous = self.start_totals.get(season - 1)
            if previous and VETERAN_MIN_STARTS <= state.starts <= VETERAN_MAX_STARTS:
                average = sum(previous) / len(previous)
                state.rating += VETERAN_REVERSION * (average - state.rating)
            state.season = season
        return state.rating

    def team(self, team: str, season: int, starter_rating: float | None) -> _Team:
        """Return a team's state for ``season``, starting it or running its preseason step."""
        state = self.teams.get(team)
        if state is None:
            initial = starter_rating if starter_rating is not None else self.debut_rating(season)
            state = _Team(rating=initial, season=season)
            self.teams[team] = state
        elif state.season < season:
            if starter_rating is not None:
                state.rating = (
                    TEAM_PRESEASON_CARRYOVER * state.rating
                    + (1.0 - TEAM_PRESEASON_CARRYOVER) * starter_rating
                )
            state.defense *= 1.0 - DEFENSE_OFFSEASON_REVERSION
            state.season = season
        return state

    def pre_game(self, team: str, starter: str | None, season: int) -> _PreGame:
        """Return one side's pre-game values, running any preseason step they need."""
        rating = None if starter is None else self.quarterback(starter, season)
        state = self.team(team, season, rating)
        return _PreGame(starter, rating, state.rating, state.defense)

    def record_start(self, qb_id: str, team: str, adjusted: float, season: int) -> None:
        """Roll a start's defense-adjusted value into the quarterback's and team's ratings."""
        quarterback = self.quarterbacks[qb_id]
        if quarterback.starts == 0 and season > self.first_season:
            self.debut_totals.setdefault(season, []).append(adjusted)
        quarterback.rating = (
            1.0 - QUARTERBACK_WEIGHT
        ) * quarterback.rating + QUARTERBACK_WEIGHT * adjusted
        quarterback.starts += 1
        state = self.teams[team]
        state.rating = (1.0 - TEAM_WEIGHT) * state.rating + TEAM_WEIGHT * adjusted
        self.start_totals.setdefault(season, []).append(adjusted)

    def record_defense(self, team: str, excess_allowed: float) -> None:
        """Roll the VALUE a defense allowed above the week's league mean into its rating."""
        state = self.teams[team]
        state.defense = (1.0 - DEFENSE_WEIGHT) * state.defense + DEFENSE_WEIGHT * excess_allowed


@dataclass(frozen=True)
class _BoxLookup:
    """The box scores keyed for the rating loop."""

    starts: Mapping[tuple[int, int, str], tuple[str, float]]
    team_values: Mapping[tuple[int, int, str], float]
    week_means: Mapping[tuple[int, int], float]


def _box_lookup(box_scores: pl.DataFrame) -> _BoxLookup:
    """Return each team-game's starter and his value, its team value, and each week's mean."""
    starts: dict[tuple[int, int, str], tuple[str, float]] = {}
    for season, week, team, qb_id, value in (
        box_scores.filter(pl.col("starter"))
        .select("season", "week", "team_abbr", "qb_id", "value")
        .iter_rows()
    ):
        starts.setdefault((season, week, team), (qb_id, value))
    team_games = box_scores.group_by("season", "week", "team_abbr", maintain_order=True).agg(
        pl.col("value").sum()
    )
    team_values = {
        (season, week, team): value
        for season, week, team, value in team_games.select(
            "season", "week", "team_abbr", "value"
        ).iter_rows()
    }
    weekly = team_games.group_by("season", "week", maintain_order=True).agg(pl.col("value").mean())
    week_means = {
        (season, week): value
        for season, week, value in weekly.select("season", "week", "value").iter_rows()
    }
    return _BoxLookup(starts, team_values, week_means)


def value_qb_adjustments(
    schedule: pl.DataFrame,
    box_scores: pl.DataFrame,
    *,
    projected_starters: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return each game's quarterback adjustments in Elo points, from 538's VALUE model.

    Args:
        schedule: Schedule rows with ``game_id``, ``season``, ``week``, ``home_abbr``,
            ``away_abbr`` and, when present, ``date``; every game, upcoming ones included.
        box_scores: ``quarterback_box_scores`` output (``season``, ``week``, ``team_abbr``,
            ``qb_id``, ``starter`` and ``value`` are read).
        projected_starters: Optional ``game_id``, ``team_abbr`` and ``qb_id`` naming the
            expected starter of a game the play-by-play does not cover (an upcoming one).

    Returns:
        ``ADJUSTMENT_SCHEMA``, one row per game in rating order: each side's starter, his and
        his team's pre-game rolling VALUE, and the adjustment ``3.3 * (starter - team)`` (0
        when no starter is known). The method is in the module docstring.

    """
    keys = [column for column in _SORT_COLUMNS if column in schedule.columns]
    games = schedule.sort(keys, maintain_order=True)
    if games.height == 0:
        return pl.DataFrame(schema=ADJUSTMENT_SCHEMA)
    lookup = _box_lookup(box_scores)
    projected: dict[tuple[str, str], str] = {}
    if projected_starters is not None:
        for game_id, team, qb_id in projected_starters.select(
            "game_id", "team_abbr", "qb_id"
        ).iter_rows():
            projected[(game_id, team)] = qb_id
    book = _Book(first_season=min(games.get_column("season").cast(pl.Int64).to_list()))
    last_starter: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for game_id, season, week, home, away in games.select(
        "game_id", "season", "week", "home_abbr", "away_abbr"
    ).iter_rows():
        sides = {home: away, away: home}
        pre: dict[str, _PreGame] = {}
        for team in sides:
            start = lookup.starts.get((season, week, team))
            starter = (
                start[0]
                if start is not None
                else projected.get((game_id, team), last_starter.get(team))
            )
            pre[team] = book.pre_game(team, starter, season)
        for team, opponent in sides.items():
            start = lookup.starts.get((season, week, team))
            if start is not None:
                qb_id, value = start
                book.record_start(qb_id, team, value - pre[opponent].defense, season)
                last_starter[team] = qb_id
            allowed = lookup.team_values.get((season, week, opponent))
            if allowed is not None:
                book.record_defense(team, allowed - lookup.week_means[(season, week)])
        rows.append(
            {
                "game_id": game_id,
                **{
                    f"{side}_{name}": value
                    for side, team in (("home", home), ("away", away))
                    for name, value in (
                        ("qb_id", pre[team].starter),
                        ("qb_value", pre[team].quarterback),
                        ("team_qb_value", pre[team].team),
                        ("qb_adj", pre[team].adjustment),
                    )
                },
            }
        )
    return pl.DataFrame(rows, schema=ADJUSTMENT_SCHEMA)
