"""This repository's own Elo ratings, computed from game results with FiveThirtyEight's method.

FiveThirtyEight (538) published its NFL Elo method in "How Our NFL Predictions Work" and the
code behind its classic ratings in ``fivethirtyeight/nfl-elo-game`` (``forecast.py``, MIT
License). This module reproduces that public framework exactly, and adds 538's later additions
as separate switches of :class:`EloSettings`: the rest, travel and rolling home-field
adjustments of ``elo_game_context``, the playoff boost (538's 2019 version multiplies a
playoff game's whole difference by 1.2 before the probability, because favorites outplay
underdogs by more in the playoffs), and the quarterback adjustment (538's 2019 VALUE model,
``elo_quarterbacks``, or any other source of per-game adjustments). With the boost on, the
boosted difference is also the one the margin-of-victory multiplier damps. A quarterback
adjustment joins its side's rating for the game only: the difference, and so the probability
and the update, include it, but the update lands on the team's own rating.

The framework
-------------
Every team carries one rating. Before a game, the home team's win probability is::

    diff = home_rating - away_rating + home_field     (home_field = 65, or 0 at a neutral site)
    p = 1 / (10 ** (-diff / 400) + 1)

After the game, the winner takes ``shift`` points from the loser::

    multiplier = ln(max(margin, 1) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
    shift = 20 * multiplier * (result - p)

where ``result`` is 1, 0.5 or 0 for a home win, tie or loss, ``margin`` the absolute point
difference and ``winner_diff`` the pre-game difference from the winner's side. The second
factor damps the credit a favorite gets for a big win; for a tie it is ``2.2 / 1``, so the
multiplier is ``2.2 * ln 2`` (1.525) whatever the gap. A team's first game of a season starts
one third of the way back from its last rating toward 1505. A team seen for the first time in
the frame's first season starts at 1505 (or at the seed it is given), and one first seen later
is an expansion team at 1300.

Games are rated in season, week, date and ``game_id`` order. Every value for a game comes from
games strictly before it: a game's pre-game ratings never include its own result, and a game
without a final score (an upcoming one) is rated but moves no rating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor.utils.polars import elo_game_context

if TYPE_CHECKING:
    from collections.abc import Mapping

# 538's published framework constants ("How Our NFL Predictions Work"; nfl-elo-game forecast.py).
MEAN_RATING = 1505.0
K_FACTOR = 20.0
# Elo points per factor of ten in the odds: p = 1 / (10 ** (-diff / LOGISTIC_SCALE) + 1).
LOGISTIC_SCALE = 400.0
OFFSEASON_REVERSION = 1.0 / 3.0
EXPANSION_RATING = 1300.0
# The margin-of-victory multiplier: ln(max(margin, 1) + 1) * DAMPING / (diff * SCALE + DAMPING).
MOV_DAMPING = 2.2
MOV_DIFF_SCALE = 0.001
# 538's 2019 playoff boost: a playoff game's whole Elo difference is multiplied by this.
PLAYOFF_MULTIPLIER = 1.2
_REGULAR_SEASON = "REG"

_REQUIRED_COLUMNS = (
    "game_id",
    "season",
    "week",
    "game_type",
    "home_abbr",
    "away_abbr",
    "home_score",
    "away_score",
    "neutral",
)
_SORT_COLUMNS = ("season", "week", "date", "game_id")
_RESULT_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.String(),
    "season": pl.Int64(),
    "week": pl.Int64(),
    "home_abbr": pl.String(),
    "away_abbr": pl.String(),
    "home_rating": pl.Float64(),
    "away_rating": pl.Float64(),
    "home_qb_adj": pl.Float64(),
    "away_qb_adj": pl.Float64(),
    "home_edge": pl.Float64(),
    "elo_diff": pl.Float64(),
    "home_win_prob": pl.Float64(),
    "home_rating_post": pl.Float64(),
    "away_rating_post": pl.Float64(),
}


@dataclass(frozen=True, kw_only=True)
class EloSettings:
    """Which of FiveThirtyEight's additions to the classic Elo are switched on.

    With every switch off (the default) the ratings are 538's public framework, described in
    the module docstring.

    Attributes:
        rest: 25 points for a team coming off a bye week (``elo_game_context``).
        travel: A 55-point home field plus 4 points per 1,000 miles of the visitor's trip,
            instead of the classic 65 (``elo_game_context``).
        playoff_boost: Multiply a playoff game's whole difference by 1.2.
        rolling_home_field: Use the rolling ten-season home field as the base
            (``elo_game_context``).
        quarterback: Add each side's quarterback adjustment (``elo_quarterbacks``) to its
            rating for the game.

    """

    rest: bool = False
    travel: bool = False
    playoff_boost: bool = False
    rolling_home_field: bool = False
    quarterback: bool = False


FRAMEWORK = EloSettings()


def win_probability(diff: float) -> float:
    """Return the home win probability for an Elo difference from the home side.

    The expression is 538's, term for term, so the result matches its published
    probabilities to the last bit.
    """
    return 1.0 / (math.pow(10.0, -diff / LOGISTIC_SCALE) + 1.0)


def margin_multiplier(home_score: float, away_score: float, diff: float) -> float:
    """Return the margin-of-victory multiplier for a final score and the pre-game difference.

    Args:
        home_score: The home team's points.
        away_score: The away team's points.
        diff: The pre-game Elo difference from the home side, home field included.

    Returns:
        ``ln(max(margin, 1) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)``, with ``2.2 / 1`` as the
        second factor for a tie.

    """
    margin = abs(home_score - away_score)
    if home_score == away_score:
        damping = 1.0
    else:
        winner_diff = diff if home_score > away_score else -diff
        damping = winner_diff * MOV_DIFF_SCALE + MOV_DAMPING
    return math.log(max(margin, 1.0) + 1.0) * (MOV_DAMPING / damping)


def _result(home_score: float, away_score: float) -> float:
    """Return 1, 0.5 or 0 for a home win, tie or loss."""
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5


def _require_columns(games: pl.DataFrame, settings: EloSettings) -> None:
    """Raise when the schedule lacks a column the ratings read or repeats a game id."""
    required = [*_REQUIRED_COLUMNS, *(("stadium_id",) if settings.travel else ())]
    missing = [column for column in required if column not in games.columns]
    if missing:
        msg = f"the schedule is missing column(s): {', '.join(missing)}"
        raise ValueError(msg)
    if games.get_column("game_id").is_duplicated().any():
        msg = "the schedule lists a game_id more than once"
        raise ValueError(msg)


def _ordered(games: pl.DataFrame) -> pl.DataFrame:
    """Return the games in rating order: season, week, date when present, then game id."""
    keys = [column for column in _SORT_COLUMNS if column in games.columns]
    return games.sort(keys, maintain_order=True)


def _home_edges(games: pl.DataFrame, settings: EloSettings) -> list[float]:
    """Return the Elo points each game credits to the home side, in the order of ``games``."""
    edges = elo_game_context.home_edges(
        games,
        rest=settings.rest,
        travel=settings.travel,
        rolling=settings.rolling_home_field,
    )
    return edges.get_column("home_edge").to_list()


def _quarterback_adjustments(
    games: pl.DataFrame, settings: EloSettings, qb_adjustments: pl.DataFrame | None
) -> tuple[list[float], list[float]]:
    """Return each side's quarterback adjustment, 0 for a game the frame lacks or when off."""
    if qb_adjustments is None:
        if settings.quarterback:
            msg = "the quarterback switch is on but no qb_adjustments were given"
            raise ValueError(msg)
        zeros = [0.0] * games.height
        return zeros, list(zeros)
    if not settings.quarterback:
        msg = "qb_adjustments were given but the quarterback switch is off"
        raise ValueError(msg)
    joined = games.select("game_id").join(
        qb_adjustments.select("game_id", "home_qb_adj", "away_qb_adj"),
        on="game_id",
        how="left",
        maintain_order="left",
    )
    return (
        joined.get_column("home_qb_adj").cast(pl.Float64).fill_null(0.0).to_list(),
        joined.get_column("away_qb_adj").cast(pl.Float64).fill_null(0.0).to_list(),
    )


def _playoff_multipliers(games: pl.DataFrame, settings: EloSettings) -> list[float]:
    """Return 1.2 for a playoff game when the boost is on, else 1, in the order of ``games``."""
    if not settings.playoff_boost:
        return [1.0] * games.height
    game_types = games.get_column("game_type").cast(pl.String).to_list()
    return [1.0 if kind == _REGULAR_SEASON else PLAYOFF_MULTIPLIER for kind in game_types]


@dataclass
class _RatingBook:
    """Every team's current rating and the season of its last game."""

    first_season: int
    initial_ratings: Mapping[str, float]
    ratings: dict[str, float]
    seasons: dict[str, int]

    def pre_game(self, team: str, season: int) -> float:
        """Return a team's rating going into a game of ``season``, starting or reverting it."""
        last = self.seasons.get(team)
        if last is None:
            default = MEAN_RATING if season == self.first_season else EXPANSION_RATING
            self.ratings[team] = float(self.initial_ratings.get(team, default))
        elif last < season:
            rating = self.ratings[team]
            self.ratings[team] = MEAN_RATING * OFFSEASON_REVERSION + rating * (
                1.0 - OFFSEASON_REVERSION
            )
        self.seasons[team] = season
        return self.ratings[team]


def compute_elo_ratings(
    games: pl.DataFrame,
    *,
    settings: EloSettings = FRAMEWORK,
    initial_ratings: Mapping[str, float] | None = None,
    qb_adjustments: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Rate every game in order and return the pre-game ratings and probabilities.

    Args:
        games: One row per game with this repository's schedule columns: ``game_id``,
            ``season``, ``week``, ``game_type``, ``home_abbr``, ``away_abbr``, ``home_score``,
            ``away_score`` (null for an unplayed game), ``neutral`` (1 at a neutral site),
            ``stadium_id`` when travel is on and, when present, ``date``. Every game of a
            season belongs in it, unplayed ones too: byes and home stadiums come from the
            whole schedule.
        settings: The components switched on; the framework alone by default.
        initial_ratings: A team's rating at its first game in the frame, by team code; a team
            without one starts at 1505 in the frame's first season and at 1300 later.
        qb_adjustments: ``game_id``, ``home_qb_adj`` and ``away_qb_adj`` in Elo points, read
            when the quarterback switch is on (and required then); a game missing from it has
            no adjustment.

    Returns:
        One row per game in rating order: the ids and teams, each side's pre-game rating
        (``home_rating``, ``away_rating``), each side's quarterback adjustment
        (``home_qb_adj``, ``away_qb_adj``), the home side's edge in Elo points
        (``home_edge``), the pre-game difference (``elo_diff``), the home win probability
        (``home_win_prob``) and each side's rating after the game (``home_rating_post``,
        ``away_rating_post``, null for an unplayed game).

    Raises:
        ValueError: When a required column is missing, a game id repeats, or the
            quarterback adjustments and the quarterback switch disagree.

    """
    _require_columns(games, settings)
    ordered = _ordered(games)
    if ordered.height == 0:
        return pl.DataFrame(schema=_RESULT_SCHEMA)
    seasons = ordered.get_column("season").cast(pl.Int64).to_list()
    homes = ordered.get_column("home_abbr").cast(pl.String).to_list()
    aways = ordered.get_column("away_abbr").cast(pl.String).to_list()
    home_scores = ordered.get_column("home_score").cast(pl.Float64).to_list()
    away_scores = ordered.get_column("away_score").cast(pl.Float64).to_list()
    edges = _home_edges(ordered, settings)
    multipliers = _playoff_multipliers(ordered, settings)
    home_qbs, away_qbs = _quarterback_adjustments(ordered, settings, qb_adjustments)
    book = _RatingBook(min(seasons), initial_ratings or {}, {}, {})

    columns: dict[str, list[float | None]] = {
        name: []
        for name in (
            "home_rating",
            "away_rating",
            "elo_diff",
            "home_win_prob",
            "home_rating_post",
            "away_rating_post",
        )
    }
    for season, home, away, home_score, away_score, edge, boost, home_qb, away_qb in zip(
        seasons,
        homes,
        aways,
        home_scores,
        away_scores,
        edges,
        multipliers,
        home_qbs,
        away_qbs,
        strict=True,
    ):
        home_rating = book.pre_game(home, season)
        away_rating = book.pre_game(away, season)
        diff = ((home_rating + home_qb) - (away_rating + away_qb) + edge) * boost
        p = win_probability(diff)
        home_post: float | None = None
        away_post: float | None = None
        if home_score is not None and away_score is not None:
            multiplier = margin_multiplier(home_score, away_score, diff)
            shift = (K_FACTOR * multiplier) * (_result(home_score, away_score) - p)
            home_post = home_rating + shift
            away_post = away_rating - shift
            book.ratings[home] = home_post
            book.ratings[away] = away_post
        columns["home_rating"].append(home_rating)
        columns["away_rating"].append(away_rating)
        columns["elo_diff"].append(diff)
        columns["home_win_prob"].append(p)
        columns["home_rating_post"].append(home_post)
        columns["away_rating_post"].append(away_post)

    return pl.DataFrame(
        {
            "game_id": ordered.get_column("game_id").cast(pl.String),
            "season": ordered.get_column("season").cast(pl.Int64),
            "week": ordered.get_column("week").cast(pl.Int64),
            "home_abbr": homes,
            "away_abbr": aways,
            "home_rating": columns["home_rating"],
            "away_rating": columns["away_rating"],
            "home_qb_adj": home_qbs,
            "away_qb_adj": away_qbs,
            "home_edge": edges,
            "elo_diff": columns["elo_diff"],
            "home_win_prob": columns["home_win_prob"],
            "home_rating_post": columns["home_rating_post"],
            "away_rating_post": columns["away_rating_post"],
        },
        schema=_RESULT_SCHEMA,
    )
