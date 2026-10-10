"""This repository's own Elo ratings, computed from game results with FiveThirtyEight's method.

FiveThirtyEight (538) published its NFL Elo method in "How Our NFL Predictions Work" and the
code behind its classic ratings in ``fivethirtyeight/nfl-elo-game`` (``forecast.py``, MIT
License). This module reproduces that public framework exactly, and adds 538's later additions
as separate switches of :class:`EloSettings`: the rest, travel and rolling home-field
adjustments of ``elo_game_context``, the playoff boost (538's 2019 version multiplies a
playoff game's whole difference by 1.2 before the probability, because favorites outplay
underdogs by more in the playoffs), the quarterback adjustment (538's 2019 VALUE model,
``elo_quarterbacks``, or any other source of per-game adjustments) and the preseason pull toward
betting win totals. With the boost on, the boosted difference is also the one the
margin-of-victory multiplier damps. A quarterback adjustment joins its side's rating for the game
only: the difference, and so the probability and the update, include it, but the update lands on
the team's own rating.

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

The win-total pull
------------------
538 sets a preseason rating as one third the regressed Elo and two thirds a rating built from
the sportsbooks' season win totals. Here the win-total rating is nfelosrs's ``wt_rating_elo``
(``elo_win_totals``, 2003 on), and its weight is fitted rather than taken from 538: for each
season, the weight on a grid of 0 to 1 in steps of 0.05 whose fixed-weight run has the lowest
Brier score on the completed regular-season games of the earlier win-total seasons (ties to the
smaller weight). The first win-total season has nothing earlier and gets no pull, and a team
without a win total keeps the plain regression. The win totals are market information, so a
setting without the switch refuses them.

Games are rated in season, week, date and ``game_id`` order. Every value for a game comes from
games strictly before it: a game's pre-game ratings never include its own result, a season's
pull weight is fitted on earlier seasons only, and a game without a final score (an upcoming
one) is rated but moves no rating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars import elo_game_context

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
# The candidate weights of the win-total pull, 0 to 1 in steps of 0.05.
WIN_TOTAL_WEIGHT_GRID: tuple[float, ...] = tuple(step / 20 for step in range(21))
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
        win_totals: Pull each preseason rating toward the team's win-total rating, with the
            weight fitted on earlier seasons. Market information: leave it off for a
            market-free rating.

    """

    rest: bool = False
    travel: bool = False
    playoff_boost: bool = False
    rolling_home_field: bool = False
    quarterback: bool = False
    win_totals: bool = False


FRAMEWORK = EloSettings()
# The components a forward selection kept on the 2007-2025 regular season, adding them in the
# order of the settings above and keeping each only when it improved the Brier score of the
# ratings' own probabilities with a paired game-bootstrap interval below zero; rest and the
# playoff boost were not kept. Market-free: no input built from betting lines or win totals.
MARKET_FREE_SETTINGS = EloSettings(travel=True, rolling_home_field=True, quarterback=True)
# The same plus the win-total pull, which the selection also kept: market information, for a
# rating that may carry it.
WIN_TOTAL_SETTINGS = EloSettings(
    travel=True, rolling_home_field=True, quarterback=True, win_totals=True
)


@dataclass(frozen=True, kw_only=True, eq=False)
class EloInputs:
    """What the ratings read besides the schedule; each input is tied to its switch.

    Attributes:
        initial_ratings: A team's rating at its first game in the frame, by team code; a team
            without one starts at 1505 in the frame's first season and at 1300 later.
        qb_adjustments: ``game_id``, ``home_qb_adj`` and ``away_qb_adj`` in Elo points
            (``elo_quarterbacks.value_qb_adjustments`` or any other source), required by the
            quarterback switch and refused without it; a game missing from it has none.
        win_totals: ``season``, ``team_abbr`` and ``win_total_elo``
            (``elo_win_totals.load_win_totals``), required by the win-total switch and refused
            without it, because they are market information.
        win_total_weights: The pull's weight by season, read with the win-total switch; fitted
            with :func:`fit_win_total_weights` when not given.

    """

    initial_ratings: Mapping[str, float] | None = None
    qb_adjustments: pl.DataFrame | None = None
    win_totals: pl.DataFrame | None = None
    win_total_weights: Mapping[int, float] | None = None


NO_INPUTS = EloInputs()


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


def _check_win_totals(settings: EloSettings, win_totals: pl.DataFrame | None) -> None:
    """Raise when the win-total switch and the win totals disagree."""
    if settings.win_totals and win_totals is None:
        msg = "the win-total switch is on but no win_totals were given"
        raise ValueError(msg)
    if not settings.win_totals and win_totals is not None:
        msg = (
            "win_totals were given without the win-total switch: they are market information, "
            "and a market-free setting never reads them"
        )
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


@dataclass(frozen=True)
class _Games:
    """The ordered games as plain lists, with every per-game input the rating loop reads."""

    frame: pl.DataFrame
    seasons: list[int]
    homes: list[str]
    aways: list[str]
    home_scores: list[float | None]
    away_scores: list[float | None]
    regular: list[bool]
    edges: list[float]
    multipliers: list[float]
    home_qbs: list[float]
    away_qbs: list[float]


def _prepare(
    ordered: pl.DataFrame, settings: EloSettings, qb_adjustments: pl.DataFrame | None
) -> _Games:
    """Return the ordered games with their home edges, playoff boosts and adjustments."""
    home_qbs, away_qbs = _quarterback_adjustments(ordered, settings, qb_adjustments)
    return _Games(
        frame=ordered,
        seasons=ordered.get_column("season").cast(pl.Int64).to_list(),
        homes=ordered.get_column("home_abbr").cast(pl.String).to_list(),
        aways=ordered.get_column("away_abbr").cast(pl.String).to_list(),
        home_scores=ordered.get_column("home_score").cast(pl.Float64).to_list(),
        away_scores=ordered.get_column("away_score").cast(pl.Float64).to_list(),
        regular=(ordered.get_column("game_type").cast(pl.String) == _REGULAR_SEASON).to_list(),
        edges=_home_edges(ordered, settings),
        multipliers=_playoff_multipliers(ordered, settings),
        home_qbs=home_qbs,
        away_qbs=away_qbs,
    )


@dataclass(frozen=True)
class _Preseason:
    """How a team's preseason rating is set: seeds, win-total ratings and pull weights."""

    initial_ratings: Mapping[str, float]
    targets: Mapping[tuple[str, int], float]
    weights: Mapping[int, float]


@dataclass
class _RatingBook:
    """Every team's current rating and the season of its last game."""

    first_season: int
    preseason: _Preseason
    ratings: dict[str, float] = field(default_factory=dict)
    seasons: dict[str, int] = field(default_factory=dict)

    def pre_game(self, team: str, season: int) -> float:
        """Return a team's rating going into a game of ``season``, starting or reverting it."""
        last = self.seasons.get(team)
        if last is None:
            default = MEAN_RATING if season == self.first_season else EXPANSION_RATING
            rating = float(self.preseason.initial_ratings.get(team, default))
        elif last < season:
            rating = MEAN_RATING * OFFSEASON_REVERSION + self.ratings[team] * (
                1.0 - OFFSEASON_REVERSION
            )
        else:
            return self.ratings[team]
        self.ratings[team] = self._pulled(team, season, rating)
        self.seasons[team] = season
        return self.ratings[team]

    def _pulled(self, team: str, season: int, rating: float) -> float:
        """Return a preseason rating pulled toward the team's win-total rating, if any."""
        weight = self.preseason.weights.get(season, 0.0)
        target = self.preseason.targets.get((team, season))
        if target is None or weight == 0.0:
            return rating
        return (1.0 - weight) * rating + weight * target


@dataclass(frozen=True)
class _Rated:
    """The rating loop's per-game outputs, in game order."""

    home_ratings: list[float]
    away_ratings: list[float]
    diffs: list[float]
    probabilities: list[float]
    home_posts: list[float | None]
    away_posts: list[float | None]


def _rate(games: _Games, preseason: _Preseason) -> _Rated:
    """Run the ratings over every game in order."""
    book = _RatingBook(min(games.seasons), preseason)
    rated = _Rated([], [], [], [], [], [])
    for index, season in enumerate(games.seasons):
        home, away = games.homes[index], games.aways[index]
        home_rating = book.pre_game(home, season)
        away_rating = book.pre_game(away, season)
        diff = (
            (home_rating + games.home_qbs[index])
            - (away_rating + games.away_qbs[index])
            + games.edges[index]
        ) * games.multipliers[index]
        p = win_probability(diff)
        home_score, away_score = games.home_scores[index], games.away_scores[index]
        home_post: float | None = None
        away_post: float | None = None
        if home_score is not None and away_score is not None:
            multiplier = margin_multiplier(home_score, away_score, diff)
            shift = (K_FACTOR * multiplier) * (_result(home_score, away_score) - p)
            home_post = home_rating + shift
            away_post = away_rating - shift
            book.ratings[home] = home_post
            book.ratings[away] = away_post
        rated.home_ratings.append(home_rating)
        rated.away_ratings.append(away_rating)
        rated.diffs.append(diff)
        rated.probabilities.append(p)
        rated.home_posts.append(home_post)
        rated.away_posts.append(away_post)
    return rated


def _win_total_targets(win_totals: pl.DataFrame) -> dict[tuple[str, int], float]:
    """Return each team's win-total rating by (team, season)."""
    return {
        (team, season): rating
        for season, team, rating in win_totals.select(
            pl.col("season").cast(pl.Int64),
            pl.col("team_abbr").cast(pl.String),
            pl.col("win_total_elo").cast(pl.Float64),
        )
        .drop_nulls()
        .iter_rows()
    }


def _season_squared_errors(games: _Games, rated: _Rated) -> dict[int, tuple[float, int]]:
    """Return each season's Brier sum and count over its completed regular-season games.

    The outcome is a home win (a tie counts as not one, as in the walk-forward).
    """
    totals: dict[int, tuple[float, int]] = {}
    for index, season in enumerate(games.seasons):
        home_score, away_score = games.home_scores[index], games.away_scores[index]
        if home_score is None or away_score is None or not games.regular[index]:
            continue
        outcome = 1.0 if home_score > away_score else 0.0
        error = (outcome - rated.probabilities[index]) ** 2
        total, count = totals.get(season, (0.0, 0))
        totals[season] = (total + error, count + 1)
    return totals


def _fit_weights(
    games: _Games,
    initial_ratings: Mapping[str, float],
    targets: Mapping[tuple[str, int], float],
    grid: Sequence[float],
) -> dict[int, float]:
    """Return each win-total season's pull weight, fitted on the earlier win-total seasons."""
    win_seasons = sorted({season for _, season in targets} & set(games.seasons))
    errors = {
        weight: _season_squared_errors(
            games,
            _rate(games, _Preseason(initial_ratings, targets, dict.fromkeys(win_seasons, weight))),
        )
        for weight in grid
    }

    def earlier_brier(weight: float, season: int) -> float:
        sums = [errors[weight].get(year, (0.0, 0)) for year in win_seasons if year < season]
        count = sum(n for _, n in sums)
        return sum(total for total, _ in sums) / count if count else math.inf

    weights: dict[int, float] = {}
    for season in win_seasons:
        if season == win_seasons[0]:
            weights[season] = 0.0
            continue
        weights[season] = min(grid, key=lambda weight: (earlier_brier(weight, season), weight))
    return weights


def fit_win_total_weights(
    games: pl.DataFrame,
    *,
    settings: EloSettings = FRAMEWORK,
    inputs: EloInputs = NO_INPUTS,
    grid: Sequence[float] = WIN_TOTAL_WEIGHT_GRID,
) -> dict[int, float]:
    """Return the win-total pull's weight for every season the win totals cover.

    Args:
        games: The schedule, as for :func:`compute_elo_ratings`.
        settings: The other components, which every candidate run uses; the win-total switch
            itself is not read.
        inputs: The seeds, quarterback adjustments and, required here, the win totals.
        grid: The candidate weights.

    Returns:
        Weight by season: for each season, the candidate whose run with that weight in every
        season has the lowest Brier score over the completed regular-season games of the
        earlier win-total seasons (ties to the smaller weight); 0 for the first one.

    Raises:
        ValueError: When the win totals are missing, or as :func:`compute_elo_ratings` does.

    """
    if inputs.win_totals is None:
        msg = "fitting the win-total weights needs inputs.win_totals"
        raise ValueError(msg)
    _require_columns(games, settings)
    prepared = _prepare(_ordered(games), settings, inputs.qb_adjustments)
    return _fit_weights(
        prepared, inputs.initial_ratings or {}, _win_total_targets(inputs.win_totals), grid
    )


def compute_elo_ratings(
    games: pl.DataFrame,
    *,
    settings: EloSettings = FRAMEWORK,
    inputs: EloInputs = NO_INPUTS,
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
        inputs: Seeds and the inputs the quarterback and win-total switches read (see
            :class:`EloInputs`).

    Returns:
        One row per game in rating order: the ids and teams, each side's pre-game rating
        (``home_rating``, ``away_rating``), each side's quarterback adjustment
        (``home_qb_adj``, ``away_qb_adj``), the home side's edge in Elo points
        (``home_edge``), the pre-game difference (``elo_diff``), the home win probability
        (``home_win_prob``) and each side's rating after the game (``home_rating_post``,
        ``away_rating_post``, null for an unplayed game).

    Raises:
        ValueError: When a required column is missing, a game id repeats, or an input and its
            switch disagree (quarterback adjustments, win totals).

    """
    _require_columns(games, settings)
    _check_win_totals(settings, inputs.win_totals)
    ordered = _ordered(games)
    if ordered.height == 0:
        return pl.DataFrame(schema=_RESULT_SCHEMA)
    prepared = _prepare(ordered, settings, inputs.qb_adjustments)
    seeds = inputs.initial_ratings or {}
    targets: dict[tuple[str, int], float] = {}
    weights: Mapping[int, float] = {}
    if settings.win_totals and inputs.win_totals is not None:
        targets = _win_total_targets(inputs.win_totals)
        weights = (
            inputs.win_total_weights
            if inputs.win_total_weights is not None
            else _fit_weights(prepared, seeds, targets, WIN_TOTAL_WEIGHT_GRID)
        )
    rated = _rate(prepared, _Preseason(seeds, targets, weights))
    return pl.DataFrame(
        {
            "game_id": ordered.get_column("game_id").cast(pl.String),
            "season": ordered.get_column("season").cast(pl.Int64),
            "week": ordered.get_column("week").cast(pl.Int64),
            "home_abbr": prepared.homes,
            "away_abbr": prepared.aways,
            "home_rating": rated.home_ratings,
            "away_rating": rated.away_ratings,
            "home_qb_adj": prepared.home_qbs,
            "away_qb_adj": prepared.away_qbs,
            "home_edge": prepared.edges,
            "elo_diff": rated.diffs,
            "home_win_prob": rated.probabilities,
            "home_rating_post": rated.home_posts,
            "away_rating_post": rated.away_posts,
        },
        schema=_RESULT_SCHEMA,
    )


def build_own_elo_features(
    schedule: pl.DataFrame,
    *,
    settings: EloSettings = MARKET_FREE_SETTINGS,
    inputs: EloInputs = NO_INPUTS,
) -> pl.DataFrame:
    """Return the own Elo feature group for every game of this repository's schedule frame.

    Args:
        schedule: The schedule, as for :func:`compute_elo_ratings`: every game, upcoming ones
            included, with the columns that function reads.
        settings: The components switched on; by default the market-free preset, whose
            quarterback adjustment needs ``inputs.qb_adjustments``
            (``elo_quarterbacks.value_qb_adjustments``).
        inputs: Seeds and the inputs the quarterback and win-total switches read.

    Returns:
        One row per schedule row, in its order: ``game_id`` and ``constants.OWN_ELO_COLUMNS``,
        that is each side's pre-game rating with its quarterback adjustment
        (``away_own_elo``, ``home_own_elo``), away minus home (``own_elo_diff``) and the home
        win probability (``own_elo_home_win_prob``), all from games strictly before.

    """
    rated = compute_elo_ratings(schedule, settings=settings, inputs=inputs)
    teams = rated.select(
        "game_id",
        (pl.col("away_rating") + pl.col("away_qb_adj")).alias("away_own_elo"),
        (pl.col("home_rating") + pl.col("home_qb_adj")).alias("home_own_elo"),
        pl.col("home_win_prob").alias("own_elo_home_win_prob"),
    ).with_columns((pl.col("away_own_elo") - pl.col("home_own_elo")).alias("own_elo_diff"))
    return schedule.select(pl.col("game_id").cast(pl.String)).join(
        teams.select("game_id", *constants.OWN_ELO_COLUMNS),
        on="game_id",
        how="left",
        maintain_order="left",
    )
