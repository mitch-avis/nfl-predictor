"""The tiebreaker card: entries for a pool's tiebreak games and the week's extreme team scores.

Yahoo's Pick'em breaks a weekly tie by the total points, then the home team's and the road team's
points, of a first and then a second designated tiebreak game, and then by the score of the
week's highest-scoring and lowest-scoring team. Yahoo designates the games each week, so the card
lists every game and suggests the latest Sunday and Monday kickoffs (the Sunday and Monday night
games) as the two.

Entries are whole points from the unrounded predictions: the total rounds half up, and the home
side takes half of that total plus half the predicted margin, rounded, so the entry keeps the
rounded total exactly. Each team's score is treated as normal around its predicted score, with a
spread from the quantile models' p10 to p90 bands (the defaults the betting page uses when a file
has none) and the margin and total errors taken as independent; the week's highest and lowest
scores are the extremes of those normals, taken as independent across teams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from nfl_predictor.api.pool.kickoff import kickoff_eastern
from nfl_predictor.api.readers import market

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from datetime import datetime

    import polars as pl

NIGHT_HOUR = 19
"""A kickoff at 7 p.m. Eastern or later is a night game."""
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
SUNDAY, MONDAY = 6, 0
TIEBREAK_GAMES = 2
BISECTION_STEPS = 100
SEARCH_SIGMAS = 10.0


@dataclass(frozen=True, kw_only=True)
class TiebreakGame:
    """One game with its whole-point tiebreak entry and the p10 to p90 bands behind it.

    The entry and band fields are ``None`` for a game without a predicted margin and total.
    """

    game_id: str
    away_abbr: str
    home_abbr: str
    kickoff: str | None
    slot: str | None
    total_points: int | None = None
    home_points: int | None = None
    away_points: int | None = None
    total_raw: float | None = None
    home_raw: float | None = None
    away_raw: float | None = None
    total_p10: float | None = None
    total_p90: float | None = None
    home_p10: float | None = None
    home_p90: float | None = None
    away_p10: float | None = None
    away_p90: float | None = None


@dataclass(frozen=True, kw_only=True)
class ScoreEstimate:
    """The week's highest (or lowest) team score: the entry, its median and its p10 to p90."""

    points: int | None
    median: float | None
    p10: float | None
    p90: float | None
    projected_team: str | None
    projected_points: float | None


@dataclass(frozen=True, kw_only=True)
class TiebreakCard:
    """Every game's entry, the suggested tiebreak games and the week's extreme scores."""

    games: list[TiebreakGame]
    default_game_ids: list[str]
    highest: ScoreEstimate
    lowest: ScoreEstimate


@dataclass(frozen=True)
class _TeamScore:
    """One team's predicted score as a normal distribution."""

    team: str
    mean: float
    sigma: float


def _first_number(row: Mapping[str, object], *keys: str) -> float | None:
    """Return the first finite number among ``keys``."""
    for key in keys:
        value = row.get(key)
        if market.is_number(value):
            return float(value)
    return None


def _round_half_up(value: float) -> int:
    """Round to the nearest whole point, halves up."""
    return math.floor(value + 0.5)


def _band(mean: float, sigma: float) -> tuple[float, float]:
    """Return the p10 to p90 band of a normal score, floored at zero points."""
    spread = market.NORMAL_Z_P90 * sigma
    return max(0.0, mean - spread), max(0.0, mean + spread)


def _slot(kickoff: datetime | None) -> str | None:
    """Name a night game by its day (``"Monday night"``); other kickoffs have no slot."""
    if kickoff is None or kickoff.hour < NIGHT_HOUR:
        return None
    return f"{WEEKDAYS[kickoff.weekday()]} night"


def _total_band(row: Mapping[str, object], total: float, sigma_total: float) -> tuple[float, float]:
    """Return the total's own p10 to p90 when the file has them, else a normal band."""
    p10, p90 = row.get("predicted_total_p10"), row.get("predicted_total_p90")
    if market.is_number(p10) and market.is_number(p90) and p90 > p10:
        return float(p10), float(p90)
    return _band(total, sigma_total)


def _game(row: Mapping[str, object]) -> tuple[TiebreakGame, datetime | None, list[_TeamScore]]:
    """Build one game's entry, its kickoff and its two teams' score distributions."""
    away, home = str(row.get("away_abbr")), str(row.get("home_abbr"))
    kickoff = kickoff_eastern(row)
    game = TiebreakGame(
        game_id=str(row.get("game_id")),
        away_abbr=away,
        home_abbr=home,
        kickoff=kickoff.isoformat() if kickoff else None,
        slot=_slot(kickoff),
    )
    margin = _first_number(row, "predicted_margin_raw", "predicted_margin")
    total = _first_number(row, "predicted_total_raw", "predicted_total")
    if margin is None or total is None:
        return game, kickoff, []
    sigma_total = market.sigma_from_quantiles(
        row.get("predicted_total_p10"), row.get("predicted_total_p90"), market.DEFAULT_SIGMA_TOTAL
    )
    sigma_margin = market.sigma_from_quantiles(
        row.get("predicted_margin_p10"),
        row.get("predicted_margin_p90"),
        market.DEFAULT_SIGMA_MARGIN,
    )
    sigma_team = math.hypot(sigma_total, sigma_margin) / 2
    home_mean, away_mean = (total + margin) / 2, (total - margin) / 2
    total_points = _round_half_up(total)
    home_points = _round_half_up((total_points + margin) / 2)
    total_p10, total_p90 = _total_band(row, total, sigma_total)
    home_p10, home_p90 = _band(home_mean, sigma_team)
    away_p10, away_p90 = _band(away_mean, sigma_team)
    game = replace(
        game,
        total_points=total_points,
        home_points=home_points,
        away_points=total_points - home_points,
        total_raw=total,
        home_raw=home_mean,
        away_raw=away_mean,
        total_p10=total_p10,
        total_p90=total_p90,
        home_p10=home_p10,
        home_p90=home_p90,
        away_p10=away_p10,
        away_p90=away_p90,
    )
    teams = [_TeamScore(home, home_mean, sigma_team), _TeamScore(away, away_mean, sigma_team)]
    return game, kickoff, teams


def _quantile(cdf: Callable[[float], float], q: float, low: float, high: float) -> float:
    """Return the ``q`` quantile of a continuous distribution by bisection on its CDF."""
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2
        if cdf(middle) < q:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _extreme(teams: list[_TeamScore], *, highest: bool) -> ScoreEstimate:
    """Estimate the week's highest (or lowest) team score from the teams' normals."""
    if not teams:
        return ScoreEstimate(
            points=None,
            median=None,
            p10=None,
            p90=None,
            projected_team=None,
            projected_points=None,
        )
    normal = market.NORMAL

    def below(x: float, team: _TeamScore) -> float:
        return normal.cdf((x - team.mean) / team.sigma)

    def cdf(x: float) -> float:
        if highest:
            return math.prod(below(x, team) for team in teams)
        return 1.0 - math.prod(1.0 - below(x, team) for team in teams)

    reach = SEARCH_SIGMAS * max(team.sigma for team in teams)
    low = min(team.mean for team in teams) - reach
    high = max(team.mean for team in teams) + reach
    p10, median, p90 = (max(0.0, _quantile(cdf, q, low, high)) for q in (0.1, 0.5, 0.9))
    projected = (max if highest else min)(teams, key=lambda team: team.mean)
    return ScoreEstimate(
        points=_round_half_up(median),
        median=median,
        p10=p10,
        p90=p90,
        projected_team=projected.team,
        projected_points=projected.mean,
    )


def _default_games(games: list[tuple[TiebreakGame, datetime | None]]) -> list[str]:
    """Suggest the latest Sunday and Monday kickoffs, filled with the latest others, in order."""
    timed = [(game, kickoff) for game, kickoff in games if kickoff is not None]
    picked: list[tuple[TiebreakGame, datetime]] = []
    for weekday in (SUNDAY, MONDAY):
        on_day = [item for item in timed if item[1].weekday() == weekday]
        if on_day:
            picked.append(on_day[-1])
    for item in reversed(timed):
        if len(picked) >= TIEBREAK_GAMES:
            break
        if item not in picked:
            picked.append(item)
    picked.sort(key=lambda item: (item[1], item[0].game_id))
    return [game.game_id for game, _kickoff in picked[:TIEBREAK_GAMES]]


def _kickoff_order(
    item: tuple[TiebreakGame, datetime | None, list[_TeamScore]],
) -> tuple[float, str]:
    """Sort key: kickoff (games without one last), then game id."""
    game, kickoff, _teams = item
    return (kickoff.timestamp() if kickoff else math.inf, game.game_id)


def build_tiebreakers(frame: pl.DataFrame) -> TiebreakCard:
    """Build the tiebreaker card for one week's predictions."""
    built = sorted((_game(row) for row in frame.to_dicts()), key=_kickoff_order)
    teams = [team for _game_entry, _kickoff, game_teams in built for team in game_teams]
    return TiebreakCard(
        games=[game for game, _kickoff, _teams in built],
        default_game_ids=_default_games([(game, kickoff) for game, kickoff, _teams in built]),
        highest=_extreme(teams, highest=True),
        lowest=_extreme(teams, highest=False),
    )
