"""A spread-to-moneyline map fitted to the market's own prices.

Many games have a spread but no moneyline (every nflverse game before 2006, and every nfelo
opener before 2024). The fixed conversion, ``game_utils.spread_to_moneyline``, prices them
with a normal curve at ``SCORE_DIFF_STD_DEV`` and a flat 5% vig, which smooths over the key
numbers: the market moves its moneylines far more crossing 3 or 7 points than crossing 5.

This map is fitted to games that carry a real spread and both real moneylines. For each
absolute spread (half points), it takes the mean implied probability, vig included, of the
favorite's moneyline and of the underdog's, then makes the favorite's non-decreasing and the
underdog's non-increasing in the spread with a weighted pool-adjacent-violators pass. Between
fitted spreads it interpolates linearly; past the largest it holds the last value. A pick'em
prices both sides at the mean of the two. The fit reads prices only, never a score, so it
carries no outcome. It is fitted per season on earlier seasons only (``fit_season_maps``), so
a season's derived moneylines use no later season's prices.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.game_utils import spread_to_moneyline

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_PAIR_COLUMNS: tuple[str, ...] = ("season", "home_spread", "home_moneyline", "away_moneyline")
_PICK_EM = 0.0


def implied_probability(moneyline: float) -> float:
    """Return the win probability a moneyline implies, vig included."""
    if moneyline < 0:
        return -moneyline / (-moneyline + 100.0)
    return 100.0 / (moneyline + 100.0)


def probability_to_moneyline(probability: float) -> int:
    """Return the American moneyline of a win probability (``+100`` at even odds)."""
    if probability > constants.EVEN_ODDS_PROBABILITY:
        return round(-100.0 * probability / (1.0 - probability))
    return round(100.0 * (1.0 - probability) / probability)


def _interpolate(x: float, xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    """Linear interpolation on ascending ``xs``, holding the end values outside them."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    right = bisect.bisect_right(xs, x)
    left = right - 1
    share = (x - xs[left]) / (xs[right] - xs[left])
    return ys[left] + share * (ys[right] - ys[left])


@dataclass(frozen=True)
class SpreadMoneylineMap:
    """The fitted prices of the favorite and the underdog at each absolute spread.

    Attributes:
        spreads: Absolute spreads, ascending, starting at the pick'em ``0``.
        favorite_prob: The favorite's implied probability (vig included) at each spread.
        underdog_prob: The underdog's implied probability (vig included) at each spread.
        games: Games the map was fitted on.
        seasons: Seasons those games came from.

    """

    spreads: tuple[float, ...]
    favorite_prob: tuple[float, ...]
    underdog_prob: tuple[float, ...]
    games: int
    seasons: tuple[int, ...]

    def implied_probabilities(self, home_spread: float) -> tuple[float, float]:
        """Return the home and away implied probabilities for a home spread."""
        distance = abs(home_spread)
        favorite = _interpolate(distance, self.spreads, self.favorite_prob)
        underdog = _interpolate(distance, self.spreads, self.underdog_prob)
        if home_spread < _PICK_EM:
            return favorite, underdog
        if home_spread > _PICK_EM:
            return underdog, favorite
        even = (favorite + underdog) / 2.0
        return even, even

    def moneylines(self, home_spread: float) -> tuple[int, int]:
        """Return the home and away moneylines for a home spread."""
        home, away = self.implied_probabilities(home_spread)
        return probability_to_moneyline(home), probability_to_moneyline(away)

    def to_dict(self) -> dict[str, object]:
        """Return the map as a JSON-ready record."""
        return {
            "games": self.games,
            "seasons": list(self.seasons),
            "spreads": list(self.spreads),
            "favorite_prob": list(self.favorite_prob),
            "underdog_prob": list(self.underdog_prob),
        }


def price_pairs(schedule: pl.DataFrame) -> pl.DataFrame:
    """Return the games with a real spread and both real moneylines, prices only.

    Args:
        schedule: Schedule rows before any moneyline is derived.

    Returns:
        ``season``, ``home_spread``, ``home_moneyline`` and ``away_moneyline`` of those games.

    """
    return schedule.select(_PAIR_COLUMNS).drop_nulls()


def _pool_adjacent_violators(
    values: list[float], weights: list[float], *, increasing: bool
) -> list[float]:
    """Return the weighted least-squares monotone fit of ``values`` (in their order)."""
    sign = 1.0 if increasing else -1.0
    blocks: list[list[float]] = []  # [weighted mean, weight, length]
    for value, weight in zip(values, weights, strict=True):
        blocks.append([sign * value, weight, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            last = blocks.pop()
            prior = blocks[-1]
            total = prior[1] + last[1]
            prior[0] = (prior[0] * prior[1] + last[0] * last[1]) / total
            prior[1] = total
            prior[2] += last[2]
    fitted: list[float] = []
    for mean, _weight, length in blocks:
        fitted.extend([sign * mean] * int(length))
    return fitted


def fit_spread_moneyline_map(
    pairs: pl.DataFrame, *, min_games: int | None = None
) -> SpreadMoneylineMap | None:
    """Fit the map to price pairs, or return ``None`` with fewer than ``min_games`` games.

    Args:
        pairs: Rows of ``price_pairs``.
        min_games: Minimum games; defaults to ``constants.MONEYLINE_MAP_MIN_GAMES``.

    Returns:
        The fitted map, or ``None`` when there are too few games.

    """
    minimum = constants.MONEYLINE_MAP_MIN_GAMES if min_games is None else min_games
    pairs = price_pairs(pairs)
    if pairs.height == 0 or pairs.height < minimum:
        return None
    home = pl.col("home_moneyline").cast(pl.Float64)
    away = pl.col("away_moneyline").cast(pl.Float64)
    home_prob = pl.when(home < 0).then(-home / (-home + 100.0)).otherwise(100.0 / (home + 100.0))
    away_prob = pl.when(away < 0).then(-away / (-away + 100.0)).otherwise(100.0 / (away + 100.0))
    spread = pl.col("home_spread")
    even = (home_prob + away_prob) / 2.0
    bins = (
        pairs.select(
            ((spread.abs() * 2.0).round(0) / 2.0).alias("distance"),
            pl.when(spread < 0)
            .then(home_prob)
            .when(spread > 0)
            .then(away_prob)
            .otherwise(even)
            .alias("favorite"),
            pl.when(spread < 0)
            .then(away_prob)
            .when(spread > 0)
            .then(home_prob)
            .otherwise(even)
            .alias("underdog"),
        )
        .group_by("distance")
        .agg(pl.len().alias("games"), pl.col("favorite").mean(), pl.col("underdog").mean())
        .sort("distance")
    )
    distances = bins.get_column("distance").to_list()
    weights = [float(games) for games in bins.get_column("games").to_list()]
    favorite = bins.get_column("favorite").to_list()
    underdog = bins.get_column("underdog").to_list()
    if distances[0] != _PICK_EM:
        # No pick'em in the fit: anchor zero at the closest spread's mean price for both sides.
        even_price = (favorite[0] + underdog[0]) / 2.0
        distances.insert(0, _PICK_EM)
        weights.insert(0, 1.0)
        favorite.insert(0, even_price)
        underdog.insert(0, even_price)
    seasons = tuple(sorted(set(pairs.get_column("season").to_list())))
    return SpreadMoneylineMap(
        spreads=tuple(float(distance) for distance in distances),
        favorite_prob=tuple(_pool_adjacent_violators(favorite, weights, increasing=True)),
        underdog_prob=tuple(_pool_adjacent_violators(underdog, weights, increasing=False)),
        games=pairs.height,
        seasons=seasons,
    )


def fit_season_maps(
    history: pl.DataFrame, seasons: Iterable[int], *, min_games: int | None = None
) -> dict[int, SpreadMoneylineMap | None]:
    """Fit one map per season, each on the price pairs of strictly earlier seasons.

    Args:
        history: Schedule rows (any seasons) before any moneyline is derived.
        seasons: Seasons that need a map.
        min_games: Minimum games per fit; see ``fit_spread_moneyline_map``.

    Returns:
        Each season's map, or ``None`` where the earlier seasons have too few games.

    """
    pairs = price_pairs(history)
    return {
        season: fit_spread_moneyline_map(
            pairs.filter(pl.col("season") < season), min_games=min_games
        )
        for season in sorted(set(seasons))
    }


def _derived_pair(
    season: int,
    home_spread: float,
    away_spread: float | None,
    maps: Mapping[int, SpreadMoneylineMap | None],
) -> tuple[int | None, int | None]:
    fitted = maps.get(season)
    if fitted is not None:
        return fitted.moneylines(home_spread)
    away = None if away_spread is None else spread_to_moneyline(away_spread)
    return spread_to_moneyline(home_spread), away


def fill_moneylines_from_maps(
    df: pl.DataFrame, maps: Mapping[int, SpreadMoneylineMap | None]
) -> pl.DataFrame:
    """Fill each missing moneyline from its game's spread with the season's map.

    A season without a map (missing from ``maps`` or ``None``) keeps the fixed conversion,
    applied per side as ``game_utils.fill_missing_moneylines`` does. A moneyline that is
    present is kept; a game without a home spread keeps its nulls.

    Args:
        df: Game rows with ``season``, the spreads and the moneylines.
        maps: Each season's fitted map, from ``fit_season_maps``.

    Returns:
        The rows with the moneylines filled, as whole numbers.

    """
    if "home_spread" not in df.columns:
        return df
    for column in ("home_moneyline", "away_moneyline", "away_spread"):
        if column not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias(column))
    cache: dict[tuple[int, float, float | None], tuple[int | None, int | None]] = {}

    def derive(game: dict[str, object]) -> dict[str, int | None]:
        season = game["season"]
        home_spread = game["home_spread"]
        away_spread = game["away_spread"]
        if not isinstance(season, int) or not isinstance(home_spread, float):
            return {"home": None, "away": None}
        away = float(away_spread) if isinstance(away_spread, (int, float)) else None
        key = (season, home_spread, away)
        if key not in cache:
            cache[key] = _derived_pair(season, home_spread, away, maps)
        home_ml, away_ml = cache[key]
        return {"home": home_ml, "away": away_ml}

    derived = pl.struct(
        pl.col("season").cast(pl.Int64),
        pl.col("home_spread").cast(pl.Float64),
        pl.col("away_spread").cast(pl.Float64),
    ).map_elements(derive, return_dtype=pl.Struct({"home": pl.Int64, "away": pl.Int64}))
    return (
        df.with_columns(derived.alias("__derived_moneylines"))
        .with_columns(
            pl.coalesce(
                pl.col("home_moneyline").cast(pl.Int64),
                pl.col("__derived_moneylines").struct.field("home"),
            ).alias("home_moneyline"),
            pl.coalesce(
                pl.col("away_moneyline").cast(pl.Int64),
                pl.col("__derived_moneylines").struct.field("away"),
            ).alias("away_moneyline"),
        )
        .drop("__derived_moneylines")
    )
