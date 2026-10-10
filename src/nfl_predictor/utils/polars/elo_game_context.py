"""Game-level adjustments to the own Elo: byes, travel and a rolling home field.

FiveThirtyEight (538) added a rest adjustment, a travel-distance adjustment and a playoff boost
to its NFL Elo in 2019 (version 2.0 of "How Our NFL Predictions Work"), and replaced the fixed
home field with a rolling ten-season average in 2022 (version 2.2). This module computes the
three game-level ones from this repository's schedule frame as Elo points credited to the home
side (``home_edges``); the playoff boost multiplies the whole difference and lives in
``elo_ratings``. Each is a separate switch there.

Home field
    0 at a neutral site. Otherwise 538's classic 65 points, or 55 when travel is on (538's
    2019 split of the same edge into a base and a distance term), or the rolling estimate below.

Rest
    25 points for a team coming off a bye week: its previous game of the season, by the
    schedule's week numbers, was two or more weeks earlier. That covers a top seed that skips
    the wild-card round; a team's first game of a season is never off a bye.

Travel
    4 points per 1,000 miles of the visitor's trip, measured from each team's home stadium to
    the game's stadium, as ``0.004 * (away_miles - home_miles)`` so a home team playing away
    from its own stadium travels too. At a neutral site there is no travel term except in the
    Super Bowl, where both trips count (538's text, and its published 2019 and 2021 numbers).
    A team's home stadium for a season is the venue of most of its non-neutral home games that
    season; distances use ``constants.STADIUM_COORDINATES`` and the great-circle formula, and a
    game whose venue or home stadium has no coordinates gets no travel term.

Rolling home field
    25 Elo points per point of the mean home margin (538's points-to-Elo factor) over completed,
    non-neutral regular-season games in the ten seasons before the predicted week: from the
    same week ten seasons earlier up to the week before, so a game's own week never counts.
    With travel on, the mean travel term of those games is subtracted, so that home field plus
    the average trip still equals the observed margin. The frame's first season has no earlier
    season to average and uses the fixed value.
"""

from __future__ import annotations

import math

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.logger import log

# 538's home field: the classic fixed value, and the 2019 base that a travel term adds to.
CLASSIC_HOME_FIELD = 65.0
TRAVEL_BASE_HOME_FIELD = 55.0
TRAVEL_ELO_PER_MILE = 4.0 / 1000.0
BYE_ELO = 25.0
# 538: "divide EloDiff by 25 to get the spread", so a point of margin is worth 25 Elo points.
POINTS_TO_ELO = 25.0
ROLLING_HOME_FIELD_SEASONS = 10
EARTH_RADIUS_MILES = 3958.8
# A team whose previous game of the season was this many weeks back or more had a bye week.
BYE_WEEK_GAP = 2

_SUPER_BOWL = "SB"
_REGULAR_SEASON = "REG"
_SIDES = ("home", "away")


def great_circle_miles(origin: tuple[float, float], destination: tuple[float, float]) -> float:
    """Return the great-circle (haversine) distance in miles between two points.

    Args:
        origin: Latitude and longitude in degrees.
        destination: Latitude and longitude in degrees.

    Returns:
        The distance along the Earth's surface, with a mean radius of 3,958.8 miles.

    """
    lat1, lon1 = (math.radians(value) for value in origin)
    lat2, lon2 = (math.radians(value) for value in destination)
    half_chord = (
        math.sin((lat2 - lat1) / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_MILES * math.asin(math.sqrt(half_chord))


def _neutral() -> pl.Expr:
    """Return whether a game is at a neutral site (a null flag means it is not)."""
    return pl.col("neutral").fill_null(0) == 1


def _team_games(games: pl.DataFrame) -> pl.DataFrame:
    """Return one row per side of every game: game id, season, week, side and team."""
    return pl.concat(
        [
            games.select(
                "game_id",
                "season",
                "week",
                pl.lit(side).alias("side"),
                pl.col(f"{side}_abbr").alias("team_abbr"),
            )
            for side in _SIDES
        ]
    )


def _per_game(games: pl.DataFrame, team_values: pl.DataFrame, column: str) -> pl.DataFrame:
    """Return ``game_id`` with ``home_<column>`` and ``away_<column>``, in the order of games."""
    out = games.select("game_id")
    for side in _SIDES:
        values = team_values.filter(pl.col("side") == side).select(
            "game_id", pl.col(column).alias(f"{side}_{column}")
        )
        out = out.join(values, on="game_id", how="left", maintain_order="left")
    return out


def bye_weeks(games: pl.DataFrame) -> pl.DataFrame:
    """Return whether each side comes off a bye week.

    Args:
        games: Schedule rows with ``game_id``, ``season``, ``week``, ``home_abbr`` and
            ``away_abbr``, every game of the season included (unplayed ones too).

    Returns:
        ``game_id``, ``home_bye`` and ``away_bye``, in the order of ``games``: true when the
        team's previous game of the same season was two or more weeks earlier.

    """
    team_games = _team_games(games).sort("season", "team_abbr", "week", "game_id")
    previous_week = pl.col("week").shift(1).over("season", "team_abbr")
    flagged = team_games.with_columns(
        ((pl.col("week") - previous_week) >= BYE_WEEK_GAP).fill_null(value=False).alias("bye")
    )
    return _per_game(games, flagged, "bye")


def home_stadiums(games: pl.DataFrame) -> pl.DataFrame:
    """Return each team's home stadium for each season.

    Args:
        games: Schedule rows with ``season``, ``home_abbr``, ``neutral`` and ``stadium_id``.

    Returns:
        ``season``, ``team_abbr`` and ``stadium_id``: the venue of most of the team's
        non-neutral home games that season (ties go to the lower id). A team with no such game
        has no row.

    """
    home_games = games.filter(~_neutral() & pl.col("stadium_id").is_not_null())
    counts = home_games.group_by("season", "home_abbr", "stadium_id").len()
    return (
        counts.sort(
            ["season", "home_abbr", "len", "stadium_id"],
            descending=[False, False, True, False],
        )
        .unique(["season", "home_abbr"], keep="first", maintain_order=True)
        .select("season", pl.col("home_abbr").alias("team_abbr"), "stadium_id")
    )


def _stadium_miles(origin: str, destination: str) -> float | None:
    """Return the miles between two stadium ids, or ``None`` when either has no coordinates."""
    coordinates = constants.STADIUM_COORDINATES
    if origin not in coordinates or destination not in coordinates:
        return None
    return great_circle_miles(coordinates[origin], coordinates[destination])


def travel_miles(games: pl.DataFrame) -> pl.DataFrame:
    """Return each side's distance from its home stadium to the game's stadium.

    Args:
        games: Schedule rows with ``game_id``, ``season``, ``week``, ``home_abbr``,
            ``away_abbr``, ``neutral`` and ``stadium_id``.

    Returns:
        ``game_id``, ``home_miles`` and ``away_miles``, in the order of ``games``; null when the
        venue or the team's home stadium (see ``home_stadiums``) has no coordinates.

    """
    venues = games.select("game_id", pl.col("stadium_id").alias("venue"))
    team_games = (
        _team_games(games)
        .join(venues, on="game_id", how="left")
        .join(
            home_stadiums(games).rename({"stadium_id": "home_stadium"}),
            on=["season", "team_abbr"],
            how="left",
        )
    )
    pairs = team_games.select("home_stadium", "venue").drop_nulls().unique(maintain_order=True)
    distances = pl.DataFrame(
        [
            (origin, destination, _stadium_miles(origin, destination))
            for origin, destination in pairs.iter_rows()
        ],
        schema={"home_stadium": pl.String, "venue": pl.String, "miles": pl.Float64},
        orient="row",
    )
    measured = team_games.join(distances, on=["home_stadium", "venue"], how="left")
    return _per_game(games, measured, "miles")


def _travel_terms(games: pl.DataFrame) -> pl.DataFrame:
    """Return ``game_id`` and ``travel_elo``: the travel term credited to the home side.

    ``0.004 * (away_miles - home_miles)`` at a non-neutral game or the Super Bowl, else 0; a
    missing distance counts as no travel term.
    """
    miles = travel_miles(games)
    counted = ~_neutral() | (pl.col("game_type") == _SUPER_BOWL)
    term = TRAVEL_ELO_PER_MILE * (pl.col("away_miles") - pl.col("home_miles"))
    return (
        games.select("game_id", "neutral", "game_type")
        .join(miles, on="game_id", how="left", maintain_order="left")
        .select(
            "game_id",
            pl.when(counted).then(term).otherwise(0.0).fill_null(0.0).alias("travel_elo"),
        )
    )


def rolling_home_field(games: pl.DataFrame, *, travel: bool) -> pl.DataFrame:
    """Return the rolling home field, in Elo points, for every week on the schedule.

    Args:
        games: Schedule rows with ``game_id``, ``season``, ``week``, ``game_type``, the teams,
            the scores, ``neutral`` and, when ``travel`` is set, ``stadium_id``.
        travel: Whether the travel term is on; its mean over the window is then subtracted.

    Returns:
        ``season``, ``week`` and ``home_field``: 25 times the mean home margin over completed,
        non-neutral regular-season games from the same week ten seasons earlier up to the week
        before, less the mean travel term with ``travel``; null in the frame's first season and
        for a window without games.

    """
    completed = games.filter(
        (pl.col("game_type") == _REGULAR_SEASON)
        & ~_neutral()
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    if travel:
        completed = completed.join(_travel_terms(games), on="game_id", how="left")
    else:
        completed = completed.with_columns(pl.lit(0.0).alias("travel_elo"))
    weekly = completed.group_by("season", "week").agg(
        pl.len().alias("games"),
        (pl.col("home_score") - pl.col("away_score")).cast(pl.Float64).sum().alias("margin"),
        pl.col("travel_elo").sum().alias("travel"),
    )
    targets = games.select("season", "week").unique().sort("season", "week")
    window = targets.join(
        weekly.rename({"season": "window_season", "week": "window_week"}), how="cross"
    )
    oldest = pl.col("season") - ROLLING_HOME_FIELD_SEASONS
    after_start = (pl.col("window_season") > oldest) | (
        (pl.col("window_season") == oldest) & (pl.col("window_week") >= pl.col("week"))
    )
    before_week = (pl.col("window_season") < pl.col("season")) | (
        (pl.col("window_season") == pl.col("season")) & (pl.col("window_week") < pl.col("week"))
    )
    sums = (
        window.filter(after_start & before_week)
        .group_by("season", "week")
        .agg(pl.col("games").sum(), pl.col("margin").sum(), pl.col("travel").sum())
    )
    first_season = games.get_column("season").min()
    estimate = POINTS_TO_ELO * pl.col("margin") / pl.col("games") - pl.col("travel") / pl.col(
        "games"
    )
    return targets.join(sums, on=["season", "week"], how="left").select(
        "season",
        "week",
        pl.when((pl.col("season") == first_season) | (pl.col("games").fill_null(0) == 0))
        .then(None)
        .otherwise(estimate)
        .alias("home_field"),
    )


def home_edges(games: pl.DataFrame, *, rest: bool, travel: bool, rolling: bool) -> pl.DataFrame:
    """Return the Elo points each game credits to the home side before the playoff boost.

    Args:
        games: Schedule rows with ``game_id``, ``season``, ``week``, ``game_type``, the teams,
            the scores, ``neutral`` and, for travel, ``stadium_id``.
        rest: Credit 25 points to a team coming off a bye week.
        travel: Use the 55-point base plus 4 points per 1,000 miles of the visitor's trip.
        rolling: Use the rolling ten-season home field as the base.

    Returns:
        ``game_id``, ``home_edge`` and its parts (``home_field``, ``travel_elo``,
        ``rest_elo``), in the order of ``games``. See the module docstring for each part.

    """
    fixed = TRAVEL_BASE_HOME_FIELD if travel else CLASSIC_HOME_FIELD
    frame = games.select("game_id", "season", "week", "neutral")
    if rolling:
        weekly = rolling_home_field(games, travel=travel)
        frame = frame.join(weekly, on=["season", "week"], how="left", maintain_order="left")
        base = pl.col("home_field").fill_null(fixed)
    else:
        base = pl.lit(fixed)
    frame = frame.with_columns(pl.when(_neutral()).then(0.0).otherwise(base).alias("home_field"))
    if travel:
        terms = _travel_terms(games)
        _log_missing_venues(games)
        frame = frame.join(terms, on="game_id", how="left", maintain_order="left")
    else:
        frame = frame.with_columns(pl.lit(0.0).alias("travel_elo"))
    if rest:
        byes = bye_weeks(games)
        frame = frame.join(byes, on="game_id", how="left", maintain_order="left").with_columns(
            (
                BYE_ELO
                * (pl.col("home_bye").cast(pl.Float64) - pl.col("away_bye").cast(pl.Float64))
            ).alias("rest_elo")
        )
    else:
        frame = frame.with_columns(pl.lit(0.0).alias("rest_elo"))
    return frame.select(
        "game_id",
        (pl.col("home_field") + pl.col("travel_elo") + pl.col("rest_elo")).alias("home_edge"),
        "home_field",
        "travel_elo",
        "rest_elo",
    )


def _log_missing_venues(games: pl.DataFrame) -> None:
    """Warn about games played at a stadium id that ``STADIUM_COORDINATES`` lacks."""
    known = set(constants.STADIUM_COORDINATES)
    venues = games.get_column("stadium_id")
    missing = games.filter(venues.is_null() | ~venues.is_in(known))
    if missing.height:
        log.warning(
            "Own Elo travel: %d games at stadiums without coordinates get no travel term: %s",
            missing.height,
            sorted({str(venue) for venue in missing.get_column("stadium_id").to_list()})[:10],
        )
