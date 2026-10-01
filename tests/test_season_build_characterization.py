"""Characterization test: a season build's game rows and strength snapshots.

A small world (four teams, a prior season and two built seasons, one playoff round, a week
with two teams on a bye, an unplayed week without ELO rows) runs through
``data_collection.process_season`` with every input family present: ELO and quarterback
trends, coach history, TeamRankings for the season and the one before it, play-by-play sums
for schedule-adjusted strength, and the stat and strength prior blends. The rows and the
strength snapshots it returns are compared with the snapshots under
``tests/fixtures/season_build_characterization/``, their column types included.

Restructuring the week builder must leave the snapshots untouched. To rewrite them after an
intended behavior change, run this module with ``NFLP_UPDATE_SNAPSHOTS=1`` and commit the
rewrite with the change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from tests import snapshots

from nfl_predictor import constants, data_collection

SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "season_build_characterization"

# Two teams from each of two divisions, so divisional and standings features have values.
_TEAMS = ("BUF", "KC", "MIA", "LV")
_PRIOR_SEASON = 2018
_BUILT_SEASONS = (2019, 2020)
# Enough weeks for the four-week trends to differ from the season-to-date means.
_WEEKS = 6
# The built seasons have 17 regular-season weeks, so week 18 is the first playoff round.
_PLAYOFF_WEEK = 18
_PLAYOFF_SEASON = 2019
# The last week of the last season is unplayed and has no ELO rows of its own.
_UNPLAYED = (2020, _WEEKS)
# Two teams are on a bye in this week, so the next week mixes teams with and without games.
_BYE_WEEK = (2020, 1)


@dataclass(frozen=True)
class _World:
    """Every input of the season builds."""

    schedule: pl.DataFrame
    team_stats: pl.DataFrame
    elo: pl.DataFrame
    rankings: dict[int, pl.DataFrame]


def _slate(season: int, week: int) -> list[tuple[str, str]]:
    """Return the week's two games as (away, home): a round robin, one round per week."""
    rest = list(_TEAMS[1:])
    shift = (season + week) % len(rest)
    order = [_TEAMS[0], *rest[shift:], *rest[:shift]]
    pairs = [(order[0], order[3]), (order[1], order[2])]
    return pairs if week % 2 else [(home, away) for away, home in pairs]


def _weeks(season: int) -> list[tuple[int, str, list[tuple[str, str]]]]:
    """Return the season's weeks as (week, game type, games)."""
    weeks = [
        (week, "REG", _slate(season, week)[: 1 if (season, week) == _BYE_WEEK else None])
        for week in range(1, _WEEKS + 1)
    ]
    if season == _PLAYOFF_SEASON:
        weeks.append((_PLAYOFF_WEEK, "WC", _slate(season, 1)[:1]))
    return weeks


def _team_game(
    rng: np.random.Generator,
    key: tuple[int, int],
    sides: tuple[str, str, bool],
    points: tuple[int, int],
) -> dict[str, object]:
    """Return one team's per-game stat row."""
    season, week = key
    team, opponent, is_home = sides
    scored, allowed = points
    return {
        "season": season,
        "week": week,
        "team_abbr": team,
        "opponent_abbr": opponent,
        "is_home": is_home,
        "offensive_snaps": float(rng.integers(55, 75)),
        "defensive_snaps": float(rng.integers(55, 75)),
        "pass_epa_sum": float(rng.normal(0.0, 6.0)),
        "rush_epa_sum": float(rng.normal(0.0, 4.0)),
        "pass_epa_allowed_sum": float(rng.normal(0.0, 6.0)),
        "rush_epa_allowed_sum": float(rng.normal(0.0, 4.0)),
        "st_epa_for": float(rng.normal(0.0, 2.0)),
        "st_epa_against": float(rng.normal(0.0, 2.0)),
        "st_plays": float(rng.integers(8, 14)),
        "points_scored": float(scored),
        "points_allowed": float(allowed),
        "pass_yards": float(rng.integers(120, 360)),
        "rush_yards": float(rng.integers(40, 180)),
        "scoring_margin": float(scored - allowed),
        "turnover_margin": float(rng.integers(-3, 4)),
    }


def _elo_row(
    rng: np.random.Generator, season: int, week: int, away: str, home: str
) -> dict[str, object]:
    """Return one game's ELO row."""
    return {
        "season": season,
        "week": week,
        "away_abbr": away,
        "home_abbr": home,
        "away_elo_pre": float(rng.normal(1500.0, 60.0)),
        "home_elo_pre": float(rng.normal(1500.0, 60.0)),
        "away_qb": f"QB {away}",
        "home_qb": f"QB {home}",
        "away_qb_elo_pre": float(rng.normal(0.0, 30.0)),
        "home_qb_elo_pre": float(rng.normal(0.0, 30.0)),
        "away_qb_value_pre": float(rng.normal(100.0, 20.0)),
        "home_qb_value_pre": float(rng.normal(100.0, 20.0)),
    }


def _rankings(rng: np.random.Generator, season: int) -> pl.DataFrame:
    """Return the season's weekly TeamRankings rows, every rating and situational stat.

    Only played regular-season weeks have rows, so the playoff round falls back to the
    latest ratings and the unplayed week has none.
    """
    weeks = [
        week
        for week, game_type, _ in _weeks(season)
        if game_type == "REG" and (season, week) != _UNPLAYED
    ]
    return pl.DataFrame(
        [
            {
                "team_abbr": team,
                "week": week,
                **{
                    column: float(rng.normal(0.0, 5.0))
                    for column in (*constants.TR_RATINGS, *constants.TR_STATS)
                },
            }
            for week in weeks
            for team in _TEAMS
        ]
    )


def _world() -> _World:
    rng = np.random.default_rng(20261001)
    games: list[dict[str, object]] = []
    team_games: list[dict[str, object]] = []
    elo_rows: list[dict[str, object]] = []
    for season in (_PRIOR_SEASON, *_BUILT_SEASONS):
        for week, game_type, slate in _weeks(season):
            played = (season, week) != _UNPLAYED
            for index, (away, home) in enumerate(slate):
                away_points = int(rng.integers(3, 38))
                home_points = int(rng.integers(3, 38))
                games.append(
                    {
                        "game_id": f"{season}_{week:02d}_{away}_{home}",
                        "season": season,
                        "week": week,
                        "game_type": game_type,
                        "date": date(season, 9, 7) + timedelta(days=7 * (week - 1) + index),
                        "away_abbr": away,
                        "home_abbr": home,
                        "away_score": away_points if played else None,
                        "home_score": home_points if played else None,
                        "away_coach": f"Coach {away}",
                        "home_coach": f"Coach {home}",
                        "spread_line": float(rng.normal(0.0, 4.0)),
                        "total_line": float(rng.normal(44.0, 3.0)),
                    }
                )
                if played:
                    elo_rows.append(_elo_row(rng, season, week, away, home))
                    team_games.extend(
                        (
                            _team_game(
                                rng, (season, week), (away, home, False), (away_points, home_points)
                            ),
                            _team_game(
                                rng, (season, week), (home, away, True), (home_points, away_points)
                            ),
                        )
                    )
    return _World(
        schedule=pl.DataFrame(games),
        team_stats=pl.DataFrame(team_games),
        elo=pl.DataFrame(elo_rows),
        rankings={season: _rankings(rng, season) for season in (_PRIOR_SEASON, *_BUILT_SEASONS)},
    )


@dataclass(frozen=True)
class _Build:
    """The game rows and the strength snapshots of every built season."""

    games: pl.DataFrame
    snapshots: pl.DataFrame


@pytest.fixture(scope="module")
def build() -> _Build:
    world = _world()
    strength_snapshots: list[pl.DataFrame] = []
    frames = [
        data_collection.process_season(
            season,
            world.schedule,
            world.team_stats,
            data_collection.SeasonInputs(
                min_season=_PRIOR_SEASON,
                elo_df=world.elo,
                tr_df=world.rankings[season],
                prev_tr_df=world.rankings.get(season - 1),
                strength_snapshots=strength_snapshots,
            ),
        )
        for season in (_PRIOR_SEASON, *_BUILT_SEASONS)
    ]
    return _Build(
        games=pl.concat(frames, how="diagonal"),
        snapshots=pl.concat(strength_snapshots, how="vertical"),
    )


def _as_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    return pd.read_csv(StringIO(frame.write_csv()))


def _schema(frame: pl.DataFrame) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in frame.schema.items()}


def test_the_world_exercises_every_week_feature_family(build: _Build) -> None:
    """Guard the fixture: the pinned rows carry real values for every family."""
    games = build.games
    for column in (
        "away_pass_yards",
        "home_wins",
        "away_elo_pre",
        "home_qb_elo_pre",
        "away_elo_4wk_trend",
        "home_qb_elo_4wk_trend",
        "away_scoring_margin_4wk_trend",
        "home_coach_win_pct_prior",
        "away_predictive_rating",
        "home_third_down_pct",
        "away_last_5_games_rating_trend",
        "home_adj_strength_composite",
        "away_sos_played_raw",
        "home_sos_remaining_adj",
        "pass_yards_diff",
        "is_divisional_matchup",
        "away_next_is_divisional_matchup",
        "home_division_rank",
    ):
        assert games.get_column(column).drop_nulls().n_unique() > 1, column
    played = games.select("season", "week").unique()
    assert played.filter(pl.col("week") == _PLAYOFF_WEEK).height == 1
    assert (
        played.filter((pl.col("season") == _UNPLAYED[0]) & (pl.col("week") == _UNPLAYED[1])).height
        == 1
    )
    unplayed_elo = games.filter(
        (pl.col("season") == _UNPLAYED[0]) & (pl.col("week") == _UNPLAYED[1])
    ).get_column("away_elo_pre")
    assert unplayed_elo.null_count() == 0, "the unplayed week takes each team's latest ELO"


def test_season_build_rows_match_snapshots(build: _Build) -> None:
    snapshots.check_csv("games.csv", _as_pandas(build.games), SNAPSHOT_DIR / "games.csv")
    snapshots.check_json(
        "games_schema.json", _schema(build.games), SNAPSHOT_DIR / "games_schema.json"
    )
    snapshots.check_csv(
        "strength_snapshots.csv",
        _as_pandas(build.snapshots),
        SNAPSHOT_DIR / "strength_snapshots.csv",
    )
    snapshots.check_json(
        "strength_snapshots_schema.json",
        _schema(build.snapshots),
        SNAPSHOT_DIR / "strength_snapshots_schema.json",
    )
    if snapshots.updating():
        pytest.skip(f"snapshots rewritten in {SNAPSHOT_DIR}")
