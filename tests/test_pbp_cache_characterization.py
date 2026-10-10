"""Characterization test: the play-by-play aggregations on a download that carries every column.

A small synthetic world (two seasons, two games each, with every kind of play nflverse codes:
completions, incompletions, interceptions, sacks with and without a lost fumble, designed runs,
scrambles coded the pre-2006 way, kneels, spikes, botched snaps, penalties, two-point tries,
the kicking plays and a timeout) is downloaded through ``loaders.load_pbp`` with nflreadpy
replaced by the fixture, then aggregated by the three consumers of the cached play-by-play:
``pbp.aggregate_pbp_team_game_stats``, ``pbp.aggregate_pbp_team_box_score_stats`` and
``qb_stats.aggregate_qb_game_stats``. Their outputs, column types included, are compared with the
snapshots under ``tests/fixtures/pbp_cache_characterization/``.

The raw rows carry every column the cache requests and more, with values chosen so that an
aggregation reading one of the additional columns would move its output: scrambles flagged only
by ``qb_scramble``, ``passing_yards`` and ``rushing_yards`` that differ from ``yards_gained``,
botched snaps, and touchdown, fumble and tackle credits. Caching more columns must leave the
snapshots untouched. To rewrite them after an intended change, run this module with
``NFLP_UPDATE_SNAPSHOTS=1`` and commit the rewrite with the change.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest
from tests import snapshots

from nfl_predictor import constants
from nfl_predictor.utils.polars import loaders, pbp, qb_stats

if TYPE_CHECKING:
    from polars.datatypes import DataType

SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "pbp_cache_characterization"

# (season, week, away, home) as nflverse writes the codes in that season.
_GAMES = (
    (2005, 1, "OAK", "SD"),
    (2005, 2, "STL", "WAS"),
    (2024, 1, "LA", "KC"),
    (2024, 2, "LV", "BUF"),
)
# Seasons before 2006 leave ``qb_dropback`` at 0 on scrambles and publish no pass depth.
_FIRST_CHARTED_SEASON = 2006
# The script each offense runs in a game, in order; every kind appears in every game.
_SCRIPT = (
    "kickoff",
    "run",
    "pass",
    "incomplete",
    "scramble",
    "pass",
    "run",
    "sack",
    "punt",
    "run",
    "pass",
    "penalty",
    "run_fumble",
    "pass",
    "interception",
    "run",
    "aborted",
    "pass",
    "strip_sack",
    "run",
    "pass",
    "field_goal",
    "run",
    "scramble",
    "pass",
    "extra_point",
    "two_point_pass",
    "run",
    "incomplete",
    "two_point_run",
    "spike",
    "kneel",
    "timeout",
)
# Flags each kind of play sets (every other flag is 0).
_KIND_FLAGS: dict[str, dict[str, object]] = {
    "pass": {
        "play_type": "pass",
        "qb_dropback": 1,
        "pass": 1,
        "pass_attempt": 1,
        "complete_pass": 1,
    },
    "incomplete": {
        "play_type": "pass",
        "qb_dropback": 1,
        "pass": 1,
        "pass_attempt": 1,
        "incomplete_pass": 1,
    },
    "interception": {
        "play_type": "pass",
        "qb_dropback": 1,
        "pass": 1,
        "pass_attempt": 1,
        "interception": 1,
    },
    "sack": {
        "play_type": "pass",
        "qb_dropback": 1,
        "pass": 1,
        "pass_attempt": 1,
        "sack": 1,
        "qb_hit": 1,
    },
    "strip_sack": {
        "play_type": "pass",
        "qb_dropback": 1,
        "pass": 1,
        "pass_attempt": 1,
        "sack": 1,
        "fumble": 1,
        "fumble_lost": 1,
        "fumble_forced": 1,
    },
    "run": {"play_type": "run", "rush": 1, "rush_attempt": 1},
    "run_fumble": {
        "play_type": "run",
        "rush": 1,
        "rush_attempt": 1,
        "fumble": 1,
        "fumble_forced": 1,
    },
    "scramble": {"play_type": "run", "pass": 1, "rush_attempt": 1, "qb_scramble": 1},
    "kneel": {"play_type": "qb_kneel", "qb_kneel": 1, "rush_attempt": 1},
    "spike": {
        "play_type": "qb_spike",
        "qb_spike": 1,
        "qb_dropback": 1,
        "pass": 1,
        "pass_attempt": 1,
        "incomplete_pass": 1,
    },
    "aborted": {"play_type": "run", "rush": 1, "rush_attempt": 1, "aborted_play": 1, "fumble": 1},
    "punt": {"play_type": "punt", "special": 1, "special_teams_play": 1, "punt_attempt": 1},
    "field_goal": {
        "play_type": "field_goal",
        "special": 1,
        "special_teams_play": 1,
        "field_goal_attempt": 1,
    },
    "kickoff": {
        "play_type": "kickoff",
        "special": 1,
        "special_teams_play": 1,
        "kickoff_attempt": 1,
    },
    "extra_point": {"play_type": "extra_point", "special": 1, "special_teams_play": 1},
    "two_point_pass": {
        "play_type": "pass",
        "qb_dropback": 1,
        "pass": 1,
        "pass_attempt": 1,
        "two_point_attempt": 1,
    },
    "two_point_run": {"play_type": "run", "rush": 1, "rush_attempt": 1, "two_point_attempt": 1},
    "penalty": {"play_type": "no_play", "penalty": 1, "qb_scramble": 1},
    "timeout": {},
}
_SPECIAL_KINDS = frozenset({"punt", "field_goal", "kickoff", "extra_point"})
_PASS_KINDS = frozenset({"pass", "incomplete", "interception", "two_point_pass"})
_RUN_KINDS = frozenset({"run", "run_fumble", "scramble", "aborted", "two_point_run"})
_FUMBLE_KINDS = frozenset({"strip_sack", "run_fumble", "aborted"})

# Every column the fixture writes, as nflverse types it. ``goal_to_go`` is typed per season.
_STRING_COLUMNS = (
    "game_id",
    "season_type",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "posteam_type",
    "play_type",
    "two_point_conv_result",
    "td_team",
    "penalty_team",
    "penalty_type",
    "fixed_drive_result",
    "drive_start_yard_line",
    "drive_end_yard_line",
    "drive_time_of_possession",
    "yrdln",
    "side_of_field",
    "end_yard_line",
    "pass_length",
    "series_result",
    "return_team",
    "timeout_team",
    "desc",
    "nfl_api_id",
    "passer_player_id",
    "passer_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "receiver_player_id",
    "receiver_player_name",
    "td_player_id",
    "penalty_player_id",
    "pass_defense_1_player_id",
    *(
        f"{credit}_{part}"
        for credit in (
            "fumbled_1",
            "fumbled_2",
            "fumble_recovery_1",
            "fumble_recovery_2",
            "forced_fumble_player_1",
            "forced_fumble_player_2",
            "solo_tackle_1",
            "solo_tackle_2",
            "assist_tackle_1",
            "assist_tackle_2",
            "assist_tackle_3",
            "assist_tackle_4",
            "tackle_with_assist_1",
            "tackle_with_assist_2",
        )
        for part in ("team", "player_id")
    ),
)
_INTEGER_COLUMNS = ("season", "week")
_FLOAT_COLUMNS = (
    "play_id",
    "qtr",
    "half_seconds_remaining",
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "score_differential_post",
    "posteam_score",
    "posteam_score_post",
    "wp",
    "wpa",
    "qb_dropback",
    "qb_kneel",
    "qb_spike",
    "qb_scramble",
    "rush",
    "pass",
    "rush_attempt",
    "pass_attempt",
    "aborted_play",
    "shotgun",
    "no_huddle",
    "epa",
    "qb_epa",
    "success",
    "yards_gained",
    "passing_yards",
    "rushing_yards",
    "return_yards",
    "air_yards",
    "air_epa",
    "yac_epa",
    "yards_after_catch",
    "xyac_mean_yardage",
    "xyac_median_yardage",
    "cpoe",
    "xpass",
    "pass_oe",
    "third_down_converted",
    "third_down_failed",
    "fourth_down_converted",
    "fourth_down_failed",
    "two_point_attempt",
    "defensive_two_point_conv",
    "touchdown",
    "sack",
    "qb_hit",
    "interception",
    "fumble",
    "fumble_forced",
    "fumble_lost",
    "complete_pass",
    "incomplete_pass",
    "pass_touchdown",
    "rush_touchdown",
    "tackled_for_loss",
    "first_down",
    "first_down_pass",
    "first_down_rush",
    "first_down_penalty",
    "penalty",
    "penalty_yards",
    "special",
    "special_teams_play",
    "punt_attempt",
    "field_goal_attempt",
    "kickoff_attempt",
    "fixed_drive",
    "drive_play_count",
    "drive_first_downs",
    "drive_inside20",
    "drive_yards_penalized",
    "ydsnet",
    "series",
    "series_success",
)
# A season from this one on publishes ``goal_to_go`` as a float in the fixture, before it as an
# integer, as nflverse does for some of its seasons.
_FLOAT_GOAL_TO_GO_SEASON = 2020
_POSTSEASON_WEEK = 19


def _player(role: str, team: str, slot: int = 1) -> str:
    """Return a fixture player id for a role on a team."""
    return f"00-{role}{team}{slot}"


def _scores(kind: str, index: int) -> bool:
    """Return whether the script's play at ``index`` is an offensive touchdown."""
    return index % 9 in {2, 6} and kind in {"pass", "run"}


def _flags(kind: str, season: int) -> dict[str, object]:
    """Return the kind's flags, every other flag at 0, with the scramble coding of its era."""
    flags: dict[str, object] = dict.fromkeys(
        (
            "qb_dropback",
            "qb_kneel",
            "qb_spike",
            "qb_scramble",
            "rush",
            "pass",
            "rush_attempt",
            "pass_attempt",
            "aborted_play",
            "complete_pass",
            "incomplete_pass",
            "interception",
            "sack",
            "qb_hit",
            "fumble",
            "fumble_forced",
            "fumble_lost",
            "special",
            "special_teams_play",
            "punt_attempt",
            "field_goal_attempt",
            "kickoff_attempt",
            "two_point_attempt",
            "penalty",
        ),
        0.0,
    )
    flags.update(
        {
            name: float(value) if isinstance(value, int) else value
            for name, value in _KIND_FLAGS[kind].items()
        }
    )
    if kind == "scramble":
        flags["qb_dropback"] = 1.0 if season >= _FIRST_CHARTED_SEASON else 0.0
    return flags


def _yards(rng: np.random.Generator, kind: str) -> float | None:
    """Return the yards a play of this kind gained."""
    ranges = {
        "pass": (3, 40),
        "sack": (-10, -1),
        "strip_sack": (-10, -1),
        "run": (-3, 25),
        "run_fumble": (-3, 25),
        "two_point_run": (0, 3),
        "scramble": (0, 16),
        "aborted": (-6, 4),
        "punt": (0, 45),
        "field_goal": (0, 1),
        "kickoff": (10, 45),
        "extra_point": (0, 1),
        "two_point_pass": (0, 3),
    }
    if kind == "timeout":
        return None
    if kind == "kneel":
        return -1.0
    low, high = ranges.get(kind, (0, 1))
    return float(rng.integers(low, high))


@dataclass(frozen=True)
class _Snap:
    """The down, distance, gain, result and teams of one fixture play."""

    down: float | None
    ydstogo: float
    yards: float
    success: bool
    offense: str
    defense: str


def _situation(kind: str, index: int, snap: _Snap) -> dict[str, object]:
    """Return the conversion, first-down, touchdown, two-point and penalty columns of a play."""
    down, yards, success = snap.down, snap.yards, snap.success
    offense, defense = snap.offense, snap.defense
    gained_first = down is not None and yards >= snap.ydstogo
    scored = _scores(kind, index)
    pick_six = kind == "interception" and index % 2 == 0
    out: dict[str, object] = {
        "third_down_converted": float(down == 3 and success),
        "third_down_failed": float(down == 3 and not success),
        "fourth_down_converted": float(down == 4 and success),
        "fourth_down_failed": float(down == 4 and not success),
        "first_down": float(gained_first),
        "first_down_pass": float(gained_first and kind in _PASS_KINDS),
        "first_down_rush": float(gained_first and kind in _RUN_KINDS),
        "first_down_penalty": float(kind == "penalty" and index % 2 == 0),
        "defensive_two_point_conv": 0.0,
        "tackled_for_loss": float(kind in _RUN_KINDS and yards < 0),
        "pass_touchdown": float(scored and kind == "pass"),
        "rush_touchdown": float(scored and kind == "run"),
        "touchdown": float(scored or pick_six),
    }
    if scored or pick_six:
        out["td_team"] = offense if scored else defense
    if kind.startswith("two_point"):
        out["two_point_conv_result"] = "success" if index % 2 else "failure"
    if kind == "penalty" or (kind == "run" and index % 11 == 1):
        on_offense = kind == "penalty" and index % 2 == 0
        out |= {
            "penalty": 1.0,
            "penalty_team": offense if on_offense else defense,
            "penalty_player_id": _player("O" if on_offense else "D", offense, 7),
            "penalty_yards": 5.0 if on_offense else 15.0,
            "penalty_type": "False Start" if on_offense else "Defensive Pass Interference",
        }
    return out


def _ball_carriers(kind: str, index: int, quarterback: str, offense: str) -> dict[str, object]:
    """Return the passer, receiver, rusher and touchdown credits of a play."""
    out: dict[str, object] = {}
    if kind in _PASS_KINDS | {"sack", "strip_sack", "spike"}:
        out |= {"passer_player_id": quarterback, "passer_player_name": f"Q.{quarterback[-4:]}"}
    if kind in {"pass", "interception"} or (kind == "incomplete" and index % 2 == 0):
        receiver = _player("W", offense, index % 3)
        out |= {"receiver_player_id": receiver, "receiver_player_name": f"W.{receiver[-4:]}"}
    if kind in _RUN_KINDS | {"kneel"}:
        rusher = quarterback if kind in {"scramble", "aborted", "kneel"} else _player("R", offense)
        out |= {"rusher_player_id": rusher, "rusher_player_name": f"R.{rusher[-4:]}"}
    if _scores(kind, index):
        out["td_player_id"] = out.get("receiver_player_id", out.get("rusher_player_id"))
    return out


def _people(
    kind: str, index: int, teams: tuple[str, str], *, two_passers: bool
) -> dict[str, object]:
    """Return the passer, rusher, receiver, fumble, tackle and return credits of a play."""
    offense, defense = teams
    quarterback = _player("Q", offense, 2 if two_passers and index > len(_SCRIPT) // 2 else 1)
    out = _ball_carriers(kind, index, quarterback, offense)
    if kind in _FUMBLE_KINDS:
        lost = kind == "strip_sack"
        fumbler = _player("C", offense) if kind == "aborted" else quarterback
        if kind == "run_fumble":
            fumbler = _player("R", offense)
        out |= {
            "fumbled_1_team": offense,
            "fumbled_1_player_id": fumbler,
            "fumble_recovery_1_team": defense if lost else offense,
            "fumble_recovery_1_player_id": _player(
                "D" if lost else "O", defense if lost else offense
            ),
            "forced_fumble_player_1_team": defense,
            "forced_fumble_player_1_player_id": _player("D", defense, 3),
        }
    if kind in {"pass", "run", "run_fumble", "scramble", "interception"}:
        out |= {"solo_tackle_1_team": defense, "solo_tackle_1_player_id": _player("D", defense)}
    if kind == "run" and index % 2 == 0:
        out |= {
            "assist_tackle_1_team": defense,
            "assist_tackle_1_player_id": _player("D", defense, 4),
            "tackle_with_assist_1_team": defense,
            "tackle_with_assist_1_player_id": _player("D", defense, 5),
        }
    if kind == "incomplete" and index % 3 == 0:
        out["pass_defense_1_player_id"] = _player("D", defense, 6)
    if kind in {"punt", "kickoff"}:
        out["return_team"] = defense
    return out


def _yardage(
    rng: np.random.Generator, kind: str, yards: float, *, charted: bool
) -> dict[str, object]:
    """Return the passing, rushing, return and pass-depth yardage of a play."""
    out: dict[str, object] = {
        "return_yards": float(rng.integers(0, 30))
        if kind in {"punt", "kickoff", "interception"}
        else 0.0,
    }
    if kind == "pass":
        out["passing_yards"] = yards + 1.0
    if kind in _RUN_KINDS | {"kneel"}:
        out["rushing_yards"] = yards - 1.0
    if charted and kind in _PASS_KINDS:
        air = float(rng.integers(-2, 30))
        out |= {
            "air_yards": air,
            "air_epa": round(float(rng.normal(0.0, 0.8)), 4),
            "pass_length": "deep" if air >= 15 else "short",
            "cpoe": round(float(rng.normal(0.0, 30.0)), 4),
        }
        if kind == "pass":
            out |= {
                "yards_after_catch": max(yards - air, 0.0),
                "yac_epa": round(float(rng.normal(0.0, 0.6)), 4),
                "xyac_mean_yardage": round(float(rng.uniform(2.0, 8.0)), 4),
            }
    if charted and kind not in _SPECIAL_KINDS:
        xpass = round(float(rng.uniform(0.1, 0.9)), 4)
        out |= {"xpass": xpass, "pass_oe": round(100 * (float(kind in _PASS_KINDS) - xpass), 4)}
    return out


def _play(
    rng: np.random.Generator, game: tuple[int, int, str, str], offense: str, index: int
) -> dict[str, object]:
    """Return one raw play-by-play row of the script for ``offense`` in ``game``."""
    season, week, away, home = game
    kind = _SCRIPT[index]
    defense = home if offense == away else away
    row: dict[str, object] = {
        "game_id": f"{season}_{week:02d}_{away}_{home}",
        "season": season,
        "week": week,
        "season_type": "REG",
        "home_team": home,
        "away_team": away,
        "play_id": float((1 if offense == away else 2) * 1000 + index * 10),
        "qtr": float(1 + index * 4 // len(_SCRIPT)),
        "half_seconds_remaining": float(1800 - (index * 53) % 1800),
        "desc": f"({index}) {kind} by {offense} against {defense}",
        "nfl_api_id": f"api-{season}-{week}-{offense}",
        "xyac_median_yardage": 3.0,
        **_flags(kind, season),
    }
    yards = _yards(rng, kind)
    if yards is None:
        return {**row, "timeout_team": offense}
    special = kind in _SPECIAL_KINDS
    two_point = kind.startswith("two_point")
    down = float(1 + index % 4) if not special and not two_point and kind != "penalty" else None
    ydstogo = float(1 + (index * 7) % 15)
    epa = round(float(rng.normal(0.0, 1.3)), 4)
    yardline = 8.0 if _scores(kind, index) else float(5 + (index * 13) % 90)
    near_side = defense if yardline < 50 else offense
    row |= {
        "posteam": offense,
        "defteam": defense,
        "posteam_type": "away" if offense == away else "home",
        "yardline_100": yardline,
        "goal_to_go": int(yardline <= 10),
        "yrdln": f"{near_side} {int(min(yardline, 100 - yardline))}",
        "side_of_field": near_side,
        "down": down,
        "ydstogo": ydstogo,
        "epa": epa,
        "qb_epa": round(epa - 0.75, 4) if kind == "strip_sack" else epa,
        "success": float(epa > 0 and not special),
        "yards_gained": yards,
        "wp": round(float(rng.uniform(0.03, 0.97)), 4),
        "wpa": round(float(rng.normal(0.0, 0.03)), 4),
        "score_differential": float(index % 15 - 7),
        "score_differential_post": float(index % 15 - 6),
        "posteam_score": float(index),
        "posteam_score_post": float(index + (7 if _scores(kind, index) else 0)),
        "shotgun": float(index % 2),
        "no_huddle": float(index % 5 == 0),
        "fixed_drive": float(index // 6 * 2 + (1 if offense == away else 2)),
        "fixed_drive_result": ("Touchdown", "Punt", "Field goal", "Turnover")[index // 6 % 4],
        "drive_start_yard_line": f"{offense} {20 + index // 6 * 5}",
        "drive_end_yard_line": f"{defense} {10 + index % 30}",
        "drive_play_count": 6.0,
        "drive_first_downs": float(index // 6 % 3),
        "drive_inside20": float(index // 6 % 2 == 0),
        "drive_time_of_possession": f"{2 + index % 4}:{index % 60:02d}",
        "drive_yards_penalized": float(5 * (index % 3)),
        "ydsnet": float(30 + index % 40),
        "series": float(1 + index // 3),
        "series_success": float(index % 3 != 0),
        "series_result": ("First down", "Punt", "Touchdown")[index % 3],
    }
    teams = (offense, defense)
    snap = _Snap(down, ydstogo, yards, success=epa > 0, offense=offense, defense=defense)
    row |= _situation(kind, index, snap)
    row |= _people(kind, index, teams, two_passers=season == _GAMES[-1][0] and week == 1)
    row |= _yardage(rng, kind, yards, charted=season >= _FIRST_CHARTED_SEASON)
    return row


def _schema(season: int) -> dict[str, DataType]:
    """Return the raw column types for one season, as nflverse publishes them."""
    goal_to_go = pl.Float64() if season >= _FLOAT_GOAL_TO_GO_SEASON else pl.Int32()
    return {
        **dict.fromkeys(_STRING_COLUMNS, pl.String()),
        **dict.fromkeys(_INTEGER_COLUMNS, pl.Int32()),
        **dict.fromkeys(_FLOAT_COLUMNS, pl.Float64()),
        "goal_to_go": goal_to_go,
    }


def _raw_season(season: int) -> pl.DataFrame:
    """Return the fixture's raw play-by-play for one season, postseason plays included."""
    rng = np.random.default_rng(season)
    rows = [
        _play(rng, game, offense, index)
        for game in _GAMES
        if game[0] == season
        for offense in game[2:]
        for index in range(len(_SCRIPT))
    ]
    postseason_game = (season, _POSTSEASON_WEEK, _GAMES[-1][3], _GAMES[-1][2])
    postseason = [
        {**_play(rng, postseason_game, postseason_game[2], index), "season_type": "POST"}
        for index in range(3)
    ]
    return pl.from_dicts([*rows, *postseason], schema=_schema(season))


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pl.DataFrame]:
    """Download the fixture seasons through the loader and aggregate them three ways."""
    seasons = sorted({game[0] for game in _GAMES})
    raw = {season: _raw_season(season) for season in seasons}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loaders.nfl, "load_pbp", lambda seasons: raw[seasons[0]])
        loaded = loaders.load_pbp(
            seasons, cache_dir=tmp_path_factory.mktemp("pbp_cache"), current_season=2025
        )
    return {
        "raw": pl.concat(raw.values(), how="diagonal_relaxed"),
        "team_game_stats": pbp.aggregate_pbp_team_game_stats(loaded),
        "team_box_scores": pbp.aggregate_pbp_team_box_score_stats(loaded),
        "qb_game_stats": qb_stats.aggregate_qb_game_stats(loaded),
    }


def _as_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    return pd.read_csv(StringIO(frame.write_csv()))


def _types(frame: pl.DataFrame) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in frame.schema.items()}


def test_the_world_carries_every_requested_column(world: dict[str, pl.DataFrame]) -> None:
    """Guard the fixture: the download holds every column the cache requests, and more."""
    # Act
    raw_columns = set(world["raw"].columns)

    # Assert
    assert set(constants.PBP_COLUMNS) <= raw_columns
    assert {"nfl_api_id", "xyac_median_yardage"} <= raw_columns


def test_the_world_exercises_every_aggregation_family(world: dict[str, pl.DataFrame]) -> None:
    """Guard the fixture: every pinned family has real values."""
    # Arrange
    team_games = world["team_game_stats"]
    box_scores = world["team_box_scores"]
    qb_games = world["qb_game_stats"]

    # Act
    team_game_totals = team_games.select(pl.exclude("team_abbr", "opponent_abbr")).sum()
    box_score_totals = box_scores.select(
        pl.exclude("team_abbr", "opponent_abbr", "season_type")
    ).sum()

    # Assert
    assert team_games.height == 2 * len(_GAMES)
    assert all(value != 0 for value in team_game_totals.row(0)), team_game_totals
    assert all(value != 0 for value in box_score_totals.row(0)), box_score_totals
    assert qb_games.get_column("qb_id").n_unique() > len(_GAMES) * 2


def test_team_game_stats_match_their_snapshot(world: dict[str, pl.DataFrame]) -> None:
    """The per-team-game counts and sums are unchanged."""
    # Arrange
    frame = world["team_game_stats"]

    # Act & Assert
    snapshots.check_csv(
        "team_game_stats.csv", _as_pandas(frame), SNAPSHOT_DIR / "team_game_stats.csv"
    )
    snapshots.check_json(
        "team_game_stats_schema.json", _types(frame), SNAPSHOT_DIR / "team_game_stats_schema.json"
    )


def test_team_box_scores_match_their_snapshot(world: dict[str, pl.DataFrame]) -> None:
    """The per-team-game box scores are unchanged."""
    # Arrange
    frame = world["team_box_scores"]

    # Act & Assert
    snapshots.check_csv(
        "team_box_scores.csv", _as_pandas(frame), SNAPSHOT_DIR / "team_box_scores.csv"
    )
    snapshots.check_json(
        "team_box_scores_schema.json", _types(frame), SNAPSHOT_DIR / "team_box_scores_schema.json"
    )


def test_qb_game_stats_match_their_snapshot(world: dict[str, pl.DataFrame]) -> None:
    """The per-quarterback-game sums are unchanged."""
    # Arrange
    frame = world["qb_game_stats"]

    # Act & Assert
    snapshots.check_csv("qb_game_stats.csv", _as_pandas(frame), SNAPSHOT_DIR / "qb_game_stats.csv")
    snapshots.check_json(
        "qb_game_stats_schema.json", _types(frame), SNAPSHOT_DIR / "qb_game_stats_schema.json"
    )
