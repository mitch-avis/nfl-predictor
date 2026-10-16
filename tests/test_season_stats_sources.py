"""Tests for the season inputs read through this repository's loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_predictor.season_stats import game_logs, sources
from nfl_predictor.utils.polars import loaders, nflverse_sources

if TYPE_CHECKING:
    from pathlib import Path


def _play_with_team_codes(team: str) -> dict[str, str]:
    """Return one play whose every team-code column names ``team``, yard lines at its 25."""
    return {
        **dict.fromkeys(sources.PBP_TEAM_COLUMNS, team),
        **dict.fromkeys(sources.PBP_YARD_LINE_COLUMNS, f"{team} 25"),
    }


@pytest.mark.parametrize(
    ("alias", "canonical"), [("LA", "LAR"), ("STL", "LAR"), ("OAK", "LV"), ("WAS", "WSH")]
)
def test_prepare_pbp_normalizes_every_team_code_column(alias: str, canonical: str) -> None:
    """A raw ``LA`` in ``penalty_team`` or a yard line never matches the normalized ``posteam``."""
    # Arrange
    pbp = pl.DataFrame([_play_with_team_codes(alias)])

    # Act
    result = sources.prepare_pbp(pbp, 2025)

    # Assert
    assert result.to_dicts() == [_play_with_team_codes(canonical)]


def test_prepare_pbp_rewrites_only_the_team_code_in_yard_lines() -> None:
    """Midfield (``50``, ``MID 50``), another code (``LAC 30``), and nulls are kept as written."""
    # Arrange
    pbp = pl.DataFrame({"yrdln": ["LA 25", "LAC 30", "50", "MID 50", "LA -10", None]})

    # Act
    result = sources.prepare_pbp(pbp, 2025)

    # Assert
    assert result.get_column("yrdln").to_list() == [
        "LAR 25",
        "LAC 30",
        "50",
        "MID 50",
        "LAR -10",
        None,
    ]


def test_prepare_pbp_treats_an_empty_team_as_missing() -> None:
    """Old play-by-play leaves ``posteam`` empty, not null, on timeouts and other non-plays."""
    # Arrange
    pbp = pl.DataFrame({"posteam": ["ATL", ""], "defteam": ["DAL", ""]})

    # Act
    result = sources.prepare_pbp(pbp, 1999)

    # Assert
    assert result.rows() == [("ATL", "DAL"), (None, None)]


def _gap_play() -> pl.DataFrame:
    """Return one play with a value in every field some season lacks."""
    return pl.DataFrame(
        {
            "air_yards": [12.0],
            "air_epa": [0.9],
            "pass_length": ["deep"],
            "yards_after_catch": [4.0],
            "yac_epa": [0.3],
            "xyac_mean_yardage": [5.1],
            "qb_hit": [0.0],
            "no_huddle": [0.0],
            "drive_yards_penalized": [-5.0],
            "epa": [1.2],
        }
    )


@pytest.mark.parametrize(
    ("season", "blank"),
    [
        (
            2000,
            {
                "air_yards",
                "air_epa",
                "pass_length",
                "yards_after_catch",
                "yac_epa",
                "xyac_mean_yardage",
                "no_huddle",
                "drive_yards_penalized",
            },
        ),
        (
            2004,
            {
                "air_yards",
                "air_epa",
                "pass_length",
                "yards_after_catch",
                "yac_epa",
                "xyac_mean_yardage",
                "qb_hit",
            },
        ),
        (2006, set[str]()),
    ],
)
def test_prepare_pbp_blanks_the_fields_nflverse_lacks_that_season(
    season: int, blank: set[str]
) -> None:
    # Act
    result = sources.prepare_pbp(_gap_play(), season)

    # Assert
    assert {
        column for column in result.columns if result.get_column(column).is_null().all()
    } == blank


def _broken_game_pbp() -> pl.DataFrame:
    """Return 2001 PIT-at-JAX plays crediting every tackle to PIT, as nflverse writes them.

    jax_1 plays for JAX and pit_1 for PIT; nflverse names PIT for both.
    """
    return pl.DataFrame(
        {
            "game_id": ["2001_01_PIT_JAX"] * 20,
            "posteam": ["PIT", "JAX"] * 10,
            "defteam": ["JAX", "PIT"] * 10,
            "solo_tackle_1_team": ["PIT"] * 20,
            "solo_tackle_1_player_id": ["jax_1", "pit_1"] * 10,
        }
    )


def _rosters(teams: list[str]) -> pl.DataFrame:
    """Return the 2001 season roster of jax_1 and pit_1 under ``teams``."""
    return pl.DataFrame({"season": [2001, 2001], "gsis_id": ["jax_1", "pit_1"], "team": teams})


def test_repair_one_team_games_rebuilds_player_teams_from_the_roster() -> None:
    # Act
    result = sources.repair_one_team_games(_broken_game_pbp(), lambda: _rosters(["JAX", "PIT"]))

    # Assert
    assert result.get_column("solo_tackle_1_team").to_list() == ["JAX", "PIT"] * 10


def test_repair_one_team_games_reads_no_roster_when_every_game_credits_both_teams() -> None:
    # Arrange
    pbp = _broken_game_pbp().with_columns(pl.Series("solo_tackle_1_team", ["JAX", "PIT"] * 10))

    def unexpected_roster() -> pl.DataFrame:
        msg = "roster read without a game credited to one team"
        raise AssertionError(msg)

    # Act
    result = sources.repair_one_team_games(pbp, unexpected_roster)

    # Assert
    assert result.equals(pbp)


def test_prepare_schedule_keeps_regular_season_games_under_the_team_names_builders_read() -> None:
    # Arrange
    schedule = pl.DataFrame(
        {
            "game_id": ["2025_01_SEA_LAR", "2025_19_SEA_LAR"],
            "game_type": ["REG", "WC"],
            "week": [1, 19],
            "home_abbr": ["LAR", "LAR"],
            "away_abbr": ["SEA", "SEA"],
            "home_score": [24, 20],
            "away_score": [17, 27],
        }
    )

    # Act
    result = sources.prepare_schedule(schedule)

    # Assert
    assert result.to_dicts() == [
        {
            "game_id": "2025_01_SEA_LAR",
            "week": 1,
            "home_team": "LAR",
            "away_team": "SEA",
            "home_score": 24,
            "away_score": 17,
        }
    ]


def test_missing_pbp_columns_names_every_column_the_builders_read_and_lack() -> None:
    # Arrange
    pbp = pl.DataFrame(schema=dict.fromkeys(sources.SEASON_STATS_PBP_COLUMNS[2:], pl.String))

    # Act
    missing = sources.missing_pbp_columns(pbp)

    # Assert
    assert missing == list(sources.SEASON_STATS_PBP_COLUMNS[:2])


def test_season_stats_pbp_columns_cover_every_field_with_a_season_gap() -> None:
    # Act
    uncovered = set(sources.PBP_FIELD_GAPS) - set(sources.SEASON_STATS_PBP_COLUMNS)

    # Assert
    assert uncovered == set()


def test_official_team_stats_source_requests_every_column_the_override_reads() -> None:
    # Act
    columns = sources.OFFICIAL_TEAM_STATS.columns

    # Assert
    assert set(game_logs.OFFICIAL_TEAM_STAT_COLUMNS) <= set(columns)
    assert {"game_id", "week", "team", "opponent_team"} <= set(columns)


def test_official_team_stats_keep_the_regular_season_with_canonical_codes(
    tmp_path: Path,
) -> None:
    # Arrange
    raw = pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [1, 1, 19],
            "season_type": ["REG", "REG", "POST"],
            "game_id": ["2025_01_SEA_LA", "2025_01_SEA_LA", "2025_19_SEA_LA"],
            "team": ["LA", "SEA", "LA"],
            "opponent_team": ["SEA", "LA", "SEA"],
            "passing_yards": [250, 180, 300],
            "passing_epa": [5.5, 1.7, 2.0],
        }
    )
    source = sources.OFFICIAL_TEAM_STATS

    # Act
    result = source.prepare(raw, regular_season_only=True)

    # Assert
    assert result.select("team", "opponent_team", "passing_yards").rows() == [
        ("LAR", "SEA", 250),
        ("SEA", "LAR", 180),
    ]
    assert result.columns == list(source.columns)
    assert source.cache_path(tmp_path, 2025, regular_season_only=True).name == (
        "official_team_stats_2025_reg.parquet"
    )


def _season_frames() -> dict[str, pl.DataFrame]:
    """Return what each loader returns for one 2025 game, SEA at LAR."""
    values: dict[str, object] = {
        "game_id": "2025_01_SEA_LAR",
        "season": 2025,
        "season_type": "REG",
        "week": 1,
        "posteam": "LAR",
        "defteam": "SEA",
        "drive_start_yard_line": "LA 25",
    }
    pbp_row = {column: values.get(column) for column in sources.SEASON_STATS_PBP_COLUMNS}
    pbp_schema = {
        column: pl.Int64 if column in {"season", "week"} else pl.String
        for column in sources.SEASON_STATS_PBP_COLUMNS
    }
    return {
        "pbp": pl.DataFrame([pbp_row], schema=pbp_schema),
        "player_stats": pl.DataFrame({"game_id": ["2025_01_SEA_LAR"], "team": ["LAR"]}),
        "schedule": pl.DataFrame(
            {
                "game_id": ["2025_01_SEA_LAR"],
                "game_type": ["REG"],
                "week": [1],
                "home_abbr": ["LAR"],
                "away_abbr": ["SEA"],
                "home_score": [24],
                "away_score": [17],
            }
        ),
        "official": pl.DataFrame({"game_id": ["2025_01_SEA_LAR"], "team": ["LAR"]}),
        "snap_counts": pl.DataFrame({"game_id": ["2025_01_SEA_LAR"], "player": ["A QB"]}),
        "identity": pl.DataFrame({"qb_id": ["00-0000001"], "qb_position": ["QB"]}),
    }


def _stub_loaders(monkeypatch: pytest.MonkeyPatch, frames: dict[str, pl.DataFrame]) -> list[str]:
    """Point every loader ``load_season_inputs`` reads at ``frames``; return the calls made."""
    calls: list[str] = []

    def stub(name: str, frame: pl.DataFrame) -> object:
        def loader(*_args: object, **_kwargs: object) -> pl.DataFrame:
            calls.append(name)
            return frame

        return loader

    monkeypatch.setattr(loaders, "load_pbp", stub("pbp", frames["pbp"]))
    monkeypatch.setattr(
        nflverse_sources, "load_weekly_player_stats", stub("player_stats", frames["player_stats"])
    )
    monkeypatch.setattr(loaders, "load_schedule", stub("schedule", frames["schedule"]))
    monkeypatch.setattr(sources, "load_official_team_stats", stub("official", frames["official"]))
    monkeypatch.setattr(
        nflverse_sources, "load_snap_counts", stub("snap_counts", frames["snap_counts"])
    )
    monkeypatch.setattr(
        nflverse_sources, "load_season_qb_identity", stub("identity", frames["identity"])
    )
    monkeypatch.setattr(nflverse_sources, "load_rosters", stub("rosters", pl.DataFrame()))
    return calls


def test_load_season_inputs_reads_each_source_through_the_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    frames = _season_frames()
    calls = _stub_loaders(monkeypatch, frames)

    # Act
    inputs = sources.load_season_inputs(2025)

    # Assert
    assert sorted(calls) == [
        "identity",
        "official",
        "pbp",
        "player_stats",
        "schedule",
        "snap_counts",
    ]
    assert inputs.pbp.get_column("drive_start_yard_line").to_list() == ["LAR 25"]
    assert inputs.schedule.select("home_team", "away_team").row(0) == ("LAR", "SEA")
    assert inputs.player_stats.equals(frames["player_stats"])
    assert inputs.official_team_stats.equals(frames["official"])
    assert inputs.snap_counts.equals(frames["snap_counts"])
    assert inputs.qb_identity.equals(frames["identity"])


def test_load_season_inputs_refuses_play_by_play_without_a_column_the_builders_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    frames = _season_frames()
    frames["pbp"] = frames["pbp"].drop("qb_scramble", "wp")
    _stub_loaders(monkeypatch, frames)

    # Act & Assert
    with pytest.raises(ValueError, match="qb_scramble, wp"):
        sources.load_season_inputs(2025)


def test_load_season_inputs_accepts_a_season_without_play_by_play(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before kickoff the current season has no plays; the loader then returns its schema."""
    # Arrange
    frames = _season_frames()
    frames["pbp"] = pl.DataFrame(schema={"game_id": pl.String, "posteam": pl.String})
    _stub_loaders(monkeypatch, frames)

    # Act
    inputs = sources.load_season_inputs(2026)

    # Assert
    assert inputs.pbp.is_empty()


def test_prepared_rams_plays_credit_their_penalties_touchdowns_and_drive_starts() -> None:
    """nflverse writes the Rams as ``LA`` in ``penalty_team``, ``td_team``, and yard lines.

    Fixture, SEA at LA: the Rams false-start, run 5 yards, and score on a 70-yard run, a drive
    from their own 25. Seattle's drive from its own 30 draws Rams pass interference (20 yards),
    then ends in a fumble the Rams return for a touchdown.
    """
    # Arrange
    game = {"game_id": "2025_01_SEA_LA", "season": 2025, "season_type": "REG", "week": 1}
    rams_drive = {
        **game,
        "posteam": "LA",
        "defteam": "SEA",
        "fixed_drive": 1,
        "fixed_drive_result": "Touchdown",
        "drive_start_yard_line": "LA 25",
    }
    seattle_drive = {
        **game,
        "posteam": "SEA",
        "defteam": "LA",
        "fixed_drive": 2,
        "fixed_drive_result": "Opp touchdown",
        "drive_start_yard_line": "SEA 30",
    }
    quiet = {
        "pass": 0,
        "rush": 0,
        "qb_dropback": 0,
        "rushing_yards": 0.0,
        "fumble": 0,
        "fumble_lost": 0,
        "touchdown": 0,
        "td_team": None,
        "penalty": 0,
        "penalty_team": None,
        "penalty_type": None,
        "penalty_yards": None,
    }
    pbp = pl.DataFrame(
        [
            {
                **rams_drive,
                **quiet,
                "penalty": 1,
                "penalty_team": "LA",
                "penalty_type": "False Start",
                "penalty_yards": 5.0,
            },
            {**rams_drive, **quiet, "rush": 1, "rushing_yards": 5.0},
            {
                **rams_drive,
                **quiet,
                "rush": 1,
                "rushing_yards": 70.0,
                "touchdown": 1,
                "td_team": "LA",
            },
            {
                **seattle_drive,
                **quiet,
                "pass": 1,
                "qb_dropback": 1,
                "penalty": 1,
                "penalty_team": "LA",
                "penalty_type": "Defensive Pass Interference",
                "penalty_yards": 20.0,
            },
            {
                **seattle_drive,
                **quiet,
                "rush": 1,
                "fumble": 1,
                "fumble_lost": 1,
                "touchdown": 1,
                "td_team": "LA",
            },
        ]
    )
    schedule = sources.prepare_schedule(
        pl.DataFrame(
            {
                "game_id": ["2025_01_SEA_LA"],
                "game_type": ["REG"],
                "week": [1],
                "home_abbr": ["LAR"],
                "away_abbr": ["SEA"],
                "home_score": [14],
                "away_score": [0],
            }
        )
    )
    rams_columns = {
        "penalties": 2,
        "penalty_yards": 25.0,
        "offensive_penalties": 1,
        "offensive_penalty_yards": 5.0,
        "defensive_penalties": 1,
        "defensive_penalty_yards": 20.0,
        "defensive_pass_interference": 1,
        # One false start and one penalty over the Rams' two snaps.
        "presnap_penalty_rate": 0.5,
        "penalty_rate_per_offensive_snap": 0.5,
        "penalty_differential": -2,
        "penalty_yards_differential": -25.0,
        "def_tds": 1,
        "fumble_recovery_tds": 1,
        "total_tds": 2,
        "avg_starting_field_position": 25.0,
        "long_field_score_pct": 1.0,
        "avg_starting_field_position_allowed": 30.0,
    }

    # Act
    result = game_logs.build_weekly_team_stats(
        sources.prepare_pbp(pbp, 2025), pl.DataFrame(), schedule, pl.DataFrame()
    )

    # Assert
    rams = result.filter(pl.col("team") == "LAR").select(list(rams_columns)).to_dicts()
    assert rams == [rams_columns]
