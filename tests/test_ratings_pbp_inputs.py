"""Tests for the adapter from this repository's play-by-play to the ratings' input rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl
import pytest

from nfl_predictor.ratings.inputs import QB_GAME_SCHEMA, TEAM_GAME_SCHEMA
from nfl_predictor.ratings.pbp_inputs import (
    PASSER_GAME_SCHEMA,
    SeasonSources,
    passer_games_from_pbp,
    quarterback_rows,
    team_games_from_pbp,
)

_SEASON = 2005
_GAME = "2005_01_AAA_BBB"
_SECOND_GAME = "2005_02_CCC_AAA"

_PLAY_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.String(),
    "season": pl.Int32(),
    "season_type": pl.String(),
    "week": pl.Int32(),
    "posteam": pl.String(),
    "defteam": pl.String(),
    "play_type": pl.String(),
    "qb_dropback": pl.Float64(),
    "qb_scramble": pl.Float64(),
    "rush": pl.Float64(),
    "pass": pl.Float64(),
    "qb_kneel": pl.Float64(),
    "qb_spike": pl.Float64(),
    "special": pl.Float64(),
    "epa": pl.Float64(),
    "qb_epa": pl.Float64(),
    "passer_player_id": pl.String(),
    "passer_player_name": pl.String(),
}


@dataclass(frozen=True)
class _Play:
    """One play-by-play row with the columns the adapter reads; ``attempt`` is ``pass``."""

    posteam: str | None
    defteam: str | None
    game_id: str = _GAME
    week: int = 1
    season: int = _SEASON
    season_type: str = "REG"
    play_type: str = "pass"
    qb_dropback: float = 0.0
    qb_scramble: float = 0.0
    rush: float = 0.0
    attempt: float = 0.0
    qb_kneel: float = 0.0
    qb_spike: float = 0.0
    special: float = 0.0
    epa: float = 0.0
    qb_epa: float | None = None
    passer_player_id: str | None = None
    passer_player_name: str | None = None

    def row(self) -> dict[str, object]:
        """Return the play as a row; ``qb_epa`` defaults to ``epa``."""
        values = asdict(self)
        values["pass"] = values.pop("attempt")
        if values["qb_epa"] is None:
            values["qb_epa"] = values["epa"]
        return values


def _plays(plays: list[_Play]) -> pl.DataFrame:
    """Return play-by-play rows in the cache's types."""
    return pl.DataFrame([play.row() for play in plays], schema=_PLAY_SCHEMA)


def _schedule(
    games: list[tuple[str, int, str, str, int, int]], season: int = _SEASON
) -> pl.DataFrame:
    """Return schedule rows ``(game_id, week, away, home, away_score, home_score)``."""
    return pl.DataFrame(
        {
            "game_id": [game[0] for game in games],
            "season": [season] * len(games),
            "game_type": ["REG"] * len(games),
            "week": [game[1] for game in games],
            "away_abbr": [game[2] for game in games],
            "home_abbr": [game[3] for game in games],
            "away_score": [game[4] for game in games],
            "home_score": [game[5] for game in games],
        },
        schema={
            "game_id": pl.String(),
            "season": pl.Int64(),
            "game_type": pl.String(),
            "week": pl.Int64(),
            "away_abbr": pl.String(),
            "home_abbr": pl.String(),
            "away_score": pl.Int64(),
            "home_score": pl.Int64(),
        },
    )


_FIRST_GAME_SCHEDULE = [(_GAME, 1, "AAA", "BBB", 10, 17)]


def _snap_plays() -> list[_Play]:
    """Return AAA's plays of the first game: five snaps, a punt, and a play with no snap."""
    return [
        _Play("AAA", "BBB", qb_dropback=1.0, attempt=1.0, epa=0.5),
        _Play("AAA", "BBB", play_type="run", rush=1.0, epa=-0.25),
        _Play("AAA", "BBB", play_type="run", qb_scramble=1.0, epa=0.125),
        _Play("AAA", "BBB", play_type="qb_kneel", qb_kneel=1.0, epa=-0.0625),
        _Play("AAA", "BBB", play_type="qb_spike", qb_spike=1.0, attempt=1.0, epa=-0.03125),
        _Play("AAA", "BBB", play_type="no_play", epa=0.75),
        _Play("AAA", "BBB", play_type="punt", special=1.0, epa=0.375),
        _Play("BBB", "AAA", qb_dropback=1.0, attempt=1.0, epa=1.0),
    ]


def _row(games: pl.DataFrame, team: str, game_id: str = _GAME) -> dict[str, object]:
    """Return ``team``'s row of ``game_id``."""
    rows = games.filter((pl.col("team") == team) & (pl.col("game_id") == game_id))
    assert rows.height == 1
    return rows.row(0, named=True)


def test_team_games_have_the_season_and_the_team_game_columns() -> None:
    # Arrange
    plays = _plays(_snap_plays())

    # Act
    games = team_games_from_pbp(plays, _schedule(_FIRST_GAME_SCHEDULE))

    # Assert
    assert games.schema == pl.Schema({"season": pl.Int64(), **TEAM_GAME_SCHEMA})


def test_team_games_count_dropbacks_runs_scramble_runs_kneels_and_spikes_as_snaps() -> None:
    # Arrange
    plays = _plays(_snap_plays())

    # Act
    games = team_games_from_pbp(plays, _schedule(_FIRST_GAME_SCHEDULE))

    # Assert
    assert _row(games, "AAA")["offensive_snaps"] == 5


def test_team_games_sum_the_epa_of_the_snaps() -> None:
    # Arrange
    plays = _plays(_snap_plays())

    # Act
    games = team_games_from_pbp(plays, _schedule(_FIRST_GAME_SCHEDULE))

    # Assert
    assert _row(games, "AAA")["offensive_epa"] == 0.5 - 0.25 + 0.125 - 0.0625 - 0.03125


def test_team_games_count_special_teams_plays_for_the_team_with_the_ball() -> None:
    # Arrange
    plays = _plays(_snap_plays())

    # Act
    games = team_games_from_pbp(plays, _schedule(_FIRST_GAME_SCHEDULE))

    # Assert
    assert (_row(games, "AAA")["st_plays"], _row(games, "AAA")["st_epa"]) == (1, 0.375)
    assert (_row(games, "BBB")["st_plays"], _row(games, "BBB")["st_epa"]) == (0, 0.0)


def test_team_games_count_a_scramble_only_on_a_run_play() -> None:
    # Arrange
    plays = _plays(
        [
            _Play("AAA", "BBB", play_type="pass", qb_scramble=1.0, epa=2.0),
            _Play("BBB", "AAA", qb_dropback=1.0, epa=1.0),
            _Play("AAA", "BBB", play_type="run", rush=1.0, epa=0.5),
        ]
    )

    # Act
    games = team_games_from_pbp(plays, _schedule(_FIRST_GAME_SCHEDULE))

    # Assert
    assert (_row(games, "AAA")["offensive_snaps"], _row(games, "AAA")["offensive_epa"]) == (1, 0.5)


def test_team_games_count_a_scramble_flagged_as_a_rush_once() -> None:
    # Arrange
    plays = _plays(
        [
            _Play("AAA", "BBB", play_type="run", qb_scramble=1.0, rush=1.0, epa=0.5),
            _Play("BBB", "AAA", qb_dropback=1.0, epa=1.0),
        ]
    )

    # Act
    games = team_games_from_pbp(plays, _schedule(_FIRST_GAME_SCHEDULE))

    # Assert
    assert (_row(games, "AAA")["offensive_snaps"], _row(games, "AAA")["offensive_epa"]) == (1, 0.5)


def test_team_games_leave_out_plays_without_both_teams() -> None:
    # Arrange
    plays = _plays(
        [
            *_snap_plays(),
            _Play("", "", qb_dropback=1.0, epa=4.0),
            _Play(None, "BBB", qb_dropback=1.0, epa=4.0),
            _Play("AAA", None, qb_dropback=1.0, epa=4.0),
            _Play("AAA", "", special=1.0, epa=4.0),
        ]
    )

    # Act
    games = team_games_from_pbp(plays, _schedule(_FIRST_GAME_SCHEDULE))

    # Assert
    assert games.get_column("team").to_list() == ["AAA", "BBB"]
    assert (_row(games, "AAA")["offensive_snaps"], _row(games, "AAA")["st_plays"]) == (5, 1)


def test_team_games_leave_out_a_team_without_a_snap() -> None:
    # Arrange
    plays = _plays(
        [
            _Play("AAA", "BBB", qb_dropback=1.0, epa=0.5),
            _Play("BBB", "AAA", play_type="punt", special=1.0, epa=0.25),
        ]
    )

    # Act
    games = team_games_from_pbp(plays, _schedule(_FIRST_GAME_SCHEDULE))

    # Assert
    assert games.get_column("team").to_list() == ["AAA"]


def test_team_games_leave_out_postseason_plays() -> None:
    # Arrange
    postseason = _Play("AAA", "BBB", season_type="POST", qb_dropback=1.0, epa=3.0)
    plays = _plays([*_snap_plays(), postseason])

    # Act
    games = team_games_from_pbp(plays, _schedule(_FIRST_GAME_SCHEDULE))

    # Assert
    assert _row(games, "AAA")["offensive_snaps"] == 5


def test_team_games_take_the_margin_and_the_home_side_from_the_schedule() -> None:
    # Arrange
    plays = _plays(_snap_plays())

    # Act
    games = team_games_from_pbp(plays, _schedule(_FIRST_GAME_SCHEDULE))

    # Assert
    assert (_row(games, "AAA")["point_margin"], _row(games, "AAA")["is_home"]) == (-7, False)
    assert (_row(games, "BBB")["point_margin"], _row(games, "BBB")["is_home"]) == (7, True)


def test_team_games_sort_by_season_team_week_and_game() -> None:
    # Arrange
    plays = _plays(
        [
            _Play("CCC", "AAA", game_id=_SECOND_GAME, week=2, qb_dropback=1.0),
            _Play("AAA", "CCC", game_id=_SECOND_GAME, week=2, qb_dropback=1.0),
            _Play("BBB", "AAA", qb_dropback=1.0),
            _Play("AAA", "BBB", qb_dropback=1.0),
            _Play("AAA", "BBB", game_id="2004_01_AAA_BBB", season=2004, qb_dropback=1.0),
            _Play("BBB", "AAA", game_id="2004_01_AAA_BBB", season=2004, qb_dropback=1.0),
        ]
    )
    schedule = pl.concat(
        [
            _schedule([*_FIRST_GAME_SCHEDULE, (_SECOND_GAME, 2, "CCC", "AAA", 3, 6)]),
            _schedule([("2004_01_AAA_BBB", 1, "AAA", "BBB", 0, 3)], season=2004),
        ]
    )

    # Act
    games = team_games_from_pbp(plays, schedule)

    # Assert
    assert games.select("season", "team", "week").rows() == [
        (2004, "AAA", 1),
        (2004, "BBB", 1),
        (2005, "AAA", 1),
        (2005, "AAA", 2),
        (2005, "BBB", 1),
        (2005, "CCC", 2),
    ]


def test_team_games_without_the_play_type_column_raise_naming_it() -> None:
    # Arrange
    plays = _plays(_snap_plays()).drop("play_type")

    # Act & Assert
    with pytest.raises(ValueError, match="play_type"):
        team_games_from_pbp(plays, _schedule(_FIRST_GAME_SCHEDULE))


# Quarterback rows. The starter is the players file's quarterback; the others test the identity.
_STARTER = "00-0000001"
_TIGHT_END = "00-0000002"  # players file TE, weekly roster QB from week 3
_RECEIVER = "00-0000003"  # players file WR, never a quarterback
_UNPLACED = "00-0000004"  # in no players file row and no roster week
_LATE_BACK = "00-0000005"  # no players file row, weekly roster RB from week 3
_LATE_QB = "00-0000006"  # no players file row, weekly roster QB from week 3
_BACKUP = "00-0000007"  # players file QB, in the snap counts only
_THIRD_GAME = "2005_03_AAA_DDD"
_GAMES = [
    (_GAME, 1, "AAA", "BBB", 10, 17),
    (_SECOND_GAME, 2, "CCC", "AAA", 3, 6),
    (_THIRD_GAME, 3, "AAA", "DDD", 21, 20),
]
_OPPONENTS = {_GAME: "BBB", _SECOND_GAME: "CCC", _THIRD_GAME: "DDD"}


def _players() -> pl.DataFrame:
    """Return the players file rows of the passers above."""
    return pl.DataFrame(
        {
            "gsis_id": [_STARTER, _TIGHT_END, _RECEIVER, _BACKUP],
            "display_name": ["Alpha Passer", "Tee Passer", "Wide Passer", "Backup Passer"],
            "position": ["QB", "TE", "WR", "QB"],
            "pfr_id": ["PassAl00", "PassTe00", "PassWi00", "PassBa00"],
            "espn_id": [None, None, None, None],
        },
        schema=dict.fromkeys(
            ("gsis_id", "display_name", "position", "pfr_id", "espn_id"), pl.String()
        ),
    )


def _weekly_rosters() -> pl.DataFrame:
    """Return the weekly roster rows: the starter every week, the late listings from week 3."""
    rows = [(_STARTER, week, "QB", "Alpha Passer") for week in (1, 2, 3)]
    rows += [(_TIGHT_END, 3, "QB", "Tee Passer")]
    rows += [(_LATE_BACK, 3, "RB", "Late Back"), (_LATE_QB, 3, "QB", "Late Quarterback")]
    return pl.DataFrame(
        {
            "season": [_SEASON] * len(rows),
            "week": [row[1] for row in rows],
            "game_type": ["REG"] * len(rows),
            "team": ["AAA"] * len(rows),
            "position": [row[2] for row in rows],
            "full_name": [row[3] for row in rows],
            "gsis_id": [row[0] for row in rows],
            "pfr_id": [None] * len(rows),
        },
        schema={
            "season": pl.Int64(),
            "week": pl.Int64(),
            "game_type": pl.String(),
            "team": pl.String(),
            "position": pl.String(),
            "full_name": pl.String(),
            "gsis_id": pl.String(),
            "pfr_id": pl.String(),
        },
    )


def _player_stats(rows: list[tuple[str, str, str, int, int, float]]) -> pl.DataFrame:
    """Return weekly player stats rows ``(player_id, position, game_id, week, attempts, epa)``."""
    return pl.DataFrame(
        {
            "player_id": [row[0] for row in rows],
            "position": [row[1] for row in rows],
            "season": [_SEASON] * len(rows),
            "week": [row[3] for row in rows],
            "season_type": ["REG"] * len(rows),
            "game_id": [row[2] for row in rows],
            "team": ["AAA"] * len(rows),
            "attempts": [row[4] for row in rows],
            "passing_epa": [row[5] for row in rows],
        },
        schema={
            "player_id": pl.String(),
            "position": pl.String(),
            "season": pl.Int64(),
            "week": pl.Int64(),
            "season_type": pl.String(),
            "game_id": pl.String(),
            "team": pl.String(),
            "attempts": pl.Int64(),
            "passing_epa": pl.Float64(),
        },
    )


def _snap_counts(rows: list[tuple[str, str, str, int, float]]) -> pl.DataFrame:
    """Return snap count rows ``(pfr_player_id, player, game_id, week, offense_snaps)``."""
    return pl.DataFrame(
        {
            "game_id": [row[2] for row in rows],
            "season": [_SEASON] * len(rows),
            "game_type": ["REG"] * len(rows),
            "week": [row[3] for row in rows],
            "player": [row[1] for row in rows],
            "pfr_player_id": [row[0] for row in rows],
            "position": ["QB"] * len(rows),
            "team": ["AAA"] * len(rows),
            "opponent": [None] * len(rows),
            "offense_snaps": [row[4] for row in rows],
            "offense_pct": [None] * len(rows),
        },
        schema={
            "game_id": pl.String(),
            "season": pl.Int64(),
            "game_type": pl.String(),
            "week": pl.Int64(),
            "player": pl.String(),
            "pfr_player_id": pl.String(),
            "position": pl.String(),
            "team": pl.String(),
            "opponent": pl.String(),
            "offense_snaps": pl.Float64(),
            "offense_pct": pl.Float64(),
        },
    )


def _passes(passer: str, name: str, game_id: str = _GAME) -> list[_Play]:
    """Return a passer's plays for AAA in one game: two passes, a sack, and a spike."""
    week = int(game_id[5:7])
    opponent = _OPPONENTS[game_id]
    return [
        _Play(
            "AAA",
            opponent,
            game_id=game_id,
            week=week,
            qb_dropback=1.0,
            attempt=attempt,
            epa=epa,
            passer_player_id=passer,
            passer_player_name=name,
        )
        for attempt, epa in ((1.0, 0.5), (1.0, -0.25), (0.0, 0.125))
    ] + [
        _Play(
            "AAA",
            opponent,
            game_id=game_id,
            week=week,
            play_type="qb_spike",
            qb_spike=1.0,
            attempt=1.0,
            epa=-0.5,
            passer_player_id=passer,
            passer_player_name=name,
        )
    ]


_PASSES_DROPBACK_EPA = (0.5 - 0.25 + 0.125) + -0.5 - -0.5


def _sources(
    plays: pl.DataFrame,
    player_stats: pl.DataFrame | None = None,
    snap_counts: pl.DataFrame | None = None,
) -> SeasonSources:
    """Return the season's sources around ``plays``."""
    return SeasonSources(
        season=_SEASON,
        plays=plays,
        player_stats=_player_stats([]) if player_stats is None else player_stats,
        snap_counts=_snap_counts([]) if snap_counts is None else snap_counts,
        players=_players(),
        weekly_rosters=_weekly_rosters(),
    )


def _passer_games(
    plays: list[_Play],
    player_stats: pl.DataFrame | None = None,
    snap_counts: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return the passer-game rows of ``plays`` with the fixture's identity sources."""
    frame = _plays(plays)
    team_games = team_games_from_pbp(frame, _schedule(_GAMES))
    return passer_games_from_pbp(_sources(frame, player_stats, snap_counts), team_games)


def _qb(rows: pl.DataFrame, qb_id: str, game_id: str = _GAME) -> dict[str, object]:
    """Return ``qb_id``'s row of ``game_id``."""
    found = rows.filter((pl.col("qb_id") == qb_id) & (pl.col("game_id") == game_id))
    assert found.height == 1
    return found.row(0, named=True)


def test_passer_games_have_the_passer_game_columns() -> None:
    # Arrange
    plays = _plays(_passes(_STARTER, "A.Passer"))
    team_games = team_games_from_pbp(plays, _schedule(_GAMES))

    # Act
    passers = passer_games_from_pbp(_sources(plays), team_games)

    # Assert
    assert passers.schema == pl.Schema(PASSER_GAME_SCHEMA)


def test_quarterback_rows_have_the_quarterback_game_columns() -> None:
    # Arrange
    passers = _passer_games(_passes(_STARTER, "A.Passer"))

    # Act
    rows = quarterback_rows(passers)

    # Assert
    assert rows.schema == pl.Schema(QB_GAME_SCHEMA)


def test_quarterback_rows_count_dropbacks_and_the_pass_attempts_among_them() -> None:
    # Arrange
    passers = _passer_games(_passes(_STARTER, "A.Passer"))

    # Act
    rows = quarterback_rows(passers)

    # Assert
    assert (_qb(rows, _STARTER)["qb_dropbacks"], _qb(rows, _STARTER)["qb_attempts"]) == (3, 2)


def test_quarterback_rows_leave_spikes_out_of_the_dropback_epa() -> None:
    # Arrange
    passers = _passer_games(_passes(_STARTER, "A.Passer"))

    # Act
    rows = quarterback_rows(passers)

    # Assert
    assert _qb(rows, _STARTER)["qb_dropback_epa"] == _PASSES_DROPBACK_EPA


def test_quarterback_rows_take_a_listed_quarterbacks_official_attempts_and_epa() -> None:
    # Arrange
    stats = _player_stats([(_STARTER, "QB", _GAME, 1, 4, 1.5)])
    passers = _passer_games(_passes(_STARTER, "A.Passer"), player_stats=stats)

    # Act
    rows = quarterback_rows(passers)

    # Assert
    assert (_qb(rows, _STARTER)["qb_attempts"], _qb(rows, _STARTER)["qb_dropback_epa"]) == (
        4,
        1.5 - -0.5,
    )


def test_quarterback_rows_keep_the_play_by_play_values_of_a_passer_listed_elsewhere() -> None:
    # Arrange
    stats = _player_stats([(_UNPLACED, "RB", _GAME, 1, 9, 9.0)])
    passers = _passer_games(_passes(_UNPLACED, "U.Passer"), player_stats=stats)

    # Act
    rows = quarterback_rows(passers)

    # Assert
    assert (_qb(rows, _UNPLACED)["qb_attempts"], _qb(rows, _UNPLACED)["qb_dropback_epa"]) == (
        2,
        _PASSES_DROPBACK_EPA,
    )


@pytest.mark.parametrize(
    ("passer", "name"),
    [
        (_STARTER, "Alpha Passer"),
        (_LATE_QB, "Late Quarterback"),
        (_UNPLACED, "U.Passer"),
    ],
    ids=["players-file", "weekly-roster", "play-by-play"],
)
def test_quarterback_rows_name_a_passer_from_the_players_file_the_roster_or_the_plays(
    passer: str, name: str
) -> None:
    # Arrange
    passers = _passer_games(_passes(passer, "U.Passer"))

    # Act
    rows = quarterback_rows(passers)

    # Assert
    assert _qb(rows, passer)["qb_name"] == name


@pytest.mark.parametrize(
    ("passer", "kept"),
    [
        (_STARTER, True),
        (_TIGHT_END, True),
        (_RECEIVER, False),
        (_UNPLACED, True),
        (_LATE_BACK, False),
        (_LATE_QB, True),
    ],
    ids=["players-file-qb", "roster-qb", "receiver", "unplaced", "roster-back", "roster-only-qb"],
)
def test_quarterback_rows_keep_passers_the_season_lists_at_quarterback_or_nowhere(
    passer: str, *, kept: bool
) -> None:
    # Arrange
    passers = _passer_games(_passes(passer, "X.Passer"))

    # Act
    rows = quarterback_rows(passers)

    # Assert
    assert (passer in rows.get_column("qb_id").to_list()) is kept


@pytest.mark.parametrize(
    ("passer", "before_week", "kept"),
    [
        (_TIGHT_END, 3, False),
        (_TIGHT_END, 4, True),
        (_LATE_BACK, 3, True),
        (_LATE_BACK, 4, False),
        (_LATE_QB, 3, True),
        (_LATE_QB, 4, True),
        (_RECEIVER, 4, False),
        (_STARTER, 2, True),
    ],
    ids=[
        "roster-qb-not-yet",
        "roster-qb-listed",
        "roster-back-not-yet-placed",
        "roster-back-placed",
        "roster-only-qb-not-yet-placed",
        "roster-only-qb-listed",
        "receiver",
        "players-file-qb",
    ],
)
def test_quarterback_rows_before_a_week_read_only_the_roster_weeks_before_it(
    passer: str, before_week: int, *, kept: bool
) -> None:
    # Arrange
    passers = _passer_games(_passes(passer, "X.Passer"))

    # Act
    rows = quarterback_rows(passers, before_week=before_week)

    # Assert
    assert (passer in rows.get_column("qb_id").to_list()) is kept


@pytest.mark.parametrize(
    ("before_week", "expected"),
    [(3, (2, _PASSES_DROPBACK_EPA)), (4, (7, 0.75 - -0.5))],
    ids=["not-yet-a-quarterback", "listed-at-quarterback"],
)
def test_quarterback_rows_before_a_week_take_official_values_once_he_is_a_quarterback(
    before_week: int, expected: tuple[int, float]
) -> None:
    # Arrange
    stats = _player_stats([(_LATE_QB, "TE", _GAME, 1, 7, 0.75)])
    passers = _passer_games(_passes(_LATE_QB, "L.Passer"), player_stats=stats)

    # Act
    rows = quarterback_rows(passers, before_week=before_week)

    # Assert
    row = _qb(rows, _LATE_QB)
    assert (row["qb_attempts"], row["qb_dropback_epa"]) == expected


def test_quarterback_rows_before_a_week_leave_out_that_week_and_later() -> None:
    # Arrange
    plays = [*_passes(_STARTER, "A.Passer"), *_passes(_STARTER, "A.Passer", _SECOND_GAME)]
    passers = _passer_games(plays)

    # Act
    rows = quarterback_rows(passers, before_week=2)

    # Assert
    assert rows.get_column("game_id").to_list() == [_GAME]


def test_quarterback_rows_take_the_opponent_and_home_side_from_the_team_games() -> None:
    # Arrange
    passers = _passer_games(_passes(_STARTER, "A.Passer", _SECOND_GAME))

    # Act
    rows = quarterback_rows(passers)

    # Assert
    row = _qb(rows, _STARTER, _SECOND_GAME)
    assert (row["opponent_team"], row["is_home"]) == ("CCC", True)


def test_quarterback_rows_sort_by_team_week_game_and_name() -> None:
    # Arrange
    passers = _passer_games([*_passes(_TIGHT_END, "T.Passer"), *_passes(_UNPLACED, "A.Unplaced")])

    # Act
    rows = quarterback_rows(passers)

    # Assert
    assert rows.get_column("qb_id").to_list() == [_UNPLACED, _TIGHT_END]


def test_quarterback_rows_before_a_week_sort_by_team_week_game_and_id() -> None:
    # Arrange
    passers = _passer_games([*_passes(_TIGHT_END, "T.Passer"), *_passes(_UNPLACED, "A.Unplaced")])

    # Act
    rows = quarterback_rows(passers, before_week=4)

    # Assert
    assert rows.get_column("qb_id").to_list() == [_TIGHT_END, _UNPLACED]


def test_passer_games_add_a_snap_count_quarterback_without_a_dropback() -> None:
    # Arrange
    snaps = _snap_counts([("PassBa00", "Backup Passer", _GAME, 1, 3.0)])
    passers = _passer_games(_passes(_STARTER, "A.Passer"), snap_counts=snaps)

    # Act
    rows = quarterback_rows(passers)

    # Assert
    assert (_qb(rows, _BACKUP)["qb_dropbacks"], _qb(rows, _BACKUP)["qb_name"]) == (
        0,
        "Backup Passer",
    )


def test_passer_games_merge_a_passers_snap_count_row_into_his_dropbacks() -> None:
    # Arrange
    snaps = _snap_counts([("PassAl00", "Alpha Passer", _GAME, 1, 60.0)])
    passers = _passer_games(_passes(_STARTER, "A.Passer"), snap_counts=snaps)

    # Act
    rows = quarterback_rows(passers)

    # Assert
    assert rows.select("qb_id", "qb_dropbacks").rows() == [(_STARTER, 3)]


def test_passer_games_key_an_unmapped_snap_count_quarterback_by_his_snap_id() -> None:
    # Arrange
    snaps = _snap_counts([("PassZe00", "Zed Passer", _GAME, 1, 2.0)])
    passers = _passer_games(_passes(_STARTER, "A.Passer"), snap_counts=snaps)

    # Act
    rows = quarterback_rows(passers)

    # Assert
    assert _qb(rows, "PassZe00")["qb_name"] == "Zed Passer"


def test_passer_games_leave_out_snap_count_rows_without_an_offensive_snap() -> None:
    # Arrange
    snaps = _snap_counts([("PassBa00", "Backup Passer", _GAME, 1, 0.0)])
    passers = _passer_games(_passes(_STARTER, "A.Passer"), snap_counts=snaps)

    # Act
    rows = quarterback_rows(passers)

    # Assert
    assert rows.get_column("qb_id").to_list() == [_STARTER]


def test_passer_games_leave_out_dropbacks_without_a_passer_name() -> None:
    # Arrange
    scramble = _Play("AAA", "BBB", qb_dropback=1.0, epa=3.0, passer_player_id=_STARTER)
    passers = _passer_games([*_passes(_STARTER, "A.Passer"), scramble])

    # Act
    rows = quarterback_rows(passers)

    # Assert
    assert _qb(rows, _STARTER)["qb_dropbacks"] == 3
