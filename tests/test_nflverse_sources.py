"""Tests for the cached loaders of the nflverse player sources.

Every nflreadpy call is replaced by a fake that returns a small frame shaped like the source (its
integer columns published as 32-bit, as nflverse does) or fails the way nflreadpy fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils.polars import nflverse_sources

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

CURRENT = 2024


def _player_stats(season: int, rows: list[dict[str, object]]) -> pl.DataFrame:
    """Return raw weekly player stats: one row per entry, with nflverse's column types."""
    base: dict[str, object] = {
        "player_id": "00-0000001",
        "player_name": "P.Player",
        "player_display_name": "Pat Player",
        "position": "QB",
        "position_group": "QB",
        "season": season,
        "week": 1,
        "season_type": "REG",
        "game_id": f"{season}_01_KC_DEN",
        "team": "KC",
        "opponent_team": "DEN",
        "attempts": 30,
        "passing_yards": 250,
        "passing_epa": 4.5,
        "def_tackles_for_loss": 0,
        "def_qb_hits": 0,
    }
    frame = pl.DataFrame([{**base, **row} for row in rows])
    return frame.with_columns(pl.col(pl.Int64).cast(pl.Int32))


def _fake(
    frames: dict[int, pl.DataFrame], calls: list[int] | None = None
) -> Callable[..., pl.DataFrame]:
    """Return a fake nflreadpy season loader serving ``frames`` and recording each season asked."""

    def load(seasons: list[int], **_kwargs: object) -> pl.DataFrame:
        (season,) = seasons
        if calls is not None:
            calls.append(season)
        return frames[season]

    return load


def _failing(error: Exception, calls: list[int] | None = None) -> Callable[..., pl.DataFrame]:
    """Return a fake nflreadpy season loader that raises ``error``."""

    def load(seasons: list[int], **_kwargs: object) -> pl.DataFrame:
        if calls is not None:
            calls.extend(seasons)
        raise error

    return load


def _no_rosters(seasons: list[int], **_kwargs: object) -> pl.DataFrame:
    """Fail if the roster loader is called."""
    msg = f"rosters should not be loaded for {seasons}"
    raise AssertionError(msg)


@pytest.fixture
def no_rosters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any roster download, for tests whose stats need no team repair."""
    monkeypatch.setattr(nflverse_sources.nfl, "load_rosters", _no_rosters)


@pytest.mark.usefixtures("no_rosters")
def test_a_historical_season_downloads_once_and_then_reads_its_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    calls: list[int] = []
    monkeypatch.setattr(
        nflverse_sources.nfl, "load_player_stats", _fake({2020: _player_stats(2020, [{}])}, calls)
    )
    first = nflverse_sources.load_weekly_player_stats(
        [2020], cache_dir=tmp_path, current_season=CURRENT
    )

    # Act
    second = nflverse_sources.load_weekly_player_stats(
        [2020], cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    assert calls == [2020]
    assert second.equals(first)
    assert (tmp_path / "player_stats_week_2020_reg.parquet").is_file()


@pytest.mark.usefixtures("no_rosters")
def test_the_current_season_is_downloaded_again_even_when_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(
        nflverse_sources.nfl,
        "load_player_stats",
        _fake({CURRENT: _player_stats(CURRENT, [{"passing_yards": 100}])}),
    )
    nflverse_sources.load_weekly_player_stats([CURRENT], cache_dir=tmp_path, current_season=CURRENT)
    monkeypatch.setattr(
        nflverse_sources.nfl,
        "load_player_stats",
        _fake({CURRENT: _player_stats(CURRENT, [{"passing_yards": 300}])}),
    )

    # Act
    stats = nflverse_sources.load_weekly_player_stats(
        [CURRENT], cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    assert stats.get_column("passing_yards").to_list() == [300]


@pytest.mark.usefixtures("no_rosters")
def test_force_refresh_downloads_a_cached_historical_season_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    calls: list[int] = []
    monkeypatch.setattr(
        nflverse_sources.nfl, "load_player_stats", _fake({2020: _player_stats(2020, [{}])}, calls)
    )
    nflverse_sources.load_weekly_player_stats([2020], cache_dir=tmp_path, current_season=CURRENT)

    # Act
    nflverse_sources.load_weekly_player_stats(
        [2020], cache_dir=tmp_path, current_season=CURRENT, force_refresh=True
    )

    # Assert
    assert calls == [2020, 2020]


@pytest.mark.usefixtures("no_rosters")
def test_a_current_season_the_source_cannot_serve_falls_back_to_its_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(
        nflverse_sources.nfl,
        "load_player_stats",
        _fake({CURRENT: _player_stats(CURRENT, [{"passing_yards": 111}])}),
    )
    nflverse_sources.load_weekly_player_stats([CURRENT], cache_dir=tmp_path, current_season=CURRENT)
    monkeypatch.setattr(
        nflverse_sources.nfl, "load_player_stats", _failing(ConnectionError("404 Not Found"))
    )

    # Act
    stats = nflverse_sources.load_weekly_player_stats(
        [CURRENT], cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    assert stats.get_column("passing_yards").to_list() == [111]


@pytest.mark.parametrize(
    "error",
    [ConnectionError("404 Not Found"), ValueError("Season must be between 2012 and 2023")],
    ids=["not-published", "refused"],
)
def test_a_current_season_the_source_cannot_serve_without_a_cache_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: Exception
) -> None:
    # Arrange
    monkeypatch.setattr(nflverse_sources.nfl, "load_snap_counts", _failing(error))

    # Act
    snaps = nflverse_sources.load_snap_counts([CURRENT], cache_dir=tmp_path, current_season=CURRENT)

    # Assert
    assert snaps.is_empty()
    assert snaps.columns == constants.NFLREADPY_SNAP_COUNT_COLUMNS


def test_a_historical_download_failure_is_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(
        nflverse_sources.nfl, "load_snap_counts", _failing(ConnectionError("network down"))
    )

    # Act & Assert
    with pytest.raises(ConnectionError, match="network down"):
        nflverse_sources.load_snap_counts([2020], cache_dir=tmp_path, current_season=CURRENT)


def test_a_season_before_the_source_publishes_is_skipped_without_asking_nflreadpy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    calls: list[int] = []
    monkeypatch.setattr(
        nflverse_sources.nfl, "load_snap_counts", _failing(ValueError("too early"), calls)
    )
    early = constants.NFLREADPY_SNAP_COUNTS_MIN_SEASON - 1

    # Act
    snaps = nflverse_sources.load_snap_counts([early], cache_dir=tmp_path, current_season=CURRENT)

    # Assert
    assert calls == []
    assert snaps.is_empty()
    assert snaps.columns == constants.NFLREADPY_SNAP_COUNT_COLUMNS


@pytest.mark.parametrize(
    ("frame", "rows"),
    [(pl.DataFrame(), 0), (pl.DataFrame({"other": [1, 2, 3]}), 3)],
    ids=["no-columns", "only-other-columns"],
)
def test_conforming_a_frame_without_any_requested_column_keeps_its_rows(
    frame: pl.DataFrame, rows: int
) -> None:
    # Arrange
    schema = pl.Schema({"season": pl.Int64(), "team": pl.String()})

    # Act
    conformed = nflverse_sources.conform(frame, schema)

    # Assert
    assert conformed.schema == schema
    assert conformed.height == rows


@pytest.mark.parametrize("value", [0.5, -2.5], ids=["half", "negative-half"])
def test_conforming_a_fractional_float_into_an_integer_column_is_an_error(value: float) -> None:
    # Arrange
    schema = pl.Schema({"attempts": pl.Int64()})
    frame = pl.DataFrame({"attempts": [30.0, value]})

    # Act & Assert
    with pytest.raises(ValueError, match="attempts"):
        nflverse_sources.conform(frame, schema)


def test_conforming_whole_floats_into_an_integer_column_keeps_their_values() -> None:
    # Arrange
    schema = pl.Schema({"attempts": pl.Int64()})
    frame = pl.DataFrame({"attempts": [30.0, None, -2.0]})

    # Act
    conformed = nflverse_sources.conform(frame, schema)

    # Assert
    assert conformed.get_column("attempts").to_list() == [30, None, -2]
    assert conformed.schema == schema


def test_no_seasons_give_an_empty_frame_with_every_requested_column(tmp_path: Path) -> None:
    # Act
    rosters = nflverse_sources.load_rosters([], cache_dir=tmp_path, current_season=CURRENT)

    # Assert
    assert rosters.is_empty()
    assert rosters.columns == constants.NFLREADPY_ROSTER_COLUMNS


@pytest.mark.usefixtures("no_rosters")
def test_every_frame_carries_every_requested_column_with_a_fixed_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(
        nflverse_sources.nfl,
        "load_player_stats",
        _fake({2019: _player_stats(2019, [{}]), 2020: _player_stats(2020, [{}])}),
    )

    # Act
    stats = nflverse_sources.load_weekly_player_stats(
        [2019, 2020], cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    assert stats.columns == constants.NFLREADPY_PLAYER_STATS_COLUMNS
    assert stats.schema["season"] == pl.Int64
    assert stats.schema["passing_yards"] == pl.Int64
    assert stats.schema["passing_epa"] == pl.Float64
    assert stats.schema["def_sacks"] == pl.Float64
    assert stats.schema["player_id"] == pl.String
    assert stats.get_column("carries").to_list() == [None, None]


@pytest.mark.usefixtures("no_rosters")
def test_a_cache_that_lacks_a_requested_column_is_downloaded_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    stale = pl.DataFrame({"season": [2020], "week": [1], "passing_yards": [1]})
    stale.write_parquet(tmp_path / "player_stats_week_2020_reg.parquet")
    monkeypatch.setattr(
        nflverse_sources.nfl,
        "load_player_stats",
        _fake({2020: _player_stats(2020, [{"passing_yards": 222}])}),
    )

    # Act
    stats = nflverse_sources.load_weekly_player_stats(
        [2020], cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    assert stats.get_column("passing_yards").to_list() == [222]


@pytest.mark.usefixtures("no_rosters")
def test_regular_season_only_leaves_out_the_postseason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(
        nflverse_sources.nfl,
        "load_player_stats",
        _fake({2020: _player_stats(2020, [{"season_type": "REG"}, {"season_type": "POST"}])}),
    )

    # Act
    stats = nflverse_sources.load_weekly_player_stats(
        [2020], cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    assert stats.get_column("season_type").to_list() == ["REG"]


@pytest.mark.usefixtures("no_rosters")
def test_the_postseason_is_kept_on_request_in_its_own_cache_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(
        nflverse_sources.nfl,
        "load_player_stats",
        _fake({2020: _player_stats(2020, [{"season_type": "REG"}, {"season_type": "POST"}])}),
    )

    # Act
    stats = nflverse_sources.load_weekly_player_stats(
        [2020], cache_dir=tmp_path, current_season=CURRENT, regular_season_only=False
    )

    # Assert
    assert stats.get_column("season_type").to_list() == ["REG", "POST"]
    assert (tmp_path / "player_stats_week_2020_all.parquet").is_file()


@pytest.mark.usefixtures("no_rosters")
def test_player_stat_team_codes_are_canonical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(
        nflverse_sources.nfl,
        "load_player_stats",
        _fake({2020: _player_stats(2020, [{"team": "LA", "opponent_team": "WAS"}])}),
    )

    # Act
    stats = nflverse_sources.load_weekly_player_stats(
        [2020], cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    assert stats.select("team", "opponent_team").rows() == [("LAR", "WSH")]


@pytest.mark.usefixtures("no_rosters")
def test_stats_nflverse_never_charted_in_a_season_are_null(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    seasons = [2002, 2003, 2006, 2011, 2012]
    monkeypatch.setattr(
        nflverse_sources.nfl,
        "load_player_stats",
        _fake(
            {
                season: _player_stats(season, [{"def_tackles_for_loss": 1, "def_qb_hits": 2}])
                for season in seasons
            }
        ),
    )

    # Act
    stats = nflverse_sources.load_weekly_player_stats(
        seasons, cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    assert stats.select("season", "def_tackles_for_loss", "def_qb_hits").rows() == [
        (2002, 1, 2),
        (2003, None, None),
        (2006, None, 2),
        (2011, None, 2),
        (2012, 1, 2),
    ]


@pytest.mark.usefixtures("no_rosters")
def test_the_cache_keeps_the_stats_as_nflverse_published_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(
        nflverse_sources.nfl,
        "load_player_stats",
        _fake({2004: _player_stats(2004, [{"def_qb_hits": 0}])}),
    )

    # Act
    nflverse_sources.load_weekly_player_stats([2004], cache_dir=tmp_path, current_season=CURRENT)

    # Assert
    cached = pl.read_parquet(tmp_path / "player_stats_week_2004_reg.parquet")
    assert cached.get_column("def_qb_hits").to_list() == [0]


def _jacksonville_home_game(season: int) -> list[dict[str, object]]:
    """Return a game whose every player row names the visitor, as nflverse writes 2001's."""
    game = f"{season}_01_WAS_JAX"
    return [
        {"game_id": game, "player_id": f"{side}_{index}", "team": "WAS", "opponent_team": "JAX"}
        for side in ("was", "jax")
        for index in range(constants.ONE_TEAM_GAME_MIN_VALUES // 2)
    ]


def _roster_rows(season: int, players: list[tuple[str, str]]) -> pl.DataFrame:
    """Return raw season-roster rows of ``(gsis_id, team)``, as nflverse publishes them."""
    return pl.DataFrame(
        {
            "season": [season] * len(players),
            "team": [team for _player, team in players],
            "position": ["WR"] * len(players),
            "full_name": [player for player, _team in players],
            "gsis_id": [player for player, _team in players],
        }
    ).with_columns(pl.col("season").cast(pl.Int32))


def test_a_game_nflverse_credits_to_one_team_takes_its_teams_from_the_rosters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    half = constants.ONE_TEAM_GAME_MIN_VALUES // 2
    monkeypatch.setattr(
        nflverse_sources.nfl,
        "load_player_stats",
        _fake({2002: _player_stats(2002, _jacksonville_home_game(2002))}),
    )
    roster = [(f"was_{index}", "WAS") for index in range(half)] + [
        (f"jax_{index}", "JAX") for index in range(half)
    ]
    monkeypatch.setattr(
        nflverse_sources.nfl, "load_rosters", _fake({2002: _roster_rows(2002, roster)})
    )

    # Act
    stats = nflverse_sources.load_weekly_player_stats(
        [2002], cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    teams = stats.group_by("team", "opponent_team", maintain_order=True).len().sort("team")
    assert teams.rows() == [("JAX", "WSH", half), ("WSH", "JAX", half)]


def test_a_player_the_rosters_cannot_place_keeps_no_team_and_is_logged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    half = constants.ONE_TEAM_GAME_MIN_VALUES // 2
    monkeypatch.setattr(
        nflverse_sources.nfl,
        "load_player_stats",
        _fake({2002: _player_stats(2002, _jacksonville_home_game(2002))}),
    )
    roster = [(f"was_{index}", "WAS") for index in range(half)] + [
        (f"jax_{index}", "JAX") for index in range(1, half)
    ]
    monkeypatch.setattr(
        nflverse_sources.nfl, "load_rosters", _fake({2002: _roster_rows(2002, roster)})
    )

    # Act
    stats = nflverse_sources.load_weekly_player_stats(
        [2002], cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    unplaced = stats.filter(pl.col("team").is_null())
    assert unplaced.select("player_id", "opponent_team").rows() == [("jax_0", None)]
    assert "1 player row(s) in those games stay without a team" in caplog.text


def test_roster_team_codes_are_canonical_including_the_league_codes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    roster = _roster_rows(
        2002, [("a", "ARZ"), ("b", "BLT"), ("c", "CLV"), ("d", "HST"), ("e", "SL"), ("f", "OAK")]
    )
    monkeypatch.setattr(nflverse_sources.nfl, "load_rosters", _fake({2002: roster}))

    # Act
    rosters = nflverse_sources.load_rosters([2002], cache_dir=tmp_path, current_season=CURRENT)

    # Assert
    assert rosters.get_column("team").to_list() == ["ARI", "BAL", "CLE", "HOU", "LAR", "LV"]
    assert (tmp_path / "rosters_2002.parquet").is_file()


def test_weekly_rosters_keep_the_regular_season_and_start_in_2002(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    calls: list[int] = []
    weekly = _roster_rows(2002, [("a", "SL"), ("b", "SL")]).with_columns(
        pl.Series("game_type", ["REG", "WC"]), pl.Series("week", [17, 18], dtype=pl.Int32)
    )
    monkeypatch.setattr(nflverse_sources.nfl, "load_rosters_weekly", _fake({2002: weekly}, calls))

    # Act
    rosters = nflverse_sources.load_weekly_rosters(
        [2001, 2002], cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    assert calls == [2002]
    assert rosters.select("season", "week", "game_type", "team").rows() == [
        (2002, 17, "REG", "LAR")
    ]


def test_snap_counts_keep_the_regular_season_with_canonical_teams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    snaps = pl.DataFrame(
        {
            "game_id": ["2020_01_LA_DAL", "2020_19_LA_SEA"],
            "pfr_game_id": ["x", "y"],
            "season": [2020, 2020],
            "game_type": ["REG", "WC"],
            "week": [1, 19],
            "player": ["Jared Goff", "Jared Goff"],
            "pfr_player_id": ["GoffJa00", "GoffJa00"],
            "position": ["QB", "QB"],
            "team": ["LA", "LA"],
            "opponent": ["DAL", "SEA"],
            "offense_snaps": [70.0, 60.0],
            "offense_pct": [1.0, 1.0],
        }
    )
    monkeypatch.setattr(nflverse_sources.nfl, "load_snap_counts", _fake({2020: snaps}))

    # Act
    loaded = nflverse_sources.load_snap_counts([2020], cache_dir=tmp_path, current_season=CURRENT)

    # Assert
    assert loaded.columns == constants.NFLREADPY_SNAP_COUNT_COLUMNS
    assert loaded.select("week", "team", "opponent", "offense_snaps").rows() == [
        (1, "LAR", "DAL", 70.0)
    ]


def _players_file(position: str) -> pl.DataFrame:
    """Return a one-player players file listing him at ``position``."""
    return pl.DataFrame(
        {
            "gsis_id": ["00-0033357"],
            "display_name": ["Taysom Hill"],
            "short_name": ["T.Hill"],
            "position": [position],
            "pfr_id": ["HillTa00"],
            "espn_id": ["2468609"],
        }
    )


def _players_loader(position: str, calls: list[str]) -> Callable[[], pl.DataFrame]:
    """Return a fake players-file download that records each call."""

    def load() -> pl.DataFrame:
        calls.append(position)
        return _players_file(position)

    return load


def _failing_players() -> pl.DataFrame:
    """Simulate an outage of the players-file download."""
    msg = "players file unavailable"
    raise ConnectionError(msg)


def test_the_players_file_downloads_once_and_then_reads_its_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    calls: list[str] = []
    monkeypatch.setattr(nflverse_sources.nfl, "load_players", _players_loader("TE", calls))
    nflverse_sources.load_players(cache_dir=tmp_path)

    # Act
    players = nflverse_sources.load_players(cache_dir=tmp_path)

    # Assert
    assert calls == ["TE"]
    assert players.columns == constants.NFLREADPY_PLAYERS_COLUMNS
    assert (tmp_path / "players.parquet").is_file()


def test_force_refresh_downloads_the_players_file_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    calls: list[str] = []
    monkeypatch.setattr(nflverse_sources.nfl, "load_players", _players_loader("TE", calls))
    nflverse_sources.load_players(cache_dir=tmp_path)
    monkeypatch.setattr(nflverse_sources.nfl, "load_players", _players_loader("QB", calls))

    # Act
    players = nflverse_sources.load_players(cache_dir=tmp_path, force_refresh=True)

    # Assert
    assert calls == ["TE", "QB"]
    assert players.get_column("position").to_list() == ["QB"]


def test_a_failed_players_download_is_raised_unless_told_to_degrade(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(nflverse_sources.nfl, "load_players", _failing_players)

    # Act & Assert
    with pytest.raises(ConnectionError, match="players file unavailable"):
        nflverse_sources.load_players(cache_dir=tmp_path)


def test_a_degraded_players_download_falls_back_to_the_cached_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(nflverse_sources.nfl, "load_players", _players_loader("TE", []))
    nflverse_sources.load_players(cache_dir=tmp_path)
    monkeypatch.setattr(nflverse_sources.nfl, "load_players", _failing_players)

    # Act
    players = nflverse_sources.load_players(
        cache_dir=tmp_path, force_refresh=True, degrade_on_failure=True
    )

    # Assert
    assert players.get_column("gsis_id").to_list() == ["00-0033357"]


def test_a_degraded_players_download_without_a_cache_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(nflverse_sources.nfl, "load_players", _failing_players)

    # Act
    players = nflverse_sources.load_players(cache_dir=tmp_path, degrade_on_failure=True)

    # Assert
    assert players.is_empty()
    assert players.columns == constants.NFLREADPY_PLAYERS_COLUMNS


def _weekly_roster(season: int, position: str) -> pl.DataFrame:
    """Return a one-week regular-season roster listing Taysom Hill at ``position``."""
    return _roster_rows(season, [("00-0033357", "NO")]).with_columns(
        pl.lit("REG").alias("game_type"),
        pl.lit(1, dtype=pl.Int32).alias("week"),
        pl.lit(position).alias("position"),
        pl.lit("Taysom Hill").alias("full_name"),
        pl.lit("HillTa00").alias("pfr_id"),
    )


def test_the_qb_identity_counts_a_weekly_roster_listing_at_qb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(nflverse_sources.nfl, "load_players", _players_loader("TE", []))
    monkeypatch.setattr(
        nflverse_sources.nfl, "load_rosters_weekly", _fake({2020: _weekly_roster(2020, "QB")})
    )

    # Act
    identity = nflverse_sources.load_season_qb_identity(
        2020, cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    assert identity.rows() == [("00-0033357", "HillTa00", "Taysom Hill", "QB")]


def test_the_qb_identity_before_weekly_rosters_reads_the_players_file_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    calls: list[int] = []
    monkeypatch.setattr(nflverse_sources.nfl, "load_players", _players_loader("TE", []))
    monkeypatch.setattr(
        nflverse_sources.nfl, "load_rosters_weekly", _failing(ValueError("before 2002"), calls)
    )

    # Act
    identity = nflverse_sources.load_season_qb_identity(
        2001, cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    assert calls == []
    assert identity.get_column("qb_position").to_list() == ["TE"]


def test_the_current_season_qb_identity_reads_a_fresh_players_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    calls: list[str] = []
    monkeypatch.setattr(nflverse_sources.nfl, "load_players", _players_loader("TE", calls))
    nflverse_sources.load_players(cache_dir=tmp_path)
    monkeypatch.setattr(nflverse_sources.nfl, "load_players", _players_loader("QB", calls))
    monkeypatch.setattr(
        nflverse_sources.nfl,
        "load_rosters_weekly",
        _failing(ConnectionError("not published yet")),
    )

    # Act
    identity = nflverse_sources.load_season_qb_identity(
        CURRENT, cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    assert calls == ["TE", "QB"]
    assert identity.get_column("qb_position").to_list() == ["QB"]


def test_the_current_season_qb_identity_survives_a_players_file_outage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(nflverse_sources.nfl, "load_players", _failing_players)
    monkeypatch.setattr(
        nflverse_sources.nfl,
        "load_rosters_weekly",
        _fake({CURRENT: _weekly_roster(CURRENT, "QB")}),
    )

    # Act
    identity = nflverse_sources.load_season_qb_identity(
        CURRENT, cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    assert identity.rows() == [("00-0033357", "HillTa00", "Taysom Hill", "QB")]
