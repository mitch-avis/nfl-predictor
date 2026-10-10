"""The season statistics port against fixtures cut from nfl-sos-ratings' snapshot and files.

The fixtures (``tests/fixtures/sos_season_stats_parity/``, with their provenance and hashes) hold
the nflverse inputs of five games, as nflreadpy served them to that project's build, and the game
logs and win-probability bins it published for them; and, for 1999 and 2024, a column subset of
the published game logs with the published season tables built from them. Floats must agree
within ``FLOAT_TOLERANCE``; integers, text, flags, columns, and row order exactly. Washington is
``WSH`` here and ``WAS`` there.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import nflreadpy
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nfl_predictor import constants
from nfl_predictor.season_stats import game_logs, published_logs, sources
from nfl_predictor.season_stats import season as season_tables
from nfl_predictor.season_stats.metrics import get_registry
from nfl_predictor.season_stats.opponent_stats import compute_all_opponent_profiles
from nfl_predictor.season_stats.wp_bins import compute_qb_wp_bins, compute_team_wp_bins

if TYPE_CHECKING:
    from collections.abc import Callable

FIXTURES = Path(__file__).parent / "fixtures" / "sos_season_stats_parity"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
FLOAT_TOLERANCE = 1e-9
GAME_SEASONS = sorted(int(season) for season in MANIFEST["games"])
TABLE_SEASONS = (1999, 2024)
GAME_OUTPUTS = ("team_game_logs", "qb_game_logs", "team_wp_bins", "qb_wp_bins")
SEASON_TABLES = (
    "team_per_game_stats",
    "qb_per_game_stats",
    "qb_opponent_profiles",
    "opponent_profiles",
    "combined",
    "qb_combined",
)
TEAM_CODE_COLUMNS = ("team", "opponent_team")


def _fixture(name: str) -> pl.DataFrame:
    """Return one fixture file."""
    return pl.read_parquet(FIXTURES / name)


def _as_here(frame: pl.DataFrame) -> pl.DataFrame:
    """Return a published frame with Washington's code as this repository writes it."""
    present = [column for column in TEAM_CODE_COLUMNS if column in frame.columns]
    return frame.with_columns(pl.col(column).replace("WAS", "WSH") for column in present)


def _assert_matches(published: pl.DataFrame, rebuilt: pl.DataFrame) -> None:
    """Assert the same columns and rows in the same order, floats within the tolerance."""
    assert rebuilt.columns == published.columns
    assert_frame_equal(
        rebuilt,
        _as_here(published),
        check_dtypes=False,
        check_exact=False,
        rel_tol=0,
        abs_tol=FLOAT_TOLERANCE,
    )


def test_every_fixture_matches_its_recorded_hash() -> None:
    # Act
    hashes = {
        record["fixture"]: hashlib.sha256((FIXTURES / record["fixture"]).read_bytes()).hexdigest()
        for record in MANIFEST["files"]
    }

    # Assert
    assert hashes == {record["fixture"]: record["sha256"] for record in MANIFEST["files"]}


def _season_cut(name: str, season: int) -> pl.DataFrame:
    """Return the rows of a fixture holding every fixture game that belong to ``season``."""
    return _fixture(name).filter(pl.col("game_id").str.starts_with(f"{season}_"))


def _serve(frames: dict[str, pl.DataFrame]) -> Callable[..., pl.DataFrame]:
    """Return a stand-in for one nflreadpy loader that serves ``frames`` by season."""

    def load(seasons: list[int] | None = None, **_kwargs: object) -> pl.DataFrame:
        if seasons is None:
            return frames["all"]
        (season,) = seasons
        return frames[str(season)]

    return load


def _serve_snapshot_cut(monkeypatch: pytest.MonkeyPatch, season: int) -> None:
    """Make nflreadpy serve the season's fixture games as it served the snapshot."""

    def per_season(name: str) -> dict[str, pl.DataFrame]:
        path = FIXTURES / f"{season}_games_{name}.parquet"
        return {str(season): pl.read_parquet(path)} if path.exists() else {}

    schedule = _fixture("games_schedule.parquet").filter(pl.col("season") == season)
    monkeypatch.setattr(nflreadpy, "load_pbp", _serve(per_season("pbp")))
    monkeypatch.setattr(nflreadpy, "load_player_stats", _serve(per_season("player_stats")))
    monkeypatch.setattr(nflreadpy, "load_team_stats", _serve(per_season("team_stats")))
    monkeypatch.setattr(nflreadpy, "load_schedules", _serve({str(season): schedule}))
    monkeypatch.setattr(nflreadpy, "load_snap_counts", _serve(per_season("snap_counts")))
    monkeypatch.setattr(nflreadpy, "load_rosters_weekly", _serve(per_season("weekly_rosters")))
    monkeypatch.setattr(nflreadpy, "load_rosters", _serve(per_season("rosters")))
    players = _fixture(f"{season}_games_players.parquet")
    monkeypatch.setattr(nflreadpy, "load_players", _serve({"all": players}))
    # The play-by-play cache keeps only its listed columns; the build needs these.
    monkeypatch.setattr(constants, "PBP_COLUMNS", list(sources.SEASON_STATS_PBP_COLUMNS))


def _game_output(inputs: sources.SeasonInputs, kind: str) -> pl.DataFrame:
    """Return one published-file kind built from a season's inputs, in its file's row order."""
    games = season_tables.build_season_games(inputs)
    built = {
        "team_game_logs": lambda: game_logs.build_team_game_logs(games.team_games),
        "qb_game_logs": lambda: game_logs.build_qb_game_logs(games.qb_games, games.team_games),
        "team_wp_bins": lambda: compute_team_wp_bins(inputs.pbp),
        "qb_wp_bins": lambda: compute_qb_wp_bins(inputs.pbp),
    }[kind]()
    return season_tables.finalize_table(built, kind)


@pytest.mark.parametrize("kind", GAME_OUTPUTS)
@pytest.mark.parametrize("season", GAME_SEASONS)
def test_games_built_through_the_loaders_match_the_published_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, season: int, kind: str
) -> None:
    # Arrange
    _serve_snapshot_cut(monkeypatch, season)
    published = _season_cut(f"games_{kind}.parquet", season)
    inputs = sources.load_season_inputs(season, cache_dir=tmp_path, current_season=2027)

    # Act
    rebuilt = _game_output(inputs, kind)

    # Assert
    assert published.height
    _assert_matches(published, rebuilt)


def _games_from_published_logs(season: int) -> season_tables.SeasonGames:
    """Return a season's game rows from the published log fixtures and its played schedule."""
    schedule = sources.normalize_team_codes(
        _fixture(f"{season}_season_schedule.parquet"), ("home_team", "away_team")
    )
    return season_tables.SeasonGames(
        team_games=published_logs.team_games_from_published(
            _as_here(_fixture(f"{season}_season_team_game_logs.parquet"))
        ),
        qb_games=published_logs.qb_games_from_published(
            _as_here(_fixture(f"{season}_season_qb_game_logs.parquet"))
        ),
        schedule=season_tables.played_schedule(sources.prepare_schedule(schedule)),
    )


@pytest.mark.parametrize("kind", SEASON_TABLES)
@pytest.mark.parametrize("season", TABLE_SEASONS)
def test_season_tables_from_the_published_logs_match_the_published_tables(
    season: int, kind: str
) -> None:
    # Arrange
    games = _games_from_published_logs(season)
    ratings = _as_here(_fixture(f"{season}_season_ratings.parquet"))
    qb_ratings = _fixture(f"{season}_season_qb_ratings.parquet")

    # Act
    tables = season_tables.build_season_tables(
        games, pl.DataFrame(), team_ratings=ratings, qb_ratings=qb_ratings
    ).tables()

    # Assert
    _assert_matches(_fixture(f"{season}_season_{kind}.parquet"), tables[kind])


def test_the_season_without_a_games_play_by_play_still_counts_it_as_an_opponent() -> None:
    """1999 BAL at STL has a final score but no play-by-play, so no game rows on either side."""
    # Arrange
    games = _games_from_published_logs(1999)

    # Act
    profiles, details = compute_all_opponent_profiles(games.team_games, games.schedule)

    # Assert
    assert profiles is not None
    assert "1999_01_BAL_STL" not in games.team_games.get_column("game_id").to_list()
    assert "LAR" in [detail["opponent"] for detail in details["BAL"]]


def test_every_published_fixture_column_resolves_in_the_registry() -> None:
    # Arrange
    published = [
        record["fixture"]
        for record in MANIFEST["files"]
        if str(record["source"]).startswith("data/") or isinstance(record["source"], list)
    ]

    # Act
    unknown = {
        name: get_registry().validate_columns(pl.read_parquet_schema(FIXTURES / name))
        for name in published
    }

    # Assert
    assert published
    assert {name: columns for name, columns in unknown.items() if columns} == {}


def test_published_team_bins_add_up_to_the_published_game_logs() -> None:
    # Arrange
    logs = _fixture("games_team_game_logs.parquet")
    unit = pl.col("wp_unit")

    # Act
    summed = (
        _fixture("games_team_wp_bins.parquet")
        .group_by("game_id", "team")
        .agg(
            pl.col("wp_bin_plays").filter(unit == "scrimmage").sum().alias("offensive_snaps"),
            pl.col("wp_bin_epa").filter(unit == "scrimmage").sum().alias("offensive_epa"),
            pl.col("wp_bin_plays").filter(unit == "special_teams").sum().alias("st_plays"),
            pl.col("wp_bin_epa").filter(unit == "special_teams").sum().alias("st_epa"),
        )
    )

    # Assert
    columns = ["game_id", "team", "offensive_snaps", "offensive_epa", "st_plays", "st_epa"]
    assert_frame_equal(
        summed.select(columns).sort("game_id", "team"),
        logs.select(columns).sort("game_id", "team"),
        check_dtypes=False,
        abs_tol=FLOAT_TOLERANCE,
    )


def test_published_qb_bins_add_up_to_each_passers_dropbacks() -> None:
    # Arrange
    logs = _fixture("games_qb_game_logs.parquet")

    # Act
    summed = (
        _fixture("games_qb_wp_bins.parquet")
        .group_by("game_id", "qb_id")
        .agg(pl.col("qb_wp_bin_dropbacks").sum())
    )

    # Assert
    joined = logs.join(summed, on=["game_id", "qb_id"], how="left")
    assert joined.select(
        (pl.col("qb_dropbacks") == pl.col("qb_wp_bin_dropbacks").fill_null(0)).all()
    ).item()
