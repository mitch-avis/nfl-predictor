"""Tests for the ETL's season statistics: each season's tables from the frames the ETL holds."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import nflreadpy
import numpy as np
import polars as pl
import pytest
from nflreadpy.downloader import get_downloader
from polars.testing import assert_frame_equal
from tests.ratings_league import (
    BACKUP_ID,
    SEASONS,
    TEAMS,
    League,
    league,
    starter_id,
    with_player_sources,
    with_season_stats_columns,
)

from nfl_predictor import constants
from nfl_predictor.ratings.season import BootstrapSettings, build_qb_season
from nfl_predictor.season_stats import season as season_tables
from nfl_predictor.season_stats import sources as season_stats_sources
from nfl_predictor.utils import polars_utils, season_cache
from nfl_predictor.utils.polars import (
    loaders,
    nflverse_sources,
    ratings_build,
    season_stats_build,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

FIXTURES = Path(__file__).parent / "fixtures" / "sos_season_stats_parity"
_FAST = BootstrapSettings(resamples=20, seed=0)
_LAST = SEASONS[-1]
_SEASON = SEASONS[1]
_TEAM_RATING_COLUMNS = list(season_tables.TEAM_RATING_INPUT_COLUMNS[1:])
_QB_RATING_COLUMNS = list(season_tables.QB_RATING_INPUT_COLUMNS[1:])


def _league(seed: int = 0, *, player_sources: bool = True) -> League:
    """Return the synthetic league as the ETL holds it, with its player sources by default."""
    data = league(seed)
    return with_season_stats_columns(with_player_sources(data) if player_sources else data, seed)


def _loaded(data: League) -> ratings_build.LoadedSources:
    """Return the league as the ratings' loaded sources."""
    return ratings_build.LoadedSources(
        plays=data.plays,
        schedule=data.schedule,
        player_stats=data.player_stats,
        snap_counts=data.snap_counts,
        players=data.players,
        weekly_rosters=data.weekly_rosters,
        starter_identity=data.starter_identity,
    )


def _official(data: League) -> pl.DataFrame:
    """Return official weekly team stats for every team-game of the league, 200 passing yards each.

    Each team-game's passing yards are 200 plus its week, so the tables' totals show which
    season's rows they read.
    """
    games = data.plays.select("game_id", "season", "week", "posteam", "defteam").unique(
        maintain_order=True
    )
    return nflverse_sources.conform(
        games.select(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.lit("REG").alias("season_type"),
            "game_id",
            pl.col("posteam").alias("team"),
            pl.col("defteam").alias("opponent_team"),
            (pl.lit(200) + pl.col("week")).cast(pl.Int64).alias("passing_yards"),
        ),
        season_stats_sources.OFFICIAL_TEAM_STATS.schema,
    )


def _sources(
    data: League, official: pl.DataFrame | None = None
) -> season_stats_build.SeasonStatsSources:
    """Return the season statistics' sources of the league."""
    return season_stats_build.SeasonStatsSources(
        plays=data.plays,
        schedule=data.schedule,
        player_stats=data.player_stats,
        snap_counts=data.snap_counts,
        official_team_stats=_official(data) if official is None else official,
        identity=nflverse_sources.QbIdentitySources(data.players, data.weekly_rosters),
    )


def _season_stats(
    data: League | None = None, official: pl.DataFrame | None = None
) -> season_stats_build.SeasonStatsBuild:
    """Return the season statistics build of the league."""
    return season_stats_build.SeasonStatsBuild(_sources(data or _league(), official))


def _team_ratings() -> pl.DataFrame:
    """Return a team ratings table as the ratings build publishes it, one rating per team."""
    return pl.DataFrame(
        {
            "team": list(TEAMS),
            "games_played": [6] * len(TEAMS),
            **{column: [1.5, 0.5, -0.5, -1.5] for column in _TEAM_RATING_COLUMNS},
        }
    )


def _qb_ratings() -> pl.DataFrame:
    """Return every starter's rating and faced pass defense."""
    return pl.DataFrame(
        {
            "qb_id": [starter_id(team) for team in TEAMS],
            "adj_qb_epa_per_dropback": [0.2, 0.1, 0.0, -0.1],
            "qb_faced_pass_defense": [0.01, 0.0, -0.01, 0.02],
        }
    )


def _tables(build: season_stats_build.SeasonStatsBuild, season: int) -> dict[str, pl.DataFrame]:
    """Return a season's tables with the fixed ratings above."""
    return build.tables(season, team_ratings=_team_ratings(), qb_ratings=_qb_ratings())


def test_the_play_by_play_cache_keeps_every_column_the_season_statistics_read() -> None:
    # Act
    missing = [
        column
        for column in season_stats_sources.SEASON_STATS_PBP_COLUMNS
        if column not in constants.PBP_COLUMNS
    ]

    # Assert
    assert missing == []


def test_the_season_statistics_kinds_are_the_names_of_the_published_files() -> None:
    # Act
    kinds = season_stats_build.SEASON_STATS_KINDS

    # Assert
    assert kinds == (
        "team_game_logs",
        "qb_game_logs",
        "team_per_game_stats",
        "qb_per_game_stats",
        "qb_opponent_profiles",
        "opponent_profiles",
        "combined",
        "qb_combined",
        "team_wp_bins",
        "qb_wp_bins",
    )


def _refuse(name: str) -> Callable[..., pl.DataFrame]:
    """Return a loader that fails the test when called: its source is already loaded."""

    def load(*_args: object, **_kwargs: object) -> pl.DataFrame:
        msg = f"{name} was asked for a source the ETL already holds"
        raise AssertionError(msg)

    return load


@pytest.fixture
def official_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[int], dict[str, Any]]]:
    """Record every official team stats load, and refuse every source the ratings loaded."""
    calls: list[tuple[list[int], dict[str, Any]]] = []
    data = _league()

    def load_official(seasons: list[int], **kwargs: Any) -> pl.DataFrame:
        calls.append((list(seasons), kwargs))
        return _official(data)

    monkeypatch.setattr(season_stats_sources, "load_official_team_stats", load_official)
    for name in (
        "load_weekly_player_stats",
        "load_snap_counts",
        "load_players",
        "load_weekly_rosters",
        "load_rosters",
        "load_season_qb_identity",
    ):
        monkeypatch.setattr(nflverse_sources, name, _refuse(name))
    for module in (loaders, polars_utils):
        monkeypatch.setattr(module, "load_pbp", _refuse("load_pbp"))
        monkeypatch.setattr(module, "load_schedule", _refuse("load_schedule"))
    return calls


def test_sources_load_the_official_team_stats_of_the_runs_seasons_once(
    official_calls: list[tuple[list[int], dict[str, Any]]],
) -> None:
    # Act
    season_stats_build.load_season_stats_sources(
        [2000, 2001], _loaded(_league()), current_season=2001, force_refresh=True
    )

    # Assert
    assert official_calls == [([2000, 2001], {"force_refresh": True, "current_season": 2001})]


def test_sources_reuse_every_other_frame_the_ratings_loaded(
    official_calls: list[tuple[list[int], dict[str, Any]]],
) -> None:
    # Arrange
    data = _league()

    # Act
    sources = season_stats_build.load_season_stats_sources(
        [2000, 2001], _loaded(data), current_season=2001, force_refresh=False
    )

    # Assert
    assert_frame_equal(sources.plays, data.plays)
    assert_frame_equal(sources.schedule, data.schedule)
    assert_frame_equal(sources.player_stats, data.player_stats)
    assert_frame_equal(sources.snap_counts, data.snap_counts)
    assert_frame_equal(sources.identity.players, data.players)
    assert_frame_equal(sources.identity.weekly_rosters, data.weekly_rosters)


def test_sources_refuse_play_by_play_without_a_column_the_season_statistics_read(
    official_calls: list[tuple[list[int], dict[str, Any]]],
) -> None:
    # Arrange
    data = _league()
    loaded = replace(_loaded(data), plays=data.plays.drop("qb_scramble", "wp"))

    # Act & Assert
    with pytest.raises(ValueError, match="qb_scramble, wp"):
        season_stats_build.load_season_stats_sources(
            [2000], loaded, current_season=2001, force_refresh=False
        )


class _Response:
    """A downloaded file, as ``requests`` hands it to nflreadpy."""

    def __init__(self, content: bytes) -> None:
        self.content = content
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        """Report a successful download."""


@pytest.fixture
def empty_nflreadpy_cache() -> Generator[None]:
    """Start and end with nflreadpy's in-memory cache empty."""
    nflreadpy.clear_cache()
    yield
    nflreadpy.clear_cache()


@pytest.mark.usefixtures("empty_nflreadpy_cache")
def test_the_current_seasons_team_stats_file_is_downloaded_once_for_both_loaders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ETL's team-stat loader and the official team stats ask nflreadpy for one file."""
    # Arrange
    content = (FIXTURES / "2024_games_team_stats.parquet").read_bytes()
    downloads: list[str] = []

    def get(url: str, **_kwargs: object) -> _Response:
        downloads.append(url)
        return _Response(content)

    monkeypatch.setattr(get_downloader().session, "get", get)
    loaders.load_team_stats([2024], cache_dir=tmp_path, current_season=2024)

    # Act
    official = season_stats_sources.load_official_team_stats(
        [2024], cache_dir=tmp_path, current_season=2024
    )

    # Assert
    assert len(downloads) == 1
    assert official.height == 4


def test_tables_of_a_played_season_hold_every_kind() -> None:
    # Act
    tables = _tables(_season_stats(), _SEASON)

    # Assert
    assert tuple(tables) == season_stats_build.SEASON_STATS_KINDS


def _serve_loaders(monkeypatch: pytest.MonkeyPatch, data: League, official: pl.DataFrame) -> None:
    """Point every loader the season statistics' own loader reads at the league's frames."""

    def by_season(frame: pl.DataFrame) -> Callable[..., pl.DataFrame]:
        def load(seasons: list[int], **_kwargs: object) -> pl.DataFrame:
            return frame.filter(pl.col("season").is_in(list(seasons)))

        return load

    def identity(season: int, **_kwargs: object) -> pl.DataFrame:
        return nflverse_sources.QbIdentitySources(
            data.players, data.weekly_rosters
        ).season_identity(season)

    monkeypatch.setattr(loaders, "load_pbp", by_season(data.plays))
    monkeypatch.setattr(loaders, "load_schedule", by_season(data.schedule))
    monkeypatch.setattr(nflverse_sources, "load_weekly_player_stats", by_season(data.player_stats))
    monkeypatch.setattr(nflverse_sources, "load_snap_counts", by_season(data.snap_counts))
    monkeypatch.setattr(season_stats_sources, "load_official_team_stats", by_season(official))
    monkeypatch.setattr(nflverse_sources, "load_season_qb_identity", identity)
    monkeypatch.setattr(nflverse_sources, "load_rosters", _refuse("load_rosters"))


@pytest.mark.parametrize("season", SEASONS)
def test_tables_equal_those_built_from_the_season_statistics_own_loader(
    monkeypatch: pytest.MonkeyPatch, season: int
) -> None:
    """The step feeds the builders what ``season_stats.sources.load_season_inputs`` would."""
    # Arrange
    data = _league()
    official = _official(data)
    _serve_loaders(monkeypatch, data, official)
    inputs = season_stats_sources.load_season_inputs(season)
    expected = season_tables.build_season_tables(
        season_tables.build_season_games(inputs),
        inputs.pbp,
        team_ratings=_team_ratings(),
        qb_ratings=_qb_ratings(),
    ).tables()

    # Act
    tables = _tables(_season_stats(data, official), season)

    # Assert
    assert list(tables) == list(expected)
    for kind, table in tables.items():
        assert_frame_equal(table, expected[kind], check_exact=True)


def test_tables_read_the_official_team_stats_of_their_season() -> None:
    # Arrange
    build = _season_stats()

    # Act
    logs = _tables(build, _SEASON)["team_game_logs"]

    # Assert
    assert logs.get_column("passing_yards").to_list() == [
        200 + week for week in logs.get_column("week").to_list()
    ]


def test_a_season_without_a_game_has_no_tables(caplog: pytest.LogCaptureFixture) -> None:
    # Arrange
    data = _league()
    pre_kickoff = replace(data, plays=data.plays.filter(pl.col("season") != _LAST))
    build = _season_stats(pre_kickoff)

    # Act
    with caplog.at_level(logging.INFO):
        tables = _tables(build, _LAST)

    # Assert
    assert tables == {}
    assert f"season {_LAST} has no game played yet" in caplog.text


@pytest.mark.parametrize(
    ("source", "named"),
    [
        ("official_team_stats", "official weekly team stats"),
        ("player_stats", "weekly player stats"),
    ],
)
def test_a_played_season_without_a_source_logs_what_its_tables_lack(
    caplog: pytest.LogCaptureFixture, source: str, named: str
) -> None:
    # Arrange
    sources = _sources(_league())
    frame: pl.DataFrame = getattr(sources, source)
    build = season_stats_build.SeasonStatsBuild(
        replace(sources, **{source: frame.filter(pl.col("season") != _SEASON)})
    )

    # Act
    with caplog.at_level(logging.WARNING):
        tables = _tables(build, _SEASON)

    # Assert
    assert tables
    assert f"season {_SEASON} has no {named}" in caplog.text


def test_a_season_whose_snap_counts_nflverse_publishes_logs_their_absence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    season = season_stats_build.SNAP_COUNTS_FIRST_SEASON
    inputs = season_stats_sources.SeasonInputs(
        season=season,
        pbp=pl.DataFrame({"game_id": ["g1"]}),
        player_stats=pl.DataFrame({"game_id": ["g1"]}),
        schedule=pl.DataFrame(),
        official_team_stats=pl.DataFrame({"game_id": ["g1"]}),
        snap_counts=pl.DataFrame(),
        qb_identity=pl.DataFrame(),
    )

    # Act
    with caplog.at_level(logging.WARNING):
        season_stats_build.log_missing_sources(inputs)

    # Assert
    assert f"season {season} has no snap counts" in caplog.text


def test_a_season_before_nflverse_has_snap_counts_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    inputs = season_stats_sources.SeasonInputs(
        season=season_stats_build.SNAP_COUNTS_FIRST_SEASON - 1,
        pbp=pl.DataFrame({"game_id": ["g1"]}),
        player_stats=pl.DataFrame({"game_id": ["g1"]}),
        schedule=pl.DataFrame(),
        official_team_stats=pl.DataFrame({"game_id": ["g1"]}),
        snap_counts=pl.DataFrame(),
        qb_identity=pl.DataFrame(),
    )

    # Act
    with caplog.at_level(logging.WARNING):
        season_stats_build.log_missing_sources(inputs)

    # Assert
    assert caplog.text == ""


def test_tables_without_ratings_leave_the_rating_columns_null() -> None:
    # Act
    tables = _season_stats().tables(_SEASON, team_ratings=None, qb_ratings=None)

    # Assert
    assert tables["combined"].select(_TEAM_RATING_COLUMNS).null_count().row(0) == (4,) * 6
    qb_combined = tables["qb_combined"]
    assert qb_combined.select(_QB_RATING_COLUMNS).null_count().row(0) == (qb_combined.height,) * 2


def test_key_frames_hold_the_seasons_game_rows_schedule_and_plays() -> None:
    # Act
    frames = _season_stats().key_frames(_SEASON)

    # Assert
    assert sorted(frames) == [
        "season_stats_plays",
        "season_stats_qb_games",
        "season_stats_schedule",
        "season_stats_team_games",
    ]
    assert frames["season_stats_plays"].get_column("season").unique().to_list() == [_SEASON]
    assert frames["season_stats_team_games"].get_column("season").unique().to_list() == [_SEASON]
    assert frames["season_stats_schedule"].height == 12


def test_tables_after_the_key_frames_equal_tables_without_them() -> None:
    # Arrange
    expected = _tables(_season_stats(), _SEASON)
    build = _season_stats()
    build.key_frames(_SEASON)

    # Act
    tables = _tables(build, _SEASON)

    # Assert
    for kind, table in tables.items():
        assert_frame_equal(table, expected[kind], check_exact=True)


def _key_digests(build: season_stats_build.SeasonStatsBuild, season: int) -> dict[str, str]:
    """Return the season cache's value digest of each key frame."""
    frames = build.key_frames(season)
    return {name: season_cache.frame_digest(frame) for name, frame in frames.items()}


def test_key_frames_ignore_a_players_file_row_no_source_of_the_season_names() -> None:
    """The players file is downloaded again every in-season run, with new players each week."""
    # Arrange
    data = _league()
    signed = pl.DataFrame(
        {"gsis_id": ["00-0099999"], "display_name": ["New Signing"], "position": ["QB"]}
    )
    newer = replace(data, players=pl.concat([data.players, signed], how="diagonal"))
    expected = _key_digests(_season_stats(data), _SEASON)

    # Act
    digests = _key_digests(_season_stats(newer), _SEASON)

    # Assert
    assert digests == expected


def test_key_frames_change_when_the_players_file_renames_a_passer_of_the_season() -> None:
    # Arrange
    data = _league()
    renamed = data.players.with_columns(
        pl.when(pl.col("gsis_id") == starter_id("AAA"))
        .then(pl.lit("Renamed Starter"))
        .otherwise(pl.col("display_name"))
        .alias("display_name")
    )
    expected = _key_digests(_season_stats(data), _SEASON)

    # Act
    digests = _key_digests(_season_stats(replace(data, players=renamed)), _SEASON)

    # Assert
    assert digests["season_stats_qb_games"] != expected["season_stats_qb_games"]


# The ratings build: the season statistics beside each season's fitted ratings.
def _ratings(data: League, *, current_season: int = _LAST + 1) -> ratings_build.RatingsBuild:
    """Return the league's ratings build with the season statistics, every season built."""
    build = ratings_build.RatingsBuild(
        ratings_build.ratings_inputs(_loaded(data)),
        current_season=current_season,
        bootstrap=_FAST,
        season_stats=_season_stats(data),
    )
    for season in SEASONS:
        build.build(season)
    return build


@pytest.fixture(scope="module")
def built() -> ratings_build.RatingsBuild:
    """Return the league's ratings build with the season statistics (shared: it fits ratings)."""
    return _ratings(_league())


def test_build_adds_each_seasons_statistics_tables_to_its_frames(
    built: ratings_build.RatingsBuild,
) -> None:
    # Act
    frames = built.build_frames(_SEASON)

    # Assert
    assert set(season_stats_build.SEASON_STATS_KINDS) <= set(frames)
    assert set(ratings_build.PUBLISHED_KINDS) <= set(frames)


def test_published_tables_name_the_season_statistics_tables_by_season_and_kind(
    built: ratings_build.RatingsBuild,
) -> None:
    # Act
    tables = built.published_tables()

    # Assert
    assert sorted(tables) == sorted(
        f"{season}_{kind}"
        for season in SEASONS
        for kind in (*ratings_build.PUBLISHED_KINDS, *season_stats_build.SEASON_STATS_KINDS)
    )


def test_combined_takes_the_rating_columns_of_the_seasons_team_ratings(
    built: ratings_build.RatingsBuild,
) -> None:
    # Arrange
    ratings = built.published_tables()[f"{_SEASON}_ratings"]

    # Act
    combined = built.published_tables()[f"{_SEASON}_combined"]

    # Assert
    columns = ["team", *_TEAM_RATING_COLUMNS]
    assert_frame_equal(
        combined.select(columns).sort("team"),
        ratings.select(columns).sort("team"),
        check_exact=True,
    )


def test_qb_combined_takes_the_rating_columns_of_the_seasons_quarterback_fit(
    built: ratings_build.RatingsBuild,
) -> None:
    """Every passer the fit rates, the backup who never qualifies for the rankings included."""
    # Arrange
    fit = build_qb_season(
        built._history.qb_games(_LAST), built._season_team_games(_LAST), bootstrap=_FAST
    ).qb_season
    expected = fit.select("qb_id", *_QB_RATING_COLUMNS).sort("qb_id")

    # Act
    qb_combined = built.published_tables()[f"{_LAST}_qb_combined"]

    # Assert
    rated = qb_combined.filter(pl.col("qb_id").is_in(expected.get_column("qb_id").implode()))
    assert_frame_equal(
        rated.select("qb_id", *_QB_RATING_COLUMNS).sort("qb_id"), expected, check_exact=True
    )
    backup = rated.filter(pl.col("qb_id") == BACKUP_ID)
    assert backup.get_column("qb_is_eligible").to_list() == [False]
    assert backup.get_column("adj_qb_epa_per_dropback").is_not_null().all()


def test_a_season_without_a_qualified_passer_leaves_the_passer_ratings_null(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Without player stats a passer's attempts come from his plays, three a game here."""
    # Arrange
    data = _league(player_sources=False)
    first_attempts = pl.int_range(pl.len()).over("game_id", "posteam") < 3
    sparse = replace(
        data,
        plays=data.plays.filter(
            (pl.col("season") != _SEASON) | (pl.col("qb_dropback") == 0.0) | first_attempts
        ),
    )
    build = ratings_build.RatingsBuild(
        ratings_build.ratings_inputs(_loaded(sparse)),
        current_season=_LAST + 1,
        bootstrap=_FAST,
        season_stats=_season_stats(sparse),
    )

    # Act
    with caplog.at_level(logging.INFO):
        frames = build.build(_SEASON)

    # Assert
    assert "qb_ratings" not in frames
    qb_combined = frames["qb_combined"]
    assert qb_combined.select(_QB_RATING_COLUMNS).null_count().row(0) == (qb_combined.height,) * 2
    assert f"season {_SEASON} has no quarterback ratings" in caplog.text


def test_season_key_frames_are_the_season_statistics_key_frames(
    built: ratings_build.RatingsBuild,
) -> None:
    # Arrange
    expected = _season_stats().key_frames(_SEASON)

    # Act
    frames = built.season_key_frames(_SEASON)

    # Assert
    assert list(frames) == list(expected)
    for name, frame in frames.items():
        assert_frame_equal(frame, expected[name], check_exact=True)


def test_season_key_frames_are_empty_without_season_statistics() -> None:
    # Arrange
    build = ratings_build.RatingsBuild(
        ratings_build.ratings_inputs(_loaded(_league())), current_season=_LAST, bootstrap=_FAST
    )

    # Act
    frames = build.season_key_frames(_SEASON)

    # Assert
    assert frames == {}


def test_reused_season_frames_publish_the_same_season_statistics_tables(
    built: ratings_build.RatingsBuild,
) -> None:
    # Arrange
    reused = ratings_build.RatingsBuild(
        ratings_build.ratings_inputs(_loaded(_league())),
        current_season=_LAST + 1,
        bootstrap=_FAST,
        season_stats=_season_stats(),
    )
    for season in SEASONS:
        reused.reuse(season, built.build_frames(season))

    # Act
    tables = reused.published_tables()

    # Assert
    assert tables.keys() == built.published_tables().keys()
    for name, table in tables.items():
        assert_frame_equal(table, built.published_tables()[name], check_exact=True)


# Determinism: two runs, the second fed its inputs in another order, write the same bytes.
def _reordered(data: League, seed: int) -> League:
    """Return the league with whole games, and every other source's rows, in another order.

    Each game's plays keep their order: the sums over a game's plays follow nflverse's play order,
    which the play-by-play cache keeps, so only whole games move.
    """
    rng = np.random.default_rng(seed)
    games = data.plays.get_column("game_id").unique(maintain_order=True).to_list()
    order = {game: position for position, game in enumerate(rng.permutation(games).tolist())}
    plays = (
        data.plays.with_row_index("_play")
        .with_columns(pl.col("game_id").replace_strict(order, return_dtype=pl.Int64).alias("_game"))
        .sort("_game", "_play")
        .drop("_game", "_play")
    )

    def shuffled(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.sample(fraction=1.0, shuffle=True, seed=seed)

    return replace(
        data,
        plays=plays,
        schedule=shuffled(data.schedule),
        player_stats=shuffled(data.player_stats),
        snap_counts=shuffled(data.snap_counts),
        weekly_rosters=shuffled(data.weekly_rosters),
        players=shuffled(data.players),
        starter_identity=shuffled(data.starter_identity),
    )


def test_two_runs_write_byte_identical_season_statistics_tables(
    built: ratings_build.RatingsBuild, tmp_path: Path
) -> None:
    # Arrange
    second = _ratings(_reordered(_league(), seed=3))
    ratings_build.write_published_tables(built.published_tables(), tmp_path / "first")

    # Act
    ratings_build.write_published_tables(second.published_tables(), tmp_path / "second")

    # Assert
    names = sorted(path.name for path in (tmp_path / "first").iterdir())
    assert names == sorted(path.name for path in (tmp_path / "second").iterdir())
    assert any(name.endswith("_qb_combined.parquet") for name in names)
    differing = [
        name
        for name in names
        if (tmp_path / "first" / name).read_bytes() != (tmp_path / "second" / name).read_bytes()
    ]
    assert differing == []
