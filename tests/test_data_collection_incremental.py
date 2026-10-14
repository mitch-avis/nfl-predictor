"""Characterization tests: the incremental ETL writes exactly what the full rebuild writes.

A small three-season world (two finished seasons and the current one) runs through the real
season and week builders; only the upstream loaders, the play-by-play quarterback join and
the SurvivorGrid scrape are replaced, and the ratings and the season statistics read the world's
own play-by-play. Each test compares every written CSV, ratings table and season statistics table
byte for byte with a full rebuild of the same inputs, and the combined frame exactly, after the
cache was filled, after a cached input changed, after the code fingerprint changed and after an
entry was damaged.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import shutil
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from tests.ratings_league import (
    plays_for,
    rewritten_plays,
    starter_id,
    with_season_stats_play_columns,
)

from nfl_predictor import constants, data_collection
from nfl_predictor.ratings.season import BootstrapSettings
from nfl_predictor.season_stats.sources import OFFICIAL_TEAM_STATS
from nfl_predictor.utils import clock, polars_utils, season_cache
from nfl_predictor.utils.polars import nflverse_sources, ratings_build, season_stats_build

if TYPE_CHECKING:
    from pathlib import Path

_TEAMS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
_PRIOR_SEASON = 2018
_SEASONS = (2019, 2020, 2021)
_CURRENT_SEASON = 2021
_CURRENT_WEEK = 3
_WEEKS = 3
_CACHED_SEASONS = (2019, 2020)


def _slate(season: int, week: int) -> list[tuple[str, str]]:
    """Return the week's three games as (away, home): a round robin, one round per week."""
    rest = list(_TEAMS[1:])
    shift = (season + week) % len(rest)
    order = [_TEAMS[0], *rest[shift:], *rest[:shift]]
    pairs = [(order[0], order[5]), (order[1], order[4]), (order[2], order[3])]
    return pairs if week % 2 else [(home, away) for away, home in pairs]


def _played(season: int, week: int, played_through: int) -> bool:
    return season < _CURRENT_SEASON or week <= played_through


@dataclass
class _World:
    """Every upstream input of one ETL run."""

    schedule: pl.DataFrame
    team_stats: pl.DataFrame
    elo: pl.DataFrame
    # Every played game's plays, the prior season's included, with every column the season
    # statistics read, and every season's schedule.
    plays: pl.DataFrame
    ratings_schedule: pl.DataFrame
    # The players file: each team's starter.
    players: pl.DataFrame
    rankings: dict[int, pl.DataFrame] = field(default_factory=dict)


def _world(seed: int = 7, *, played_through: int = _CURRENT_WEEK - 1) -> _World:
    """Build the world; ``played_through`` is the current season's last played week."""
    rng = np.random.default_rng(seed)
    ranking_rng = np.random.default_rng(seed + 1)
    games: list[dict[str, object]] = []
    team_games: list[dict[str, object]] = []
    elo_rows: list[dict[str, object]] = []
    for season in (_PRIOR_SEASON, *_SEASONS):
        for week in range(1, _WEEKS + 1):
            for index, (away, home) in enumerate(_slate(season, week)):
                played = _played(season, week, played_through)
                away_points = int(rng.integers(3, 38))
                home_points = int(rng.integers(3, 38))
                game_id = f"{season}_{week:02d}_{away}_{home}"
                games.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "game_type": "REG",
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
                elo_rows.append(
                    {
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
                )
                if not played:
                    continue
                for team, opponent, is_home, scored, allowed in (
                    (away, home, False, away_points, home_points),
                    (home, away, True, home_points, away_points),
                ):
                    team_games.append(
                        {
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
                    )
    schedule = pl.DataFrame(games)
    rankings = {
        season: pl.DataFrame(
            [
                {
                    "team_abbr": team,
                    "week": week,
                    **{
                        column: float(ranking_rng.normal(0.0, 5.0))
                        for column in constants.TR_RATINGS
                    },
                }
                for week in range(1, _WEEKS + 1)
                for team in _TEAMS
            ]
        )
        for season in (_PRIOR_SEASON, *_SEASONS)
    }
    played = schedule.filter(pl.col("away_score").is_not_null())
    return _World(
        schedule=schedule.filter(pl.col("season") >= _SEASONS[0]),
        team_stats=pl.DataFrame(team_games),
        elo=pl.DataFrame(elo_rows).filter(pl.col("season") >= _SEASONS[0]),
        plays=with_season_stats_play_columns(plays_for(played, seed), seed),
        ratings_schedule=schedule,
        players=pl.DataFrame(
            {
                "gsis_id": [starter_id(team) for team in _TEAMS],
                "position": ["QB"] * len(_TEAMS),
                "display_name": [None] * len(_TEAMS),
                "pfr_id": [None] * len(_TEAMS),
            },
            schema_overrides={"display_name": pl.String, "pfr_id": pl.String},
        ),
        rankings=rankings,
    )


# Twenty resamples keep the rank ranges quick; the published thousand change nothing tested here.
_FAST_BOOTSTRAP = BootstrapSettings(resamples=20, seed=0)


def _ratings(world: _World) -> ratings_build.RatingsBuild:
    """Return the ratings build of the world, with its season statistics.

    Both read the world's plays, schedule and starters; the world has no player stats, snap
    counts or official team stats, as the current season has none before nflverse publishes them.
    """
    sources = ratings_build.LoadedSources(
        plays=world.plays,
        schedule=world.ratings_schedule,
        player_stats=nflverse_sources.PLAYER_STATS.empty(),
        snap_counts=nflverse_sources.SNAP_COUNTS.empty(),
        players=world.players,
        weekly_rosters=pl.DataFrame(
            schema={
                "season": pl.Int64(),
                "week": pl.Int64(),
                "game_type": pl.String(),
                "gsis_id": pl.String(),
                "position": pl.String(),
                "full_name": pl.String(),
                "pfr_id": pl.String(),
            }
        ),
        starter_identity=pl.DataFrame(
            {
                "qb_name": [f"QB {team}" for team in _TEAMS],
                "qb_id": [starter_id(team) for team in _TEAMS],
            }
        ),
    )
    season_stats = season_stats_build.SeasonStatsBuild(
        season_stats_build.SeasonStatsSources(
            plays=sources.plays,
            schedule=sources.schedule,
            player_stats=sources.player_stats,
            snap_counts=sources.snap_counts,
            official_team_stats=OFFICIAL_TEAM_STATS.empty(),
            identity=nflverse_sources.QbIdentitySources(sources.players, sources.weekly_rosters),
        )
    )
    return ratings_build.RatingsBuild(
        ratings_build.ratings_inputs(sources),
        current_season=_CURRENT_SEASON,
        first_season=_PRIOR_SEASON,
        bootstrap=_FAST_BOOTSTRAP,
        season_stats=season_stats,
    )


@dataclass
class _Run:
    """The world one ETL run reads, the seasons it built and what it returned."""

    world: _World
    built: list[int] = field(default_factory=list)
    rated: list[int] = field(default_factory=list)
    games: pl.DataFrame | None = None
    snapshots: pl.DataFrame | None = None


def _split_in_two(frame: pl.DataFrame) -> pl.DataFrame:
    """Hold the frame as two chunks split at its midpoint, as a large loaded frame can be.

    The split point moves with the frame's length, which is what lets later seasons' rows
    reach the last bits of an eager reduction over an earlier season's slice.
    """
    middle = frame.height // 2
    return pl.concat([frame.slice(0, middle), frame.slice(middle)], rechunk=False)


def _install(monkeypatch: pytest.MonkeyPatch, run: _Run) -> None:
    """Replace the loaders and the network steps, and record what each run builds."""

    def load_sources(_seasons: list[int], _config: object) -> object:
        world = run.world
        return data_collection._EtlSources(
            current_season=_CURRENT_SEASON,
            current_week=_CURRENT_WEEK,
            schedule_df=world.schedule,
            team_stats_df=_split_in_two(world.team_stats),
            pbp_df=pl.DataFrame(),
            elo_df=world.elo,
            raw_elo_df=world.elo,
            ratings=_ratings(world),
        )

    build_ratings = ratings_build.RatingsBuild.build

    def counting_build_ratings(
        self: ratings_build.RatingsBuild, season: int
    ) -> dict[str, pl.DataFrame]:
        run.rated.append(season)
        return build_ratings(self, season)

    process_season = data_collection.process_season
    collect_all_data = data_collection.collect_all_data

    def counting_process_season(
        season: int,
        schedule_df: pl.DataFrame,
        team_stats_df: pl.DataFrame,
        inputs: data_collection.SeasonInputs,
    ) -> pl.DataFrame:
        run.built.append(season)
        return process_season(season, schedule_df, team_stats_df, inputs)

    def recording_collect_all_data(
        seasons: list[int],
        *,
        config: data_collection.DataCollectionConfig,
        records: data_collection.CollectionRecords,
    ) -> pl.DataFrame:
        run.games = collect_all_data(seasons, config=config, records=records)
        run.snapshots = data_collection.combine_strength_snapshots(records.strength_snapshots)
        return run.games

    monkeypatch.setattr(data_collection, "_load_sources", load_sources)
    monkeypatch.setattr(data_collection, "process_season", counting_process_season)
    monkeypatch.setattr(ratings_build.RatingsBuild, "build", counting_build_ratings)
    monkeypatch.setattr(data_collection, "collect_all_data", recording_collect_all_data)
    monkeypatch.setattr(
        polars_utils,
        "load_team_rankings",
        lambda season, *_args, **_kwargs: run.world.rankings[season],
    )
    # The world has no play-by-play: no quarterback history, so no quarterback join.
    monkeypatch.setattr(data_collection, "_quarterback_history", lambda *_a, **_k: None)
    monkeypatch.setattr(data_collection, "_attach_qb_features", lambda games, *_a, **_k: games)
    monkeypatch.setattr(data_collection.game_utils, "fill_future_game_lines", lambda df: df)
    monkeypatch.setattr(clock, "local_today", lambda: date(_CURRENT_SEASON, 9, 22))
    monkeypatch.setattr(clock, "nfl_week", lambda _today: _CURRENT_WEEK)


# The fixture stubs the stored-line fills; the pick-time source would download nfelo's lines.
_ARGS = (
    "--min-season",
    str(_SEASONS[0]),
    "--max-season",
    str(_SEASONS[-1]),
    "--line-source",
    constants.LINE_SOURCE_STORED,
)


@dataclass(frozen=True)
class _Result:
    """Every file one run wrote, the combined frame, the snapshot file and the seasons built."""

    files: dict[str, bytes]
    games: pl.DataFrame
    snapshots: pl.DataFrame
    built: list[int]
    rated: list[int]


def _main(run: _Run, data_dir: Path, *extra: str) -> _Result:
    run.built = []
    run.rated = []
    data_collection.main([*_ARGS, "--data-dir", str(data_dir), *extra])
    assert run.games is not None
    assert run.snapshots is not None
    files = {
        path.relative_to(data_dir).as_posix(): path.read_bytes()
        for path in sorted(data_dir.rglob("*"))
        if path.suffix in {".csv", ".parquet"} and "cache" not in path.relative_to(data_dir).parts
    }
    return _Result(files, run.games, run.snapshots, list(run.built), list(run.rated))


def _assert_identical(incremental: _Result, full: _Result) -> None:
    assert_frame_equal(incremental.games, full.games, check_exact=True)
    assert_frame_equal(incremental.snapshots, full.snapshots, check_exact=True)
    assert sorted(incremental.files) == sorted(full.files)
    csv = [name for name in full.files if name.endswith(".csv")]
    assert len(csv) == 6, "five datasets and one games-to-predict file"
    ratings = [name for name in full.files if name.startswith(f"{constants.RATINGS_DIRNAME}/")]
    assert len(ratings) == len(_SEASONS) * (
        len(ratings_build.PUBLISHED_KINDS) + len(season_stats_build.SEASON_STATS_KINDS)
    ) + len(ratings_build.IN_PROGRESS_KINDS), (
        "every season's ratings and season statistics tables and the current season's weekly ranges"
    )
    for name, content in full.files.items():
        assert incremental.files[name] == content, f"{name} differs from the full rebuild"


def _cache_dir(data_dir: Path) -> Path:
    return data_dir / "cache" / constants.ETL_SEASON_CACHE_DIRNAME


def _stamps(directory: Path) -> dict[Path, int]:
    return {path: path.stat().st_mtime_ns for path in sorted(directory.rglob("*"))}


@dataclass(frozen=True)
class _Baseline:
    """A cold incremental run of the unchanged world, then a full rebuild in the same place.

    The cache's file stamps are kept from before and after the full rebuild.
    """

    cold: _Result
    full: _Result
    cold_dir: Path
    stamps_before_full: dict[Path, int]
    stamps_after_full: dict[Path, int]


@pytest.fixture(scope="module")
def baseline(tmp_path_factory: pytest.TempPathFactory) -> _Baseline:
    """Run the unchanged world once per module: the runs every test compares against."""
    run = _Run(world=_world())
    data_dir = tmp_path_factory.mktemp("baseline")
    with pytest.MonkeyPatch.context() as monkeypatch:
        _install(monkeypatch, run)
        cold = _main(run, data_dir, "--incremental")
        before = _stamps(_cache_dir(data_dir))
        full = _main(run, data_dir)
        after = _stamps(_cache_dir(data_dir))
    return _Baseline(cold, full, data_dir, before, after)


@pytest.fixture
def etl(monkeypatch: pytest.MonkeyPatch) -> _Run:
    """Install a fresh copy of the world for one test."""
    run = _Run(world=_world())
    _install(monkeypatch, run)
    return run


@pytest.fixture
def warm_dir(baseline: _Baseline, tmp_path: Path) -> Path:
    """Return a data directory holding a copy of the baseline's filled season cache."""
    shutil.copytree(_cache_dir(baseline.cold_dir), _cache_dir(tmp_path))
    return tmp_path


def test_the_world_exercises_every_season_feature_family(baseline: _Baseline) -> None:
    """Guard the fixture: the compared rows carry real values, not all-null columns."""
    # Act
    finished = baseline.full.games.filter(pl.col("season") < _CURRENT_SEASON)

    # Assert
    for column in (
        "away_adj_strength_composite",
        "home_sos_played_raw",
        "away_elo_4wk_trend",
        "home_coach_win_pct_prior",
        "away_predictive_rating",
        "home_pass_yards",
        "away_wins",
        "away_ratings_team",
        "home_ratings_sos",
        "away_ratings_qb",
        "home_ratings_qb_dropbacks",
    ):
        assert finished.get_column(column).drop_nulls().n_unique() > 1, column
    assert set(baseline.full.games.get_column("season").unique()) == set(_SEASONS)
    assert baseline.full.built == list(_SEASONS)
    assert baseline.full.rated == list(_SEASONS)


def test_a_cold_incremental_run_builds_every_season_and_equals_the_full_rebuild(
    baseline: _Baseline,
) -> None:
    # Act
    cold = baseline.cold

    # Assert
    _assert_identical(cold, baseline.full)
    assert cold.built == list(_SEASONS)


def test_a_cold_run_caches_only_the_finished_seasons(baseline: _Baseline) -> None:
    # Act
    entries = sorted(path.name for path in _cache_dir(baseline.cold_dir).iterdir())

    # Assert
    assert entries == [f"season_{season}" for season in _CACHED_SEASONS]


def test_a_warm_incremental_run_reuses_finished_seasons_and_equals_the_full_rebuild(
    baseline: _Baseline, etl: _Run, warm_dir: Path
) -> None:
    # Act
    warm = _main(etl, warm_dir, "--incremental")

    # Assert
    _assert_identical(warm, baseline.full)
    assert warm.built == [_CURRENT_SEASON]
    assert warm.rated == [_CURRENT_SEASON]


_RATING_COLUMNS = [
    column
    for stat in constants.RATINGS_FEATURE_STATS
    for column in (f"away_{stat}", f"home_{stat}", f"{stat}_diff")
]


def _rating_features(games: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Return the rating features of one week's game rows, by game."""
    return (
        games.filter((pl.col("season") == season) & (pl.col("week") == week))
        .select("game_id", *_RATING_COLUMNS)
        .sort("game_id")
    )


def test_a_weeks_rating_features_never_see_its_or_later_plays_through_the_etl(
    baseline: _Baseline, etl: _Run, tmp_path: Path
) -> None:
    """Week 3, the first week with every rating feature, ``ratings_sos`` included."""
    # Arrange
    later = (pl.col("season") > 2020) | ((pl.col("season") == 2020) & (pl.col("week") >= 3))
    etl.world.plays = rewritten_plays(etl.world.plays, later)
    etl.world.ratings_schedule = etl.world.ratings_schedule.with_columns(
        pl.when(later).then(pl.col("home_score") + 21).otherwise("home_score").alias("home_score")
    )

    # Act
    rewritten = _main(etl, tmp_path)

    # Assert
    expected = _rating_features(baseline.full.games, 2020, 3)
    assert expected.select(_RATING_COLUMNS).null_count().row(0) == (0,) * len(_RATING_COLUMNS)
    assert_frame_equal(_rating_features(rewritten.games, 2020, 3), expected, check_exact=True)
    assert not _rating_features(rewritten.games, 2021, 2).equals(
        _rating_features(baseline.full.games, 2021, 2)
    )


def test_a_changed_play_in_a_cached_season_rebuilds_from_that_season_on(
    baseline: _Baseline, etl: _Run, warm_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # Arrange
    etl.world.plays = etl.world.plays.with_columns(
        pl.when((pl.col("season") == 2020) & (pl.col("week") == 2))
        .then(pl.col("epa") + 0.5)
        .otherwise(pl.col("epa"))
        .alias("epa")
    )
    full = _main(etl, tmp_path_factory.mktemp("full"))

    # Act
    warm = _main(etl, warm_dir, "--incremental")

    # Assert
    assert full.files["all_data_ml.csv"] != baseline.full.files["all_data_ml.csv"]
    _assert_identical(warm, full)
    assert warm.built == [2020, 2021]
    assert warm.rated == [2020, 2021]


def test_a_changed_play_only_the_season_statistics_read_rebuilds_that_season(
    baseline: _Baseline, etl: _Run, warm_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The win probability feeds only the season statistics' win-probability bins."""
    # Arrange
    etl.world.plays = etl.world.plays.with_columns(
        pl.when((pl.col("season") == 2019) & (pl.col("week") == 2))
        .then(pl.col("wp") / 2.0)
        .otherwise(pl.col("wp"))
        .alias("wp")
    )
    full = _main(etl, tmp_path_factory.mktemp("full"))

    # Act
    warm = _main(etl, warm_dir, "--incremental")

    # Assert
    bins = f"{constants.RATINGS_DIRNAME}/2019_team_wp_bins.parquet"
    assert full.files[bins] != baseline.full.files[bins]
    assert full.files["all_data_ml.csv"] == baseline.full.files["all_data_ml.csv"]
    _assert_identical(warm, full)
    assert warm.rated == [2019, _CURRENT_SEASON]


def test_a_changed_input_in_a_cached_season_rebuilds_from_that_season_on(
    baseline: _Baseline, etl: _Run, warm_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # Arrange
    etl.world.team_stats = etl.world.team_stats.with_columns(
        pl.when((pl.col("season") == 2020) & (pl.col("week") == 2))
        .then(pl.col("pass_epa_sum") + 1.5)
        .otherwise(pl.col("pass_epa_sum"))
        .alias("pass_epa_sum")
    )
    full = _main(etl, tmp_path_factory.mktemp("full"))

    # Act
    warm = _main(etl, warm_dir, "--incremental")

    # Assert
    assert full.files["all_data_ml.csv"] != baseline.full.files["all_data_ml.csv"]
    _assert_identical(warm, full)
    assert warm.built == [2020, 2021]


def test_a_changed_value_in_the_current_season_keeps_finished_seasons_cached(
    baseline: _Baseline, etl: _Run, warm_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Values change but the row count does not, so every frame keeps its chunk layout."""
    # Arrange
    etl.world.team_stats = etl.world.team_stats.with_columns(
        pl.when(pl.col("season") == _CURRENT_SEASON)
        .then(pl.col("rush_epa_sum") - 2.0)
        .otherwise(pl.col("rush_epa_sum"))
        .alias("rush_epa_sum")
    )
    full = _main(etl, tmp_path_factory.mktemp("full"))

    # Act
    warm = _main(etl, warm_dir, "--incremental")

    # Assert
    assert full.files["all_data_ml.csv"] != baseline.full.files["all_data_ml.csv"]
    _assert_identical(warm, full)
    assert warm.built == [_CURRENT_SEASON]


def test_a_new_week_in_the_current_season_keeps_finished_seasons_equal_to_a_full_rebuild(
    etl: _Run, warm_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # Arrange
    etl.world = _world(played_through=_CURRENT_WEEK)
    full = _main(etl, tmp_path_factory.mktemp("full"))

    # Act
    warm = _main(etl, warm_dir, "--incremental")

    # Assert
    assert warm.built == [_CURRENT_SEASON]
    _assert_identical(warm, full)


def test_a_code_fingerprint_change_rebuilds_every_season(
    baseline: _Baseline, etl: _Run, warm_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(season_cache, "etl_code_fingerprint", lambda: "edited code")

    # Act
    warm = _main(etl, warm_dir, "--incremental")

    # Assert
    _assert_identical(warm, baseline.full)
    assert warm.built == list(_SEASONS)


def test_a_damaged_entry_is_rebuilt_and_rewritten_with_the_output_unchanged(
    baseline: _Baseline, etl: _Run, warm_dir: Path
) -> None:
    # Arrange
    for path in sorted((_cache_dir(warm_dir) / "season_2020").iterdir()):
        path.write_bytes(b"\x00garbage")

    # Act
    warm = _main(etl, warm_dir, "--incremental")
    rewarm = _main(etl, warm_dir, "--incremental")

    # Assert
    _assert_identical(warm, baseline.full)
    assert warm.built == [2020, 2021]
    _assert_identical(rewarm, baseline.full)
    assert rewarm.built == [_CURRENT_SEASON]


def test_an_entry_whose_manifest_lists_no_ratings_is_rebuilt_with_the_output_unchanged(
    baseline: _Baseline, etl: _Run, warm_dir: Path
) -> None:
    # Arrange
    manifest = _cache_dir(warm_dir) / "season_2020" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["ratings"] = []
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    # Act
    warm = _main(etl, warm_dir, "--incremental")

    # Assert
    _assert_identical(warm, baseline.full)
    assert (warm.built, warm.rated) == ([2020, 2021], [2020, 2021])


def test_the_full_rebuild_neither_reads_nor_writes_the_cache(baseline: _Baseline) -> None:
    # Act
    full = baseline.full

    # Assert
    assert full.built == list(_SEASONS)
    assert baseline.stamps_before_full
    assert baseline.stamps_after_full == baseline.stamps_before_full


def _is_strength_column(column: str) -> bool:
    return any(stat in column for stat in constants.ADJUSTED_STRENGTH_STATS)


def test_the_default_strength_blend_games_given_explicitly_writes_the_same_files(
    baseline: _Baseline, etl: _Run, tmp_path: Path
) -> None:
    # Act
    explicit = _main(etl, tmp_path, "--strength-prior-blend-games", "4")

    # Assert
    _assert_identical(explicit, baseline.full)


def test_a_smaller_strength_blend_games_moves_only_the_strength_family(
    baseline: _Baseline, etl: _Run, tmp_path: Path
) -> None:
    # Act
    tuned = _main(etl, tmp_path, "--strength-prior-blend-games", "2")

    # Assert
    assert tuned.games.columns == baseline.full.games.columns
    changed = {
        column
        for column in baseline.full.games.columns
        if not tuned.games.get_column(column).equals(baseline.full.games.get_column(column))
    }
    assert "away_adj_strength_composite" in changed
    assert all(_is_strength_column(column) for column in changed), changed
    assert not tuned.snapshots.equals(baseline.full.snapshots)
    assert tuned.files["strength_snapshots.csv"] != baseline.full.files["strength_snapshots.csv"]


def test_a_changed_strength_blend_games_rebuilds_every_cached_season(
    etl: _Run, warm_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # Arrange
    full = _main(etl, tmp_path_factory.mktemp("full"), "--strength-prior-blend-games", "2")

    # Act
    warm = _main(etl, warm_dir, "--incremental", "--strength-prior-blend-games", "2")

    # Assert
    _assert_identical(warm, full)
    assert warm.built == list(_SEASONS)


def _season_keys(
    world: _World, config: data_collection.DataCollectionConfig
) -> dict[int, str | None]:
    """Return the cache key of each finished season, as an incremental run would build it."""
    sources = data_collection._EtlSources(
        current_season=_CURRENT_SEASON,
        current_week=_CURRENT_WEEK,
        schedule_df=world.schedule,
        team_stats_df=world.team_stats,
        pbp_df=pl.DataFrame(),
        elo_df=world.elo,
        raw_elo_df=world.elo,
        ratings=_ratings(world),
    )
    cache = data_collection._open_season_cache(config, sources, _SEASONS[0])
    assert cache is not None
    return {
        season: cache.keys.key(
            season,
            data_collection._season_frames(
                season,
                sources,
                data_collection.SeasonInputs(
                    min_season=_SEASONS[0],
                    tr_df=world.rankings[season],
                    prev_tr_df=world.rankings[season - 1] if season > _SEASONS[0] else None,
                ),
            ),
        )
        for season in _CACHED_SEASONS
    }


def _incremental_config(**changes: object) -> data_collection.DataCollectionConfig:
    config = data_collection.replace(
        data_collection._default_config(list(_SEASONS)), incremental=True
    )
    return data_collection.replace(config, **changes)


@pytest.mark.parametrize("source", ["schedule", "team_stats", "elo", "rankings", "plays"])
def test_a_changed_first_season_row_changes_every_finished_season_key(source: str) -> None:
    """The first season feeds the next one's coach history and week-1 priors and fallbacks."""
    # Arrange
    world = _world()
    baseline = _season_keys(world, _incremental_config())
    first = pl.col("season") == _SEASONS[0]
    if source == "schedule":
        world.schedule = world.schedule.with_columns(
            pl.when(first)
            .then(pl.lit("Coach Elsewhere"))
            .otherwise("home_coach")
            .alias("home_coach")
        )
    elif source == "team_stats":
        world.team_stats = world.team_stats.with_columns(
            pl.when(first)
            .then(pl.col("rush_yards") + 1.0)
            .otherwise("rush_yards")
            .alias("rush_yards")
        )
    elif source == "elo":
        world.elo = world.elo.with_columns(
            pl.when(first)
            .then(pl.col("away_elo_pre") + 25.0)
            .otherwise("away_elo_pre")
            .alias("away_elo_pre")
        )
    elif source == "plays":
        world.plays = world.plays.with_columns(
            pl.when(first).then(pl.col("qb_epa") - 0.5).otherwise("qb_epa").alias("qb_epa")
        )
    else:
        world.rankings[_SEASONS[0]] = world.rankings[_SEASONS[0]].with_columns(
            pl.col("predictive_rating") + 1.0
        )

    # Act
    changed = _season_keys(world, _incremental_config())

    # Assert
    assert all(changed[season] != baseline[season] for season in _CACHED_SEASONS)


def test_a_changed_prior_season_play_changes_every_finished_season_key() -> None:
    """The prior season sets the penalties and the preseason prior of the ratings."""
    # Arrange
    world = _world()
    baseline = _season_keys(world, _incremental_config())
    world.plays = world.plays.with_columns(
        pl.when(pl.col("season") == _PRIOR_SEASON)
        .then(pl.col("epa") + 0.25)
        .otherwise("epa")
        .alias("epa")
    )

    # Act
    changed = _season_keys(world, _incremental_config())

    # Assert
    assert all(changed[season] != baseline[season] for season in _CACHED_SEASONS)


def test_a_changed_season_statistics_play_changes_only_that_seasons_key() -> None:
    """A season's statistics read no other season, so the change reaches no later key."""
    # Arrange
    world = _world()
    baseline = _season_keys(world, _incremental_config())
    world.plays = world.plays.with_columns(
        pl.when(pl.col("season") == 2019).then(pl.col("wp") / 2.0).otherwise("wp").alias("wp")
    )

    # Act
    changed = _season_keys(world, _incremental_config())

    # Assert
    assert changed[2019] != baseline[2019]
    assert changed[2020] == baseline[2020]


def test_a_new_player_in_the_players_file_keeps_every_finished_season_cached(
    baseline: _Baseline, etl: _Run, warm_dir: Path
) -> None:
    """An in-season run downloads the players file again, with new players every week."""
    # Arrange
    signing = pl.DataFrame(
        {"gsis_id": ["00-0099999"], "position": ["QB"], "display_name": ["New Signing"]}
    )
    etl.world.players = pl.concat([etl.world.players, signing], how="diagonal")

    # Act
    warm = _main(etl, warm_dir, "--incremental")

    # Assert
    _assert_identical(warm, baseline.full)
    assert warm.rated == [_CURRENT_SEASON]


def test_a_changed_current_season_play_keeps_every_finished_season_key() -> None:
    # Arrange
    world = _world()
    baseline = _season_keys(world, _incremental_config())
    world.plays = world.plays.with_columns(
        pl.when(pl.col("season") == _CURRENT_SEASON)
        .then(pl.col("epa") + 0.25)
        .otherwise("epa")
        .alias("epa")
    )

    # Act
    changed = _season_keys(world, _incremental_config())

    # Assert
    assert changed == baseline


def test_a_changed_prior_season_team_stat_changes_every_finished_season_key() -> None:
    # Arrange
    world = _world()
    baseline = _season_keys(world, _incremental_config())
    world.team_stats = world.team_stats.with_columns(
        pl.when(pl.col("season") == _PRIOR_SEASON)
        .then(pl.col("pass_yards") + 1.0)
        .otherwise("pass_yards")
        .alias("pass_yards")
    )

    # Act
    changed = _season_keys(world, _incremental_config())

    # Assert
    assert all(changed[season] != baseline[season] for season in _CACHED_SEASONS)


@pytest.mark.parametrize(
    "change",
    [
        {"blend_strength_prior": False},
        {"blend_stat_prior": False},
        {"stat_prior_blend_games": 6.0},
        {"strength_prior_blend_games": 2.0},
        {"team_stats_source": "nflverse"},
        {"tr_stats_source": "scrape"},
    ],
    ids=lambda change: next(iter(change)),
)
def test_every_option_that_changes_a_season_build_changes_its_key(
    change: dict[str, object],
) -> None:
    # Arrange
    world = _world()
    baseline = _season_keys(world, _incremental_config())

    # Act
    changed = _season_keys(world, _incremental_config(**change))

    # Assert
    assert all(changed[season] != baseline[season] for season in _CACHED_SEASONS)


@pytest.mark.parametrize(
    "change",
    [
        {"enable_timing": True},
        {"enable_debug": True},
        {"force_refresh_nflreadpy": True},
        {"max_season": 2030},
    ],
    ids=lambda change: next(iter(change)),
)
def test_options_that_cannot_change_a_season_build_keep_its_key(
    change: dict[str, object], tmp_path: Path
) -> None:
    # Arrange
    world = _world()
    baseline = _season_keys(world, _incremental_config())

    # Act
    changed = _season_keys(world, _incremental_config(**change, data_dir=tmp_path))

    # Assert
    assert changed == baseline


def test_an_incremental_run_keeps_its_cache_under_its_data_directory(tmp_path: Path) -> None:
    # Arrange
    sources = data_collection._EtlSources(
        current_season=_CURRENT_SEASON,
        current_week=_CURRENT_WEEK,
        schedule_df=pl.DataFrame(),
        team_stats_df=pl.DataFrame(),
        pbp_df=pl.DataFrame(),
        elo_df=pl.DataFrame(),
        raw_elo_df=pl.DataFrame(),
    )

    # Act
    cache = data_collection._open_season_cache(
        _incremental_config(data_dir=tmp_path), sources, _SEASONS[0]
    )
    full = data_collection._open_season_cache(
        data_collection._default_config(list(_SEASONS)), sources, _SEASONS[0]
    )

    # Assert
    assert cache is not None
    assert cache.cache.directory == _cache_dir(tmp_path)
    assert full is None


def test_a_season_without_a_key_is_rebuilt_and_not_cached(
    baseline: _Baseline, etl: _Run, warm_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(season_cache.SeasonKeys, "key", lambda *_args: None)
    stamps = _stamps(_cache_dir(warm_dir))

    # Act
    warm = _main(etl, warm_dir, "--incremental")

    # Assert
    _assert_identical(warm, baseline.full)
    assert warm.built == list(_SEASONS)
    assert _stamps(_cache_dir(warm_dir)) == stamps


# Everything a season build reads, and how the cache key covers it. A new parameter or field
# fails the guards below until it is keyed or shown to leave a season's rows unchanged.
_PROCESS_SEASON_PARAMETERS = {
    "season": "keyed: the key is per season",
    "schedule_df": "keyed: the schedule's rows through the season",
    "team_stats_df": "keyed: the team stats' rows through the season",
    "inputs": "covered field by field in _SEASON_INPUT_FIELDS",
}
_KEYED_OPTIONS = (
    "min_season",
    "blend_strength_prior",
    "blend_stat_prior",
    "stat_prior_blend_games",
    "strength_prior_blend_games",
    "team_stats_source",
    "tr_stats_source",
    "line_source",
)
_DERIVED_SEASON_INPUTS = (
    "team_elo_trends",
    "qb_trends",
    "team_stat_trends",
    "coach_features",
    "prior_strength_snapshot",
    "prior_season_stats",
)
_SEASON_INPUT_FIELDS = {
    **dict.fromkeys(
        set(_KEYED_OPTIONS) - {"team_stats_source", "line_source"}, "keyed: a run option"
    ),
    "elo_df": "keyed: the Elo rows through the season",
    "tr_df": "keyed: the season's TeamRankings",
    "prev_tr_df": "keyed: the previous season's TeamRankings",
    "strength_snapshots": "output-neutral: the sink a build appends its snapshots to",
    "timing_enabled": "output-neutral: timing logs only",
    "timing_totals": "output-neutral: timing logs only",
    **dict.fromkeys(_DERIVED_SEASON_INPUTS, "recomputed by process_season from keyed inputs"),
}
_CONFIG_FIELDS = {
    **dict.fromkeys(_KEYED_OPTIONS, "keyed: a run option"),
    "enable_timing": "output-neutral: logging only",
    "enable_debug": "output-neutral: logging only",
    "force_refresh_nflreadpy": "output-neutral: changes only the loaded rows, keyed by value",
    "max_season": "output-neutral: adds later seasons, which no earlier key reads",
    "data_dir": "output-neutral: where the files and the cache are written",
    "incremental": "output-neutral: whether the cache is used",
}


def test_every_process_season_parameter_is_keyed() -> None:
    # Act
    parameters = inspect.signature(data_collection.process_season).parameters

    # Assert
    assert set(parameters) == set(_PROCESS_SEASON_PARAMETERS)


def test_every_season_input_is_keyed_or_output_neutral() -> None:
    # Act
    fields = {field.name for field in dataclasses.fields(data_collection.SeasonInputs)}

    # Assert
    assert fields == set(_SEASON_INPUT_FIELDS)


def test_every_config_field_is_keyed_or_output_neutral_and_the_key_holds_the_keyed_ones() -> None:
    # Arrange
    fields = {field.name for field in dataclasses.fields(data_collection.DataCollectionConfig)}
    world = _world()
    sources = data_collection._EtlSources(
        current_season=_CURRENT_SEASON,
        current_week=_CURRENT_WEEK,
        schedule_df=world.schedule,
        team_stats_df=world.team_stats,
        pbp_df=pl.DataFrame(),
        elo_df=world.elo,
        raw_elo_df=world.elo,
    )

    # Act
    cache = data_collection._open_season_cache(_incremental_config(), sources, _SEASONS[0])

    # Assert
    assert fields == set(_CONFIG_FIELDS)
    assert cache is not None
    options = cache.keys._context["options"]
    assert isinstance(options, dict)
    assert set(options) == set(_KEYED_OPTIONS)
    assert set(cache.keys._frames) == {"schedule", "team_stats", "elo"}


def test_process_season_recomputes_every_derived_season_input() -> None:
    # Arrange
    world = _world()
    sentinel = pl.DataFrame({"sentinel": [1]})
    supplied = data_collection.SeasonInputs(
        min_season=_SEASONS[0],
        elo_df=world.elo,
        team_elo_trends=sentinel,
        qb_trends=sentinel,
        team_stat_trends=sentinel,
        coach_features=sentinel,
        prior_strength_snapshot=sentinel,
        prior_season_stats=sentinel,
    )

    # Act
    derived = data_collection._with_season_features(
        _SEASONS[1], world.schedule, world.team_stats, supplied
    )

    # Assert
    assert all(getattr(derived, name) is not sentinel for name in _DERIVED_SEASON_INPUTS)
