"""Characterization tests: the incremental ETL writes exactly what the full rebuild writes.

A small three-season world (two finished seasons and the current one) runs through the real
season and week builders; only the upstream loaders, the play-by-play quarterback join and
the SurvivorGrid scrape are replaced. Each test compares every written CSV byte for byte
with a full rebuild of the same inputs, and the combined frame exactly, after the cache was
filled, after a cached input changed, after the code fingerprint changed and after an entry
was damaged.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nfl_predictor import constants, data_collection
from nfl_predictor.utils import clock, polars_utils, season_cache

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


def _played(season: int, week: int) -> bool:
    return season < _CURRENT_SEASON or week < _CURRENT_WEEK


@dataclass
class _World:
    """Every upstream input of one ETL run."""

    schedule: pl.DataFrame
    team_stats: pl.DataFrame
    elo: pl.DataFrame
    rankings: dict[int, pl.DataFrame] = field(default_factory=dict)


def _world(seed: int = 7) -> _World:
    rng = np.random.default_rng(seed)
    games: list[dict[str, object]] = []
    team_games: list[dict[str, object]] = []
    elo_rows: list[dict[str, object]] = []
    for season in (_PRIOR_SEASON, *_SEASONS):
        for week in range(1, _WEEKS + 1):
            for index, (away, home) in enumerate(_slate(season, week)):
                played = _played(season, week)
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
                    **{column: float(rng.normal(0.0, 5.0)) for column in constants.TR_RATINGS},
                }
                for week in range(1, _WEEKS + 1)
                for team in _TEAMS
            ]
        )
        for season in (_PRIOR_SEASON, *_SEASONS)
    }
    return _World(
        schedule=schedule.filter(pl.col("season") >= _SEASONS[0]),
        team_stats=pl.DataFrame(team_games),
        elo=pl.DataFrame(elo_rows).filter(pl.col("season") >= _SEASONS[0]),
        rankings=rankings,
    )


@dataclass
class _Run:
    """The world one ETL run reads, the seasons it built and what it returned."""

    world: _World
    built: list[int] = field(default_factory=list)
    games: pl.DataFrame | None = None
    snapshots: pl.DataFrame | None = None


def _install(monkeypatch: pytest.MonkeyPatch, run: _Run) -> None:
    """Replace the loaders and the network steps, and record what each run builds."""

    def load_sources(_seasons: list[int], _config: object) -> object:
        world = run.world
        return data_collection._EtlSources(
            current_season=_CURRENT_SEASON,
            current_week=_CURRENT_WEEK,
            schedule_df=world.schedule,
            team_stats_df=world.team_stats,
            pbp_df=pl.DataFrame(),
            elo_df=world.elo,
            raw_elo_df=world.elo,
        )

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
        strength_snapshots: list[pl.DataFrame],
    ) -> pl.DataFrame:
        run.games = collect_all_data(seasons, config=config, strength_snapshots=strength_snapshots)
        run.snapshots = data_collection.combine_strength_snapshots(strength_snapshots)
        return run.games

    monkeypatch.setattr(data_collection, "_load_sources", load_sources)
    monkeypatch.setattr(data_collection, "process_season", counting_process_season)
    monkeypatch.setattr(data_collection, "collect_all_data", recording_collect_all_data)
    monkeypatch.setattr(
        polars_utils,
        "load_team_rankings",
        lambda season, *_args, **_kwargs: run.world.rankings[season],
    )
    monkeypatch.setattr(data_collection, "_attach_qb_features", lambda games, *_a, **_k: games)
    monkeypatch.setattr(data_collection.game_utils, "fill_future_game_lines", lambda df: df)
    monkeypatch.setattr(clock, "local_today", lambda: date(_CURRENT_SEASON, 9, 22))
    monkeypatch.setattr(clock, "nfl_week", lambda _today: _CURRENT_WEEK)


_ARGS = ("--min-season", str(_SEASONS[0]), "--max-season", str(_SEASONS[-1]))


@dataclass(frozen=True)
class _Result:
    """Every file one run wrote, the combined frame, the snapshot file and the seasons built."""

    files: dict[str, bytes]
    games: pl.DataFrame
    snapshots: pl.DataFrame
    built: list[int]


def _main(run: _Run, data_dir: Path, *extra: str) -> _Result:
    run.built = []
    data_collection.main([*_ARGS, "--data-dir", str(data_dir), *extra])
    assert run.games is not None
    assert run.snapshots is not None
    files = {
        path.relative_to(data_dir).as_posix(): path.read_bytes()
        for path in sorted(data_dir.rglob("*.csv"))
        if "cache" not in path.relative_to(data_dir).parts
    }
    return _Result(files, run.games, run.snapshots, list(run.built))


def _assert_identical(incremental: _Result, full: _Result) -> None:
    assert_frame_equal(incremental.games, full.games, check_exact=True)
    assert_frame_equal(incremental.snapshots, full.snapshots, check_exact=True)
    assert sorted(incremental.files) == sorted(full.files)
    assert len(full.files) == 6, "five datasets and one games-to-predict file"
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
    finished = baseline.full.games.filter(pl.col("season") < _CURRENT_SEASON)

    for column in (
        "away_adj_strength_composite",
        "home_sos_played_raw",
        "away_elo_4wk_trend",
        "home_coach_win_pct_prior",
        "away_predictive_rating",
        "home_pass_yards",
        "away_wins",
    ):
        assert finished.get_column(column).drop_nulls().n_unique() > 1, column
    assert set(baseline.full.games.get_column("season").unique()) == set(_SEASONS)
    assert baseline.full.built == list(_SEASONS)


def test_a_cold_incremental_run_builds_every_season_and_equals_the_full_rebuild(
    baseline: _Baseline,
) -> None:
    _assert_identical(baseline.cold, baseline.full)
    assert baseline.cold.built == list(_SEASONS)


def test_a_cold_run_caches_only_the_finished_seasons(baseline: _Baseline) -> None:
    entries = sorted(path.name for path in _cache_dir(baseline.cold_dir).iterdir())

    assert entries == [f"season_{season}" for season in _CACHED_SEASONS]


def test_a_warm_incremental_run_reuses_finished_seasons_and_equals_the_full_rebuild(
    baseline: _Baseline, etl: _Run, warm_dir: Path
) -> None:
    warm = _main(etl, warm_dir, "--incremental")

    _assert_identical(warm, baseline.full)
    assert warm.built == [_CURRENT_SEASON]


def test_a_changed_input_in_a_cached_season_rebuilds_from_that_season_on(
    baseline: _Baseline, etl: _Run, warm_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    etl.world.team_stats = etl.world.team_stats.with_columns(
        pl.when((pl.col("season") == 2020) & (pl.col("week") == 2))
        .then(pl.col("pass_epa_sum") + 1.5)
        .otherwise(pl.col("pass_epa_sum"))
        .alias("pass_epa_sum")
    )
    full = _main(etl, tmp_path_factory.mktemp("full"))

    warm = _main(etl, warm_dir, "--incremental")

    assert full.files["all_data_ml.csv"] != baseline.full.files["all_data_ml.csv"]
    _assert_identical(warm, full)
    assert warm.built == [2020, 2021]


def test_a_change_in_the_current_season_keeps_finished_seasons_cached(
    baseline: _Baseline, etl: _Run, warm_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    etl.world.team_stats = etl.world.team_stats.with_columns(
        pl.when(pl.col("season") == _CURRENT_SEASON)
        .then(pl.col("rush_epa_sum") - 2.0)
        .otherwise(pl.col("rush_epa_sum"))
        .alias("rush_epa_sum")
    )
    full = _main(etl, tmp_path_factory.mktemp("full"))

    warm = _main(etl, warm_dir, "--incremental")

    assert full.files["all_data_ml.csv"] != baseline.full.files["all_data_ml.csv"]
    _assert_identical(warm, full)
    assert warm.built == [_CURRENT_SEASON]


def test_a_code_fingerprint_change_rebuilds_every_season(
    baseline: _Baseline, etl: _Run, warm_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(season_cache, "etl_code_fingerprint", lambda: "edited code")

    warm = _main(etl, warm_dir, "--incremental")

    _assert_identical(warm, baseline.full)
    assert warm.built == list(_SEASONS)


def test_a_damaged_entry_is_rebuilt_and_rewritten_with_the_output_unchanged(
    baseline: _Baseline, etl: _Run, warm_dir: Path
) -> None:
    for path in sorted((_cache_dir(warm_dir) / "season_2020").iterdir()):
        path.write_bytes(b"\x00garbage")

    warm = _main(etl, warm_dir, "--incremental")
    rewarm = _main(etl, warm_dir, "--incremental")

    _assert_identical(warm, baseline.full)
    assert warm.built == [2020, 2021]
    _assert_identical(rewarm, baseline.full)
    assert rewarm.built == [_CURRENT_SEASON]


def test_the_full_rebuild_neither_reads_nor_writes_the_cache(baseline: _Baseline) -> None:
    assert baseline.full.built == list(_SEASONS)
    assert baseline.stamps_before_full
    assert baseline.stamps_after_full == baseline.stamps_before_full


def _season_keys(world: _World, config: data_collection.DataCollectionConfig) -> dict[int, str]:
    """Return the cache key of each finished season, as an incremental run would build it."""
    sources = data_collection._EtlSources(
        current_season=_CURRENT_SEASON,
        current_week=_CURRENT_WEEK,
        schedule_df=world.schedule,
        team_stats_df=world.team_stats,
        pbp_df=pl.DataFrame(),
        elo_df=world.elo,
        raw_elo_df=world.elo,
    )
    cache = data_collection._open_season_cache(config, sources, _SEASONS[0])
    assert cache is not None
    return {
        season: cache.keys.key(
            season,
            {
                "tr": world.rankings[season],
                "prev_tr": world.rankings[season - 1] if season > _SEASONS[0] else None,
            },
        )
        for season in _CACHED_SEASONS
    }


def _incremental_config(**changes: object) -> data_collection.DataCollectionConfig:
    config = data_collection.replace(
        data_collection._default_config(list(_SEASONS)), incremental=True
    )
    return data_collection.replace(config, **changes)


@pytest.mark.parametrize("source", ["schedule", "team_stats", "elo", "rankings"])
def test_a_changed_first_season_row_changes_every_finished_season_key(source: str) -> None:
    """The first season feeds the next one's coach history and week-1 priors and fallbacks."""
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
    else:
        world.rankings[_SEASONS[0]] = world.rankings[_SEASONS[0]].with_columns(
            pl.col("predictive_rating") + 1.0
        )

    changed = _season_keys(world, _incremental_config())

    assert all(changed[season] != baseline[season] for season in _CACHED_SEASONS)


def test_a_changed_prior_season_team_stat_changes_every_finished_season_key() -> None:
    world = _world()
    baseline = _season_keys(world, _incremental_config())
    world.team_stats = world.team_stats.with_columns(
        pl.when(pl.col("season") == _PRIOR_SEASON)
        .then(pl.col("pass_yards") + 1.0)
        .otherwise("pass_yards")
        .alias("pass_yards")
    )

    changed = _season_keys(world, _incremental_config())

    assert all(changed[season] != baseline[season] for season in _CACHED_SEASONS)


@pytest.mark.parametrize(
    "change",
    [
        {"blend_strength_prior": False},
        {"blend_stat_prior": False},
        {"stat_prior_blend_games": 6.0},
        {"team_stats_source": "nflverse"},
        {"tr_stats_source": "scrape"},
    ],
    ids=lambda change: next(iter(change)),
)
def test_every_option_that_changes_a_season_build_changes_its_key(
    change: dict[str, object],
) -> None:
    world = _world()

    baseline = _season_keys(world, _incremental_config())
    changed = _season_keys(world, _incremental_config(**change))

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
    world = _world()

    baseline = _season_keys(world, _incremental_config())
    changed = _season_keys(world, _incremental_config(**change, data_dir=tmp_path))

    assert changed == baseline


def test_an_incremental_run_keeps_its_cache_under_its_data_directory(tmp_path: Path) -> None:
    sources = data_collection._EtlSources(
        current_season=_CURRENT_SEASON,
        current_week=_CURRENT_WEEK,
        schedule_df=pl.DataFrame(),
        team_stats_df=pl.DataFrame(),
        pbp_df=pl.DataFrame(),
        elo_df=pl.DataFrame(),
        raw_elo_df=pl.DataFrame(),
    )

    cache = data_collection._open_season_cache(
        _incremental_config(data_dir=tmp_path), sources, _SEASONS[0]
    )
    full = data_collection._open_season_cache(
        data_collection._default_config(list(_SEASONS)), sources, _SEASONS[0]
    )

    assert cache is not None
    assert cache.cache.directory == _cache_dir(tmp_path)
    assert full is None


def test_a_season_without_a_key_is_rebuilt_and_not_cached(
    baseline: _Baseline, etl: _Run, warm_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(season_cache.SeasonKeys, "key", lambda *_args: None)
    stamps = _stamps(_cache_dir(warm_dir))

    warm = _main(etl, warm_dir, "--incremental")

    _assert_identical(warm, baseline.full)
    assert warm.built == list(_SEASONS)
    assert _stamps(_cache_dir(warm_dir)) == stamps
