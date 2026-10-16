"""Tests for the per-season cache behind the incremental ETL rebuild.

The cache may only ever return a season's build when every input that build reads is
unchanged, so most tests here change one input and require a miss. A damaged entry must
read as a miss, never as an error.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from importlib import metadata
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nfl_predictor.utils import season_cache

if TYPE_CHECKING:
    from pathlib import Path


def _frame() -> pl.DataFrame:
    """Return rows from three seasons, interleaved, with nulls, NaN and several dtypes."""
    return pl.DataFrame(
        {
            "season": [2020, 2021, 2020, 2022, 2021, 2022],
            "week": [1, 1, 2, 1, 2, 2],
            "team_abbr": ["AAA", "BBB", None, "AAA", "", "CCC"],
            "value": [0.1, -0.0, float("nan"), None, 1 / 3, 2.5],
            "count": pl.Series([1, 2, 3, None, 5, 6], dtype=pl.Int32),
            "kickoff": [date(2020, 9, 1)] * 6,
        }
    )


def _keys(frame: pl.DataFrame, **context: object) -> season_cache.SeasonKeys:
    return season_cache.SeasonKeys({"team_stats": frame}, {"min_season": 2020, **context})


def _key(frame: pl.DataFrame, season: int, **context: object) -> str | None:
    return _keys(frame, **context).key(season, {"tr": None})


def test_a_key_is_stable_for_the_same_inputs() -> None:
    # Arrange
    first = _key(_frame(), 2021)

    # Act
    second = _key(_frame(), 2021)

    # Assert
    assert second == first


def test_a_key_ignores_how_polars_laid_the_rows_out_in_memory() -> None:
    # Arrange
    frame = _frame()
    pieces = pl.concat([frame.slice(0, 1), frame.slice(1, 2), frame.slice(3)], rechunk=False)
    expected = _key(frame, 2021)

    # Act
    key = _key(pieces, 2021)

    # Assert
    assert key == expected


def test_a_key_ignores_changed_rows_from_later_seasons() -> None:
    # Arrange
    later_changed = _frame().with_columns(
        pl.when(pl.col("season") == 2022).then(99.0).otherwise(pl.col("value")).alias("value")
    )
    expected = _key(_frame(), 2021)

    # Act
    key = _key(later_changed, 2021)

    # Assert
    assert key == expected


def test_a_key_ignores_added_rows_from_later_seasons() -> None:
    # Arrange
    later_added = pl.concat([_frame(), _frame().filter(pl.col("season") == 2022)])
    expected = _key(_frame(), 2021)

    # Act
    key = _key(later_added, 2021)

    # Assert
    assert key == expected


@pytest.mark.parametrize(
    "change",
    [
        pl.when(pl.col("season") == 2020).then(0.2).otherwise(pl.col("value")).alias("value"),
        pl.when(pl.col("season") == 2021).then(None).otherwise(pl.col("value")).alias("value"),
        pl.when(pl.col("season") == 2021).then(0.0).otherwise(pl.col("value")).alias("value"),
        pl.col("team_abbr").fill_null(""),
        pl.col("count").cast(pl.Int64),
        pl.col("value").cast(pl.Float32),
    ],
    ids=[
        "earlier-season-value",
        "value-to-null",
        "negative-zero-to-zero",
        "null-to-empty-string",
        "dtype-int",
        "dtype-float",
    ],
)
def test_a_key_changes_when_a_row_through_the_season_changes(change: pl.Expr) -> None:
    # Arrange
    baseline = _key(_frame(), 2021)

    # Act
    key = _key(_frame().with_columns(change), 2021)

    # Assert
    assert key != baseline


def test_a_key_changes_when_rows_through_the_season_are_reordered() -> None:
    # Arrange
    frame = _frame()
    swapped = pl.concat([frame.slice(2, 1), frame.slice(0, 2), frame.slice(3)])
    baseline = _key(frame, 2021)

    # Act
    key = _key(swapped, 2021)

    # Assert
    assert key != baseline


def test_a_key_changes_with_a_column_added_even_when_it_is_null_through_the_season() -> None:
    # Arrange
    widened = _frame().with_columns(
        pl.when(pl.col("season") == 2022).then(1.0).otherwise(None).alias("late_column")
    )
    baseline = _key(_frame(), 2021)

    # Act
    key = _key(widened, 2021)

    # Assert
    assert key != baseline


_TR = pl.DataFrame({"team_abbr": ["AAA"], "week": [1], "predictive_rating": [1.0]})


def test_a_key_changes_when_a_season_frame_appears() -> None:
    # Arrange
    keys = _keys(_frame())
    baseline = keys.key(2021, {"tr": None})

    # Act
    key = keys.key(2021, {"tr": _TR})

    # Assert
    assert key != baseline


def test_a_key_changes_when_a_season_frame_value_changes() -> None:
    # Arrange
    keys = _keys(_frame())
    baseline = keys.key(2021, {"tr": _TR})

    # Act
    key = keys.key(2021, {"tr": _TR.with_columns(pl.lit(1.5).alias("predictive_rating"))})

    # Assert
    assert key != baseline


def test_a_key_changes_with_the_first_season_in_the_context() -> None:
    # Arrange
    baseline = _key(_frame(), 2021)

    # Act
    key = _key(_frame(), 2021, min_season=2021)

    # Assert
    assert key != baseline


def test_a_key_changes_with_a_flag_in_the_context() -> None:
    # Arrange
    baseline = _key(_frame(), 2021, blend=True)

    # Act
    key = _key(_frame(), 2021, blend=False)

    # Assert
    assert key != baseline


def test_a_key_changes_with_the_season() -> None:
    # Arrange
    keys = _keys(_frame())
    baseline = keys.key(2021, {"tr": None})

    # Act
    key = keys.key(2022, {"tr": None})

    # Assert
    assert key != baseline


def test_a_key_changes_when_the_frame_turns_empty() -> None:
    # Arrange
    frame = _frame()
    baseline = _key(frame.filter(pl.col("season") == 2022), 2021)

    # Act
    key = _key(frame.clear(), 2021)

    # Assert
    assert key != baseline


def test_a_key_is_stable_for_a_frame_without_a_season_column() -> None:
    # Arrange
    no_season = _frame().drop("season")
    first = _key(no_season, 2021)

    # Act
    second = _key(no_season, 2021)

    # Assert
    assert second == first


def test_a_key_changes_with_any_row_of_a_frame_without_a_season_column() -> None:
    # Arrange
    no_season = _frame().drop("season")
    baseline = _key(no_season, 2021)

    # Act
    key = _key(no_season.with_columns(pl.lit(7).alias("week")), 2021)

    # Assert
    assert key != baseline


def test_a_key_changes_with_the_code(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    baseline = _key(_frame(), 2021)
    monkeypatch.setattr(season_cache, "etl_code_fingerprint", lambda: "different code")

    # Act
    key = _key(_frame(), 2021)

    # Assert
    assert key != baseline


def test_a_key_changes_with_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    baseline = _key(_frame(), 2021)
    monkeypatch.setattr(season_cache, "environment_fingerprint", lambda: {"polars": "0.0.0"})

    # Act
    key = _key(_frame(), 2021)

    # Assert
    assert key != baseline


def _fake_package(tmp_path: Path) -> Path:
    """Write a minimal ``nfl_predictor`` package tree with ETL and model modules."""
    package = tmp_path / "nfl_predictor"
    (package / "utils" / "polars").mkdir(parents=True)
    (package / "ml").mkdir()
    for name in ("data_collection.py", "constants.py", "ml/model.py"):
        (package / name).write_text("x = 1\n", encoding="utf-8")
    (package / "utils" / "polars" / "features.py").write_text("y = 1\n", encoding="utf-8")
    return package


def test_the_code_fingerprint_covers_the_etl_modules(tmp_path: Path) -> None:
    # Arrange
    package = _fake_package(tmp_path)
    baseline = season_cache.etl_code_fingerprint(package)
    (package / "utils" / "polars" / "features.py").write_text("y = 2\n", encoding="utf-8")
    edited = season_cache.etl_code_fingerprint(package)
    (package / "constants.py").write_text("x = 2\n", encoding="utf-8")

    # Act
    constants_edited = season_cache.etl_code_fingerprint(package)

    # Assert
    assert len({baseline, edited, constants_edited}) == 3


def test_the_code_fingerprint_ignores_the_model_modules(tmp_path: Path) -> None:
    # Arrange
    package = _fake_package(tmp_path)
    (package / "utils" / "polars" / "features.py").write_text("y = 2\n", encoding="utf-8")
    (package / "constants.py").write_text("x = 2\n", encoding="utf-8")
    constants_edited = season_cache.etl_code_fingerprint(package)
    (package / "ml" / "model.py").write_text("z = 3\n", encoding="utf-8")

    # Act
    model_edited = season_cache.etl_code_fingerprint(package)

    # Assert
    assert model_edited == constants_edited


def test_the_environment_fingerprint_names_the_libraries_and_the_thread_count() -> None:
    # Act
    environment = season_cache.environment_fingerprint()

    # Assert
    assert environment["polars"] == pl.__version__
    assert environment["polars_threads"] == str(pl.thread_pool_size())
    assert {"python", "numpy", "machine"} <= set(environment)


def test_the_environment_fingerprint_names_the_polars_runtime_package() -> None:
    # Arrange
    runtimes = [
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in metadata.distributions()
        if distribution.metadata["Name"].lower().startswith("polars-runtime")
    ]
    assert runtimes, "Polars installs its compiled runtime as a polars-runtime-* package"

    # Act
    runtime = season_cache.environment_fingerprint()["polars_runtime"]

    # Assert
    assert runtime == ",".join(sorted(runtimes))


def _build() -> season_cache.SeasonBuild:
    return season_cache.SeasonBuild(
        games=_frame(),
        snapshots=pl.DataFrame(
            {"season": [2021], "week": [1], "team_abbr": ["AAA"], "adj_srs": [0.25]}
        ),
    )


def _assert_same_build(left: season_cache.SeasonBuild, right: season_cache.SeasonBuild) -> None:
    assert_frame_equal(left.games, right.games, check_exact=True)
    assert_frame_equal(left.snapshots, right.snapshots, check_exact=True)


def test_a_stored_build_reads_back_exactly(tmp_path: Path) -> None:
    # Arrange
    cache = season_cache.SeasonCache(tmp_path / "seasons")
    cache.store(2021, "key-a", _build())

    # Act
    loaded = cache.load(2021, "key-a")

    # Assert
    assert loaded is not None
    _assert_same_build(loaded, _build())


def test_an_empty_build_reads_back_exactly(tmp_path: Path) -> None:
    # Arrange
    cache = season_cache.SeasonCache(tmp_path)
    empty = season_cache.SeasonBuild(games=pl.DataFrame(), snapshots=_build().snapshots.clear())
    cache.store(2021, "key-a", empty)

    # Act
    loaded = cache.load(2021, "key-a")

    # Assert
    assert loaded is not None
    _assert_same_build(loaded, empty)


@pytest.mark.parametrize(
    ("season", "key"),
    [
        pytest.param(2021, "key-b", id="different_key"),
        pytest.param(2020, "key-a", id="different_season"),
    ],
)
def test_a_different_key_or_season_is_a_miss(tmp_path: Path, season: int, key: str) -> None:
    # Arrange
    cache = season_cache.SeasonCache(tmp_path)
    cache.store(2021, "key-a", _build())

    # Act
    loaded = cache.load(season, key)

    # Assert
    assert loaded is None


def test_storing_a_season_again_replaces_its_entry(tmp_path: Path) -> None:
    # Arrange
    cache = season_cache.SeasonCache(tmp_path)
    cache.store(2021, "key-a", _build())
    replacement = season_cache.SeasonBuild(games=_frame().head(2), snapshots=_build().snapshots)
    cache.store(2021, "key-b", replacement)

    # Act
    loaded = cache.load(2021, "key-b")

    # Assert
    assert loaded is not None
    _assert_same_build(loaded, replacement)


def test_storing_a_season_again_drops_the_old_key(tmp_path: Path) -> None:
    # Arrange
    cache = season_cache.SeasonCache(tmp_path)
    cache.store(2021, "key-a", _build())
    replacement = season_cache.SeasonBuild(games=_frame().head(2), snapshots=_build().snapshots)
    cache.store(2021, "key-b", replacement)

    # Act
    loaded = cache.load(2021, "key-a")

    # Assert
    assert loaded is None


def _entry_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


@pytest.mark.parametrize("damage", ["truncate", "garbage", "delete"])
def test_a_damaged_entry_is_a_miss(tmp_path: Path, damage: str) -> None:
    # Arrange
    cache = season_cache.SeasonCache(tmp_path)
    cache.store(2021, "key-a", _build())

    # Act
    loads: list[season_cache.SeasonBuild | None] = []
    for path in _entry_files(tmp_path):
        if damage == "delete":
            path.unlink()
        elif damage == "truncate":
            path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])
        else:
            path.write_bytes(b"not a cache entry")
        loads.append(cache.load(2021, "key-a"))
        cache.store(2021, "key-a", _build())

    # Assert
    assert loads
    assert loads == [None] * len(loads)


def test_a_data_file_with_a_valid_manifest_but_other_content_is_a_miss(tmp_path: Path) -> None:
    # Arrange
    cache = season_cache.SeasonCache(tmp_path)
    cache.store(2021, "key-a", _build())
    other = season_cache.SeasonCache(tmp_path / "other")
    other.store(2021, "key-a", season_cache.SeasonBuild(games=_frame().head(1), snapshots=_frame()))
    games = next(path for path in _entry_files(tmp_path / "other") if "games" in path.name)
    target = next(
        path
        for path in _entry_files(tmp_path)
        if "games" in path.name and "other" not in path.parts
    )
    target.write_bytes(games.read_bytes())

    # Act
    loaded = cache.load(2021, "key-a")

    # Assert
    assert loaded is None


def test_a_manifest_from_another_format_version_is_a_miss(tmp_path: Path) -> None:
    # Arrange
    cache = season_cache.SeasonCache(tmp_path)
    cache.store(2021, "key-a", _build())
    manifest = next(path for path in _entry_files(tmp_path) if path.suffix == ".json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["format"] = -1
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    # Act
    loaded = cache.load(2021, "key-a")

    # Assert
    assert loaded is None


def test_a_missing_cache_directory_is_a_miss(tmp_path: Path) -> None:
    # Arrange
    cache = season_cache.SeasonCache(tmp_path / "absent")

    # Act
    loaded = cache.load(2021, "key-a")

    # Assert
    assert loaded is None


def test_a_store_that_cannot_write_does_not_raise(tmp_path: Path) -> None:
    # Arrange
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the cache directory should be", encoding="utf-8")
    cache = season_cache.SeasonCache(blocker)

    # Act
    cache.store(2021, "key-a", _build())

    # Assert
    assert cache.load(2021, "key-a") is None


_UNRENDERABLE_COLUMNS = [
    pl.concat_list("value", "value"),
    pl.struct("value", "count"),
    pl.concat_list("value", "value").list.to_array(2),
    pl.duration(seconds="count"),
    pl.col("team_abbr").cast(pl.Binary),
    pl.col("value").map_elements(lambda value: (value,), return_dtype=pl.Object),
]
_UNRENDERABLE_IDS = ["list", "struct", "array", "duration", "binary", "object"]


@pytest.mark.parametrize("column", _UNRENDERABLE_COLUMNS, ids=_UNRENDERABLE_IDS)
def test_a_column_the_value_digest_cannot_render_leaves_the_season_without_a_key(
    column: pl.Expr,
) -> None:
    # Arrange
    unrenderable = _frame().with_columns(column.alias("extra"))

    # Act
    key = _keys(unrenderable).key(2021, {"tr": None})

    # Assert
    assert key is None


@pytest.mark.parametrize("column", _UNRENDERABLE_COLUMNS, ids=_UNRENDERABLE_IDS)
def test_a_season_frame_the_value_digest_cannot_render_leaves_the_season_without_a_key(
    column: pl.Expr,
) -> None:
    # Arrange
    unrenderable = _frame().with_columns(column.alias("extra"))
    season_frame = unrenderable.filter(pl.col("season") == 2021)

    # Act
    key = _keys(_frame()).key(2021, {"tr": season_frame})

    # Assert
    assert key is None


def test_frames_the_value_digest_can_render_give_the_season_a_key() -> None:
    # Act
    key = _keys(_frame()).key(2021, {"tr": None})

    # Assert
    assert key is not None


def test_a_season_left_without_a_key_is_logged_as_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A season that can never be reused is reported at WARNING, not only at INFO."""
    # Arrange
    unrenderable = _frame().with_columns(pl.struct("value", "count").alias("extra"))
    caplog.set_level(logging.INFO)

    # Act
    key = _keys(unrenderable).key(2021, {"tr": None})

    # Assert
    assert key is None
    records = [record for record in caplog.records if "no value digest" in record.getMessage()]
    assert [record.levelno for record in records] == [logging.WARNING]
