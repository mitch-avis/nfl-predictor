"""Per-season cache of the ETL's season builds, for the incremental rebuild.

A season's build is its game rows and its weekly strength snapshots, as
``data_collection.process_season`` produces them. For a finished season those depend on the
code, the library environment, the run's options and the values of the input rows from that
season and earlier, so ``SeasonKeys`` hashes exactly those and the cache returns a stored build
only when the key matches. Anything else, including a later season's rows, cannot change the
key, and nothing else can change the build: a reused build equals a full rebuild's even after
later seasons gain rows. That holds only while a season's build reads no later season, which
includes where Polars splits the run-wide frames into chunks, since those boundaries move with
the frames' total length. A reduction over an eagerly filtered slice sums chunk by chunk, so
season-build code rechunks such a slice before reducing it (``calculate_league_means`` does).

Every input frame is digested by value, not by memory layout: each season's rows are written
as CSV (exact float text, nulls distinct from empty strings) and hashed with the frame's
schema, the order rows appear in across seasons and whether the frame has any rows at all.

The cache directory is disposable. A missing, stale, unreadable or tampered entry is a miss,
and a failed write leaves at worst an entry that later reads as a miss; neither ever stops
the ETL.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor.utils.logger import log

if TYPE_CHECKING:
    from collections.abc import Mapping

# Bumped whenever the entry layout or the key payload changes shape.
FORMAT_VERSION = 1

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST = "manifest.json"
_GAMES = "games.arrow"
_SNAPSHOTS = "snapshots.arrow"
# A season build reads Polars and NumPy (the strength ridge); their versions can move results.
_LIBRARIES = ("polars", "numpy")
# The modules a season build runs: the orchestration, the constants and every utility module.
_ETL_SOURCES = ("data_collection.py", "constants.py")
_ETL_SOURCE_DIRECTORIES = ("utils",)


@dataclass(frozen=True)
class SeasonBuild:
    """One season's game rows and its strength snapshots, stacked in the order recorded."""

    games: pl.DataFrame
    snapshots: pl.DataFrame


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def etl_code_fingerprint(package_root: Path = _PACKAGE_ROOT) -> str:
    """Return a hash of the source of every module a season build runs.

    Args:
        package_root: The ``nfl_predictor`` package directory.

    Returns:
        The SHA-256 of each file's package-relative path and bytes, in path order.

    """
    files = [package_root / name for name in _ETL_SOURCES]
    for directory in _ETL_SOURCE_DIRECTORIES:
        files.extend((package_root / directory).rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda path: path.relative_to(package_root).as_posix()):
        digest.update(path.relative_to(package_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(_sha256(path.read_bytes()).encode() if path.is_file() else b"missing")
        digest.update(b"\0")
    return digest.hexdigest()


def environment_fingerprint() -> dict[str, str]:
    """Return the runtime facts a season build's floats can depend on.

    Polars splits large sums into per-thread partial sums, so its thread count is part of
    the result, as are the library versions, the Python version and the CPU architecture.
    """
    environment = {name: metadata.version(name) for name in _LIBRARIES}
    # Polars ships its compiled engine as a separate polars-runtime-* package.
    environment["polars_runtime"] = ",".join(
        sorted(
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in metadata.distributions()
            if distribution.metadata["Name"].lower().startswith("polars-runtime")
        )
    )
    environment["python"] = sys.version
    environment["machine"] = platform.machine()
    environment["polars_threads"] = str(pl.thread_pool_size())
    return environment


def _schema_text(frame: pl.DataFrame) -> str:
    return json.dumps([[name, str(dtype)] for name, dtype in frame.schema.items()])


def _rows_digest(frame: pl.DataFrame) -> str:
    """Hash a frame's rows by value: every column's exact text, in row order.

    Non-numeric values are always quoted, so a null (written bare and empty) never reads
    the same as an empty string, and floats are written at full precision, signed zero and
    NaN included. The schema is hashed alongside because the text alone cannot tell an
    integer column from a float one with the same values.
    """
    text = frame.write_csv(quote_style="non_numeric") if frame.width else ""
    return _sha256(f"{_schema_text(frame)}\n{text}".encode())


def frame_digest(frame: pl.DataFrame | None) -> str:
    """Return the value digest of a whole frame, or a fixed marker for no frame."""
    return "none" if frame is None else _rows_digest(frame)


class _ThroughSeasonDigests:
    """Digest each season's prefix of one frame: the rows from that season and earlier."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self._has_season = "season" in frame.columns
        self._header = {"schema": _schema_text(frame), "has_rows": frame.height > 0}
        self._partitions: dict[int, str] = {}
        self._whole: str | None = None

    def through(self, season: int) -> dict[str, object]:
        if not self._has_season:
            if self._whole is None:
                self._whole = _rows_digest(self.frame)
            return {**self._header, "rows": self._whole}
        prefix = self.frame.filter(pl.col("season") <= season)
        seasons = prefix.get_column("season")
        partitions = []
        for value in sorted(seasons.unique().to_list()):
            if value not in self._partitions:
                self._partitions[value] = _rows_digest(self.frame.filter(pl.col("season") == value))
            partitions.append([value, self._partitions[value]])
        # Each season's rows are digested in their own order; the order the seasons
        # interleave in is the season column of the prefix.
        return {
            **self._header,
            "partitions": partitions,
            "interleave": _rows_digest(seasons.to_frame()),
        }


class SeasonKeys:
    """Cache keys for every season of one ETL run.

    The run-wide frames (the schedule, the team stats and the ELO ratings) contribute only
    their rows through the keyed season, plus their schema and whether they have rows at
    all. Frames that belong to one season (that season's TeamRankings, for example) are
    passed to ``key`` and contribute whole.
    """

    def __init__(self, frames: Mapping[str, pl.DataFrame], context: Mapping[str, object]) -> None:
        """Remember the run-wide frames and the options, code and environment of the run.

        Args:
            frames: The run-wide input frames, by name.
            context: Every run option that can change a season's build, JSON-serializable.

        """
        self._frames = {name: _ThroughSeasonDigests(frame) for name, frame in frames.items()}
        self._context = {
            "format": FORMAT_VERSION,
            "options": dict(context),
            "code": etl_code_fingerprint(),
            "environment": environment_fingerprint(),
        }

    def key(self, season: int, season_frames: Mapping[str, pl.DataFrame | None]) -> str | None:
        """Return the key of one season's build.

        Args:
            season: The season being built.
            season_frames: The frames that belong to this season alone, by name.

        Returns:
            A SHA-256 hex digest, or None when an input has a column the value digest cannot
            render as CSV (a list, struct, array, duration, binary or object column); such a
            season is never cached.

        """
        try:
            payload = {
                **self._context,
                "season": season,
                "frames": {name: digests.through(season) for name, digests in self._frames.items()},
                "season_frames": {
                    name: frame_digest(frame) for name, frame in season_frames.items()
                },
            }
        except pl.exceptions.ComputeError as error:
            log.warning(
                "Season cache: %d has an input with no value digest, so it is never reused (%s)",
                season,
                error,
            )
            return None
        return _sha256(json.dumps(payload, sort_keys=True, default=str).encode())


# What reading an entry can raise when the entry is missing, partial or not an entry at all.
_READ_ERRORS = (OSError, ValueError, KeyError, TypeError, pl.exceptions.PolarsError)


class SeasonCache:
    """Season builds on disk, one entry per season, each validated by key and checksum."""

    def __init__(self, directory: Path) -> None:
        """Use ``directory`` for the entries; it is created on the first store."""
        self.directory = directory

    def _entry(self, season: int) -> Path:
        return self.directory / f"season_{season}"

    def load(self, season: int, key: str) -> SeasonBuild | None:
        """Return the stored build for ``season`` when its key is ``key``, else None."""
        entry = self._entry(season)
        try:
            manifest = json.loads((entry / _MANIFEST).read_text(encoding="utf-8"))
            if (
                manifest["format"] != FORMAT_VERSION
                or manifest["season"] != season
                or manifest["key"] != key
            ):
                log.info("Season cache: %d is stale; rebuilding it", season)
                return None
            frames = {}
            for name in (_GAMES, _SNAPSHOTS):
                data = (entry / name).read_bytes()
                if _sha256(data) != manifest["files"][name]:
                    log.warning("Season cache: %s for %d fails its checksum", name, season)
                    return None
                frames[name] = pl.read_ipc(data, memory_map=False)
        except FileNotFoundError:
            log.info("Season cache: no entry for %d", season)
            return None
        except _READ_ERRORS as error:
            log.warning("Season cache: unreadable entry for %d (%s)", season, error)
            return None
        return SeasonBuild(games=frames[_GAMES], snapshots=frames[_SNAPSHOTS])

    def store(self, season: int, key: str, build: SeasonBuild) -> None:
        """Write ``build`` as the entry for ``season``; a failure is logged, not raised.

        The manifest is removed first and written last, so an interrupted store leaves an
        entry that reads as a miss.
        """
        entry = self._entry(season)
        try:
            entry.mkdir(parents=True, exist_ok=True)
            (entry / _MANIFEST).unlink(missing_ok=True)
            checksums = {}
            for name, frame in ((_GAMES, build.games), (_SNAPSHOTS, build.snapshots)):
                path = entry / name
                frame.write_ipc(path)
                checksums[name] = _sha256(path.read_bytes())
            manifest = {"format": FORMAT_VERSION, "season": season, "key": key, "files": checksums}
            staged = entry / f"{_MANIFEST}.tmp"
            staged.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            staged.replace(entry / _MANIFEST)
        except (OSError, pl.exceptions.PolarsError) as error:
            log.warning("Season cache: could not store %d (%s)", season, error)
            shutil.rmtree(entry, ignore_errors=True)
