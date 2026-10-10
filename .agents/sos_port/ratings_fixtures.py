"""Cut the ratings parity test fixtures from the published ``nfl-sos-ratings`` files.

Run from the repository root:

    .venv/bin/python .agents/sos_port/ratings_fixtures.py [--source PATH]

``--source`` is the ``nfl-sos-ratings`` checkout (default: the one beside this repository's main
checkout). The script asserts that checkout is at the frozen commit, reads its published Parquet
files without writing there, and writes under ``tests/fixtures/sos_ratings_parity/``:

- one Parquet file per cut, named like its source (``{season}_{name}.parquet``): the game logs cut
  to the columns the ratings read, and published ratings tables cut to a subset of their rows;
- ``manifest.json``: per fixture its source file and that file's SHA-256, the columns and rows
  kept, its row count, and its own SHA-256, which ``tests/test_ratings_published_parity.py``
  checks;
- ``PROVENANCE.md``: the same, as a table.

The fixtures feed the fast parity tests: 1999 (the first season, cross-validating its own
penalties, with the published rank ranges and pairs), 1999-2003 (the first season with a
preseason prior, and its week-by-week history), and the 2024 quarterbacks (a recent season with
passers who played without a dropback). The full comparison of every season is
``ratings_parity.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ratings_source import (  # a sibling module, importable once its dir is on the path
    check_source,
    default_source,
    published_file,
    repo_root,
    sha256_file,
)

if TYPE_CHECKING:
    from collections.abc import Callable

FIXTURE_DIR = repo_root() / "tests" / "fixtures" / "sos_ratings_parity"
TEAM_LOG_COLUMNS = (
    "game_id",
    "week",
    "team",
    "opponent_team",
    "is_home",
    "offensive_snaps",
    "offensive_epa",
    "st_plays",
    "st_epa",
    "point_margin",
)
QB_LOG_COLUMNS = (
    "game_id",
    "week",
    "team",
    "opponent_team",
    "is_home",
    "qb_id",
    "qb_name",
    "qb_dropbacks",
    "qb_attempts",
    "qb_passing_epa",
    "qb_spike_epa",
)
# Every fourth row of a ranges table (by published rank) keeps ranks across the whole range.
_RANGE_STRIDE = 4
_HISTORY_WEEKS = (1, 2, 9)


@dataclass(frozen=True, slots=True)
class Cut:
    """One fixture: a published file, the columns kept, and which rows."""

    season: int
    name: str
    columns: tuple[str, ...] | None
    rows: str
    keep: Callable[[pl.DataFrame, Path], pl.DataFrame]


def _every_row(frame: pl.DataFrame, _source: Path) -> pl.DataFrame:
    """Keep every row."""
    return frame


def _every_fourth(frame: pl.DataFrame, _source: Path) -> pl.DataFrame:
    """Keep every fourth row, starting with the first."""
    return frame.gather_every(_RANGE_STRIDE)


def _kept_ids(source: Path, season: int, ranges: str, id_column: str) -> list[str]:
    """Return the ids the ranges cut keeps for ``season``."""
    published = pl.read_parquet(published_file(source, season, ranges))
    return published.gather_every(_RANGE_STRIDE).get_column(id_column).to_list()


def _pairs_among(ranges: str, id_column: str, other_column: str, season: int) -> Callable:
    """Return a cut keeping the pairs whose two units are both in the ranges cut."""

    def keep(frame: pl.DataFrame, source: Path) -> pl.DataFrame:
        ids = _kept_ids(source, season, ranges, id_column)
        return frame.filter(pl.col(id_column).is_in(ids) & pl.col(other_column).is_in(ids))

    return keep


def _history_weeks(frame: pl.DataFrame, _source: Path) -> pl.DataFrame:
    """Keep weeks 1, 2, 9, and the last week."""
    last = frame.get_column("week").max()
    return frame.filter(pl.col("week").is_in([*_HISTORY_WEEKS, last]))


CUTS: tuple[Cut, ...] = (
    *(
        Cut(season, "team_game_logs", TEAM_LOG_COLUMNS, "every row", _every_row)
        for season in (1999, 2000, 2001, 2002, 2003, 2024)
    ),
    *(
        Cut(season, "qb_game_logs", QB_LOG_COLUMNS, "every row", _every_row)
        for season in (1999, 2024)
    ),
    Cut(1999, "ratings", None, "every row", _every_row),
    Cut(1999, "rating_ranges", None, "every fourth row by published rank", _every_fourth),
    Cut(
        1999,
        "rating_pairs",
        None,
        "pairs of the teams in the rating_ranges cut",
        _pairs_among("rating_ranges", "team", "other_team", 1999),
    ),
    Cut(1999, "qb_ratings", None, "every row", _every_row),
    Cut(1999, "qb_ratings_by_week", None, "weeks 1, 2, 9, and the last", _history_weeks),
    Cut(1999, "qb_rating_ranges", None, "every fourth row by published rank", _every_fourth),
    Cut(
        1999,
        "qb_rating_pairs",
        None,
        "pairs of the passers in the qb_rating_ranges cut",
        _pairs_among("qb_rating_ranges", "qb_id", "other_qb_id", 1999),
    ),
    Cut(2003, "ratings", None, "every row", _every_row),
    Cut(2003, "ratings_by_week", None, "every row", _every_row),
    Cut(2024, "qb_ratings", None, "every row", _every_row),
    Cut(2024, "qb_ratings_by_week", None, "weeks 1, 2, 9, and the last", _history_weeks),
    Cut(2024, "qb_rating_ranges", None, "every fourth row by published rank", _every_fourth),
    Cut(
        2024,
        "qb_rating_pairs",
        None,
        "pairs of the passers in the qb_rating_ranges cut",
        _pairs_among("qb_rating_ranges", "qb_id", "other_qb_id", 2024),
    ),
)


def _write_provenance(manifest: dict[str, object], files: list[dict[str, object]]) -> None:
    """Write ``PROVENANCE.md``: the manifest as a table."""
    header = "| fixture | source file (SHA-256, first 12) | columns | rows kept | rows | SHA-256 |"
    lines = [
        "# Ratings parity fixtures",
        "",
        "Generated by `.agents/sos_port/ratings_fixtures.py`; do not edit by hand. Each file is",
        "cut from a published `nfl-sos-ratings` Parquet file, read at the checkout's commit",
        f"`{manifest['source_commit']}`. `manifest.json` holds the same records with full",
        "hashes, and `tests/test_ratings_published_parity.py` checks every fixture against its",
        "hash. The SHA-256 columns show the first 12 characters.",
        "",
        header,
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for record in files:
        columns = record["columns"]
        column_text = "every column" if columns is None else ", ".join(f"`{c}`" for c in columns)
        lines.append(
            f"| `{record['fixture']}` | `{record['source_file']}` "
            f"(`{str(record['source_sha256'])[:12]}`) | {column_text} | {record['rows']} | "
            f"{record['row_count']} | `{str(record['sha256'])[:12]}` |"
        )
    (FIXTURE_DIR / "PROVENANCE.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    """Write every fixture, the manifest, and the provenance table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=None, help="nfl-sos-ratings checkout")
    args = parser.parse_args(argv)
    source = (args.source or default_source()).resolve()
    head = check_source(source)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    for cut in CUTS:
        path = published_file(source, cut.season, cut.name)
        frame = pl.read_parquet(path)
        if cut.columns is not None:
            frame = frame.select(cut.columns)
        frame = cut.keep(frame, source)
        fixture = FIXTURE_DIR / path.name
        frame.write_parquet(fixture)
        files.append(
            {
                "fixture": fixture.name,
                "source_file": f"data/{path.name}",
                "source_sha256": sha256_file(path),
                "columns": None if cut.columns is None else list(cut.columns),
                "rows": cut.rows,
                "row_count": frame.height,
                "sha256": sha256_file(fixture),
            }
        )
    manifest: dict[str, object] = {
        "source_repository": "nfl-sos-ratings",
        "source_commit": head,
        "generator": ".agents/sos_port/ratings_fixtures.py",
        "files": files,
    }
    (FIXTURE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_provenance(manifest, files)
    total = sum((FIXTURE_DIR / str(record["fixture"])).stat().st_size for record in files)
    sys.stdout.write(f"wrote {len(files)} fixtures ({total:,} bytes) to {FIXTURE_DIR}\n")


if __name__ == "__main__":
    main()
