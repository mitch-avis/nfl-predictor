"""Measure facts about nflverse play-by-play that the column decisions rest on.

Run from the root of an ``nfl-predictor`` checkout (network access needed unless ``--raw-dir``
holds the seasons; nothing is written except under ``--work-dir``):

    .venv/bin/python .agents/sos_port/decisions/pbp_column_facts.py --work-dir "$(mktemp -d)"

Over the regular-season plays of each season (downloaded into memory with nflreadpy, or read
from ``--raw-dir``) it reports:

- every raw column whose dtype differs between seasons;
- for each column of ``constants.PBP_TEAM_COLUMNS``, the values that ``ALIAS_TO_CANONICAL``
  changes, the values it does not know (they pass through normalization unchanged), and the
  empty strings (``nfl-sos-ratings`` turns them into nulls; this cache keeps them);
- for each yard-line text column, the plays whose opening team code normalization would change;
- for each field ``nfl-sos-ratings`` blanks in the seasons it judges nflverse lacks it
  (``SOS_FIELD_GAPS``), the plays with a value, and with a nonzero or nonempty value, in each of
  those seasons and in the first season after them;
- for ``--size-season``, the size of its cache file written with the loader's own writer with
  every requested column, without ``desc``, and without the 28 fumble and tackle credit columns.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars import loaders

YARD_LINE_COLUMNS = ("yrdln", "drive_start_yard_line", "drive_end_yard_line", "end_yard_line")
CREDIT_PREFIXES = ("fumble", "forced_fumble", "solo_tackle", "assist_tackle", "tackle_with")
# ``_PBP_FIELD_GAPS`` of ``nfl_sos_ratings/data_loader.py`` at 9081bce: the seasons whose values
# of each field its loader sets to null.
_BEFORE_PASS_DEPTH = range(1999, 2006)
SOS_FIELD_GAPS: dict[str, range] = {
    "air_yards": _BEFORE_PASS_DEPTH,
    "air_epa": _BEFORE_PASS_DEPTH,
    "pass_length": _BEFORE_PASS_DEPTH,
    "yards_after_catch": _BEFORE_PASS_DEPTH,
    "yac_epa": _BEFORE_PASS_DEPTH,
    "xyac_mean_yardage": _BEFORE_PASS_DEPTH,
    "qb_hit": range(2003, 2006),
    "no_huddle": range(1999, 2003),
    "drive_yards_penalized": range(1999, 2001),
}


def _gap_row(field: str, season: int, plays: pl.DataFrame) -> str:
    """Return one line: the plays with a value, and with a nonzero or nonempty value, of a field."""
    column = plays.get_column(field)
    with_value = column.len() - column.null_count()
    if column.dtype == pl.String:
        meaningful = int((column.fill_null("") != "").sum())
    else:
        meaningful = int((column.fill_null(0) != 0).sum())
    window = "blanked" if season in SOS_FIELD_GAPS[field] else "first season kept"
    return (
        f"{field} {season} ({window}): {plays.height} plays, {with_value} with a value, "
        f"{meaningful} nonzero or nonempty\n"
    )


def _raw_season(season: int, raw_dir: Path | None) -> pl.DataFrame:
    """Return one season's raw play-by-play from ``raw_dir`` when saved there, else nflreadpy."""
    path = raw_dir / f"play_by_play_{season}.parquet" if raw_dir else None
    if path is not None and path.exists():
        return pl.read_parquet(path)
    return nfl.load_pbp(seasons=[season])


def _changed_codes() -> list[str]:
    """Return the team codes normalization rewrites."""
    return [code for code, canonical in constants.ALIAS_TO_CANONICAL.items() if code != canonical]


def _size(frame: pl.DataFrame, columns: list[str], path: Path) -> int:
    """Write ``frame[columns]`` with the loader's cache writer and return the file size."""
    loaders._write_cached_frame(frame.select(columns), path, columns)
    return path.stat().st_size


def main() -> int:
    """Print the facts and return the exit status."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--work-dir", type=Path, required=True, help="Scratch directory (written).")
    parser.add_argument("--raw-dir", type=Path, help="Raw season files to reuse.")
    parser.add_argument("--first-season", type=int, default=constants.NFLREADPY_MIN_SEASON)
    parser.add_argument("--last-season", type=int, default=2025)
    parser.add_argument("--size-season", type=int, default=2024)
    arguments = parser.parse_args()
    arguments.work_dir.mkdir(parents=True, exist_ok=True)
    changed = _changed_codes()
    known = set(constants.ALIAS_TO_CANONICAL)
    dtypes: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    team_counts: dict[str, int] = dict.fromkeys(constants.PBP_TEAM_COLUMNS, 0)
    team_codes: dict[str, set[str]] = defaultdict(set)
    unknown: dict[str, set[str]] = defaultdict(set)
    empty: dict[str, int] = dict.fromkeys(constants.PBP_TEAM_COLUMNS, 0)
    yard_counts: dict[str, int] = dict.fromkeys(YARD_LINE_COLUMNS, 0)
    yard_codes: dict[str, set[str]] = defaultdict(set)
    gap_lines: list[str] = []
    sizes: dict[str, int] = {}
    for season in range(arguments.first_season, arguments.last_season + 1):
        raw = _raw_season(season, arguments.raw_dir)
        for name, dtype in raw.schema.items():
            dtypes[name][str(dtype)].append(season)
        plays = raw.filter(pl.col("season_type") == "REG")
        for column in constants.PBP_TEAM_COLUMNS:
            values = plays.get_column(column).drop_nulls()
            hits = values.is_in(changed)
            team_counts[column] += int(hits.sum())
            team_codes[column] |= set(values.filter(hits).unique().to_list())
            unknown[column] |= {value for value in values.unique().to_list() if value not in known}
            empty[column] += int((values == "").sum())
        gap_lines.extend(
            _gap_row(field, season, plays)
            for field, gap in SOS_FIELD_GAPS.items()
            if season in gap or season == gap.stop
        )
        for column in YARD_LINE_COLUMNS:
            code = plays.get_column(column).str.extract(r"^([A-Za-z]+) ", 1)
            hits = code.is_in(changed).fill_null(value=False)
            yard_counts[column] += int(hits.sum())
            yard_codes[column] |= set(code.filter(hits).unique().to_list())
        if season == arguments.size_season:
            fresh = loaders._prepare_pbp(raw, regular_season_only=True)
            every = list(fresh.columns)
            credits = [
                column
                for column in every
                if column.startswith(CREDIT_PREFIXES) and column.endswith(("_team", "_player_id"))
            ]
            sizes = {
                f"every requested column ({len(every)})": _size(
                    fresh, every, arguments.work_dir / "every.parquet"
                ),
                "without desc": _size(
                    fresh,
                    [c for c in every if c != "desc"],
                    arguments.work_dir / "without_desc.parquet",
                ),
                f"without the {len(credits)} fumble and tackle credit columns": _size(
                    fresh,
                    [c for c in every if c not in credits],
                    arguments.work_dir / "without_credits.parquet",
                ),
            }
    out = sys.stdout
    seasons = f"{arguments.first_season}-{arguments.last_season}"
    out.write(f"## Raw columns whose dtype differs between seasons, {seasons}\n\n")
    varying = {name: by for name, by in sorted(dtypes.items()) if len(by) > 1}
    out.writelines(f"{name}: {dict(by)}\n" for name, by in varying.items())
    out.write(f"({len(varying)} of {len(dtypes)} columns)\n")
    out.write(f"\n## Team-code columns, regular season {seasons}: values normalization changes\n\n")
    out.writelines(
        f"{column}: {team_counts[column]} values, codes {sorted(team_codes[column])}, "
        f"unknown to the alias table {sorted(unknown[column])}\n"
        for column in constants.PBP_TEAM_COLUMNS
    )
    out.write(f"\n## Team-code columns, regular season {seasons}: empty strings\n\n")
    out.writelines(f"{column}: {empty[column]}\n" for column in constants.PBP_TEAM_COLUMNS)
    out.write(
        f"\n## Fields nfl-sos-ratings blanks, regular season: the blanked seasons and the first "
        f"season after them\n\n"
    )
    field_order = list(SOS_FIELD_GAPS)
    out.writelines(
        sorted(gap_lines, key=lambda line: (field_order.index(line.split()[0]), line.split()[1]))
    )
    out.write(f"\n## Yard-line text, regular season {seasons}: plays whose code would change\n\n")
    out.writelines(
        f"{column}: {yard_counts[column]} plays, codes {sorted(yard_codes[column])}\n"
        for column in YARD_LINE_COLUMNS
    )
    out.write(f"\n## Cache file size of {arguments.size_season} (bytes)\n\n")
    out.writelines(f"{label}: {size}\n" for label, size in sizes.items())
    return 0


if __name__ == "__main__":
    sys.exit(main())
