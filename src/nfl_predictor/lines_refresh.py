"""Refresh market lines in the existing datasets without rerunning the full ETL.

The weekly ETL rebuilds every feature family, which takes minutes and rewrites every dataset. Lines
move daily, so this module re-reads the lines of a single season and writes back only the five
market columns the model consumes, matching rows on ``game_id`` (the season, week and teams in a
week's ``games_to_predict`` file). Completed-game datasets are never touched: their lines are
history.

The refresh follows the line source the data was built with, read from the build's line record
(``market_lines_metadata.json`` beside the datasets), so it writes what a rebuild would:

- ``stored``: the nflverse schedule lines, missing moneylines derived with the fixed conversion
  (``game_utils.spread_to_moneyline``). A data directory without a line record predates the
  record and is read as a stored build.
- ``pick_time``: the line order of ``pick_time_lines`` (an upcoming game takes nfelo's latest line,
  then nflverse's, then SurvivorGrid's; a completed game keeps its opener or stored fallback),
  with missing moneylines priced by the season's fitted map from the build's record.

Run it as ``nfl-predictor lines --season 2026 --week 2``. Because the market features feed the
model, a prediction run should follow a refresh.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import game_utils, polars_utils
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.narrowing import as_str_dict
from nfl_predictor.utils.polars import moneyline_map, nfelo_lines, pick_time_lines

LINE_COLUMNS: tuple[str, ...] = (
    "total_line",
    "away_spread",
    "home_spread",
    "away_moneyline",
    "home_moneyline",
)
JOIN_COLUMN = "game_id"
MATCHUP_COLUMNS: tuple[str, ...] = ("season", "week", "away_abbr", "home_abbr")
SEASON_COLUMN = "season"
DEFAULT_DATA_DIR = constants.DATA_PATH
LINE_RECORD_NAME = f"{constants.MARKET_LINES_METADATA_NAME}.json"
_INCOMING_PREFIX = "__incoming_"


@dataclass(frozen=True)
class BuildLines:
    """The line source a data directory was built with, and a pick-time build's fitted maps.

    Attributes:
        line_source: ``constants.LINE_SOURCE_STORED`` or ``constants.LINE_SOURCE_PICK_TIME``.
        moneyline_maps: Each built season's spread-to-moneyline map, ``None`` where the build
            used the fixed conversion; empty for a stored build.

    """

    line_source: str
    moneyline_maps: dict[int, moneyline_map.SpreadMoneylineMap | None] = field(
        default_factory=dict[int, moneyline_map.SpreadMoneylineMap | None]
    )


@dataclass(frozen=True)
class FileRefresh:
    """What the refresh did to one dataset file.

    Attributes:
        path: The dataset that was considered.
        exists: Whether the file was present; missing files are skipped, not an error.
        matched_rows: Rows of the target season whose ``game_id`` appeared in the schedule.
        changed_rows: Rows whose line values actually differ from what was on disk.
        changed_columns: Per-column count of changed cells.
        written: Whether the file was rewritten (only when something changed).

    """

    path: Path
    exists: bool
    matched_rows: int = 0
    changed_rows: int = 0
    changed_columns: dict[str, int] = field(default_factory=dict[str, int])

    @property
    def written(self) -> bool:
        """Return whether the file was rewritten."""
        return self.changed_rows > 0


@dataclass(frozen=True)
class LinesRefreshResult:
    """The outcome of a whole refresh."""

    season: int
    week: int
    schedule_games: int
    files: tuple[FileRefresh, ...]
    line_source: str = constants.LINE_SOURCE_STORED

    @property
    def changed_rows(self) -> int:
        """Return the total number of changed rows across every file."""
        return sum(refresh.changed_rows for refresh in self.files)

    @property
    def written_paths(self) -> tuple[Path, ...]:
        """Return the files that were rewritten."""
        return tuple(refresh.path for refresh in self.files if refresh.written)


def _target_paths(data_dir: Path, week: int) -> tuple[Path, ...]:
    """Return the datasets a refresh may rewrite, in the order they are processed."""
    return (
        data_dir / "predict" / f"week_{week:02d}_games_to_predict.csv",
        data_dir / "all_data_ml.csv",
        data_dir / "all_data.csv",
    )


def read_build_lines(data_dir: Path) -> BuildLines:
    """Return the line source the datasets in ``data_dir`` were built with.

    Args:
        data_dir: Directory holding the datasets and the build's line record.

    Returns:
        The record's line source, with each season's map for a pick-time build. Without a
        record the data predates it, which makes it a stored build.

    Raises:
        ValueError: If the record is not JSON or names an unknown line source.
        TypeError: If a pick-time record has no table of moneyline maps.

    """
    path = data_dir / LINE_RECORD_NAME
    if not path.is_file():
        log.warning(
            "No line record (%s) beside the data in %s; refreshing it as a %s-line build.",
            LINE_RECORD_NAME,
            data_dir,
            constants.LINE_SOURCE_STORED,
        )
        return BuildLines(constants.LINE_SOURCE_STORED)
    try:
        record = as_str_dict(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        msg = f"The line record {path} is not valid JSON: {exc}"
        raise ValueError(msg) from exc
    source = record.get("line_source") if record is not None else None
    if record is None or source not in constants.LINE_SOURCES:
        msg = (
            f"The line record {path} names line source {source!r}; expected one of "
            f"{', '.join(constants.LINE_SOURCES)}."
        )
        raise ValueError(msg)
    if source == constants.LINE_SOURCE_STORED:
        return BuildLines(constants.LINE_SOURCE_STORED)
    maps = as_str_dict(record.get("moneyline_maps"))
    if maps is None:
        msg = f"The pick-time line record {path} has no moneyline map table."
        raise TypeError(msg)
    return BuildLines(
        constants.LINE_SOURCE_PICK_TIME,
        {
            int(season): None
            if fitted is None
            else moneyline_map.SpreadMoneylineMap.from_dict(fitted)
            for season, fitted in maps.items()
        },
    )


def _require_season_map(season: int, build: BuildLines) -> None:
    """Refuse a pick-time refresh of a season the build's record holds no map entry for."""
    if season not in build.moneyline_maps:
        msg = (
            f"The pick-time build's line record has no moneyline map for season {season}; "
            "rebuild the data with nfl-predictor data before refreshing that season's lines."
        )
        raise ValueError(msg)


def _pick_time_lines(schedule: pl.DataFrame, season: int, build: BuildLines) -> pl.DataFrame:
    """Return the schedule with the lines a pick-time build gives ``season``'s games.

    The ETL's order: ``pick_time_lines.apply_pick_time_lines`` (nfelo's latest line, then the
    nflverse line, for an upcoming game), SurvivorGrid for an upcoming game still without a
    spread, then every missing moneyline priced with the season's map from the build, which
    ``load_lines`` has already checked is there.
    """
    maps = {season: build.moneyline_maps[season]}
    snapshot = nfelo_lines.load_nfelo_lines()
    log.info(
        "Pick-time lines: nfelo %s copy, %d games (sha256 %s)",
        snapshot.origin,
        snapshot.frame.height,
        snapshot.sha256,
    )
    lines, report = pick_time_lines.apply_pick_time_lines(schedule, snapshot.frame)
    for row in report.iter_rows(named=True):
        log.info(
            "Pick-time lines, season %d: upcoming %d nfelo / %d nflverse / %d without a line",
            row["season"],
            row["upcoming_nfelo"],
            row["upcoming_nflverse"],
            row["upcoming_without_line"],
        )

    def fill(frame: pl.DataFrame) -> pl.DataFrame:
        return moneyline_map.fill_moneylines_from_maps(frame, maps)

    return fill(game_utils.fill_future_game_lines(lines, fill_moneylines=fill))


def load_lines(
    season: int, *, cache_dir: Path | None = None, build: BuildLines | None = None
) -> pl.DataFrame:
    """Return one row per game of ``season`` carrying the current market lines.

    Args:
        season: Season to refresh.
        cache_dir: Optional nflreadpy cache directory override.
        build: The line source to follow; ``None`` is the stored source.

    Returns:
        A frame of the join keys (``game_id`` and the season/week/team matchup) plus whichever
        line columns the schedule provides. Stored: the schedule's lines, missing moneylines
        derived from the spreads with the fixed conversion. Pick time: see ``_pick_time_lines``.

    Raises:
        ValueError: If the schedule holds no games for the season, or a pick-time build has no
            moneyline map for it.

    """
    if build is not None and build.line_source == constants.LINE_SOURCE_PICK_TIME:
        # Checked before anything is downloaded.
        _require_season_map(season, build)
    schedule = polars_utils.load_schedule(
        [season],
        cache_dir=cache_dir,
        force_refresh=True,
        current_season=season,
    )
    if schedule.is_empty() or JOIN_COLUMN not in schedule.columns:
        msg = f"The schedule for season {season} has no games to refresh lines from."
        raise ValueError(msg)
    if build is not None and build.line_source == constants.LINE_SOURCE_PICK_TIME:
        schedule = _pick_time_lines(schedule, season, build)
    else:
        schedule = game_utils.fill_missing_moneylines(schedule)
    keys = [JOIN_COLUMN, *(c for c in MATCHUP_COLUMNS if c in schedule.columns)]
    present = [column for column in LINE_COLUMNS if column in schedule.columns]
    return schedule.select([*keys, *present]).unique(subset=JOIN_COLUMN, keep="first")


def join_keys(frame: pl.DataFrame, lines: pl.DataFrame) -> list[str]:
    """Return the columns to match ``frame`` rows against ``lines`` rows.

    ``game_id`` is preferred, but the weekly ``games_to_predict`` files are written without it, so
    the season, week, and both team abbreviations identify the game instead.
    """
    if JOIN_COLUMN in frame.columns and JOIN_COLUMN in lines.columns:
        return [JOIN_COLUMN]
    if all(column in frame.columns and column in lines.columns for column in MATCHUP_COLUMNS):
        return list(MATCHUP_COLUMNS)
    return []


def _refresh_column(frame: pl.DataFrame, column: str, in_season: pl.Expr) -> pl.Expr:
    """Return the expression that overlays incoming values onto ``column``."""
    incoming = f"{_INCOMING_PREFIX}{column}"
    dtype = frame.schema[column]
    if dtype == pl.Null or frame[column].null_count() == frame.height:
        dtype = frame.schema[incoming]
    return (
        pl.when(in_season & pl.col(incoming).is_not_null())
        .then(pl.col(incoming).cast(dtype, strict=False))
        .otherwise(pl.col(column).cast(dtype, strict=False))
        .alias(column)
    )


def _write_atomic(frame: pl.DataFrame, path: Path) -> None:
    """Write ``frame`` to ``path`` through a temporary file in the same directory."""
    tmp_path = path.with_name(f"{path.name}.tmp")
    frame.write_csv(tmp_path)
    tmp_path.replace(path)


def refresh_file(path: Path, lines: pl.DataFrame, season: int) -> FileRefresh:
    """Overlay ``lines`` onto one dataset, rewriting it only when a value changed.

    Args:
        path: Dataset to refresh.
        lines: Frame from :func:`load_lines`.
        season: Only rows of this season are updated.

    Returns:
        A :class:`FileRefresh` describing what changed.

    """
    if not path.is_file():
        log.info("Skipping %s: not on disk.", path)
        return FileRefresh(path=path, exists=False)
    frame = pl.read_csv(path, infer_schema_length=None)
    keys = join_keys(frame, lines)
    columns = [c for c in LINE_COLUMNS if c in lines.columns and c in frame.columns]
    if not keys or not columns:
        log.warning("Skipping %s: no usable join key or line columns.", path)
        return FileRefresh(path=path, exists=True)

    incoming = lines.select([*keys, *columns]).rename(
        {column: f"{_INCOMING_PREFIX}{column}" for column in columns}
    )
    joined = frame.join(
        incoming.cast({key: frame.schema[key] for key in keys}), on=keys, how="left"
    )
    in_season = (
        (pl.col(SEASON_COLUMN) == season) if SEASON_COLUMN in frame.columns else pl.lit(True)
    )
    updated = joined.with_columns(
        [_refresh_column(joined, column, in_season) for column in columns]
    )

    matched = int(
        joined.select(
            (in_season & pl.col(f"{_INCOMING_PREFIX}{columns[0]}").is_not_null()).sum()
        ).item()
    )
    changed_columns = {
        column: int(frame[column].ne_missing(updated[column]).sum())
        for column in columns
        if frame[column].ne_missing(updated[column]).any()
    }
    changed_mask = pl.Series([False] * frame.height)
    for column in changed_columns:
        changed_mask = changed_mask | frame[column].ne_missing(updated[column])
    changed_rows = int(changed_mask.sum())

    refresh = FileRefresh(
        path=path,
        exists=True,
        matched_rows=matched,
        changed_rows=changed_rows,
        changed_columns=changed_columns,
    )
    if changed_rows:
        _write_atomic(updated.select(frame.columns), path)
        log.info(
            "Refreshed %s: %d/%d matched rows changed (%s).",
            path.name,
            changed_rows,
            matched,
            ", ".join(f"{name}={count}" for name, count in sorted(changed_columns.items())),
        )
    else:
        log.info("Left %s unchanged: %d matched rows already current.", path.name, matched)
    return refresh


def refresh_lines(
    season: int,
    week: int,
    *,
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> LinesRefreshResult:
    """Refresh the market lines of ``season`` in the prediction and all-data datasets.

    The lines follow the line source the data was built with (``read_build_lines``).

    Args:
        season: Season whose lines to refresh.
        week: Week whose ``data/predict`` file to refresh alongside the all-data datasets.
        data_dir: Data directory (defaults to the project's ``data/``).
        cache_dir: Optional nflreadpy cache directory override.

    Returns:
        A :class:`LinesRefreshResult` with per-file counts and the line source followed.

    """
    resolved_data_dir = data_dir or DEFAULT_DATA_DIR
    build = read_build_lines(resolved_data_dir)
    log.info(
        "Refreshing %s lines for season %d week %d in %s.",
        build.line_source,
        season,
        week,
        resolved_data_dir,
    )
    lines = load_lines(season, cache_dir=cache_dir, build=build)
    log.info("Loaded lines for %d games of season %d.", lines.height, season)
    files = tuple(
        refresh_file(path, lines, season) for path in _target_paths(resolved_data_dir, week)
    )
    result = LinesRefreshResult(
        season=season,
        week=week,
        schedule_games=lines.height,
        files=files,
        line_source=build.line_source,
    )
    log.info(
        "Lines refresh complete: %d rows changed across %d files.",
        result.changed_rows,
        len(result.written_paths),
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Refresh market lines in the existing datasets, following the line source the data "
            "was built with (its market_lines_metadata.json): the nflverse schedule for a stored "
            "build; nfelo's latest line, then nflverse's, then SurvivorGrid's, priced with the "
            "build's fitted moneyline map, for a pick-time build."
        )
    )
    parser.add_argument("--season", type=int, default=None, help="Season to refresh.")
    parser.add_argument("--week", type=int, default=None, help="Week whose predict file to update.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data directory override.")
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="nflreadpy cache directory override."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Refresh lines from the command line.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when omitted.

    Returns:
        A process exit code.

    """
    args = _build_parser().parse_args(argv)
    season, week = args.season, args.week
    if season is None or week is None:
        current_season, current_week = polars_utils.get_current_nfl_week()
        season = current_season if season is None else season
        week = current_week if week is None else week
    refresh_lines(season, week, data_dir=args.data_dir, cache_dir=args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
