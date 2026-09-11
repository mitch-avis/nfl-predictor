"""Refresh market lines in the existing datasets without rerunning the full ETL.

The weekly ETL rebuilds every feature family, which takes minutes and rewrites every dataset. Lines
move daily, so this module re-reads the schedule for a single season and writes back only the five
market columns the model consumes, matching rows on ``game_id`` alone. Completed-game datasets are
never touched: their lines are history.

Run it as ``python -m nfl_predictor.lines_refresh --season 2026 --week 2``. Because the market
features feed the model, a prediction run should follow a refresh.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import game_utils, polars_utils
from nfl_predictor.utils.logger import log

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
DEFAULT_DATA_DIR = Path(constants.DATA_PATH)
_INCOMING_PREFIX = "__incoming_"


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
    changed_columns: dict[str, int] = field(default_factory=dict)

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


def load_lines(season: int, *, cache_dir: Path | None = None) -> pl.DataFrame:
    """Return one row per game of ``season`` carrying the current market lines.

    Args:
        season: Season to refresh.
        cache_dir: Optional nflreadpy cache directory override.

    Returns:
        A frame of the join keys (``game_id`` and the season/week/team matchup) plus whichever
        line columns the schedule provides, with missing moneylines derived from the spreads.

    Raises:
        ValueError: If the schedule holds no games for the season.

    """
    schedule = polars_utils.load_schedule(
        [season],
        cache_dir=cache_dir,
        force_refresh=True,
        current_season=season,
    )
    if schedule.is_empty() or JOIN_COLUMN not in schedule.columns:
        raise ValueError(f"The schedule for season {season} has no games to refresh lines from.")
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
    os.replace(tmp_path, path)


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

    Args:
        season: Season whose lines to refresh.
        week: Week whose ``data/predict`` file to refresh alongside the all-data datasets.
        data_dir: Data directory (defaults to the project's ``data/``).
        cache_dir: Optional nflreadpy cache directory override.

    Returns:
        A :class:`LinesRefreshResult` with per-file counts.

    """
    resolved_data_dir = data_dir or DEFAULT_DATA_DIR
    log.info("Refreshing lines for season %d week %d from %s.", season, week, resolved_data_dir)
    lines = load_lines(season, cache_dir=cache_dir)
    log.info("Loaded lines for %d games of season %d.", lines.height, season)
    files = tuple(
        refresh_file(path, lines, season) for path in _target_paths(resolved_data_dir, week)
    )
    result = LinesRefreshResult(season=season, week=week, schedule_games=lines.height, files=files)
    log.info(
        "Lines refresh complete: %d rows changed across %d files.",
        result.changed_rows,
        len(result.written_paths),
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(description="Refresh market lines in the existing datasets.")
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
