"""Build a games-to-predict file for a week the ETL has not produced one for.

``data_collection`` writes ``data/predict/week_NN_games_to_predict.csv`` for the current week only,
so predicting further ahead has nothing to read. Every future game already has a feature row in
``all_data_ml.csv``; this module extracts one week of them with the ETL's own upcoming-game rule.

The extracted rows carry whatever the last ETL knew: a week that is still weeks away has no market
lines, and its rest, quarterback, and record features are as of the last refresh. Predictions made
from it should be regenerated once the week is close enough for an ETL run to fill those in.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.logger import log

DEFAULT_DATA_DIR = Path(constants.DATA_PATH)
SOURCE_FILENAME = "all_data_ml.csv"
MARKET_COLUMNS: tuple[str, ...] = (
    "total_line",
    "away_spread",
    "home_spread",
    "away_moneyline",
    "home_moneyline",
)


@dataclass(frozen=True)
class WeekBuild:
    """What building one week's prediction inputs produced.

    Attributes:
        path: The games-to-predict file.
        season: Season the week belongs to.
        week: Week that was built.
        games: Number of upcoming games written (or already in the file).
        created: Whether the file was written now; ``False`` when it already existed.
        missing_market_columns: Line columns that are empty for every game of the week.

    """

    path: Path
    season: int
    week: int
    games: int
    created: bool
    missing_market_columns: tuple[str, ...] = ()


def week_file_path(data_dir: Path, week: int) -> Path:
    """Return the games-to-predict path for ``week``."""
    return data_dir / "predict" / f"week_{week:02d}_games_to_predict.csv"


def _source_path(data_dir: Path) -> Path:
    """Return the ML dataset path, or explain that it is missing."""
    path = data_dir / SOURCE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run the ETL before predicting a future week.")
    return path


def available_weeks(season: int, *, data_dir: Path | None = None) -> list[int]:
    """Return the weeks of ``season`` that still have unplayed games, in order.

    Args:
        season: Season to inspect.
        data_dir: Data directory (defaults to the project's ``data/``).

    Returns:
        Sorted week numbers, or an empty list when the dataset is missing.

    """
    path = (data_dir or DEFAULT_DATA_DIR) / SOURCE_FILENAME
    if not path.is_file():
        return []
    frame = pl.read_csv(path, columns=["season", "week", "away_score", "home_score"])
    candidates = sorted(
        {int(week) for week in frame.filter(pl.col("season") == season)["week"].to_list()}
    )
    return [
        week
        for week in candidates
        if not polars_utils.filter_upcoming_games(frame, season, week).is_empty()
    ]


def build_week_file(
    season: int,
    week: int,
    *,
    data_dir: Path | None = None,
    overwrite: bool = False,
) -> WeekBuild:
    """Write the games-to-predict file for ``season`` week ``week``.

    Args:
        season: Season to build from.
        week: Week whose upcoming games to extract.
        data_dir: Data directory (defaults to the project's ``data/``).
        overwrite: Replace an existing file instead of keeping it.

    Returns:
        A :class:`WeekBuild` describing the file.

    Raises:
        FileNotFoundError: If ``all_data_ml.csv`` is missing.
        ValueError: If the week has no upcoming games in the dataset.

    """
    resolved = data_dir or DEFAULT_DATA_DIR
    path = week_file_path(resolved, week)
    if path.is_file() and not overwrite:
        with path.open(encoding="utf-8") as handle:
            games = max(sum(1 for _ in handle) - 1, 0)
        log.info("Keeping the existing %s (%d games); pass --overwrite to rebuild.", path, games)
        return WeekBuild(path=path, season=season, week=week, games=games, created=False)

    frame = pl.read_csv(_source_path(resolved), infer_schema_length=None)
    upcoming = polars_utils.filter_upcoming_games(frame, season, week)
    if upcoming.is_empty():
        raise ValueError(
            f"Season {season} week {week} has no upcoming games in {SOURCE_FILENAME}; "
            "it may already be played or beyond the schedule the ETL loaded."
        )
    missing = tuple(
        column
        for column in MARKET_COLUMNS
        if column in upcoming.columns and upcoming[column].null_count() == upcoming.height
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    upcoming.write_csv(path)
    log.info(
        "Wrote %d upcoming games for season %d week %d to %s.", upcoming.height, season, week, path
    )
    if missing:
        log.warning(
            "No market lines yet for season %d week %d (%s); refresh lines closer to kickoff.",
            season,
            week,
            ", ".join(missing),
        )
    return WeekBuild(
        path=path,
        season=season,
        week=week,
        games=upcoming.height,
        created=True,
        missing_market_columns=missing,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Build the games-to-predict file for one week from the ML dataset."
    )
    parser.add_argument("--season", type=int, default=None, help="Season to build from.")
    parser.add_argument("--week", type=int, default=None, help="Week to build.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data directory override.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing week file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build a week's prediction inputs from the command line.

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
    build_week_file(season, week, data_dir=args.data_dir, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
