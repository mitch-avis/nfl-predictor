"""The market line known at pick time, game by game.

The stored nflverse lines are one late, probably closing, snapshot, but picks are made before
the week's first game. Under the ``pick_time`` line source the ETL replaces each game's five
line columns, in this order:

- A completed game with a real nfelo opener (seasons from
  ``constants.NFELO_FIRST_REAL_OPENER_SEASON`` on, except
  ``constants.NFELO_UNRELIABLE_OPENER_SEASONS``, and an opening spread present) takes the
  opening spread. Its moneylines are the opening pair when nfelo has both (from 2024),
  otherwise they are left missing and derived later from the opening spread with the fitted
  map (``moneyline_map``); its total is the opening total, or the stored total before nfelo
  published opening totals.
- Any other completed game keeps its stored line: a counted fallback row.
- An upcoming game takes nfelo's latest line (its opener when there is no latest), with that
  line's moneyline pair and total (the stored total when nfelo has neither); without an nfelo
  line it keeps the nflverse line; without either it is left empty for SurvivorGrid.

The closing line never becomes a feature under this source. A game with no line at all still
cannot be anchored, and the model run stops on it, as with the stored source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars import moneyline_map
from nfl_predictor.utils.polars.nfelo_lines import NFELO_KEY_COLUMNS

if TYPE_CHECKING:
    from nfl_predictor.utils.polars.nfelo_lines import NfeloLinesSnapshot

LINE_COLUMNS: tuple[str, ...] = tuple(constants.LINES_COLUMNS)
_SOURCE = "__line_source"
_MATCHED = "__nfelo_matched"
_OPENER = "opener"
_STORED = "stored_fallback"
_UPCOMING_NFELO = "upcoming_nfelo"
_UPCOMING_NFLVERSE = "upcoming_nflverse"
_UPCOMING_NONE = "upcoming_without_line"
_REPORT_COUNTS: tuple[str, ...] = (
    "games",
    "matched",
    _OPENER,
    "opener_with_stored_total",
    _STORED,
    _UPCOMING_NFELO,
    _UPCOMING_NFLVERSE,
    _UPCOMING_NONE,
)
_FROM_MAP = "moneylines_from_map"
_FROM_FIXED = "moneylines_from_fixed_conversion"


def _pair(home: str, away: str) -> tuple[pl.Expr, pl.Expr]:
    """Return a moneyline pair, both null unless both sides are present."""
    both = pl.col(home).is_not_null() & pl.col(away).is_not_null()
    return (
        pl.when(both).then(pl.col(home)).otherwise(None),
        pl.when(both).then(pl.col(away)).otherwise(None),
    )


def apply_pick_time_lines(
    schedule: pl.DataFrame, nfelo: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Replace each game's line columns with the line known at pick time.

    Args:
        schedule: Schedule rows with the stored lines and the scores.
        nfelo: Normalized nfelo lines (``nfelo_lines.normalize_nfelo_lines``).

    Returns:
        The schedule with only the five line columns changed (same rows, order and types),
        and a per-season report of games, matches, openers and fallback rows.

    """
    key_types = {column: schedule.schema[column] for column in NFELO_KEY_COLUMNS}
    lookup = nfelo.with_columns(
        *(pl.col(column).cast(dtype) for column, dtype in key_types.items()),
        pl.lit(value=True).alias(_MATCHED),
    )
    joined = schedule.join(lookup, on=list(NFELO_KEY_COLUMNS), how="left", maintain_order="left")

    matched = pl.col(_MATCHED).fill_null(value=False)
    completed = pl.col("away_score").is_not_null()
    real_opener = (
        matched
        & (pl.col("season") >= constants.NFELO_FIRST_REAL_OPENER_SEASON)
        & ~pl.col("season").is_in(sorted(constants.NFELO_UNRELIABLE_OPENER_SEASONS))
        & pl.col("home_spread_open").is_not_null()
    )
    latest_spread = pl.coalesce("home_spread_last", "home_spread_open")
    source = (
        pl.when(completed & real_opener)
        .then(pl.lit(_OPENER))
        .when(completed)
        .then(pl.lit(_STORED))
        .when(latest_spread.is_not_null())
        .then(pl.lit(_UPCOMING_NFELO))
        .when(pl.col("home_spread").is_not_null())
        .then(pl.lit(_UPCOMING_NFLVERSE))
        .otherwise(pl.lit(_UPCOMING_NONE))
    )
    joined = joined.with_columns(source.alias(_SOURCE))

    is_opener = pl.col(_SOURCE) == _OPENER
    is_latest = pl.col(_SOURCE) == _UPCOMING_NFELO
    home_spread = (
        pl.when(is_opener)
        .then(pl.col("home_spread_open"))
        .when(is_latest)
        .then(latest_spread)
        .otherwise(pl.col("home_spread"))
    )
    open_home, open_away = _pair("home_ml_open", "away_ml_open")
    last_home, last_away = _pair("home_ml_last", "away_ml_last")
    replaced = joined.with_columns(
        home_spread.alias("home_spread"),
        pl.when(is_opener | is_latest)
        .then(-home_spread)
        .otherwise(pl.col("away_spread"))
        .alias("away_spread"),
        pl.when(is_opener)
        .then(open_home)
        .when(is_latest)
        .then(last_home)
        .otherwise(pl.col("home_moneyline"))
        .alias("home_moneyline"),
        pl.when(is_opener)
        .then(open_away)
        .when(is_latest)
        .then(last_away)
        .otherwise(pl.col("away_moneyline"))
        .alias("away_moneyline"),
        pl.when(is_opener)
        .then(pl.coalesce("total_line_open", "total_line"))
        .when(is_latest)
        .then(pl.coalesce("total_line_last", "total_line_open", "total_line"))
        .otherwise(pl.col("total_line"))
        .alias("total_line"),
    )

    report = (
        replaced.group_by("season", maintain_order=True)
        .agg(
            pl.len().alias("games"),
            pl.col(_MATCHED).fill_null(value=False).sum().alias("matched"),
            (pl.col(_SOURCE) == _OPENER).sum().alias(_OPENER),
            ((pl.col(_SOURCE) == _OPENER) & pl.col("total_line_open").is_null())
            .sum()
            .alias("opener_with_stored_total"),
            *((pl.col(_SOURCE) == name).sum().alias(name) for name in _REPORT_COUNTS[4:]),
        )
        .sort("season")
        .with_columns(pl.col(name).cast(pl.Int64) for name in _REPORT_COUNTS)
    )
    lines = replaced.select(
        pl.col(column).cast(dtype) if column in LINE_COLUMNS else pl.col(column)
        for column, dtype in schedule.schema.items()
    )
    return lines, report


@dataclass
class PickTimeLines:
    """What a pick-time build used: the nfelo snapshot, the line order's report and the maps.

    ``fill_moneylines`` derives each missing moneyline from its game's spread with that
    season's map and counts what it derived, so ``metadata`` can report it.
    """

    snapshot: NfeloLinesSnapshot
    report: pl.DataFrame
    maps: dict[int, moneyline_map.SpreadMoneylineMap | None]
    derived: dict[int, dict[str, int]] = field(default_factory=dict)

    def fill_moneylines(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill missing moneylines with each season's map, recording how many it derived."""
        if {"season", "home_spread"}.issubset(df.columns):
            missing = pl.lit(value=False)
            for column in ("home_moneyline", "away_moneyline"):
                if column in df.columns:
                    missing = missing | pl.col(column).is_null()
                else:
                    missing = pl.lit(value=True)
            needed = df.filter(pl.col("home_spread").is_not_null() & missing)
            for season, games in (
                needed.group_by("season", maintain_order=True).agg(pl.len()).iter_rows()
            ):
                kind = _FROM_FIXED if self.maps.get(int(season)) is None else _FROM_MAP
                counts = self.derived.setdefault(int(season), {_FROM_MAP: 0, _FROM_FIXED: 0})
                counts[kind] += int(games)
        return moneyline_map.fill_moneylines_from_maps(df, self.maps)

    def metadata(self) -> dict[str, object]:
        """Return the run-metadata record of the pick-time build."""
        seasons: dict[str, dict[str, int]] = {}
        for row in self.report.iter_rows(named=True):
            season = int(row["season"])
            counts = {name: int(row[name]) for name in _REPORT_COUNTS}
            derived = self.derived.get(season, {_FROM_MAP: 0, _FROM_FIXED: 0})
            counts.update({_FROM_MAP: derived[_FROM_MAP], _FROM_FIXED: derived[_FROM_FIXED]})
            seasons[str(season)] = counts
        return {
            "line_source": constants.LINE_SOURCE_PICK_TIME,
            "nfelo": self.snapshot.metadata(),
            "seasons": seasons,
            "moneyline_maps": {
                str(season): None if fitted is None else fitted.to_dict()
                for season, fitted in sorted(self.maps.items())
            },
        }


def prepare_pick_time_lines(
    schedule: pl.DataFrame, *, history: pl.DataFrame, snapshot: NfeloLinesSnapshot
) -> tuple[pl.DataFrame, PickTimeLines]:
    """Apply the pick-time line order and fit each season's moneyline map.

    The maps are fitted on the stored prices (``moneyline_map.price_pairs``) of the schedule
    before its lines are replaced, together with ``history`` for earlier seasons the run does
    not build; each season's map sees only earlier seasons.

    Args:
        schedule: The run's schedule rows with the stored lines.
        history: Schedule rows of earlier seasons for the map fits (may be empty); seasons
            also in ``schedule`` are ignored.
        snapshot: The nfelo lines the run uses.

    Returns:
        The schedule with pick-time lines and the record of the build.

    """
    pairs = [moneyline_map.price_pairs(schedule).with_columns(pl.col("season").cast(pl.Int64))]
    if history.height > 0:
        own_seasons = schedule.get_column("season").unique(maintain_order=True).to_list()
        pairs.insert(
            0,
            moneyline_map.price_pairs(
                history.filter(~pl.col("season").is_in(own_seasons))
            ).with_columns(pl.col("season").cast(pl.Int64)),
        )
    seasons = schedule.get_column("season").unique(maintain_order=True).to_list()
    maps = moneyline_map.fit_season_maps(pl.concat(pairs, how="vertical_relaxed"), seasons)
    lines, report = apply_pick_time_lines(schedule, snapshot.frame)
    return lines, PickTimeLines(snapshot=snapshot, report=report, maps=maps)
