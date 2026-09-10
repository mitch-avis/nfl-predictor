"""Read power rankings and projected standings, adding week-over-week movement."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nfl_predictor.api.readers.cache import cached
from nfl_predictor.api.registry import project
from nfl_predictor.api.registry.power import POWER_COLUMNS, STANDINGS_COLUMNS
from nfl_predictor.api.schemas.common import TablePayload


def _read_csv(path: Path) -> pl.DataFrame:
    """Read a CSV."""
    return pl.read_csv(path, infer_schema_length=10000)


def _with_record(df: pl.DataFrame) -> pl.DataFrame:
    """Add a ``record`` string column (``W-L`` or ``W-L-T``)."""
    if not {"wins", "losses"} <= set(df.columns):
        return df
    ties = pl.col("ties") if "ties" in df.columns else pl.lit(0)
    record = pl.concat_str(
        [
            pl.col("wins").cast(pl.Int64).cast(pl.String),
            pl.lit("-"),
            pl.col("losses").cast(pl.Int64).cast(pl.String),
            pl.when(ties.fill_null(0) > 0)
            .then(pl.concat_str([pl.lit("-"), ties.cast(pl.Int64).cast(pl.String)]))
            .otherwise(pl.lit("")),
        ]
    )
    return df.with_columns(record.alias("record"))


def rankings_frame(path: Path) -> pl.DataFrame:
    """Return the cached rankings frame sorted by rank."""
    frame: pl.DataFrame = cached(path, _read_csv)
    return frame.sort("rank") if "rank" in frame.columns else frame


def with_movement(current: pl.DataFrame, previous: pl.DataFrame | None) -> pl.DataFrame:
    """Attach ``previous_rank`` and ``rank_change`` (positive = climbed) from ``previous``."""
    if previous is None or "rank" not in previous.columns or "rank" not in current.columns:
        return current.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("previous_rank"),
            pl.lit(None, dtype=pl.Int64).alias("rank_change"),
        )
    prior = previous.select(
        pl.col("team_abbr"), pl.col("rank").cast(pl.Int64).alias("previous_rank")
    )
    joined = current.join(prior, on="team_abbr", how="left")
    return joined.with_columns(
        (pl.col("previous_rank") - pl.col("rank").cast(pl.Int64)).alias("rank_change")
    )


def read_rankings(path: Path, previous_path: Path | None) -> TablePayload:
    """Return the rankings table with movement against ``previous_path`` when given."""
    current = rankings_frame(path)
    previous = rankings_frame(previous_path) if previous_path is not None else None
    return project(_with_record(with_movement(current, previous)), POWER_COLUMNS)


def read_standings(path: Path) -> TablePayload:
    """Return a projected-standings table sorted by projected win percentage."""
    frame: pl.DataFrame = cached(path, _read_csv)
    if "projected_win_pct" in frame.columns:
        frame = frame.sort(["projected_win_pct", "team_abbr"], descending=[True, False])
    return project(_with_record(frame), STANDINGS_COLUMNS)
