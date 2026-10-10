"""Check that the season statistics parity audit's comparison catches each kind of difference.

Run from the repository root:

    .venv/bin/python .agents/sos_port/decisions/season_stats_audit_check.py

It reads the published 2024 team game logs of ``nfl-sos-ratings`` (beside this repository's main
checkout, at the frozen commit), makes one copy per kind of difference, and prints what the audit's
``compare_tables`` reports for each: a copy equal to the published file must pass, a float moved by
1e-12 must pass as within tolerance, and every other copy must fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from season_stats_parity import compare_tables, default_source, relabel_washington


def _at(row: int, column: str, value: pl.Expr) -> pl.Expr:
    """Return ``column`` with row ``row`` replaced by ``value``."""
    return pl.when(pl.int_range(pl.len()) == row).then(value).otherwise(pl.col(column)).alias(column)


def main() -> None:
    """Print the comparison of each perturbed copy with the published file."""
    published = pl.read_parquet(default_source() / "data" / "2024_team_game_logs.parquet")
    rebuilt, _ = relabel_washington(published)
    copies = {
        "identical": rebuilt,
        "text": rebuilt.with_columns(_at(3, "opponent_team", pl.lit("XXX"))),
        "integer": rebuilt.with_columns(
            _at(5, "offensive_snaps", pl.col("offensive_snaps") + 1)
        ),
        "float moved by 1e-12": rebuilt.with_columns(
            _at(7, "offensive_epa", pl.col("offensive_epa") + 1e-12)
        ),
        "float moved by 1e-6": rebuilt.with_columns(
            _at(7, "offensive_epa", pl.col("offensive_epa") + 1e-6)
        ),
        "null": rebuilt.with_columns(_at(9, "passing_epa", pl.lit(None, dtype=pl.Float64))),
        "rows reversed": rebuilt.reverse(),
        "first 10 rows": rebuilt.head(10),
        "columns reversed": rebuilt.select(list(reversed(rebuilt.columns))),
        "WAS written back in team": rebuilt.with_columns(pl.col("team").replace("WSH", "WAS")),
    }
    for name, copy in copies.items():
        result = compare_tables(2024, "team_game_logs", published, copy)
        print(
            f"{name}: passed {result.passed}, identical {result.identical:,}, within tolerance"
            f" {result.within_tolerance}, outside {result.outside:,},"
            f" structure {result.structure or 'none'}"
        )


if __name__ == "__main__":
    main()
