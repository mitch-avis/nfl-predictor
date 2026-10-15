"""The ported prior's carryover on a whole team rating, season by season (power rankings, week 1).

Run from the repository root (single-threaded, a few minutes):

    VIRTUAL_ENV="$PWD/.venv" .venv/bin/python \
        .agents/sos_port/decisions/power_rankings_carryover.py --build-dir ~/scratch/step6/data \
        > .agents/sos_port/decisions/power_rankings_carryover_output.txt

For each season from 2008 to 2026 it prints the factor the model rating's week 1 uses
(``rating_sources.carryover_factor``: the least-squares slope, through the origin over the teams
of both seasons, of each team's week-1 ported rating, read from the build's game rows, on its
published rating for the whole previous season, ``ratings/{season-1}_ratings.parquet``), beside
the ported prior's own two carryover slopes for that season (offense and defense,
``PriorHistory.slopes``), fit on the team-game rows of the nflreadpy cache snapshot. Those rows
reproduce the build's ported ratings: ``rankings_eval.py --ratings-from-snapshot`` finds no
team-week where the two differ. Reads only; writes nothing but standard output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
import rankings_eval  # a sibling module of the parent directory, importable once it is on the path

from nfl_predictor.ratings.team_prior import PriorHistory
from nfl_predictor.reporting import rating_sources

SEASONS = tuple(range(2008, 2027))


def week_one_ratings(path: Path) -> pl.DataFrame:
    """Return every season's week-1 ported ratings from the build's game rows."""
    features = list(rankings_eval.PORTED_FEATURES)
    rows = pl.read_csv(
        path,
        columns=["season", "week", "game_type", "away_abbr", "home_abbr", *features],
        schema_overrides={
            "season": pl.Int64,
            "week": pl.Int64,
            **dict.fromkeys(features, pl.Float64),
        },
        infer_schema_length=None,
    ).filter((pl.col("week") == 1) & (pl.col("game_type") == "REG"))
    return rating_sources.ported_ratings_from_games(rows)


def main(argv: list[str] | None = None) -> None:
    """Print the factor and the two slopes for every season."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--build-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    games_path = arguments.build_dir / "completed_games_ml.csv"
    ratings_dir = arguments.build_dir / "ratings"
    week_one = week_one_ratings(games_path)
    files: dict[str, str] = {}
    rows = rankings_eval.snapshot_team_games(files)
    history = PriorHistory(lambda season: rows[season])
    out = sys.stdout
    out.write(f"build game rows: {games_path} sha256 {rankings_eval.sha256_file(games_path)}\n")
    out.write(f"snapshot files read: {len(files)} (sha256 checked against its manifest)\n\n")
    out.write("season  teams  factor  offense slope  defense slope  season pairs\n")
    factors: list[float] = []
    for season in SEASONS:
        season_rows = week_one.filter(pl.col("season") == season)
        ratings = dict(
            zip(season_rows.get_column("team"), season_rows.get_column("rating"), strict=True)
        )
        previous = rating_sources.season_ported_ratings(ratings_dir, season - 1)
        factor = rating_sources.carryover_factor(ratings, previous)
        factors.append(factor)
        slopes = history.slopes(season)
        if slopes is None:
            message = f"no carryover slopes for {season}"
            raise SystemExit(message)
        out.write(
            f"{season}  {len(set(ratings) & set(previous)):>5}  {factor:6.4f}  "
            f"{slopes.offense:13.4f}  {slopes.defense:13.4f}  {slopes.pairs:>12}\n"
        )
    out.write(
        f"\nfactor over the seasons: min {min(factors):.4f}, max {max(factors):.4f}, "
        f"mean {sum(factors) / len(factors):.4f}\n"
    )


if __name__ == "__main__":
    main()
